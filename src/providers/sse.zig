const std = @import("std");
const std_compat = @import("compat");
const root = @import("root.zig");
const http_util = @import("../http_util.zig");
const error_classify = @import("error_classify.zig");
const verbose = @import("../verbose.zig");
const stream_tools = @import("sse_tool_calls.zig");
const line_reader = @import("sse_line_reader.zig");
const log = std.log.scoped(.provider_sse);

var curl_fail_fast_arg_mutex: std_compat.sync.Mutex = .{};
var curl_fail_with_body_supported_cache: ?bool = null;
const stream_stall_detection_args = [_][]const u8{
    "--speed-limit",
    "1",
    "--speed-time",
    "60",
};
const DEFAULT_STREAM_MAX_TIME_SECS: u64 = 600;
const MAX_STREAM_OUTPUT_BYTES: usize = 4 * 1024 * 1024;

pub fn appendStreamOutput(allocator: std.mem.Allocator, output: *std.ArrayListUnmanaged(u8), bytes: []const u8) !void {
    if (bytes.len > MAX_STREAM_OUTPUT_BYTES - output.items.len) return error.StreamOutputTooLarge;
    try output.appendSlice(allocator, bytes);
}

pub fn appendCurlStreamBaseArgs(
    allocator: std.mem.Allocator,
    argv_buf: [][]const u8,
    argc: *usize,
    timeout_buf: *[32]u8,
    timeout_secs: u64,
) void {
    const base = [_][]const u8{ "curl", "-s", "--no-buffer", curlFailFastArg(allocator) };
    for (base) |arg| {
        argv_buf[argc.*] = arg;
        argc.* += 1;
    }
    const seconds = if (timeout_secs > 0) timeout_secs else DEFAULT_STREAM_MAX_TIME_SECS;
    const timeout_str = std.fmt.bufPrint(timeout_buf, "{d}", .{seconds}) catch unreachable;
    argv_buf[argc.*] = "--max-time";
    argc.* += 1;
    argv_buf[argc.*] = timeout_str;
    argc.* += 1;
    appendCurlStallDetectionArgs(argv_buf, argc);
    argv_buf[argc.*] = "-X";
    argc.* += 1;
    argv_buf[argc.*] = "POST";
    argc.* += 1;
}

pub fn stopCurl(child: *std_compat.process.Child) void {
    _ = child.kill() catch {};
    _ = child.wait() catch {};
}

/// Start a stream process with caller-prepared arguments and send its body.
/// The caller owns the child and must call `wait` or `stopCurl` on every path.
pub fn spawnCurl(allocator: std.mem.Allocator, argv: []const []const u8, body: []const u8) !std_compat.process.Child {
    var child = std_compat.process.Child.init(argv, allocator);
    child.stdin_behavior = .Pipe;
    child.stdout_behavior = .Pipe;
    child.stderr_behavior = .Ignore;
    try child.spawn();
    errdefer stopCurl(&child);
    const stdin_file = child.stdin orelse return error.CurlWriteError;
    stdin_file.writeAll(body) catch return error.CurlWriteError;
    stdin_file.close();
    child.stdin = null;
    return child;
}

fn finalizeStreamResultWithTools(
    allocator: std.mem.Allocator,
    accumulated: []const u8,
    stream_usage: ?root.TokenUsage,
    collector: *stream_tools.Collector,
    allow_tool_calls: bool,
) !root.StreamChatResult {
    var content: ?[]const u8 = null;
    var reasoning_content: ?[]const u8 = null;
    if (accumulated.len > 0) {
        const split = try root.splitThinkContent(allocator, accumulated);
        content = split.visible;
        reasoning_content = split.reasoning;
    }
    errdefer {
        if (content) |text| allocator.free(text);
        if (reasoning_content) |text| allocator.free(text);
    }

    var usage = stream_usage orelse root.TokenUsage{};
    if (usage.completion_tokens == 0) {
        usage.completion_tokens = @intCast((accumulated.len + 3) / 4);
    }

    const tool_calls = if (allow_tool_calls) try collector.take(allocator) else &.{};
    return .{
        .content = content,
        .reasoning_content = reasoning_content,
        .tool_calls = tool_calls,
        .usage = usage,
        .model = "",
        .finish_reason = collector.finish_reason,
        .tool_fragments = collector.fragments,
    };
}

fn finalizeStreamResult(
    allocator: std.mem.Allocator,
    accumulated: []const u8,
    stream_usage: ?root.TokenUsage,
) !root.StreamChatResult {
    var collector = stream_tools.Collector{};
    return finalizeStreamResultWithTools(allocator, accumulated, stream_usage, &collector, false);
}

fn parseCurlVersionComponent(component: []const u8) ?u32 {
    var end: usize = 0;
    while (end < component.len and std.ascii.isDigit(component[end])) : (end += 1) {}
    if (end == 0) return null;
    return std.fmt.parseInt(u32, component[0..end], 10) catch null;
}

fn parseCurlVersionTriplet(version_line: []const u8) ?[3]u32 {
    const prefix = "curl ";
    if (!std.mem.startsWith(u8, version_line, prefix)) return null;

    const version_tail = version_line[prefix.len..];
    const version_end = std.mem.indexOfScalar(u8, version_tail, ' ') orelse version_tail.len;
    const version_token = version_tail[0..version_end];

    var parts = std.mem.splitScalar(u8, version_token, '.');
    const major = parseCurlVersionComponent(parts.next() orelse return null) orelse return null;
    const minor = parseCurlVersionComponent(parts.next() orelse return null) orelse return null;
    const patch = parseCurlVersionComponent(parts.next() orelse return null) orelse return null;
    return .{ major, minor, patch };
}

fn curlVersionSupportsFailWithBody(version_line: []const u8) bool {
    const version = parseCurlVersionTriplet(version_line) orelse return false;
    if (version[0] != 7) return version[0] > 7;
    if (version[1] != 76) return version[1] > 76;
    return version[2] >= 0;
}

fn detectCurlFailWithBodySupport(allocator: std.mem.Allocator) bool {
    const result = std_compat.process.Child.run(.{
        .allocator = allocator,
        .argv = &.{ "curl", "--version" },
        .max_output_bytes = 1024,
    }) catch return false;
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);

    switch (result.term) {
        .exited => |code| if (code != 0) return false,
        else => return false,
    }

    const trimmed = std.mem.trim(u8, result.stdout, " \n\r\t");
    var line_it = std.mem.splitScalar(u8, trimmed, '\n');
    return curlVersionSupportsFailWithBody(line_it.first());
}

/// Prefer `--fail-with-body` so JSON API errors remain classifiable, but fall
/// back to `-f` on curl releases older than 7.76.0 where the newer flag fails.
pub fn curlFailFastArg(allocator: std.mem.Allocator) []const u8 {
    curl_fail_fast_arg_mutex.lock();
    defer curl_fail_fast_arg_mutex.unlock();

    if (curl_fail_with_body_supported_cache == null) {
        curl_fail_with_body_supported_cache = detectCurlFailWithBodySupport(allocator);
    }

    return if (curl_fail_with_body_supported_cache.?) "--fail-with-body" else "-f";
}

pub fn appendCurlStallDetectionArgs(argv_buf: [][]const u8, argc: *usize) void {
    for (stream_stall_detection_args) |arg| {
        argv_buf[argc.*] = arg;
        argc.* += 1;
    }
}

/// Content delta from an SSE chunk.
pub const DeltaContent = union(enum) {
    text: []const u8,
    reasoning: []const u8,

    pub fn deinit(self: DeltaContent, allocator: std.mem.Allocator) void {
        switch (self) {
            .text => |t| allocator.free(t),
            .reasoning => |r| allocator.free(r),
        }
    }
};

/// Result of parsing a single SSE line.
pub const SseLineResult = union(enum) {
    /// Text or reasoning delta content (owned, caller frees).
    delta: DeltaContent,
    /// Stream is complete ([DONE] sentinel).
    done: void,
    /// Token usage from a stream chunk.
    usage: root.TokenUsage,
    /// Line should be skipped (empty, comment, or no content).
    skip: void,
};

const THINK_OPEN_TAG = "<think>";
const THINK_CLOSE_TAG = "</think>";

fn closeReasoningBlock(
    allocator: std.mem.Allocator,
    accumulated: *std.ArrayListUnmanaged(u8),
    in_reasoning: *bool,
    callback: root.StreamCallback,
    ctx: *anyopaque,
) !void {
    if (!in_reasoning.*) return;
    in_reasoning.* = false;
    try appendStreamOutput(allocator, accumulated, THINK_CLOSE_TAG);
    callback(ctx, root.StreamChunk.textDelta(THINK_CLOSE_TAG));
}

fn appendDeltaContent(
    allocator: std.mem.Allocator,
    accumulated: *std.ArrayListUnmanaged(u8),
    in_reasoning: *bool,
    callback: root.StreamCallback,
    ctx: *anyopaque,
    content: DeltaContent,
) !void {
    switch (content) {
        .text => |text| {
            try closeReasoningBlock(allocator, accumulated, in_reasoning, callback, ctx);
            try appendStreamOutput(allocator, accumulated, text);
            callback(ctx, root.StreamChunk.textDelta(text));
        },
        .reasoning => |reasoning| {
            if (!in_reasoning.*) {
                in_reasoning.* = true;
                try appendStreamOutput(allocator, accumulated, THINK_OPEN_TAG);
                callback(ctx, root.StreamChunk.textDelta(THINK_OPEN_TAG));
            }
            try appendStreamOutput(allocator, accumulated, reasoning);
            callback(ctx, root.StreamChunk.textDelta(reasoning));
        },
    }
}

/// Parse a single SSE line in OpenAI streaming format.
///
/// Handles:
/// - `data: [DONE]` → `.done`
/// - `data: {JSON}` → extracts `choices[0].delta.content` → `.delta`
/// - Empty lines, comments (`:`) → `.skip`
pub fn parseSseLine(allocator: std.mem.Allocator, line: []const u8) !SseLineResult {
    return parseSseLineWithTools(allocator, line, null);
}

fn parseSseLineWithTools(
    allocator: std.mem.Allocator,
    line: []const u8,
    collector: ?*stream_tools.Collector,
) !SseLineResult {
    const trimmed = std_compat.mem.trimRight(u8, line, "\r");

    if (trimmed.len == 0) return .skip;
    if (trimmed[0] == ':') return .skip;

    // SSE uses "data:" with an optional single leading space before the value.
    const prefix = "data:";
    if (!std.mem.startsWith(u8, trimmed, prefix)) return .skip;

    const data = if (trimmed.len > prefix.len and trimmed[prefix.len] == ' ')
        trimmed[prefix.len + 1 ..]
    else
        trimmed[prefix.len..];

    if (data.len == 0) return .skip;

    if (std.mem.eql(u8, data, "[DONE]")) return .done;

    const parsed = std.json.parseFromSlice(std.json.Value, allocator, data, .{}) catch |err| {
        if (err == error.OutOfMemory) return err;
        return error.InvalidSseJson;
    };
    defer parsed.deinit();
    if (collector) |tools| try tools.feedValue(allocator, parsed.value);

    const content = try extractDeltaContentValue(allocator, parsed.value) orelse {
        // No content delta — check for usage data (sent in the final chunk).
        if (extractStreamUsageValue(parsed.value)) |u| return .{ .usage = u };
        return .skip;
    };
    return .{ .delta = content };
}

/// Extract `usage` object from an OpenAI-compatible streaming chunk.
/// The final chunk typically has `choices:[]` and a top-level `usage` object.
/// OpenAI usage chunks may contain nested objects (prompt_tokens_details,
/// completion_tokens_details) so we use a generous 32 KB stack buffer.
fn extractStreamUsage(json_str: []const u8) ?root.TokenUsage {
    // 32 KB is sufficient for OpenAI's nested usage objects.
    var buf: [32 * 1024]u8 = undefined;
    var fba = std.heap.FixedBufferAllocator.init(&buf);
    const alloc = fba.allocator();

    const parsed = std.json.parseFromSlice(std.json.Value, alloc, json_str, .{}) catch
        return null;
    defer parsed.deinit();

    return extractStreamUsageValue(parsed.value);
}

fn extractStreamUsageValue(value: std.json.Value) ?root.TokenUsage {
    if (value != .object) return null;
    const obj = value.object;
    const usage_val = obj.get("usage") orelse return null;
    if (usage_val != .object) return null;

    var usage = root.TokenUsage{};
    // Handle prompt_tokens (OpenAI) or input_tokens (Anthropic/some compatible)
    const prompt_val = usage_val.object.get("prompt_tokens") orelse
        usage_val.object.get("input_tokens");
    if (prompt_val) |v| {
        switch (v) {
            .integer => |n| usage.prompt_tokens = @intCast(@max(0, n)),
            .float => |f| usage.prompt_tokens = @intFromFloat(@max(0.0, f)),
            else => {},
        }
    }

    const completion_val = usage_val.object.get("completion_tokens") orelse
        usage_val.object.get("output_tokens");
    if (completion_val) |v| {
        switch (v) {
            .integer => |n| usage.completion_tokens = @intCast(@max(0, n)),
            .float => |f| usage.completion_tokens = @intFromFloat(@max(0.0, f)),
            else => {},
        }
    }

    if (usage_val.object.get("total_tokens")) |v| {
        switch (v) {
            .integer => |n| usage.total_tokens = @intCast(@max(0, n)),
            .float => |f| usage.total_tokens = @intFromFloat(@max(0.0, f)),
            else => {},
        }
    } else {
        usage.total_tokens = usage.prompt_tokens +| usage.completion_tokens;
    }

    // Require at least one non-zero field to treat this as a valid usage chunk.
    // This guards against "usage":null being coerced into a zero struct.
    if (usage.prompt_tokens == 0 and usage.completion_tokens == 0 and usage.total_tokens == 0) {
        return null;
    }

    return usage;
}

/// Extract visible streaming text or reasoning from an SSE JSON payload.
/// Returns owned DeltaContent or null if no content found.
pub fn extractDeltaContent(allocator: std.mem.Allocator, json_str: []const u8) !?DeltaContent {
    if (verbose.isVerbose()) {
        // NOTE: No unit test for this log path; it depends on global verbose
        // logging state. Keep payload bytes out of logs because SSE chunks can
        // contain user prompts, tool results, or model output.
        log.debug("SSE JSON payload received: len={d}", .{json_str.len});
    }

    const parsed = std.json.parseFromSlice(std.json.Value, allocator, json_str, .{}) catch |err| {
        if (verbose.isVerbose()) log.err("Failed to parse SSE JSON payload: len={d} error={s}", .{ json_str.len, @errorName(err) });
        if (err == error.OutOfMemory) return err;
        return error.InvalidSseJson;
    };
    defer parsed.deinit();

    return extractDeltaContentValue(allocator, parsed.value);
}

fn extractDeltaContentValue(allocator: std.mem.Allocator, value: std.json.Value) !?DeltaContent {
    if (value != .object) return null;
    const obj = value.object;
    const choices = obj.get("choices") orelse return null;
    if (choices != .array or choices.array.items.len == 0) return null;

    const first = choices.array.items[0];
    if (first != .object) return null;

    const delta = first.object.get("delta") orelse return null;
    if (delta != .object) return null;

    // Check content first, but only if not empty
    if (delta.object.get("content")) |content| {
        if (content == .string and content.string.len > 0) {
            return .{ .text = try allocator.dupe(u8, content.string) };
        }
    }

    // Fallback to various reasoning fields
    const reasoning_keys = [_][]const u8{ "reasoning", "reasoning_content", "reasoning_details" };
    for (reasoning_keys) |key| {
        if (delta.object.get(key)) |val| {
            if (val == .string and val.string.len > 0) {
                return .{ .reasoning = try allocator.dupe(u8, val.string) };
            }
            if (std.mem.eql(u8, key, "reasoning_details") and val == .array) {
                if (try root.extractReasoningTextFromDetails(allocator, val)) |text| {
                    return .{ .reasoning = text };
                }
            }
        }
    }

    return null;
}

/// Record the provider's own words for a streamed API error so the caller can
/// report WHY the dial failed.
///
/// The blocking parse paths in every provider already do this; the streaming
/// path classified the payload and dropped the text, so an embedder saw the
/// bare error name. `ApiError` alone cannot distinguish a model that does not
/// exist from a rejected credential — `error_classify` collapses both into the
/// same bucket — which is exactly the difference an operator needs.
///
/// Best-effort by construction: a scrub allocation failure records only the
/// mapped error name, never upstream text that may echo credentials.
fn recordStreamApiErrorDetail(
    allocator: std.mem.Allocator,
    root_obj: std.json.ObjectMap,
    mapped_err: anyerror,
) void {
    var summary_buf: [1024]u8 = undefined;
    const summary = error_classify.summarizeKnownApiError(root_obj, &summary_buf) orelse @errorName(mapped_err);
    const sanitized = root.sanitizeApiError(allocator, summary) catch null;
    defer if (sanitized) |s| allocator.free(s);
    root.setLastApiErrorDetail("", sanitized orelse @errorName(mapped_err));
}

/// Record an error payload that is JSON but not a shape `error_classify`
/// recognises. The body is the only evidence there is; scrubbed, it still names
/// the fault where an unadorned `ServerError` names nothing.
fn recordStreamApiErrorBody(allocator: std.mem.Allocator, body: []const u8) void {
    const sanitized = root.sanitizeApiError(allocator, body) catch return;
    defer allocator.free(sanitized);
    root.setLastApiErrorDetail("", sanitized);
}

/// Run curl in SSE streaming mode and parse output line by line.
///
/// Spawns `curl -s --no-buffer` with the strongest supported fail-fast flag:
/// `--fail-with-body` on curl >= 7.76.0, otherwise `-f`.
/// For each SSE delta, calls `callback(ctx, chunk)`.
/// Returns accumulated result after stream completes.
pub fn curlStream(
    allocator: std.mem.Allocator,
    url: []const u8,
    body: []const u8,
    auth_header: ?[]const u8,
    extra_headers: []const []const u8,
    timeout_secs: u64,
    callback: root.StreamCallback,
    ctx: *anyopaque,
) !root.StreamChatResult {
    // Check verbose mode once at function start
    const log_enabled = verbose.isVerbose();
    const debug_log = std.log.scoped(.sse);

    // Build argv on stack (max 40 args)
    var argv_buf: [40][]const u8 = undefined;
    var argc: usize = 0;

    var timeout_buf: [32]u8 = undefined;
    appendCurlStreamBaseArgs(allocator, argv_buf[0..], &argc, &timeout_buf, timeout_secs);

    // Add proxy from environment if set
    const proxy = http_util.getProxyFromEnv(allocator) catch null;
    defer if (proxy) |p| allocator.free(p);

    if (proxy) |p| {
        argv_buf[argc] = "--proxy";
        argc += 1;
        argv_buf[argc] = p;
        argc += 1;
    }

    const resolve_entry = try http_util.buildSafeResolveEntryForRemoteUrl(allocator, url);
    defer if (resolve_entry) |entry| allocator.free(entry);
    http_util.appendCurlResolveArgs(argv_buf[0..], &argc, resolve_entry);

    var header_buf: [16][]const u8 = undefined;
    var header_count: usize = 0;
    header_buf[header_count] = "Content-Type: application/json";
    header_count += 1;
    if (auth_header) |auth| {
        if (header_count >= header_buf.len) return error.TooManyHeaders;
        header_buf[header_count] = auth;
        header_count += 1;
    }

    for (extra_headers) |hdr| {
        if (header_count >= header_buf.len) return error.TooManyHeaders;
        header_buf[header_count] = hdr;
        header_count += 1;
    }

    var prepared_headers = try http_util.prepareCurlHeaderArg(allocator, header_buf[0..header_count]);
    defer prepared_headers.deinit(allocator);
    if (prepared_headers.arg) |headers_arg| {
        argv_buf[argc] = "-H";
        argc += 1;
        argv_buf[argc] = headers_arg;
        argc += 1;
    }

    argv_buf[argc] = "--data-binary";
    argc += 1;
    argv_buf[argc] = "@-";
    argc += 1;
    argv_buf[argc] = url;
    argc += 1;

    if (log_enabled) {
        debug_log.info("curl argc={d}, body_len={d}, header_file={}", .{ argc, body.len, prepared_headers.uses_temp_file });
    }

    if (log_enabled) {
        debug_log.info("spawning curl process...", .{});
    }
    var child = try spawnCurl(allocator, argv_buf[0..argc], body);
    var child_reaped = false;
    defer if (!child_reaped) stopCurl(&child);
    if (log_enabled) {
        const pid: i64 = if (@import("builtin").os.tag == .windows) @intCast(@intFromPtr(child.id)) else child.id;
        debug_log.info("curl process spawned, pid={d}", .{pid});
    }

    // Read stdout line by line, parse SSE events
    var accumulated: std.ArrayListUnmanaged(u8) = .empty;
    defer accumulated.deinit(allocator);

    var reader = line_reader.Reader.init(allocator, child.stdout.?);
    defer reader.deinit();
    var saw_done = false;
    var stream_usage: ?root.TokenUsage = null;
    var in_reasoning = false;
    var tool_collector = stream_tools.Collector{};
    defer tool_collector.deinit(allocator);
    var tool_metadata_error: ?anyerror = null;

    var first_line = true;
    while (try reader.next()) |line| {
        // A non-SSE JSON body is an API error, not a successful empty stream.
        if (first_line and std.mem.startsWith(u8, line, "{")) {
            if (log_enabled) {
                debug_log.info("Detected JSON response, not SSE", .{});
            }
            const parsed = std.json.parseFromSlice(std.json.Value, allocator, line, .{}) catch null;
            if (parsed) |p| {
                defer p.deinit();
                if (p.value == .object) {
                    if (error_classify.classifyKnownApiError(p.value.object)) |kind| {
                        const mapped_err = error_classify.kindToError(kind);
                        recordStreamApiErrorDetail(allocator, p.value.object, mapped_err);
                        return mapped_err;
                    }
                }
            }

            recordStreamApiErrorBody(allocator, line);
            debug_log.err("Server returned JSON error payload: len={d}", .{line.len});
            return error.ServerError;
        }
        first_line = false;
        if (log_enabled) debug_log.info("parsing SSE line: len={d}", .{line.len});
        const result = parseSseLineWithTools(allocator, line, if (tool_metadata_error == null) &tool_collector else null) catch |err| {
            if (err == error.OutOfMemory) return err;
            if (tool_metadata_error == null) tool_metadata_error = err;
            continue;
        };
        switch (result) {
            .delta => |content| {
                defer content.deinit(allocator);
                try appendDeltaContent(allocator, &accumulated, &in_reasoning, callback, ctx, content);
            },
            .usage => |u| stream_usage = u,
            .done => {
                if (log_enabled) debug_log.info("SSE stream done", .{});
                saw_done = true;
                break;
            },
            .skip => {},
        }
    }

    if (log_enabled) {
        debug_log.info("stdout stream ended, saw_done={}, accumulated_len={d}, total_stdout={d}", .{ saw_done, accumulated.items.len, reader.total_read });
    }

    if (saw_done) {
        // A terminal event is complete even when the server keeps its socket open.
        _ = child.kill() catch {};
    }

    if (log_enabled) {
        debug_log.info("waiting for curl process to exit...", .{});
    }
    const term = child.wait() catch |err| {
        log.err("curlStream child.wait failed: {}", .{err});
        if (tool_metadata_error) |metadata_err| return metadata_err;
        if (root.shouldRecoverPartialStream(accumulated.items.len, saw_done)) {
            log.warn("curlStream proceeding despite wait failure after partial stream output", .{});
            try closeReasoningBlock(allocator, &accumulated, &in_reasoning, callback, ctx);
            callback(ctx, root.StreamChunk.finalChunk());
            return finalizeStreamResultWithTools(allocator, accumulated.items, stream_usage, &tool_collector, false);
        }
        return error.CurlWaitError;
    };
    child_reaped = true;
    if (log_enabled) {
        debug_log.info("curl process terminated: {}", .{term});
    }
    if (tool_metadata_error) |metadata_err| return metadata_err;
    if (!saw_done) switch (term) {
        .exited => |code| if (code != 0) {
            if (root.shouldRecoverPartialStream(accumulated.items.len, saw_done)) {
                log.warn("curlStream exit code {d} after partial stream output; returning accumulated output", .{code});
                try closeReasoningBlock(allocator, &accumulated, &in_reasoning, callback, ctx);
                callback(ctx, root.StreamChunk.finalChunk());
                return finalizeStreamResultWithTools(allocator, accumulated.items, stream_usage, &tool_collector, false);
            }
            return error.CurlFailed;
        },
        else => {
            if (root.shouldRecoverPartialStream(accumulated.items.len, saw_done)) {
                log.warn("curlStream abnormal termination after partial stream output; returning accumulated output", .{});
                try closeReasoningBlock(allocator, &accumulated, &in_reasoning, callback, ctx);
                callback(ctx, root.StreamChunk.finalChunk());
                return finalizeStreamResultWithTools(allocator, accumulated.items, stream_usage, &tool_collector, false);
            }
            return error.CurlFailed;
        },
    };

    const stream_complete = saw_done or tool_collector.finish_reason != .unknown;
    if (tool_collector.count > 0 and tool_collector.finish_reason != .tool_calls)
        return error.IncompleteStreamToolCall;
    // Signal stream completion only after curl exits and tool metadata passes validation.
    try closeReasoningBlock(allocator, &accumulated, &in_reasoning, callback, ctx);
    const result = try finalizeStreamResultWithTools(allocator, accumulated.items, stream_usage, &tool_collector, stream_complete);
    callback(ctx, root.StreamChunk.finalChunk());
    return result;
}

// ════════════════════════════════════════════════════════════════════════════
// Anthropic SSE Parsing
// ════════════════════════════════════════════════════════════════════════════

/// Result of parsing a single Anthropic SSE line.
pub const AnthropicSseResult = union(enum) {
    /// Remember this event type (caller tracks state).
    event: []const u8,
    /// Text delta content (owned, caller frees).
    delta: []const u8,
    /// Output token count from message_delta usage.
    usage: u32,
    /// Stream is complete (message_stop).
    done: void,
    /// Line should be skipped (empty, comment, or uninteresting event).
    skip: void,
};

/// Parse a single SSE line in Anthropic streaming format.
///
/// Anthropic SSE is stateful: `event:` lines set the context for subsequent `data:` lines.
/// The caller must track `current_event` across calls.
///
/// - `event: X` → `.event` (caller remembers X)
/// - `data: {JSON}` + current_event=="content_block_delta" → extracts `delta.text` → `.delta`
/// - `data: {JSON}` + current_event=="message_delta" → extracts `usage.output_tokens` → `.usage`
/// - `data: {JSON}` + current_event=="message_stop" → `.done`
/// - Everything else → `.skip`
pub fn parseAnthropicSseLine(allocator: std.mem.Allocator, line: []const u8, current_event: []const u8) !AnthropicSseResult {
    const trimmed = std_compat.mem.trimRight(u8, line, "\r");

    if (trimmed.len == 0) return .skip;
    if (trimmed[0] == ':') return .skip;

    // Handle "event: TYPE" lines
    const event_prefix = "event: ";
    if (std.mem.startsWith(u8, trimmed, event_prefix)) {
        return .{ .event = trimmed[event_prefix.len..] };
    }

    // Handle "data: {JSON}" lines
    const data_prefix = "data: ";
    if (!std.mem.startsWith(u8, trimmed, data_prefix)) return .skip;

    const data = trimmed[data_prefix.len..];

    if (std.mem.eql(u8, current_event, "message_stop")) return .done;

    if (std.mem.eql(u8, current_event, "content_block_delta")) {
        const text = try extractAnthropicDelta(allocator, data) orelse return .skip;
        return .{ .delta = text };
    }

    if (std.mem.eql(u8, current_event, "message_delta")) {
        const tokens = try extractAnthropicUsage(data) orelse return .skip;
        return .{ .usage = tokens };
    }

    return .skip;
}

/// Extract `delta.text` from an Anthropic content_block_delta JSON payload.
/// Returns owned slice or null if not a text_delta.
pub fn extractAnthropicDelta(allocator: std.mem.Allocator, json_str: []const u8) !?[]const u8 {
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, json_str, .{}) catch
        return error.InvalidSseJson;
    defer parsed.deinit();

    const obj = parsed.value.object;
    const delta = obj.get("delta") orelse return null;
    if (delta != .object) return null;

    const dtype = delta.object.get("type") orelse return null;
    if (dtype != .string or !std.mem.eql(u8, dtype.string, "text_delta")) return null;

    const text = delta.object.get("text") orelse return null;
    if (text != .string) return null;
    if (text.string.len == 0) return null;

    return try allocator.dupe(u8, text.string);
}

/// Extract `usage.output_tokens` from an Anthropic message_delta JSON payload.
/// Returns token count or null if not present.
pub fn extractAnthropicUsage(json_str: []const u8) !?u32 {
    // Use a stack buffer for parsing to avoid needing an allocator
    var buf: [4096]u8 = undefined;
    var fba = std.heap.FixedBufferAllocator.init(&buf);
    const allocator = fba.allocator();

    const parsed = std.json.parseFromSlice(std.json.Value, allocator, json_str, .{}) catch
        return error.InvalidSseJson;
    defer parsed.deinit();

    const obj = parsed.value.object;
    const usage = obj.get("usage") orelse return null;
    if (usage != .object) return null;

    const output_tokens = usage.object.get("output_tokens") orelse return null;
    if (output_tokens != .integer) return null;

    return @intCast(output_tokens.integer);
}

/// Run curl in SSE streaming mode for Anthropic and parse output line by line.
///
/// Similar to `curlStream()` but uses stateful Anthropic SSE parsing.
/// `headers` is a slice of pre-formatted header strings (e.g. "x-api-key: sk-...").
pub fn curlStreamAnthropic(
    allocator: std.mem.Allocator,
    url: []const u8,
    body: []const u8,
    headers: []const []const u8,
    callback: root.StreamCallback,
    ctx: *anyopaque,
) !root.StreamChatResult {
    // Build argv on stack (max 40 args)
    var argv_buf: [40][]const u8 = undefined;
    var argc: usize = 0;

    var timeout_buf: [32]u8 = undefined;
    appendCurlStreamBaseArgs(allocator, argv_buf[0..], &argc, &timeout_buf, 0);

    // Add proxy from environment if set
    const proxy = http_util.getProxyFromEnv(allocator) catch null;
    defer if (proxy) |p| allocator.free(p);

    if (proxy) |p| {
        argv_buf[argc] = "--proxy";
        argc += 1;
        argv_buf[argc] = p;
        argc += 1;
    }

    const resolve_entry = try http_util.buildSafeResolveEntryForRemoteUrl(allocator, url);
    defer if (resolve_entry) |entry| allocator.free(entry);
    http_util.appendCurlResolveArgs(argv_buf[0..], &argc, resolve_entry);

    var header_buf: [16][]const u8 = undefined;
    var header_count: usize = 0;
    header_buf[header_count] = "Content-Type: application/json";
    header_count += 1;
    for (headers) |hdr| {
        if (header_count >= header_buf.len) return error.TooManyHeaders;
        header_buf[header_count] = hdr;
        header_count += 1;
    }

    var prepared_headers = try http_util.prepareCurlHeaderArg(allocator, header_buf[0..header_count]);
    defer prepared_headers.deinit(allocator);
    if (prepared_headers.arg) |headers_arg| {
        argv_buf[argc] = "-H";
        argc += 1;
        argv_buf[argc] = headers_arg;
        argc += 1;
    }

    argv_buf[argc] = "--data-binary";
    argc += 1;
    argv_buf[argc] = "@-";
    argc += 1;
    argv_buf[argc] = url;
    argc += 1;

    var child = try spawnCurl(allocator, argv_buf[0..argc], body);
    var child_reaped = false;
    defer if (!child_reaped) stopCurl(&child);

    // Read stdout line by line, parse Anthropic SSE events
    var accumulated: std.ArrayListUnmanaged(u8) = .empty;
    defer accumulated.deinit(allocator);

    var reader = line_reader.Reader.init(allocator, child.stdout.?);
    defer reader.deinit();

    var current_event: []const u8 = "";
    defer if (current_event.len > 0) allocator.free(@constCast(current_event));
    var anthropic_usage: root.TokenUsage = .{};
    var saw_done = false;

    var first_line = true;
    while (try reader.next()) |line| {
        // A body that opens with '{' is a JSON error payload, not an SSE
        // stream. Without this branch every line parses as `.skip`, the loop
        // drains, and the non-zero exit falls through to `CurlFailed` — which
        // reads as a transport fault and names neither the model nor the
        // status. Anthropic answers a model that does not exist with
        // {"type":"error","error":{"type":"not_found_error","message":...}},
        // so classify it the way the OpenAI-compatible stream does and keep
        // the words.
        if (first_line and std.mem.startsWith(u8, line, "{")) {
            const parsed = std.json.parseFromSlice(std.json.Value, allocator, line, .{}) catch null;
            if (parsed) |p| {
                defer p.deinit();
                if (p.value == .object) {
                    if (error_classify.classifyKnownApiError(p.value.object)) |kind| {
                        const mapped_err = error_classify.kindToError(kind);
                        recordStreamApiErrorDetail(allocator, p.value.object, mapped_err);
                        return mapped_err;
                    }
                }
            }

            recordStreamApiErrorBody(allocator, line);
            return error.ServerError;
        }
        first_line = false;
        const result = parseAnthropicSseLine(allocator, line, current_event) catch |err| {
            if (err == error.OutOfMemory) return err;
            continue;
        };
        switch (result) {
            .event => |ev| {
                const next_event = try allocator.dupe(u8, ev);
                if (current_event.len > 0) allocator.free(@constCast(current_event));
                current_event = next_event;
            },
            .delta => |delta| {
                defer allocator.free(delta);
                try appendStreamOutput(allocator, &accumulated, delta);
                callback(ctx, root.StreamChunk.textDelta(delta));
            },
            .usage => |tokens| anthropic_usage.completion_tokens = tokens,
            .done => {
                saw_done = true;
                break;
            },
            .skip => {},
        }
    }

    if (saw_done) {
        _ = child.kill() catch {};
    }

    const term = child.wait() catch |err| {
        log.err("curlStreamAnthropic child.wait failed: {}", .{err});
        if (root.shouldRecoverPartialStream(accumulated.items.len, saw_done)) {
            log.warn("curlStreamAnthropic proceeding despite wait failure after partial stream output", .{});
            callback(ctx, root.StreamChunk.finalChunk());
            return finalizeStreamResult(allocator, accumulated.items, anthropic_usage);
        }
        return error.CurlWaitError;
    };
    child_reaped = true;
    if (!saw_done) switch (term) {
        .exited => |code| if (code != 0) {
            if (root.shouldRecoverPartialStream(accumulated.items.len, saw_done)) {
                log.warn("curlStreamAnthropic exit code {d} after partial stream output; returning accumulated output", .{code});
                callback(ctx, root.StreamChunk.finalChunk());
                return finalizeStreamResult(allocator, accumulated.items, anthropic_usage);
            }
            return error.CurlFailed;
        },
        else => {
            if (root.shouldRecoverPartialStream(accumulated.items.len, saw_done)) {
                log.warn("curlStreamAnthropic abnormal termination after partial stream output; returning accumulated output", .{});
                callback(ctx, root.StreamChunk.finalChunk());
                return finalizeStreamResult(allocator, accumulated.items, anthropic_usage);
            }
            return error.CurlFailed;
        },
    };

    callback(ctx, root.StreamChunk.finalChunk());
    return finalizeStreamResult(allocator, accumulated.items, anthropic_usage);
}

// ════════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════════

test "parseSseLine valid delta" {
    const allocator = std.testing.allocator;
    const result = try parseSseLine(allocator, "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}");
    switch (result) {
        .delta => |d| {
            defer d.deinit(allocator);
            try std.testing.expectEqualStrings("Hello", d.text);
        },
        else => return error.TestUnexpectedResult,
    }
}

test "parseSseLine valid delta without optional space" {
    const allocator = std.testing.allocator;
    const result = try parseSseLine(allocator, "data:{\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}");
    switch (result) {
        .delta => |d| {
            defer d.deinit(allocator);
            try std.testing.expectEqualStrings("Hello", d.text);
        },
        else => return error.TestUnexpectedResult,
    }
}

test "one decoded frame supplies text and tool metadata" {
    const allocator = std.testing.allocator;
    var collector = stream_tools.Collector{};
    defer collector.deinit(allocator);
    const line = "data: {\"choices\":[{\"delta\":{\"content\":\"working\",\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"probe\",\"arguments\":\"{}\"}}]},\"finish_reason\":\"tool_calls\"}]}";
    const result = try parseSseLineWithTools(allocator, line, &collector);
    switch (result) {
        .delta => |content| {
            defer content.deinit(allocator);
            try std.testing.expectEqualStrings("working", content.text);
        },
        else => return error.TestUnexpectedResult,
    }
    const calls = try collector.take(allocator);
    defer {
        for (calls) |call| {
            allocator.free(call.id);
            allocator.free(call.name);
            allocator.free(call.arguments);
        }
        allocator.free(calls);
    }
    try std.testing.expectEqual(@as(usize, 1), calls.len);
    try std.testing.expectEqualStrings("probe", calls[0].name);
    try std.testing.expectEqual(root.StreamFinishReason.tool_calls, collector.finish_reason);
}

test "appendCurlStallDetectionArgs appends curl speed flags in order" {
    // Regression: stalled SSE streams must trip curl's speed-limit instead of
    // hanging until --max-time expires with an idle-but-open connection.
    var argv_buf: [8][]const u8 = undefined;
    var argc: usize = 0;
    appendCurlStallDetectionArgs(argv_buf[0..], &argc);

    try std.testing.expectEqual(@as(usize, 4), argc);
    try std.testing.expectEqualStrings("--speed-limit", argv_buf[0]);
    try std.testing.expectEqualStrings("1", argv_buf[1]);
    try std.testing.expectEqualStrings("--speed-time", argv_buf[2]);
    try std.testing.expectEqualStrings("60", argv_buf[3]);
}

test "parseSseLine DONE sentinel" {
    const result = try parseSseLine(std.testing.allocator, "data: [DONE]");
    try std.testing.expect(result == .done);
}

test "parseSseLine DONE sentinel without optional space" {
    const result = try parseSseLine(std.testing.allocator, "data:[DONE]");
    try std.testing.expect(result == .done);
}

test "curlVersionSupportsFailWithBody rejects curl older than 7.76.0" {
    try std.testing.expect(!curlVersionSupportsFailWithBody("curl 7.68.0 (x86_64-pc-linux-gnu) libcurl/7.68.0"));
}

test "curlVersionSupportsFailWithBody accepts curl 7.76.0 and newer" {
    try std.testing.expect(curlVersionSupportsFailWithBody("curl 7.76.0 (x86_64-pc-linux-gnu) libcurl/7.76.0"));
    try std.testing.expect(curlVersionSupportsFailWithBody("curl 8.17.0 (x86_64-alpine-linux-musl) libcurl/8.17.0"));
}

test "curlVersionSupportsFailWithBody tolerates suffixes in version token" {
    try std.testing.expect(curlVersionSupportsFailWithBody("curl 8.17.0-DEV (x86_64) libcurl/8.17.0"));
}

test "parseSseLine empty line" {
    const result = try parseSseLine(std.testing.allocator, "");
    try std.testing.expect(result == .skip);
}

test "parseSseLine comment" {
    const result = try parseSseLine(std.testing.allocator, ":keep-alive");
    try std.testing.expect(result == .skip);
}

test "parseSseLine empty data field" {
    const result = try parseSseLine(std.testing.allocator, "data:");
    try std.testing.expect(result == .skip);
}

test "parseSseLine delta without content" {
    const result = try parseSseLine(std.testing.allocator, "data: {\"choices\":[{\"delta\":{}}]}");
    try std.testing.expect(result == .skip);
}

test "parseSseLine empty choices" {
    const result = try parseSseLine(std.testing.allocator, "data: {\"choices\":[]}");
    try std.testing.expect(result == .skip);
}

test "parseSseLine invalid JSON" {
    try std.testing.expectError(error.InvalidSseJson, parseSseLine(std.testing.allocator, "data: not-json{{{"));
}

test "extractDeltaContent with content" {
    const allocator = std.testing.allocator;
    const d = (try extractDeltaContent(allocator, "{\"choices\":[{\"delta\":{\"content\":\"world\"}}]}")).?;
    defer d.deinit(allocator);
    try std.testing.expectEqualStrings("world", d.text);
}

test "extractDeltaContent without content" {
    const result = try extractDeltaContent(std.testing.allocator, "{\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}");
    try std.testing.expect(result == null);
}

test "extractDeltaContent empty content" {
    const result = try extractDeltaContent(std.testing.allocator, "{\"choices\":[{\"delta\":{\"content\":\"\"}}]}");
    try std.testing.expect(result == null);
}

test "extractDeltaContent falls back to reasoning_content when content empty" {
    const allocator = std.testing.allocator;
    const d = (try extractDeltaContent(allocator, "{\"choices\":[{\"delta\":{\"content\":\"\",\"reasoning_content\":\"step by step\"}}]}")).?;
    defer d.deinit(allocator);
    try std.testing.expectEqualStrings("step by step", d.reasoning);
}

// Regression: OpenRouter's documented chat stream uses delta.reasoning.
test "extractDeltaContent falls back to reasoning when content missing" {
    const allocator = std.testing.allocator;
    const d = (try extractDeltaContent(allocator, "{\"choices\":[{\"delta\":{\"reasoning\":\"step by step\"}}]}")).?;
    defer d.deinit(allocator);
    try std.testing.expectEqualStrings("step by step", d.reasoning);
}

test "extractDeltaContent falls back to reasoning_content when content missing" {
    const allocator = std.testing.allocator;
    const d = (try extractDeltaContent(allocator, "{\"choices\":[{\"delta\":{\"reasoning_content\":\"step by step\"}}]}")).?;
    defer d.deinit(allocator);
    try std.testing.expectEqualStrings("step by step", d.reasoning);
}

// Regression: OpenRouter's current normalized streaming shape uses reasoning_details.
test "extractDeltaContent falls back to reasoning_details when content missing" {
    const allocator = std.testing.allocator;
    const d = (try extractDeltaContent(
        allocator,
        "{\"choices\":[{\"delta\":{\"reasoning_details\":[{\"type\":\"reasoning.summary\",\"summary\":\"plan\"},{\"type\":\"reasoning.text\",\"text\":\"step by step\"}]}}]}",
    )).?;
    defer d.deinit(allocator);
    try std.testing.expectEqualStrings("plan\nstep by step", d.reasoning);
}

test "extractDeltaContent prefers visible content over reasoning_content" {
    const allocator = std.testing.allocator;
    const d = (try extractDeltaContent(allocator, "{\"choices\":[{\"delta\":{\"content\":\"final answer\",\"reasoning_content\":\"private\"}}]}")).?;
    defer d.deinit(allocator);
    try std.testing.expectEqualStrings("final answer", d.text);
}

test "appendDeltaContent closes reasoning before final" {
    const Collector = struct {
        buf: std.ArrayListUnmanaged(u8) = .empty,
        saw_final: bool = false,

        fn callback(ctx: *anyopaque, chunk: root.StreamChunk) void {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            if (chunk.is_final) {
                self.saw_final = true;
                return;
            }
            self.buf.appendSlice(std.testing.allocator, chunk.delta) catch unreachable;
        }
    };

    const allocator = std.testing.allocator;
    var collector = Collector{};
    defer collector.buf.deinit(allocator);
    var accumulated: std.ArrayListUnmanaged(u8) = .empty;
    defer accumulated.deinit(allocator);

    var in_reasoning = false;
    const content = DeltaContent{ .reasoning = try allocator.dupe(u8, "private") };
    defer content.deinit(allocator);

    try appendDeltaContent(allocator, &accumulated, &in_reasoning, Collector.callback, @ptrCast(&collector), content);
    try closeReasoningBlock(allocator, &accumulated, &in_reasoning, Collector.callback, @ptrCast(&collector));
    Collector.callback(@ptrCast(&collector), root.StreamChunk.finalChunk());

    try std.testing.expect(collector.saw_final);
    try std.testing.expect(!in_reasoning);
    try std.testing.expectEqualStrings("<think>private</think>", accumulated.items);
    try std.testing.expectEqualStrings("<think>private</think>", collector.buf.items);
}

test "extractDeltaContent empty reasoning_content returns null" {
    const result = try extractDeltaContent(std.testing.allocator, "{\"choices\":[{\"delta\":{\"reasoning_content\":\"\"}}]}");
    try std.testing.expect(result == null);
}

test "StreamChunk textDelta token estimate" {
    const chunk = root.StreamChunk.textDelta("12345678");
    try std.testing.expect(chunk.token_count == 2);
    try std.testing.expect(!chunk.is_final);
    try std.testing.expectEqualStrings("12345678", chunk.delta);
}

test "StreamChunk finalChunk" {
    const chunk = root.StreamChunk.finalChunk();
    try std.testing.expect(chunk.is_final);
    try std.testing.expectEqualStrings("", chunk.delta);
    try std.testing.expect(chunk.token_count == 0);
}

// ── Anthropic SSE Tests ─────────────────────────────────────────

test "parseAnthropicSseLine event line returns event" {
    const result = try parseAnthropicSseLine(std.testing.allocator, "event: content_block_delta", "");
    switch (result) {
        .event => |ev| try std.testing.expectEqualStrings("content_block_delta", ev),
        else => return error.TestUnexpectedResult,
    }
}

test "parseAnthropicSseLine data with content_block_delta returns delta" {
    const allocator = std.testing.allocator;
    const json = "data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello\"}}";
    const result = try parseAnthropicSseLine(allocator, json, "content_block_delta");
    switch (result) {
        .delta => |text| {
            defer allocator.free(text);
            try std.testing.expectEqualStrings("Hello", text);
        },
        else => return error.TestUnexpectedResult,
    }
}

test "parseAnthropicSseLine data with message_delta returns usage" {
    const json = "data: {\"type\":\"message_delta\",\"delta\":{},\"usage\":{\"output_tokens\":42}}";
    const result = try parseAnthropicSseLine(std.testing.allocator, json, "message_delta");
    switch (result) {
        .usage => |tokens| try std.testing.expect(tokens == 42),
        else => return error.TestUnexpectedResult,
    }
}

test "parseAnthropicSseLine data with message_stop returns done" {
    const result = try parseAnthropicSseLine(std.testing.allocator, "data: {\"type\":\"message_stop\"}", "message_stop");
    try std.testing.expect(result == .done);
}

test "parseAnthropicSseLine empty line returns skip" {
    const result = try parseAnthropicSseLine(std.testing.allocator, "", "");
    try std.testing.expect(result == .skip);
}

test "parseAnthropicSseLine comment returns skip" {
    const result = try parseAnthropicSseLine(std.testing.allocator, ":keep-alive", "");
    try std.testing.expect(result == .skip);
}

test "parseAnthropicSseLine data with unknown event returns skip" {
    const json = "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_123\"}}";
    const result = try parseAnthropicSseLine(std.testing.allocator, json, "message_start");
    try std.testing.expect(result == .skip);
}

test "extractAnthropicDelta correct JSON returns text" {
    const allocator = std.testing.allocator;
    const json = "{\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"world\"}}";
    const result = (try extractAnthropicDelta(allocator, json)).?;
    defer allocator.free(result);
    try std.testing.expectEqualStrings("world", result);
}

test "extractAnthropicDelta without text returns null" {
    const json = "{\"type\":\"content_block_delta\",\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{}\"}}";
    const result = try extractAnthropicDelta(std.testing.allocator, json);
    try std.testing.expect(result == null);
}

test "extractAnthropicUsage correct JSON returns token count" {
    const json = "{\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":57}}";
    const result = (try extractAnthropicUsage(json)).?;
    try std.testing.expect(result == 57);
}

// ── Stream Usage Extraction Tests ───────────────────────────────

test "extractStreamUsage returns full usage from final chunk" {
    const json = "{\"id\":\"chatcmpl-abc\",\"choices\":[],\"usage\":{\"prompt_tokens\":100,\"completion_tokens\":263,\"total_tokens\":363}}";
    const usage = extractStreamUsage(json).?;
    try std.testing.expectEqual(@as(u32, 100), usage.prompt_tokens);
    try std.testing.expectEqual(@as(u32, 263), usage.completion_tokens);
    try std.testing.expectEqual(@as(u32, 363), usage.total_tokens);
}

test "extractStreamUsage returns null for chunk without usage" {
    const json = "{\"id\":\"chatcmpl-abc\",\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}";
    try std.testing.expect(extractStreamUsage(json) == null);
}

test "extractStreamUsage returns null for invalid JSON" {
    try std.testing.expect(extractStreamUsage("not-json{{{") == null);
}

test "finalizeStreamResult separates think blocks into reasoning content" {
    const result = try finalizeStreamResult(
        std.testing.allocator,
        "<think>private trace</think>Visible answer",
        .{ .completion_tokens = 4, .total_tokens = 4 },
    );
    defer {
        if (result.content) |content| std.testing.allocator.free(content);
        if (result.reasoning_content) |reasoning| std.testing.allocator.free(reasoning);
    }

    try std.testing.expectEqualStrings("Visible answer", result.content.?);
    try std.testing.expectEqualStrings("private trace", result.reasoning_content.?);
}

test "parseSseLine extracts usage from final chunk" {
    const allocator = std.testing.allocator;
    const line = "data: {\"id\":\"chatcmpl-abc\",\"choices\":[],\"usage\":{\"prompt_tokens\":50,\"completion_tokens\":20,\"total_tokens\":70}}";
    const result = try parseSseLine(allocator, line);
    switch (result) {
        .usage => |u| {
            try std.testing.expectEqual(@as(u32, 50), u.prompt_tokens);
            try std.testing.expectEqual(@as(u32, 20), u.completion_tokens);
            try std.testing.expectEqual(@as(u32, 70), u.total_tokens);
        },
        else => return error.TestUnexpectedResult,
    }
}

// Regression: OpenAI usage chunks contain nested objects (prompt_tokens_details,
// completion_tokens_details) that exceeded the old 4096-byte FixedBufferAllocator,
// silently returning null from extractStreamUsage and causing prompt_tokens=0.
test "parseSseLine extracts usage from OpenAI nested usage chunk" {
    const allocator = std.testing.allocator;
    const line = "data: {\"id\":\"chatcmpl-De08d\",\"object\":\"chat.completion.chunk\"," ++
        "\"created\":1778426023,\"model\":\"gpt-4o-2024-08-06\"," ++
        "\"service_tier\":\"default\",\"system_fingerprint\":\"fp_5acb5510d6\"," ++
        "\"choices\":[],\"usage\":{\"prompt_tokens\":9,\"completion_tokens\":9,\"total_tokens\":18," ++
        "\"prompt_tokens_details\":{\"cached_tokens\":0,\"audio_tokens\":0}," ++
        "\"completion_tokens_details\":{\"reasoning_tokens\":0,\"audio_tokens\":0," ++
        "\"accepted_prediction_tokens\":0,\"rejected_prediction_tokens\":0}}}";
    const result = try parseSseLine(allocator, line);
    switch (result) {
        .usage => |u| {
            try std.testing.expectEqual(@as(u32, 9), u.prompt_tokens);
            try std.testing.expectEqual(@as(u32, 9), u.completion_tokens);
            try std.testing.expectEqual(@as(u32, 18), u.total_tokens);
        },
        else => return error.TestUnexpectedResult,
    }
}

test "extractStreamUsage returns null for null usage field" {
    // Intermediate OpenAI chunks have "usage":null — must not produce a zero struct.
    const json =
        \\{"id":"chatcmpl-abc","choices":[{"delta":{"content":"Hi"}}],"usage":null}
    ;
    try std.testing.expect(extractStreamUsage(json) == null);
}

test "a streamed 404 error payload records the provider's own words" {
    // The regression: a fleet pinned to a model no provider serves streamed its
    // dial, so the 404 body went through this file — which classified it and
    // threw the text away. The embedder then reported the bare word `ApiError`,
    // indistinguishable from a rejected credential.
    const allocator = std.testing.allocator;
    root.clearLastApiErrorDetail();
    defer root.clearLastApiErrorDetail();

    const body =
        \\{"error":{"object":"error","type":"invalid_request_error","message":"Model not found, inaccessible, and/or not deployed","status":404}}
    ;
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, body, .{});
    defer parsed.deinit();

    const kind = error_classify.classifyKnownApiError(parsed.value.object).?;
    const mapped_err = error_classify.kindToError(kind);
    try std.testing.expectEqual(error.ApiError, mapped_err);
    recordStreamApiErrorDetail(allocator, parsed.value.object, mapped_err);

    const detail = (try root.snapshotLastApiErrorDetail(allocator)).?;
    defer allocator.free(detail);
    try std.testing.expect(std.mem.indexOf(u8, detail, "404") != null);
    try std.testing.expect(std.mem.indexOf(u8, detail, "Model not found") != null);
}

test "a streamed rate-limit payload keeps its status alongside the mapped error" {
    const allocator = std.testing.allocator;
    root.clearLastApiErrorDetail();
    defer root.clearLastApiErrorDetail();

    const body =
        \\{"error":{"message":"Rate limit exceeded","type":"rate_limit_error","status":429}}
    ;
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, body, .{});
    defer parsed.deinit();

    const kind = error_classify.classifyKnownApiError(parsed.value.object).?;
    try std.testing.expectEqual(error.RateLimited, error_classify.kindToError(kind));
    recordStreamApiErrorDetail(allocator, parsed.value.object, error_classify.kindToError(kind));

    const detail = (try root.snapshotLastApiErrorDetail(allocator)).?;
    defer allocator.free(detail);
    try std.testing.expect(std.mem.indexOf(u8, detail, "429") != null);
    try std.testing.expect(std.mem.indexOf(u8, detail, "Rate limit exceeded") != null);
}

test "an unrecognised JSON error payload still leaves the body as evidence" {
    const allocator = std.testing.allocator;
    root.clearLastApiErrorDetail();
    defer root.clearLastApiErrorDetail();

    recordStreamApiErrorBody(allocator, "{\"detail\":\"upstream connect error\"}");

    const detail = (try root.snapshotLastApiErrorDetail(allocator)).?;
    defer allocator.free(detail);
    try std.testing.expect(std.mem.indexOf(u8, detail, "upstream connect error") != null);
}

test "an Anthropic streamed model-not-found payload names the model, not the transport" {
    // Before: the body parsed as SSE lines that all skipped, the loop drained,
    // and the non-zero exit surfaced CurlFailed — a transport error name for a
    // provider rejection, with the message discarded.
    const allocator = std.testing.allocator;
    root.clearLastApiErrorDetail();
    defer root.clearLastApiErrorDetail();

    const body =
        \\{"type":"error","error":{"type":"not_found_error","message":"model: claude-does-not-exist"}}
    ;
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, body, .{});
    defer parsed.deinit();

    const kind = error_classify.classifyKnownApiError(parsed.value.object).?;
    const mapped_err = error_classify.kindToError(kind);
    try std.testing.expectEqual(error.ApiError, mapped_err);
    recordStreamApiErrorDetail(allocator, parsed.value.object, mapped_err);

    const detail = (try root.snapshotLastApiErrorDetail(allocator)).?;
    defer allocator.free(detail);
    try std.testing.expect(std.mem.indexOf(u8, detail, "not_found_error") != null);
    try std.testing.expect(std.mem.indexOf(u8, detail, "claude-does-not-exist") != null);
}

test "an Anthropic streamed overload payload maps to the rate-limit bucket" {
    const allocator = std.testing.allocator;
    root.clearLastApiErrorDetail();
    defer root.clearLastApiErrorDetail();

    const body =
        \\{"type":"error","error":{"type":"rate_limit_error","message":"Number of requests has exceeded your rate limit","status":429}}
    ;
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, body, .{});
    defer parsed.deinit();

    const kind = error_classify.classifyKnownApiError(parsed.value.object).?;
    try std.testing.expectEqual(error.RateLimited, error_classify.kindToError(kind));
    recordStreamApiErrorDetail(allocator, parsed.value.object, error_classify.kindToError(kind));

    const detail = (try root.snapshotLastApiErrorDetail(allocator)).?;
    defer allocator.free(detail);
    try std.testing.expect(std.mem.indexOf(u8, detail, "429") != null);
}

const LOCAL_STREAM_CALLS = 100;
const LOCAL_STREAM_WORKERS = 10;
const LOCAL_STREAM_RESPONSE_BODY = "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\ndata: [DONE]\n\n";
const LOCAL_STREAM_RESPONSE = std.fmt.comptimePrint(
    "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {d}\r\nConnection: close\r\n\r\n{s}",
    .{ LOCAL_STREAM_RESPONSE_BODY.len, LOCAL_STREAM_RESPONSE_BODY },
);

fn readLocalStreamRequest(stream: std_compat.net.Stream) !void {
    var request: [2048]u8 = undefined;
    var used: usize = 0;
    while (used < request.len) {
        const n = try stream.read(request[used..]);
        if (n == 0) return error.TestUnexpectedResult;
        used += n;
        const end = std.mem.indexOf(u8, request[0..used], "\r\n\r\n") orelse continue;
        if (used >= end + 6) return; // Header terminator plus the two-byte body.
    }
    return error.TestUnexpectedResult;
}

const LocalStreamServer = struct {
    server: *std_compat.net.Server,
    stop: std.atomic.Value(bool) = .init(false),
    accepted: usize = 0,

    fn stopAndJoin(self: *LocalStreamServer, thread: *std.Thread) void {
        self.stop.store(true, .release);
        const unblock = std_compat.net.tcpConnectToAddress(self.server.listen_address) catch null;
        if (unblock) |conn| conn.close();
        thread.join();
    }

    fn run(self: *LocalStreamServer) void {
        while (true) {
            var conn = self.server.accept() catch return;
            defer conn.stream.close();
            if (self.stop.load(.acquire)) return;
            self.accepted += 1;
            readLocalStreamRequest(conn.stream) catch return;
            conn.stream.writeAll(LOCAL_STREAM_RESPONSE) catch return;
        }
    }
};

const LocalStreamSample = struct {
    first_ms: u64 = 0,
    final_ms: u64 = 0,
    success: bool = false,
};

const LocalStreamCallback = struct {
    first: i96 = 0,
    final_calls: usize = 0,

    fn onChunk(ptr: *anyopaque, chunk: root.StreamChunk) void {
        const self: *LocalStreamCallback = @ptrCast(@alignCast(ptr));
        if (chunk.is_final) {
            self.final_calls += 1;
        } else if (chunk.delta.len > 0 and self.first == 0) {
            self.first = std.Io.Clock.awake.now(std_compat.io()).nanoseconds;
        }
    }
};

const LocalStreamWorker = struct {
    url: []const u8,
    samples: []LocalStreamSample,

    fn run(self: *LocalStreamWorker) void {
        for (self.samples) |*sample| {
            const started = std.Io.Clock.awake.now(std_compat.io()).nanoseconds;
            var callback = LocalStreamCallback{};
            const result = curlStream(std.heap.page_allocator, self.url, "{}", null, &.{}, 5, LocalStreamCallback.onChunk, &callback) catch continue;
            defer {
                if (result.content) |content| std.heap.page_allocator.free(content);
                if (result.reasoning_content) |reasoning| std.heap.page_allocator.free(reasoning);
            }
            const finished = std.Io.Clock.awake.now(std_compat.io()).nanoseconds;
            sample.success = callback.first != 0 and callback.final_calls == 1 and
                result.content != null and std.mem.eql(u8, result.content.?, "ok");
            if (sample.success) {
                sample.first_ms = @intCast(@divTrunc(callback.first - started, std.time.ns_per_ms));
                sample.final_ms = @intCast(@divTrunc(finished - started, std.time.ns_per_ms));
            }
        }
    }
};

test "100 local SSE calls return complete replies with ten concurrent workers" {
    if (comptime @import("builtin").os.tag == .wasi) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const addr = try std_compat.net.Address.resolveIp("127.0.0.1", 0);
    var server = try addr.listen(.{});
    defer server.deinit();
    var serving = LocalStreamServer{ .server = &server };
    var server_thread = try std.Thread.spawn(.{}, LocalStreamServer.run, .{&serving});
    var server_joined = false;
    defer if (!server_joined) serving.stopAndJoin(&server_thread);

    const url = try std.fmt.allocPrint(allocator, "http://127.0.0.1:{d}/stream", .{server.listen_address.in.getPort()});
    defer allocator.free(url);
    var samples = [_]LocalStreamSample{.{}} ** LOCAL_STREAM_CALLS;
    var workers: [LOCAL_STREAM_WORKERS]LocalStreamWorker = undefined;
    var threads: [LOCAL_STREAM_WORKERS]std.Thread = undefined;
    for (&workers, 0..) |*worker, i| {
        const begin = i * (LOCAL_STREAM_CALLS / LOCAL_STREAM_WORKERS);
        worker.* = .{ .url = url, .samples = samples[begin .. begin + LOCAL_STREAM_CALLS / LOCAL_STREAM_WORKERS] };
        threads[i] = try std.Thread.spawn(.{}, LocalStreamWorker.run, .{worker});
    }
    for (&threads) |*thread| thread.join();
    serving.stopAndJoin(&server_thread);
    server_joined = true;

    var first: [LOCAL_STREAM_CALLS]u64 = undefined;
    var final: [LOCAL_STREAM_CALLS]u64 = undefined;
    for (samples, 0..) |sample, i| {
        try std.testing.expect(sample.success);
        first[i] = sample.first_ms;
        final[i] = sample.final_ms;
    }
    try std.testing.expectEqual(@as(usize, LOCAL_STREAM_CALLS), serving.accepted);
    std.mem.sortUnstable(u64, &first, {}, std.sort.asc(u64));
    std.mem.sortUnstable(u64, &final, {}, std.sort.asc(u64));
    std.debug.print("local SSE 100/100: first p50={d}ms p95={d}ms p99={d}ms; final p50={d}ms p95={d}ms p99={d}ms\n", .{
        first[49], first[94], first[98], final[49], final[94], final[98],
    });
}

const HeldOpenStreamServer = struct {
    server: *std_compat.net.Server,
    caller_returned: *std.atomic.Value(bool),
    saw_caller_return: bool = false,

    fn run(self: *HeldOpenStreamServer) void {
        var conn = self.server.accept() catch return;
        defer conn.stream.close();
        readLocalStreamRequest(conn.stream) catch return;
        const response = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: keep-alive\r\n\r\n" ++ LOCAL_STREAM_RESPONSE_BODY;
        conn.stream.writeAll(response) catch return;
        for (0..200) |_| {
            if (self.caller_returned.load(.acquire)) {
                self.saw_caller_return = true;
                return;
            }
            std_compat.thread.sleep(10 * std.time.ns_per_ms);
        }
    }
};

test "terminal SSE event completes before a provider closes its socket" {
    if (comptime @import("builtin").os.tag == .wasi) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const addr = try std_compat.net.Address.resolveIp("127.0.0.1", 0);
    var server = try addr.listen(.{});
    defer server.deinit();
    var returned = std.atomic.Value(bool).init(false);
    var held_open = HeldOpenStreamServer{ .server = &server, .caller_returned = &returned };
    var thread = try std.Thread.spawn(.{}, HeldOpenStreamServer.run, .{&held_open});
    var joined = false;
    defer if (!joined) thread.join();
    const url = try std.fmt.allocPrint(allocator, "http://127.0.0.1:{d}/stream", .{server.listen_address.in.getPort()});
    defer allocator.free(url);
    var callback = LocalStreamCallback{};
    const result = try curlStream(allocator, url, "{}", null, &.{}, 4, LocalStreamCallback.onChunk, &callback);
    defer if (result.content) |content| allocator.free(content);
    returned.store(true, .release);
    thread.join();
    joined = true;
    try std.testing.expect(held_open.saw_caller_return);
    try std.testing.expectEqual(@as(usize, 1), callback.final_calls);
    try std.testing.expectEqualStrings("ok", result.content.?);
}
