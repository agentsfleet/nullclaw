const std = @import("std");
const std_compat = @import("compat");
const root = @import("root.zig");
const error_classify = @import("error_classify.zig");
const verbose = @import("../verbose.zig");
const stream_tools = @import("sse_tool_calls.zig");
const native_sse = @import("native_sse.zig");
const log = std.log.scoped(.provider_sse);

const MAX_STREAM_OUTPUT_BYTES: usize = 4 * 1024 * 1024;

pub fn appendStreamOutput(allocator: std.mem.Allocator, output: *std.ArrayListUnmanaged(u8), bytes: []const u8) !void {
    if (bytes.len > MAX_STREAM_OUTPUT_BYTES - output.items.len) return error.StreamOutputTooLarge;
    try output.appendSlice(allocator, bytes);
}

fn finalizeStreamResultWithTools(
    allocator: std.mem.Allocator,
    accumulated: *std.ArrayListUnmanaged(u8),
    stream_usage: ?root.TokenUsage,
    collector: *stream_tools.Collector,
    allow_tool_calls: bool,
) !root.StreamChatResult {
    var content: ?[]const u8 = null;
    var reasoning_content: ?[]const u8 = null;
    const output_len = accumulated.items.len;
    if (output_len > 0) {
        if (std.mem.indexOf(u8, accumulated.items, "<think>") == null and
            std.mem.indexOf(u8, accumulated.items, "</think>") == null)
        {
            content = try accumulated.toOwnedSlice(allocator);
        } else {
            const split = try root.splitThinkContent(allocator, accumulated.items);
            content = split.visible;
            reasoning_content = split.reasoning;
        }
    }
    errdefer {
        if (content) |text| allocator.free(text);
        if (reasoning_content) |text| allocator.free(text);
    }

    var usage = stream_usage orelse root.TokenUsage{};
    if (usage.completion_tokens == 0) {
        usage.completion_tokens = @intCast((output_len + 3) / 4);
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
    accumulated: *std.ArrayListUnmanaged(u8),
    stream_usage: ?root.TokenUsage,
) !root.StreamChatResult {
    var collector = stream_tools.Collector{};
    return finalizeStreamResultWithTools(allocator, accumulated, stream_usage, &collector, false);
}

fn validateToolFinish(collector: *const stream_tools.Collector) !void {
    if (collector.count > 0 and collector.finish_reason != .tool_calls)
        return error.IncompleteStreamToolCall;
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
            callback(ctx, root.StreamChunk.answerDelta(text));
        },
        .reasoning => |reasoning| {
            if (!in_reasoning.*) {
                in_reasoning.* = true;
                try appendStreamOutput(allocator, accumulated, THINK_OPEN_TAG);
                callback(ctx, root.StreamChunk.textDelta(THINK_OPEN_TAG));
            }
            try appendStreamOutput(allocator, accumulated, reasoning);
            callback(ctx, root.StreamChunk.reasoningDelta(reasoning));
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
    const data = openAiDataPayload(line) orelse return .skip;
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

fn openAiDataPayload(line: []const u8) ?[]const u8 {
    const trimmed = std_compat.mem.trimRight(u8, line, "\r");
    if (!std.mem.startsWith(u8, trimmed, "data:")) return null;
    const value = trimmed[5..];
    return if (std.mem.startsWith(u8, value, " ")) value[1..] else value;
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

/// Classify and scrub an initial JSON provider error shared by SSE dialects.
pub fn initialJsonError(allocator: std.mem.Allocator, line: []const u8) ?anyerror {
    if (!std.mem.startsWith(u8, line, "{")) return null;
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, line, .{}) catch null;
    if (parsed) |payload| {
        defer payload.deinit();
        if (payload.value == .object) {
            if (error_classify.classifyKnownApiError(payload.value.object)) |kind| {
                const mapped = error_classify.kindToError(kind);
                recordStreamApiErrorDetail(allocator, payload.value.object, mapped);
                return mapped;
            }
        }
    }
    recordStreamApiErrorBody(allocator, line);
    return error.ServerError;
}

const OpenAiStream = struct {
    allocator: std.mem.Allocator,
    callback: root.StreamCallback,
    ctx: *anyopaque,
    accumulated: std.ArrayListUnmanaged(u8) = .empty,
    usage: ?root.TokenUsage = null,
    in_reasoning: bool = false,
    tools: stream_tools.Collector = .{},
    metadata_error: ?anyerror = null,
    first_line: bool = true,
    saw_done: bool = false,

    fn deinit(self: *OpenAiStream) void {
        self.accumulated.deinit(self.allocator);
        self.tools.deinit(self.allocator);
    }

    fn onLine(context: *anyopaque, line: []const u8) anyerror!bool {
        const self: *OpenAiStream = @ptrCast(@alignCast(context));
        if (self.first_line) if (initialJsonError(self.allocator, line)) |err| return err;
        self.first_line = false;
        const result = parseSseLineWithTools(self.allocator, line, if (self.metadata_error == null) &self.tools else null) catch |err| {
            if (err == error.OutOfMemory) return err;
            // Plain-text provider heartbeats can be skipped before tool
            // assembly. A malformed JSON object might be a tool argument
            // fragment, so it must fail the turn rather than alter a call.
            if (err == error.InvalidSseJson and self.tools.count == 0) {
                if (openAiDataPayload(line)) |data| {
                    if (data.len > 0 and data[0] != '{' and data[0] != '[') return true;
                }
            }
            if (self.metadata_error == null) self.metadata_error = err;
            return true;
        };
        switch (result) {
            .delta => |content| {
                defer content.deinit(self.allocator);
                try appendDeltaContent(self.allocator, &self.accumulated, &self.in_reasoning, self.callback, self.ctx, content);
            },
            .usage => |usage| self.usage = usage,
            .done => {
                self.saw_done = true;
                return false;
            },
            .skip => {},
        }
        return true;
    }
};

test "OpenAI stream skips plain-text heartbeat frames after visible deltas" {
    const Ignore = struct {
        fn onChunk(_: *anyopaque, _: root.StreamChunk) void {}
    };
    var marker: u8 = 0;
    var stream = OpenAiStream{ .allocator = std.testing.allocator, .callback = Ignore.onChunk, .ctx = &marker };
    defer stream.deinit();
    try std.testing.expect(try OpenAiStream.onLine(&stream, "data: {\"choices\":[{\"delta\":{\"content\":\"first\"}}]}"));
    try std.testing.expect(try OpenAiStream.onLine(&stream, "data: heartbeat"));
    try std.testing.expect(try OpenAiStream.onLine(&stream, "data:heartbeat"));
    try std.testing.expect(try OpenAiStream.onLine(&stream, "data: {\"choices\":[{\"delta\":{\"content\":\" second\"}}]}"));
    try std.testing.expect(!(try OpenAiStream.onLine(&stream, "data: [DONE]")));
    try std.testing.expect(stream.metadata_error == null);
    try std.testing.expectEqualStrings("first second", stream.accumulated.items);
}

test "OpenAI stream rejects malformed JSON between tool argument fragments" {
    const Ignore = struct {
        fn onChunk(_: *anyopaque, _: root.StreamChunk) void {}
    };
    var marker: u8 = 0;
    var stream = OpenAiStream{ .allocator = std.testing.allocator, .callback = Ignore.onChunk, .ctx = &marker };
    defer stream.deinit();
    try std.testing.expect(try OpenAiStream.onLine(&stream, "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"function\":{\"name\":\"memory_store\",\"arguments\":\"{\\\"content\\\":\\\"first\"}}]}}]}"));
    try std.testing.expect(try OpenAiStream.onLine(&stream, "data: {not-json}"));
    try std.testing.expect(stream.metadata_error.? == error.InvalidSseJson);
}

/// Stream a provider reply through in-process libcurl and parse each frame once.
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
    var headers: [16][]const u8 = undefined;
    var count: usize = 0;
    if (auth_header) |auth| {
        if (count >= headers.len) return error.TooManyHeaders;
        headers[count] = auth;
        count += 1;
    }
    for (extra_headers) |header| {
        if (count >= headers.len) return error.TooManyHeaders;
        headers[count] = header;
        count += 1;
    }

    var stream = OpenAiStream{ .allocator = allocator, .callback = callback, .ctx = ctx };
    defer stream.deinit();
    const transfer = try native_sse.postJson(allocator, url, body, headers[0..count], timeout_secs, &stream, OpenAiStream.onLine);
    if (stream.metadata_error) |err| return err;
    if (transfer.status < 200 or transfer.status >= 300) return error.ServerError;
    if (!transfer.ok) {
        if (!root.shouldRecoverPartialStream(stream.accumulated.items.len, stream.saw_done)) return error.CurlFailed;
        try closeReasoningBlock(allocator, &stream.accumulated, &stream.in_reasoning, callback, ctx);
        callback(ctx, root.StreamChunk.finalChunk());
        return finalizeStreamResultWithTools(allocator, &stream.accumulated, stream.usage, &stream.tools, false);
    }
    try validateToolFinish(&stream.tools);
    const complete = stream.saw_done or stream.tools.finish_reason != .unknown;
    try closeReasoningBlock(allocator, &stream.accumulated, &stream.in_reasoning, callback, ctx);
    const result = try finalizeStreamResultWithTools(allocator, &stream.accumulated, stream.usage, &stream.tools, complete);
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

const AnthropicStream = struct {
    allocator: std.mem.Allocator,
    callback: root.StreamCallback,
    ctx: *anyopaque,
    accumulated: std.ArrayListUnmanaged(u8) = .empty,
    current_event: []const u8 = "",
    usage: root.TokenUsage = .{},
    first_line: bool = true,
    saw_done: bool = false,

    fn deinit(self: *AnthropicStream) void {
        self.accumulated.deinit(self.allocator);
        if (self.current_event.len > 0) self.allocator.free(@constCast(self.current_event));
    }

    fn onLine(context: *anyopaque, line: []const u8) anyerror!bool {
        const self: *AnthropicStream = @ptrCast(@alignCast(context));
        if (self.first_line) if (initialJsonError(self.allocator, line)) |err| return err;
        self.first_line = false;
        const result = parseAnthropicSseLine(self.allocator, line, self.current_event) catch |err| {
            if (err == error.OutOfMemory) return err;
            return true;
        };
        switch (result) {
            .event => |event| {
                const next = try self.allocator.dupe(u8, event);
                if (self.current_event.len > 0) self.allocator.free(@constCast(self.current_event));
                self.current_event = next;
            },
            .delta => |delta| {
                defer self.allocator.free(delta);
                try appendStreamOutput(self.allocator, &self.accumulated, delta);
                self.callback(self.ctx, root.StreamChunk.textDelta(delta));
            },
            .usage => |tokens| self.usage.completion_tokens = tokens,
            .done => {
                self.saw_done = true;
                return false;
            },
            .skip => {},
        }
        return true;
    }
};

/// Stream an Anthropic reply through in-process libcurl.
pub fn curlStreamAnthropic(
    allocator: std.mem.Allocator,
    url: []const u8,
    body: []const u8,
    headers: []const []const u8,
    callback: root.StreamCallback,
    ctx: *anyopaque,
) !root.StreamChatResult {
    var stream = AnthropicStream{ .allocator = allocator, .callback = callback, .ctx = ctx };
    defer stream.deinit();
    const transfer = try native_sse.postJson(allocator, url, body, headers, 0, &stream, AnthropicStream.onLine);
    if (transfer.status < 200 or transfer.status >= 300) return error.ServerError;
    if (!transfer.ok and !root.shouldRecoverPartialStream(stream.accumulated.items.len, stream.saw_done))
        return error.CurlFailed;
    callback(ctx, root.StreamChunk.finalChunk());
    return finalizeStreamResult(allocator, &stream.accumulated, stream.usage);
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

test "tool fragments require a tool_calls finish reason" {
    var collector = stream_tools.Collector{};
    defer collector.deinit(std.testing.allocator);
    try collector.feedLine(std.testing.allocator, "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"probe\",\"arguments\":\"{}\"}}]}}]}");
    try collector.feedLine(std.testing.allocator, "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}");
    try std.testing.expectError(error.IncompleteStreamToolCall, validateToolFinish(&collector));
    collector.finish_reason = .tool_calls;
    try validateToolFinish(&collector);
}

test "parseSseLine DONE sentinel" {
    const result = try parseSseLine(std.testing.allocator, "data: [DONE]");
    try std.testing.expect(result == .done);
}

test "parseSseLine DONE sentinel without optional space" {
    const result = try parseSseLine(std.testing.allocator, "data:[DONE]");
    try std.testing.expect(result == .done);
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
        saw_reasoning: bool = false,

        fn callback(ctx: *anyopaque, chunk: root.StreamChunk) void {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            if (chunk.is_final) {
                self.saw_final = true;
                return;
            }
            if (chunk.kind == .reasoning) self.saw_reasoning = true;
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
    try std.testing.expect(collector.saw_reasoning);
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
    var accumulated: std.ArrayListUnmanaged(u8) = .empty;
    defer accumulated.deinit(std.testing.allocator);
    try accumulated.appendSlice(std.testing.allocator, "<think>private trace</think>Visible answer");
    const result = try finalizeStreamResult(
        std.testing.allocator,
        &accumulated,
        .{ .completion_tokens = 4, .total_tokens = 4 },
    );
    defer {
        if (result.content) |content| std.testing.allocator.free(content);
        if (result.reasoning_content) |reasoning| std.testing.allocator.free(reasoning);
    }

    try std.testing.expectEqualStrings("Visible answer", result.content.?);
    try std.testing.expectEqualStrings("private trace", result.reasoning_content.?);
}

test "finalizeStreamResult transfers plain output ownership" {
    var accumulated: std.ArrayListUnmanaged(u8) = .empty;
    defer accumulated.deinit(std.testing.allocator);
    try accumulated.appendSlice(std.testing.allocator, "visible answer");
    const result = try finalizeStreamResult(std.testing.allocator, &accumulated, null);
    defer std.testing.allocator.free(result.content.?);
    try std.testing.expectEqualStrings("visible answer", result.content.?);
    try std.testing.expectEqual(@as(usize, 0), accumulated.items.len);
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
    response: []const u8 = LOCAL_STREAM_RESPONSE,
    response_delay_ns: u64 = 0,
    stop: std.atomic.Value(bool) = .init(false),
    accepted: std.atomic.Value(usize) = .init(0),
    active: std.atomic.Value(usize) = .init(0),
    peak: std.atomic.Value(usize) = .init(0),

    fn stopAndJoin(self: *LocalStreamServer, thread: *std.Thread) void {
        self.stop.store(true, .release);
        self.unblock();
        thread.join();
    }

    fn stopAndJoinMany(self: *LocalStreamServer, threads: []std.Thread) void {
        self.stop.store(true, .release);
        for (threads) |_| self.unblock();
        for (threads) |*thread| thread.join();
    }

    fn unblock(self: *LocalStreamServer) void {
        const conn = std_compat.net.tcpConnectToAddress(self.server.listen_address) catch null;
        if (conn) |stream| stream.close();
    }

    fn run(self: *LocalStreamServer) void {
        while (true) {
            var conn = self.server.accept() catch return;
            defer conn.stream.close();
            if (self.stop.load(.acquire)) return;
            _ = self.accepted.fetchAdd(1, .monotonic);
            const concurrent = self.active.fetchAdd(1, .monotonic) + 1;
            defer _ = self.active.fetchSub(1, .monotonic);
            _ = self.peak.fetchMax(concurrent, .monotonic);
            readLocalStreamRequest(conn.stream) catch return;
            if (self.response_delay_ns > 0) std_compat.thread.sleep(self.response_delay_ns);
            conn.stream.writeAll(self.response) catch return;
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
    if (!@import("build_options").stream_transport_tests) return error.SkipZigTest;
    if (comptime @import("builtin").os.tag == .wasi) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const addr = try std_compat.net.Address.resolveIp("127.0.0.1", 0);
    var server = try addr.listen(.{});
    defer server.deinit();
    var serving = LocalStreamServer{ .server = &server, .response_delay_ns = 10 * std.time.ns_per_ms };
    var server_threads: [LOCAL_STREAM_WORKERS]std.Thread = undefined;
    for (&server_threads) |*thread| thread.* = try std.Thread.spawn(.{}, LocalStreamServer.run, .{&serving});
    var server_joined = false;
    defer if (!server_joined) serving.stopAndJoinMany(&server_threads);

    const url = try std.fmt.allocPrint(allocator, "http://127.0.0.1:{d}/stream", .{server.listen_address.in.getPort()});
    defer allocator.free(url);
    var samples = [_]LocalStreamSample{.{}} ** LOCAL_STREAM_CALLS;
    var workers: [LOCAL_STREAM_WORKERS]LocalStreamWorker = undefined;
    var threads: [LOCAL_STREAM_WORKERS]std.Thread = undefined;
    const concurrent_started = std.Io.Clock.awake.now(std_compat.io()).nanoseconds;
    for (&workers, 0..) |*worker, i| {
        const begin = i * (LOCAL_STREAM_CALLS / LOCAL_STREAM_WORKERS);
        worker.* = .{ .url = url, .samples = samples[begin .. begin + LOCAL_STREAM_CALLS / LOCAL_STREAM_WORKERS] };
        threads[i] = try std.Thread.spawn(.{}, LocalStreamWorker.run, .{worker});
    }
    for (&threads) |*thread| thread.join();
    const concurrent_ms: u64 = @intCast(@divTrunc(std.Io.Clock.awake.now(std_compat.io()).nanoseconds - concurrent_started, std.time.ns_per_ms));

    var serial_samples = [_]LocalStreamSample{.{}} ** LOCAL_STREAM_CALLS;
    var serial_worker = LocalStreamWorker{ .url = url, .samples = &serial_samples };
    const serial_started = std.Io.Clock.awake.now(std_compat.io()).nanoseconds;
    serial_worker.run();
    const serial_ms: u64 = @intCast(@divTrunc(std.Io.Clock.awake.now(std_compat.io()).nanoseconds - serial_started, std.time.ns_per_ms));
    serving.stopAndJoinMany(&server_threads);
    server_joined = true;

    var first: [LOCAL_STREAM_CALLS]u64 = undefined;
    var final: [LOCAL_STREAM_CALLS]u64 = undefined;
    for (samples, 0..) |sample, i| {
        try std.testing.expect(sample.success);
        first[i] = sample.first_ms;
        final[i] = sample.final_ms;
    }
    for (serial_samples) |sample| try std.testing.expect(sample.success);
    try std.testing.expectEqual(@as(usize, 2 * LOCAL_STREAM_CALLS), serving.accepted.load(.acquire));
    try std.testing.expect(serving.peak.load(.acquire) >= 2);
    std.mem.sortUnstable(u64, &first, {}, std.sort.asc(u64));
    std.mem.sortUnstable(u64, &final, {}, std.sort.asc(u64));
    std.debug.print("local SSE 100/100: concurrent={d}ms serial={d}ms; first p50={d}ms p95={d}ms p99={d}ms; final p50={d}ms p95={d}ms p99={d}ms\n", .{
        concurrent_ms, serial_ms, first[49], first[94], first[98], final[49], final[94], final[98],
    });
}

test "local libcurl stream assembles fragmented tool calls" {
    if (!@import("build_options").stream_transport_tests) return error.SkipZigTest;
    if (comptime @import("builtin").os.tag == .wasi) return error.SkipZigTest;
    const body =
        "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_\",\"type\":\"function\",\"function\":{\"name\":\"memory_\",\"arguments\":\"{\\\"key\\\":\"}}]}}]}\n" ++
        "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"1\",\"function\":{\"name\":\"store\",\"arguments\":\"\\\"lantern\\\"}\"}}]}}]}\n" ++
        "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n" ++
        "data: [DONE]\n";
    const response = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n" ++ body;
    const allocator = std.testing.allocator;
    const addr = try std_compat.net.Address.resolveIp("127.0.0.1", 0);
    var server = try addr.listen(.{});
    defer server.deinit();
    var serving = LocalStreamServer{ .server = &server, .response = response };
    var thread = try std.Thread.spawn(.{}, LocalStreamServer.run, .{&serving});
    var joined = false;
    defer if (!joined) serving.stopAndJoin(&thread);
    const url = try std.fmt.allocPrint(allocator, "http://127.0.0.1:{d}/stream", .{server.listen_address.in.getPort()});
    defer allocator.free(url);
    var callback = LocalStreamCallback{};
    const result = try curlStream(allocator, url, "{}", null, &.{}, 4, LocalStreamCallback.onChunk, &callback);
    defer {
        if (result.content) |content| allocator.free(content);
        if (result.reasoning_content) |reasoning| allocator.free(reasoning);
        for (result.tool_calls) |call| {
            allocator.free(call.id);
            allocator.free(call.name);
            allocator.free(call.arguments);
        }
        allocator.free(result.tool_calls);
    }
    serving.stopAndJoin(&thread);
    joined = true;
    try std.testing.expectEqual(@as(usize, 1), callback.final_calls);
    try std.testing.expectEqual(@as(usize, 1), result.tool_calls.len);
    try std.testing.expectEqualStrings("memory_store", result.tool_calls[0].name);
    try std.testing.expectEqualStrings("{\"key\":\"lantern\"}", result.tool_calls[0].arguments);
}

test "local HTTP errors and redirects cannot become successful partial replies" {
    if (!@import("build_options").stream_transport_tests) return error.SkipZigTest;
    if (comptime @import("builtin").os.tag == .wasi) return error.SkipZigTest;
    const partial = "data: {\"choices\":[{\"delta\":{\"content\":\"misleading\"}}]}\n";
    const responses = [_][]const u8{
        "HTTP/1.1 401 Unauthorized\r\nContent-Type: text/event-stream\r\nContent-Length: 10000\r\nConnection: close\r\n\r\n" ++ partial,
        "HTTP/1.1 302 Found\r\nLocation: https://example.com/\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
    };
    for (responses) |response| {
        const addr = try std_compat.net.Address.resolveIp("127.0.0.1", 0);
        var server = try addr.listen(.{});
        defer server.deinit();
        var serving = LocalStreamServer{ .server = &server, .response = response };
        var thread = try std.Thread.spawn(.{}, LocalStreamServer.run, .{&serving});
        defer serving.stopAndJoin(&thread);
        const url = try std.fmt.allocPrint(std.testing.allocator, "http://127.0.0.1:{d}/stream", .{server.listen_address.in.getPort()});
        defer std.testing.allocator.free(url);
        var callback = LocalStreamCallback{};
        try std.testing.expectError(error.ServerError, curlStream(
            std.testing.allocator,
            url,
            "{}",
            null,
            &.{},
            4,
            LocalStreamCallback.onChunk,
            &callback,
        ));
        try std.testing.expectEqual(@as(usize, 0), callback.final_calls);
    }
}

const HeldOpenStreamServer = struct {
    server: *std_compat.net.Server,
    caller_returned: *std.atomic.Value(bool),
    sent: ?*std.atomic.Value(bool) = null,
    saw_caller_return: bool = false,

    fn run(self: *HeldOpenStreamServer) void {
        var conn = self.server.accept() catch return;
        defer conn.stream.close();
        readLocalStreamRequest(conn.stream) catch return;
        const response = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: keep-alive\r\n\r\n" ++ LOCAL_STREAM_RESPONSE_BODY;
        conn.stream.writeAll(response) catch return;
        if (self.sent) |sent| sent.store(true, .release);
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
    if (!@import("build_options").stream_transport_tests) return error.SkipZigTest;
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

test "idle native HTTP stream stops within one poll interval budget" {
    if (!@import("build_options").stream_transport_tests) return error.SkipZigTest;
    if (comptime @import("builtin").os.tag == .wasi) return error.SkipZigTest;
    const native_http = @import("../native_http.zig");
    const allocator = std.testing.allocator;
    const addr = try std_compat.net.Address.resolveIp("127.0.0.1", 0);
    var server = try addr.listen(.{});
    defer server.deinit();
    var release_server = std.atomic.Value(bool).init(false);
    var sent = std.atomic.Value(bool).init(false);
    var held_open = HeldOpenStreamServer{ .server = &server, .caller_returned = &release_server, .sent = &sent };
    var server_thread = try std.Thread.spawn(.{}, HeldOpenStreamServer.run, .{&held_open});
    defer {
        release_server.store(true, .release);
        server_thread.join();
    }
    const url = try std.fmt.allocPrint(allocator, "http://127.0.0.1:{d}/stream", .{server.listen_address.in.getPort()});
    defer allocator.free(url);
    var interrupted = std.atomic.Value(bool).init(false);
    const CancelRequest = struct {
        url: []const u8,
        flag: *const std.atomic.Value(bool),
        result: ?anyerror = null,

        fn run(self: *@This()) void {
            var response = native_http.perform(std.heap.page_allocator, .{
                .method = .post,
                .url = self.url,
                .body = "{}",
                .timeout_secs = 4,
                .interrupt_flag = self.flag,
            }) catch |err| {
                self.result = err;
                return;
            };
            response.deinit(std.heap.page_allocator);
        }
    };
    var request = CancelRequest{ .url = url, .flag = &interrupted };
    var caller_thread = try std.Thread.spawn(.{}, CancelRequest.run, .{&request});
    var caller_joined = false;
    defer if (!caller_joined) {
        interrupted.store(true, .release);
        caller_thread.join();
    };
    for (0..100) |_| {
        if (sent.load(.acquire)) break;
        std_compat.thread.sleep(10 * std.time.ns_per_ms);
    }
    try std.testing.expect(sent.load(.acquire));
    const started = std.Io.Clock.awake.now(std_compat.io()).nanoseconds;
    interrupted.store(true, .release);
    caller_thread.join();
    caller_joined = true;
    const elapsed_ms = @divTrunc(std.Io.Clock.awake.now(std_compat.io()).nanoseconds - started, std.time.ns_per_ms);
    try std.testing.expectEqual(error.CurlInterrupted, request.result.?);
    try std.testing.expect(elapsed_ms < 500);
}
