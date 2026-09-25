//! Provider Server-Sent Events over the shared in-process libcurl transport.

const std = @import("std");
const http_util = @import("../http_util.zig");
const native_http = @import("../native_http.zig");
const line_reader = @import("sse_line_reader.zig");

const DEFAULT_MAX_SECONDS: u64 = 600;
const LOW_SPEED_SECONDS: u64 = 60;

pub const Result = struct {
    status: u16,
    ok: bool,
    terminal: bool,
    transport_error: ?anyerror,
};

pub const Options = struct {
    timeout_secs: u64,
    connect_timeout_secs: u64 = 30,
    low_speed_secs: u64 = LOW_SPEED_SECONDS,
};

pub const LineCallback = *const fn (*anyopaque, []const u8) anyerror!bool;

const StreamState = struct {
    reader: line_reader.FeedReader,
    ctx: *anyopaque,
    on_line: LineCallback,

    fn onBytes(ptr: *anyopaque, bytes: []const u8) anyerror!bool {
        const self: *StreamState = @ptrCast(@alignCast(ptr));
        return self.reader.feed(bytes, self.ctx, self.on_line);
    }
};

test "native stream rejects CR LF and NUL in caller headers before network I/O" {
    const Ignore = struct {
        fn onLine(_: *anyopaque, _: []const u8) !bool {
            return error.UnexpectedLine;
        }
    };
    var marker: u8 = 0;
    for ([_][]const u8{
        "X-Test: ok\rX-Injected: yes",
        "X-Test: ok\nX-Injected: yes",
        "X-Test: ok\x00X-Injected: yes",
    }) |header| {
        try std.testing.expectError(error.InvalidHeader, postJson(
            std.testing.allocator,
            "http://127.0.0.1:1/",
            "{}",
            &.{header},
            1,
            &marker,
            Ignore.onLine,
        ));
    }
}

/// POST one JSON stream. A `false` line callback ends the transfer after a
/// provider terminal event. Each caller owns its parser state and easy handle.
pub fn postJson(
    allocator: std.mem.Allocator,
    url: []const u8,
    body: []const u8,
    headers: []const []const u8,
    timeout_secs: u64,
    ctx: *anyopaque,
    on_line: LineCallback,
) !Result {
    return postJsonWithOptions(allocator, url, body, headers, .{ .timeout_secs = timeout_secs }, ctx, on_line);
}

pub fn postJsonWithOptions(
    allocator: std.mem.Allocator,
    url: []const u8,
    body: []const u8,
    headers: []const []const u8,
    options: Options,
    ctx: *anyopaque,
    on_line: LineCallback,
) !Result {
    if (headers.len >= 16) return error.TooManyHeaders;
    var all_headers: [17][]const u8 = undefined;
    all_headers[0] = "Content-Type: application/json";
    for (headers, 0..) |header, i| all_headers[i + 1] = header;
    const resolve_entry = try http_util.buildSafeResolveEntryForRemoteUrl(allocator, url);
    defer if (resolve_entry) |entry| allocator.free(entry);
    const proxy = try http_util.getProxyForUrl(allocator, url);
    defer if (proxy) |value| allocator.free(value);
    var state = StreamState{ .reader = line_reader.FeedReader.init(allocator), .ctx = ctx, .on_line = on_line };
    defer state.reader.deinit();
    var response = try native_http.perform(allocator, .{
        .method = .post,
        .url = url,
        .body = body,
        .headers = all_headers[0 .. headers.len + 1],
        .proxy = proxy,
        .resolve_entry = resolve_entry,
        .timeout_secs = if (options.timeout_secs == 0) DEFAULT_MAX_SECONDS else options.timeout_secs,
        .connect_timeout_secs = options.connect_timeout_secs,
        .low_speed_secs = options.low_speed_secs,
        .interrupt_flag = http_util.currentThreadInterruptFlag(),
        .sink = StreamState.onBytes,
        .sink_ctx = &state,
    });
    defer response.deinit(allocator);
    const terminal = response.terminal or !(try state.reader.finish(ctx, on_line));
    return .{ .status = response.status, .ok = response.transport_error == null or terminal, .terminal = terminal, .transport_error = response.transport_error };
}
