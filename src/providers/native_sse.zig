//! In-process libcurl transport for provider Server-Sent Events.
//! One transfer runs on the calling steer thread; different steers use
//! independent handles and never share mutable libcurl state.

const std = @import("std");
const compat = @import("compat");
const http_util = @import("../http_util.zig");
const line_reader = @import("sse_line_reader.zig");
const curl = @cImport({
    @cInclude("curl/curl.h");
});

const DEFAULT_MAX_SECONDS: u64 = 600;
const LOW_SPEED_SECONDS: c_long = 60;
const LOW_SPEED_BYTES_PER_SECOND: c_long = 1;

var init_mutex: compat.sync.Mutex = .{};
var initialized = false;

pub const Result = struct {
    status: u16,
    ok: bool,
    terminal: bool,
};

pub const LineCallback = *const fn (*anyopaque, []const u8) anyerror!bool;

const WriteState = struct {
    reader: line_reader.FeedReader,
    ctx: *anyopaque,
    on_line: LineCallback,
    failure: ?anyerror = null,
    terminal: bool = false,
};

fn onBytes(ptr: [*c]u8, size: usize, count: usize, context: ?*anyopaque) callconv(.c) usize {
    const state: *WriteState = @ptrCast(@alignCast(context orelse return 0));
    const len = std.math.mul(usize, size, count) catch return 0;
    const keep_reading = state.reader.feed(ptr[0..len], state.ctx, state.on_line) catch |err| {
        state.failure = err;
        return 0;
    };
    if (!keep_reading) {
        state.terminal = true;
        return 0;
    }
    return len;
}

fn ensureInitialized() !void {
    init_mutex.lock();
    defer init_mutex.unlock();
    if (initialized) return;
    if (curl.curl_global_init(curl.CURL_GLOBAL_DEFAULT) != curl.CURLE_OK)
        return error.CurlInitFailed;
    initialized = true;
}

fn setOption(easy: *curl.CURL, option: curl.CURLoption, value: anytype) !void {
    if (curl.curl_easy_setopt(easy, option, value) != curl.CURLE_OK)
        return error.CurlOptionFailed;
}

fn appendHeader(allocator: std.mem.Allocator, list: *?*curl.struct_curl_slist, header: []const u8) !void {
    try http_util.validateCurlHeaderLine(header);
    const terminated = try allocator.dupeZ(u8, header);
    defer allocator.free(terminated);
    list.* = curl.curl_slist_append(list.*, terminated.ptr) orelse return error.OutOfMemory;
}

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

/// POST one JSON stream. A `false` line callback ends the HTTP transfer
/// immediately after a provider terminal event. The caller owns all parser state.
pub fn postJson(
    allocator: std.mem.Allocator,
    url: []const u8,
    body: []const u8,
    headers: []const []const u8,
    timeout_secs: u64,
    ctx: *anyopaque,
    on_line: LineCallback,
) !Result {
    if (headers.len >= 16) return error.TooManyHeaders;
    if (body.len > std.math.maxInt(curl.curl_off_t)) return error.RequestTooLarge;
    try ensureInitialized();
    const easy = curl.curl_easy_init() orelse return error.CurlInitFailed;
    defer curl.curl_easy_cleanup(easy);

    const url_z = try allocator.dupeZ(u8, url);
    defer allocator.free(url_z);
    var header_list: ?*curl.struct_curl_slist = null;
    defer if (header_list) |list| curl.curl_slist_free_all(list);
    try appendHeader(allocator, &header_list, "Content-Type: application/json");
    for (headers) |header| try appendHeader(allocator, &header_list, header);

    const resolve_entry = try http_util.buildSafeResolveEntryForRemoteUrl(allocator, url);
    defer if (resolve_entry) |entry| allocator.free(entry);
    var resolve_list: ?*curl.struct_curl_slist = null;
    defer if (resolve_list) |list| curl.curl_slist_free_all(list);
    if (resolve_entry) |entry| {
        try appendHeader(allocator, &resolve_list, entry);
        try setOption(easy, curl.CURLOPT_RESOLVE, resolve_list);
        // A proxy would resolve the origin itself and bypass the pinned address.
        try setOption(easy, curl.CURLOPT_NOPROXY, "*");
    }

    const proxy = http_util.getProxyFromEnv(allocator) catch null;
    defer if (proxy) |value| allocator.free(value);
    const proxy_z = if (proxy) |value| try allocator.dupeZ(u8, value) else null;
    defer if (proxy_z) |value| allocator.free(value);
    if (proxy_z) |value| try setOption(easy, curl.CURLOPT_PROXY, value.ptr);

    var state = WriteState{ .reader = line_reader.FeedReader.init(allocator), .ctx = ctx, .on_line = on_line };
    defer state.reader.deinit();
    const seconds = if (timeout_secs == 0) DEFAULT_MAX_SECONDS else timeout_secs;
    const bounded_seconds: c_long = @intCast(@min(seconds, std.math.maxInt(c_long)));
    try setOption(easy, curl.CURLOPT_URL, url_z.ptr);
    try setOption(easy, curl.CURLOPT_HTTPHEADER, header_list);
    // Keep provider auth headers off an HTTPS proxy's CONNECT request.
    try setOption(easy, curl.CURLOPT_HEADEROPT, @as(c_long, curl.CURLHEADER_SEPARATE));
    try setOption(easy, curl.CURLOPT_POST, @as(c_long, 1));
    try setOption(easy, curl.CURLOPT_POSTFIELDS, body.ptr);
    try setOption(easy, curl.CURLOPT_POSTFIELDSIZE_LARGE, @as(curl.curl_off_t, @intCast(body.len)));
    try setOption(easy, curl.CURLOPT_WRITEFUNCTION, &onBytes);
    try setOption(easy, curl.CURLOPT_WRITEDATA, &state);
    try setOption(easy, curl.CURLOPT_TIMEOUT, bounded_seconds);
    try setOption(easy, curl.CURLOPT_LOW_SPEED_LIMIT, LOW_SPEED_BYTES_PER_SECOND);
    try setOption(easy, curl.CURLOPT_LOW_SPEED_TIME, LOW_SPEED_SECONDS);
    try setOption(easy, curl.CURLOPT_NOSIGNAL, @as(c_long, 1));
    try setOption(easy, curl.CURLOPT_FOLLOWLOCATION, @as(c_long, 0));

    const code = curl.curl_easy_perform(easy);
    if (state.failure) |err| return err;
    if (!state.terminal and !(try state.reader.finish(ctx, on_line))) state.terminal = true;
    var response_code: c_long = 0;
    if (curl.curl_easy_getinfo(easy, curl.CURLINFO_RESPONSE_CODE, &response_code) != curl.CURLE_OK)
        return error.CurlInfoFailed;
    return .{
        .status = @intCast(@max(0, @min(response_code, std.math.maxInt(u16)))),
        .ok = code == curl.CURLE_OK or state.terminal,
        .terminal = state.terminal,
    };
}
