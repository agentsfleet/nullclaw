//! HTTPS reads with per-hop address checks and explicit redirect handling.
const std = @import("std");
const compat = @import("compat");
const native_http = @import("native_http.zig");
const http_util = @import("http_util.zig");
const net_security = @import("net_security.zig");

const MAX_REDIRECTS = 5;

pub const GetOptions = struct {
    timeout_secs: u64,
    max_body_bytes: usize,
    headers: []const []const u8 = &.{},
    sink: ?native_http.BodySink = null,
    sink_ctx: ?*anyopaque = null,
};

const LimitedSink = struct {
    downstream: native_http.BodySink,
    context: *anyopaque,
    limit: usize,
    count: usize = 0,

    fn onBytes(ptr: *anyopaque, bytes: []const u8) anyerror!bool {
        const self: *LimitedSink = @ptrCast(@alignCast(ptr));
        if (bytes.len > self.limit -| self.count) return error.ResponseTooLarge;
        self.count += bytes.len;
        return self.downstream(self.context, bytes);
    }
};

fn remainingTimeoutSeconds(timeout_secs: u64, started_ns: i128, now_ns: i128) !u64 {
    const limit_ns: i128 = @as(i128, timeout_secs) * std.time.ns_per_s;
    const elapsed = @max(0, now_ns - started_ns);
    if (elapsed >= limit_ns) return error.CurlTimeout;
    return @intCast(@max(1, @divTrunc(limit_ns - elapsed + std.time.ns_per_s - 1, std.time.ns_per_s)));
}

test "limited file sink rejects bytes before exceeding its response cap" {
    const Sink = struct {
        written: usize = 0,

        fn onBytes(ptr: *anyopaque, bytes: []const u8) !bool {
            const self: *@This() = @ptrCast(@alignCast(ptr));
            self.written += bytes.len;
            return true;
        }
    };
    var sink = Sink{};
    var limited = LimitedSink{ .downstream = Sink.onBytes, .context = &sink, .limit = 4 };
    try std.testing.expect(try LimitedSink.onBytes(&limited, "ab"));
    try std.testing.expect(try LimitedSink.onBytes(&limited, "cd"));
    try std.testing.expectError(error.ResponseTooLarge, LimitedSink.onBytes(&limited, "e"));
    try std.testing.expectEqual(@as(usize, 4), sink.written);
}

test "redirect timeout uses one remaining budget across hops" {
    try std.testing.expectEqual(@as(u64, 3), try remainingTimeoutSeconds(5, 0, 2 * std.time.ns_per_s));
    try std.testing.expectEqual(@as(u64, 1), try remainingTimeoutSeconds(5, 0, 4 * std.time.ns_per_s + 1));
    try std.testing.expectError(error.CurlTimeout, remainingTimeoutSeconds(5, 0, 5 * std.time.ns_per_s));
}

pub fn get(allocator: std.mem.Allocator, initial_url: []const u8, options: GetOptions) !native_http.Response {
    if (options.sink != null and options.sink_ctx == null) return error.MissingBodySinkContext;
    var current = try allocator.dupe(u8, initial_url);
    defer allocator.free(current);
    const started = std.Io.Clock.now(.awake, compat.io()).nanoseconds;
    var limited: LimitedSink = undefined;
    if (options.sink) |sink| limited = .{
        .downstream = sink,
        .context = options.sink_ctx.?,
        .limit = options.max_body_bytes,
    };

    for (0..MAX_REDIRECTS + 1) |hop| {
        try net_security.validateOutboundUrl(current);
        const uri = std.Uri.parse(current) catch return error.InvalidUrl;
        const host = net_security.extractHost(current) orelse return error.InvalidUrl;
        const port: u16 = uri.port orelse 443;
        const target = try net_security.resolveConnectHost(allocator, host, port);
        defer allocator.free(target);
        const pin = if (http_util.shouldUsePinnedResolve(host, target))
            try http_util.buildCurlResolveEntry(allocator, host, port, target)
        else
            null;
        defer if (pin) |entry| allocator.free(entry);
        const proxy = try http_util.getProxyForUrl(allocator, current);
        defer if (proxy) |value| allocator.free(value);
        const now = std.Io.Clock.now(.awake, compat.io()).nanoseconds;
        const remaining = try remainingTimeoutSeconds(options.timeout_secs, started, now);
        var response = try native_http.perform(allocator, .{
            .method = .get,
            .url = current,
            .headers = options.headers,
            .proxy = proxy,
            .resolve_entry = pin,
            .timeout_secs = remaining,
            .max_body_bytes = options.max_body_bytes,
            .capture_headers = true,
            .interrupt_flag = http_util.currentThreadInterruptFlag(),
            .sink = if (options.sink != null) LimitedSink.onBytes else null,
            .sink_ctx = if (options.sink != null) &limited else null,
            // A file sink must never receive an error page or redirect body.
            // Buffered callers still need the body to explain HTTP failures.
            .sink_success_only = options.sink != null,
        });
        if (response.transport_error) |err| {
            response.deinit(allocator);
            return err;
        }
        switch (response.status) {
            301, 302, 303, 307, 308 => {
                const location = native_http.responseHeader(response.headers, "Location") orelse {
                    response.deinit(allocator);
                    return error.MissingRedirectLocation;
                };
                if (hop == MAX_REDIRECTS) {
                    response.deinit(allocator);
                    return error.TooManyRedirects;
                }
                const next = native_http.resolveRedirect(allocator, current, location) catch |err| {
                    response.deinit(allocator);
                    return err;
                };
                response.deinit(allocator);
                allocator.free(current);
                current = next;
            },
            else => return response,
        }
    }
    unreachable;
}
