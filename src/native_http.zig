//! One in-process libcurl transfer per request. Handles are private to the
//! calling thread; libcurl's global setup is serialized once for the process.

const std = @import("std");
const compat = @import("compat");
const curl = @cImport({
    @cInclude("curl/curl.h");
});

const AtomicBool = std.atomic.Value(bool);
const MAX_HEADERS: usize = 32;
const MAX_HEADER_BYTES: usize = 128 * 1024;

var init_mutex: compat.sync.Mutex = .{};
var initialized = AtomicBool.init(false);

pub const Method = enum { get, post, put, patch, delete, head, options };
pub const BodySink = *const fn (*anyopaque, []const u8) anyerror!bool;

pub const Request = struct {
    method: Method,
    url: []const u8,
    body: ?[]const u8 = null,
    body_file: ?*compat.fs.File = null,
    body_file_size: u64 = 0,
    headers: []const []const u8 = &.{},
    proxy: ?[]const u8 = null,
    resolve_entry: ?[]const u8 = null,
    timeout_secs: ?u64 = null,
    connect_timeout_secs: u64 = 30,
    low_speed_secs: ?u64 = null,
    max_body_bytes: usize = 8 * 1024 * 1024,
    capture_headers: bool = false,
    interrupt_flag: ?*const AtomicBool = null,
    sink: ?BodySink = null,
    sink_ctx: ?*anyopaque = null,
    sink_success_only: bool = false,
};

pub const Response = struct {
    status: u16,
    headers: []u8,
    body: []u8,
    terminal: bool,
    transport_error: ?anyerror,

    pub fn deinit(self: *Response, allocator: std.mem.Allocator) void {
        allocator.free(self.headers);
        allocator.free(self.body);
        self.* = undefined;
    }
};

const State = struct {
    allocator: std.mem.Allocator,
    body: std.ArrayList(u8) = .empty,
    headers: std.ArrayList(u8) = .empty,
    max_body_bytes: usize,
    capture_headers: bool,
    sink: ?BodySink,
    sink_ctx: ?*anyopaque,
    sink_success_only: bool,
    header_bytes_seen: usize = 0,
    response_status: u16 = 0,
    body_file: ?*compat.fs.File,
    interrupt_flag: ?*const AtomicBool,
    failure: ?anyerror = null,
    terminal: bool = false,
};

fn onBody(ptr: [*c]u8, size: usize, count: usize, context: ?*anyopaque) callconv(.c) usize {
    const state: *State = @ptrCast(@alignCast(context orelse return 0));
    const len = std.math.mul(usize, size, count) catch {
        state.failure = error.ResponseTooLarge;
        return 0;
    };
    const bytes = ptr[0..len];
    if (state.sink_success_only and (state.response_status < 200 or state.response_status >= 300)) return len;
    if (state.sink) |sink| {
        const keep = sink(state.sink_ctx orelse return 0, bytes) catch |err| {
            state.failure = err;
            return 0;
        };
        if (!keep) {
            state.terminal = true;
            return 0;
        }
    } else {
        if (len > state.max_body_bytes -| state.body.items.len) {
            state.failure = error.ResponseTooLarge;
            return 0;
        }
        state.body.appendSlice(state.allocator, bytes) catch |err| {
            state.failure = err;
            return 0;
        };
    }
    return len;
}

fn onHeaders(ptr: [*c]u8, size: usize, count: usize, context: ?*anyopaque) callconv(.c) usize {
    const state: *State = @ptrCast(@alignCast(context orelse return 0));
    const len = std.math.mul(usize, size, count) catch {
        state.failure = error.ResponseHeadersTooLarge;
        return 0;
    };
    if (len > MAX_HEADER_BYTES -| state.header_bytes_seen) {
        state.failure = error.ResponseHeadersTooLarge;
        return 0;
    }
    state.header_bytes_seen += len;
    const line = ptr[0..len];
    if (std.mem.startsWith(u8, line, "HTTP/")) {
        if (std.mem.indexOfScalar(u8, line, ' ')) |space| {
            if (space + 4 <= line.len)
                state.response_status = std.fmt.parseInt(u16, line[space + 1 .. space + 4], 10) catch 0;
        }
    }
    if (state.capture_headers) {
        state.headers.appendSlice(state.allocator, ptr[0..len]) catch |err| {
            state.failure = err;
            return 0;
        };
    }
    return len;
}

test "response headers are bounded even when the caller does not capture them" {
    const allocator = std.testing.allocator;
    var state = State{
        .allocator = allocator,
        .max_body_bytes = 0,
        .capture_headers = false,
        .sink = null,
        .sink_ctx = null,
        .sink_success_only = false,
        .body_file = null,
        .interrupt_flag = null,
    };
    const header = try allocator.alloc(u8, MAX_HEADER_BYTES);
    defer allocator.free(header);
    @memset(header, 'a');
    try std.testing.expectEqual(MAX_HEADER_BYTES, onHeaders(header.ptr, 1, header.len, &state));
    try std.testing.expectEqual(@as(usize, 0), onHeaders(header.ptr, 1, 1, &state));
    try std.testing.expectEqual(error.ResponseHeadersTooLarge, state.failure.?);
}

fn onRead(ptr: [*c]u8, size: usize, count: usize, context: ?*anyopaque) callconv(.c) usize {
    const state: *State = @ptrCast(@alignCast(context orelse return curl.CURL_READFUNC_ABORT));
    const len = std.math.mul(usize, size, count) catch return curl.CURL_READFUNC_ABORT;
    const file = state.body_file orelse return curl.CURL_READFUNC_ABORT;
    return file.read(ptr[0..len]) catch |err| {
        state.failure = err;
        return curl.CURL_READFUNC_ABORT;
    };
}

fn onProgress(context: ?*anyopaque, _: curl.curl_off_t, _: curl.curl_off_t, _: curl.curl_off_t, _: curl.curl_off_t) callconv(.c) c_int {
    const state: *State = @ptrCast(@alignCast(context orelse return 1));
    if (state.interrupt_flag) |flag| {
        if (flag.load(.acquire)) {
            state.failure = error.CurlInterrupted;
            return 1;
        }
    }
    return 0;
}

fn ensureInitialized() !void {
    if (initialized.load(.acquire)) return;
    init_mutex.lock();
    defer init_mutex.unlock();
    if (initialized.load(.acquire)) return;
    if (curl.curl_global_init(curl.CURL_GLOBAL_DEFAULT) != curl.CURLE_OK)
        return error.CurlInitFailed;
    initialized.store(true, .release);
}

fn setOption(easy: *curl.CURL, option: curl.CURLoption, value: anytype) !void {
    if (curl.curl_easy_setopt(easy, option, value) != curl.CURLE_OK)
        return error.CurlOptionFailed;
}

pub fn validateHeaderLine(header: []const u8) !void {
    if (std.mem.indexOfAny(u8, header, "\r\n\x00") != null) return error.InvalidHeader;
}

fn appendHeader(allocator: std.mem.Allocator, list: *?*curl.struct_curl_slist, header: []const u8) !void {
    try validateHeaderLine(header);
    const terminated = try allocator.dupeZ(u8, header);
    defer allocator.free(terminated);
    list.* = curl.curl_slist_append(list.*, terminated.ptr) orelse return error.OutOfMemory;
}

fn connectToEntry(allocator: std.mem.Allocator, resolve_entry: []const u8) ![]u8 {
    const host_end = std.mem.indexOfScalar(u8, resolve_entry, ':') orelse return error.InvalidResolveEntry;
    const port_end = std.mem.indexOfScalarPos(u8, resolve_entry, host_end + 1, ':') orelse return error.InvalidResolveEntry;
    if (host_end == 0 or port_end == host_end + 1 or port_end + 1 == resolve_entry.len)
        return error.InvalidResolveEntry;
    const host = resolve_entry[0..host_end];
    const port = resolve_entry[host_end + 1 .. port_end];
    const address = resolve_entry[port_end + 1 ..];
    return std.fmt.allocPrint(allocator, "{s}:{s}:{s}:{s}", .{ host, port, address, port });
}

fn seconds(value: u64) c_long {
    return @intCast(@min(value, std.math.maxInt(c_long)));
}

fn transportError(code: curl.CURLcode) anyerror {
    return switch (code) {
        curl.CURLE_COULDNT_RESOLVE_HOST, curl.CURLE_COULDNT_RESOLVE_PROXY => error.CurlDnsError,
        curl.CURLE_COULDNT_CONNECT => error.CurlConnectError,
        curl.CURLE_OPERATION_TIMEDOUT => error.CurlTimeout,
        curl.CURLE_SSL_CONNECT_ERROR, curl.CURLE_PEER_FAILED_VERIFICATION, curl.CURLE_SSL_CERTPROBLEM => error.CurlTlsError,
        curl.CURLE_ABORTED_BY_CALLBACK => error.CurlInterrupted,
        else => error.CurlFailed,
    };
}

pub fn resolveRedirect(allocator: std.mem.Allocator, base: []const u8, location: []const u8) ![]u8 {
    try validateHeaderLine(location);
    const handle = curl.curl_url() orelse return error.CurlInitFailed;
    defer curl.curl_url_cleanup(handle);
    const base_z = try allocator.dupeZ(u8, base);
    defer allocator.free(base_z);
    const location_z = try allocator.dupeZ(u8, location);
    defer allocator.free(location_z);
    if (curl.curl_url_set(handle, curl.CURLUPART_URL, base_z.ptr, 0) != curl.CURLUE_OK or
        curl.curl_url_set(handle, curl.CURLUPART_URL, location_z.ptr, 0) != curl.CURLUE_OK)
        return error.InvalidRedirect;
    var raw: [*c]u8 = null;
    if (curl.curl_url_get(handle, curl.CURLUPART_URL, &raw, 0) != curl.CURLUE_OK or raw == null)
        return error.InvalidRedirect;
    defer curl.curl_free(raw);
    return allocator.dupe(u8, std.mem.span(raw));
}

pub fn responseHeader(headers: []const u8, name: []const u8) ?[]const u8 {
    var found: ?[]const u8 = null;
    var lines = std.mem.splitScalar(u8, headers, '\n');
    while (lines.next()) |raw| {
        const line = std.mem.trim(u8, raw, "\r");
        if (std.mem.startsWith(u8, line, "HTTP/")) found = null;
        const colon = std.mem.indexOfScalar(u8, line, ':') orelse continue;
        if (std.ascii.eqlIgnoreCase(line[0..colon], name))
            found = std.mem.trim(u8, line[colon + 1 ..], " \t");
    }
    return found;
}

pub fn perform(allocator: std.mem.Allocator, request: Request) !Response {
    if (request.headers.len > MAX_HEADERS) return error.TooManyHeaders;
    if (request.sink != null and request.sink_ctx == null) return error.MissingBodySinkContext;
    if (request.body != null and request.body_file != null) return error.MultipleRequestBodies;
    if (request.body_file_size > std.math.maxInt(curl.curl_off_t)) return error.RequestTooLarge;
    if (request.body) |body| if (body.len > std.math.maxInt(curl.curl_off_t)) return error.RequestTooLarge;
    try validateHeaderLine(request.url);
    if (request.proxy) |proxy| try validateHeaderLine(proxy);
    try ensureInitialized();
    const easy = curl.curl_easy_init() orelse return error.CurlInitFailed;
    defer curl.curl_easy_cleanup(easy);

    const url = try allocator.dupeZ(u8, request.url);
    defer allocator.free(url);
    var header_list: ?*curl.struct_curl_slist = null;
    defer if (header_list) |list| curl.curl_slist_free_all(list);
    for (request.headers) |header| try appendHeader(allocator, &header_list, header);
    var resolve_list: ?*curl.struct_curl_slist = null;
    defer if (resolve_list) |list| curl.curl_slist_free_all(list);
    if (request.resolve_entry) |entry| try appendHeader(allocator, &resolve_list, entry);
    var connect_to_list: ?*curl.struct_curl_slist = null;
    defer if (connect_to_list) |list| curl.curl_slist_free_all(list);
    if (request.resolve_entry != null and request.proxy != null) {
        const entry = try connectToEntry(allocator, request.resolve_entry.?);
        defer allocator.free(entry);
        try appendHeader(allocator, &connect_to_list, entry);
    }
    const proxy = if (request.proxy) |value| try allocator.dupeZ(u8, value) else null;
    defer if (proxy) |value| allocator.free(value);

    var state = State{
        .allocator = allocator,
        .max_body_bytes = request.max_body_bytes,
        .capture_headers = request.capture_headers,
        .sink = request.sink,
        .sink_ctx = request.sink_ctx,
        .sink_success_only = request.sink_success_only,
        .body_file = request.body_file,
        .interrupt_flag = request.interrupt_flag,
    };
    defer state.body.deinit(allocator);
    defer state.headers.deinit(allocator);

    try setOption(easy, curl.CURLOPT_URL, url.ptr);
    try setOption(easy, curl.CURLOPT_PROTOCOLS_STR, "http,https");
    try setOption(easy, curl.CURLOPT_HTTPHEADER, header_list);
    try setOption(easy, curl.CURLOPT_HEADEROPT, @as(c_long, curl.CURLHEADER_SEPARATE));
    try setOption(easy, curl.CURLOPT_NOSIGNAL, @as(c_long, 1));
    try setOption(easy, curl.CURLOPT_FOLLOWLOCATION, @as(c_long, 0));
    try setOption(easy, curl.CURLOPT_CONNECTTIMEOUT, seconds(request.connect_timeout_secs));
    try setOption(easy, curl.CURLOPT_WRITEFUNCTION, &onBody);
    try setOption(easy, curl.CURLOPT_WRITEDATA, &state);
    try setOption(easy, curl.CURLOPT_HEADERFUNCTION, &onHeaders);
    try setOption(easy, curl.CURLOPT_HEADERDATA, &state);
    try setOption(easy, curl.CURLOPT_NOPROGRESS, @as(c_long, 0));
    try setOption(easy, curl.CURLOPT_XFERINFOFUNCTION, &onProgress);
    try setOption(easy, curl.CURLOPT_XFERINFODATA, &state);
    if (request.timeout_secs) |timeout| try setOption(easy, curl.CURLOPT_TIMEOUT, seconds(timeout));
    if (request.low_speed_secs) |idle| {
        try setOption(easy, curl.CURLOPT_LOW_SPEED_LIMIT, @as(c_long, 1));
        try setOption(easy, curl.CURLOPT_LOW_SPEED_TIME, seconds(idle));
    }
    if (request.resolve_entry != null) {
        try setOption(easy, curl.CURLOPT_RESOLVE, resolve_list);
        if (connect_to_list) |list|
            try setOption(easy, curl.CURLOPT_CONNECT_TO, list)
        else
            try setOption(easy, curl.CURLOPT_NOPROXY, "*");
    }
    if (proxy) |value| {
        try setOption(easy, curl.CURLOPT_PROXY, value.ptr);
    }
    const method: [*:0]const u8 = switch (request.method) {
        .get => "GET",
        .post => "POST",
        .put => "PUT",
        .patch => "PATCH",
        .delete => "DELETE",
        .head => "HEAD",
        .options => "OPTIONS",
    };
    try setOption(easy, curl.CURLOPT_CUSTOMREQUEST, method);
    if (request.method == .head) try setOption(easy, curl.CURLOPT_NOBODY, @as(c_long, 1));
    if (request.body) |body| {
        try setOption(easy, curl.CURLOPT_POSTFIELDS, body.ptr);
        try setOption(easy, curl.CURLOPT_POSTFIELDSIZE_LARGE, @as(curl.curl_off_t, @intCast(body.len)));
    } else if (request.body_file != null) {
        try setOption(easy, curl.CURLOPT_POST, @as(c_long, 1));
        try setOption(easy, curl.CURLOPT_READFUNCTION, &onRead);
        try setOption(easy, curl.CURLOPT_READDATA, &state);
        try setOption(easy, curl.CURLOPT_POSTFIELDSIZE_LARGE, @as(curl.curl_off_t, @intCast(request.body_file_size)));
    }

    const multi = curl.curl_multi_init() orelse return error.CurlInitFailed;
    defer _ = curl.curl_multi_cleanup(multi);
    if (curl.curl_multi_add_handle(multi, easy) != curl.CURLM_OK) return error.CurlInitFailed;
    defer _ = curl.curl_multi_remove_handle(multi, easy);

    var running: c_int = 0;
    var code: curl.CURLcode = curl.CURLE_OK;
    while (true) {
        if (request.interrupt_flag) |flag| {
            if (flag.load(.acquire)) {
                state.failure = error.CurlInterrupted;
                break;
            }
        }
        if (curl.curl_multi_perform(multi, &running) != curl.CURLM_OK) {
            state.failure = error.CurlFailed;
            break;
        }
        if (running == 0 or state.failure != null) break;
        var active_fds: c_int = 0;
        if (curl.curl_multi_poll(multi, null, 0, 50, &active_fds) != curl.CURLM_OK) {
            state.failure = error.CurlFailed;
            break;
        }
    }
    if (state.failure == null) {
        var messages_left: c_int = 0;
        const message = curl.curl_multi_info_read(multi, &messages_left);
        if (message == null or message.*.msg != curl.CURLMSG_DONE) return error.CurlFailed;
        code = message.*.data.result;
    }
    if (state.failure) |err| return err;
    var response_code: c_long = 0;
    if (curl.curl_easy_getinfo(easy, curl.CURLINFO_RESPONSE_CODE, &response_code) != curl.CURLE_OK)
        return error.CurlInfoFailed;
    const headers = try state.headers.toOwnedSlice(allocator);
    errdefer allocator.free(headers);
    const body = try state.body.toOwnedSlice(allocator);
    return .{
        .status = @intCast(@max(0, @min(response_code, std.math.maxInt(u16)))),
        .headers = headers,
        .body = body,
        .terminal = state.terminal,
        .transport_error = if (code == curl.CURLE_OK or state.terminal) null else transportError(code),
    };
}

test "a configured proxy tunnels to the pinned address" {
    if (!@import("build_options").stream_transport_tests) return error.SkipZigTest;
    if (comptime @import("builtin").os.tag == .wasi) return error.SkipZigTest;
    const address = try compat.net.Address.resolveIp("127.0.0.1", 0);
    var server = try address.listen(.{});
    defer server.deinit();
    const Proxy = struct {
        server: *compat.net.Server,
        saw_pinned_target: bool = false,

        fn readHeaders(stream: compat.net.Stream, buf: *[4096]u8) !usize {
            var used: usize = 0;
            while (used < buf.len) {
                const count = try stream.read(buf[used..]);
                if (count == 0) return error.TestUnexpectedResult;
                used += count;
                if (std.mem.indexOf(u8, buf[0..used], "\r\n\r\n") != null) return used;
            }
            return error.TestUnexpectedResult;
        }

        fn run(self: *@This()) void {
            var connection = self.server.accept() catch return;
            defer connection.stream.close();
            var first: [4096]u8 = undefined;
            const first_len = readHeaders(connection.stream, &first) catch return;
            self.saw_pinned_target = std.mem.startsWith(u8, first[0..first_len], "CONNECT 203.0.113.9:80 ");
            if (!self.saw_pinned_target) return;
            connection.stream.writeAll("HTTP/1.1 200 Connection established\r\n\r\n") catch return;
            var second: [4096]u8 = undefined;
            _ = readHeaders(connection.stream, &second) catch return;
            connection.stream.writeAll("HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok") catch {};
        }
    };
    var proxy_server = Proxy{ .server = &server };
    var thread = try std.Thread.spawn(.{}, Proxy.run, .{&proxy_server});
    var joined = false;
    defer if (!joined) thread.join();
    const proxy_url = try std.fmt.allocPrint(std.testing.allocator, "http://127.0.0.1:{d}", .{server.listen_address.in.getPort()});
    defer std.testing.allocator.free(proxy_url);
    var response = try perform(std.testing.allocator, .{
        .method = .get,
        .url = "http://pinned.test:80/",
        .proxy = proxy_url,
        .resolve_entry = "pinned.test:80:203.0.113.9",
        .timeout_secs = 5,
    });
    defer response.deinit(std.testing.allocator);
    thread.join();
    joined = true;
    try std.testing.expect(response.transport_error == null);
    try std.testing.expectEqual(@as(u16, 200), response.status);
    try std.testing.expectEqualStrings("ok", response.body);
    try std.testing.expect(proxy_server.saw_pinned_target);
}
