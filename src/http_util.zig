//! Shared HTTP utilities.
//!
//! Buffered and streaming requests share the in-process libcurl transport.
//! Existing public helper names remain for provider compatibility.

const std = @import("std");
const std_compat = @import("compat");
const Allocator = std.mem.Allocator;
const AtomicBool = std.atomic.Value(bool);
const net_security = @import("net_security.zig");
const native_http = @import("native_http.zig");

const log = std.log.scoped(.http_util);
threadlocal var thread_interrupt_flag: ?*const AtomicBool = null;
const DEFAULT_CURL_GET_MAX_BYTES: usize = 4 * 1024 * 1024;
const DEFAULT_CURL_POST_MAX_BYTES: usize = 8 * 1024 * 1024;

pub fn isCurlTransportError(err: anyerror) bool {
    return switch (err) {
        error.CurlDnsError,
        error.CurlConnectError,
        error.CurlTimeout,
        error.CurlTlsError,
        error.CurlReadError,
        error.CurlWriteError,
        error.CurlWaitError,
        error.CurlFailed,
        error.CurlInterrupted,
        => true,
        else => false,
    };
}

pub fn preserveCurlTransportError(err: anyerror, fallback: anyerror) anyerror {
    return if (isCurlTransportError(err)) err else fallback;
}

pub fn setThreadInterruptFlag(flag: ?*const AtomicBool) void {
    thread_interrupt_flag = flag;
}

pub fn currentThreadInterruptFlag() ?*const AtomicBool {
    return thread_interrupt_flag;
}

pub const HttpResponse = struct {
    status_code: u16,
    body: []u8,
};

pub const HttpResponseWithHeaders = struct {
    status_code: u16,
    headers: []u8,
    body: []u8,
};

fn parseHeader(header: []const u8) ?std.http.Header {
    const colon = std.mem.indexOfScalar(u8, header, ':') orelse return null;
    const name = std.mem.trim(u8, header[0..colon], " \t\r\n");
    const value = std.mem.trim(u8, header[colon + 1 ..], " \t\r\n");
    if (name.len == 0) return null;
    return .{ .name = name, .value = value };
}

fn contentTypeHeaderValue(header: []const u8) ?[]const u8 {
    const parsed = parseHeader(header) orelse return null;
    if (!std.ascii.eqlIgnoreCase(parsed.name, "content-type")) return null;
    return parsed.value;
}

fn initProxyClientWithOptionalProxy(allocator: Allocator, proxy: ?[]const u8) !ProxyHttpClient {
    var proxy_client = try ProxyHttpClient.init(allocator);
    if (proxy == null) return proxy_client;

    proxy_client.deinit();
    var proxy_arena = std.heap.ArenaAllocator.init(allocator);
    errdefer proxy_arena.deinit();
    var client: std.http.Client = .{ .allocator = allocator, .io = std_compat.io() };
    errdefer client.deinit();
    var env_map = std_compat.process.EnvMap.init(proxy_arena.allocator());
    try env_map.put("HTTPS_PROXY", proxy.?);
    try env_map.put("https_proxy", proxy.?);
    try env_map.put("HTTP_PROXY", proxy.?);
    try env_map.put("http_proxy", proxy.?);
    try client.initDefaultProxies(proxy_arena.allocator(), &env_map);
    return .{ .proxy_arena = proxy_arena, .client = client };
}

pub fn httpRequestWithStatusAndHeaders(
    allocator: Allocator,
    method: std.http.Method,
    url: []const u8,
    body: ?[]const u8,
    headers: []const []const u8,
    content_type: ?[]const u8,
    proxy: ?[]const u8,
) !HttpResponseWithHeaders {
    var header_buf: [20]std.http.Header = undefined;
    var header_count: usize = 0;
    if (content_type) |ct| {
        header_buf[header_count] = .{ .name = "Content-Type", .value = ct };
        header_count += 1;
    }
    for (headers) |header| {
        if (header_count >= header_buf.len) return error.TooManyHeaders;
        header_buf[header_count] = parseHeader(header) orelse return error.InvalidHeader;
        header_count += 1;
    }

    var client = try initProxyClientWithOptionalProxy(allocator, proxy);
    defer client.deinit();

    const uri = try std.Uri.parse(url);
    const redirect_behavior: std.http.Client.Request.RedirectBehavior =
        if (body == null) @enumFromInt(3) else .unhandled;
    var req = try client.client.request(method, uri, .{
        .redirect_behavior = redirect_behavior,
        .headers = .{ .accept_encoding = .default },
        .extra_headers = header_buf[0..header_count],
    });
    defer req.deinit();

    if (body) |payload| {
        req.transfer_encoding = .{ .content_length = payload.len };
        var request_body = try req.sendBodyUnflushed(&.{});
        try request_body.writer.writeAll(payload);
        try request_body.end();
        try req.connection.?.flush();
    } else {
        try req.sendBodiless();
    }

    var redirect_buffer: [8 * 1024]u8 = undefined;
    var response = try req.receiveHead(&redirect_buffer);
    const response_headers = try allocator.dupe(u8, response.head.bytes);
    errdefer allocator.free(response_headers);

    var aw: std.Io.Writer.Allocating = .init(allocator);
    defer aw.deinit();

    const decompress_buffer: []u8 = switch (response.head.content_encoding) {
        .identity => &.{},
        .zstd => try allocator.alloc(u8, std.compress.zstd.default_window_len),
        .deflate, .gzip => try allocator.alloc(u8, std.compress.flate.max_window_len),
        .compress => return error.UnsupportedCompressionMethod,
    };
    defer if (response.head.content_encoding != .identity) allocator.free(decompress_buffer);

    var transfer_buffer: [64]u8 = undefined;
    var decompress: std.http.Decompress = undefined;
    const reader = response.readerDecompressing(&transfer_buffer, &decompress, decompress_buffer);
    _ = reader.streamRemaining(&aw.writer) catch |err| switch (err) {
        error.ReadFailed => return response.bodyErr().?,
        else => |e| return e,
    };

    const response_body = aw.writer.buffer[0..aw.writer.end];
    return .{
        .status_code = @as(u16, @intFromEnum(response.head.status)),
        .headers = response_headers,
        .body = try allocator.dupe(u8, response_body),
    };
}

pub fn httpRequestWithStatus(
    allocator: Allocator,
    method: std.http.Method,
    url: []const u8,
    body: ?[]const u8,
    headers: []const []const u8,
    content_type: ?[]const u8,
    proxy: ?[]const u8,
) !HttpResponse {
    const resp = try httpRequestWithStatusAndHeaders(allocator, method, url, body, headers, content_type, proxy);
    allocator.free(resp.headers);
    return .{
        .status_code = resp.status_code,
        .body = resp.body,
    };
}

pub fn httpRequest(
    allocator: Allocator,
    method: std.http.Method,
    url: []const u8,
    body: ?[]const u8,
    headers: []const []const u8,
    content_type: ?[]const u8,
    proxy: ?[]const u8,
) ![]u8 {
    const resp = try httpRequestWithStatus(allocator, method, url, body, headers, content_type, proxy);
    errdefer allocator.free(resp.body);
    if (resp.status_code < 200 or resp.status_code >= 300) return error.HttpStatusError;
    return resp.body;
}

pub fn httpPostJsonWithProxy(
    allocator: Allocator,
    url: []const u8,
    body: []const u8,
    headers: []const []const u8,
    proxy: ?[]const u8,
) ![]u8 {
    return httpRequest(allocator, .POST, url, body, headers, "application/json", proxy);
}

pub fn httpGetWithProxy(
    allocator: Allocator,
    url: []const u8,
    headers: []const []const u8,
    proxy: ?[]const u8,
) ![]u8 {
    return httpRequest(allocator, .GET, url, null, headers, null, proxy);
}

const proxy_env_var_names = [_][]const u8{
    "http_proxy",
    "HTTP_PROXY",
    "https_proxy",
    "HTTPS_PROXY",
    "all_proxy",
    "ALL_PROXY",
};
const http_proxy_env_var_names = [_][]const u8{
    "http_proxy",
    "HTTP_PROXY",
    "all_proxy",
    "ALL_PROXY",
};
const https_proxy_env_var_names = [_][]const u8{
    "https_proxy",
    "HTTPS_PROXY",
    "all_proxy",
    "ALL_PROXY",
};

pub const ProxyHttpClient = struct {
    proxy_arena: std.heap.ArenaAllocator,
    client: std.http.Client,

    pub fn init(allocator: Allocator) !ProxyHttpClient {
        var proxy_arena = std.heap.ArenaAllocator.init(allocator);
        errdefer proxy_arena.deinit();

        var client: std.http.Client = .{ .allocator = allocator, .io = std_compat.io() };
        errdefer client.deinit();

        try initClientDefaultProxies(&client, proxy_arena.allocator());

        return .{
            .proxy_arena = proxy_arena,
            .client = client,
        };
    }

    pub fn deinit(self: *ProxyHttpClient) void {
        self.client.deinit();
        self.proxy_arena.deinit();
        self.* = undefined;
    }
};

pub const SafeResolveEntryError = Allocator.Error || error{
    InvalidUrl,
    HostResolutionFailed,
    LocalAddressBlocked,
};

fn defaultPortForScheme(uri: std.Uri) ?u16 {
    if (uri.port) |port| return port;
    if (std.ascii.eqlIgnoreCase(uri.scheme, "https")) return 443;
    if (std.ascii.eqlIgnoreCase(uri.scheme, "http")) return 80;
    return null;
}

fn shouldUseCurlResolve(host: []const u8) bool {
    return std.mem.indexOfScalar(u8, net_security.stripHostBrackets(host), ':') == null;
}

pub fn shouldUsePinnedResolve(host: []const u8, connect_host: []const u8) bool {
    return shouldUseCurlResolve(host) and !std.mem.eql(u8, host, connect_host);
}

pub fn buildCurlResolveEntry(
    allocator: Allocator,
    host: []const u8,
    port: u16,
    connect_host: []const u8,
) ![]u8 {
    const host_for_resolve = net_security.stripHostBrackets(host);
    const connect_target = if (std.mem.indexOfScalar(u8, connect_host, ':') != null)
        try std.fmt.allocPrint(allocator, "[{s}]", .{connect_host})
    else
        try allocator.dupe(u8, connect_host);
    defer allocator.free(connect_target);

    return std.fmt.allocPrint(allocator, "{s}:{d}:{s}", .{ host_for_resolve, port, connect_target });
}

/// Build an optional libcurl address pin for remote provider requests.
/// Remote hosts are pinned to a concrete globally-routable address; explicit
/// local/private hosts are left untouched so intentional local providers still work.
///
/// DNS resolution failures are fail-closed. Falling back to curl's resolver would
/// bypass the single resolved-address check and weaken SSRF protection against
/// DNS answers that resolve to local/private networks.
pub fn buildSafeResolveEntryForRemoteUrl(
    allocator: Allocator,
    url: []const u8,
) SafeResolveEntryError!?[]u8 {
    const uri = std.Uri.parse(url) catch return error.InvalidUrl;
    const port = defaultPortForScheme(uri) orelse return error.InvalidUrl;
    const host = net_security.extractHost(url) orelse return error.InvalidUrl;

    if (net_security.isLocalHost(host)) return null;

    const connect_host = net_security.resolveConnectHost(allocator, host, port) catch |err|
        return mapResolveConnectHostError(host, err);
    defer allocator.free(connect_host);

    if (!shouldUsePinnedResolve(host, connect_host)) return null;
    return try buildCurlResolveEntry(allocator, host, port, connect_host);
}

fn mapResolveConnectHostError(host: []const u8, err: net_security.ResolveConnectHostError) SafeResolveEntryError {
    return switch (err) {
        error.HostResolutionFailed => blk: {
            log.debug("host resolution unavailable for {s}; failing closed", .{host});
            break :blk error.HostResolutionFailed;
        },
        error.LocalAddressBlocked => error.LocalAddressBlocked,
        error.OutOfMemory => error.OutOfMemory,
    };
}

/// HTTP POST via in-process libcurl with optional proxy and timeout.
///
/// `headers` is a slice of header strings (e.g. `"Authorization: Bearer xxx"`).
/// `proxy` is an optional proxy URL (e.g. `"socks5://host:port"`).
/// `max_time` is an optional total timeout as a string (e.g. `"300"`).
/// Returns the response body. Caller owns returned memory.
pub fn curlPostWithProxy(
    allocator: Allocator,
    url: []const u8,
    body: []const u8,
    headers: []const []const u8,
    proxy: ?[]const u8,
    max_time: ?[]const u8,
) ![]u8 {
    return curlPostWithProxyAndResolve(allocator, url, body, headers, proxy, max_time, null);
}

pub fn curlPostWithProxyAndResolve(
    allocator: Allocator,
    url: []const u8,
    body: []const u8,
    headers: []const []const u8,
    proxy: ?[]const u8,
    max_time: ?[]const u8,
    resolve_entry: ?[]const u8,
) ![]u8 {
    return curlRequestWithProxy(
        allocator,
        "POST",
        "Content-Type: application/json",
        url,
        body,
        headers,
        proxy,
        max_time,
        resolve_entry,
    );
}

/// HTTP POST with application/x-www-form-urlencoded body via libcurl,
/// with optional proxy and timeout.
pub fn curlPostFormWithProxy(
    allocator: Allocator,
    url: []const u8,
    body: []const u8,
    proxy: ?[]const u8,
    max_time: ?[]const u8,
) ![]u8 {
    return curlPostFormWithProxyAndResolve(allocator, url, body, proxy, max_time, null);
}

pub fn curlPostFormWithProxyAndResolve(
    allocator: Allocator,
    url: []const u8,
    body: []const u8,
    proxy: ?[]const u8,
    max_time: ?[]const u8,
    resolve_entry: ?[]const u8,
) ![]u8 {
    return curlRequestWithProxy(
        allocator,
        "POST",
        "Content-Type: application/x-www-form-urlencoded",
        url,
        body,
        &.{},
        proxy,
        max_time,
        resolve_entry,
    );
}

fn nativeRequest(
    allocator: Allocator,
    method: native_http.Method,
    url: []const u8,
    body: ?[]const u8,
    headers: []const []const u8,
    content_type_header: ?[]const u8,
    proxy: ?[]const u8,
    max_time: ?[]const u8,
    resolve_entry: ?[]const u8,
    max_body_bytes: usize,
    capture_headers: bool,
    fail_http_status: bool,
) !native_http.Response {
    const additional: usize = if (content_type_header == null) 0 else 1;
    if (headers.len + additional > 32) return error.TooManyHeaders;
    var all_headers: [32][]const u8 = undefined;
    if (content_type_header) |header| all_headers[0] = header;
    for (headers, 0..) |header, i| all_headers[i + additional] = header;
    const timeout: ?u64 = if (max_time) |value|
        std.fmt.parseInt(u64, value, 10) catch return error.CurlFailed
    else
        null;
    var response = try native_http.perform(allocator, .{
        .method = method,
        .url = url,
        .body = body,
        .headers = all_headers[0 .. headers.len + additional],
        .proxy = proxy,
        .resolve_entry = resolve_entry,
        .timeout_secs = timeout,
        .max_body_bytes = max_body_bytes,
        .capture_headers = capture_headers,
        .interrupt_flag = thread_interrupt_flag,
    });
    if (response.transport_error) |err| {
        response.deinit(allocator);
        return err;
    }
    if (fail_http_status and response.status >= 400) {
        response.deinit(allocator);
        return error.CurlFailed;
    }
    return response;
}

fn bodyOnly(allocator: Allocator, response: native_http.Response) []u8 {
    allocator.free(response.headers);
    return response.body;
}

fn curlRequestWithProxy(
    allocator: Allocator,
    method: []const u8,
    content_type_header: []const u8,
    url: []const u8,
    body: []const u8,
    headers: []const []const u8,
    proxy: ?[]const u8,
    max_time: ?[]const u8,
    resolve_entry: ?[]const u8,
) ![]u8 {
    const parsed_method: native_http.Method = if (std.mem.eql(u8, method, "POST")) .post else if (std.mem.eql(u8, method, "PUT")) .put else return error.UnsupportedHttpMethod;
    const response = try nativeRequest(allocator, parsed_method, url, body, headers, content_type_header, proxy, max_time, resolve_entry, DEFAULT_CURL_POST_MAX_BYTES, false, false);
    return bodyOnly(allocator, response);
}

/// HTTP POST via in-process libcurl (no proxy, no timeout).
pub fn curlPost(allocator: Allocator, url: []const u8, body: []const u8, headers: []const []const u8) ![]u8 {
    return curlPostWithProxy(allocator, url, body, headers, null, null);
}

/// HTTP POST with application/x-www-form-urlencoded body via libcurl.
///
/// `body` must already be percent-encoded form data (e.g. `"key=val&key2=val2"`).
/// Returns the response body. Caller owns returned memory.
pub fn curlPostForm(allocator: Allocator, url: []const u8, body: []const u8) ![]u8 {
    return curlPostFormWithProxy(allocator, url, body, null, null);
}

/// HTTP POST via in-process libcurl and include HTTP status code in response.
/// Caller owns `response.body`.
pub fn curlPostWithStatus(
    allocator: Allocator,
    url: []const u8,
    body: []const u8,
    headers: []const []const u8,
) !HttpResponse {
    return curlPostWithStatusAndTimeout(allocator, url, body, headers, null);
}

pub fn curlGetWithStatus(
    allocator: Allocator,
    url: []const u8,
    headers: []const []const u8,
) !HttpResponse {
    return curlGetWithStatusAndTimeout(allocator, url, headers, null);
}

/// HTTP POST via in-process libcurl and include HTTP status code in response,
/// with optional --max-time timeout.
/// Caller owns `response.body`.
pub fn curlPostWithStatusAndTimeout(
    allocator: Allocator,
    url: []const u8,
    body: []const u8,
    headers: []const []const u8,
    max_time: ?[]const u8,
) !HttpResponse {
    return curlPostWithStatusAndTimeoutAndResolve(allocator, url, body, headers, max_time, null);
}

pub fn curlPostWithStatusAndTimeoutAndResolve(
    allocator: Allocator,
    url: []const u8,
    body: []const u8,
    headers: []const []const u8,
    max_time: ?[]const u8,
    resolve_entry: ?[]const u8,
) !HttpResponse {
    const response = try nativeRequest(allocator, .post, url, body, headers, "Content-Type: application/json", null, max_time, resolve_entry, DEFAULT_CURL_POST_MAX_BYTES, false, false);
    return .{ .status_code = response.status, .body = bodyOnly(allocator, response) };
}

/// HTTP POST via in-process libcurl and include HTTP status code and response headers,
/// with optional --max-time timeout.
/// Caller owns `response.headers` and `response.body`.
pub fn curlPostWithStatusHeadersAndTimeout(
    allocator: Allocator,
    url: []const u8,
    body: []const u8,
    headers: []const []const u8,
    max_time: ?[]const u8,
) !HttpResponseWithHeaders {
    return curlPostWithStatusHeadersAndTimeoutAndResolve(allocator, url, body, headers, max_time, null);
}

pub fn curlPostWithStatusHeadersAndTimeoutAndResolve(
    allocator: Allocator,
    url: []const u8,
    body: []const u8,
    headers: []const []const u8,
    max_time: ?[]const u8,
    resolve_entry: ?[]const u8,
) !HttpResponseWithHeaders {
    const response = try nativeRequest(allocator, .post, url, body, headers, "Content-Type: application/json", null, max_time, resolve_entry, 1024 * 1024, true, false);
    return .{ .status_code = response.status, .headers = response.headers, .body = response.body };
}

pub fn curlGetWithStatusAndTimeout(
    allocator: Allocator,
    url: []const u8,
    headers: []const []const u8,
    max_time: ?[]const u8,
) !HttpResponse {
    return curlGetWithStatusAndTimeoutAndResolve(allocator, url, headers, max_time, null);
}

pub fn curlGetWithStatusAndTimeoutAndResolve(
    allocator: Allocator,
    url: []const u8,
    headers: []const []const u8,
    max_time: ?[]const u8,
    resolve_entry: ?[]const u8,
) !HttpResponse {
    const response = try nativeRequest(allocator, .get, url, null, headers, null, null, max_time, resolve_entry, DEFAULT_CURL_GET_MAX_BYTES, false, false);
    return .{ .status_code = response.status, .body = bodyOnly(allocator, response) };
}

/// HTTP PUT via in-process libcurl (no proxy, no timeout).
pub fn curlPut(allocator: Allocator, url: []const u8, body: []const u8, headers: []const []const u8) ![]u8 {
    return curlRequestWithProxy(
        allocator,
        "PUT",
        "Content-Type: application/json",
        url,
        body,
        headers,
        null,
        null,
        null,
    );
}

/// HTTP GET via in-process libcurl with optional proxy.
///
/// `headers` is a slice of header strings (e.g. `"Authorization: Bearer xxx"`).
/// `timeout_secs` sets the total transfer deadline. Returns the response body. Caller owns returned memory.
fn curlGetWithProxyAndResolve(
    allocator: Allocator,
    url: []const u8,
    headers: []const []const u8,
    timeout_secs: []const u8,
    proxy: ?[]const u8,
    resolve_entry: ?[]const u8,
    max_bytes: usize,
) ![]u8 {
    const response = try nativeRequest(allocator, .get, url, null, headers, null, proxy, timeout_secs, resolve_entry, max_bytes, false, true);
    return bodyOnly(allocator, response);
}

/// HTTP GET via in-process libcurl with optional proxy.
///
/// `headers` is a slice of header strings (e.g. `"Authorization: Bearer xxx"`).
/// `timeout_secs` sets the total transfer deadline. Returns the response body. Caller owns returned memory.
pub fn curlGetWithProxy(
    allocator: Allocator,
    url: []const u8,
    headers: []const []const u8,
    timeout_secs: []const u8,
    proxy: ?[]const u8,
) ![]u8 {
    return curlGetWithProxyAndResolve(allocator, url, headers, timeout_secs, proxy, null, DEFAULT_CURL_GET_MAX_BYTES);
}

/// HTTP GET via in-process libcurl with a pinned host mapping.
///
/// `resolve_entry` must be in curl `--resolve` format: `host:port:address`.
pub fn curlGetWithResolve(
    allocator: Allocator,
    url: []const u8,
    headers: []const []const u8,
    timeout_secs: []const u8,
    resolve_entry: []const u8,
) ![]u8 {
    return curlGetWithProxyAndResolve(allocator, url, headers, timeout_secs, null, resolve_entry, DEFAULT_CURL_GET_MAX_BYTES);
}

/// HTTP GET via in-process libcurl (no proxy).
pub fn curlGet(allocator: Allocator, url: []const u8, headers: []const []const u8, timeout_secs: []const u8) ![]u8 {
    return curlGetWithProxy(allocator, url, headers, timeout_secs, null);
}

/// HTTP GET via in-process libcurl with a caller-provided response size cap.
pub fn curlGetMaxBytes(
    allocator: Allocator,
    url: []const u8,
    headers: []const []const u8,
    timeout_secs: []const u8,
    max_bytes: usize,
) ![]u8 {
    return curlGetWithProxyAndResolve(allocator, url, headers, timeout_secs, null, null, max_bytes);
}

/// Read proxy URL from standard environment variables.
/// Checks https_proxy/HTTPS_PROXY first, then http_proxy/HTTP_PROXY,
/// then all_proxy/ALL_PROXY.
/// Returns null if no proxy is set.
/// Caller owns returned memory.
var proxy_override_value: ?[]u8 = null;
var proxy_override_mutex: std_compat.sync.Mutex = .{};

pub const ProxyOverrideError = error{OutOfMemory};

/// Set process-wide proxy override from config.
/// When set, this value has higher priority than proxy environment variables.
pub fn setProxyOverride(proxy: ?[]const u8) ProxyOverrideError!void {
    proxy_override_mutex.lock();
    defer proxy_override_mutex.unlock();

    if (proxy_override_value) |existing| {
        std.heap.page_allocator.free(existing);
        proxy_override_value = null;
    }

    if (proxy) |raw| {
        const trimmed = std.mem.trim(u8, raw, " \t\r\n");
        if (trimmed.len == 0) return;
        proxy_override_value = try std.heap.page_allocator.dupe(u8, trimmed);
    }
}

fn normalizeProxyEnvValue(allocator: Allocator, val: []const u8) !?[]const u8 {
    const trimmed = std.mem.trim(u8, val, " \t\r\n");
    if (trimmed.len == 0) return null;
    return try allocator.dupe(u8, trimmed);
}

fn applyProxyOverrideToEnvMap(env_map: *std_compat.process.EnvMap) !bool {
    proxy_override_mutex.lock();
    defer proxy_override_mutex.unlock();

    const override = proxy_override_value orelse return false;
    for (proxy_env_var_names) |key| {
        try env_map.put(key, override);
    }
    return true;
}

fn putProxyEnvVarFromProcess(
    env_map: *std_compat.process.EnvMap,
    allocator: Allocator,
    key: []const u8,
) !void {
    if (std_compat.process.getEnvVarOwned(allocator, key)) |raw_value| {
        defer allocator.free(raw_value);
        if (try normalizeProxyEnvValue(allocator, raw_value)) |proxy| {
            defer allocator.free(proxy);
            try env_map.put(key, proxy);
        }
    } else |_| {}
}

fn buildProxyEnvMapFromProcess(allocator: Allocator) !std_compat.process.EnvMap {
    var env_map = std_compat.process.EnvMap.init(allocator);
    errdefer env_map.deinit();

    for (proxy_env_var_names) |key| {
        try putProxyEnvVarFromProcess(&env_map, allocator, key);
    }
    _ = try applyProxyOverrideToEnvMap(&env_map);

    return env_map;
}

fn getProxyFromEnvMap(
    allocator: Allocator,
    env_map: *const std_compat.process.EnvMap,
    env_vars: []const []const u8,
) !?[]const u8 {
    for (env_vars) |var_name| {
        const raw_value = env_map.get(var_name) orelse continue;
        if (try normalizeProxyEnvValue(allocator, raw_value)) |proxy| {
            return proxy;
        }
    }
    return null;
}

fn initClientDefaultProxiesFromEnvMap(
    client: *std.http.Client,
    arena: Allocator,
    env_map: *const std_compat.process.EnvMap,
) !void {
    var merged_env_map = try env_map.clone(arena);
    _ = try applyProxyOverrideToEnvMap(&merged_env_map);
    try client.initDefaultProxies(arena, &merged_env_map);
}

pub fn initClientDefaultProxies(client: *std.http.Client, arena: Allocator) !void {
    var env_map = try buildProxyEnvMapFromProcess(arena);
    try client.initDefaultProxies(arena, &env_map);
}

pub fn getProxyFromEnv(allocator: Allocator) !?[]const u8 {
    const override = blk: {
        proxy_override_mutex.lock();
        defer proxy_override_mutex.unlock();
        if (proxy_override_value) |value| break :blk try allocator.dupe(u8, value);
        break :blk null;
    };
    if (override) |value| return value;
    for (https_proxy_env_var_names ++ http_proxy_env_var_names) |key| {
        const raw = std_compat.process.getEnvVarOwned(allocator, key) catch continue;
        defer allocator.free(raw);
        if (try normalizeProxyEnvValue(allocator, raw)) |value| return value;
    }
    return null;
}

// ── Tests ───────────────────────────────────────────────────────────

const CredentialedNativeHelper = enum {
    get_body,
    get_status,
    post_body,
    post_status,
    post_status_headers,
    put_body,
};

const CredentialedNativeServerCtx = struct {
    server: *std_compat.net.Server,
    expected_method: []const u8,
    saw_request: AtomicBool = AtomicBool.init(false),
    saw_expected_method: AtomicBool = AtomicBool.init(false),
    saw_authorization: AtomicBool = AtomicBool.init(false),
};

fn serveCredentialedNativeTest(ctx: *CredentialedNativeServerCtx) void {
    var conn = ctx.server.accept() catch return;
    defer conn.stream.close();

    var buf: [2048]u8 = undefined;
    var filled: usize = 0;
    while (filled < buf.len) {
        const n = conn.stream.read(buf[filled..]) catch return;
        if (n == 0) break;
        filled += n;
        if (std.mem.indexOf(u8, buf[0..filled], "\r\n\r\n") != null) break;
    }

    const request = buf[0..filled];
    ctx.saw_request.store(true, .release);
    if (std.mem.startsWith(u8, request, ctx.expected_method) and
        request.len > ctx.expected_method.len and
        request[ctx.expected_method.len] == ' ')
    {
        ctx.saw_expected_method.store(true, .release);
    }
    if (std.mem.indexOf(u8, request, "Authorization: Bearer test-token") != null) {
        ctx.saw_authorization.store(true, .release);
    }

    const response =
        "HTTP/1.1 200 OK\r\n" ++
        "Content-Type: application/json\r\n" ++
        "Content-Length: 11\r\n" ++
        "Connection: close\r\n" ++
        "\r\n" ++
        "{\"ok\":true}";
    conn.stream.writeAll(response) catch {};
}

fn unblockCredentialedNativeServer(server: *std_compat.net.Server) void {
    var conn = std_compat.net.tcpConnectToAddress(server.listen_address) catch return;
    conn.close();
}

fn expectCredentialedNative(helper: CredentialedNativeHelper, expected_method: []const u8) !void {
    if (comptime @import("builtin").os.tag == .wasi) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const addr = try std_compat.net.Address.resolveIp("127.0.0.1", 0);
    var server = try addr.listen(.{});
    defer server.deinit();

    var ctx = CredentialedNativeServerCtx{
        .server = &server,
        .expected_method = expected_method,
    };
    var thread = try std.Thread.spawn(.{}, serveCredentialedNativeTest, .{&ctx});

    const url = try std.fmt.allocPrint(allocator, "http://127.0.0.1:{d}/legacy", .{server.listen_address.in.getPort()});
    defer allocator.free(url);
    const headers = [_][]const u8{"Authorization: Bearer test-token"};
    var request_err: ?anyerror = null;

    switch (helper) {
        .get_body => {
            const body = curlGet(allocator, url, &headers, "5") catch |err| blk: {
                request_err = err;
                break :blk null;
            };
            if (body) |b| {
                defer allocator.free(b);
                try std.testing.expectEqualStrings("{\"ok\":true}", b);
            }
        },
        .get_status => {
            const resp = curlGetWithStatus(allocator, url, &headers) catch |err| blk: {
                request_err = err;
                break :blk null;
            };
            if (resp) |r| {
                defer allocator.free(r.body);
                try std.testing.expectEqual(@as(u16, 200), r.status_code);
                try std.testing.expectEqualStrings("{\"ok\":true}", r.body);
            }
        },
        .post_body => {
            const body = curlPost(allocator, url, "{\"ping\":true}", &headers) catch |err| blk: {
                request_err = err;
                break :blk null;
            };
            if (body) |b| {
                defer allocator.free(b);
                try std.testing.expectEqualStrings("{\"ok\":true}", b);
            }
        },
        .post_status => {
            const resp = curlPostWithStatus(allocator, url, "{\"ping\":true}", &headers) catch |err| blk: {
                request_err = err;
                break :blk null;
            };
            if (resp) |r| {
                defer allocator.free(r.body);
                try std.testing.expectEqual(@as(u16, 200), r.status_code);
                try std.testing.expectEqualStrings("{\"ok\":true}", r.body);
            }
        },
        .post_status_headers => {
            const resp = curlPostWithStatusHeadersAndTimeout(allocator, url, "{\"ping\":true}", &headers, null) catch |err| blk: {
                request_err = err;
                break :blk null;
            };
            if (resp) |r| {
                defer allocator.free(r.headers);
                defer allocator.free(r.body);
                try std.testing.expectEqual(@as(u16, 200), r.status_code);
                try std.testing.expectEqualStrings("{\"ok\":true}", r.body);
            }
        },
        .put_body => {
            const body = curlPut(allocator, url, "{\"ping\":true}", &headers) catch |err| blk: {
                request_err = err;
                break :blk null;
            };
            if (body) |b| {
                defer allocator.free(b);
                try std.testing.expectEqualStrings("{\"ok\":true}", b);
            }
        },
    }

    if (!ctx.saw_request.load(.acquire)) {
        unblockCredentialedNativeServer(&server);
    }
    thread.join();

    if (request_err) |err| return err;
    try std.testing.expect(ctx.saw_expected_method.load(.acquire));
    try std.testing.expect(ctx.saw_authorization.load(.acquire));
}

fn expectCredentialedResolveEntry(helper: CredentialedNativeHelper, expected_method: []const u8) !void {
    if (comptime @import("builtin").os.tag == .wasi) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const addr = try std_compat.net.Address.resolveIp("127.0.0.1", 0);
    var server = try addr.listen(.{});
    defer server.deinit();

    var ctx = CredentialedNativeServerCtx{
        .server = &server,
        .expected_method = expected_method,
    };
    var thread = try std.Thread.spawn(.{}, serveCredentialedNativeTest, .{&ctx});

    const host = "credentialed-curl.test";
    const port = server.listen_address.in.getPort();
    const url = try std.fmt.allocPrint(allocator, "http://{s}:{d}/legacy", .{ host, port });
    defer allocator.free(url);
    const resolve_entry = try std.fmt.allocPrint(allocator, "{s}:{d}:127.0.0.1", .{ host, port });
    defer allocator.free(resolve_entry);
    const headers = [_][]const u8{"Authorization: Bearer test-token"};
    var request_err: ?anyerror = null;

    switch (helper) {
        .get_body => {
            const body = curlGetWithResolve(allocator, url, &headers, "5", resolve_entry) catch |err| blk: {
                request_err = err;
                break :blk null;
            };
            if (body) |b| {
                defer allocator.free(b);
                try std.testing.expectEqualStrings("{\"ok\":true}", b);
            }
        },
        .get_status => {
            const resp = curlGetWithStatusAndTimeoutAndResolve(allocator, url, &headers, "5", resolve_entry) catch |err| blk: {
                request_err = err;
                break :blk null;
            };
            if (resp) |r| {
                defer allocator.free(r.body);
                try std.testing.expectEqual(@as(u16, 200), r.status_code);
                try std.testing.expectEqualStrings("{\"ok\":true}", r.body);
            }
        },
        .post_body => {
            const body = curlPostWithProxyAndResolve(allocator, url, "{\"ping\":true}", &headers, null, "5", resolve_entry) catch |err| blk: {
                request_err = err;
                break :blk null;
            };
            if (body) |b| {
                defer allocator.free(b);
                try std.testing.expectEqualStrings("{\"ok\":true}", b);
            }
        },
        .post_status => {
            const resp = curlPostWithStatusAndTimeoutAndResolve(allocator, url, "{\"ping\":true}", &headers, "5", resolve_entry) catch |err| blk: {
                request_err = err;
                break :blk null;
            };
            if (resp) |r| {
                defer allocator.free(r.body);
                try std.testing.expectEqual(@as(u16, 200), r.status_code);
                try std.testing.expectEqualStrings("{\"ok\":true}", r.body);
            }
        },
        .post_status_headers => {
            const resp = curlPostWithStatusHeadersAndTimeoutAndResolve(allocator, url, "{\"ping\":true}", &headers, "5", resolve_entry) catch |err| blk: {
                request_err = err;
                break :blk null;
            };
            if (resp) |r| {
                defer allocator.free(r.headers);
                defer allocator.free(r.body);
                try std.testing.expectEqual(@as(u16, 200), r.status_code);
                try std.testing.expectEqualStrings("{\"ok\":true}", r.body);
            }
        },
        .put_body => unreachable,
    }

    if (!ctx.saw_request.load(.acquire)) {
        unblockCredentialedNativeServer(&server);
    }
    thread.join();

    if (request_err) |err| return err;
    try std.testing.expect(ctx.saw_expected_method.load(.acquire));
    try std.testing.expect(ctx.saw_authorization.load(.acquire));
}

test "credentialed native body helpers do not reject authorization headers" {
    // Channel callers retain their headers when using native HTTP.
    try expectCredentialedNative(.get_body, "GET");
    try expectCredentialedNative(.post_body, "POST");
    try expectCredentialedNative(.put_body, "PUT");
}

test "credentialed native status helpers do not reject authorization headers" {
    // Regression: status-returning helpers used by Lark/QQ/OneBot must preserve
    // behavior through the in-process transport.
    try expectCredentialedNative(.get_status, "GET");
    try expectCredentialedNative(.post_status, "POST");
    try expectCredentialedNative(.post_status_headers, "POST");
}

test "credentialed curl helpers preserve resolve pinning" {
    // Native requests retain the validated address pin.
    try expectCredentialedResolveEntry(.get_body, "GET");
    try expectCredentialedResolveEntry(.post_body, "POST");
    try expectCredentialedResolveEntry(.get_status, "GET");
    try expectCredentialedResolveEntry(.post_status, "POST");
    try expectCredentialedResolveEntry(.post_status_headers, "POST");
}

test "native request rejects injected URL headers before network access" {
    try std.testing.expectError(
        error.InvalidHeader,
        curlGetWithResolve(
            std.testing.allocator,
            "https://example.com/v1\r\nX-Injected: value",
            &.{},
            "5",
            "example.com:443:203.0.113.10",
        ),
    );
}

test "buildSafeResolveEntryForRemoteUrl allows explicit local host without pinning" {
    try std.testing.expect((try buildSafeResolveEntryForRemoteUrl(std.testing.allocator, "http://127.0.0.1:11434/api/chat")) == null);
}

test "buildSafeResolveEntryForRemoteUrl rejects loopback integer alias" {
    try std.testing.expectError(error.LocalAddressBlocked, buildSafeResolveEntryForRemoteUrl(std.testing.allocator, "https://2130706433/v1"));
}

test "buildSafeResolveEntryForRemoteUrl maps resolution failure to fail closed" {
    // Resolver failure must not bypass private-address screening.
    try std.testing.expect(mapResolveConnectHostError("example.com", error.HostResolutionFailed) == error.HostResolutionFailed);
}

test "buildSafeResolveEntryForRemoteUrl rejects malformed URL" {
    try std.testing.expectError(error.InvalidUrl, buildSafeResolveEntryForRemoteUrl(std.testing.allocator, "notaurl"));
}

test "curl post max bytes is increased for large provider responses" {
    try std.testing.expect(DEFAULT_CURL_POST_MAX_BYTES >= 8 * 1024 * 1024);
}

test "preserveCurlTransportError preserves curl transport failures" {
    // Regression: provider probes need raw curl transport failures instead of a
    // provider-specific API error so they can report network_error correctly.
    try std.testing.expect(preserveCurlTransportError(error.CurlDnsError, error.ApiError) == error.CurlDnsError);
    try std.testing.expect(preserveCurlTransportError(error.CurlConnectError, error.ApiError) == error.CurlConnectError);
    try std.testing.expect(preserveCurlTransportError(error.CurlTimeout, error.ApiError) == error.CurlTimeout);
    try std.testing.expect(preserveCurlTransportError(error.CurlTlsError, error.ApiError) == error.CurlTlsError);
    try std.testing.expect(preserveCurlTransportError(error.CurlReadError, error.ApiError) == error.CurlReadError);
    try std.testing.expect(preserveCurlTransportError(error.CurlWriteError, error.ApiError) == error.CurlWriteError);
    try std.testing.expect(preserveCurlTransportError(error.CurlWaitError, error.ApiError) == error.CurlWaitError);
    try std.testing.expect(preserveCurlTransportError(error.CurlFailed, error.ApiError) == error.CurlFailed);
    try std.testing.expect(preserveCurlTransportError(error.CurlInterrupted, error.ApiError) == error.CurlInterrupted);
}

test "preserveCurlTransportError returns fallback for non-transport failures" {
    try std.testing.expect(preserveCurlTransportError(error.RateLimited, error.ApiError) == error.ApiError);
    try std.testing.expect(preserveCurlTransportError(error.InvalidUrl, error.ApiError) == error.ApiError);
}

test "normalizeProxyEnvValue trims surrounding whitespace" {
    const alloc = std.testing.allocator;
    const normalized = try normalizeProxyEnvValue(alloc, "  socks5://127.0.0.1:1080 \r\n");
    defer if (normalized) |v| alloc.free(v);
    try std.testing.expect(normalized != null);
    try std.testing.expectEqualStrings("socks5://127.0.0.1:1080", normalized.?);
}

test "normalizeProxyEnvValue rejects empty values" {
    const normalized = try normalizeProxyEnvValue(std.testing.allocator, " \t\r\n");
    try std.testing.expect(normalized == null);
}

test "setProxyOverride applies and clears process-wide override" {
    const override = "  socks5://proxy-override-test.invalid:1080  ";
    const normalized_override = "socks5://proxy-override-test.invalid:1080";

    try setProxyOverride(override);
    const from_override = try getProxyFromEnv(std.testing.allocator);
    defer if (from_override) |v| std.testing.allocator.free(v);
    try std.testing.expect(from_override != null);
    try std.testing.expectEqualStrings(normalized_override, from_override.?);

    try setProxyOverride(null);
    const after_clear = try getProxyFromEnv(std.testing.allocator);
    defer if (after_clear) |v| std.testing.allocator.free(v);
    if (after_clear) |proxy| {
        // Environment may define a proxy; only assert our override no longer leaks.
        try std.testing.expect(!std.mem.eql(u8, proxy, normalized_override));
    }
}

test "setProxyOverride accepts long proxy URLs" {
    const allocator = std.testing.allocator;
    var long_proxy = try allocator.alloc(u8, 1600);
    defer allocator.free(long_proxy);

    @memcpy(long_proxy[0.."http://".len], "http://");
    @memset(long_proxy["http://".len..], 'a');

    try setProxyOverride(long_proxy);
    defer setProxyOverride(null) catch unreachable;

    const from_override = try getProxyFromEnv(allocator);
    defer if (from_override) |v| allocator.free(v);
    try std.testing.expect(from_override != null);
    try std.testing.expectEqual(long_proxy.len, from_override.?.len);
}

test "getProxyFromEnvMap honors lowercase https_proxy before http_proxy" {
    var env_map = std_compat.process.EnvMap.init(std.testing.allocator);
    defer env_map.deinit();

    try env_map.put("http_proxy", "http://http-only.example:8080");
    try env_map.put("https_proxy", "https://secure.example:8443");

    const proxy = try getProxyFromEnvMap(std.testing.allocator, &env_map, &https_proxy_env_var_names);
    defer if (proxy) |value| std.testing.allocator.free(value);

    try std.testing.expect(proxy != null);
    try std.testing.expectEqualStrings("https://secure.example:8443", proxy.?);
}

test "applyProxyOverrideToEnvMap overwrites existing proxy values" {
    var env_map = std_compat.process.EnvMap.init(std.testing.allocator);
    defer env_map.deinit();

    try env_map.put("HTTPS_PROXY", "https://old.example:9443");
    try setProxyOverride("  socks5://override.example:1080  ");
    defer setProxyOverride(null) catch unreachable;

    try std.testing.expect(try applyProxyOverrideToEnvMap(&env_map));
    try std.testing.expectEqualStrings("socks5://override.example:1080", env_map.get("HTTPS_PROXY").?);
    try std.testing.expectEqualStrings("socks5://override.example:1080", env_map.get("http_proxy").?);
}

test "initClientDefaultProxiesFromEnvMap parses proxy settings" {
    var env_map = std_compat.process.EnvMap.init(std.testing.allocator);
    defer env_map.deinit();

    try env_map.put("http_proxy", "http://proxy-http.example:8080");
    try env_map.put("HTTPS_PROXY", "https://proxy-https.example:8443");

    var proxy_arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer proxy_arena.deinit();

    var client: std.http.Client = .{ .allocator = std.testing.allocator, .io = std.testing.io };
    defer client.deinit();

    // Regression: Zig 0.16 requires an explicit environ map for initDefaultProxies.
    try initClientDefaultProxiesFromEnvMap(&client, proxy_arena.allocator(), &env_map);

    try std.testing.expect(client.http_proxy != null);
    try std.testing.expect(client.https_proxy != null);
    try std.testing.expectEqual(@as(u16, 8080), client.http_proxy.?.port);
    try std.testing.expectEqual(@as(u16, 8443), client.https_proxy.?.port);
    try std.testing.expect(client.http_proxy.?.host.eql(try std.Io.net.HostName.init("proxy-http.example")));
    try std.testing.expect(client.https_proxy.?.host.eql(try std.Io.net.HostName.init("proxy-https.example")));
}
