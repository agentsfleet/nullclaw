const builtin = @import("builtin");
const std = @import("std");
const root = @import("root.zig");
const Tool = root.Tool;
const ToolResult = root.ToolResult;
const JsonObjectMap = root.JsonObjectMap;
const net_security = @import("../root.zig").net_security;
const http_util = @import("../http_util.zig");
const native_http = @import("../native_http.zig");

const log = std.log.scoped(.http_request);

/// HTTP request tool for API interactions.
/// Supports GET, POST, PUT, DELETE, PATCH, HEAD, OPTIONS methods with
/// domain allowlisting, SSRF protection, and header redaction.
pub const HttpRequestTool = struct {
    allowed_domains: []const []const u8 = &.{}, // empty = allow all
    max_response_size: u32 = 1_000_000,
    timeout_secs: u64 = 60,

    pub const tool_name = "http_request";
    pub const tool_description = "Make HTTPS requests to external APIs. Supports GET, POST, PUT, DELETE, PATCH, HEAD, OPTIONS methods. " ++
        "Security: allowlist-only domains, SSRF protection, and allowlisted hosts may reach local/private addresses.";
    pub const tool_params =
        \\{"type":"object","properties":{"url":{"type":"string","description":"HTTPS URL to request"},"method":{"type":"string","description":"HTTP method (GET, POST, PUT, DELETE, PATCH, HEAD, OPTIONS)","default":"GET"},"headers":{"type":"object","description":"Optional HTTP headers as key-value pairs"},"body":{"type":"string","description":"Optional request body"}},"required":["url"]}
    ;

    const vtable = root.ToolVTable(@This());

    pub fn tool(self: *HttpRequestTool) Tool {
        return .{
            .ptr = @ptrCast(self),
            .vtable = &vtable,
        };
    }

    pub fn execute(self: *HttpRequestTool, allocator: std.mem.Allocator, args: JsonObjectMap) !ToolResult {
        const url = root.getString(args, "url") orelse
            return ToolResult.fail("Missing 'url' parameter");

        const method_str = root.getString(args, "method") orelse "GET";

        // Validate method first (cheap local operation, no network calls)
        const method = validateMethod(method_str) orelse {
            const msg = try std.fmt.allocPrint(allocator, "Unsupported HTTP method: {s}", .{method_str});
            return ToolResult{ .success = false, .output = "", .error_msg = msg };
        };

        // Validate URL scheme - HTTPS only for security (AGENTS.md policy)
        net_security.validateOutboundUrl(url) catch {
            return ToolResult.fail("Only HTTPS URLs are allowed for security");
        };

        // Build URI
        const uri = std.Uri.parse(url) catch
            return ToolResult.fail("Invalid URL format");

        const resolved_port: u16 = uri.port orelse 443;

        // Extract host
        const host = net_security.extractHost(url) orelse
            return ToolResult.fail("Invalid URL: cannot extract host");

        // Check domain allowlist BEFORE any DNS resolution.
        // This prevents DNS exfiltration and avoids unnecessary network calls.
        const is_allowlisted = if (self.allowed_domains.len > 0)
            net_security.hostMatchesAllowlist(host, self.allowed_domains)
        else
            false;

        // Reject non-allowlisted hosts when allowlist is configured
        if (self.allowed_domains.len > 0 and !is_allowlisted) {
            return ToolResult.fail("Host is not in http_request.allowed_domains");
        }

        // SSRF protection: skip for allowlisted hosts (fixes #393).
        // Allowlisted hosts can resolve to private IPs (e.g., local searxng).
        // Non-allowlisted hosts require global address validation.
        //
        // Security trade-off for allowlisted hosts:
        // - resolveConnectHost normally pins DNS results through libcurl,
        //   preventing DNS rebinding attacks between resolution and connection.
        // - For allowlisted hosts, we skip this and let libcurl resolve the hostname.
        //   This is acceptable because the operator explicitly trusts these domains
        //   (e.g., internal services like searxng on private IPs).
        // - DNS rebinding protection is intentionally traded for operational flexibility.
        const connect_host: []const u8 = if (is_allowlisted)
            // Allowlisted: trust the operator, skip SSRF check and DNS pinning.
            // libcurl will resolve the hostname itself (no address pinning).
            try allocator.dupe(u8, host)
        else
            // No allowlist configured: enforce SSRF for all external hosts.
            // DNS results are pinned to prevent rebinding attacks.
            net_security.resolveConnectHost(allocator, host, resolved_port) catch |err| switch (err) {
                error.LocalAddressBlocked => return ToolResult.fail("Blocked local/private host"),
                else => return ToolResult.fail("Unable to verify host safety"),
            };
        defer allocator.free(connect_host);

        // Parse custom headers from ObjectMap
        const headers_val = root.getValue(args, "headers");
        var header_list: std.ArrayList([2][]const u8) = .empty;
        defer {
            for (header_list.items) |h| {
                allocator.free(h[0]);
                allocator.free(h[1]);
            }
            header_list.deinit(allocator);
        }
        if (headers_val) |hv| {
            if (hv == .object) {
                var it = hv.object.iterator();
                while (it.next()) |entry| {
                    const val_str = switch (entry.value_ptr.*) {
                        .string => |s| s,
                        else => continue,
                    };
                    const name = try allocator.dupe(u8, entry.key_ptr.*);
                    const value = allocator.dupe(u8, val_str) catch |err| {
                        allocator.free(name);
                        return err;
                    };
                    header_list.append(allocator, .{ name, value }) catch |err| {
                        allocator.free(name);
                        allocator.free(value);
                        return err;
                    };
                }
            }
        }
        const custom_headers = header_list.items;

        const body: ?[]const u8 = root.getString(args, "body");

        if (builtin.is_test) {
            return ToolResult.fail("Network disabled in tests");
        }

        const status_result = runNativeRequestWithStatus(
            allocator,
            methodToSlice(method),
            url,
            host,
            resolved_port,
            connect_host,
            custom_headers,
            body,
            self.timeout_secs,
            @intCast(self.max_response_size),
        ) catch |err| {
            if (err == error.CurlInterrupted) {
                return ToolResult.fail("Interrupted by /stop");
            }
            const msg = try std.fmt.allocPrint(allocator, "HTTP request failed: {}", .{err});
            return ToolResult{ .success = false, .output = "", .error_msg = msg };
        };
        defer allocator.free(status_result.body);

        const status_code = status_result.status_code;
        const success = status_code >= 200 and status_code < 300;

        // Build redacted headers display for custom request headers
        const redacted = redactHeadersForDisplay(allocator, custom_headers) catch try allocator.dupe(u8, "");
        defer allocator.free(redacted);

        const output = if (redacted.len > 0)
            try std.fmt.allocPrint(
                allocator,
                "Status: {d}\nRequest Headers: {s}\n\nResponse Body:\n{s}",
                .{ status_code, redacted, status_result.body },
            )
        else
            try std.fmt.allocPrint(
                allocator,
                "Status: {d}\n\nResponse Body:\n{s}",
                .{ status_code, status_result.body },
            );

        if (success) {
            return ToolResult{ .success = true, .output = output };
        } else {
            const err_msg = try std.fmt.allocPrint(allocator, "HTTP {d}", .{status_code});
            return ToolResult{ .success = false, .output = output, .error_msg = err_msg };
        }
    }
};

fn methodToSlice(method: std.http.Method) []const u8 {
    return switch (method) {
        .GET => "GET",
        .POST => "POST",
        .PUT => "PUT",
        .DELETE => "DELETE",
        .PATCH => "PATCH",
        .HEAD => "HEAD",
        .OPTIONS => "OPTIONS",
        else => "GET",
    };
}

fn runNativeRequestWithStatus(
    allocator: std.mem.Allocator,
    method: []const u8,
    url: []const u8,
    host: []const u8,
    resolved_port: u16,
    connect_host: []const u8,
    headers: []const [2][]const u8,
    body: ?[]const u8,
    timeout_secs: u64,
    max_response_size: usize,
) !http_util.HttpResponse {
    const native_method: native_http.Method = if (std.mem.eql(u8, method, "GET")) .get else if (std.mem.eql(u8, method, "POST")) .post else if (std.mem.eql(u8, method, "PUT")) .put else if (std.mem.eql(u8, method, "PATCH")) .patch else if (std.mem.eql(u8, method, "DELETE")) .delete else if (std.mem.eql(u8, method, "HEAD")) .head else if (std.mem.eql(u8, method, "OPTIONS")) .options else return error.UnsupportedMethod;
    var resolve_entry: ?[]u8 = null;
    defer if (resolve_entry) |entry| allocator.free(entry);
    if (http_util.shouldUsePinnedResolve(host, connect_host))
        resolve_entry = try http_util.buildCurlResolveEntry(allocator, host, resolved_port, connect_host);

    if (headers.len > 32) return error.TooManyHeaders;
    var header_lines: [32][]u8 = undefined;
    var header_count: usize = 0;
    defer for (header_lines[0..header_count]) |line| allocator.free(line);
    for (headers) |header| {
        header_lines[header_count] = try std.fmt.allocPrint(allocator, "{s}: {s}", .{ header[0], header[1] });
        header_count += 1;
    }
    const proxy = try http_util.getProxyFromEnv(allocator);
    defer if (proxy) |value| allocator.free(value);
    var response = try native_http.perform(allocator, .{
        .method = native_method,
        .url = url,
        .body = body,
        .headers = header_lines[0..header_count],
        .proxy = proxy,
        .resolve_entry = resolve_entry,
        .timeout_secs = timeout_secs,
        .max_body_bytes = max_response_size,
        .interrupt_flag = http_util.currentThreadInterruptFlag(),
    });
    if (response.transport_error) |err| {
        response.deinit(allocator);
        return err;
    }
    allocator.free(response.headers);
    return .{ .status_code = response.status, .body = response.body };
}

fn ensureTlsCaBundleLoaded(client: *std.http.Client) !void {
    if (@atomicLoad(bool, &client.next_https_rescan_certs, .acquire)) {
        client.ca_bundle_mutex.lock();
        defer client.ca_bundle_mutex.unlock();

        if (client.next_https_rescan_certs) {
            client.ca_bundle.rescan(client.allocator) catch |err| switch (err) {
                error.OutOfMemory => return error.OutOfMemory,
                else => return error.CertificateBundleLoadFailure,
            };
            @atomicStore(bool, &client.next_https_rescan_certs, false, .release);
        }
    }
}

fn isTlsSetupError(err: anyerror) bool {
    return err == error.TlsInitializationFailed or err == error.CertificateBundleLoadFailure;
}

fn buildHttpRequestErrorMessage(allocator: std.mem.Allocator, prefix: []const u8, err: anyerror) ![]u8 {
    if (isTlsSetupError(err)) {
        return std.fmt.allocPrint(
            allocator,
            "{s}: {s}. Ensure system CA certificates are available in the runtime, or use an endpoint with a publicly trusted certificate chain.",
            .{ prefix, @errorName(err) },
        );
    }
    return std.fmt.allocPrint(allocator, "{s}: {}", .{ prefix, err });
}

fn validateMethod(method: []const u8) ?std.http.Method {
    if (std.ascii.eqlIgnoreCase(method, "GET")) return .GET;
    if (std.ascii.eqlIgnoreCase(method, "POST")) return .POST;
    if (std.ascii.eqlIgnoreCase(method, "PUT")) return .PUT;
    if (std.ascii.eqlIgnoreCase(method, "DELETE")) return .DELETE;
    if (std.ascii.eqlIgnoreCase(method, "PATCH")) return .PATCH;
    if (std.ascii.eqlIgnoreCase(method, "HEAD")) return .HEAD;
    if (std.ascii.eqlIgnoreCase(method, "OPTIONS")) return .OPTIONS;
    return null;
}

/// Disable auto-follow redirects so every hop can be explicitly validated.
fn buildRequestOptions(
    extra_headers: []const std.http.Header,
    connection: ?*std.http.Client.Connection,
) std.http.Client.RequestOptions {
    return .{
        .extra_headers = extra_headers,
        .redirect_behavior = .unhandled,
        .connection = connection,
    };
}

/// Parse headers from a JSON object string: {"Key": "Value", ...}
/// Returns array of [2][]const u8 pairs. Caller owns memory.
fn parseHeaders(allocator: std.mem.Allocator, headers_json: ?[]const u8) ![]const [2][]const u8 {
    const json = headers_json orelse return &.{};
    if (json.len < 2) return &.{};

    var list: std.ArrayList([2][]const u8) = .empty;
    errdefer {
        for (list.items) |h| {
            allocator.free(h[0]);
            allocator.free(h[1]);
        }
        list.deinit(allocator);
    }

    // Simple JSON object parser: find "key": "value" pairs
    var pos: usize = 0;
    while (pos < json.len) {
        // Find next key (quoted string)
        const key_start = std.mem.indexOfScalarPos(u8, json, pos, '"') orelse break;
        const key_end = std.mem.indexOfScalarPos(u8, json, key_start + 1, '"') orelse break;
        const key = json[key_start + 1 .. key_end];

        // Skip to colon and value
        pos = key_end + 1;
        const colon = std.mem.indexOfScalarPos(u8, json, pos, ':') orelse break;
        pos = colon + 1;

        // Skip whitespace
        while (pos < json.len and (json[pos] == ' ' or json[pos] == '\t' or json[pos] == '\n')) : (pos += 1) {}

        if (pos >= json.len or json[pos] != '"') {
            pos += 1;
            continue;
        }
        const val_start = pos;
        const val_end = std.mem.indexOfScalarPos(u8, json, val_start + 1, '"') orelse break;
        const value = json[val_start + 1 .. val_end];
        pos = val_end + 1;

        try list.append(allocator, .{
            try allocator.dupe(u8, key),
            try allocator.dupe(u8, value),
        });
    }

    return list.toOwnedSlice(allocator);
}

/// Redact sensitive headers for display output.
/// Headers with names containing authorization, api-key, apikey, token, secret,
/// or password (case-insensitive) get their values replaced with "***REDACTED***".
fn redactHeadersForDisplay(allocator: std.mem.Allocator, headers: []const [2][]const u8) ![]const u8 {
    if (headers.len == 0) return allocator.dupe(u8, "");

    var buf: std.ArrayList(u8) = .empty;
    errdefer buf.deinit(allocator);

    for (headers, 0..) |h, i| {
        if (i > 0) try buf.appendSlice(allocator, ", ");
        try buf.appendSlice(allocator, h[0]);
        try buf.appendSlice(allocator, ": ");
        if (isSensitiveHeader(h[0])) {
            try buf.appendSlice(allocator, "***REDACTED***");
        } else {
            try buf.appendSlice(allocator, h[1]);
        }
    }

    return buf.toOwnedSlice(allocator);
}

/// Check if a header name is sensitive (case-insensitive substring check).
fn isSensitiveHeader(name: []const u8) bool {
    // Convert to lowercase for comparison
    var lower_buf: [256]u8 = undefined;
    // Fail closed: oversized header names are treated as sensitive.
    if (name.len > lower_buf.len) return true;
    const lower = lower_buf[0..name.len];
    for (name, 0..) |c, i| {
        lower[i] = std.ascii.toLower(c);
    }
    if (std.mem.indexOf(u8, lower, "authorization") != null) return true;
    if (std.mem.indexOf(u8, lower, "api-key") != null) return true;
    if (std.mem.indexOf(u8, lower, "apikey") != null) return true;
    if (std.mem.indexOf(u8, lower, "token") != null) return true;
    if (std.mem.indexOf(u8, lower, "secret") != null) return true;
    if (std.mem.indexOf(u8, lower, "password") != null) return true;
    return false;
}

// ── Tests ───────────────────────────────────────────────────────────

test "http_request tool name" {
    var ht = HttpRequestTool{};
    const t = ht.tool();
    try std.testing.expectEqualStrings("http_request", t.name());
}

test "http_request tool description not empty" {
    var ht = HttpRequestTool{};
    const t = ht.tool();
    const description = t.description();
    try std.testing.expect(description.len > 0);
    try std.testing.expect(std.mem.indexOf(u8, description, "HTTPS") != null);
    try std.testing.expect(std.mem.indexOf(u8, description, "allowlisted hosts may reach local/private addresses") != null);
}

test "http_request schema has url" {
    var ht = HttpRequestTool{};
    const t = ht.tool();
    const schema = t.parametersJson();
    try std.testing.expect(std.mem.indexOf(u8, schema, "url") != null);
    try std.testing.expect(std.mem.indexOf(u8, schema, "HTTPS URL to request") != null);
    try std.testing.expect(std.mem.indexOf(u8, schema, "HTTP or HTTPS URL to request") == null);
}

test "http_request schema has headers" {
    var ht = HttpRequestTool{};
    const t = ht.tool();
    const schema = t.parametersJson();
    try std.testing.expect(std.mem.indexOf(u8, schema, "headers") != null);
}

test "validateMethod accepts valid methods" {
    try std.testing.expect(validateMethod("GET") != null);
    try std.testing.expect(validateMethod("POST") != null);
    try std.testing.expect(validateMethod("PUT") != null);
    try std.testing.expect(validateMethod("DELETE") != null);
    try std.testing.expect(validateMethod("PATCH") != null);
    try std.testing.expect(validateMethod("HEAD") != null);
    try std.testing.expect(validateMethod("OPTIONS") != null);
    try std.testing.expect(validateMethod("get") != null); // case insensitive
}

test "validateMethod rejects invalid" {
    try std.testing.expect(validateMethod("INVALID") == null);
}

// ── redactHeadersForDisplay tests ──────────────────────────

test "redactHeadersForDisplay redacts Authorization" {
    const headers = [_][2][]const u8{
        .{ "Authorization", "Bearer secret-token" },
    };
    const result = try redactHeadersForDisplay(std.testing.allocator, &headers);
    defer std.testing.allocator.free(result);
    try std.testing.expect(std.mem.indexOf(u8, result, "***REDACTED***") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "secret-token") == null);
}

test "redactHeadersForDisplay preserves Content-Type" {
    const headers = [_][2][]const u8{
        .{ "Content-Type", "application/json" },
    };
    const result = try redactHeadersForDisplay(std.testing.allocator, &headers);
    defer std.testing.allocator.free(result);
    try std.testing.expect(std.mem.indexOf(u8, result, "application/json") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "REDACTED") == null);
}

test "redactHeadersForDisplay redacts api-key and token" {
    const headers = [_][2][]const u8{
        .{ "X-API-Key", "my-key" },
        .{ "X-Secret-Token", "tok-123" },
    };
    const result = try redactHeadersForDisplay(std.testing.allocator, &headers);
    defer std.testing.allocator.free(result);
    try std.testing.expect(std.mem.indexOf(u8, result, "my-key") == null);
    try std.testing.expect(std.mem.indexOf(u8, result, "tok-123") == null);
}

test "redactHeadersForDisplay empty returns empty" {
    const result = try redactHeadersForDisplay(std.testing.allocator, &.{});
    defer std.testing.allocator.free(result);
    try std.testing.expectEqualStrings("", result);
}

test "isSensitiveHeader checks" {
    try std.testing.expect(isSensitiveHeader("Authorization"));
    try std.testing.expect(isSensitiveHeader("X-API-Key"));
    try std.testing.expect(isSensitiveHeader("X-Secret-Token"));
    try std.testing.expect(isSensitiveHeader("password-header"));
    try std.testing.expect(!isSensitiveHeader("Content-Type"));
    try std.testing.expect(!isSensitiveHeader("Accept"));
}

test "isSensitiveHeader treats oversized names as sensitive" {
    var long_name: [300]u8 = undefined;
    @memset(long_name[0..], 'a');
    try std.testing.expect(isSensitiveHeader(long_name[0..]));
}

test "http_request disables automatic redirects" {
    const opts = buildRequestOptions(&.{}, null);
    try std.testing.expect(opts.redirect_behavior == .unhandled);
    try std.testing.expect(opts.connection == null);
}

test "http_request request options keep provided connection" {
    const fake_ptr_value = @as(usize, @alignOf(std.http.Client.Connection));
    const fake_connection: *std.http.Client.Connection = @ptrFromInt(fake_ptr_value);
    const opts = buildRequestOptions(&.{}, fake_connection);
    try std.testing.expect(opts.connection != null);
    try std.testing.expectEqual(@intFromPtr(fake_connection), @intFromPtr(opts.connection.?));
}

// ── execute-level tests ──────────────────────────────────────

test "execute rejects missing url parameter" {
    var ht = HttpRequestTool{};
    const t = ht.tool();
    const parsed = try root.parseTestArgs("{}");
    defer parsed.deinit();
    const result = try t.execute(std.testing.allocator, parsed.value.object);
    try std.testing.expect(!result.success);
    try std.testing.expect(std.mem.indexOf(u8, result.error_msg.?, "url") != null);
}

test "execute rejects non-http scheme" {
    var ht = HttpRequestTool{};
    const t = ht.tool();
    const parsed = try root.parseTestArgs("{\"url\": \"ftp://example.com\"}");
    defer parsed.deinit();
    const result = try t.execute(std.testing.allocator, parsed.value.object);
    try std.testing.expect(!result.success);
    try std.testing.expect(std.mem.indexOf(u8, result.error_msg.?, "HTTPS") != null);
}

test "execute accepts uppercase https scheme before allowlist checks" {
    const domains = [_][]const u8{"allowed.example"};
    var ht = HttpRequestTool{ .allowed_domains = &domains };
    const t = ht.tool();
    const parsed = try root.parseTestArgs("{\"url\": \"HTTPS://blocked.example/path\"}");
    defer parsed.deinit();
    const result = try t.execute(std.testing.allocator, parsed.value.object);
    try std.testing.expect(!result.success);
    try std.testing.expectEqualStrings("Host is not in http_request.allowed_domains", result.error_msg.?);
}

test "execute rejects localhost SSRF" {
    var ht = HttpRequestTool{};
    const t = ht.tool();
    const parsed = try root.parseTestArgs("{\"url\": \"https://127.0.0.1:8080/admin\"}");
    defer parsed.deinit();
    const result = try t.execute(std.testing.allocator, parsed.value.object);
    try std.testing.expect(!result.success);
    try std.testing.expect(std.mem.indexOf(u8, result.error_msg.?, "local") != null);
}

test "execute rejects localhost SSRF with URL userinfo" {
    var ht = HttpRequestTool{};
    const t = ht.tool();
    const parsed = try root.parseTestArgs("{\"url\": \"https://user:pass@127.0.0.1:8080/admin\"}");
    defer parsed.deinit();
    const result = try t.execute(std.testing.allocator, parsed.value.object);
    try std.testing.expect(!result.success);
    try std.testing.expect(std.mem.indexOf(u8, result.error_msg.?, "local") != null);
}

test "execute rejects localhost SSRF with unbracketed ipv6 authority" {
    var ht = HttpRequestTool{};
    const t = ht.tool();
    const parsed = try root.parseTestArgs("{\"url\": \"https://[::1]:8080/admin\"}");
    defer parsed.deinit();
    const result = try t.execute(std.testing.allocator, parsed.value.object);
    try std.testing.expect(!result.success);
    try std.testing.expect(std.mem.indexOf(u8, result.error_msg.?, "local") != null);
}

test "execute rejects private IP SSRF" {
    var ht = HttpRequestTool{};
    const t = ht.tool();
    const parsed = try root.parseTestArgs("{\"url\": \"https://192.168.1.1/admin\"}");
    defer parsed.deinit();
    const result = try t.execute(std.testing.allocator, parsed.value.object);
    try std.testing.expect(!result.success);
}

test "execute rejects 10.x private range" {
    var ht = HttpRequestTool{};
    const t = ht.tool();
    const parsed = try root.parseTestArgs("{\"url\": \"https://10.0.0.1/secret\"}");
    defer parsed.deinit();
    const result = try t.execute(std.testing.allocator, parsed.value.object);
    try std.testing.expect(!result.success);
}

test "execute rejects loopback decimal alias SSRF" {
    var ht = HttpRequestTool{};
    const t = ht.tool();
    const parsed = try root.parseTestArgs("{\"url\": \"https://2130706433/admin\"}");
    defer parsed.deinit();
    const result = try t.execute(std.testing.allocator, parsed.value.object);
    try std.testing.expect(!result.success);
    try std.testing.expect(std.mem.indexOf(u8, result.error_msg.?, "local") != null);
}

test "execute rejects unsupported method" {
    var ht = HttpRequestTool{};
    const t = ht.tool();
    const parsed = try root.parseTestArgs("{\"url\": \"https://example.com\", \"method\": \"INVALID\"}");
    defer parsed.deinit();
    const result = try t.execute(std.testing.allocator, parsed.value.object);
    defer if (result.error_msg) |e| std.testing.allocator.free(e);
    try std.testing.expect(!result.success);
    try std.testing.expect(std.mem.indexOf(u8, result.error_msg.?, "Unsupported") != null);
}

test "execute rejects invalid URL format" {
    var ht = HttpRequestTool{};
    const t = ht.tool();
    const parsed = try root.parseTestArgs("{\"url\": \"http://\"}");
    defer parsed.deinit();
    const result = try t.execute(std.testing.allocator, parsed.value.object);
    try std.testing.expect(!result.success);
}

test "execute rejects non-allowlisted domain" {
    const domains = [_][]const u8{"example.com"};
    var ht = HttpRequestTool{ .allowed_domains = &domains };
    const t = ht.tool();
    const parsed = try root.parseTestArgs("{\"url\": \"https://evil.com/path\"}");
    defer parsed.deinit();
    const result = try t.execute(std.testing.allocator, parsed.value.object);
    try std.testing.expect(!result.success);
    try std.testing.expect(std.mem.indexOf(u8, result.error_msg.?, "allowed_domains") != null);
}

// ── Allowlist SSRF bypass tests (Issue #393) ───────────────

test "execute allows allowlisted private IP (fixes #393)" {
    // When a domain is in the allowlist, SSRF protection is skipped,
    // allowing access to private IPs (e.g., local searxng instance)
    const domains = [_][]const u8{"127.0.0.1"};
    var ht = HttpRequestTool{ .allowed_domains = &domains };
    const t = ht.tool();
    const parsed = try root.parseTestArgs("{\"url\": \"https://127.0.0.1:8080/admin\"}");
    defer parsed.deinit();
    const result = try t.execute(std.testing.allocator, parsed.value.object);
    try std.testing.expect(!result.success);
    try std.testing.expectEqualStrings("Network disabled in tests", result.error_msg.?);
}

test "execute rejects non-allowlisted domain before DNS resolution" {
    // Non-allowlisted domains should be rejected immediately without DNS lookup
    const domains = [_][]const u8{"example.com"};
    var ht = HttpRequestTool{ .allowed_domains = &domains };
    const t = ht.tool();
    const parsed = try root.parseTestArgs("{\"url\": \"https://evil.com/path\"}");
    defer parsed.deinit();
    const result = try t.execute(std.testing.allocator, parsed.value.object);
    try std.testing.expect(!result.success);
    try std.testing.expect(std.mem.indexOf(u8, result.error_msg.?, "allowed_domains") != null);
    // Should fail before any DNS resolution (no "Unable to verify host safety" error)
    try std.testing.expect(std.mem.indexOf(u8, result.error_msg.?, "verify host") == null);
}

test "http_request parameters JSON is valid" {
    var ht = HttpRequestTool{};
    const t = ht.tool();
    const schema = t.parametersJson();
    try std.testing.expect(schema[0] == '{');
    try std.testing.expect(std.mem.indexOf(u8, schema, "method") != null);
    try std.testing.expect(std.mem.indexOf(u8, schema, "body") != null);
    try std.testing.expect(std.mem.indexOf(u8, schema, "headers") != null);
}

test "shouldUsePinnedResolve skips allowlisted hostname" {
    try std.testing.expect(!http_util.shouldUsePinnedResolve("searx.internal", "searx.internal"));
}

test "shouldUsePinnedResolve keeps pinning resolved hostname" {
    try std.testing.expect(http_util.shouldUsePinnedResolve("example.com", "93.184.216.34"));
}

test "validateMethod case insensitive" {
    try std.testing.expect(validateMethod("get") != null);
    try std.testing.expect(validateMethod("Post") != null);
    try std.testing.expect(validateMethod("pUt") != null);
    try std.testing.expect(validateMethod("delete") != null);
    try std.testing.expect(validateMethod("patch") != null);
    try std.testing.expect(validateMethod("head") != null);
    try std.testing.expect(validateMethod("options") != null);
}

test "validateMethod rejects empty string" {
    try std.testing.expect(validateMethod("") == null);
}

test "validateMethod rejects CONNECT TRACE" {
    try std.testing.expect(validateMethod("CONNECT") == null);
    try std.testing.expect(validateMethod("TRACE") == null);
}

test "isTlsSetupError detects TLS setup failures" {
    try std.testing.expect(isTlsSetupError(error.TlsInitializationFailed));
    try std.testing.expect(isTlsSetupError(error.CertificateBundleLoadFailure));
    try std.testing.expect(!isTlsSetupError(error.EndOfStream));
}

test "buildHttpRequestErrorMessage includes TLS hint" {
    const msg = try buildHttpRequestErrorMessage(std.testing.allocator, "HTTP request failed", error.TlsInitializationFailed);
    defer std.testing.allocator.free(msg);
    try std.testing.expect(std.mem.indexOf(u8, msg, "system CA certificates") != null);
}

test "parseHeaders basic" {
    const headers = try parseHeaders(std.testing.allocator, "{\"Content-Type\": \"application/json\"}");
    defer {
        for (headers) |h| {
            std.testing.allocator.free(h[0]);
            std.testing.allocator.free(h[1]);
        }
        std.testing.allocator.free(headers);
    }
    try std.testing.expectEqual(@as(usize, 1), headers.len);
    try std.testing.expectEqualStrings("Content-Type", headers[0][0]);
    try std.testing.expectEqualStrings("application/json", headers[0][1]);
}

test "parseHeaders null returns empty" {
    const headers = try parseHeaders(std.testing.allocator, null);
    try std.testing.expectEqual(@as(usize, 0), headers.len);
}
