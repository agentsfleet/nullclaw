//! OpenAI-compatible streaming tool-call fragments. JSON decoding is delegated
//! to std.json; this module only assembles the indexed fields across frames.

const std = @import("std");
const root = @import("root.zig");

const MAX_TOOL_CALLS = 16;
const MAX_TOOL_ID_BYTES = 256;
const MAX_TOOL_NAME_BYTES = 256;
const MAX_TOOL_ARGUMENT_BYTES = 256 * 1024;
const MAX_TOTAL_TOOL_ARGUMENT_BYTES = 1024 * 1024;

const Part = struct {
    id: std.ArrayListUnmanaged(u8) = .empty,
    name: std.ArrayListUnmanaged(u8) = .empty,
    arguments: std.ArrayListUnmanaged(u8) = .empty,

    fn deinit(self: *Part, allocator: std.mem.Allocator) void {
        self.id.deinit(allocator);
        self.name.deinit(allocator);
        self.arguments.deinit(allocator);
    }
};

pub const Collector = struct {
    parts: [MAX_TOOL_CALLS]Part = undefined,
    count: usize = 0,
    total_argument_bytes: usize = 0,
    fragments: u32 = 0,
    finish_reason: root.StreamFinishReason = .unknown,

    pub fn deinit(self: *Collector, allocator: std.mem.Allocator) void {
        for (self.parts[0..self.count]) |*part| part.deinit(allocator);
        self.count = 0;
        self.total_argument_bytes = 0;
    }

    /// Consume only metadata-bearing Server-Sent Event lines. Content-only
    /// frames stay on the existing text path and incur no extra JSON parse.
    pub fn feedLine(self: *Collector, allocator: std.mem.Allocator, line: []const u8) !void {
        const trimmed = std.mem.trimEnd(u8, line, "\r");
        if (!std.mem.startsWith(u8, trimmed, "data:")) return;
        const data = std.mem.trimStart(u8, trimmed[5..], " ");
        if (std.mem.indexOf(u8, data, "\"tool_calls\"") == null and
            std.mem.indexOf(u8, data, "\"finish_reason\"") == null) return;

        const parsed = std.json.parseFromSlice(std.json.Value, allocator, data, .{}) catch
            return error.InvalidStreamMetadata;
        defer parsed.deinit();
        try self.feedValue(allocator, parsed.value);
    }

    /// Consume metadata from the JSON tree already decoded for text and usage.
    pub fn feedValue(self: *Collector, allocator: std.mem.Allocator, value: std.json.Value) !void {
        if (value != .object) return error.InvalidStreamMetadata;
        const choices = value.object.get("choices") orelse return;
        if (choices != .array or choices.array.items.len == 0) return;
        const first = choices.array.items[0];
        if (first != .object) return error.InvalidStreamMetadata;

        if (first.object.get("finish_reason")) |reason| {
            if (reason == .string) self.finish_reason = parseFinishReason(reason.string);
        }

        const delta = first.object.get("delta") orelse return;
        if (delta != .object) return error.InvalidStreamMetadata;
        const calls = delta.object.get("tool_calls") orelse return;
        if (calls != .array) return error.InvalidStreamMetadata;
        for (calls.array.items) |call| try self.feedCall(allocator, call);
    }

    fn feedCall(self: *Collector, allocator: std.mem.Allocator, call: std.json.Value) !void {
        if (call != .object) return error.InvalidStreamToolCall;
        const index = call.object.get("index") orelse return error.InvalidStreamToolCall;
        if (index != .integer or index.integer < 0 or index.integer >= MAX_TOOL_CALLS)
            return error.InvalidStreamToolCall;
        const at: usize = @intCast(index.integer);
        while (self.count <= at) : (self.count += 1) self.parts[self.count] = .{};
        const part = &self.parts[at];
        self.fragments +|= 1;

        if (call.object.get("type")) |kind| {
            if (kind != .string or !std.mem.eql(u8, kind.string, "function"))
                return error.InvalidStreamToolCall;
        }
        if (call.object.get("id")) |id| {
            if (id != .string) return error.InvalidStreamToolCall;
            try appendBounded(&part.id, allocator, id.string, MAX_TOOL_ID_BYTES);
        }
        if (call.object.get("function")) |function| {
            if (function != .object) return error.InvalidStreamToolCall;
            if (function.object.get("name")) |name| {
                if (name != .string) return error.InvalidStreamToolCall;
                try appendBounded(&part.name, allocator, name.string, MAX_TOOL_NAME_BYTES);
            }
            if (function.object.get("arguments")) |arguments| {
                if (arguments != .string) return error.InvalidStreamToolCall;
                try self.appendArguments(part, allocator, arguments.string);
            }
        }
    }

    fn appendArguments(self: *Collector, part: *Part, allocator: std.mem.Allocator, fragment: []const u8) !void {
        if (fragment.len > MAX_TOTAL_TOOL_ARGUMENT_BYTES - self.total_argument_bytes)
            return error.StreamToolFieldTooLarge;
        try appendBounded(&part.arguments, allocator, fragment, MAX_TOOL_ARGUMENT_BYTES);
        self.total_argument_bytes += fragment.len;
    }

    /// Transfer complete calls to the caller. A missing field rejects the
    /// whole batch, so no partial action reaches the agent dispatcher.
    pub fn take(self: *Collector, allocator: std.mem.Allocator) ![]const root.ToolCall {
        if (self.count == 0) return &.{};
        for (self.parts[0..self.count]) |part| {
            if (part.id.items.len == 0 or part.name.items.len == 0 or part.arguments.items.len == 0)
                return error.IncompleteStreamToolCall;
            const parsed = std.json.parseFromSlice(std.json.Value, allocator, part.arguments.items, .{}) catch |err|
                return if (err == error.OutOfMemory) err else error.InvalidStreamToolArguments;
            defer parsed.deinit();
            if (parsed.value != .object) return error.InvalidStreamToolArguments;
        }

        const calls = try allocator.alloc(root.ToolCall, self.count);
        var transferred: usize = 0;
        errdefer {
            for (calls[0..transferred]) |call| {
                allocator.free(call.id);
                allocator.free(call.name);
                allocator.free(call.arguments);
            }
            allocator.free(calls);
        }
        for (self.parts[0..self.count], 0..) |*part, i| {
            const id = try part.id.toOwnedSlice(allocator);
            errdefer allocator.free(id);
            const name = try part.name.toOwnedSlice(allocator);
            errdefer allocator.free(name);
            const arguments = try part.arguments.toOwnedSlice(allocator);
            calls[i] = .{ .id = id, .name = name, .arguments = arguments };
            transferred += 1;
        }
        self.count = 0;
        self.total_argument_bytes = 0;
        return calls;
    }
};

fn appendBounded(list: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator, fragment: []const u8, limit: usize) !void {
    if (fragment.len > limit - list.items.len) return error.StreamToolFieldTooLarge;
    try list.appendSlice(allocator, fragment);
}

fn parseFinishReason(reason: []const u8) root.StreamFinishReason {
    if (std.mem.eql(u8, reason, "stop")) return .stop;
    if (std.mem.eql(u8, reason, "tool_calls")) return .tool_calls;
    if (std.mem.eql(u8, reason, "length")) return .length;
    if (std.mem.eql(u8, reason, "content_filter")) return .content_filter;
    return .other;
}

test "fragmented indexed tool calls become complete owned calls" {
    const alloc = std.testing.allocator;
    var collector = Collector{};
    defer collector.deinit(alloc);
    try collector.feedLine(alloc, "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_\",\"type\":\"function\",\"function\":{\"name\":\"memory_\",\"arguments\":\"{\\\"key\\\":\"}}]}}]}");
    try collector.feedLine(alloc, "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"1\",\"function\":{\"name\":\"store\",\"arguments\":\"\\\"lantern\\\"}\"}}]}}]}");
    try collector.feedLine(alloc, "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}");
    const calls = try collector.take(alloc);
    defer {
        for (calls) |call| {
            alloc.free(call.id);
            alloc.free(call.name);
            alloc.free(call.arguments);
        }
        alloc.free(calls);
    }
    try std.testing.expectEqual(@as(usize, 1), calls.len);
    try std.testing.expectEqualStrings("call_1", calls[0].id);
    try std.testing.expectEqualStrings("memory_store", calls[0].name);
    try std.testing.expectEqualStrings("{\"key\":\"lantern\"}", calls[0].arguments);
    try std.testing.expectEqual(root.StreamFinishReason.tool_calls, collector.finish_reason);
    try std.testing.expectEqual(@as(u32, 2), collector.fragments);
}

test "incomplete or oversized streamed tool calls are rejected" {
    const alloc = std.testing.allocator;
    var collector = Collector{};
    defer collector.deinit(alloc);
    try std.testing.expectError(error.InvalidStreamToolCall, collector.feedLine(alloc, "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":16}]}}]}"));
    try collector.feedLine(alloc, "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\"}]}}]}");
    try std.testing.expectError(error.IncompleteStreamToolCall, collector.take(alloc));
    const too_long = try alloc.alloc(u8, MAX_TOOL_NAME_BYTES + 1);
    defer alloc.free(too_long);
    @memset(too_long, 'x');
    try std.testing.expectError(error.StreamToolFieldTooLarge, appendBounded(&collector.parts[0].name, alloc, too_long, MAX_TOOL_NAME_BYTES));
}

test "invalid streamed arguments cannot reach tool dispatch" {
    const alloc = std.testing.allocator;
    var collector = Collector{};
    defer collector.deinit(alloc);
    try collector.feedLine(alloc, "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"function\":{\"name\":\"probe\",\"arguments\":\"{\\\"x\\\":\"}}]}}]}");
    try std.testing.expectError(error.InvalidStreamToolArguments, collector.take(alloc));
}

test "streamed tool argument budget rejects one large call and aggregate overflow" {
    const allocator = std.testing.allocator;
    var collector = Collector{};
    defer collector.deinit(allocator);
    collector.parts[0] = .{};
    collector.count = 1;
    const allowed = try allocator.alloc(u8, MAX_TOOL_ARGUMENT_BYTES);
    defer allocator.free(allowed);
    @memset(allowed, 'x');
    try collector.appendArguments(&collector.parts[0], allocator, allowed);
    try std.testing.expectError(error.StreamToolFieldTooLarge, collector.appendArguments(&collector.parts[0], allocator, "x"));
    collector.parts[1] = .{};
    collector.count = 2;
    collector.total_argument_bytes = MAX_TOTAL_TOOL_ARGUMENT_BYTES;
    try std.testing.expectError(error.StreamToolFieldTooLarge, collector.appendArguments(&collector.parts[1], allocator, "x"));
}
