//! Bounded line framing for provider Server-Sent Events streams.
//! Each reader belongs to one request thread and receives libcurl byte slices.

const std = @import("std");

pub const MAX_LINE_BYTES: usize = 2 * 1024 * 1024;

/// Frames bytes delivered by an in-process HTTP client. The callback can stop
/// immediately after a terminal event without waiting for the server to close.
pub const FeedReader = struct {
    allocator: std.mem.Allocator,
    line: std.ArrayListUnmanaged(u8) = .empty,

    pub fn init(allocator: std.mem.Allocator) FeedReader {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *FeedReader) void {
        self.line.deinit(self.allocator);
    }

    pub fn feed(self: *FeedReader, bytes: []const u8, ctx: *anyopaque, on_line: *const fn (*anyopaque, []const u8) anyerror!bool) !bool {
        var remaining = bytes;
        while (std.mem.indexOfScalar(u8, remaining, '\n')) |newline| {
            try self.append(remaining[0..newline]);
            const keep_reading = try on_line(ctx, self.line.items);
            self.line.clearRetainingCapacity();
            if (!keep_reading) return false;
            remaining = remaining[newline + 1 ..];
        }
        try self.append(remaining);
        return true;
    }

    pub fn finish(self: *FeedReader, ctx: *anyopaque, on_line: *const fn (*anyopaque, []const u8) anyerror!bool) !bool {
        if (self.line.items.len > 0) {
            const keep_reading = try on_line(ctx, self.line.items);
            self.line.clearRetainingCapacity();
            return keep_reading;
        }
        return true;
    }

    fn append(self: *FeedReader, bytes: []const u8) !void {
        if (bytes.len > MAX_LINE_BYTES - self.line.items.len) return error.SseLineTooLarge;
        try self.line.appendSlice(self.allocator, bytes);
    }
};

test "feed reader frames split chunks and stops at terminal event" {
    const Collector = struct {
        count: usize = 0,
        fn onLine(ctx: *anyopaque, line: []const u8) !bool {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            if (self.count == 0) try std.testing.expectEqualStrings("data: one", line);
            if (self.count == 1) try std.testing.expectEqualStrings("data: [DONE]", line);
            self.count += 1;
            return self.count < 2;
        }
    };
    var reader = FeedReader.init(std.testing.allocator);
    defer reader.deinit();
    var collector = Collector{};
    try std.testing.expect(try reader.feed("data: o", &collector, Collector.onLine));
    try std.testing.expect(!(try reader.feed("ne\ndata: [DONE]\nignored", &collector, Collector.onLine)));
    try std.testing.expectEqual(@as(usize, 2), collector.count);
}

test "feed reader reports terminal event in a trailing line" {
    const Terminal = struct {
        fn onLine(_: *anyopaque, line: []const u8) !bool {
            try std.testing.expectEqualStrings("data: [DONE]", line);
            return false;
        }
    };
    var reader = FeedReader.init(std.testing.allocator);
    defer reader.deinit();
    var marker: u8 = 0;
    try std.testing.expect(try reader.feed("data: [DONE]", &marker, Terminal.onLine));
    try std.testing.expect(!(try reader.finish(&marker, Terminal.onLine)));
}

test "feed reader rejects an oversized frame without retaining extra bytes" {
    var reader = FeedReader.init(std.testing.allocator);
    defer reader.deinit();
    const oversized = try std.testing.allocator.alloc(u8, MAX_LINE_BYTES + 1);
    defer std.testing.allocator.free(oversized);
    @memset(oversized, 'x');
    var marker: u8 = 0;
    const Never = struct {
        fn onLine(_: *anyopaque, _: []const u8) !bool {
            return error.UnexpectedLine;
        }
    };
    try std.testing.expectError(error.SseLineTooLarge, reader.feed(oversized, &marker, Never.onLine));
    try std.testing.expectEqual(@as(usize, 0), reader.line.items.len);
}
