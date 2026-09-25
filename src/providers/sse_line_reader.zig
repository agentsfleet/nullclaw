//! Bounded line framing for provider Server-Sent Events streams.
//! Each reader belongs to one request thread and borrows the child's stdout.

const std = @import("std");
const compat = @import("compat");

pub const MAX_LINE_BYTES: usize = 2 * 1024 * 1024;

pub const Reader = struct {
    allocator: std.mem.Allocator,
    file: compat.fs.File,
    buffer: [4096]u8 = undefined,
    pos: usize = 0,
    end: usize = 0,
    eof: bool = false,
    line: std.ArrayListUnmanaged(u8) = .empty,
    total_read: usize = 0,

    pub fn init(allocator: std.mem.Allocator, file: compat.fs.File) Reader {
        return .{ .allocator = allocator, .file = file };
    }

    pub fn deinit(self: *Reader) void {
        self.line.deinit(self.allocator);
        self.* = undefined;
    }

    /// The returned line is borrowed until the next `next` call. No newline is included.
    pub fn next(self: *Reader) !?[]const u8 {
        self.line.clearRetainingCapacity();
        while (true) {
            if (self.pos == self.end) {
                if (self.eof) return null;
                self.end = try self.file.read(&self.buffer);
                self.total_read += self.end;
                self.pos = 0;
                if (self.end == 0) {
                    self.eof = true;
                    return if (self.line.items.len > 0) self.line.items else null;
                }
            }

            const pending = self.buffer[self.pos..self.end];
            const newline = std.mem.indexOfScalar(u8, pending, '\n');
            const piece = pending[0 .. newline orelse pending.len];
            if (piece.len > MAX_LINE_BYTES - self.line.items.len) return error.SseLineTooLarge;
            try self.line.appendSlice(self.allocator, piece);
            self.pos += piece.len;
            if (newline != null) {
                self.pos += 1;
                return self.line.items;
            }
        }
    }
};

test "reader frames split lines and the trailing line" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = compat.fs.Dir.wrap(tmp.dir);
    const output = try dir.createFile("frames", .{});
    try output.writeAll("data: first\ndata: second\nlast");
    output.close();

    const input = try dir.openFile("frames", .{});
    defer input.close();
    var reader = Reader.init(std.testing.allocator, input);
    defer reader.deinit();
    try std.testing.expectEqualStrings("data: first", (try reader.next()).?);
    try std.testing.expectEqualStrings("data: second", (try reader.next()).?);
    try std.testing.expectEqualStrings("last", (try reader.next()).?);
    try std.testing.expect((try reader.next()) == null);
}

test "reader rejects an oversized line" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = compat.fs.Dir.wrap(tmp.dir);
    const output = try dir.createFile("large", .{});
    const oversized = try allocator.alloc(u8, MAX_LINE_BYTES + 1);
    defer allocator.free(oversized);
    @memset(oversized, 'x');
    try output.writeAll(oversized);
    output.close();

    const input = try dir.openFile("large", .{});
    defer input.close();
    var reader = Reader.init(allocator, input);
    defer reader.deinit();
    try std.testing.expectError(error.SseLineTooLarge, reader.next());
}
