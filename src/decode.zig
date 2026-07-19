//! Typed decoding from `Value` into native Zig types.
//!
//! Maps a parsed JSON `Value` tree onto a target struct via comptime
//! reflection, the way `serde::Deserialize` does in Rust. Strings and slices
//! are zero-copy where possible; everything else lives in the caller's arena.
//!
//! ```zig
//! const Config = struct {
//!     title: []const u8,
//!     port: u16 = 8080,
//!     tags: []const []const u8,
//!     server: struct {
//!         host: []const u8,
//!         tls: bool = false,
//!     },
//! };
//!
//! const cfg = try json.parseInto(Config, arena, src, .{});
//! ```
//!
//! Field defaults satisfy missing-field cases. Optional fields (`?T`) become
//! `null` when absent or explicitly `null`. Unknown JSON keys are an error by
//! default; opt out with `ParseOptions{ .ignore_unknown_fields = true }`.

const std = @import("std");
const Allocator = std.mem.Allocator;
const ArenaAllocator = std.heap.ArenaAllocator;
const testing = std.testing;
const value_mod = @import("value.zig");
const Value = value_mod.Value;
const parser_mod = @import("parser.zig");
const lev = @import("levenshtein.zig");

pub const DecodeError = error{
    TypeMismatch,
    MissingField,
    UnknownField,
    InvalidEnumValue,
    Overflow,
    OutOfMemory,
};

/// Decode diagnostics have no source location (the value tree carries none),
/// so every entry gets a zero span and the dotted path is folded into the
/// message text instead.
const no_span: value_mod.Span = .{ .start = 0, .end = 0 };

fn appendDiag(list: *std.ArrayList(parser_mod.Diagnostic), arena: Allocator, path: *const PathBuilder, msg: []const u8, suggestion: ?[]const u8) Allocator.Error!void {
    const full = if (path.slice().len > 0)
        try std.fmt.allocPrint(arena, "{s} (at {s})", .{ msg, path.slice() })
    else
        msg;
    try list.append(arena, .{ .message = full, .span = no_span, .suggestion = suggestion });
}

const PathBuilder = struct {
    buf: std.ArrayList(u8),

    pub fn pushSegment(self: *PathBuilder, arena: Allocator, segment: []const u8) Allocator.Error!usize {
        const prev = self.buf.items.len;
        if (prev > 0) try self.buf.append(arena, '.');
        try self.buf.appendSlice(arena, segment);
        return prev;
    }

    pub fn pushIndex(self: *PathBuilder, arena: Allocator, idx: usize) Allocator.Error!usize {
        const prev = self.buf.items.len;
        var tmp: [24]u8 = undefined;
        // [24]u8 fits '[' + max u64 decimal (20 digits) + ']' + NUL -- always in range.
        const s = std.fmt.bufPrint(&tmp, "[{d}]", .{idx}) catch unreachable;
        try self.buf.appendSlice(arena, s);
        return prev;
    }

    pub fn restore(self: *PathBuilder, prev_len: usize) void {
        self.buf.shrinkRetainingCapacity(prev_len);
    }

    pub fn slice(self: *const PathBuilder) []const u8 {
        return self.buf.items;
    }
};

/// Comptime check that every annotation entry on `T` names a real field
/// (struct) or variant (union): `json_rename` keys, `json_skip` entries,
/// and `json_flatten` entries. A typo'd annotation fails the build with
/// `@compileError` instead of silently never applying. Runs at the top of
/// struct and tagged-union decoding (and typed encoding). Compile errors
/// cannot be asserted from the test suite.
pub fn validateAnnotations(comptime T: type) void {
    comptime {
        const kind = if (@typeInfo(T) == .@"union") "variant" else "field";
        if (@hasDecl(T, "json_rename")) {
            for (@typeInfo(@TypeOf(T.json_rename)).@"struct".fields) |rf| {
                if (!@hasField(T, rf.name)) {
                    @compileError("json_rename entry `" ++ rf.name ++ "` does not match any " ++ kind ++ " of " ++ @typeName(T));
                }
            }
        }
        if (@hasDecl(T, "json_skip")) {
            for (T.json_skip) |name| {
                if (!@hasField(T, name)) {
                    @compileError("json_skip entry `" ++ name ++ "` does not match any " ++ kind ++ " of " ++ @typeName(T));
                }
            }
        }
        if (@hasDecl(T, "json_flatten")) {
            for (T.json_flatten) |name| {
                if (!@hasField(T, name)) {
                    @compileError("json_flatten entry `" ++ name ++ "` does not match any " ++ kind ++ " of " ++ @typeName(T));
                }
            }
        }
    }
}

/// Returns the effective JSON key for `field_name` on type `T`,
/// consulting `T.json_rename` if present.
pub fn renamedKey(comptime T: type, comptime TAnnotation: type, comptime field_name: []const u8) []const u8 {
    const renames = blk: {
        if (TAnnotation.getOrEmpty(T)) |annotation| {
            if (annotation.json_rename) |renames| break :blk renames;
        }
        if (@hasDecl(T, "json_rename")) break :blk T.json_rename;
        break :blk null;
    };

    if (renames) |r| {
        if (@hasField(@TypeOf(r), field_name)) {
            return @field(r, field_name);
        }
    }
    return field_name;
}

/// Returns true if `field_name` on type `T` is listed in `T.json_skip`.
pub fn isSkipped(comptime T: type, comptime TAnnotation: type, comptime field_name: []const u8) bool {
    const skip = if (TAnnotation.getOrEmpty(T)) |annotation| {
        if (annotation.json_skip) |skips| {
            skips;
        }
    } else if (@hasDecl(T, "json_skip"))
        T.json_skip
    else
        return false;

    inline for (skip) |name| {
        if (comptime std.mem.eql(u8, name, field_name)) return true;
    }
    return false;
}

/// Returns true if `field_name` on type `T` is listed in `T.json_flatten`.
pub fn isFlattened(comptime T: type, comptime TAnnotation: type, comptime field_name: []const u8) bool {
    const flat = if (TAnnotation.getOrEmpty(T)) |annotation| {
        if (annotation.json_flatten) |flatten| {
            flatten;
        }
    } else if (@hasDecl(T, "json_flatten"))
        T.json_flatten
    else
        return false;

    inline for (flat) |name| {
        if (comptime std.mem.eql(u8, name, field_name)) return true;
    }
    return false;
}

/// Returns the full set of JSON keys that decoding `T` expects to see
/// at the object's level -- i.e., renamed names for non-flattened fields,
/// plus the expectedKeys of each flattened field's type (recursive).
fn expectedKeys(comptime T: type, comptime TAnnotation: type) []const []const u8 {
    comptime {
        const s = @typeInfo(T).@"struct";
        var keys: []const []const u8 = &.{};
        for (s.fields) |field| {
            if (isSkipped(T, TAnnotation, field.name)) continue;
            if (isFlattened(T, TAnnotation, field.name)) {
                const inner = expectedKeys(field.type, TAnnotation);
                keys = keys ++ inner;
            } else {
                keys = keys ++ &[_][]const u8{renamedKey(T, TAnnotation, field.name)};
            }
        }
        return keys;
    }
}

/// Decode a `Value` into an instance of `T`.
///
/// Number policy: float targets accept `.integer` values (converted via
/// `@floatFromInt`), but integer targets do NOT accept `.float` values --
/// `1e2` parses as `.float` and stays one, so it never decodes into an
/// integer field. In typed mode, integer literals in the range
/// [minInt(i128), maxInt(i128)] parse as `.integer` and decode into any
/// integer target that fits (via overflow-checked cast). For u128 or values
/// beyond i128 range, use `number_mode = .raw` so the lexeme is preserved
/// and decoded directly into the target. JSON `null` decodes only into
/// optional targets; for any other target it errors like an absent field
/// (`error.MissingField`).
pub fn decode(comptime T: type, comptime TAnnotation: type, arena: Allocator, value: Value, options: parser_mod.ParseOptions) DecodeError!T {
    var path: PathBuilder = .{ .buf = .empty };
    return decodeInner(T, TAnnotation, arena, value, options, &path);
}

/// Parse + decode in one call. See `decode` for the decoding rules.
///
/// Fast path: types without `Value` fields, `fromJson` hooks, or tagged
/// unions decode in a single streaming pass with no intermediate `Value`
/// tree. On any error the input is re-decoded through the tree path, so
/// diagnostics and error selection are always the canonical ones. Callers
/// requesting `options.spans` use the tree path unconditionally.
pub fn parseInto(comptime T: type, comptime TAnnotation: type, arena: Allocator, src: []const u8, options: parser_mod.ParseOptions) (parser_mod.Error || DecodeError)!T {
    if (comptime needsTree(T, TAnnotation)) return parseIntoTree(T, TAnnotation, arena, src, options);
    if (options.spans != null) return parseIntoTree(T, TAnnotation, arena, src, options);
    return streamParseInto(T, TAnnotation, arena, src, options) catch |err| switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        else => parseIntoTree(T, TAnnotation, arena, src, options),
    };
}

fn parseIntoTree(comptime T: type, comptime TAnnotation: type, arena: Allocator, src: []const u8, options: parser_mod.ParseOptions) (parser_mod.Error || DecodeError)!T {
    const value = try parser_mod.parse(arena, src, options);
    return decode(T, TAnnotation, arena, value, options);
}

/// Reader-input variant of `parseInto`: drains the reader into arena
/// memory, then decodes the slice (streaming when the type allows).
pub fn parseIntoReader(comptime T: type, arena: Allocator, reader: *std.Io.Reader, options: parser_mod.ParseOptions) (parser_mod.ReaderError || DecodeError)!T {
    const input = try reader.allocRemaining(arena, .unlimited);
    return parseInto(T, arena, input, options);
}

fn decodeInner(comptime T: type, comptime TAnnotation: type, arena: Allocator, value: Value, options: parser_mod.ParseOptions, path: *PathBuilder) DecodeError!T {
    if (T == Value) return value;

    // Custom fromJson hook short-circuit.
    if (comptime (@typeInfo(T) == .@"struct")) {
        if (@hasDecl(T, "fromJson")) {
            comptime {
                const fn_info = @typeInfo(@TypeOf(T.fromJson)).@"fn";
                if (fn_info.params.len != 3) {
                    @compileError(@typeName(T) ++ ".fromJson must take exactly 3 params: (Allocator, Value, ParseOptions)");
                }
            }
            return T.fromJson(arena, value, options);
        }

        if (comptime (TAnnotation.getOrEmpty(T))) |provider| {
            if (provider.fromJson) |fromJson| {
                return fromJson(arena, value, options);
            }
        }
    }

    // JSON `null` satisfies optionals only (handled in decodeOptional). For
    // any other target the field is effectively absent, so the error matches
    // the missing-field case.
    if (comptime @typeInfo(T) != .optional) {
        if (value == .null) {
            if (options.errors) |list| {
                const msg = try std.fmt.allocPrint(arena, "expected {s}, got null", .{@typeName(T)});
                try appendDiag(list, arena, path, msg, null);
            }
            return error.MissingField;
        }
    }

    // Tagged-union dispatch.
    if (comptime (@typeInfo(T) == .@"union" and (@hasDecl(T, "json_tag") or (TAnnotation.has(T) and TAnnotation.get(T).json_tag != null)))) {
        return decodeTaggedUnion(T, TAnnotation, arena, value, options, path);
    }

    return switch (@typeInfo(T)) {
        .bool => decodeBool(value, arena, options, path),
        .int => decodeInt(T, value, arena, options, path),
        .float => decodeFloat(T, value, arena, options, path),
        .pointer => |p| decodePointer(T, TAnnotation, p, arena, value, options, path),
        .array => |a| decodeArray(T, TAnnotation, a, arena, value, options, path),
        .optional => |o| decodeOptional(o.child, TAnnotation, arena, value, options, path),
        .@"struct" => |s| decodeStruct(T, TAnnotation, s, arena, value, options, path),
        .@"enum" => decodeEnum(T, value, arena, options, path),
        else => @compileError("json decode: unsupported type " ++ @typeName(T)),
    };
}

fn decodeBool(value: Value, arena: Allocator, options: parser_mod.ParseOptions, path: *PathBuilder) DecodeError!bool {
    if (value != .bool) {
        if (options.errors) |list| {
            const msg = try std.fmt.allocPrint(arena, "expected boolean, got {s}", .{@tagName(value)});
            try appendDiag(list, arena, path, msg, null);
        }
        return error.TypeMismatch;
    }
    return value.bool;
}

fn decodeInt(comptime T: type, value: Value, arena: Allocator, options: parser_mod.ParseOptions, path: *PathBuilder) DecodeError!T {
    // Raw-mode lexeme: parse directly into T so wide targets (u64, i128,
    // u128) are not bottlenecked by an i128 intermediate. Float-syntax
    // lexemes fail parseInt (InvalidCharacter) and become TypeMismatch,
    // matching the typed-mode policy of never coercing floats to ints.
    switch (value) {
        .integer => |n| {
            if (std.math.cast(T, n)) |v| return v;
            if (options.errors) |list| {
                const msg = try std.fmt.allocPrint(arena, "integer {d} out of range for {s}", .{ n, @typeName(T) });
                try appendDiag(list, arena, path, msg, null);
            }
            return error.Overflow;
        },
        .number_raw => |raw| {
            return std.fmt.parseInt(T, raw, 10) catch |err| {
                if (options.errors) |list| {
                    const msg = switch (err) {
                        error.Overflow => try std.fmt.allocPrint(arena, "integer {s} out of range for {s}", .{ raw, @typeName(T) }),
                        error.InvalidCharacter => try std.fmt.allocPrint(arena, "expected integer, got number {s}", .{raw}),
                    };
                    try appendDiag(list, arena, path, msg, null);
                }
                return switch (err) {
                    error.Overflow => error.Overflow,
                    error.InvalidCharacter => error.TypeMismatch,
                };
            };
        },
        else => {
            if (options.errors) |list| {
                const msg = try std.fmt.allocPrint(arena, "expected integer, got {s}", .{@tagName(value)});
                try appendDiag(list, arena, path, msg, null);
            }
            return error.TypeMismatch;
        },
    }
}

fn decodeFloat(comptime T: type, value: Value, arena: Allocator, options: parser_mod.ParseOptions, path: *PathBuilder) DecodeError!T {
    return switch (value) {
        .float => |f| blk: {
            const result: T = @floatCast(f);
            // A finite f64 that overflows the narrower target is a caller error,
            // not a lossless cast. A genuine inf/nan source passes through.
            if (!std.math.isInf(f) and std.math.isInf(result)) return error.Overflow;
            break :blk result;
        },
        .integer => |n| blk: {
            const result: T = @floatFromInt(n);
            if (std.math.isInf(result)) return error.Overflow;
            break :blk result;
        },
        .number_raw => |raw| blk: {
            const f = std.fmt.parseFloat(f64, raw) catch {
                if (options.errors) |list| {
                    const msg = try std.fmt.allocPrint(arena, "expected float, got number {s}", .{raw});
                    try appendDiag(list, arena, path, msg, null);
                }
                return error.TypeMismatch;
            };
            const result: T = @floatCast(f);
            if (!std.math.isInf(f) and std.math.isInf(result)) return error.Overflow;
            break :blk result;
        },
        else => {
            if (options.errors) |list| {
                const msg = try std.fmt.allocPrint(arena, "expected float, got {s}", .{@tagName(value)});
                try appendDiag(list, arena, path, msg, null);
            }
            return error.TypeMismatch;
        },
    };
}

fn decodePointer(comptime T: type, comptime TAnnotation: type, comptime p: std.builtin.Type.Pointer, arena: Allocator, value: Value, options: parser_mod.ParseOptions, path: *PathBuilder) DecodeError!T {
    if (p.size != .slice) @compileError("json decode: only slice pointers supported, got " ++ @typeName(T));
    if (p.child == u8 and p.is_const) {
        if (value != .string) {
            if (options.errors) |list| {
                const msg = try std.fmt.allocPrint(arena, "expected string, got {s}", .{@tagName(value)});
                try appendDiag(list, arena, path, msg, null);
            }
            return error.TypeMismatch;
        }
        return value.string;
    }
    if (value != .array) {
        if (options.errors) |list| {
            const msg = try std.fmt.allocPrint(arena, "expected array, got {s}", .{@tagName(value)});
            try appendDiag(list, arena, path, msg, null);
        }
        return error.TypeMismatch;
    }
    const items = value.array;
    const out = try arena.alloc(p.child, items.len);
    for (items, 0..) |item, i| {
        const prev = try path.pushIndex(arena, i);
        defer path.restore(prev);
        out[i] = try decodeInner(p.child, TAnnotation, arena, item, options, path);
    }
    return out;
}

fn decodeArray(comptime T: type, comptime TAnnotation: type, comptime a: std.builtin.Type.Array, arena: Allocator, value: Value, options: parser_mod.ParseOptions, path: *PathBuilder) DecodeError!T {
    if (value != .array) {
        if (options.errors) |list| {
            const msg = try std.fmt.allocPrint(arena, "expected array, got {s}", .{@tagName(value)});
            try appendDiag(list, arena, path, msg, null);
        }
        return error.TypeMismatch;
    }
    if (value.array.len != a.len) {
        if (options.errors) |list| {
            const msg = try std.fmt.allocPrint(arena, "array length mismatch: expected {d}, got {d}", .{ a.len, value.array.len });
            try appendDiag(list, arena, path, msg, null);
        }
        return error.TypeMismatch;
    }
    var out: T = undefined;
    // Zig rejects indexing an empty array even in unreachable loop bodies,
    // so skip the loop entirely for zero-length array types.
    if (comptime a.len > 0) {
        for (value.array, 0..) |item, i| {
            const prev = try path.pushIndex(arena, i);
            defer path.restore(prev);
            out[i] = try decodeInner(a.child, TAnnotation, arena, item, options, path);
        }
    }
    return out;
}

fn decodeOptional(comptime Child: type, comptime TAnnotation: type, arena: Allocator, value: Value, options: parser_mod.ParseOptions, path: *PathBuilder) DecodeError!?Child {
    if (value == .null) return null;
    return try decodeInner(Child, TAnnotation, arena, value, options, path);
}

fn decodeStruct(comptime T: type, comptime TAnnotation: type, comptime s: std.builtin.Type.Struct, arena: Allocator, value: Value, options: parser_mod.ParseOptions, path: *PathBuilder) DecodeError!T {
    comptime validateAnnotations(T);
    if (value != .object) {
        if (options.errors) |list| {
            const msg = try std.fmt.allocPrint(arena, "expected object, got {s}", .{@tagName(value)});
            try appendDiag(list, arena, path, msg, null);
        }
        return error.TypeMismatch;
    }
    const obj = value.object;

    // Unknown-field check runs before field assignment so that an
    // unrecognized key is reported as UnknownField rather than being
    // shadowed by a subsequent MissingField on a required field.
    if (!options.ignore_unknown_fields) {
        var it = obj.iterator();
        outer: while (it.next()) |entry| {
            inline for (comptime expectedKeys(T, TAnnotation)) |expected| {
                if (std.mem.eql(u8, entry.key_ptr.*, expected)) continue :outer;
            }
            // Unknown key. Try a suggestion.
            const key = entry.key_ptr.*;
            const suggestion = lev.closestMatch(key, comptime expectedKeys(T, TAnnotation), lev.suggestionThreshold(key.len));

            if (options.errors) |list| {
                const msg = try std.fmt.allocPrint(arena, "unknown field `{s}`", .{key});
                const suggestion_owned: ?[]const u8 = if (suggestion) |s_str| try arena.dupe(u8, s_str) else null;
                try appendDiag(list, arena, path, msg, suggestion_owned);
            }
            return error.UnknownField;
        }
    }

    var out: T = undefined;

    inline for (s.fields) |field| {
        if (comptime isSkipped(T, TAnnotation, field.name)) {
            const dv = comptime field.defaultValue() orelse
                @compileError("json_skip field `" ++ field.name ++ "` on " ++ @typeName(T) ++ " has no default value");
            @field(out, field.name) = dv;
        } else if (comptime isFlattened(T, TAnnotation, field.name)) {
            // Decode the inner struct from the SAME parent value (no key lookup).
            // The parent's expectedKeys already validated all keys, so suppress
            // unknown-field errors in the inner struct to avoid false positives
            // on sibling fields the inner type doesn't know about.
            const prev = try path.pushSegment(arena, field.name);
            defer path.restore(prev);
            var flat_opts = options;
            flat_opts.ignore_unknown_fields = true;
            @field(out, field.name) = try decodeInner(field.type, TAnnotation, arena, value, flat_opts, path);
        } else {
            const eff_key = comptime renamedKey(T, TAnnotation, field.name);
            if (obj.get(eff_key)) |fv| {
                const prev = try path.pushSegment(arena, eff_key);
                defer path.restore(prev);
                @field(out, field.name) = try decodeInner(field.type, TAnnotation, arena, fv, options, path);
            } else if (field.defaultValue()) |dv| {
                @field(out, field.name) = dv;
            } else if (@typeInfo(field.type) == .optional) {
                @field(out, field.name) = null;
            } else {
                if (options.errors) |list| {
                    const msg = try std.fmt.allocPrint(arena, "missing required field `{s}`", .{eff_key});
                    try appendDiag(list, arena, path, msg, null);
                }
                return error.MissingField;
            }
        }
    }

    return out;
}

/// Effective (renamed) wire names of every variant of union `T`.
fn variantNames(comptime T: type, comptime TAnnotation: type) []const []const u8 {
    comptime {
        var names: []const []const u8 = &.{};
        for (@typeInfo(T).@"union".fields) |field| {
            names = names ++ &[_][]const u8{renamedKey(T, TAnnotation, field.name)};
        }
        return names;
    }
}

fn decodeTaggedUnion(comptime T: type, comptime TAnnotation: type, arena: Allocator, value: Value, options: parser_mod.ParseOptions, path: *PathBuilder) DecodeError!T {
    comptime validateAnnotations(T);
    if (value != .object) {
        if (options.errors) |list| {
            const msg = try std.fmt.allocPrint(arena, "expected object for {s}, got {s}", .{ @typeName(T), @tagName(value) });
            try appendDiag(list, arena, path, msg, null);
        }
        return error.TypeMismatch;
    }
    const obj = value.object;

    const tag_field = if (TAnnotation.getOrEmpty(T)) |annotation| block: {
        if (annotation.json_tag) |json_tag| {
            break :block json_tag;
        }
    } else T.json_tag;

    const tag_value = obj.get(tag_field) orelse {
        if (options.errors) |list| {
            const msg = try std.fmt.allocPrint(arena, "missing discriminator field `{s}` for {s}", .{ tag_field, @typeName(T) });
            try appendDiag(list, arena, path, msg, null);
        }
        return error.MissingField;
    };
    if (tag_value != .string) {
        if (options.errors) |list| {
            const msg = try std.fmt.allocPrint(arena, "expected string for discriminator `{s}`, got {s}", .{ tag_field, @tagName(tag_value) });
            try appendDiag(list, arena, path, msg, null);
        }
        return error.TypeMismatch;
    }

    inline for (@typeInfo(T).@"union".fields) |union_field| {
        const variant_name = union_field.name;
        const effective_name = comptime renamedKey(T, TAnnotation, variant_name);
        if (std.mem.eql(u8, tag_value.string, effective_name)) {
            const PayloadType = union_field.type;

            if (PayloadType == void) {
                return @unionInit(T, variant_name, {});
            }

            // Build a filtered object view that drops the discriminator field.
            var filtered: value_mod.ObjectMap = .empty;
            var it = obj.iterator();
            while (it.next()) |entry| {
                if (std.mem.eql(u8, entry.key_ptr.*, tag_field)) continue;
                const key_dup = try arena.dupe(u8, entry.key_ptr.*);
                try filtered.put(arena, key_dup, entry.value_ptr.*);
            }
            const filtered_value = Value{ .object = filtered };
            const payload = try decodeInner(PayloadType, TAnnotation, arena, filtered_value, options, path);
            return @unionInit(T, variant_name, payload);
        }
    }
    if (options.errors) |list| {
        const tag = tag_value.string;
        const suggestion = lev.closestMatch(tag, comptime variantNames(T, TAnnotation), lev.suggestionThreshold(tag.len));
        const msg = try std.fmt.allocPrint(arena, "unknown variant `{s}` for {s}", .{ tag, @typeName(T) });
        const suggestion_owned: ?[]const u8 = if (suggestion) |s_str| try arena.dupe(u8, s_str) else null;
        try appendDiag(list, arena, path, msg, suggestion_owned);
    }
    return error.InvalidEnumValue;
}

fn decodeEnum(comptime T: type, value: Value, arena: Allocator, options: parser_mod.ParseOptions, path: *PathBuilder) DecodeError!T {
    switch (value) {
        .string => |s| {
            if (std.meta.stringToEnum(T, s)) |v| return v;
            if (options.errors) |list| {
                const msg = try std.fmt.allocPrint(arena, "invalid enum value `{s}` for {s}", .{ s, @typeName(T) });
                try appendDiag(list, arena, path, msg, null);
            }
            return error.InvalidEnumValue;
        },
        .integer => |n| {
            if (std.enums.fromInt(T, n)) |v| return v;
            if (options.errors) |list| {
                const msg = try std.fmt.allocPrint(arena, "integer {d} is not a valid value of {s}", .{ n, @typeName(T) });
                try appendDiag(list, arena, path, msg, null);
            }
            return error.InvalidEnumValue;
        },
        else => {
            if (options.errors) |list| {
                const msg = try std.fmt.allocPrint(arena, "expected string or integer for enum {s}, got {s}", .{ @typeName(T), @tagName(value) });
                try appendDiag(list, arena, path, msg, null);
            }
            return error.TypeMismatch;
        },
    }
}

// Streaming typed decode (no Value tree)

const tokenizer_mod = @import("tokenizer.zig");
const RawToken = tokenizer_mod.RawToken;

/// Comptime: true when decoding `T` requires a materialized `Value` (or a
/// whole-object view) somewhere in its type closure: `Value` targets,
/// `fromJson` hooks, unions (the `json_tag` discriminator may follow the
/// payload), flattened non-struct fields, and effective-key collisions
/// between a struct and its flattened fields. Those decode through the
/// tree path; everything else streams token-to-field.
fn needsTree(comptime T: type, comptime TAnnotation: type) bool {
    return comptime needsTreeImpl(T, TAnnotation, &.{});
}

fn needsTreeImpl(comptime T: type, comptime TAnnotation: type, comptime seen: []const type) bool {
    comptime {
        for (seen) |S| if (S == T) return false;
        if (T == Value) return true;
        const seen2 = seen ++ &[_]type{T};
        return switch (@typeInfo(T)) {
            .@"struct" => |s| blk: {
                if (@hasDecl(T, "fromJson")) break :blk true;
                if (TAnnotation.getOrEmpty(T)) |annotation| {
                    if (annotation.fromJson) break :blk true;
                }

                for (s.fields) |f| {
                    if (isFlattened(T, TAnnotation, f.name) and @typeInfo(f.type) != .@"struct") break :blk true;
                }
                if (hasKeyCollisions(T, TAnnotation)) break :blk true;
                for (s.fields) |f| {
                    if (needsTreeImpl(f.type, TAnnotation, seen2)) break :blk true;
                }
                break :blk false;
            },
            .@"union" => true,
            .pointer => |p| p.size == .slice and !(p.child == u8 and p.is_const) and needsTreeImpl(p.child, TAnnotation, seen2),
            .array => |a| needsTreeImpl(a.child, TAnnotation, seen2),
            .optional => |o| needsTreeImpl(o.child, TAnnotation, seen2),
            else => false,
        };
    }
}

/// One streamable destination: the effective wire key, the field path
/// from the outer struct (flattened fields contribute nested paths),
/// and the leaf type.
const EffField = struct {
    key: []const u8,
    path: []const []const u8,
    Type: type,
};

/// Effective field list of `T` with flattened inner structs expanded,
/// skipped fields excluded. Mirrors `expectedKeys` exactly.
fn effFieldsOf(comptime T: type, comptime TAnnotation: type, comptime prefix: []const []const u8) []const EffField {
    comptime {
        var out: []const EffField = &.{};
        for (@typeInfo(T).@"struct".fields) |f| {
            if (isSkipped(T, TAnnotation, f.name)) continue;
            const p2 = prefix ++ &[_][]const u8{f.name};
            if (isFlattened(T, TAnnotation, f.name)) {
                out = out ++ effFieldsOf(f.type, TAnnotation, p2);
            } else {
                out = out ++ &[_]EffField{.{ .key = renamedKey(T, TAnnotation, f.name), .path = p2, .Type = f.type }};
            }
        }
        return out;
    }
}

/// Two effective fields sharing one wire key (an outer field colliding
/// with a flattened inner one). The tree path decodes such a key into
/// every destination; a single token stream cannot, so collide -> tree.
fn hasKeyCollisions(comptime T: type, comptime TAnnotation: type) bool {
    comptime {
        const fs = effFieldsOf(T, TAnnotation, &.{});
        for (fs, 0..) |a, i| {
            for (fs[i + 1 ..]) |b| {
                if (std.mem.eql(u8, a.key, b.key)) return true;
            }
        }
        return false;
    }
}

fn PathType(comptime T: type, comptime path: []const []const u8) type {
    comptime {
        var C = T;
        for (path) |seg| C = @FieldType(C, seg);
        return C;
    }
}

fn pathPtr(comptime T: type, comptime path: []const []const u8, base: *T) *PathType(T, path) {
    if (comptime path.len == 0) return base;
    return pathPtr(@FieldType(T, path[0]), path[1..], &@field(base.*, path[0]));
}

/// Assign defaults to every `json_skip` field of `T`, recursing through
/// flattened inner structs. Mirrors the skip branch of `decodeStruct`.
fn assignSkippedDefaults(comptime T: type, comptime TAnnotation: type, comptime prefix: []const []const u8, comptime Outer: type, out: *Outer) void {
    inline for (@typeInfo(T).@"struct".fields) |f| {
        if (comptime isSkipped(T, TAnnotation, f.name)) {
            const dv = comptime f.defaultValue() orelse
                @compileError("json_skip field `" ++ f.name ++ "` on " ++ @typeName(T) ++ " has no default value");
            pathPtr(Outer, prefix ++ &[_][]const u8{f.name}, out).* = dv;
        } else if (comptime isFlattened(T, TAnnotation, f.name)) {
            assignSkippedDefaults(f.type, TAnnotation, prefix ++ &[_][]const u8{f.name}, Outer, out);
        }
    }
}

/// Streaming `parseInto`: one tokenizer pass decoding directly into `T`.
/// Success semantics match parse-then-decode exactly (same tokenizer,
/// same scalar decoders, same skip validation via `parseValue`); every
/// error abandons the pass and the caller reruns the tree path, whose
/// error selection and diagnostics are canonical.
fn streamParseInto(comptime T: type, comptime TAnnotation: type, arena: Allocator, src: []const u8, options: parser_mod.ParseOptions) (parser_mod.Error || DecodeError)!T {
    var stream_options = options;
    stream_options.errors = null;
    stream_options.spans = null;
    var p = parser_mod.Parser{
        .arena = arena,
        .input = src,
        .tokenizer = .init(src, options.dialect),
        .options = stream_options,
    };
    const t = (try p.next()) orelse return error.JsonParseError;
    const out = try streamValue(T, TAnnotation, &p, t, 0);
    if (try p.next()) |_| return error.JsonParseError;
    return out;
}

fn streamValue(comptime T: type, comptime TAnnotation: type, p: *parser_mod.Parser, t: RawToken, depth: usize) (parser_mod.Error || DecodeError)!T {
    const info = @typeInfo(T);
    if (comptime info == .optional) {
        if (t.kind == .literal_null) return null;
        return try streamValue(info.optional.child, TAnnotation, p, t, depth);
    }
    // JSON null into a non-optional mirrors decodeInner's policy.
    if (t.kind == .literal_null) return error.MissingField;

    // Scalars reuse the tree path's decoders on a scalar Value built by
    // the parser's own primitives, so numeric and enum semantics stay
    // single-sourced. Diagnostics are off (errors == null), so the
    // throwaway path builder never allocates.
    var path: PathBuilder = .{ .buf = .empty };
    return switch (comptime @typeInfo(T)) {
        .bool => switch (t.kind) {
            .literal_true => true,
            .literal_false => false,
            else => error.TypeMismatch,
        },
        .int => switch (t.kind) {
            .number => try decodeInt(T, try p.parseNumber(t), p.arena, p.options, &path),
            else => error.TypeMismatch,
        },
        .float => switch (t.kind) {
            .number => try decodeFloat(T, try p.parseNumber(t), p.arena, p.options, &path),
            else => error.TypeMismatch,
        },
        .@"enum" => switch (t.kind) {
            .string => try decodeEnum(T, .{ .string = try p.decodeString(t) }, p.arena, p.options, &path),
            .number => try decodeEnum(T, try p.parseNumber(t), p.arena, p.options, &path),
            else => error.TypeMismatch,
        },
        .pointer => |ptr| try streamPointer(T, TAnnotation, ptr, p, t, depth),
        .array => |arr| try streamFixedArray(T, TAnnotation, arr, p, t, depth),
        .@"struct" => try streamStruct(T, TAnnotation, p, t, depth),
        else => @compileError("json decode: unsupported type " ++ @typeName(T)),
    };
}

fn streamPointer(comptime T: type, comptime TAnnotation: type, comptime ptr: std.builtin.Type.Pointer, p: *parser_mod.Parser, t: RawToken, depth: usize) (parser_mod.Error || DecodeError)!T {
    if (comptime ptr.size != .slice) @compileError("json decode: only slice pointers supported, got " ++ @typeName(T));
    if (comptime (ptr.child == u8 and ptr.is_const)) {
        if (t.kind != .string) return error.TypeMismatch;
        return try p.decodeString(t);
    }
    if (t.kind != .array_begin) return error.TypeMismatch;
    if (depth >= p.depthLimit()) return error.NestingTooDeep;
    var items: std.ArrayList(ptr.child) = .empty;
    var at_first = true;
    while (true) {
        const et = (try p.next()) orelse return error.JsonParseError;
        if (et.kind == .array_end) {
            if (at_first or p.options.dialect == .jsonc) break;
            return error.JsonParseError;
        }
        at_first = false;
        try items.append(p.arena, try streamValue(ptr.child, TAnnotation, p, et, depth + 1));
        const sep = (try p.next()) orelse return error.JsonParseError;
        if (sep.kind == .array_end) break;
        if (sep.kind != .comma) return error.JsonParseError;
    }
    return items.toOwnedSlice(p.arena);
}

fn streamFixedArray(comptime T: type, comptime TAnnotation: type, comptime arr: std.builtin.Type.Array, p: *parser_mod.Parser, t: RawToken, depth: usize) (parser_mod.Error || DecodeError)!T {
    if (t.kind != .array_begin) return error.TypeMismatch;
    if (depth >= p.depthLimit()) return error.NestingTooDeep;
    var out: T = undefined;
    var i: usize = 0;
    var at_first = true;
    while (true) {
        const et = (try p.next()) orelse return error.JsonParseError;
        if (et.kind == .array_end) {
            if (at_first or p.options.dialect == .jsonc) break;
            return error.JsonParseError;
        }
        at_first = false;
        if (comptime arr.len == 0) return error.TypeMismatch;
        if (i >= arr.len) return error.TypeMismatch;
        out[i] = try streamValue(arr.child, TAnnotation, p, et, depth + 1);
        i += 1;
        const sep = (try p.next()) orelse return error.JsonParseError;
        if (sep.kind == .array_end) break;
        if (sep.kind != .comma) return error.JsonParseError;
    }
    if (i != arr.len) return error.TypeMismatch;
    return out;
}

fn streamStruct(comptime T: type, comptime TAnnotation: type, p: *parser_mod.Parser, t: RawToken, depth: usize) (parser_mod.Error || DecodeError)!T {
    comptime validateAnnotations(T);
    if (t.kind != .object_begin) return error.TypeMismatch;
    if (depth >= p.depthLimit()) return error.NestingTooDeep;

    const eff = comptime effFieldsOf(T, TAnnotation, &.{});
    var seen = [_]bool{false} ** eff.len;
    var out: T = undefined;
    assignSkippedDefaults(T, TAnnotation, &.{}, T, &out);

    var at_first = true;
    while (true) {
        const kt = (try p.next()) orelse return error.JsonParseError;
        if (kt.kind == .object_end) {
            if (at_first or p.options.dialect == .jsonc) break;
            return error.JsonParseError;
        }
        at_first = false;
        if (kt.kind != .string) return error.JsonParseError;
        const key = try p.decodeString(kt);
        const ct = (try p.next()) orelse return error.JsonParseError;
        if (ct.kind != .colon) return error.JsonParseError;
        const vt = (try p.next()) orelse return error.JsonParseError;

        var matched = false;
        inline for (eff, 0..) |f, idx| {
            if (!matched and std.mem.eql(u8, key, f.key)) {
                // Duplicate keys re-decode and overwrite: last wins,
                // matching the tree parser's object semantics.
                pathPtr(T, f.path, &out).* = try streamValue(f.Type, TAnnotation, p, vt, depth + 1);
                seen[idx] = true;
                matched = true;
            }
        }
        if (!matched) {
            if (!p.options.ignore_unknown_fields) return error.UnknownField;
            // Structural skip with identical validation and depth limits;
            // the discarded Value is arena garbage, same as the tree path.
            _ = try p.parseValue(vt, depth + 1);
        }

        const sep = (try p.next()) orelse return error.JsonParseError;
        if (sep.kind == .object_end) break;
        if (sep.kind != .comma) return error.JsonParseError;
    }

    inline for (eff, 0..) |f, idx| {
        if (!seen[idx]) {
            const Parent = PathType(T, f.path[0 .. f.path.len - 1]);
            const fi = comptime blk: {
                for (@typeInfo(Parent).@"struct".fields) |sf| {
                    if (std.mem.eql(u8, sf.name, f.path[f.path.len - 1])) break :blk sf;
                }
                unreachable;
            };
            const dv_opt = comptime fi.defaultValue();
            if (dv_opt) |dv| {
                pathPtr(T, f.path, &out).* = dv;
            } else if (comptime @typeInfo(f.Type) == .optional) {
                pathPtr(T, f.path, &out).* = null;
            } else {
                return error.MissingField;
            }
        }
    }
    return out;
}

const parse = @import("parser.zig").parse;
