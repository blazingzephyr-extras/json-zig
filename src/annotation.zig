//! Type annotation utilities.
const std = @import("std");
const decode = @import("decode.zig");
const parser = @import("parser.zig");
const v = @import("value.zig");

const Allocator = std.mem.Allocator;
const DecodeError = decode.DecodeError;
const ParseOptions = parser.ParseOptions;
const Value = v.Value;

/// Provides JSON tags and parsing hooks for typed decoding.
/// Currently supports structs and tagged unions.
pub fn TypeAnnotationProvider(comptime T: type) type {
    return struct {
        /// Name overrides.
        json_rename: ?type,
        /// Sub-fields are decoded from the parent object
        json_flatten: ?[]const []const u8,
        /// Excluded from decode/encode.
        json_skip: ?[]const []const u8,
        /// Custom deserialization of T,
        fromJson: ?*const fn (arena: Allocator, value: Value, options: ParseOptions) DecodeError!T,
        /// Custom serialization of T,
        toJson: ?*const fn (self: T, arena: Allocator) Allocator.Error!Value,
        /// Discriminator member for tagged unions,
        json_tag: ?[]const u8,
    };
}

/// Default, empty type annotation registry.
/// Types will be decoded using internal annotation options.
pub const DefaultTypeAnnotation = TypeAnnotationOptions(.{});

/// Constructs annotation options for typed encoding and decoding.
/// Reasoning:
/// 1. Support types from other packages;
/// 2. Support compile-time generated types from @Struct and @Union, where
/// embedding annotations by declaring fields is currently impossible /
/// undesirable behavior;
/// 3. Allow to overwrite default annotation.
pub fn TypeAnnotationOptions(comptime options: anytype) type {
    comptime {
        for (options) |type_annotation| {
            if (type_annotation.len < 2) break;

            const T = type_annotation[0];
            const annotation = type_annotation[1];
            if (@TypeOf(T) != type) break;
            if (@TypeOf(annotation) != TypeAnnotationProvider(T)) break;

            const kind = if (@typeInfo(T) == .@"union") "variant" else "field";
            if (annotation.json_rename) {
                for (annotation.json_rename.fields) |rf| {
                    if (!@hasField(T, rf.name)) {
                        @compileError("json_rename entry `" ++ rf.name ++ "` does not match any " ++ kind ++ " of " ++ @typeName(T));
                    }
                }
            }
            if (annotation.json_skip) {
                for (annotation.json_skip) |name| {
                    if (!@hasField(T, name)) {
                        @compileError("json_skip entry `" ++ name ++ "` does not match any " ++ kind ++ " of " ++ @typeName(T));
                    }
                }
            }
            if (annotation.json_flatten) {
                for (annotation.json_flatten) |name| {
                    if (!@hasField(T, name)) {
                        @compileError("json_flatten entry `" ++ name ++ "` does not match any " ++ kind ++ " of " ++ @typeName(T));
                    }
                }
            }
        } else return struct {
            const annotation = options;

            /// Determines whether an entry for T exists.
            pub fn has(comptime T: type) bool {
                return inline for (annotation) |entry| {
                    if (entry[0] == T) break true;
                } else false;
            }

            /// Retrieves entry for T.
            pub fn get(comptime T: type) TypeAnnotationProvider(T) {
                inline for (annotation) |entry| {
                    if (entry[0] == T) return entry[1];
                } else @compileError("Annotation registry lacks entry for " ++ T ++ ".");
            }

            pub fn getOrEmpty(comptime T: type) ?TypeAnnotationProvider(T) {
                return inline for (annotation) |entry| {
                    if (entry[0] == T) break entry[1];
                } else null;
            }
        };

        @compileError("Type annotation should be exactly (Type, TypeAnnotationProvider(T) instance)");
    }
}
