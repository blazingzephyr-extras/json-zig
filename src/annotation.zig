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
        ///
        pub const annotation_type: type = T;

        /// Name overrides.
        json_rename: ?type = null,
        /// Sub-fields are decoded from the parent object
        json_flatten: ?[]const []const u8 = null,
        /// Excluded from decode/encode.
        json_skip: ?[]const []const u8 = null,
        /// Custom deserialization of T,
        fromJson: ?*const fn (arena: Allocator, value: Value, options: ParseOptions) DecodeError!T = null,
        /// Custom serialization of T,
        toJson: ?*const fn (self: T, arena: Allocator) Allocator.Error!Value = null,
        /// Discriminator member for tagged unions,
        json_tag: ?[]const u8 = null,
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
        for (options) |annotation_entry| {
            const TOption = @TypeOf(annotation_entry);
            if (!@hasDecl(TOption, "annotation_type")) break;
            if (!@hasField(TOption, "json_rename")) break;
            if (!@hasField(TOption, "json_flatten")) break;
            if (!@hasField(TOption, "json_skip")) break;
            if (!@hasField(TOption, "fromJson")) break;
            if (!@hasField(TOption, "toJson")) break;
            if (!@hasField(TOption, "json_tag")) break;

            const T = TOption.annotation_type;
            const kind = if (@typeInfo(T) == .@"union") "variant" else "field";
            if (annotation_entry.json_rename) |rename| {
                for (rename.fields) |rf| {
                    if (!@hasField(T, rf.name)) {
                        @compileError("json_rename entry `" ++ rf.name ++ "` does not match any " ++ kind ++ " of " ++ @typeName(T));
                    }
                }
            }
            if (annotation_entry.json_skip) |skip| {
                for (skip) |name| {
                    if (!@hasField(T, name)) {
                        @compileError("json_skip entry `" ++ name ++ "` does not match any " ++ kind ++ " of " ++ @typeName(T));
                    }
                }
            }
            if (annotation_entry.json_flatten) |flatten| {
                for (flatten) |name| {
                    if (!@hasField(T, name)) {
                        @compileError("json_flatten entry `" ++ name ++ "` does not match any " ++ kind ++ " of " ++ @typeName(T));
                    }
                }
            }
        } else return struct {
            const annotation = options;

            /// Determines whether an entry for T exists.
            pub fn has(comptime T: type) bool {
                return inline for (annotation) |annotation_entry| {
                    const TOption = @TypeOf(annotation_entry);
                    if (TOption.annotation_type == T) break true;
                } else false;
            }

            /// Retrieves entry for T.
            pub fn get(comptime T: type) TypeAnnotationProvider(T) {
                inline for (annotation) |annotation_entry| {
                    const TOption = @TypeOf(annotation_entry);
                    if (TOption.annotation_type == T) return annotation_entry;
                } else @compileError("Annotation registry lacks entry for " ++ T ++ ".");
            }

            pub fn getOrEmpty(comptime T: type) ?TypeAnnotationProvider(T) {
                return inline for (annotation) |annotation_entry| {
                    const TOption = @TypeOf(annotation_entry);
                    if (TOption.annotation_type == T) break annotation_entry;
                } else null;
            }
        };

        @compileError("Type annotation should be exactly a TypeAnnotationProvider(T) instance.");
    }
}
