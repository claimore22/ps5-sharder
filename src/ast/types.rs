use crate::error::Span;
use serde::{Deserialize, Serialize};

/// A scalar base component type for vector and matrix types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScalarType {
    Float,
    Half,
    Int,
    UInt,
    Bool,
}

impl ScalarType {
    pub fn name(self) -> &'static str {
        match self {
            ScalarType::Float => "float",
            ScalarType::Half => "half",
            ScalarType::Int => "int",
            ScalarType::UInt => "uint",
            ScalarType::Bool => "bool",
        }
    }
}

/// A PSSL type.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Type {
    Void,
    Bool,
    Int,
    UInt,
    Float,
    Half,
    Double,
    Vector {
        base: ScalarType,
        width: u8,
    },
    Matrix {
        base: ScalarType,
        rows: u8,
        columns: u8,
    },
/// A user-defined (or as-yet-unknown) named type, e.g. a struct name.
Named(String),
/// A resource type used as a struct/constant-buffer member or value, e.g.
/// `Texture2D<float4>`, `RW_Texture2D<float>`, or `MS_Texture2D<float>`.
///
/// `kind` is the normalized (lowercase, underscore-stripped) PSSL spelling
/// such as `texture2d`, `rwtexture2d`, `texturecube`, or `mstexture2d`.
/// `element` is the bound element type, e.g. `float4` or `float`.
/// `access` records whether the resource is read-only or read-write, which
/// keeps `MS_Texture2D<float>` distinct from `RW_Texture2D<float>` — both may
/// share a dimension kind (`texture2d`) but differ in access.
Resource {
    kind: String,
    access: ResourceAccess,
    element: Box<Type>,
},
}

impl Type {
    /// The span of source text from which this type was parsed.
    pub fn display(&self) -> String {
        match self {
            Type::Void => "void".to_string(),
            Type::Bool => "bool".to_string(),
            Type::Int => "int".to_string(),
            Type::UInt => "uint".to_string(),
            Type::Float => "float".to_string(),
            Type::Half => "half".to_string(),
            Type::Double => "double".to_string(),
            Type::Vector { base, width } => format!("{}{width}", base.name()),
            Type::Matrix {
                base,
                rows,
                columns,
            } => {
                format!("{}{rows}x{columns}", base.name())
            }
            Type::Named(name) => name.clone(),
            Type::Resource {
                kind,
                access: _,
                element,
            } => {
                let kind_display = kind.replace("mstexture2d", "MS_Texture2D");
                let kind_display = match kind_display.as_str() {
                    "texture1d" => "Texture1D",
                    "texture2d" => "Texture2D",
                    "rwtexture2d" => "RW_Texture2D",
                    "texturecube" => "TextureCube",
                    "rwtexturecube" => "RW_TextureCube",
                    "mstexture2d" => "MS_Texture2D",
                    other => other,
                };
                format!("{kind_display}<{}>", element.display())
            }
        }
    }
}

/// Access mode for a resource type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResourceAccess {
    /// Read-only resource (e.g. `Texture2D`, `MS_Texture2D`, `TextureCube`).
    ReadOnly,
    /// Read-write resource (e.g. `RW_Texture2D`, `RW_TextureCube`).
    ReadWrite,
}

impl ResourceAccess {
    pub fn as_str(self) -> &'static str {
        match self {
            ResourceAccess::ReadOnly => "read_only",
            ResourceAccess::ReadWrite => "read_write",
        }
    }
}

/// A semantic attached to a field, parameter, or return value.
///
/// These are PSSL semantic names (e.g. `S_POSITION`). The parser treats the
/// lexical name as an opaque identifier; the *semantic mapping* layer decides
/// what the name means and how it maps to Vulkan HLSL semantics.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Semantic {
    /// The raw semantic name as written in source, e.g. "S_POSITION", "COLOR".
    Name(String),
}

impl Semantic {
    pub fn name(&self) -> &str {
        match self {
            Semantic::Name(name) => name,
        }
    }
}

/// Parameter direction qualifier (PSSL `in` / `out` / `inout`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Direction {
    In,
    Out,
    InOut,
}

impl Direction {
    pub fn as_str(self) -> &'static str {
        match self {
            Direction::In => "in",
            Direction::Out => "out",
            Direction::InOut => "inout",
        }
    }
}

/// A type together with the span where it appeared (used for diagnostics).
pub struct SpannedType {
    pub ty: Type,
    pub span: Span,
}
