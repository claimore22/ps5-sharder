use serde::{Deserialize, Serialize};
/// A typed Shader IR value.
///
/// This mirrors the PSSL type system. It is deliberately HLSL-agnostic.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum IrType {
    Void,
    Bool,
    Int,
    UInt,
    Float,
    Half,
    Double,
    /// A vector, e.g. float3.
    Vector {
        base: IrScalar,
        width: u8,
    },
    /// A matrix, e.g. float3x4.
    Matrix {
        base: IrScalar,
        rows: u8,
        columns: u8,
    },
    /// A named/composite type (e.g. a struct) referenced by name.
    Named(String),
    /// A resource type carried as a struct/constant-buffer member, e.g.
    /// `Texture2D<float4>`, `RW_Texture2D<float>`, or `MS_Texture2D<float>`.
    /// `kind` is the normalized (lowercase, underscore-stripped) PSSL spelling.
    /// `access` records the read-only/read-write mode so that, e.g.
    /// `MS_Texture2D<float>` (read-only) stays distinct from `RW_Texture2D<float>`
    /// (read-write) even when both share a dimension kind.
    Resource {
        kind: String,
        access: IrAccess,
        element: Box<IrType>,
    },
}

/// Read-only vs. read-write mode for an IR resource type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IrAccess {
    ReadOnly,
    ReadWrite,
}

impl IrAccess {
    pub fn as_str(self) -> &'static str {
        match self {
            IrAccess::ReadOnly => "read_only",
            IrAccess::ReadWrite => "read_write",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IrScalar {
    Float,
    Half,
    Int,
    UInt,
    Bool,
}

impl IrScalar {
    pub fn name(self) -> &'static str {
        match self {
            IrScalar::Float => "float",
            IrScalar::Half => "half",
            IrScalar::Int => "int",
            IrScalar::UInt => "uint",
            IrScalar::Bool => "bool",
        }
    }
}

impl IrType {
    pub fn display(&self) -> String {
        match self {
            IrType::Void => "void".to_string(),
            IrType::Bool => "bool".to_string(),
            IrType::Int => "int".to_string(),
            IrType::UInt => "uint".to_string(),
            IrType::Float => "float".to_string(),
            IrType::Half => "half".to_string(),
            IrType::Double => "double".to_string(),
            IrType::Vector { base, width } => format!("{}{width}", base.name()),
            IrType::Matrix {
                base,
                rows,
                columns,
            } => {
                format!("{}{rows}x{columns}", base.name())
            }
            IrType::Named(name) => name.clone(),
            IrType::Resource {
                kind,
                access: _,
                element,
            } => format!(
                "{}{}",
                resource_hlsl_kind_name(kind),
                element.display()
            ),
        }
    }
}

/// Maps a normalized PSSL resource kind to its HLSL spelling (without the
/// `<element>` suffix, which `display` appends).
fn resource_hlsl_kind_name(kind: &str) -> String {
    match kind {
        "texture1d" => "Texture1D<".to_string(),
        "texture2d" => "Texture2D<".to_string(),
        "texturecube" => "TextureCube<".to_string(),
        "rwtexture2d" => "RWTexture2D<".to_string(),
        "rwtexturecube" => "RWTextureCube<".to_string(),
        "mstexture2d" => "Texture2DMS<".to_string(),
        "rwmstexture2d" => "RWTexture2DArray<".to_string(),
        "regularbuffer" | "structuredbuffer" => "StructuredBuffer<".to_string(),
        "rwregularbuffer" | "rwstructuredbuffer" => "RWStructuredBuffer<".to_string(),
        "databuffer" => "Buffer<".to_string(),
        "rwdatabuffer" | "rwbuffer" => "RWBuffer<".to_string(),
        other => format!("{other}<"),
    }
}
