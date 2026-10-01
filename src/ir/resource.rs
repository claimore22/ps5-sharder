use serde::{Deserialize, Serialize};
/// The kind of a shader resource.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResourceType {
    Texture,
    Sampler,
    ConstantBuffer,
    StructuredBuffer,
    RWBuffer,
    RWTexture,
}

impl ResourceType {
    pub fn as_str(self) -> &'static str {
        match self {
            ResourceType::Texture => "texture",
            ResourceType::Sampler => "sampler",
            ResourceType::ConstantBuffer => "constant_buffer",
            ResourceType::StructuredBuffer => "structured_buffer",
            ResourceType::RWBuffer => "rw_buffer",
            ResourceType::RWTexture => "rw_texture",
        }
    }
}

/// A shader resource with explicit binding/register metadata.
///
/// Descriptor sets, bindings, registers, and spaces are kept as explicit
/// fields here (not buried inside HLSL strings) so that the information can
/// later be reconciled against compiled-shader / AGC metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Resource {
    pub name: String,
    pub resource_type: ResourceType,
    /// Normalized (lowercase, underscore-stripped) PSSL resource kind, e.g.
    /// `texture2darray`, `texture3d`, `rwtexture2d`. Preserved so the emitter
    /// can reconstruct the exact HLSL resource type instead of collapsing
    /// every texture-like resource to a bound `Texture2D`.
    pub kind: String,
    pub element_type: Option<crate::ir::IrType>,
    pub descriptor_set: Option<u32>,
    pub binding: Option<u32>,
    pub register: Option<u32>,
    pub space: Option<u32>,
    pub array_size: Option<u32>,
    pub fields: Vec<crate::ir::FieldDecl>,
}
