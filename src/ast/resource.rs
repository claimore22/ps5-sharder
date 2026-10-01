use crate::error::Span;
use serde::{Deserialize, Serialize};

use super::structure::FieldDecl;
use super::types::{Semantic, Type};

/// A top-level resource declaration, e.g. `Texture2D<float4> tex : register(t0);`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResourceDecl {
    pub name: String,
    pub kind: String,
    pub element_ty: Option<Type>,
    pub register: Option<(char, u32)>,
    pub register_space: Option<u32>,
    /// `: SEMANTIC` binding instead of `register(...)` (e.g. SRT-bound
    /// `TextureCube<float3> env : S_SRT_DATA;`). Mutually exclusive with
    /// `register` in practice.
    pub semantic: Option<Semantic>,
    pub fields: Vec<FieldDecl>,
    pub span: Span,
}
