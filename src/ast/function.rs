
use crate::error::Span;

use super::attribute::Attribute;
use super::expr::Expr;
use super::stmt::Stmt;
use super::types::{Direction, Semantic, Type};
use serde::{Deserialize, Serialize};

/// A function declaration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Function {
    pub attributes: Vec<Attribute>,
    pub return_type: Type,
    pub name: String,
    /// Template parameters from a `template<...>` prefix, in order:
    /// (name, is_value). `typename T`/`class T` are type params;
    /// `bool B`/`int N`/etc. are value params. Absent in older
    /// snapshots. Drives monomorphization (absent = non-template).
    #[serde(default)]
    pub template_params: Vec<(String, bool)>,
    /// Out-of-class scope (`Scope` in `Ret Scope::method(...)`), if any.
    /// Lets semantic analysis resolve bare member accesses against the
    /// scope's struct fields.
    pub scope: Option<String>,
    pub parameters: Vec<Parameter>,
    pub return_semantic: Option<Semantic>,
    pub body: Vec<Stmt>,
    pub span: Span,
}

/// A function parameter.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Parameter {
    pub ty: Type,
    pub name: String,
    pub direction: Option<Direction>,
    pub semantic: Option<Semantic>,
    /// Array dimension (`float3 L[4]` => Some(4)). Sizes that aren't
    /// compile-time constants (e.g. `k_lds_size`) become Some(0) (unsized),
    /// matching the array-size convention used for globals/fields.
    pub array_size: Option<u32>,
    /// Geometry-shader input primitive topology (`point`, `line`,
    /// `triangle`, …): `Point T x` in PSSL. Lowercase HLSL spelling.
    pub gs_primitive: Option<String>,
    /// C++ default argument (`float gamma = 2.2`): filled in at call
    /// sites during lowering (HLSL has no default args). Absent in
    /// older snapshots.
    #[serde(default)]
    pub default: Option<Expr>,
    /// Explicit `uniform` qualifier (`uniform float4x4 wvp` on entries):
    /// host-provided constants, lowered to cbuffer members (Vulkan
    /// entries take only varyings/resources). Absent in older snapshots.
    #[serde(default)]
    pub is_uniform: bool,
    pub span: Span,
}
