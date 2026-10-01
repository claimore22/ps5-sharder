use crate::error::Span;

use super::expr::Expr;
use super::types::{Semantic, Type};
use serde::{Deserialize, Serialize};

/// A `struct Name { ... }` declaration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StructDecl {
    pub name: String,
    pub fields: Vec<FieldDecl>,
    /// Base structs from `struct Derived : Base, ...` (flattened into
    /// `fields` post-parse; kept for provenance). Absent in older
    /// snapshots.
    #[serde(default)]
    pub bases: Vec<String>,
    /// Template parameters from a `template<...>` prefix, in order:
    /// (name, is_value). Absent in older snapshots. Drives struct
    /// monomorphization (absent = non-template).
    #[serde(default)]
    pub template_params: Vec<(String, bool)>,
    pub span: Span,
}

/// A single struct field.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FieldDecl {
    pub ty: Type,
    pub name: String,
    pub semantic: Option<Semantic>,
    pub array_size: Option<u32>,
    pub span: Span,
}

/// A global variable declaration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GlobalDecl {
    pub ty: Type,
    pub name: String,
    pub semantic: Option<Semantic>,
    pub array_size: Option<u32>,
    /// True when declared `thread_group_memory`/`groupshared`/`shared`
    /// (per-workgroup LDS): HLSL emits `groupshared` (writable), not a
    /// read-only const global.
    pub shared: bool,
    /// The `= init` expression when one was written. Most global
    /// initializers are consumed at parse (constant-folded or skipped);
    /// retained so lowering can resolve alias-style globals
    /// (`static const auto x = srt->member;`). Absent in older
    /// snapshots.
    #[serde(default)]
    pub init: Option<Expr>,
    pub span: Span,
}
