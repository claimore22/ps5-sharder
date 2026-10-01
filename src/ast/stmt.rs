use crate::error::Span;

use super::expr::Expr;
use super::types::Type;
use serde::{Deserialize, Serialize};

/// A statement in a function body.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Stmt {
    /// `return [expr];`
    Return(Option<Expr>),
    /// `if (cond) then_block [else else_block]`
    If {
        condition: Box<Expr>,
        then_branch: Vec<Stmt>,
        else_branch: Option<Vec<Stmt>>,
    },
    /// `{ ... }`
    Block(Vec<Stmt>),
    /// `for (init; cond; update) body`
    For {
        init: Box<Stmt>,
        condition: Option<Expr>,
        update: Option<Expr>,
        body: Vec<Stmt>,
    },
    /// `while (cond) body`
    While {
        condition: Box<Expr>,
        body: Vec<Stmt>,
    },
    /// `switch (scrutinee) { case label: ...; default: ...; }`
    ///
    /// Case bodies are stored per label exactly as written, so C
    /// fallthrough semantics survive lowering verbatim; `label: None`
    /// is `default`. Statements before the first label (legal but
    /// unreachable in C) are prepended to the first case body.
    Switch {
        scrutinee: Box<Expr>,
        cases: Vec<SwitchCase>,
    },
    /// `[storage] type name [array_size] [= init];`
    VarDecl {
        storage: Option<StorageClass>,
        ty: Type,
        name: String,
        array_size: Option<u32>,
        init: Option<Expr>,
    },
    /// C++17 structured binding `auto [a, b] = expr;` (only multi-name
    /// bindings take this shape; single names lower directly to VarDecl).
    /// Desugared in lowering once the init type is known.
    Destructure {
        storage: Option<StorageClass>,
        ty: Type,
        names: Vec<String>,
        init: Option<Expr>,
    },
    /// A bare expression statement, including assignments.
    Expr(Expr),
}

/// One `case label:` / `default:` group inside a `switch`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SwitchCase {
    /// `None` is `default`.
    pub label: Option<Expr>,
    pub body: Vec<Stmt>,
}

/// Storage class qualifier on a variable declaration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StorageClass {
    Const,
    Static,
}

impl StorageClass {
    pub fn as_str(self) -> &'static str {
        match self {
            StorageClass::Const => "const",
            StorageClass::Static => "static",
        }
    }
}

/// A statement together with its source span.
pub struct SpannedStmt {
    pub stmt: Stmt,
    pub span: Span,
}
