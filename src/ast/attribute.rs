use super::expr::Expr;
use serde::{Deserialize, Serialize};

/// A `[Name(args...)]` attribute attached to a declaration.
///
/// Attributes are represented generically so that unknown target attributes
/// (e.g. `[CxxSymbol("Shader::ps")]`) survive parsing without being
/// hardcoded into the grammar.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Attribute {
    pub name: String,
    pub arguments: Vec<Expr>,
}
