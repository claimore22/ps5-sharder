use serde::{Deserialize, Serialize};
/// Describe a single IR instruction. These are HLSL-agnostic operations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Instruction {
    /// Load a value from a variable/location.
    Load { result: IrValue, source: IrValue },
    /// Store a value into a variable/location.
    Store { target: IrValue, value: IrValue },
    /// Construct a composite (vector/matrix/struct) from components.
    Construct {
        result: IrValue,
        ty: IrType,
        elements: Vec<IrValue>,
    },
    /// Arithmetic / comparison operations.
    Op {
        result: IrValue,
        op: IrOp,
        a: IrValue,
        b: Option<IrValue>,
    },
    /// Call a function.
    Call {
        result: Option<IrValue>,
        function: String,
        arguments: Vec<IrValue>,
    },
    /// Unconditional branching to a basic block.
    Branch { target: String },
    /// Conditional branching on a condition value.
    ConditionalBranch {
        condition: IrValue,
        then_target: String,
        else_target: String,
    },
    /// Return a value (or empty for `void`).
    Return(IrValue),
    /// A runtime constant (for literals) bound to a value.
    Constant { result: IrValue, literal: IrLiteral },
}

/// A named IR value (SSA-like). Identifiers are carried as strings.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct IrValue {
    pub name: String,
}

impl IrValue {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }
}

/// Binary/unary operations supported by the IR.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IrOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Equal,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    LogicalAnd,
    LogicalOr,
    BitAnd,
    BitOr,
    BitXor,
    ShiftLeft,
    ShiftRight,
    Not,
    Negate,
    BitNot,
    Comma,
}

impl IrOp {
    pub fn as_str(self) -> &'static str {
        match self {
            IrOp::Add => "add",
            IrOp::Sub => "sub",
            IrOp::Mul => "mul",
            IrOp::Div => "div",
            IrOp::Mod => "mod",
            IrOp::Equal => "eq",
            IrOp::NotEqual => "neq",
            IrOp::Less => "lt",
            IrOp::LessEqual => "le",
            IrOp::Greater => "gt",
            IrOp::GreaterEqual => "ge",
            IrOp::LogicalAnd => "and",
            IrOp::LogicalOr => "or",
            IrOp::BitAnd => "and",
            IrOp::BitOr => "or",
            IrOp::BitXor => "xor",
            IrOp::ShiftLeft => "shl",
            IrOp::ShiftRight => "shr",
            IrOp::Not => "not",
            IrOp::Negate => "neg",
            IrOp::BitNot => "not",
            IrOp::Comma => "comma",
        }
    }
}

/// Literal constants attached to [`Instruction::Constant`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum IrLiteral {
    Bool(bool),
    /// Numeric literal, lexeme preserved verbatim.
    Number(String),
    String(String),
}

use super::types::IrType;
