use super::stmt::Stmt;
use super::types::Type;
use serde::{Deserialize, Serialize};

/// Binary operators in expressions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BinaryOp {
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
    /// C comma operator `a, b` (e.g. multi-update `for` loops).
    Comma,
}

impl BinaryOp {
    pub fn symbol(self) -> &'static str {
        match self {
            BinaryOp::Add => "+",
            BinaryOp::Sub => "-",
            BinaryOp::Mul => "*",
            BinaryOp::Div => "/",
            BinaryOp::Mod => "%",
            BinaryOp::Equal => "==",
            BinaryOp::NotEqual => "!=",
            BinaryOp::Less => "<",
            BinaryOp::LessEqual => "<=",
            BinaryOp::Greater => ">",
            BinaryOp::GreaterEqual => ">=",
            BinaryOp::LogicalAnd => "&&",
            BinaryOp::LogicalOr => "||",
            BinaryOp::BitAnd => "&",
            BinaryOp::BitOr => "|",
            BinaryOp::BitXor => "^",
            BinaryOp::ShiftLeft => "<<",
            BinaryOp::ShiftRight => ">>",
            BinaryOp::Comma => ",",
        }
    }
}

/// Unary operators in expressions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UnaryOp {
    Plus,
    Minus,
    Not,
    BitNot,
}

impl UnaryOp {
    pub fn symbol(self) -> &'static str {
        match self {
            UnaryOp::Plus => "+",
            UnaryOp::Minus => "-",
            UnaryOp::Not => "!",
            UnaryOp::BitNot => "~",
        }
    }
}

/// A PSSL expression.
#[derive(Debug, Clone, PartialEq,Serialize, Deserialize)]
pub enum Expr {
    /// Integer or floating-point literal, lexeme preserved verbatim.
    Number(String),
    /// String literal, unquoted.
    String(String),
    Bool(bool),
    Variable(String),
    /// `object.member`
    Member {
        object: Box<Expr>,
        member: String,
    },
    /// `object[index]`
    Index {
        object: Box<Expr>,
        index: Box<Expr>,
    },
    /// `callee(args...)` — an arbitrary function call.
    Call {
        function: Box<Expr>,
        arguments: Vec<Expr>,
    },
    /// `TypeName(args...)` — disambiguated from a call by the parser/type table.
    Constructor {
        ty: Type,
        arguments: Vec<Expr>,
    },
    /// `lhs = rhs`
    Assign {
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },
    Unary {
        op: UnaryOp,
        expr: Box<Expr>,
    },
    Binary {
        left: Box<Expr>,
        op: BinaryOp,
        right: Box<Expr>,
    },
    /// `a ? b : c`
    Conditional {
        condition: Box<Expr>,
        then_branch: Box<Expr>,
        else_branch: Box<Expr>,
    },
    /// `{ a, b, c }` — brace initializer list (including nested, e.g. `{{1,2},{3,4}}`).
    InitializerList(Vec<Expr>),
    /// C++ lambda (`[](ulong v) { return ...; }`, `[&]() { ... };`).
    /// The capture items (verbatim: `&`, `=`, `srt`, `&x`), parameter
    /// NAMES (in order), and body are kept. Only capture-less lambdas
    /// beta-reduce (types come from `auto` deduction after inlining);
    /// capturing lambdas inline solely as immediate callees, else keep
    /// the legacy opaque `__lambda` behavior downstream.
    Lambda {
        captures: Vec<String>,
        params: Vec<String>,
        body: Vec<Stmt>,
    },
}
