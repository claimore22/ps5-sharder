pub mod function;
pub mod instruction;
pub mod resource;
pub mod stage;
pub mod types;

pub use function::{function_calls, reachable_methods, Function, IrStmt, SwitchCase};
pub use instruction::{Instruction, IrLiteral, IrOp, IrValue};
pub use resource::{Resource, ResourceType};
pub use stage::Stage;
pub use types::{IrAccess, IrScalar, IrType};
use serde::{Deserialize, Serialize};

/// A struct declaration in the IR.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StructDecl {
    pub name: String,
    pub fields: Vec<FieldDecl>,
}

/// A single struct field in the IR.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FieldDecl {
    pub ty: IrType,
    pub name: String,
    /// The raw PSSL semantic name (e.g. "S_POSITION"). Keep as-is; the
    /// emitter maps it to a Vulkan HLSL semantic.
    pub semantic: Option<String>,
    pub array_size: Option<u32>,
}

/// An IR expression (HLSL-agnostic).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum IrExpr {
    Literal(IrLiteral),
    Variable(String),
    /// `object.member`
    Member {
        object: Box<IrExpr>,
        member: String,
    },
    /// `object[index]`
    Index {
        object: Box<IrExpr>,
        index: Box<IrExpr>,
    },
    /// `resource[coord]` — indexing a resource handle (e.g. RW texture).
    ResourceIndex {
        resource: Box<IrExpr>,
        index: Box<IrExpr>,
    },
    /// `resource.Load(coords[, sample])` — an explicit resource load.
    ResourceLoad {
        resource: Box<IrExpr>,
        coordinates: Box<IrExpr>,
        sample: Option<Box<IrExpr>>,
    },
    /// `callee(args...)`
    Call {
        function: String,
        arguments: Vec<IrExpr>,
    },
    /// `TypeName(args...)`
    Construct {
        ty: IrType,
        arguments: Vec<IrExpr>,
    },
    Unary {
        op: IrOp,
        expr: Box<IrExpr>,
    },
    Binary {
        left: Box<IrExpr>,
        op: IrOp,
        right: Box<IrExpr>,
    },
    Conditional {
        condition: Box<IrExpr>,
        then_branch: Box<IrExpr>,
        else_branch: Box<IrExpr>,
    },
    Assign {
        lhs: Box<IrExpr>,
        rhs: Box<IrExpr>,
    },
    /// `{ a, b, c }` — brace initializer list.
    InitializerList(Vec<IrExpr>),
}

/// The top-level Shader IR module.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ShaderModule {
    pub stage: Option<Stage>,
    pub structs: Vec<StructDecl>,
    pub resources: Vec<Resource>,
    pub constants: Vec<IrValue>,
    pub globals: Vec<GlobalVar>,
    pub functions: Vec<Function>,
    pub entry_point: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GlobalVar {
    pub ty: IrType,
    pub name: String,
    pub array_size: Option<u32>,
    /// For compile-time constants (e.g. `static constexpr uint k_size = 8;`),
    /// the resolved literal value, which the emitter renders as
    /// `static const uint k_size = 8;` rather than an uninitialized global.
    pub const_value: Option<String>,
    /// Per-workgroup LDS (`thread_group_memory`/`groupshared` in PSSL):
    /// emitted as HLSL `groupshared` (writable), never a plain global
    /// (which SPIR-V treats as read-only). Absent in older snapshots.
    #[serde(default)]
    pub shared: bool,
}
