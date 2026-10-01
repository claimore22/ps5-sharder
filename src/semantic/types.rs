use std::collections::HashMap;

use crate::Span;
use crate::ast::*;
use crate::error::{Phase, Ps5ShaderError};
use crate::ir::IrType;

/// A mapping from variable/parameter names to their types within a function.
#[derive(Debug, Default)]
pub struct TypeEnv {
    locals: HashMap<String, Type>,
    /// Names declared with an array size (`float4 AABB[48]`); the stored
    /// [`Type`] is the element type, so indexing yields it directly instead
    /// of descending into vector components.
    arrays: std::collections::HashSet<String>,
}

impl TypeEnv {
    pub fn insert(&mut self, name: impl Into<String>, ty: Type) {
        self.locals.insert(name.into(), ty);
    }

    pub fn get(&self, name: &str) -> Option<&Type> {
        self.locals.get(name)
    }

    pub fn mark_array(&mut self, name: impl Into<String>) {
        self.arrays.insert(name.into());
    }

    pub fn is_array(&self, name: &str) -> bool {
        self.arrays.contains(name)
    }
}

/// Converts an AST [`Type`] into an IR [`IrType`].
pub fn to_ir_type(ty: &Type) -> IrType {
    match ty {
        Type::Void => IrType::Void,
        Type::Bool => IrType::Bool,
        Type::Int => IrType::Int,
        Type::UInt => IrType::UInt,
        Type::Float => IrType::Float,
        Type::Half => IrType::Half,
        Type::Double => IrType::Double,
        Type::Vector { base, width } => IrType::Vector {
            base: to_ir_scalar(*base),
            width: *width,
        },
        Type::Matrix {
            base,
            rows,
            columns,
        } => IrType::Matrix {
            base: to_ir_scalar(*base),
            rows: *rows,
            columns: *columns,
        },
        Type::Named(name) => IrType::Named(name.clone()),
        Type::Resource {
            kind,
            access,
            element,
        } => IrType::Resource {
            kind: kind.clone(),
            access: match access {
                crate::ast::ResourceAccess::ReadOnly => crate::ir::IrAccess::ReadOnly,
                crate::ast::ResourceAccess::ReadWrite => crate::ir::IrAccess::ReadWrite,
            },
            element: Box::new(to_ir_type(element)),
        },
    }
}

fn to_ir_scalar(base: ScalarType) -> crate::ir::IrScalar {
    use crate::ir::IrScalar;
    match base {
        ScalarType::Float => IrScalar::Float,
        ScalarType::Half => IrScalar::Half,
        ScalarType::Int => IrScalar::Int,
        ScalarType::UInt => IrScalar::UInt,
        ScalarType::Bool => IrScalar::Bool,
    }
}

/// A convenience constructor for type errors emitted during semantic analysis.
pub fn type_error(message: impl Into<String>, span: Span) -> Ps5ShaderError {
    Ps5ShaderError::at(message, Phase::Semantic, span)
}

/// A convenience constructor for a resolved-type mismatch diagnostic that
/// mirrors the plan's example output (`expected: float4 / found: float3`).
pub fn mismatch(expected: &Type, found: &Type, span: Span) -> Ps5ShaderError {
    Ps5ShaderError::at(
        format!(
            "type mismatch: expected {}, found {}",
            expected.display(),
            found.display()
        ),
        Phase::Semantic,
        span,
    )
}
