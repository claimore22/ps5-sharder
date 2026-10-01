//! C++ default arguments (`float f(float y, C c = RRT_PARAMS)`).
//!
//! HLSL has no default args, so call sites with fewer arguments are
//! filled from the callee's lowered defaults. Only unambiguous calls
//! qualify: a plain free-function name with exactly one overload whose
//! trailing defaults cover the gap, and no exact-arity overload (which
//! DXC would bind instead). Methods (`obj.m`, `S::m`), overload sets,
//! and missing defaults keep today's loud path.

use std::collections::HashMap;

use crate::ir::{IrExpr, IrStmt, ShaderModule};

/// Per-overload trailing defaults of one free-function name.
type Signatures = Vec<Vec<Option<IrExpr>>>;

/// Fills default arguments at every call site in every function body.
/// Shaders without defaulted calls are untouched.
pub(crate) fn lower_default_args(ir: &mut ShaderModule) {
    // Free-function name -> per-overload defaults (owned: the body walk
    // below mutates the same function list).
    let mut table: HashMap<String, Signatures> = HashMap::new();
    for f in &ir.functions {
        if f.scope.is_none() {
            table
                .entry(f.name.clone())
                .or_default()
                .push(f.parameters.iter().map(|p| p.default.clone()).collect());
        }
    }
    for function in &mut ir.functions {
        lower_defargs_in_stmts(&mut function.body, &table);
    }
}

fn lower_defargs_in_stmts(stmts: &mut Vec<IrStmt>, table: &HashMap<String, Signatures>) {
    for stmt in stmts.iter_mut() {
        lower_defargs_in_stmt(stmt, table);
    }
}

fn lower_defargs_in_stmt(stmt: &mut IrStmt, table: &HashMap<String, Signatures>) {
    match stmt {
        IrStmt::Return(e) => {
            if let Some(e) = e {
                lower_defargs_in_expr(e, table);
            }
        }
        IrStmt::VarDecl { init, .. } => {
            if let Some(init) = init {
                lower_defargs_in_expr(init, table);
            }
        }
        IrStmt::If {
            condition,
            then_branch,
            else_branch,
        } => {
            lower_defargs_in_expr(condition, table);
            lower_defargs_in_stmts(then_branch, table);
            if let Some(else_branch) = else_branch {
                lower_defargs_in_stmts(else_branch, table);
            }
        }
        IrStmt::For {
            init,
            condition,
            update,
            body,
        } => {
            lower_defargs_in_stmt(init, table);
            if let Some(condition) = condition {
                lower_defargs_in_expr(condition, table);
            }
            if let Some(update) = update {
                lower_defargs_in_expr(update, table);
            }
            lower_defargs_in_stmts(body, table);
        }
        IrStmt::While { condition, body } => {
            lower_defargs_in_expr(condition, table);
            lower_defargs_in_stmts(body, table);
        }
        IrStmt::Switch { scrutinee, cases } => {
            lower_defargs_in_expr(scrutinee, table);
            for case in cases {
                if let Some(label) = &mut case.label {
                    lower_defargs_in_expr(label, table);
                }
                lower_defargs_in_stmts(&mut case.body, table);
            }
        }
        IrStmt::Block(inner) => lower_defargs_in_stmts(inner, table),
        IrStmt::Expr(e) => lower_defargs_in_expr(e, table),
    }
}

fn lower_defargs_in_expr(expr: &mut IrExpr, table: &HashMap<String, Signatures>) {
    // Fill this call first (bounded: defaults cannot name later
    // parameters, so chained fills terminate), then recurse into all
    // arguments including appended defaults.
    if let IrExpr::Call { function, arguments } = expr {
        if !function.contains('.') && !function.contains("::") {
            let mut depth = 0u32;
            while depth < 8 {
                let Some(filled) = fill_call(function, arguments, table) else {
                    break;
                };
                *arguments = filled;
                depth += 1;
            }
        }
    }
    match expr {
        IrExpr::Call { arguments, .. } | IrExpr::Construct { arguments, .. } => {
            for a in arguments {
                lower_defargs_in_expr(a, table);
            }
        }
        IrExpr::Member { object, .. } => lower_defargs_in_expr(object, table),
        IrExpr::Index { object, index } | IrExpr::ResourceIndex { resource: object, index } => {
            lower_defargs_in_expr(object, table);
            lower_defargs_in_expr(index, table);
        }
        IrExpr::ResourceLoad {
            resource,
            coordinates,
            sample,
        } => {
            lower_defargs_in_expr(resource, table);
            lower_defargs_in_expr(coordinates, table);
            if let Some(sample) = sample {
                lower_defargs_in_expr(sample, table);
            }
        }
        IrExpr::Unary { expr: inner, .. } => lower_defargs_in_expr(inner, table),
        IrExpr::Binary { left, right, .. } => {
            lower_defargs_in_expr(left, table);
            lower_defargs_in_expr(right, table);
        }
        IrExpr::Conditional {
            condition,
            then_branch,
            else_branch,
        } => {
            lower_defargs_in_expr(condition, table);
            lower_defargs_in_expr(then_branch, table);
            lower_defargs_in_expr(else_branch, table);
        }
        IrExpr::Assign { lhs, rhs } => {
            lower_defargs_in_expr(lhs, table);
            lower_defargs_in_expr(rhs, table);
        }
        IrExpr::InitializerList(elems) => {
            for e in elems {
                lower_defargs_in_expr(e, table);
            }
        }
        _ => {}
    }
}

/// If `arguments` fits exactly one same-name overload's trailing
/// defaults (and no exact-arity overload exists), returns the filled
/// argument list. Otherwise `None`.
fn fill_call(
    function: &str,
    arguments: &[IrExpr],
    table: &HashMap<String, Signatures>,
) -> Option<Vec<IrExpr>> {
    let overloads = table.get(function)?;
    // An exact-arity overload binds at DXC instead: never steal it.
    if overloads.iter().any(|params| params.len() == arguments.len()) {
        return None;
    }
    let mut winner: Option<&Vec<Option<IrExpr>>> = None;
    for params in overloads {
        if params.len() <= arguments.len() {
            continue;
        }
        if params[arguments.len()..].iter().all(|d| d.is_some()) {
            if winner.is_some() {
                return None;
            }
            winner = Some(params);
        }
    }
    let winner = winner?;
    let mut filled = arguments.to_vec();
    for d in &winner[arguments.len()..] {
        filled.push(d.clone()?);
    }
    Some(filled)
}
