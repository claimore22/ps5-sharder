//! Renaming of reserved PSSL identifiers for HLSL (`sampler` -> `sampler_`,
//! `texture` -> `texture_`).

use crate::ir::{IrExpr, IrStmt};

/// HLSL-reserved words PSSL allows as identifiers, with their renames.
/// `sampler` and `texture` are both HLSL keywords (legacy sampler/texture
/// types); a PSSL variable, resource, or parameter of either name must be
/// renamed at its declaration and at every use site to stay consistent.
pub(crate) fn reserved_hlsl_rename(name: &str) -> Option<&'static str> {
    if name.eq_ignore_ascii_case("sampler") {
        Some("sampler_")
    } else if name.eq_ignore_ascii_case("texture") {
        Some("texture_")
    } else {
        None
    }
}

/// Renames the root of a flattened member-call path (`texture.Sample` →
/// `texture_.Sample`). Only bare roots qualify: locals, globals, params,
/// and resources are renamed at declaration, while struct fields are not
/// (so dotted paths like `srt.texture` keep their spelling on both sides).
pub(crate) fn rename_member_root(path: &str) -> String {
    match path.split_once('.') {
        None => reserved_hlsl_rename(path).unwrap_or(path).to_string(),
        Some(_) => path.to_string(),
    }
}

pub(crate) fn rename_sampler_in_stmt(stmt: &mut IrStmt) {
    match stmt {
        IrStmt::Return(Some(expr)) => {
            rename_sampler_in_expr(expr);
        }

        IrStmt::VarDecl {
            init: Some(expr), ..
        } => {
            rename_sampler_in_expr(expr);
        }

        IrStmt::Expr(expr) => {
            rename_sampler_in_expr(expr);
        }

        IrStmt::Block(stmts) => {
            for s in stmts {
                rename_sampler_in_stmt(s);
            }
        }

        IrStmt::If {
            condition,
            then_branch,
            else_branch,
        } => {
            rename_sampler_in_expr(condition);

            for s in then_branch {
                rename_sampler_in_stmt(s);
            }

            if let Some(branch) = else_branch {
                for s in branch {
                    rename_sampler_in_stmt(s);
                }
            }
        }

        IrStmt::For {
            init,
            condition,
            update,
            body,
        } => {
            rename_sampler_in_stmt(init);

            if let Some(c) = condition {
                rename_sampler_in_expr(c);
            }

            if let Some(u) = update {
                rename_sampler_in_expr(u);
            }

            for s in body {
                rename_sampler_in_stmt(s);
            }
        }

        IrStmt::While { condition, body } => {
            rename_sampler_in_expr(condition);

            for s in body {
                rename_sampler_in_stmt(s);
            }
        }

        IrStmt::Switch { scrutinee, cases } => {
            rename_sampler_in_expr(scrutinee);

            for case in cases {
                if let Some(label) = &mut case.label {
                    rename_sampler_in_expr(label);
                }

                for s in &mut case.body {
                    rename_sampler_in_stmt(s);
                }
            }
        }

        _ => {}
    }
}

pub(crate) fn rename_sampler_in_expr(expr: &mut IrExpr) {
    match expr {
        IrExpr::Variable(name) => {
            if let Some(renamed) = reserved_hlsl_rename(name) {
                *name = renamed.to_string();
            }
        }

        IrExpr::Member { object, .. } => {
            rename_sampler_in_expr(object);
        }

        IrExpr::Index { object, index } => {
            rename_sampler_in_expr(object);
            rename_sampler_in_expr(index);
        }

        IrExpr::ResourceIndex { resource, index } => {
            rename_sampler_in_expr(resource);
            rename_sampler_in_expr(index);
        }

        IrExpr::ResourceLoad {
            resource,
            coordinates,
            sample,
        } => {
            rename_sampler_in_expr(resource);
            rename_sampler_in_expr(coordinates);
            if let Some(s) = sample {
                rename_sampler_in_expr(s);
            }
        }

        IrExpr::Call {
            function,
            arguments,
        } => {
            if let Some(renamed) = reserved_hlsl_rename(function) {
                *function = renamed.to_string();
            }

            for arg in arguments {
                rename_sampler_in_expr(arg);
            }
        }

        IrExpr::Construct { arguments, .. } => {
            for arg in arguments {
                rename_sampler_in_expr(arg);
            }
        }

        IrExpr::Unary { expr: inner, .. } => {
            rename_sampler_in_expr(inner);
        }

        IrExpr::Binary { left, right, .. } => {
            rename_sampler_in_expr(left);
            rename_sampler_in_expr(right);
        }

        IrExpr::Conditional {
            condition,
            then_branch,
            else_branch,
        } => {
            rename_sampler_in_expr(condition);
            rename_sampler_in_expr(then_branch);
            rename_sampler_in_expr(else_branch);
        }

        IrExpr::Assign { lhs, rhs } => {
            rename_sampler_in_expr(lhs);
            rename_sampler_in_expr(rhs);
        }

        IrExpr::InitializerList(elems) => {
            for e in elems {
                rename_sampler_in_expr(e);
            }
        }

        _ => {}
    }
}
