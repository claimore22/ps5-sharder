//! Function-level lowering.

use std::collections::HashMap;

use crate::ast::stmt::Stmt;
use crate::ast::Expr;
use crate::ast::Type;
use crate::error::Result;
use crate::ir::function::{Function, Parameter};
use crate::semantic::to_ir_type;

use super::rename::rename_sampler_in_stmt;
use super::stmt::lower_stmt;

/// Maps a `BuiltinSamplers` struct member to its externally-bound global.
/// Vulkan samplers are immutable and externally bound, so the
/// `getBuiltinSamplers()` factory (which builds undefined locals from
/// `sce::Gnm::Sampler` descriptors) lowers to these globals instead.
fn builtin_sampler_global(member: &str) -> Option<&'static str> {
    let m = member.to_ascii_lowercase();
    if m.starts_with("point") {
        Some("g_point_sampler")
    } else if m.starts_with("linear") || m.starts_with("bilinear") {
        Some("g_bilinear_sampler")
    } else if m.starts_with("aniso") || m.starts_with("trilinear") {
        Some("g_trilinear_sampler")
    } else {
        None
    }
}

/// Whether an expression is a `getBuiltinSamplers()` call (bare or scoped).
fn is_builtin_sampler_factory(e: &Expr) -> bool {
    match e {
        Expr::Call { function, arguments } => {
            if !arguments.is_empty() {
                return false;
            }
            match function.as_ref() {
                Expr::Variable(name) => name.eq_ignore_ascii_case("getBuiltinSamplers"),
                Expr::Member { member, .. } => member.eq_ignore_ascii_case("getBuiltinSamplers"),
                _ => false,
            }
        }
        _ => false,
    }
}

/// Rewrites `getBuiltinSamplers().<member>` to its global (`g_point_sampler`,
/// etc.) everywhere inside an expression. Returns the rewritten expression.
fn rewrite_expr_builtin_samplers(e: Expr) -> Expr {
    match e {
        Expr::Member { object, member } => {
            let object = Box::new(rewrite_expr_builtin_samplers(*object));
            if is_builtin_sampler_factory(&object) {
                if let Some(global) = builtin_sampler_global(&member) {
                    return Expr::Variable(global.to_string());
                }
            }
            Expr::Member { object, member }
        }
        Expr::Call { function, arguments } => Expr::Call {
            function: Box::new(rewrite_expr_builtin_samplers(*function)),
            arguments: arguments.into_iter().map(rewrite_expr_builtin_samplers).collect(),
        },
        Expr::Conditional { condition, then_branch, else_branch } => Expr::Conditional {
            condition: Box::new(rewrite_expr_builtin_samplers(*condition)),
            then_branch: Box::new(rewrite_expr_builtin_samplers(*then_branch)),
            else_branch: Box::new(rewrite_expr_builtin_samplers(*else_branch)),
        },
        Expr::Binary { left, op, right } => Expr::Binary {
            left: Box::new(rewrite_expr_builtin_samplers(*left)),
            op,
            right: Box::new(rewrite_expr_builtin_samplers(*right)),
        },
        Expr::Unary { op, expr } => Expr::Unary {
            op,
            expr: Box::new(rewrite_expr_builtin_samplers(*expr)),
        },
        Expr::Assign { lhs, rhs } => Expr::Assign {
            lhs: Box::new(rewrite_expr_builtin_samplers(*lhs)),
            rhs: Box::new(rewrite_expr_builtin_samplers(*rhs)),
        },
        Expr::Index { object, index } => Expr::Index {
            object: Box::new(rewrite_expr_builtin_samplers(*object)),
            index: Box::new(rewrite_expr_builtin_samplers(*index)),
        },
        Expr::Constructor { ty, arguments } => Expr::Constructor {
            ty,
            arguments: arguments.into_iter().map(rewrite_expr_builtin_samplers).collect(),
        },
        Expr::InitializerList(elems) => {
            Expr::InitializerList(elems.into_iter().map(rewrite_expr_builtin_samplers).collect())
        }
        Expr::Lambda { captures, params, body } => {
            let mut body = body;
            rewrite_stmts_builtin_samplers(&mut body);
            Expr::Lambda { captures, params, body }
        }
        other => other,
    }
}

fn rewrite_stmt_builtin_samplers(s: &mut Stmt) {
    match s {
        Stmt::Return(e) => {
            if let Some(e) = e {
                *e = rewrite_expr_builtin_samplers(std::mem::replace(e, Expr::Bool(true)));
            }
        }
        Stmt::If { condition, then_branch, else_branch } => {
            *condition = Box::new(rewrite_expr_builtin_samplers(std::mem::replace(
                condition.as_mut(),
                Expr::Bool(true),
            )));
            rewrite_stmts_builtin_samplers(then_branch);
            if let Some(els) = else_branch {
                rewrite_stmts_builtin_samplers(els);
            }
        }
        Stmt::Block(inner) => rewrite_stmts_builtin_samplers(inner),
        Stmt::For { init, condition, update, body } => {
            rewrite_stmt_builtin_samplers(init);
            if let Some(c) = condition {
                *c = rewrite_expr_builtin_samplers(std::mem::replace(c, Expr::Bool(true)));
            }
            if let Some(u) = update {
                *u = rewrite_expr_builtin_samplers(std::mem::replace(u, Expr::Bool(true)));
            }
            rewrite_stmts_builtin_samplers(body);
        }
        Stmt::While { condition, body } => {
            *condition = Box::new(rewrite_expr_builtin_samplers(std::mem::replace(
                condition.as_mut(),
                Expr::Bool(true),
            )));
            rewrite_stmts_builtin_samplers(body);
        }
        Stmt::Switch { scrutinee, cases } => {
            *scrutinee = Box::new(rewrite_expr_builtin_samplers(std::mem::replace(
                scrutinee.as_mut(),
                Expr::Bool(true),
            )));
            for c in cases {
                rewrite_stmts_builtin_samplers(&mut c.body);
            }
        }
        Stmt::VarDecl { init, .. } => {
            if let Some(e) = init {
                *e = rewrite_expr_builtin_samplers(std::mem::replace(e, Expr::Bool(true)));
            }
        }
        Stmt::Destructure { init, .. } => {
            if let Some(e) = init {
                *e = rewrite_expr_builtin_samplers(std::mem::replace(e, Expr::Bool(true)));
            }
        }
        Stmt::Expr(e) => {
            *e = rewrite_expr_builtin_samplers(std::mem::replace(e, Expr::Bool(true)));
        }
    }
}

fn rewrite_stmts_builtin_samplers(stmts: &mut Vec<Stmt>) {
    for s in stmts.iter_mut() {
        rewrite_stmt_builtin_samplers(s);
    }
}

/// Whether an expression mentions a sampler (heuristic over the AST shape:
/// a sampler global/variable, a `SamplerState` constructor, or the builtin
/// factory). Numeric ternaries never contain these, so a conditional whose
/// branch mentions one is a sampler select that HLSL cannot express as `?:`.
fn expr_mentions_sampler(e: &Expr) -> bool {
    match e {
        Expr::Variable(name) => {
            let l = name.to_ascii_lowercase();
            l.contains("sampler") || l == "g_point_sampler" || l == "g_bilinear_sampler" || l == "g_trilinear_sampler"
        }
        Expr::Member { object, member } => {
            let l = member.to_ascii_lowercase();
            l.contains("sampler")
                || l == "pointclamp" || l == "linearclamp"
                || expr_mentions_sampler(object)
        }
        Expr::Call { function, arguments } => {
            if is_builtin_sampler_factory(function) {
                return true;
            }
            if let Expr::Variable(name) = function.as_ref() {
                let l = name.to_ascii_lowercase();
                if l.contains("sampler") {
                    return true;
                }
            }
            expr_mentions_sampler(function) || arguments.iter().any(expr_mentions_sampler)
        }
        Expr::Constructor { ty, arguments } => {
            if ty.display().to_ascii_lowercase().contains("sampler") {
                return true;
            }
            arguments.iter().any(expr_mentions_sampler)
        }
        Expr::Conditional { condition, then_branch, else_branch } => {
            expr_mentions_sampler(condition)
                || expr_mentions_sampler(then_branch)
                || expr_mentions_sampler(else_branch)
        }
        Expr::Binary { left, right, .. } => {
            expr_mentions_sampler(left) || expr_mentions_sampler(right)
        }
        Expr::Unary { expr, .. } => expr_mentions_sampler(expr),
        Expr::Assign { lhs, rhs } => {
            expr_mentions_sampler(lhs) || expr_mentions_sampler(rhs)
        }
        Expr::Index { object, index } => {
            expr_mentions_sampler(object) || expr_mentions_sampler(index)
        }
        Expr::InitializerList(elems) => elems.iter().any(expr_mentions_sampler),
        Expr::Lambda { body, .. } => body.iter().any(stmt_mentions_sampler_var),
        _ => false,
    }
}

fn stmt_mentions_sampler_var(s: &Stmt) -> bool {
    match s {
        Stmt::VarDecl { init, .. } => init.as_ref().map(expr_mentions_sampler).unwrap_or(false),
        Stmt::Expr(e) | Stmt::Return(Some(e)) => expr_mentions_sampler(e),
        _ => false,
    }
}

/// Whether a call expression is a texture sample whose first argument is the
/// sampler variable `var` (e.g. `tex.Sample(samp, uv)`).
fn sample_call_with_sampler_var(e: &Expr, var: &str) -> bool {
    if let Expr::Call { function, arguments } = e {
        if let Expr::Member { member, .. } = function.as_ref() {
            let l = member.to_ascii_lowercase();
            let is_sample = matches!(
                l.as_str(),
                "sample" | "samplelevel" | "samplelod" | "samplebias" | "samplegrad"
                    | "samplecmp" | "samplecmplevelzero" | "gather" | "gatherred"
                    | "gathergreen" | "gatherblue" | "gatheralpha" | "gatherredcmp"
            );
            if !is_sample {
                return false;
            }
            return matches!(arguments.first(), Some(Expr::Variable(v)) if v == var);
        }
    }
    false
}

/// Builds `tex.Sample(then_branch, ...)` / `tex.Sample(else_branch, ...)`
/// from a sample call whose sampler arg is a variable. Returns `None` when the
/// shape is not a rewritable sample call.
fn split_sample_call(e: &Expr, then_s: Expr, else_s: Expr) -> Option<(Expr, Expr)> {
    if let Expr::Call { function, arguments } = e {
        if let Expr::Member { object, member } = function.as_ref() {
            let mut args_t = arguments.clone();
            let mut args_e = arguments.clone();
            if args_t.is_empty() {
                return None;
            }
            args_t[0] = then_s;
            args_e[0] = else_s;
            let mk = |args: Vec<Expr>| Expr::Call {
                function: Box::new(Expr::Member {
                    object: object.clone(),
                    member: member.clone(),
                }),
                arguments: args,
            };
            return Some((mk(args_t), mk(args_e)));
        }
    }
    None
}

// Placeholder to keep the match above total; replaced below.

/// Distributes a sampler ternary into its sample consumers:
/// `auto samp = c ? A : B; T out = tex.Sample(samp, ...);`
/// becomes
/// `T out; if (c) out = tex.Sample(A, ...); else out = tex.Sample(B, ...);`
/// HLSL/DXC has no value semantics for `SamplerState`, so a ternary over
/// samplers is illegal even when the samplers themselves are valid.
/// Counts uses of variable `var` in an expression (declarations excluded;
// lambda parameters shadowing `var` are treated as uses to stay
/// conservative — the distributor vetoes on any out-of-position mention).
fn expr_var_mentions(e: &Expr, var: &str) -> usize {
    match e {
        Expr::Variable(v) => usize::from(v == var),
        Expr::Member { object, .. } => expr_var_mentions(object, var),
        Expr::Index { object, index } => {
            expr_var_mentions(object, var) + expr_var_mentions(index, var)
        }
        Expr::Call { function, arguments } => {
            expr_var_mentions(function, var)
                + arguments.iter().map(|a| expr_var_mentions(a, var)).sum::<usize>()
        }
        Expr::Constructor { arguments, .. } => {
            arguments.iter().map(|a| expr_var_mentions(a, var)).sum()
        }
        Expr::Assign { lhs, rhs } => {
            expr_var_mentions(lhs, var) + expr_var_mentions(rhs, var)
        }
        Expr::Unary { expr, .. } => expr_var_mentions(expr, var),
        Expr::Binary { left, right, .. } => {
            expr_var_mentions(left, var) + expr_var_mentions(right, var)
        }
        Expr::Conditional { condition, then_branch, else_branch } => {
            expr_var_mentions(condition, var)
                + expr_var_mentions(then_branch, var)
                + expr_var_mentions(else_branch, var)
        }
        Expr::InitializerList(elems) => {
            elems.iter().map(|x| expr_var_mentions(x, var)).sum()
        }
        Expr::Lambda { params, body, .. } => {
            if params.iter().any(|p| p == var) {
                1
            } else {
                body.iter().map(|s| stmt_var_mentions(s, var)).sum()
            }
        }
        _ => 0,
    }
}

/// Counts uses of `var` in a statement, including nested blocks. A nested
/// redeclaration of the same name counts as a use (veto): it may shadow
/// the outer temporary.
fn stmt_var_mentions(s: &Stmt, var: &str) -> usize {
    match s {
        Stmt::Return(e) => e.as_ref().map(|e| expr_var_mentions(e, var)).unwrap_or(0),
        Stmt::If { condition, then_branch, else_branch } => {
            expr_var_mentions(condition, var)
                + then_branch.iter().map(|x| stmt_var_mentions(x, var)).sum::<usize>()
                + else_branch.as_ref().map(|els| els.iter().map(|x| stmt_var_mentions(x, var)).sum()).unwrap_or(0)
        }
        Stmt::Block(inner) => inner.iter().map(|x| stmt_var_mentions(x, var)).sum(),
        Stmt::For { init, condition, update, body } => {
            stmt_var_mentions(init, var)
                + condition.as_ref().map(|c| expr_var_mentions(c, var)).unwrap_or(0)
                + update.as_ref().map(|u| expr_var_mentions(u, var)).unwrap_or(0)
                + body.iter().map(|x| stmt_var_mentions(x, var)).sum::<usize>()
        }
        Stmt::While { condition, body } => {
            expr_var_mentions(condition, var)
                + body.iter().map(|x| stmt_var_mentions(x, var)).sum::<usize>()
        }
        Stmt::Switch { scrutinee, cases } => {
            expr_var_mentions(scrutinee, var)
                + cases.iter().map(|c| {
                    c.label.as_ref().map(|l| expr_var_mentions(l, var)).unwrap_or(0)
                        + c.body.iter().map(|x| stmt_var_mentions(x, var)).sum::<usize>()
                }).sum::<usize>()
        }
        Stmt::VarDecl { name, init, .. } => {
            usize::from(name == var)
                + init.as_ref().map(|e| expr_var_mentions(e, var)).unwrap_or(0)
        }
        Stmt::Destructure { names, init, .. } => {
            usize::from(names.iter().any(|n| n == var))
                + init.as_ref().map(|e| expr_var_mentions(e, var)).unwrap_or(0)
        }
        Stmt::Expr(e) => expr_var_mentions(e, var),
    }
}

fn distribute_sampler_select(stmts: &mut Vec<Stmt>) {
    // Recurse first so nested blocks are already normalized.
    for s in stmts.iter_mut() {
        match s {
            Stmt::Block(inner) => distribute_sampler_select(inner),
            Stmt::If { then_branch, else_branch, .. } => {
                distribute_sampler_select(then_branch);
                if let Some(els) = else_branch {
                    distribute_sampler_select(els);
                }
            }
            Stmt::For { body, .. } | Stmt::While { body, .. } => {
                distribute_sampler_select(body)
            }
            Stmt::Switch { cases, .. } => {
                for c in cases {
                    distribute_sampler_select(&mut c.body);
                }
            }
            _ => {}
        }
    }

    let mut i = 0;
    while i < stmts.len() {
        let (samp, cond, then_s, else_s) = match &stmts[i] {
            Stmt::VarDecl { name, init: Some(Expr::Conditional { condition, then_branch, else_branch }), .. } => {
                let is_select = expr_mentions_sampler(then_branch) || expr_mentions_sampler(else_branch);
                if !is_select {
                    i += 1;
                    continue;
                }
                (name.clone(), (**condition).clone(), (**then_branch).clone(), (**else_branch).clone())
            }
            _ => {
                i += 1;
                continue;
            }
        };
        // Find consumers of `samp` as a sampler arg in the rest of this block.
        // Only same-level statements whose ENTIRE use of `samp` is one
        // rewritable sample call qualify; `return samp`-shaped consumers
        // cannot branch a value (keep loud instead). Any other mention of
        // `samp` anywhere later in this block (nested scopes included, and
        // shadowing redeclarations counted as mentions) vetoes the whole
        // transform: removing the temporary would leave dangling uses.
        // (Nested-block sample consumers are left for the recursive pass
        // only when the temporary is declared in the same nested block;
        // an outer temporary used inside a nested block vetoes here.)
        let mut consumers: Vec<usize> = Vec::new();
        let mut veto = false;
        for (j, s) in stmts.iter().enumerate().skip(i + 1) {
            let total = stmt_var_mentions(s, &samp);
            if total == 0 {
                continue;
            }
            let rewritable = match s {
                Stmt::VarDecl { init: Some(e), .. } => {
                    sample_call_with_sampler_var(e, &samp) && expr_var_mentions(e, &samp) == 1
                }
                Stmt::Expr(Expr::Assign { rhs, .. }) => {
                    sample_call_with_sampler_var(rhs, &samp) && expr_var_mentions(rhs, &samp) == 1
                }
                Stmt::Expr(e) => {
                    sample_call_with_sampler_var(e, &samp) && expr_var_mentions(e, &samp) == 1
                }
                _ => false,
            };
            if rewritable && total == 1 {
                consumers.push(j);
            } else {
                veto = true;
                break;
            }
        }
        if veto || consumers.is_empty() {
            i += 1;
            continue;
        }
        // Remove the sampler temporary; rewrite each consumer in place.
        stmts.remove(i);
        // Rebuild: walk forward, replacing consumers one by one with offset tracking.
        let mut offset: isize = -1; // removal shifted everything after i down by 1
        for &orig_j in &consumers {
            let idx = (orig_j as isize + offset) as usize;
            let replacement: Stmt = match stmts[idx].clone() {
                Stmt::VarDecl { storage, ty, name, array_size, init: Some(e) } => {
                    let (t_call, e_call) = split_sample_call(&e, then_s.clone(), else_s.clone())
                        .expect("guarded by sample_call_with_sampler_var");
                    let decl = Stmt::VarDecl { storage, ty, name: name.clone(), array_size, init: None };
                    let assign_t = Stmt::Expr(Expr::Assign {
                        lhs: Box::new(Expr::Variable(name.clone())),
                        rhs: Box::new(t_call),
                    });
                    let assign_e = Stmt::Expr(Expr::Assign {
                        lhs: Box::new(Expr::Variable(name.clone())),
                        rhs: Box::new(e_call),
                    });
                    // Insert decl + if; replace current slot with decl then insert if.
                    stmts[idx] = decl;
                    stmts.insert(
                        idx + 1,
                        Stmt::If {
                            condition: Box::new(cond.clone()),
                            then_branch: vec![assign_t],
                            else_branch: Some(vec![assign_e]),
                        },
                    );
                    offset += 1;
                    i += 1;
                    continue;
                }
                Stmt::Expr(Expr::Assign { lhs, rhs }) => {
                    let (t_call, e_call) = split_sample_call(&rhs, then_s.clone(), else_s.clone())
                        .expect("guarded by sample_call_with_sampler_var");
                    Stmt::If {
                        condition: Box::new(cond.clone()),
                        then_branch: vec![Stmt::Expr(Expr::Assign { lhs: lhs.clone(), rhs: Box::new(t_call) })],
                        else_branch: Some(vec![Stmt::Expr(Expr::Assign { lhs: lhs.clone(), rhs: Box::new(e_call) })]),
                    }
                }
                Stmt::Expr(e) => {
                    // Bare sample call statement (result discarded): branch both calls.
                    if let Some((t_call, e_call)) = split_sample_call(&e, then_s.clone(), else_s.clone()) {
                        Stmt::If {
                            condition: Box::new(cond.clone()),
                            then_branch: vec![Stmt::Expr(t_call)],
                            else_branch: Some(vec![Stmt::Expr(e_call)]),
                        }
                    } else {
                        continue;
                    }
                }
                Stmt::Return(_) => {
                    // `return tex.Sample(samp, ...)`: cannot branch void; keep loud.
                    continue;
                }
                _ => continue,
            };
            stmts[idx] = replacement;
            // `If` replaces 1-for-1 except the VarDecl case above (handled).
        }
        // Continue scanning after the rewritten region.
        i += 1;
    }
}

/// Distributes struct ternaries (`S x = c ? A : B`, `x = c ? A : B`,
/// `return c ? A : B` with struct-typed operands) into if/else assignment.
/// HLSL `?:` requires numeric scalar/vector/matrix operands; PSSL headers
/// select whole structs (`ViewDataSRT viewData = eyeIndex == 1 ? __srtarg_0
/// : __srtarg_1`, `const ViewDataSRT *viewData = c ? pSrt->a : pSrt->b`
/// with pointers desugared away). Runs on the AST with the function's
/// type map so struct-typed branches resolve; repeats to fixpoint for
/// nested ternaries. `static` temporaries are skipped (deferred
/// assignment would run every call instead of once); `const` rewrites
/// with the qualifier dropped (every path assigns exactly once before
/// any use, so values are identical). Anything else keeps today's loud
/// path.
fn distribute_struct_select(
    stmts: &mut Vec<Stmt>,
    var_types: &HashMap<String, crate::ast::Type>,
    struct_fields: &HashMap<String, HashMap<String, crate::ast::Type>>,
) {
    /// Dotted path for variables and member chains (`srt.tx`), if pathable.
    fn expr_path(e: &Expr) -> Option<String> {
        match e {
            Expr::Variable(v) => Some(v.clone()),
            Expr::Member { object, member } => {
                expr_path(object).map(|p| format!("{p}.{member}"))
            }
            _ => None,
        }
    }
    /// Whether an expression has a struct type (declared struct name).
    fn is_struct_expr(
        e: &Expr,
        var_types: &HashMap<String, crate::ast::Type>,
        struct_fields: &HashMap<String, HashMap<String, crate::ast::Type>>,
    ) -> bool {
        match e {
            Expr::Constructor { ty: crate::ast::Type::Named(s), .. } => {
                struct_fields.contains_key(s.as_str())
            }
            Expr::Conditional { then_branch, else_branch, .. } => {
                is_struct_expr(then_branch, var_types, struct_fields)
                    || is_struct_expr(else_branch, var_types, struct_fields)
            }
            _ => expr_path(e).is_some_and(|p| {
                matches!(var_types.get(&p), Some(crate::ast::Type::Named(s)) if struct_fields.contains_key(s.as_str()))
            }),
        }
    }
    fn is_struct_ty(
        ty: &crate::ast::Type,
        struct_fields: &HashMap<String, HashMap<String, crate::ast::Type>>,
    ) -> bool {
        matches!(ty, crate::ast::Type::Named(s) if struct_fields.contains_key(s.as_str()))
    }
    /// `const`/`static` declared temporaries: `static` is skipped
    /// (deferred assignment would run every call instead of once,
    /// changing persistent-state semantics); `const` is rewritten with
    /// the qualifier dropped (every path assigns exactly once before any
    /// use, so values are identical, and HLSL requires `const` locals to
    /// initialize at declaration).
    fn storage_rewritable(storage: &Option<crate::ast::StorageClass>) -> bool {
        !matches!(storage, Some(crate::ast::StorageClass::Static))
    }
    /// Rewrites one statement; returns the replacement list and whether
    /// anything changed.
    fn rewrite_one(
        s: Stmt,
        var_types: &HashMap<String, crate::ast::Type>,
        struct_fields: &HashMap<String, HashMap<String, crate::ast::Type>>,
    ) -> (Vec<Stmt>, bool) {
        match s {
            Stmt::VarDecl { storage, ty, name, array_size, init: Some(Expr::Conditional { condition, then_branch, else_branch }) }
                if storage_rewritable(&storage)
                    && array_size.is_none()
                    && (is_struct_ty(&ty, struct_fields)
                        || is_struct_expr(&then_branch, var_types, struct_fields)
                        || is_struct_expr(&else_branch, var_types, struct_fields)) =>
            {
                let decl = Stmt::VarDecl { storage: None, ty, name: name.clone(), array_size, init: None };
                let assign_t = Stmt::Expr(Expr::Assign {
                    lhs: Box::new(Expr::Variable(name.clone())),
                    rhs: then_branch,
                });
                let assign_e = Stmt::Expr(Expr::Assign {
                    lhs: Box::new(Expr::Variable(name)),
                    rhs: else_branch,
                });
                (
                    vec![
                        decl,
                        Stmt::If {
                            condition,
                            then_branch: vec![assign_t],
                            else_branch: Some(vec![assign_e]),
                        },
                    ],
                    true,
                )
            }
            Stmt::Expr(Expr::Assign { lhs, rhs }) => {
                if let Expr::Conditional { condition, then_branch, else_branch } = *rhs {
                    let lhs_struct = expr_path(&lhs).is_some_and(|p| {
                        matches!(var_types.get(&p), Some(crate::ast::Type::Named(s)) if struct_fields.contains_key(s.as_str()))
                    });
                    if lhs_struct
                        || is_struct_expr(&then_branch, var_types, struct_fields)
                        || is_struct_expr(&else_branch, var_types, struct_fields)
                    {
                        return (
                            vec![Stmt::If {
                                condition,
                                then_branch: vec![Stmt::Expr(Expr::Assign { lhs: lhs.clone(), rhs: then_branch })],
                                else_branch: Some(vec![Stmt::Expr(Expr::Assign { lhs, rhs: else_branch })]),
                            }],
                            true,
                        );
                    }
                    (
                        vec![Stmt::Expr(Expr::Assign {
                            lhs,
                            rhs: Box::new(Expr::Conditional { condition, then_branch, else_branch }),
                        })],
                        false,
                    )
                } else {
                    (vec![Stmt::Expr(Expr::Assign { lhs, rhs })], false)
                }
            }
            Stmt::Return(Some(Expr::Conditional { condition, then_branch, else_branch }))
                if is_struct_expr(&then_branch, var_types, struct_fields)
                    || is_struct_expr(&else_branch, var_types, struct_fields) =>
            {
                (
                    vec![Stmt::If {
                        condition,
                        then_branch: vec![Stmt::Return(Some(*then_branch))],
                        else_branch: Some(vec![Stmt::Return(Some(*else_branch))]),
                    }],
                    true,
                )
            }
            other => (vec![other], false),
        }
    }
    fn pass_list(
        stmts: &mut Vec<Stmt>,
        var_types: &HashMap<String, crate::ast::Type>,
        struct_fields: &HashMap<String, HashMap<String, crate::ast::Type>>,
    ) -> bool {
        // Recurse into nested statement lists first.
        for s in stmts.iter_mut() {
            match s {
                Stmt::Block(inner) => {
                    pass_list(inner, var_types, struct_fields);
                }
                Stmt::If { then_branch, else_branch, .. } => {
                    pass_list(then_branch, var_types, struct_fields);
                    if let Some(els) = else_branch {
                        pass_list(els, var_types, struct_fields);
                    }
                }
                Stmt::For { body, .. } | Stmt::While { body, .. } => {
                    pass_list(body, var_types, struct_fields);
                }
                Stmt::Switch { cases, .. } => {
                    for c in cases {
                        pass_list(&mut c.body, var_types, struct_fields);
                    }
                }
                _ => {}
            }
        }
        // Rewrite current level (fixpoint handled by the caller for
        // nested ternaries surfacing inside fresh If branches).
        let mut changed = false;
        let mut i = 0;
        while i < stmts.len() {
            let taken = std::mem::replace(&mut stmts[i], Stmt::Block(Vec::new()));
            let (mut repl, hit) = rewrite_one(taken, var_types, struct_fields);
            changed |= hit;
            if repl.len() == 1 && !hit {
                stmts[i] = repl.pop().unwrap();
                i += 1;
            } else {
                stmts.splice(i..=i, repl);
                i += 1;
                // Visit the spliced If body next pass (outer fixpoint).
            }
        }
        changed
    }
    let mut guard = 0;
    while pass_list(stmts, var_types, struct_fields) {
        guard += 1;
        if guard > 32 {
            break;
        }
    }
}

/// Registers a variable and, for struct types, its dotted member entries
/// (`psIn.uv`) in `map`. First registration wins (shadowing); depth caps
/// recursive structs. Shared by parameter/global seeding and local
/// collection.
pub(crate) fn record_var_type(
    map: &mut HashMap<String, crate::ast::Type>,
    structs: &HashMap<String, HashMap<String, crate::ast::Type>>,
    name: &str,
    ty: &crate::ast::Type,
) {
    fn inner(
        map: &mut HashMap<String, crate::ast::Type>,
        structs: &HashMap<String, HashMap<String, crate::ast::Type>>,
        name: &str,
        ty: &crate::ast::Type,
        depth: usize,
    ) {
        if map.contains_key(name) {
            return;
        }
        map.insert(name.to_string(), ty.clone());
        if depth >= 4 {
            return;
        }
        if let crate::ast::Type::Named(s) = ty {
            if let Some(fields) = structs.get(s) {
                for (fname, fty) in fields {
                    let dotted = format!("{name}.{fname}");
                    if !map.contains_key(&dotted) {
                        map.insert(dotted.clone(), fty.clone());
                        inner(map, structs, &dotted, fty, depth + 1);
                    }
                }
            }
        }
    }
    inner(map, structs, name, ty, 0);
}

pub(crate) fn lower_function(
    function: &crate::ast::Function,
    resource_fields: &HashMap<(String, String), Type>,
    var_struct: &HashMap<String, String>,
    constants: &HashMap<String, crate::ast::Expr>,
    global_types: &HashMap<String, crate::ast::Type>,
    struct_fields: &HashMap<String, HashMap<String, crate::ast::Type>>,
    is_entry: bool,
) -> Result<Function> {
    // Depth-compare-configured locals/params select `SamplerComparisonState`
    // (see `retype_comparison_samplers`); work on the retyped clone so the
    // parameter/variable-type maps below agree with declarations.
    let mut owned = function.clone();
    super::stmt::retype_comparison_samplers(&mut owned);
    // Sampler factories lower to externally-bound globals, and sampler
    // ternaries (illegal in HLSL) distribute into their sample consumers.
    // Runs on the AST before type maps are built so `auto samp` never
    // reaches the emitter as a `SamplerState ?: ` select.
    rewrite_stmts_builtin_samplers(&mut owned.body);
    distribute_sampler_select(&mut owned.body);
    // Build a per-function variable -> type map so lowering can distinguish
    // PSSL scalar-from-vector casts. This lets `(float)vec` (where `vec` is a
    // vector) lower to `float(vec.x)`, while `float(scalar)` stays unchanged.
    // Built here (before struct-ternary distribution) so branch struct
    // types resolve.
    let mut var_types: HashMap<String, crate::ast::Type> = HashMap::new();

    for p in &owned.parameters {
        record_var_type(&mut var_types, &struct_fields, &p.name, &p.ty);
    }
    collect_stmt_var_types(&owned.body, &mut var_types, &struct_fields);
    // Module globals seed last so params/locals keep their existing
    // precedence. Besides grounding builtin-constant fallback (an unknown
    // `PI` is the PSSL pi only when no variable of that name exists),
    // this sharpens cast resolution for global vectors.
    for (name, ty) in global_types {
        record_var_type(&mut var_types, &struct_fields, name, ty);
    }
    // Struct ternaries (`S x = c ? A : B`) are illegal in HLSL (no struct
    // `?:` operands) — distribute into if/else assignment. Runs on the
    // AST with the type map above so struct-typed branches are known.
    distribute_struct_select(&mut owned.body, &var_types, struct_fields);
    let function = &owned;
    // Parameter types for lowering C++ default arguments (which may
    // reference earlier parameters).
    let param_env: HashMap<String, crate::ast::Type> = function
        .parameters
        .iter()
        .map(|p| (p.name.clone(), p.ty.clone()))
        .collect();
    let mut parameters: Vec<Parameter> = function
        .parameters
        .iter()
        .filter(|p| {
            // Only the ENTRY imports its SRT table from the host (the
            // param becomes flattened globals); helpers keep their
            // `S_SRT_DATA`-tagged params so calls keep matching, and
            // `flatten_srt`'s helper elision rewrites their bodies and
            // call sites uniformly. Dropping helper params here orphaned
            // every call that still passes the argument (meshlet
            // `main_meshlets` arity mismatch with no diagnostic).
            if !is_entry {
                return true;
            }
            p.semantic
                .as_ref()
                .map(|s| !s.name().eq_ignore_ascii_case("S_SRT_DATA"))
                .unwrap_or(true)
        })
        .map(|p| {
            // Keep the parameter declaration consistent with the identifier
            // rewriting performed by `rename_sampler_in_stmt`. PSSL may use
            // HLSL-reserved words (`sampler`, `texture`) as names, so rename
            // them exactly as resources and body use-sites are renamed.
            let name = if let Some(renamed) = super::rename::reserved_hlsl_rename(&p.name) {
                tracing::trace!(
                    target: "pssl_shader::lower::sampler",
                    parameter = %p.name,
                    renamed = renamed,
                    "renaming reserved parameter"
                );
                renamed.to_string()
            } else {
                p.name.clone()
            };

            Parameter {
                ty: to_ir_type(&p.ty),
                name,
                semantic: p.semantic.as_ref().map(|s| s.name().to_string()),
                array_size: p.array_size,
                direction: p.direction.map(|d| d.as_str().to_string()),
                gs_primitive: p.gs_primitive.clone(),
                default: p.default.as_ref().map(|d| {
                    super::expr::lower_expr(d, &param_env, resource_fields, var_struct)
                }),
                is_uniform: p.is_uniform,
                mesh_object: None,
            }
        })
        .collect();

    let mut body = Vec::new();

    for statement in &function.body {
        body.push(lower_stmt(statement, &var_types, resource_fields, var_struct)?);
    }

    // Rewrite `sampler` -> `sampler_` to avoid HLSL keyword clash.
    for stmt in &mut body {
        rename_sampler_in_stmt(stmt);
    }

    let num_threads = extract_num_threads(&function.attributes, constants);

    Ok(Function {
        name: function.name.clone(),
        return_type: to_ir_type(&function.return_type),
        return_semantic: function
            .return_semantic
            .as_ref()
            .map(|s| s.name().to_string()),
        parameters,
        body,
        num_threads,
        scope: function.scope.clone(),
        max_vertex_count: extract_max_vertex_count(&function.attributes, constants),
        output_topology: extract_output_topology(&function.attributes),
        mesh_vert_count: extract_mesh_count(&function.attributes, constants, "num_verts"),
        mesh_prim_count: extract_mesh_count(&function.attributes, constants, "num_prims"),
        hull_domain: extract_hull_keyword(&function.attributes, "DOMAIN_PATCH_TYPE", &["quad", "tri", "isoline"]),
        hull_partitioning: extract_hull_keyword(
            &function.attributes,
            "PARTITIONING_TYPE",
            &["integer", "pow2", "fractional_odd", "fractional_even"],
        ),
        hull_control_points: extract_hull_count(&function.attributes, constants, "OUTPUT_CONTROL_POINTS"),
        hull_patch_func: extract_hull_string(&function.attributes, "PATCH_CONSTANT_FUNC"),
        hull_max_tess: extract_hull_count(&function.attributes, constants, "MAX_TESS_FACTOR"),
    })
}

/// Extracts `[output_primitive_type("tri")]` (mesh shaders) to an HLSL
/// `[outputtopology("triangle")]` kind. Only mesh output topologies
/// (`tri`/`triangle` → `triangle`, `point` → `point`, `line` → `line`)
/// map; adjacency/patch/unknown spellings stay `None` (loud at DXC if a
/// mesh entry lacks topology). Case-insensitive; quotes stripped by the
/// attribute parser (bare and quoted spellings both match).
pub(crate) fn extract_output_topology(
    attributes: &[crate::ast::Attribute],
) -> Option<String> {
    for attr in attributes {
        if attr.name.eq_ignore_ascii_case("output_primitive_type") && attr.arguments.len() == 1 {
            let kind = match &attr.arguments[0] {
                crate::ast::Expr::String(s) => s.to_ascii_lowercase(),
                crate::ast::Expr::Variable(v) => v.to_ascii_lowercase(),
                _ => continue,
            };
            // Strip surrounding quotes if the parser kept them.
            let kind = kind.trim_matches('"').trim();
            match kind {
                "tri" | "triangle" | "triangles" => return Some("triangle".to_string()),
                "point" | "points" => return Some("point".to_string()),
                "line" | "lines" => return Some("line".to_string()),
                _ => continue,
            }
        }
        // Hull `[OUTPUT_TOPOLOGY_TYPE("triangle_cw")]`: HLSL winding is
        // part of the topology (`triangle_cw`/`triangle_ccw`); a bare
        // `triangle` has no winding and stays `None` (loud) rather than
        // guessed.
        if attr.name.eq_ignore_ascii_case("output_topology_type") && attr.arguments.len() == 1 {
            let kind = match &attr.arguments[0] {
                crate::ast::Expr::String(s) => s.to_ascii_lowercase(),
                crate::ast::Expr::Variable(v) => v.to_ascii_lowercase(),
                _ => continue,
            };
            let kind = kind.trim_matches('"').trim();
            if kind.contains("ccw") {
                return Some("triangle_ccw".to_string());
            }
            if kind.contains("cw") {
                return Some("triangle_cw".to_string());
            }
            match kind {
                "point" | "points" => return Some("point".to_string()),
                "line" | "lines" => return Some("line".to_string()),
                _ => continue,
            }
        }
    }

    None
}

pub(crate) fn extract_num_threads(
    attributes: &[crate::ast::Attribute],
    constants: &HashMap<String, crate::ast::Expr>,
) -> Option<(u32, u32, u32)> {
    for attr in attributes {
        if attr.name.eq_ignore_ascii_case("NUM_THREADS") && attr.arguments.len() == 3 {
            let x = eval_uint_resolved(&attr.arguments[0], constants)?;
            let y = eval_uint_resolved(&attr.arguments[1], constants)?;
            let z = eval_uint_resolved(&attr.arguments[2], constants)?;

            return Some((x, y, z));
        }
    }

    None
}

/// Extracts `[MAX_VERTEX_COUNT(N)]` (geometry shaders) to a literal bound.
/// Unresolvable expressions (scoped consts like `ParticleInstance::kMax`)
/// stay `None`, leaving the attribute unmapped and loud at DXC.
pub(crate) fn extract_max_vertex_count(    attributes: &[crate::ast::Attribute],
    constants: &HashMap<String, crate::ast::Expr>,
) -> Option<u32> {
    for attr in attributes {
        if attr.name.eq_ignore_ascii_case("MAX_VERTEX_COUNT") && attr.arguments.len() == 1 {
            if let Some(n) = eval_uint_resolved(&attr.arguments[0], constants) {
                return Some(n);
            }
        }
    }

    None
}

/// Extracts mesh-shader `[num_verts(N)]` / `[num_prims(N)]` bounds to a
/// literal (max vertices/primitives the entry can export). Same
/// single-constant-argument shape as `MAX_VERTEX_COUNT`; unresolvable
/// expressions stay `None` (the mesh lowering falls back to
/// `numthreads.x`, then declines loud).
pub(crate) fn extract_mesh_count(
    attributes: &[crate::ast::Attribute],
    constants: &HashMap<String, crate::ast::Expr>,
    name: &str,
) -> Option<u32> {
    for attr in attributes {
        if attr.name.eq_ignore_ascii_case(name) && attr.arguments.len() == 1 {
            if let Some(n) = eval_uint_resolved(&attr.arguments[0], constants) {
                return Some(n);
            }
        }
    }

    None
}

/// Extracts a hull-shader keyword attribute (`[DOMAIN_PATCH_TYPE("quad")]`)
/// to its lowercase spelling when it names one of `allowed` (quotes
/// stripped, case-insensitive); anything else stays `None` (loud at DXC).
pub(crate) fn extract_hull_keyword(
    attributes: &[crate::ast::Attribute],
    name: &str,
    allowed: &[&str],
) -> Option<String> {
    for attr in attributes {
        if attr.name.eq_ignore_ascii_case(name) && attr.arguments.len() == 1 {
            let kind = match &attr.arguments[0] {
                crate::ast::Expr::String(s) => s.to_ascii_lowercase(),
                crate::ast::Expr::Variable(v) => v.to_ascii_lowercase(),
                _ => continue,
            };
            let kind = kind.trim_matches('"').trim();
            if allowed.iter().any(|a| kind == *a) {
                return Some(kind.to_string());
            }
        }
    }

    None
}

/// Extracts a hull-shader single-string attribute
/// (`[PATCH_CONSTANT_FUNC("ConstantsHS")]`) verbatim (quotes stripped).
pub(crate) fn extract_hull_string(
    attributes: &[crate::ast::Attribute],
    name: &str,
) -> Option<String> {
    for attr in attributes {
        if attr.name.eq_ignore_ascii_case(name) && attr.arguments.len() == 1 {
            match &attr.arguments[0] {
                crate::ast::Expr::String(s) => {
                    return Some(s.trim_matches('"').to_string());
                }
                crate::ast::Expr::Variable(v) => {
                    return Some(v.clone());
                }
                _ => continue,
            }
        }
    }

    None
}

/// Extracts a hull-shader single-count attribute
/// (`[OUTPUT_CONTROL_POINTS(1)]`, `[MAX_TESS_FACTOR(64)]`) to a literal;
/// unresolvable expressions stay `None` (loud at DXC).
pub(crate) fn extract_hull_count(
    attributes: &[crate::ast::Attribute],
    constants: &HashMap<String, crate::ast::Expr>,
    name: &str,
) -> Option<u32> {
    for attr in attributes {
        if attr.name.eq_ignore_ascii_case(name) && attr.arguments.len() == 1 {
            if let Some(n) = eval_uint_resolved(&attr.arguments[0], constants) {
                return Some(n);
            }
        }
    }

    None
}

/// Resolves a recorded constant `Expr` to a literal string, following chains of
/// named constants (e.g. `A = B; B = 8;`). Returns `None` for non-literal
/// expressions such as struct initializers or function calls.
pub(crate) fn resolve_const_literal(
    constants: &HashMap<String, crate::ast::Expr>,
    expr: &crate::ast::Expr,
) -> Option<String> {
    match expr {
        crate::ast::Expr::Number(s) => Some(s.clone()),
        crate::ast::Expr::Bool(b) => Some(if *b { "true" } else { "false" }.to_string()),
        crate::ast::Expr::Variable(name) => {
            resolve_const_literal(constants, constants.get(name)?)
        }
        crate::ast::Expr::Constructor { ty, arguments } => {
            // `ushort2(4,16)` etc. – emit as HLSL constructor with resolved args.
            // Preserve the constructor name's HLSL spelling (e.g. `ushort2` → `uint2` via lower,
            // but keep literal form here; hlsl_type in emit will normalize the global's type).
            let args: Vec<String> = arguments
                .iter()
                .filter_map(|a| resolve_const_literal(constants, a))
                .collect();
            if args.len() == arguments.len() {
                return Some(format!("{}({})", ty.display(), args.join(", ")));
            }
            None
        }
        crate::ast::Expr::Binary { left, op, right } => {
            // Integer const-folding for globals (`kBucketNum = 1 << kBits`):
            // previously only literals/chains/constructors resolved, so a
            // shift-based size emitted an uninitialized global and its
            // array decayed to `[]` (DXC `needs an explicit size`). Only
            // integer shapes fold (floats keep today's loud path).
            if let (Some(a), Some(b)) = (
                eval_uint_resolved(left, constants),
                eval_uint_resolved(right, constants),
            ) {
                use crate::ast::BinaryOp::*;
                let v = match op {
                    Add => a.checked_add(b),
                    Sub => a.checked_sub(b),
                    Mul => a.checked_mul(b),
                    Div if b != 0 => a.checked_div(b),
                    Mod if b != 0 => a.checked_rem(b),
                    ShiftLeft => a.checked_shl(b),
                    ShiftRight => a.checked_shr(b),
                    BitAnd => Some(a & b),
                    BitOr => Some(a | b),
                    BitXor => Some(a ^ b),
                    _ => None,
                };
                if let Some(v) = v {
                    return Some(v.to_string());
                }
            }
            None
        }
        _ => None,
    }
}

/// Evaluates an attribute integer, dereferencing named constants such as
/// `k_resolve_group_size` (which may themselves reference other constants).
/// Also handles `k_wavefront_size.x` / `.y` where `k_wavefront_size` is
/// `ushort2(4,16)` etc. – the individual components are extracted from the
/// constructor's arguments.
pub(crate) fn eval_uint_resolved(
    expr: &crate::ast::Expr,
    constants: &HashMap<String, crate::ast::Expr>,
) -> Option<u32> {
    match expr {
        crate::ast::Expr::Number(s) => {
            let t = s.trim_end_matches(|c| c == 'f' || c == 'F' || c == 'h' || c == 'H');
            if t.contains('.') || t.contains('e') || t.contains('E') {
                return None;
            }
            t.parse::<u32>().ok()
        }
        crate::ast::Expr::Variable(name) => {
            // Exact match first (bare globals, qualified struct statics).
            if let Some(v) = constants.get(name) {
                return eval_uint_resolved(v, constants);
            }
            // Namespace-erased fallback (`sce::Got::Blnd::Gpu::k` → `k`):
            // the parser strips namespaces to bare names, so a qualified
            // scope-resolution spelling denotes the bare recorded constant.
            // Only fires on `::` names (never dot-accesses); unrecorded
            // stays loud. Assumes no cross-namespace collisions (same
            // assumption as the parser's bare recording).
            if name.contains("::") {
                if let Some(bare) = name.rsplit("::").next() {
                    if let Some(v) = constants.get(bare) {
                        // Clone to avoid borrowing across the call.
                        let v = v.clone();
                        return eval_uint_resolved(&v, constants);
                    }
                }
            }
            return None;
        }
        crate::ast::Expr::Member { object, member } => {
            // `k_wavefront_size.x` → look up `k_wavefront_size` constant
            if let crate::ast::Expr::Variable(var) = object.as_ref() {
                if let Some(const_expr) = constants.get(var) {
                    // Expect `ushort2(4,16)` etc. – Constructor with 2 args
                    if let crate::ast::Expr::Constructor { arguments, .. } = const_expr {
                        let idx = match member.to_ascii_lowercase().as_str() {
                            "x" | "r" => 0,
                            "y" | "g" => 1,
                            "z" | "b" => 2,
                            "w" | "a" => 3,
                            _ => return None,
                        };
                        if idx < arguments.len() {
                            return eval_uint_resolved(&arguments[idx], constants);
                        }
                    }
                    // Fallback: if const is not a constructor, try member as variable?
                    // e.g. `k_size.x` where `k_size` is `int` – not valid, but handle.
                    tracing::trace!(
                        target: "pssl_shader::lower::numthreads",
                        var = %var,
                        member = %member,
                        "failed to resolve member from non-constructor constant"
                    );
                }
                // Qualified struct static (`ParticleInstance::kMax`):
                // recorded under `S::m` during struct parsing.
                if let Some(q) = constants.get(&format!("{var}::{member}")) {
                    // Clone to avoid borrowing `constants` across the call.
                    let q = q.clone();
                    return eval_uint_resolved(&q, constants);
                }
            }
            None
        }
        crate::ast::Expr::Constructor { arguments, .. } => {
            // Single-arg constructor wrapping a number: `ushort2(4,16)` already handled above,
            // but ` (ushort2)4` etc. – just handle first arg.
            if arguments.len() == 1 {
                return eval_uint_resolved(&arguments[0], constants);
            }
            None
        }
        crate::ast::Expr::Binary { left, op, right } => {
            // Const-arithmetic bounds (`MAX_VERTEX_COUNT(NUM_VERTS * 3)`).
            let a = eval_uint_resolved(left, constants)?;
            let b = eval_uint_resolved(right, constants)?;
            use crate::ast::BinaryOp::*;
            match op {
                Add => a.checked_add(b),
                Sub => a.checked_sub(b),
                Mul => a.checked_mul(b),
                Div if b != 0 => a.checked_div(b),
                Mod if b != 0 => a.checked_rem(b),
                ShiftLeft => a.checked_shl(b),
                ShiftRight => a.checked_shr(b),
                BitAnd => Some(a & b),
                BitOr => Some(a | b),
                BitXor => Some(a ^ b),
                _ => None,
            }
        }
        crate::ast::Expr::Unary {
            op: crate::ast::UnaryOp::Minus,
            expr,
        } => {
            let v = eval_uint_resolved(expr, constants)?;
            (v == 0).then_some(0)
        }
        crate::ast::Expr::Conditional { condition, then_branch, else_branch } => {
            // Constant ternary (`((64 > 64) ? 64 : 64)` from macro-expanded
            // mesh `num_threads` bounds): evaluate a literal comparison
            // test, take the selected branch. Unconditionally sound for
            // literal tests (no variables/calls/side effects involved).
            // Anything non-literal declines (stays loud).
            let test = match condition.as_ref() {
                crate::ast::Expr::Binary { left, op, right } => {
                    let (a, b) = (
                        eval_uint_resolved(left, constants)?,
                        eval_uint_resolved(right, constants)?,
                    );
                    use crate::ast::BinaryOp::*;
                    match op {
                        Equal => a == b,
                        NotEqual => a != b,
                        Less => a < b,
                        LessEqual => a <= b,
                        Greater => a > b,
                        GreaterEqual => a >= b,
                        _ => return None,
                    }
                }
                crate::ast::Expr::Bool(b) => *b,
                _ => return None,
            };
            if test {
                eval_uint_resolved(then_branch, constants)
            } else {
                eval_uint_resolved(else_branch, constants)
            }
        }
        _ => None,
    }
}

/// Walk statements collecting every local variable declaration into `types`,
/// mapping variable name -> declared AST type. Used so lowering can tell a
/// PSSL scalar-from-vector cast (`(float)vec`) apart from a scalar identity
/// cast (`float(scalar)`). Struct-typed locals also register dotted member
/// entries (see `record_var_type` above).
pub(crate) fn collect_stmt_var_types(
    stmts: &[crate::ast::Stmt],
    types: &mut HashMap<String, crate::ast::Type>,
    structs: &HashMap<String, HashMap<String, crate::ast::Type>>,
) {
    for stmt in stmts {
        match stmt {
            crate::ast::Stmt::VarDecl {
                storage: _,
                ty,
                name,
                array_size: _,
                init: _,
            } => {
                record_var_type(types, structs, name, ty);
            }
            crate::ast::Stmt::Block(inner) => collect_stmt_var_types(inner, types, structs),
            crate::ast::Stmt::If {
                then_branch,
                else_branch,
                ..
            } => {
                collect_stmt_var_types(then_branch, types, structs);
                if let Some(branch) = else_branch {
                    collect_stmt_var_types(branch, types, structs);
                }
            }
            crate::ast::Stmt::For { body, .. } => {
                collect_stmt_var_types(body, types, structs)
            }
            crate::ast::Stmt::While { body, .. } => {
                collect_stmt_var_types(body, types, structs)
            }
            _ => {}
        }
    }
}
