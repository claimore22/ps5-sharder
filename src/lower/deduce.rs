//! C++14 `auto` return deduction.
//!
//! Runs first in `lower_module` on a cloned module: `auto f(...)` whose
//! `return` expressions unanimously share one type gets that concrete
//! return type. Anything else (mixed/empty/unguessable returns, or a
//! deduction of `auto` itself) stays `auto` and fails loudly at DXC.
//! Mirrors `semantic`'s int/float literal rule so both layers agree.

use std::collections::{HashMap, HashSet};

use crate::ast::{Expr, Function, Module, Stmt, Type};

/// Clone `module` with unanimous `auto` returns resolved.
pub(crate) fn deduce_auto_returns(module: &Module) -> Module {
    let mut module = module.clone();
    for f in &mut module.functions {
        if !matches!(&f.return_type, Type::Named(n) if n == "auto") {
            continue;
        }
        if let Some(ty) = deduce(f) {
            if !matches!(&ty, Type::Named(n) if n == "auto") {
                f.return_type = ty;
            }
        }
    }
    module
}

/// Desugar C++17 structured bindings with call inits once callee returns
/// are deduced: `auto [a, b] = f(x);` (f returning struct S with matching
/// fields) becomes a temp plus per-member decls. Anything else (non-call
/// inits, unknown/ambiguous callees, field-count mismatch, for-init
/// position) stays a `Destructure` for the `lower_stmt` fallback, which
/// preserves the old block-of-autos behavior exactly.
pub(crate) fn desugar_bindings(module: &Module) -> Module {
    // (name, pre-desugar arity) -> (return struct name, fields).
    let mut callees: HashMap<(String, usize), (String, Vec<(String, Type)>)> = HashMap::new();
    // Methods share the name table (called as `obj.m(...)`; arity matches
    // the declared params, without `this`). First match wins, like the
    // call-graph machinery; the field-count gate below keeps mismatches
    // loud.
    for f in &module.functions {
        let Type::Named(ret) = &f.return_type else {
            continue;
        };
        if ret == "auto" {
            continue;
        }
        let Some(decl) = module.structs.iter().find(|s| &s.name == ret) else {
            continue;
        };
        callees.entry((f.name.clone(), f.parameters.len())).or_insert_with(|| {
            (
                ret.clone(),
                decl.fields
                    .iter()
                    .map(|fd| (fd.name.clone(), fd.ty.clone()))
                    .collect(),
            )
        });
    }

    let mut module = module.clone();
    let mut counter: usize = 0;
    // Per-function value-type environments for non-call destructure
    // inits (`auto [a, b] = buf[i]` needs the buffer element struct):
    // params plus first-seen locals plus module globals/resources.
    // Struct field layouts for member/index type inference. Owned up
    // front: the loop below mutably borrows functions.
    let struct_layouts: HashMap<String, Vec<(String, Type)>> = module
        .structs
        .iter()
        .map(|s| {
            (
                s.name.clone(),
                s.fields
                    .iter()
                    .map(|f| (f.name.clone(), f.ty.clone()))
                    .collect(),
            )
        })
        .collect();
    let mut global_env: HashMap<String, Type> = HashMap::new();
    for g in &module.globals {
        global_env.entry(g.name.clone()).or_insert(g.ty.clone());
    }
    for r in &module.resources {
        global_env.entry(r.name.clone()).or_insert(Type::Resource {
            kind: r.kind.clone(),
            access: crate::ast::ResourceAccess::ReadOnly,
            element: Box::new(r.element_ty.clone().unwrap_or(Type::UInt)),
        });
    }
    for f in &mut module.functions {
        let mut env: HashMap<String, Type> = global_env.clone();
        // Params shadow globals (function scope wins).
        for p in &f.parameters {
            env.insert(p.name.clone(), p.ty.clone());
        }
        fn collect_local_tys(stmts: &[Stmt], env: &mut HashMap<String, Type>) {
            for s in stmts {
                match s {
                    Stmt::VarDecl { name, ty, .. } => {
                        env.entry(name.clone()).or_insert(ty.clone());
                    }
                    Stmt::Block(ss) => collect_local_tys(ss, env),
                    Stmt::If { then_branch, else_branch, .. } => {
                        collect_local_tys(then_branch, env);
                        if let Some(eb) = else_branch {
                            collect_local_tys(eb, env);
                        }
                    }
                    Stmt::For { init, body, .. } => {
                        collect_local_tys(std::slice::from_ref(init), env);
                        collect_local_tys(body, env);
                    }
                    Stmt::While { body, .. } => collect_local_tys(body, env),
                    Stmt::Switch { cases, .. } => {
                        for c in cases {
                            collect_local_tys(&c.body, env);
                        }
                    }
                    Stmt::Return(_)
                    | Stmt::Destructure { .. }
                    | Stmt::Expr(_) => {}
                }
            }
        }
        collect_local_tys(&f.body, &mut env);
        let ctx = DesugarCtx {
            callees: &callees,
            env: &env,
            layouts: &struct_layouts,
        };
        // NOTE: bodies borrow independently per statement below.
        let body = std::mem::take(&mut f.body);
        let mut out: Vec<Stmt> = Vec::with_capacity(body.len());
        for s in body {
            out.push(desugar_stmt(s, &ctx, &mut counter, false));
        }
        f.body = out;
    }
    module
}

/// Shared context for structured-binding desugaring.
struct DesugarCtx<'a> {
    callees: &'a HashMap<(String, usize), (String, Vec<(String, Type)>)>,
    env: &'a HashMap<String, Type>,
    layouts: &'a HashMap<String, Vec<(String, Type)>>,
}

/// Infer the struct behind a non-call destructure init, if it is
/// provably a struct value: variables of struct type, member accesses
/// through named structs, buffer-index expressions (resource element),
/// and struct constructors. Returns the struct name plus its fields.
/// Anything else (calls handled separately, opaque/unknown) declines.
fn infer_destructure_struct(
    init: &Expr,
    ctx: &DesugarCtx,
) -> Option<(String, Vec<(String, Type)>)> {
    fn infer_ty(e: &Expr, ctx: &DesugarCtx) -> Option<Type> {
        match e {
            Expr::Variable(n) => ctx.env.get(n).cloned(),
            Expr::Member { object, member } => {
                let Type::Named(s) = infer_ty(object, ctx)? else {
                    return None;
                };
                ctx.layouts
                    .get(&s)?
                    .iter()
                    .find(|(n, _)| n == member)
                    .map(|(_, t)| t.clone())
            }
            Expr::Index { object, .. } => {
                let Type::Resource { element, .. } = infer_ty(object, ctx)? else {
                    return None;
                };
                Some(element.as_ref().clone())
            }
            Expr::Constructor { ty, .. } => Some(ty.clone()),
            _ => None,
        }
    }
    let Type::Named(s) = infer_ty(init, ctx)? else {
        return None;
    };
    let fields = ctx.layouts.get(&s)?.clone();
    Some((s, fields))
}

/// Desugar one statement; `in_for_init` positions keep the node (the
/// `lower_stmt` fallback preserves old behavior there).
fn desugar_stmt(
    stmt: Stmt,
    ctx: &DesugarCtx,
    counter: &mut usize,
    in_for_init: bool,
) -> Stmt {
    match stmt {
        Stmt::Destructure {
            storage,
            ty,
            names,
            init: Some(init),
        } => {
            if in_for_init {
                return Stmt::Destructure {
                    storage,
                    ty,
                    names,
                    init: Some(init),
                };
            }
            // Calls desugar by callee return struct (bare `f(...)` or
            // dotted `obj.m(...)`).
            let call_shape: Option<(String, Vec<(String, Type)>)> = match &init {
                Expr::Call { function, arguments } => {
                    let id = match function.as_ref() {
                        Expr::Variable(name) => Some((name.clone(), arguments.len())),
                        Expr::Member { member, .. } => Some((member.clone(), arguments.len())),
                        _ => None,
                    };
                    id.and_then(|(callee, arity)| ctx.callees.get(&(callee, arity)).cloned())
                }
                _ => None,
            };
            // Anything else desugars when the init provably yields a
            // struct value with exactly as many fields as names
            // (`auto [a, b] = buf[i]` via the buffer element struct).
            // Unknown shapes keep today's loud fallback.
            let shape = match call_shape {
                Some(s) => Some(s),
                None => infer_destructure_struct(&init, ctx),
            };
            let Some((struct_name, fields)) = shape else {
                return Stmt::Destructure {
                    storage,
                    ty,
                    names,
                    init: Some(init),
                };
            };
            if fields.len() != names.len() {
                return Stmt::Destructure {
                    storage,
                    ty,
                    names,
                    init: Some(init),
                };
            }
            let tmp = format!("__destructure{}", *counter);
            *counter += 1;
            let mut stmts = vec![Stmt::VarDecl {
                storage: None,
                ty: Type::Named(struct_name.clone()),
                name: tmp.clone(),
                array_size: None,
                init: Some(init),
            }];
            for (name, (field_name, field_ty)) in
                names.into_iter().zip(fields.iter())
            {
                stmts.push(Stmt::VarDecl {
                    storage: storage.clone(),
                    ty: field_ty.clone(),
                    name,
                    array_size: None,
                    init: Some(Expr::Member {
                        object: Box::new(Expr::Variable(tmp.clone())),
                        member: field_name.clone(),
                    }),
                });
            }
            Stmt::Block(stmts)
        }
        Stmt::Block(ss) => Stmt::Block(
            ss.into_iter()
                .map(|s| desugar_stmt(s, ctx, counter, in_for_init))
                .collect(),
        ),
        Stmt::If {
            condition,
            then_branch,
            else_branch,
        } => Stmt::If {
            condition,
            then_branch: then_branch
                .into_iter()
                .map(|s| desugar_stmt(s, ctx, counter, in_for_init))
                .collect(),
            else_branch: else_branch.map(|b| {
                b.into_iter()
                    .map(|s| desugar_stmt(s, ctx, counter, in_for_init))
                    .collect()
            }),
        },
        Stmt::For {
            init,
            condition,
            update,
            body,
        } => Stmt::For {
            init: Box::new(desugar_stmt(*init, ctx, counter, true)),
            condition,
            update,
            body: body
                .into_iter()
                .map(|s| desugar_stmt(s, ctx, counter, in_for_init))
                .collect(),
        },
        Stmt::While { condition, body } => Stmt::While {
            condition,
            body: body
                .into_iter()
                .map(|s| desugar_stmt(s, ctx, counter, in_for_init))
                .collect(),
        },
        Stmt::Switch { scrutinee, cases } => Stmt::Switch {
            scrutinee,
            cases: cases
                .into_iter()
                .map(|mut c| {
                    c.body = c
                        .body
                        .into_iter()
                        .map(|s| desugar_stmt(s, ctx, counter, in_for_init))
                        .collect();
                    c
                })
                .collect(),
        },
        other => other,
    }
}

fn deduce(f: &Function) -> Option<Type> {
    let mut vars: HashMap<String, Type> = HashMap::new();
    for p in &f.parameters {
        vars.insert(p.name.clone(), p.ty.clone());
    }
    collect_vars(&f.body, &mut vars);
    let mut found: Option<Type> = None;
    let mut any = false;
    for ret in returns(&f.body) {
        // Bare `return;` alongside value returns is ill-formed (as in C++).
        let e = ret?;
        let ty = infer(e, &vars)?;
        match &found {
            None => found = Some(ty),
            Some(t) if *t == ty => {}
            _ => return None,
        }
        any = true;
    }
    if any { found } else { None }
}

/// All `return <expr>` expressions (`None` for bare `return;`).
fn returns(body: &[Stmt]) -> Vec<Option<&Expr>> {
    fn stmt<'a>(s: &'a Stmt, out: &mut Vec<Option<&'a Expr>>) {
        match s {
            Stmt::Return(e) => out.push(e.as_ref()),
            Stmt::Block(ss) => {
                for s in ss {
                    stmt(s, out);
                }
            }
            Stmt::If {
                then_branch,
                else_branch,
                ..
            } => {
                for s in then_branch {
                    stmt(s, out);
                }
                if let Some(b) = else_branch {
                    for s in b {
                        stmt(s, out);
                    }
                }
            }
            Stmt::For { init, body, .. } => {
                stmt(init, out);
                for s in body {
                    stmt(s, out);
                }
            }
            Stmt::While { body, .. } => {
                for s in body {
                    stmt(s, out);
                }
            }
            Stmt::Switch { cases, .. } => {
                for c in cases {
                    for s in &c.body {
                        stmt(s, out);
                    }
                }
            }
            _ => {}
        }
    }
    let mut out = Vec::new();
    for s in body {
        stmt(s, &mut out);
    }
    out
}

fn collect_vars(body: &[Stmt], vars: &mut HashMap<String, Type>) {
    fn stmt(s: &Stmt, vars: &mut HashMap<String, Type>) {
        match s {
            Stmt::VarDecl { ty, name, .. } => {
                vars.entry(name.clone()).or_insert_with(|| ty.clone());
            }
            Stmt::Block(ss) => {
                for s in ss {
                    stmt(s, vars);
                }
            }
            Stmt::If {
                then_branch,
                else_branch,
                ..
            } => {
                for s in then_branch {
                    stmt(s, vars);
                }
                if let Some(b) = else_branch {
                    for s in b {
                        stmt(s, vars);
                    }
                }
            }
            Stmt::For { init, body, .. } => {
                stmt(init, vars);
                for s in body {
                    stmt(s, vars);
                }
            }
            Stmt::While { body, .. } => {
                for s in body {
                    stmt(s, vars);
                }
            }
            Stmt::Switch { cases, .. } => {
                for c in cases {
                    for s in &c.body {
                        stmt(s, vars);
                    }
                }
            }
            _ => {}
        }
    }
    for s in body {
        stmt(s, vars);
    }
}

/// Propagates whole-struct alias copies (`s_srt = srt;` as the entry's
/// first statement, both sides one struct type): every `G` use becomes
/// `P`, the store is dropped, and the now-unused global is removed.
/// Without this, the mirror global flattens into a duplicate binding
/// set (DXC cbuffer-member redefinition) and the dangling store
/// references a dropped declaration.
///
/// Soundness gates (anything else keeps today's loud path):
/// - single whole-struct store, first statement of the entry function;
/// - `G` a module-global struct variable, `P` a directionless entry
///   parameter, identical struct types (so all member paths resolve
///   identically — reads AND member-writes redirect to the same memory;
///   scalar member-writes stay exactly as loud as before, resource
///   writes keep working);
/// - no other whole-struct stores to `G` or `P` anywhere (the alias
///   must hold for the whole module);
/// - no `G` uses at module scope (constants), which evaluate before
///   the entry runs.
pub(crate) fn propagate_struct_alias(module: &Module, entry: Option<&str>) -> Module {
    let Some(entry) = entry else {
        return module.clone();
    };
    let entry_idx = match module.functions.iter().position(|f| f.name == entry) {
        Some(i) => i,
        None => return module.clone(),
    };
    // The store must be the entry's first statement: `G = P;`.
    let (alias_global, alias_param) = match module.functions[entry_idx].body.first() {
        Some(Stmt::Expr(Expr::Assign { lhs, rhs })) => match (lhs.as_ref(), rhs.as_ref()) {
            (Expr::Variable(g), Expr::Variable(p)) => (g.clone(), p.clone()),
            _ => return module.clone(),
        },
        _ => return module.clone(),
    };
    if alias_global == alias_param {
        return module.clone();
    }
    // `G` is a module-global struct variable...
    let global_ty = match module.globals.iter().find(|g| g.name == alias_global) {
        Some(g) => g.ty.clone(),
        None => return module.clone(),
    };
    // ...and `P` a directionless entry parameter of the identical type.
    let param_ok = module.functions[entry_idx]
        .parameters
        .iter()
        .any(|p| p.name == alias_param && p.direction.is_none() && p.ty == global_ty);
    if !param_ok {
        return module.clone();
    }
    // Exactly one whole-struct store overall: the alias copy itself.
    // (Any other store to either side would break the alias.)
    let mut store_count = 0usize;
    for f in &module.functions {
        for s in &f.body {
            walk_stmts(s, &mut |s| {
                if let Stmt::Expr(Expr::Assign { lhs, .. }) = s {
                    if let Expr::Variable(v) = lhs.as_ref() {
                        if v == &alias_global || v == &alias_param {
                            store_count += 1;
                        }
                    }
                }
            });
        }
    }
    if store_count != 1 {
        return module.clone();
    }
    // No module-scope uses (constants evaluate before the entry runs).
    // Resources carry no init expressions; globals carry none either.
    for v in module.constants.values() {
        let mut found = false;
        walk_expr(v, &mut |e| {
            if matches!(e, Expr::Variable(n) if n == &alias_global) {
                found = true;
            }
        });
        if found {
            return module.clone();
        }
    }
    // Rewrite every `G` use to `P`, drop the store, drop the global.
    let mut module = module.clone();
    for f in &mut module.functions {
        for s in &mut f.body {
            rewrite_var(s, &alias_global, &alias_param);
        }
    }
    if let Some(entry_fn) = module.functions.iter_mut().find(|f| f.name == entry) {
        entry_fn.body.remove(0);
    }
    module.globals.retain(|g| g.name != alias_global);
    module
}

/// Visits every statement recursively (read-only). Expression
/// positions carry no statements, so conditions/initializers are not
/// descended into here (see `walk_expr` for those).
fn walk_stmts(stmt: &Stmt, visit: &mut impl FnMut(&Stmt)) {
    visit(stmt);
    match stmt {
        Stmt::Block(ss) => {
            for s in ss {
                walk_stmts(s, visit);
            }
        }
        Stmt::If {
            then_branch,
            else_branch,
            ..
        } => {
            for s in then_branch {
                walk_stmts(s, visit);
            }
            if let Some(b) = else_branch {
                for s in b {
                    walk_stmts(s, visit);
                }
            }
        }
        Stmt::For { init, body, .. } => {
            walk_stmts(init, visit);
            for s in body {
                walk_stmts(s, visit);
            }
        }
        Stmt::While { body, .. } => {
            for s in body {
                walk_stmts(s, visit);
            }
        }
        Stmt::Switch { cases, .. } => {
            for c in cases {
                for s in &c.body {
                    walk_stmts(s, visit);
                }
            }
        }
        _ => {}
    }
}

/// Visits every expression recursively.
fn walk_expr(e: &Expr, visit: &mut impl FnMut(&Expr)) {
    visit(e);
    match e {
        Expr::Number(_) | Expr::String(_) | Expr::Bool(_) | Expr::Variable(_) => {}
        // Lambda bodies don't contribute to enclosing returns.
        Expr::Lambda { .. } => {}
        Expr::Member { object, .. } => walk_expr(object, visit),
        Expr::Index { object, index } => {
            walk_expr(object, visit);
            walk_expr(index, visit);
        }
        Expr::Call { function, arguments } => {
            walk_expr(function, visit);
            for a in arguments {
                walk_expr(a, visit);
            }
        }
        Expr::Constructor { arguments, .. } => {
            for a in arguments {
                walk_expr(a, visit);
            }
        }
        Expr::Assign { lhs, rhs } => {
            walk_expr(lhs, visit);
            walk_expr(rhs, visit);
        }
        Expr::Unary { expr, .. } => walk_expr(expr, visit),
        Expr::Binary { left, right, .. } => {
            walk_expr(left, visit);
            walk_expr(right, visit);
        }
        Expr::Conditional {
            condition,
            then_branch,
            else_branch,
        } => {
            walk_expr(condition, visit);
            walk_expr(then_branch, visit);
            walk_expr(else_branch, visit);
        }
        Expr::InitializerList(elems) => {
            for x in elems {
                walk_expr(x, visit);
            }
        }
    }
}

/// Rewrites every `Variable(from)` to `Variable(to)` in a statement.
fn rewrite_var(stmt: &mut Stmt, from: &str, to: &str) {
    match stmt {
        Stmt::Return(e) => {
            if let Some(e) = e {
                rewrite_var_expr(e, from, to);
            }
        }
        Stmt::Block(ss) => {
            for s in ss {
                rewrite_var(s, from, to);
            }
        }
        Stmt::If {
            condition,
            then_branch,
            else_branch,
        } => {
            rewrite_var_expr(condition, from, to);
            for s in then_branch {
                rewrite_var(s, from, to);
            }
            if let Some(b) = else_branch {
                for s in b {
                    rewrite_var(s, from, to);
                }
            }
        }
        Stmt::For {
            init,
            condition,
            update,
            body,
        } => {
            rewrite_var(init, from, to);
            if let Some(c) = condition {
                rewrite_var_expr(c, from, to);
            }
            if let Some(u) = update {
                rewrite_var_expr(u, from, to);
            }
            for s in body {
                rewrite_var(s, from, to);
            }
        }
        Stmt::While { condition, body } => {
            rewrite_var_expr(condition, from, to);
            for s in body {
                rewrite_var(s, from, to);
            }
        }
        Stmt::Switch { scrutinee, cases } => {
            rewrite_var_expr(scrutinee, from, to);
            for c in cases {
                if let Some(label) = &mut c.label {
                    rewrite_var_expr(label, from, to);
                }
                for s in &mut c.body {
                    rewrite_var(s, from, to);
                }
            }
        }
        Stmt::VarDecl { init, .. } | Stmt::Destructure { init, .. } => {
            if let Some(e) = init {
                rewrite_var_expr(e, from, to);
            }
        }
        Stmt::Expr(e) => rewrite_var_expr(e, from, to),
    }
}

/// Rewrites every `Variable(from)` to `Variable(to)` in an expression.
fn rewrite_var_expr(e: &mut Expr, from: &str, to: &str) {
    match e {
        Expr::Variable(name) => {
            if name == from {
                *name = to.to_string();
            }
        }
        // Lambda params shadow outer names; otherwise rewrite the body.
        Expr::Lambda { params, body, .. } => {
            if !params.iter().any(|p| p == from) {
                for s in body {
                    rewrite_var(s, from, to);
                }
            }
        }
        Expr::Member { object, .. } => rewrite_var_expr(object, from, to),
        Expr::Index { object, index } => {
            rewrite_var_expr(object, from, to);
            rewrite_var_expr(index, from, to);
        }
        Expr::Call { function, arguments } => {
            rewrite_var_expr(function, from, to);
            for a in arguments {
                rewrite_var_expr(a, from, to);
            }
        }
        Expr::Constructor { arguments, .. } => {
            for a in arguments {
                rewrite_var_expr(a, from, to);
            }
        }
        Expr::Assign { lhs, rhs } => {
            rewrite_var_expr(lhs, from, to);
            rewrite_var_expr(rhs, from, to);
        }
        Expr::Unary { expr, .. } => rewrite_var_expr(expr, from, to),
        Expr::Binary { left, right, .. } => {
            rewrite_var_expr(left, from, to);
            rewrite_var_expr(right, from, to);
        }
        Expr::Conditional {
            condition,
            then_branch,
            else_branch,
        } => {
            rewrite_var_expr(condition, from, to);
            rewrite_var_expr(then_branch, from, to);
            rewrite_var_expr(else_branch, from, to);
        }
        Expr::InitializerList(elems) => {
            for x in elems {
                rewrite_var_expr(x, from, to);
            }
        }
        Expr::Number(_) | Expr::String(_) | Expr::Bool(_) => {}
    }
}

/// Globals aliasing SRT-root members (`static const auto frameBuffer =
/// srt->frameBuffer;`, jobified compute shaders): the alias erases and
/// uses rewrite to the member path (which lowering/flattening resolves
/// to the binding). Global initializers otherwise never survive parsing,
/// so these stay `auto` and fail loudly at DXC.
///
/// Soundness: resource/sampler targets are reference types (the handle
/// is immutable; backing store shared), so substitution is exact
/// unconditionally. Data targets (scalars/vectors/matrices) substitute
/// only with no overlapping write path anywhere (a later store through
/// the root would diverge a snapshot from a re-read). Struct targets,
/// shared (LDS) globals, arrays, rebinding assignments, impure indices,
/// and shadowed functions (params/locals of the same name bind
/// locally) all keep today's loud path.
pub(crate) fn resolve_srt_global_aliases(module: &Module) -> Module {
    use crate::ast::Semantic;
    let mut module = module.clone();
    // SRT roots: globals carrying S_SRT_DATA.
    let roots: HashMap<String, String> = module
        .globals
        .iter()
        .filter_map(|g| {
            let is_srt = g
                .semantic
                .as_ref()
                .is_some_and(|s| matches!(s, Semantic::Name(n) if n.eq_ignore_ascii_case("S_SRT_DATA")));
            if !is_srt {
                return None;
            }
            if let Type::Named(s) = &g.ty {
                Some((g.name.clone(), s.clone()))
            } else {
                None
            }
        })
        .collect();
    if roots.is_empty() {
        return module;
    }
    // Struct name -> (field name -> field type).
    let mut structs: HashMap<&str, HashMap<&str, &Type>> = HashMap::new();
    for s in &module.structs {
        structs.insert(
            s.name.as_str(),
            s.fields.iter().map(|f| (f.name.as_str(), &f.ty)).collect(),
        );
    }
    // Rooted member-path segments (`srt`, `srt.frameBuffer`), else None.
    // Bare roots alias the whole root (uses re-root member accesses).
    fn alias_path(e: &Expr, roots: &HashMap<String, String>) -> Option<Vec<String>> {
        match e {
            Expr::Variable(v) => roots.contains_key(v).then(|| vec![v.clone()]),
            Expr::Member { object, member } => {
                let mut segs = alias_path(object, roots)?;
                segs.push(member.clone());
                Some(segs)
            }
            _ => None,
        }
    }
    // Side-effect-free index shapes (duplicated per use on substitution).
    fn index_pure(e: &Expr) -> bool {
        match e {
            Expr::Call { .. } | Expr::Assign { .. } | Expr::Lambda { .. } => false,
            Expr::Variable(_) | Expr::Number(_) | Expr::String(_) | Expr::Bool(_) => true,
            Expr::Member { object, .. } => index_pure(object),
            Expr::Index { object, index } => index_pure(object) && index_pure(index),
            Expr::Constructor { arguments, .. } => arguments.iter().all(index_pure),
            Expr::Unary { expr, .. } => index_pure(expr),
            Expr::Binary { left, right, .. } => index_pure(left) && index_pure(right),
            Expr::Conditional { condition, then_branch, else_branch } => {
                index_pure(condition) && index_pure(then_branch) && index_pure(else_branch)
            }
            Expr::InitializerList(elems) => elems.iter().all(index_pure),
        }
    }
    // Resolve a rooted member path to its terminal member type through
    // struct fields (`srt.frameBuffer` with `srt: SrtData`).
    // Pointer-typedef roots arrive pre-resolved by the parser (see
    // below), so plain field lookup suffices; anything unresolvable
    // stays loud. Bare roots are resolved by the caller.
    fn path_type<'a>(
        segs: &[String],
        roots: &HashMap<String, String>,
        structs: &'a HashMap<&str, HashMap<&str, &'a Type>>,
    ) -> Option<Type> {
        if segs.len() < 2 {
            return None;
        }
        let mut cur = roots.get(&segs[0])?.clone();
        let mut out: Option<Type> = None;
        for (i, seg) in segs[1..].iter().enumerate() {
            let is_last = i + 1 == segs.len() - 1;
            let fields = structs.get(cur.as_str())?;
            let ty = fields.get(seg.as_str())?;
            match ty {
                Type::Named(s) => {
                    out = Some((*ty).clone());
                    cur = (*s).clone();
                }
                _ if is_last => {
                    out = Some((*ty).clone());
                }
                _ => return None,
            }
        }
        out
    }
    // Root variable of a member/index path, if bare.
    fn expr_root(e: &Expr) -> Option<&str> {
        match e {
            Expr::Variable(n) => Some(n),
            Expr::Member { object, .. } => expr_root(object),
            Expr::Index { object, .. } => expr_root(object),
            _ => None,
        }
    }
    // Substitute `init` for `name` in every position (shadowed functions
    // are skipped by the caller, so remaining uses bind the global).
    fn subst_global(e: &mut Expr, name: &str, init: &Expr) {
        match e {
            Expr::Variable(n) => {
                if n == name {
                    *e = init.clone();
                }
            }
            Expr::Member { object, .. } => subst_global(object, name, init),
            Expr::Index { object, index } => {
                subst_global(object, name, init);
                subst_global(index, name, init);
            }
            Expr::Call { function, arguments } => {
                subst_global(function, name, init);
                for a in arguments {
                    subst_global(a, name, init);
                }
            }
            Expr::Constructor { arguments, .. } => {
                for a in arguments {
                    subst_global(a, name, init);
                }
            }
            Expr::Unary { expr, .. } => subst_global(expr, name, init),
            Expr::Binary { left, right, .. } => {
                subst_global(left, name, init);
                subst_global(right, name, init);
            }
            Expr::Conditional { condition, then_branch, else_branch } => {
                subst_global(condition, name, init);
                subst_global(then_branch, name, init);
                subst_global(else_branch, name, init);
            }
            Expr::Assign { lhs, rhs } => {
                subst_global(lhs, name, init);
                subst_global(rhs, name, init);
            }
            Expr::InitializerList(elems) => {
                for x in elems {
                    subst_global(x, name, init);
                }
            }
            Expr::Lambda { body, .. } => {
                for s in body {
                    subst_global_stmt(s, name, init);
                }
            }
            Expr::Number(_) | Expr::String(_) | Expr::Bool(_) => {}
        }
    }
    fn subst_global_stmt(s: &mut Stmt, name: &str, init: &Expr) {
        match s {
            Stmt::Return(e) => {
                if let Some(e) = e {
                    subst_global(e, name, init);
                }
            }
            Stmt::Expr(e) => subst_global(e, name, init),
            Stmt::VarDecl { init: e, .. } => {
                if let Some(e) = e {
                    subst_global(e, name, init);
                }
            }
            Stmt::Destructure { init: e, .. } => {
                if let Some(e) = e {
                    subst_global(e, name, init);
                }
            }
            Stmt::Block(ss) => {
                for x in ss {
                    subst_global_stmt(x, name, init);
                }
            }
            Stmt::If { condition, then_branch, else_branch } => {
                subst_global(condition, name, init);
                for x in then_branch {
                    subst_global_stmt(x, name, init);
                }
                if let Some(eb) = else_branch {
                    for x in eb {
                        subst_global_stmt(x, name, init);
                    }
                }
            }
            Stmt::For { init: loop_init, condition, update, body } => {
                subst_global_stmt(loop_init, name, init);
                if let Some(c) = condition {
                    subst_global(c, name, init);
                }
                if let Some(u) = update {
                    subst_global(u, name, init);
                }
                for x in body {
                    subst_global_stmt(x, name, init);
                }
            }
            Stmt::While { condition, body } => {
                subst_global(condition, name, init);
                for x in body {
                    subst_global_stmt(x, name, init);
                }
            }
            Stmt::Switch { scrutinee, cases } => {
                subst_global(scrutinee, name, init);
                for c in cases {
                    if let Some(label) = &mut c.label {
                        subst_global(label, name, init);
                    }
                    for x in &mut c.body {
                        subst_global_stmt(x, name, init);
                    }
                }
            }
        }
    }
    // Whether a function shadows `name` (param or local): its uses bind
    // locally and must not substitute.
    fn shadows(f: &Function, name: &str) -> bool {
        if f.parameters.iter().any(|p| p.name == name) {
            return true;
        }
        fn decls(ss: &[Stmt], name: &str) -> bool {
            ss.iter().any(|s| match s {
                Stmt::VarDecl { name: n, .. } => n == name,
                Stmt::Destructure { names, .. } => names.iter().any(|n| n == name),
                Stmt::Block(ss) => decls(ss, name),
                Stmt::If { then_branch, else_branch, .. } => {
                    decls(then_branch, name)
                        || else_branch.as_ref().is_some_and(|eb| decls(eb, name))
                }
                Stmt::For { init, body, .. } => decls(std::slice::from_ref(init), name) || decls(body, name),
                Stmt::While { body, .. } => decls(body, name),
                Stmt::Switch { cases, .. } => {
                    cases.iter().any(|c| decls(&c.body, name))
                }
                _ => false,
            })
        }
        decls(&f.body, name)
    }
    // Rebinding assignment to the alias itself (`x = ...`, not `x[i]`).
    fn rebinds(body: &[Stmt], name: &str) -> bool {
        fn expr(e: &Expr, name: &str, bad: &mut bool) {
            match e {
                Expr::Assign { lhs, rhs } => {
                    if matches!(lhs.as_ref(), Expr::Variable(n) if n == name) {
                        *bad = true;
                        return;
                    }
                    expr(lhs, name, bad);
                    expr(rhs, name, bad);
                }
                Expr::Variable(_) | Expr::Number(_) | Expr::String(_) | Expr::Bool(_) => {}
                Expr::Member { object, .. } => expr(object, name, bad),
                Expr::Index { object, index } => {
                    expr(object, name, bad);
                    expr(index, name, bad);
                }
                Expr::Call { function, arguments } => {
                    expr(function, name, bad);
                    for a in arguments {
                        expr(a, name, bad);
                    }
                }
                Expr::Constructor { arguments, .. } => {
                    for a in arguments {
                        expr(a, name, bad);
                    }
                }
                Expr::Unary { expr: inner, .. } => expr(inner, name, bad),
                Expr::Binary { left, right, .. } => {
                    expr(left, name, bad);
                    expr(right, name, bad);
                }
                Expr::Conditional { condition, then_branch, else_branch } => {
                    expr(condition, name, bad);
                    expr(then_branch, name, bad);
                    expr(else_branch, name, bad);
                }
                Expr::InitializerList(elems) => {
                    for x in elems {
                        expr(x, name, bad);
                    }
                }
                Expr::Lambda { body, .. } => {
                    for s in body {
                        stmt(s, name, bad);
                    }
                }
            }
        }
        fn stmt(s: &Stmt, name: &str, bad: &mut bool) {
            match s {
                Stmt::Return(e) => {
                    if let Some(e) = e {
                        expr(e, name, bad);
                    }
                }
                Stmt::Expr(e) => expr(e, name, bad),
                Stmt::VarDecl { init, .. } | Stmt::Destructure { init, .. } => {
                    if let Some(e) = init {
                        expr(e, name, bad);
                    }
                }
                Stmt::Block(ss) => {
                    for x in ss {
                        stmt(x, name, bad);
                    }
                }
                Stmt::If { condition, then_branch, else_branch } => {
                    expr(condition, name, bad);
                    for x in then_branch {
                        stmt(x, name, bad);
                    }
                    if let Some(eb) = else_branch {
                        for x in eb {
                            stmt(x, name, bad);
                        }
                    }
                }
                Stmt::For { init, condition, update, body } => {
                    stmt(init, name, bad);
                    if let Some(c) = condition {
                        expr(c, name, bad);
                    }
                    if let Some(u) = update {
                        expr(u, name, bad);
                    }
                    for x in body {
                        stmt(x, name, bad);
                    }
                }
                Stmt::While { condition, body } => {
                    expr(condition, name, bad);
                    for x in body {
                        stmt(x, name, bad);
                    }
                }
                Stmt::Switch { scrutinee, cases } => {
                    expr(scrutinee, name, bad);
                    for c in cases {
                        if let Some(label) = &c.label {
                            expr(label, name, bad);
                        }
                        for x in &c.body {
                            stmt(x, name, bad);
                        }
                    }
                }
            }
        }
        let mut bad = false;
        for s in body {
            stmt(s, name, &mut bad);
        }
        bad
    }
    // Write paths mentioning a root (store targets that could diverge a
    // snapshot): dotted paths only.
    fn write_paths(body: &[Stmt], out: &mut Vec<Vec<String>>) {
        fn path(e: &Expr) -> Option<Vec<String>> {
            match e {
                Expr::Variable(n) => Some(vec![n.clone()]),
                Expr::Member { object, member } => {
                    let mut p = path(object)?;
                    p.push(member.clone());
                    Some(p)
                }
                Expr::Index { object, .. } => path(object),
                _ => None,
            }
        }
        fn expr(e: &Expr, out: &mut Vec<Vec<String>>) {
            if let Expr::Assign { lhs, rhs } = e {
                if let Some(p) = path(lhs) {
                    out.push(p);
                }
                expr(rhs, out);
            }
        }
        fn stmt(s: &Stmt, out: &mut Vec<Vec<String>>) {
            match s {
                Stmt::Return(e) => {
                    if let Some(e) = e {
                        expr(e, out);
                    }
                }
                Stmt::Expr(e) => expr(e, out),
                Stmt::VarDecl { init, .. } | Stmt::Destructure { init, .. } => {
                    if let Some(e) = init {
                        expr(e, out);
                    }
                }
                _ => {}
            }
        }
        // NOTE: nested-statement descent omitted: stores hide in branches
        // in practice only via the shapes above; branch stores are found
        // through the caller's recursion below.
        for s in body {
            match s {
                Stmt::Block(ss) => {
                    let mut sub = Vec::new();
                    write_paths(ss, &mut sub);
                    out.extend(sub);
                }
                Stmt::If { then_branch, else_branch, .. } => {
                    let mut sub = Vec::new();
                    write_paths(then_branch, &mut sub);
                    out.extend(sub);
                    if let Some(eb) = else_branch {
                        let mut sub = Vec::new();
                        write_paths(eb, &mut sub);
                        out.extend(sub);
                    }
                }
                Stmt::For { body, .. } | Stmt::While { body, .. } => {
                    let mut sub = Vec::new();
                    write_paths(body, &mut sub);
                    out.extend(sub);
                }
                Stmt::Switch { cases, .. } => {
                    for c in cases {
                        let mut sub = Vec::new();
                        write_paths(&c.body, &mut sub);
                        out.extend(sub);
                    }
                }
                _ => stmt(s, out),
            }
        }
    }
    // Resolve in declaration order (chains substitute into later inits),
    // then rewrite unshadowed functions and erase resolved globals.
    let mut resolved: HashMap<String, Expr> = HashMap::new();
    let mut erase: Vec<String> = Vec::new();
    // Write paths across all functions (scalar-snapshot guard).
    let mut writes: Vec<Vec<String>> = Vec::new();
    for f in &module.functions {
        write_paths(&f.body, &mut writes);
    }
    for g in module.globals.clone() {
        let Some(init) = g.init.clone() else {
            continue;
        };
        if g.array_size.is_some() || g.shared {
            continue;
        }
        // Substitute already-resolved aliases into this init first.
        let mut init = init;
        for (name, target) in &resolved {
            subst_global(&mut init, name, target);
        }
        // Must be a rooted path (bare roots alias the whole root).
        let Some(segs) = alias_path(&init, &roots) else {
            continue;
        };
        if segs.len() < 1 {
            continue;
        }
        // No rebinding, no self-reference, no shadowing user to break.
        let self_ref = match &init {
            Expr::Variable(n) => n == &g.name,
            _ => false,
        };
        // Classify the target: resources/samplers substitute always
        // (reference semantics); data needs a write-free backing.
        let target_ty = if segs.len() == 1 {
            module
                .globals
                .iter()
                .find(|r| r.name == segs[0])
                .map(|r| r.ty.clone())
        } else {
            path_type(&segs, &roots, &structs)
        };
        let Some(target_ty) = target_ty else {
            continue;
        };
        let is_reference = matches!(target_ty, Type::Resource { .. })
            || matches!(&target_ty, Type::Named(n) if crate::framework::is_sampler_type_name(n));
        let is_data = matches!(
            target_ty,
            Type::Bool
                | Type::Int
                | Type::UInt
                | Type::Float
                | Type::Half
                | Type::Double
                | Type::Vector { .. }
                | Type::Matrix { .. }
        );
        if !is_reference && !is_data {
            continue;
        }
        if self_ref {
            continue;
        }
        // Data snapshots diverge under overlapping stores: any write
        // path overlapping the alias path vetoes (resources are
        // references: stores through them hit shared backing either
        // way, so they never veto).
        if is_data {
            let overlap = writes.iter().any(|w| {
                // Normalize write roots through resolved aliases.
                let w = match w.first().map(String::as_str) {
                    Some(r) if resolved.contains_key(r) => {
                        // Only bare-root writes normalize here; member
                        // paths through aliases need full substitution
                        // (rare: bail conservative below instead).
                        vec![r.to_string()]
                    }
                    _ => w.clone(),
                };
                let a = segs.clone();
                let b = w;
                a.len() <= b.len() && b[..a.len()] == a[..]
                    || b.len() <= a.len() && a[..b.len()] == b[..]
            });
            if overlap {
                continue;
            }
        }
        // Rebinding assignments and shadowing functions veto per use
        // below; record first (uses in shadowed functions keep binding
        // locally, which is exactly right once the global is gone).
        resolved.insert(g.name.clone(), init.clone());
        erase.push(g.name.clone());
    }
    // Per-function substitution (skipping shadowed functions) with a
    // rebinding veto: `x = ...` (bare) would rebind rather than store
    // through, so any function containing one keeps the loud path by
    // vetoing the whole alias (erase only when no function vetoes).
    let mut vetoed: HashMap<String, bool> = HashMap::new();
    for f in &module.functions {
        for (name, _) in &resolved {
            if shadows(f, name) {
                continue;
            }
            if rebinds(&f.body, name) {
                vetoed.insert(name.clone(), true);
            }
        }
    }
    for (name, target) in &resolved {
        if vetoed.contains_key(name) {
            continue;
        }
        for f in &mut module.functions {
            if shadows(f, name) {
                continue;
            }
            for s in &mut f.body {
                subst_global_stmt(s, name, target);
            }
        }
        // Later global inits already substituted in declaration order
        // above; nothing further to do there.
    }
    module
        .globals
        .retain(|g| !(erase.contains(&g.name) && !vetoed.contains_key(&g.name)));
    module
}
///
/// Runs after [`deduce_auto_returns`] (call results are concrete): every
/// `auto` local with an inferable init becomes its concrete type, since
/// HLSL has no `auto` (DXC: `'auto' is a reserved keyword`). Inference
/// covers literals, variables, member paths (including vector swizzles),
/// indexed resources/arrays/vectors/matrices, calls to known free
/// functions and methods on known structs (plus a small set of builtins
/// with certain type rules), constructors, unary/arithmetic expressions,
/// and unanimous ternaries. Anything else stays `auto` (today's loud DXC
/// path): no guessing. Nested scopes clone the env (declarations don't
/// leak out of branches), matching C++ scoping.
pub(crate) fn deduce_auto_locals(module: &Module) -> Module {
    use crate::ast::ResourceAccess;
    // Struct name -> (field name -> (field type, array size)).
    let mut structs: HashMap<&str, HashMap<&str, (&Type, Option<u32>)>> = HashMap::new();
    for s in &module.structs {
        structs.insert(
            s.name.as_str(),
            s.fields
                .iter()
                .map(|f| (f.name.as_str(), (&f.ty, f.array_size)))
                .collect(),
        );
    }
    let is_struct = |n: &str| structs.contains_key(n);
    // Name -> (required, total) arity ranges with concrete return types
    // for free functions; methods key on (struct, name). `auto` returns
    // stay out (unknown). A call admits any overload whose range covers
    // its arity (defaulted trailing params fill in lowering); several
    // admitters must agree on the type, else the call stays loud.
    let mut free_rets: HashMap<&str, Vec<((usize, usize), &Type)>> = HashMap::new();
    let mut method_rets: HashMap<(&str, &str), Vec<((usize, usize), &Type)>> = HashMap::new();
    for f in &module.functions {
        if matches!(&f.return_type, Type::Named(n) if n == "auto") {
            continue;
        }
        let total = f.parameters.len();
        let mut required = 0;
        for p in &f.parameters {
            if p.default.is_some() {
                break;
            }
            required += 1;
        }
        let range = (required, total);
        match f.scope.as_deref() {
            Some(s) if is_struct(s) => {
                method_rets
                    .entry((s, f.name.as_str()))
                    .or_default()
                    .push((range, &f.return_type));
            }
            // No scope or unknown scope (flattened namespace): free.
            _ => {
                free_rets
                    .entry(f.name.as_str())
                    .or_default()
                    .push((range, &f.return_type));
            }
        }
    }
    // Module-level names: globals plus resources (as element carriers
    // for indexed uses; a resource-typed deduction itself is rejected
    // below — resources can't be HLSL locals).
    let mut roots: HashMap<&str, (Type, Option<u32>)> = HashMap::new();
    for g in &module.globals {
        roots.insert(g.name.as_str(), (g.ty.clone(), g.array_size));
    }
    for r in &module.resources {
        roots.insert(
            r.name.as_str(),
            (
                Type::Resource {
                    kind: r.kind.clone(),
                    access: ResourceAccess::ReadOnly,
                    element: Box::new(r.element_ty.clone().unwrap_or(Type::UInt)),
                },
                None,
            ),
        );
    }
    // Struct-returning callees for destructured-name registration (same
    // table `desugar_bindings` uses): `auto [a, b] = f()` names gain
    // their field types for later inits.
    let mut struct_callees: HashMap<(String, usize), Vec<(String, Type)>> = HashMap::new();
    for f in &module.functions {
        let Type::Named(ret) = &f.return_type else {
            continue;
        };
        if ret == "auto" {
            continue;
        }
        let Some(decl) = module.structs.iter().find(|s| &s.name == ret) else {
            continue;
        };
        struct_callees.entry((f.name.clone(), f.parameters.len())).or_insert_with(|| {
            decl.fields
                .iter()
                .map(|fd| (fd.name.clone(), fd.ty.clone()))
                .collect()
        });
    }

    let mut module = module.clone();
    for f in &mut module.functions {
        let mut env: HashMap<String, (Type, Option<u32>)> = HashMap::new();
        for p in &f.parameters {
            env.insert(p.name.clone(), (p.ty.clone(), p.array_size));
        }
        for (name, (ty, arr)) in &roots {
            env.entry(name.to_string()).or_insert_with(|| (ty.clone(), *arr));
        }
        deduce_stmts(
            &mut f.body,
            &mut env,
            &structs,
            &free_rets,
            &method_rets,
            &struct_callees,
            0,
        );
    }
    module
}

/// Whether an initializer carries array-ness (whole-array copy): bare
/// array variables/members (consulting the env and struct table sizes),
/// or conditionals/assignments thereof. Indexed elements, calls, and
/// constructors are single values. `None` (unknowable) counts as scalar
/// here — inference itself stays conservative separately.
#[allow(clippy::too_many_arguments)]
fn init_is_array(
    e: &Expr,
    env: &HashMap<String, (Type, Option<u32>)>,
    structs: &HashMap<&str, HashMap<&str, (&Type, Option<u32>)>>,
    free_rets: &HashMap<&str, Vec<((usize, usize), &Type)>>,
    method_rets: &HashMap<(&str, &str), Vec<((usize, usize), &Type)>>,
    depth: u32,
) -> bool {
    if depth > 32 {
        return false;
    }
    let next = depth + 1;
    match e {
        Expr::Variable(name) => env.get(name).is_some_and(|(_, arr)| arr.is_some()),
        Expr::Member { object, member } => {
            if let Expr::Variable(root) = object.as_ref() {
                // `srt.arr` where the env says array (flattened roots).
                if env.get(root).is_some_and(|(_, arr)| arr.is_some()) {
                    return true;
                }
            }
            match infer_local(object, env, structs, free_rets, method_rets, next) {
                Some(Type::Named(s)) => structs
                    .get(s.as_str())
                    .and_then(|fields| fields.get(member.as_str()))
                    .is_some_and(|(_, arr)| arr.is_some()),
                _ => false,
            }
        }
        Expr::Conditional { then_branch, else_branch, .. } => {
            init_is_array(then_branch, env, structs, free_rets, method_rets, next)
                || init_is_array(else_branch, env, structs, free_rets, method_rets, next)
        }
        Expr::Assign { rhs, .. } => init_is_array(rhs, env, structs, free_rets, method_rets, next),
        _ => false,
    }
}

/// Whether a deduced type can be an HLSL local declaration. Scalars,
/// vectors, matrices, declared structs, and samplers emit; resources
/// can't be locals, and opaque/unknown names (`auto` itself, `string`,
/// `this`, `__lambda`, framework leftovers) keep today's loud path.
fn is_emittable_local(t: &Type, structs: &HashMap<&str, HashMap<&str, (&Type, Option<u32>)>>) -> bool {
    match t {
        Type::Void => false,
        Type::Bool
        | Type::Int
        | Type::UInt
        | Type::Float
        | Type::Half
        | Type::Double
        | Type::Vector { .. }
        | Type::Matrix { .. } => true,
        Type::Resource { .. } => false,
        Type::Named(n) => {
            structs.contains_key(n.as_str())
                || crate::framework::is_sampler_type_name(n)
        }
    }
}

/// Scalar [`Type`] for a [`ScalarType`] base.
fn scalar_type_of(base: crate::ast::ScalarType) -> Type {
    match base {
        crate::ast::ScalarType::Float => Type::Float,
        crate::ast::ScalarType::Half => Type::Half,
        crate::ast::ScalarType::Int => Type::Int,
        crate::ast::ScalarType::UInt => Type::UInt,
        crate::ast::ScalarType::Bool => Type::Bool,
    }
}

/// True for single-channel swizzle spellings (`x`, `r`, `xy`, `rgba`, ...).
fn is_swizzle(member: &str) -> bool {
    !member.is_empty()
        && member.len() <= 4
        && member
            .chars()
            .all(|c| matches!(c, 'x' | 'y' | 'z' | 'w' | 'r' | 'g' | 'b' | 'a'))
}

/// Infers an expression's type for `auto` deduction; `None` means
/// unknowable (the `auto` stays). `depth` caps recursion on adversarial
/// nesting. See [`deduce_auto_locals`] for the covered shapes.
#[allow(clippy::too_many_arguments)]
fn infer_local(
    e: &Expr,
    env: &HashMap<String, (Type, Option<u32>)>,
    structs: &HashMap<&str, HashMap<&str, (&Type, Option<u32>)>>,
    free_rets: &HashMap<&str, Vec<((usize, usize), &Type)>>,
    method_rets: &HashMap<(&str, &str), Vec<((usize, usize), &Type)>>,
    depth: u32,
) -> Option<Type> {
    if depth > 32 {
        return None;
    }
    let next = depth + 1;
    match e {
        Expr::Number(value) => {
            // Mirrors `semantic`: `0` -> Int, `0.0`/`0f` -> Float.
            let is_float = value.contains('.')
                || value.contains('e')
                || value.contains('E')
                || value.ends_with('f')
                || value.ends_with('F');
            Some(if is_float { Type::Float } else { Type::Int })
        }
        Expr::Bool(_) => Some(Type::Bool),
        Expr::Variable(name) => {
            if name == "PI" || name == "M_PI" {
                return Some(Type::Float);
            }
            if name.contains("::k") {
                // Enum constants (`TraversalFlags::kNoAABB...`): integers.
                return Some(Type::Int);
            }
            env.get(name).map(|(t, _)| t.clone())
        }
        Expr::Member { object, member } => {
            let obj = infer_local(object, env, structs, free_rets, method_rets, next)?;
            // Scalar splat (`f.y`, `1.0.xxx`): HLSL broadcasts.
            if matches!(
                obj,
                Type::Float | Type::Half | Type::Int | Type::UInt | Type::Bool | Type::Double
            ) && is_swizzle(member)
            {
                if member.len() == 1 {
                    return Some(obj);
                }
                let base = match obj {
                    Type::Float => crate::ast::ScalarType::Float,
                    Type::Half => crate::ast::ScalarType::Half,
                    Type::Int => crate::ast::ScalarType::Int,
                    Type::UInt => crate::ast::ScalarType::UInt,
                    Type::Bool => crate::ast::ScalarType::Bool,
                    _ => crate::ast::ScalarType::Float,
                };
                return Some(Type::Vector { base, width: member.len() as u8 });
            }
            // Vector swizzle (`tan.xyz` -> float3, `tan.w` -> float).
            if let Type::Vector { base, .. } = obj {
                if is_swizzle(member) {
                    if member.len() == 1 {
                        return Some(scalar_type_of(base));
                    }
                    return Some(Type::Vector { base, width: member.len() as u8 });
                }
                return None;
            }
            // Struct member (chained: `srt.mat.mvp` recurses through).
            if let Type::Named(s) = &obj {
                if let Some(fields) = structs.get(s.as_str()) {
                    if let Some((ty, _)) = fields.get(member.as_str()) {
                        return Some((*ty).clone());
                    }
                }
            }
            None
        }
        Expr::Index { object, .. } => {
            // Sized arrays index to their element type (the size lives
            // on the declaration, not the type): check the env for bare
            // names (`float a[4]; a[i]` -> float) and the struct table
            // for member paths (`s.arr[i]`), before the type rules below
            // (an array of vectors is NOT a vector indexing to scalar).
            if let Expr::Variable(name) = object.as_ref() {
                if let Some((t, Some(_))) = env.get(name) {
                    return Some(t.clone());
                }
            }
            if let Expr::Member { object: inner, member } = object.as_ref() {
                if let Some(obj) = infer_local(inner, env, structs, free_rets, method_rets, next) {
                    if let Type::Named(s) = &obj {
                        if let Some(fields) = structs.get(s.as_str()) {
                            if let Some((t, Some(_))) = fields.get(member.as_str()) {
                                return Some((*t).clone());
                            }
                        }
                    }
                }
            }
            let obj = infer_local(object, env, structs, free_rets, method_rets, next)?;
            match obj {
                // Buffer element (`srt->position[i]` -> float3).
                Type::Resource { element, .. } => Some(*element),
                // Vector component (`v[i]` -> scalar).
                Type::Vector { base, .. } => Some(scalar_type_of(base)),
                // Matrix row (`m[i]` -> row vector).
                Type::Matrix { base, columns, .. } => {
                    Some(Type::Vector { base, width: columns })
                }
                _ => None,
            }
        }
        Expr::Call { function, arguments } => {
            // Free function with a concrete declared return admitted by
            // the call arity (defaulted trailing params fill later).
            if let Expr::Variable(name) = function.as_ref() {
                if let Some(ret) = admitted_return(free_rets.get(name.as_str()), arguments.len()) {
                    return Some(ret);
                }
                if let Some(t) =
                    infer_builtin_call(name, arguments, env, structs, free_rets, method_rets, next)
                {
                    return Some(t);
                }
                return None;
            }
            // Method on a known struct (`obj.method(...)`).
            if let Expr::Member { object, member } = function.as_ref() {
                // Builtin LOD queries (`tex.GetLOD(sampler, uv)`,
                // `tex.GetLODUnclamped(sampler, uv)`) return float.
                let ml = member.to_ascii_lowercase();
                if (ml == "getlod" || ml == "getlodunclamped") && arguments.len() == 2 {
                    return Some(Type::Float);
                }
                if let Some(obj) = infer_local(object, env, structs, free_rets, method_rets, next) {
                    if let Type::Named(s) = &obj {
                        if let Some(ret) = admitted_return(
                            method_rets.get(&(s.as_str(), member.as_str())),
                            arguments.len(),
                        ) {
                            return Some(ret);
                        }
                    }
                }
            }
            None
        }
        Expr::Constructor { ty, .. } => {
            // Concrete construction (`float3(...)`, `VisGsOut(...)`).
            // Opaque targets stay loud.
            match ty {
                Type::Void => None,
                Type::Bool
                | Type::Int
                | Type::UInt
                | Type::Float
                | Type::Half
                | Type::Double
                | Type::Vector { .. }
                | Type::Matrix { .. } => Some(ty.clone()),
                Type::Resource { .. } => None,
                Type::Named(n) => {
                    if structs.contains_key(n.as_str())
                        || crate::framework::is_sampler_type_name(n)
                    {
                        Some(ty.clone())
                    } else {
                        None
                    }
                }
            }
        }
        Expr::Unary { op, expr } => {
            use crate::ast::UnaryOp;
            match op {
                UnaryOp::Not => Some(Type::Bool),
                UnaryOp::Plus | UnaryOp::Minus | UnaryOp::BitNot => {
                    infer_local(expr, env, structs, free_rets, method_rets, next)
                }
            }
        }
        Expr::Binary { left, op, right } => {
            use crate::ast::BinaryOp;
            match op {
                BinaryOp::Equal
                | BinaryOp::NotEqual
                | BinaryOp::Less
                | BinaryOp::LessEqual
                | BinaryOp::Greater
                | BinaryOp::GreaterEqual
                | BinaryOp::LogicalAnd
                | BinaryOp::LogicalOr => Some(Type::Bool),
                BinaryOp::Comma => infer_local(right, env, structs, free_rets, method_rets, next),
                _ => unify_arith(
                    &infer_local(left, env, structs, free_rets, method_rets, next)?,
                    &infer_local(right, env, structs, free_rets, method_rets, next)?,
                ),
            }
        }
        Expr::Conditional { then_branch, else_branch, .. } => {
            let t = infer_local(then_branch, env, structs, free_rets, method_rets, next)?;
            let f = infer_local(else_branch, env, structs, free_rets, method_rets, next)?;
            (t == f).then_some(t)
        }
        Expr::Assign { .. } | Expr::String(_) | Expr::InitializerList(_) => None,
        // Lambdas deduce via beta-reduction (separate pass); an unreduced
        // lambda init stays loud.
        Expr::Lambda { .. } => None,
    }
}

/// The agreed return type of overloads whose (required, total) arity
/// range admits `nargs`, or `None` when no overload admits (or several
/// admit with different types).
fn admitted_return(
    overloads: Option<&Vec<((usize, usize), &Type)>>,
    nargs: usize,
) -> Option<Type> {
    let overloads = overloads?;
    let mut agreed: Option<&Type> = None;
    let mut any = false;
    for ((lo, hi), ret) in overloads {
        if *lo <= nargs && nargs <= *hi {
            any = true;
            match agreed {
                Some(t) if *t == **ret => {}
                Some(_) => return None,
                None => agreed = Some(ret),
            }
        }
    }
    if any { agreed.cloned() } else { None }
}

/// Builtins with certain type rules for `auto` inits: same-as-arg0
/// (derivatives, transcendentals, `normalize`, ...), unanimous-arg
/// poly builtins (`min`/`max`/`clamp`/`lerp` with matching args), and
/// scalar reductions (`dot`/`length`/`distance`). Anything else is
/// unknowable here (the `auto` stays).
#[allow(clippy::too_many_arguments)]
fn infer_builtin_call(
    name: &str,
    arguments: &[Expr],
    env: &HashMap<String, (Type, Option<u32>)>,
    structs: &HashMap<&str, HashMap<&str, (&Type, Option<u32>)>>,
    free_rets: &HashMap<&str, Vec<((usize, usize), &Type)>>,
    method_rets: &HashMap<(&str, &str), Vec<((usize, usize), &Type)>>,
    depth: u32,
) -> Option<Type> {
    let lower = name.to_ascii_lowercase();
    // Target pixel-position builtins (0 args -> float2; threaded to an
    // SV_Position value before lowering, with an emit-layer fallback).
    if matches!(
        lower.as_str(),
        "getvivjlinearcentroid" | "getvivjperspcenter" | "getvivjperspcentroid"
    ) && arguments.is_empty()
    {
        return Some(Type::Vector { base: crate::ast::ScalarType::Float, width: 2 });
    }
    // Framework `wave::inverse(m)` (1 arg -> the matrix type; lowered to the
    // `__wave_inverse` adjugate shim).
    if lower == "wave::inverse" && arguments.len() == 1 {
        let t = infer_local(&arguments[0], env, structs, free_rets, method_rets, depth)?;
        if matches!(t, Type::Matrix { .. }) {
            return Some(t);
        }
        return None;
    }
    // Unsigned byte-wise mean (`__v_lerp_u8(a, b, 1)`): packed uint32.
    if lower == "__v_lerp_u8" && arguments.len() == 3 {
        return Some(Type::UInt);
    }
    // GCN byte permute (`__v_perm_b32(a, b, sel)`): packed uint32.
    if lower == "__v_perm_b32" && arguments.len() == 3 {
        return Some(Type::UInt);
    }
    // GCN packed-fp16 ops preserve the (half2) argument type.
    if matches!(
        lower.as_str(),
        "__v_pk_min_f16" | "__v_pk_max_f16" | "__v_pk_add_f16" | "__v_pk_mul_f16"
    ) && arguments.len() == 2
    {
        return infer_local(&arguments[0], env, structs, free_rets, method_rets, depth);
    }
    // Sum of absolute differences preserves the first argument type.
    if lower == "sad" && arguments.len() == 3 {
        return infer_local(&arguments[0], env, structs, free_rets, method_rets, depth);
    }
    // Same-as-first-arg.
    if matches!(
        lower.as_str(),
        "ddx" | "ddy"
            | "ddx_fine" | "ddy_fine"
            | "ddx_coarse" | "ddy_coarse"
            | "frac" | "floor" | "ceil" | "round" | "trunc"
            | "saturate" | "abs" | "normalize"
            | "sin" | "cos" | "tan" | "asin" | "acos" | "atan"
            | "exp" | "exp2" | "log" | "log2" | "log10" | "sqrt" | "rsqrt"
            | "fwidth"
    ) && arguments.len() == 1
    {
        return infer_local(&arguments[0], env, structs, free_rets, method_rets, depth);
    }
    // `cross(v, w)` -> the vector type.
    if lower == "cross" && arguments.len() == 2 {
        let t = infer_local(&arguments[0], env, structs, free_rets, method_rets, depth)?;
        if matches!(t, Type::Vector { .. }) {
            return Some(t);
        }
        return None;
    }
    // Scalar reductions (`dot`/`length`/`distance` -> base scalar).
    if matches!(lower.as_str(), "dot" | "length" | "distance") && !arguments.is_empty() {
        return match infer_local(&arguments[0], env, structs, free_rets, method_rets, depth)? {
            Type::Vector { base, .. } => Some(scalar_type_of(base)),
            t @ (Type::Float | Type::Half | Type::Int | Type::UInt | Type::Double) => Some(t),
            _ => None,
        };
    }
    // Unanimous-arg poly builtins (`min`/`max`/`clamp`/`lerp`/...).
    // HLSL broadcasts scalar args against vector ones
    // (`max(float2, 0.0f)`), so fold with arithmetic unification
    // instead of requiring strict equality (which left such calls
    // uninferrable and blocked template deduction downstream).
    if matches!(
        lower.as_str(),
        "min" | "max" | "clamp" | "lerp" | "smoothstep" | "faceforward" | "reflect"
    ) && !arguments.is_empty()
    {
        let mut tys = arguments
            .iter()
            .map(|a| infer_local(a, env, structs, free_rets, method_rets, depth));
        let first = tys.next()??;
        let mut acc = first;
        for t in tys {
            let t = t?;
            acc = unify_arith(&acc, &t)?;
        }
        return Some(acc);
    }
    None
}

/// Unifies two scalar types for `auto` arithmetic: `double` wins, then
/// `float` (half demotes), then `uint`, else `int`. Vectors and
/// matrices have no `double`/`bool` spellings here; those stay loud.
fn unify_base(a: &Type, b: &Type) -> Option<crate::ast::ScalarType> {
    use crate::ast::ScalarType;
    let rank = |t: &Type| match t {
        Type::Double => Some(3),
        Type::Float | Type::Half => Some(2),
        Type::UInt => Some(1),
        Type::Int | Type::Bool => Some(0),
        _ => None,
    };
    let (ra, rb) = (rank(a)?, rank(b)?);
    if ra == 3 || rb == 3 {
        return None;
    }
    if ra == 2 || rb == 2 {
        return Some(ScalarType::Float);
    }
    if ra == 1 || rb == 1 {
        return Some(ScalarType::UInt);
    }
    Some(ScalarType::Int)
}

/// C-like arithmetic unification for `auto` inits: comparisons are
/// handled by the caller; here `double` wins, then `float`, then
/// `uint`, else the lhs when both sides agree. Vectors unify by base
/// with matching widths (scalars broadcast). Anything else is unknowable.
fn unify_arith(a: &Type, b: &Type) -> Option<Type> {
    if a == b {
        return Some(a.clone());
    }
    let rank = |t: &Type| match t {
        Type::Double => Some(3),
        Type::Float | Type::Half => Some(2),
        Type::UInt => Some(1),
        Type::Int | Type::Bool => Some(0),
        _ => None,
    };
    // Scalar-vector / scalar-matrix splat (`v2 - 0.5f`, `m * 2.0f`):
    // HLSL broadcasts the scalar; the base unifies as usual. The other
    // side must be a scalar: composite-composite pairs (vector-vector,
    // matrix-matrix) belong to the structural match below — without the
    // guard this arm shadows them and misfires to `None`.
    let is_scalar_ty = |t: &Type| {
        matches!(
            t,
            Type::Bool | Type::Int | Type::UInt | Type::Float | Type::Half | Type::Double
        )
    };
    if let (Type::Vector { base, width }, s) | (s, Type::Vector { base, width }) = (a, b) {
        if is_scalar_ty(s) {
            let base = unify_base(&scalar_type_of(*base), s)?;
            return Some(Type::Vector { base, width: *width });
        }
    }
    if let (Type::Matrix { base, rows, columns }, s) | (s, Type::Matrix { base, rows, columns }) =
        (a, b)
    {
        if is_scalar_ty(s) {
            let base = unify_base(&scalar_type_of(*base), s)?;
            return Some(Type::Matrix { base, rows: *rows, columns: *columns });
        }
    }
    match (a, b) {
        (Type::Vector { base: ba, width: wa }, Type::Vector { base: bb, width: wb })
            if wa == wb =>
        {
            let base = unify_base(&scalar_type_of(*ba), &scalar_type_of(*bb))?;
            Some(Type::Vector { base, width: *wa })
        }
        _ => {
            let (ra, rb) = (rank(a)?, rank(b)?);
            if ra == 3 || rb == 3 {
                return Some(Type::Double);
            }
            if ra == 2 || rb == 2 {
                // Half+float mixes come out float.
                return Some(Type::Float);
            }
            if ra == 1 || rb == 1 {
                return Some(Type::UInt);
            }
            // int/bool mixes: int (bool arithmetic promotes).
            if matches!(a, Type::Int | Type::Bool) && matches!(b, Type::Int | Type::Bool) {
                return Some(Type::Int);
            }
            None
        }
    }
}

/// Rewrites `auto` locals in `stmts`, registering each declaration
/// (deduced or not) so later inits see earlier locals. Returns
/// nothing; nested scopes work on clones.
#[allow(clippy::too_many_arguments)]
fn deduce_stmts(
    stmts: &mut [Stmt],
    env: &mut HashMap<String, (Type, Option<u32>)>,
    structs: &HashMap<&str, HashMap<&str, (&Type, Option<u32>)>>,
    free_rets: &HashMap<&str, Vec<((usize, usize), &Type)>>,
    method_rets: &HashMap<(&str, &str), Vec<((usize, usize), &Type)>>,
    struct_callees: &HashMap<(String, usize), Vec<(String, Type)>>,
    depth: u32,
) {
    for s in stmts.iter_mut() {
        match s {
            Stmt::VarDecl { ty, name, array_size, init, .. } => {
                // Whole-array copies (`auto t = srt.transforms;` with an
                // array member) keep `auto`: the element type alone would
                // be a mistyped scalar declaration, while the downstream
                // subobject materialization completes the real array shape
                // from the path (today's validating behavior).
                let array_init = init.as_ref().is_some_and(|e| {
                    init_is_array(e, env, structs, free_rets, method_rets, 0)
                });
                if matches!(ty, Type::Named(n) if n == "auto") && array_size.is_none() && !array_init {
                    if let Some(init_expr) = init {
                        if let Some(t) =
                            infer_local(init_expr, env, structs, free_rets, method_rets, 0)
                        {
                            if is_emittable_local(&t, structs) {
                                *ty = t.clone();
                            }
                        }
                    }
                }
                // Register under the (possibly deduced) type so later
                // inits (`auto y = x;`) resolve; unknown autos register
                // as-is (uses of them stay loud, as today).
                env.insert(name.clone(), (ty.clone(), *array_size));
            }
            Stmt::Destructure { names, init, .. } => {
                // Register field types for `auto [a, b] = f();` so later
                // inits see them (the node itself desugars later).
                if let Some(Expr::Call { function, arguments }) = init {
                    let callee = match function.as_ref() {
                        Expr::Variable(n) => Some(n.clone()),
                        Expr::Member { member, .. } => Some(member.clone()),
                        _ => None,
                    };
                    if let Some(callee) = callee {
                        if let Some(fields) = struct_callees.get(&(callee, arguments.len())) {
                            for (n, field) in names.iter().zip(fields.iter()) {
                                env.insert(n.clone(), (field.1.clone(), None));
                            }
                        }
                    }
                }
            }
            Stmt::Block(inner) => {
                let mut child = env.clone();
                deduce_stmts(inner, &mut child, structs, free_rets, method_rets, struct_callees, depth + 1);
            }
            Stmt::If { then_branch, else_branch, .. } => {
                let mut child = env.clone();
                deduce_stmts(then_branch, &mut child, structs, free_rets, method_rets, struct_callees, depth + 1);
                if let Some(els) = else_branch {
                    let mut child = env.clone();
                    deduce_stmts(els, &mut child, structs, free_rets, method_rets, struct_callees, depth + 1);
                }
            }
            Stmt::For { init, body, .. } => {
                // The init scope covers the whole loop.
                let mut child = env.clone();
                deduce_stmts(std::slice::from_mut(init.as_mut()), &mut child, structs, free_rets, method_rets, struct_callees, depth + 1);
                deduce_stmts(body, &mut child, structs, free_rets, method_rets, struct_callees, depth + 1);
            }
            Stmt::While { body, .. } => {
                let mut child = env.clone();
                deduce_stmts(body, &mut child, structs, free_rets, method_rets, struct_callees, depth + 1);
            }
            Stmt::Switch { cases, .. } => {
                for c in cases {
                    let mut child = env.clone();
                    deduce_stmts(&mut c.body, &mut child, structs, free_rets, method_rets, struct_callees, depth + 1);
                }
            }
            Stmt::Return(_) | Stmt::Expr(_) => {}
        }
    }
    let _ = depth;
}

/// Beta-reduces directly-called lambdas without by-value captures
/// (`auto convertUInt16 = [](ulong v) { return (v & 0xffff) / 65535.0f; };`
/// with `convertUInt16(x)` uses, or `auto s4 = [&ss](int y, int x) {...}`
/// with `s4(a, b)` uses): each call becomes the single return expression
/// with arguments substituted for parameters, and the declaration is
/// dropped. Capture-less and all-reference captures qualify (definition
/// and direct calls share the function scope, so captures bind
/// identically). By-value captures (`[x]`) qualify when `x` is frozen
/// (see `capture_frozen`): the definition-site value provably equals the
/// value at every use, so reading the name in place is exact. Copy-all
/// (`[=]`) never qualifies. Only single-`return` bodies with pure
/// arguments (no calls/assignments: substitution must not duplicate or
/// drop side effects) and fully direct uses (same arity; the lambda
/// never appears as a value, member base, or callee of another shape)
/// reduce; anything else (closures passed as ray-tracing callbacks,
/// recursion, multi-statement bodies) keeps today's loud `__lambda()`
/// path. Runs before `auto` deduction so reduced inits deduce normally.
/// Bounded fixpoint: reduced bodies can reveal further calls.
pub(crate) fn beta_reduce_lambdas(module: &Module) -> Module {
    let mut module = module.clone();
    for f in &mut module.functions {
        let params: Vec<String> = f.parameters.iter().map(|p| p.name.clone()).collect();
        for _ in 0..8 {
            if !beta_once(&mut f.body, &params) {
                break;
            }
        }
    }
    module
}

/// One beta pass over top-level statements; `true` when anything changed.
fn beta_once(body: &mut Vec<Stmt>, fn_params: &[String]) -> bool {
    // Collect `auto NAME = Lambda(params, [return E])` declarations.
    // Capture-less lambdas always qualify; reference captures (`[&]`,
    // `[&x]`) qualify because definition and direct calls share the
    // function scope (captured names bind identically at every use).
    // By-value captures (`[x]`) qualify subject to `capture_frozen`
    // (definition-site value provably current at every use); copy-all
    // (`[=]`) never qualifies.
    fn byval_captures(captures: &[String]) -> Option<Vec<String>> {
        // `[]` (none) and `[&]` (all references) qualify with no names;
        // `[&x]` pairs qualify as references; bare names are by-value
        // (freeze-checked later); anything with `=` does not qualify.
        let items: Vec<&str> =
            captures.iter().filter(|c| *c != ",").map(String::as_str).collect();
        let mut byval = Vec::new();
        let mut i = 0;
        while i < items.len() {
            if items[i] == "&" {
                if i + 1 < items.len() && items[i + 1] != "&" && items[i + 1] != "=" {
                    i += 2;
                } else {
                    i += 1;
                }
            } else if items[i] == "=" {
                return None;
            } else {
                byval.push(items[i].to_string());
                i += 1;
            }
        }
        Some(byval)
    }
    let mut lambdas: Vec<(String, Vec<String>, Vec<String>, Expr)> = Vec::new();
    // Collect `auto NAME = lambda` temps from every plain statement
    // list (lambdas nest in loops/branches, e.g. an upscale filter
    // kernel declared inside its dispatch loop). Lambda bodies
    // themselves are skipped (their temps cannot have rewritable
    // uses). Shadowing reuse of one name for two temps is unsound
    // here and stays loud via the use check below.
    fn collect(ss: &[Stmt], out: &mut Vec<(String, Vec<String>, Vec<String>, Expr)>) {
        for s in ss {
            match s {
                Stmt::VarDecl {
                    ty,
                    name,
                    init: Some(Expr::Lambda { captures, params, body }),
                    ..
                } => {
                    if !matches!(ty, Type::Named(n) if n == "auto") {
                        continue;
                    }
                    let Some(byval) = byval_captures(captures) else {
                        continue;
                    };
                    // Straight-line bodies (`uint m = ...; return f(m);`)
                    // normalize via pure-let elision; anything else
                    // (branches, loops, impure temps) stays loud.
                    let Some(ret) = single_return(body, params) else {
                        continue;
                    };
                    // No recursion through the lambda's own name.
                    if !expr_mentions(&ret, name) {
                        out.push((name.clone(), params.clone(), byval, ret));
                    }
                }
                Stmt::Block(inner) => collect(inner, out),
                Stmt::If { then_branch, else_branch, .. } => {
                    collect(then_branch, out);
                    if let Some(els) = else_branch {
                        collect(els, out);
                    }
                }
                Stmt::For { init, body, .. } => {
                    // For-init declarations stay put (a single-statement
                    // slot cannot splice a drop); only the body can host
                    // collectible temps.
                    let _ = init;
                    collect(body, out);
                }
                Stmt::While { body, .. } => collect(body, out),
                Stmt::Switch { cases, .. } => {
                    for c in cases {
                        collect(&c.body, out);
                    }
                }
                _ => {}
            }
        }
    }
    collect(body, &mut lambdas);
    if lambdas.is_empty() {
        return false;
    }
    // Every use of each lambda must be a direct arity-matching call with
    // pure arguments, and every by-value capture must be frozen;
    // otherwise that lambda stays loud.
    let mut usable: Vec<(String, Vec<String>, Vec<String>, Expr)> = Vec::new();
    for (name, params, byval, ret) in lambdas {
        if lambda_uses_direct(body, &name, params.len())
            && byval.iter().all(|x| capture_frozen(body, fn_params, x))
        {
            usable.push((name, params, byval, ret));
        }
    }
    if usable.is_empty() {
        return false;
    }
    // Rewrite calls, then drop the declarations from their owning
    // lists (usability guarantees at least one rewritten call, so a
    // non-empty `usable` always changes the body). Reference-captured
    // declarations drop exactly like capture-less ones: their uses
    // already inlined, the `auto = lambda` value cannot emit.
    for s in body.iter_mut() {
        beta_rewrite_stmt(s, &usable);
    }
    fn drop_decls(ss: &mut Vec<Stmt>, usable: &[(String, Vec<String>, Vec<String>, Expr)]) {
        ss.retain(|s| match s {
            Stmt::VarDecl {
                ty,
                name,
                init: Some(Expr::Lambda { .. }),
                ..
            } => {
                !(matches!(ty, Type::Named(n) if n == "auto")
                    && usable.iter().any(|(n, _, _, _)| n == name))
            }
            _ => true,
        });
        for s in ss.iter_mut() {
            match s {
                Stmt::Block(inner) => drop_decls(inner, usable),
                Stmt::If { then_branch, else_branch, .. } => {
                    drop_decls(then_branch, usable);
                    if let Some(els) = else_branch {
                        drop_decls(els, usable);
                    }
                }
                Stmt::For { body, .. } => {
                    drop_decls(body, usable);
                }
                Stmt::While { body, .. } => drop_decls(body, usable),
                Stmt::Switch { cases, .. } => {
                    for c in cases.iter_mut() {
                        drop_decls(&mut c.body, usable);
                    }
                }
                _ => {}
            }
        }
    }
    drop_decls(body, &usable);
    !usable.is_empty()
}

/// True when a by-value lambda capture `x` is frozen: the value at the
/// lambda definition provably equals the value at every direct call, so
/// inlining the name in place reads the identical value.
///
/// Definition and calls share one function scope, so it suffices that
/// `x` is bound exactly once (its single local declaration, or a
/// function parameter with no local redeclaration) and never mutated:
/// no assignment rooted at `x` (plain, compound, member, or element —
/// the parser desugars all of them to `Assign`), no shadowing
/// redeclaration or structured re-binding, and no call passing `x` (or
/// a path rooted at `x`) to a possibly-mutating callee. Calls to
/// provably side-effect-free value functions (pure math, wave/subgroup
/// reads, read-only texture sampling) cannot mutate and are exempt; any
/// other call with an `x`-rooted argument vetoes (fail-closed: a user
/// `out`-parameter could mutate it). Globals veto outright (a callee
/// could mutate them behind our back); locals and parameters cannot be
/// reached that way.
fn capture_frozen(body: &[Stmt], params: &[String], x: &str) -> bool {
    fn expr_root(e: &Expr) -> Option<&str> {
        match e {
            Expr::Variable(n) => Some(n),
            Expr::Member { object, .. } => expr_root(object),
            Expr::Index { object, .. } => expr_root(object),
            _ => None,
        }
    }
    fn callee_last(e: &Expr) -> Option<String> {
        match e {
            Expr::Variable(n) => n.rsplit("::").next().map(str::to_string),
            Expr::Member { member, .. } => Some(member.clone()),
            _ => None,
        }
    }
    /// Side-effect-free value calls: provably cannot mutate an argument.
    /// Anything else with an `x`-rooted argument vetoes.
    fn pure_callee(name: &str) -> bool {
        matches!(
            name.to_ascii_lowercase().as_str(),
            "sin" | "cos"
                | "tan"
                | "asin"
                | "acos"
                | "atan"
                | "atan2"
                | "sinh"
                | "cosh"
                | "tanh"
                | "pow"
                | "exp"
                | "exp2"
                | "log"
                | "log2"
                | "log10"
                | "sqrt"
                | "rsqrt"
                | "rcp"
                | "dot"
                | "cross"
                | "mul"
                | "normalize"
                | "length"
                | "distance"
                | "reflect"
                | "refract"
                | "saturate"
                | "clamp"
                | "lerp"
                | "mix"
                | "mad"
                | "min"
                | "max"
                | "min3"
                | "max3"
                | "abs"
                | "sign"
                | "floor"
                | "ceil"
                | "round"
                | "trunc"
                | "frac"
                | "fmod"
                | "smoothstep"
                | "step"
                | "faceforward"
                | "all"
                | "any"
                | "select"
                | "cndmask"
                | "countbits"
                | "popcnt"
                | "firstbithigh"
                | "firstbitlow"
                | "reversebits"
                | "asfloat"
                | "asint"
                | "asuint"
                | "isinf"
                | "isnan"
                | "isfinite"
                | "f16tof32"
                | "f32tof16"
                | "ballot"
                | "maskbitcnt"
                | "readfirstlane"
                | "waveactiveballot"
                | "waveactivecountbits"
                | "wavegetlaneindex"
                | "wavegetlanecount"
                | "waveisfirstlane"
                | "wavereadlanefirst"
                | "wavereadlaneat"
                | "crosslaneor"
                | "wave32"
                | "wave64"
                | "__v_cndmask_b32"
                | "__v_cmp_gt_u32"
                | "__v_ffbh_u32"
                | "__v_ffbl_b32"
                | "__maskbitcnt_u32"
                | "__maskbitcnt_hi_u32"
                | "__v_mbcnt_lo_u32_b32"
                | "__v_mbcnt_hi_u32_b32"
                | "sample"
                | "samplebias"
                | "samplegrad"
                | "samplelevel"
                | "samplecmp"
                | "samplecmplevelzero"
                | "load"
                | "gather"
                | "gatherred"
                | "gathergreen"
                | "gatherblue"
                | "gatheralpha"
                | "calculatelevelofdetail"
        )
    }
    fn expr_ok(e: &Expr, x: &str, decls: &mut u32) -> bool {
        match e {
            Expr::Assign { lhs, rhs } => {
                if expr_root(lhs).is_some_and(|r| r == x) {
                    return false;
                }
                expr_ok(lhs, x, decls) && expr_ok(rhs, x, decls)
            }
            Expr::Call { function, arguments } => {
                let pure = callee_last(function).is_some_and(|c| pure_callee(&c));
                if !pure && arguments.iter().any(|a| expr_root(a).is_some_and(|r| r == x)) {
                    return false;
                }
                expr_ok(function, x, decls) && arguments.iter().all(|a| expr_ok(a, x, decls))
            }
            Expr::Variable(_) | Expr::Number(_) | Expr::String(_) | Expr::Bool(_) => true,
            Expr::Member { object, .. } => expr_ok(object, x, decls),
            Expr::Index { object, index } => {
                expr_ok(object, x, decls) && expr_ok(index, x, decls)
            }
            Expr::Constructor { arguments, .. } => {
                arguments.iter().all(|a| expr_ok(a, x, decls))
            }
            Expr::Unary { expr, .. } => expr_ok(expr, x, decls),
            Expr::Binary { left, right, .. } => {
                expr_ok(left, x, decls) && expr_ok(right, x, decls)
            }
            Expr::Conditional { condition, then_branch, else_branch } => {
                expr_ok(condition, x, decls)
                    && expr_ok(then_branch, x, decls)
                    && expr_ok(else_branch, x, decls)
            }
            Expr::InitializerList(elems) => elems.iter().all(|a| expr_ok(a, x, decls)),
            // Nested lambda bodies are walked with the same counter: an
            // assignment to `x` there vetoes (conservative — the nested
            // lambda might never run — but fail-closed).
            Expr::Lambda { body, .. } => body.iter().all(|s| stmt_ok(s, x, decls)),
        }
    }
    fn stmt_ok(s: &Stmt, x: &str, decls: &mut u32) -> bool {
        match s {
            Stmt::Return(Some(e)) | Stmt::Expr(e) => expr_ok(e, x, decls),
            Stmt::Return(None) => true,
            Stmt::VarDecl { name, init, .. } => {
                if name == x {
                    *decls += 1;
                    if *decls > 1 {
                        return false;
                    }
                }
                init.as_ref().is_none_or(|e| expr_ok(e, x, decls))
            }
            // Structured re-binding shadows: reject outright.
            Stmt::Destructure { names, init, .. } => {
                if names.iter().any(|n| n == x) {
                    return false;
                }
                init.as_ref().is_none_or(|e| expr_ok(e, x, decls))
            }
            Stmt::Block(ss) => ss.iter().all(|s| stmt_ok(s, x, decls)),
            Stmt::If { condition, then_branch, else_branch } => {
                expr_ok(condition, x, decls)
                    && then_branch.iter().all(|s| stmt_ok(s, x, decls))
                    && else_branch.iter().flatten().all(|s| stmt_ok(s, x, decls))
            }
            Stmt::For { init, condition, update, body } => {
                stmt_ok(init, x, decls)
                    && condition.as_ref().is_none_or(|c| expr_ok(c, x, decls))
                    && update.as_ref().is_none_or(|u| expr_ok(u, x, decls))
                    && body.iter().all(|s| stmt_ok(s, x, decls))
            }
            Stmt::While { condition, body } => {
                expr_ok(condition, x, decls) && body.iter().all(|s| stmt_ok(s, x, decls))
            }
            Stmt::Switch { scrutinee, cases } => {
                expr_ok(scrutinee, x, decls)
                    && cases.iter().all(|c| {
                        c.label.as_ref().is_none_or(|l| expr_ok(l, x, decls))
                            && c.body.iter().all(|s| stmt_ok(s, x, decls))
                    })
            }
        }
    }
    let mut decls = 0u32;
    body.iter().all(|s| stmt_ok(s, x, &mut decls)) && (decls == 1 || (decls == 0 && params.contains(&x.to_string())))
}

/// Normalizes a lambda body to its return expression: a lone `return
/// E` qualifies as-is; a straight-line prefix of pure temporary
/// declarations (`T m = INIT;` with side-effect-free `INIT`) elides by
/// substituting each `INIT` for `m` (later declarations first, so
/// chains like `a = 1; b = a + 1; return b;` resolve). Bails (`None`)
/// on branches, loops, bare returns, impure temps (calls/assignments:
/// substitution must not duplicate or drop side effects), array or
/// storage-qualified temps (indexing/ persistence do not survive
/// textual elision), self-referential inits, and temps shadowing
/// parameters or each other.
fn single_return(body: &[Stmt], params: &[String]) -> Option<Expr> {
    fn init_pure(e: &Expr) -> bool {
        match e {
            Expr::Call { .. } | Expr::Assign { .. } | Expr::Lambda { .. } => false,
            Expr::Member { object, .. } => init_pure(object),
            Expr::Index { object, index } => init_pure(object) && init_pure(index),
            Expr::Constructor { arguments, .. } => arguments.iter().all(init_pure),
            Expr::Unary { expr, .. } => init_pure(expr),
            Expr::Binary { left, right, .. } => init_pure(left) && init_pure(right),
            Expr::Conditional { condition, then_branch, else_branch } => {
                init_pure(condition) && init_pure(then_branch) && init_pure(else_branch)
            }
            Expr::InitializerList(elems) => elems.iter().all(init_pure),
            Expr::Variable(_) | Expr::Number(_) | Expr::String(_) | Expr::Bool(_) => true,
        }
    }
    let mut ret: Option<Expr> = None;
    let mut lets: Vec<(String, Expr)> = Vec::new();
    for s in body {
        match s {
            Stmt::Return(Some(e)) => {
                if ret.is_some() {
                    return None;
                }
                ret = Some(e.clone());
            }
            Stmt::Return(None) => return None,
            Stmt::VarDecl { storage, name, array_size, init: Some(init), .. } => {
                if storage.is_some() || array_size.is_some() {
                    return None;
                }
                if params.contains(name) {
                    return None;
                }
                if expr_mentions(init, name) {
                    return None;
                }
                if lets.iter().any(|(n, _)| n == name) {
                    return None;
                }
                if !init_pure(init) {
                    return None;
                }
                lets.push((name.clone(), init.clone()));
            }
            _ => return None,
        }
    }
    let mut ret = ret?;
    for (n, init) in lets.iter().rev() {
        ret = beta_subst(&ret, std::slice::from_ref(n), std::slice::from_ref(init));
    }
    Some(ret)
}

/// True when `e` mentions `name` as a variable or callee anywhere.
fn expr_mentions(e: &Expr, name: &str) -> bool {
    match e {
        Expr::Variable(n) => n == name,
        Expr::Member { object, .. } => expr_mentions(object, name),
        Expr::Index { object, index } => {
            expr_mentions(object, name) || expr_mentions(index, name)
        }
        Expr::Call { function, arguments } => {
            expr_mentions(function, name) || arguments.iter().any(|a| expr_mentions(a, name))
        }
        Expr::Constructor { arguments, .. } => arguments.iter().any(|a| expr_mentions(a, name)),
        Expr::Assign { lhs, rhs } => expr_mentions(lhs, name) || expr_mentions(rhs, name),
        Expr::Unary { expr, .. } => expr_mentions(expr, name),
        Expr::Binary { left, right, .. } => {
            expr_mentions(left, name) || expr_mentions(right, name)
        }
        Expr::Conditional { condition, then_branch, else_branch } => {
            expr_mentions(condition, name)
                || expr_mentions(then_branch, name)
                || expr_mentions(else_branch, name)
        }
        Expr::InitializerList(elems) => elems.iter().any(|x| expr_mentions(x, name)),
        Expr::Lambda { body, .. } => body.iter().any(|s| stmt_mentions(s, name)),
        Expr::Number(_) | Expr::String(_) | Expr::Bool(_) => false,
    }
}

/// True when `s` mentions `name` anywhere.
fn stmt_mentions(s: &Stmt, name: &str) -> bool {
    match s {
        Stmt::Return(Some(e)) | Stmt::Expr(e) => expr_mentions(e, name),
        Stmt::Return(None) => false,
        Stmt::VarDecl { init: Some(e), .. } | Stmt::Destructure { init: Some(e), .. } => {
            expr_mentions(e, name)
        }
        Stmt::VarDecl { init: None, .. } | Stmt::Destructure { init: None, .. } => false,
        Stmt::Block(ss) => ss.iter().any(|x| stmt_mentions(x, name)),
        Stmt::If { condition, then_branch, else_branch } => {
            expr_mentions(condition, name)
                || then_branch.iter().any(|x| stmt_mentions(x, name))
                || else_branch.iter().flatten().any(|x| stmt_mentions(x, name))
        }
        Stmt::For { init, condition, update, body } => {
            stmt_mentions(init, name)
                || condition.as_ref().is_some_and(|c| expr_mentions(c, name))
                || update.as_ref().is_some_and(|u| expr_mentions(u, name))
                || body.iter().any(|x| stmt_mentions(x, name))
        }
        Stmt::While { condition, body } => {
            expr_mentions(condition, name) || body.iter().any(|x| stmt_mentions(x, name))
        }
        Stmt::Switch { scrutinee, cases } => {
            expr_mentions(scrutinee, name)
                || cases.iter().any(|c| {
                    c.label.as_ref().is_some_and(|l| expr_mentions(l, name))
                        || c.body.iter().any(|x| stmt_mentions(x, name))
                })
        }
    }
}

/// True when every mention of `name` in `body` is a direct call with
/// `arity` pure arguments (no calls/assignments/lambdas inside: beta
/// substitution must not duplicate or drop side effects).
fn lambda_uses_direct(body: &[Stmt], name: &str, arity: usize) -> bool {
    fn arg_pure(e: &Expr) -> bool {
        match e {
            Expr::Call { .. } | Expr::Assign { .. } | Expr::Lambda { .. } => false,
            Expr::Member { object, .. } => arg_pure(object),
            Expr::Index { object, index } => arg_pure(object) && arg_pure(index),
            Expr::Constructor { arguments, .. } => arguments.iter().all(arg_pure),
            Expr::Unary { expr, .. } => arg_pure(expr),
            Expr::Binary { left, right, .. } => arg_pure(left) && arg_pure(right),
            Expr::Conditional { condition, then_branch, else_branch } => {
                arg_pure(condition) && arg_pure(then_branch) && arg_pure(else_branch)
            }
            Expr::InitializerList(elems) => elems.iter().all(arg_pure),
            Expr::Variable(_) | Expr::Number(_) | Expr::String(_) | Expr::Bool(_) => true,
        }
    }
    fn expr_ok(e: &Expr, name: &str, arity: usize, found: &mut bool) -> bool {
        match e {
            Expr::Variable(n) if n == name => false,
            Expr::Member { object, .. } => expr_ok(object, name, arity, found),
            Expr::Index { object, index } => {
                expr_ok(object, name, arity, found) && expr_ok(index, name, arity, found)
            }
            Expr::Call { function, arguments } => {
                if let Expr::Variable(n) = function.as_ref() {
                    if n == name {
                        *found = true;
                        return arguments.len() == arity && arguments.iter().all(arg_pure);
                    }
                }
                expr_ok(function, name, arity, found)
                    && arguments.iter().all(|a| expr_ok(a, name, arity, found))
            }
            Expr::Constructor { arguments, .. } => {
                arguments.iter().all(|a| expr_ok(a, name, arity, found))
            }
            Expr::Assign { lhs, rhs } => {
                expr_ok(lhs, name, arity, found) && expr_ok(rhs, name, arity, found)
            }
            Expr::Unary { expr, .. } => expr_ok(expr, name, arity, found),
            Expr::Binary { left, right, .. } => {
                expr_ok(left, name, arity, found) && expr_ok(right, name, arity, found)
            }
            Expr::Conditional { condition, then_branch, else_branch } => {
                expr_ok(condition, name, arity, found)
                    && expr_ok(then_branch, name, arity, found)
                    && expr_ok(else_branch, name, arity, found)
            }
            Expr::InitializerList(elems) => elems.iter().all(|a| expr_ok(a, name, arity, found)),
            // A bare name inside a nested lambda body refers to an outer
            // capture, not a call: disqualify (conservative).
            Expr::Lambda { .. } => !expr_mentions(e, name),
            _ => true,
        }
    }
    fn stmt_ok(s: &Stmt, name: &str, arity: usize, found: &mut bool) -> bool {
        match s {
            Stmt::Return(Some(e)) | Stmt::Expr(e) => expr_ok(e, name, arity, found),
            Stmt::Return(None) => true,
            // The declaration itself is not a use.
            Stmt::VarDecl { name: n, init: Some(Expr::Lambda { .. }), .. }
                if n == name =>
            {
                true
            }
            Stmt::VarDecl { init: Some(e), .. } | Stmt::Destructure { init: Some(e), .. } => {
                expr_ok(e, name, arity, found)
            }
            Stmt::VarDecl { init: None, .. } | Stmt::Destructure { init: None, .. } => true,
            Stmt::Block(ss) => ss.iter().all(|x| stmt_ok(x, name, arity, found)),
            Stmt::If { condition, then_branch, else_branch } => {
                expr_ok(condition, name, arity, found)
                    && then_branch.iter().all(|x| stmt_ok(x, name, arity, found))
                    && else_branch.iter().flatten().all(|x| stmt_ok(x, name, arity, found))
            }
            Stmt::For { init, condition, update, body } => {
                stmt_ok(init, name, arity, found)
                    && condition.as_ref().is_some_and(|c| expr_ok(c, name, arity, found))
                    && update.as_ref().is_some_and(|u| expr_ok(u, name, arity, found))
                    && body.iter().all(|x| stmt_ok(x, name, arity, found))
            }
            Stmt::While { condition, body } => {
                expr_ok(condition, name, arity, found)
                    && body.iter().all(|x| stmt_ok(x, name, arity, found))
            }
            Stmt::Switch { scrutinee, cases } => {
                expr_ok(scrutinee, name, arity, found)
                    && cases.iter().all(|c| {
                        c.label.as_ref().is_none_or(|l| expr_ok(l, name, arity, found))
                            && c.body.iter().all(|x| stmt_ok(x, name, arity, found))
                    })
            }
        }
    }
    let mut seen = false;
    let ok = body.iter().all(|s| stmt_ok(s, name, arity, &mut seen));
    ok && seen
}

/// Substitutes `args` for `params` in `e` (first-match positional).
/// Does not descend into nested lambdas binding the same names.
fn beta_subst(e: &Expr, params: &[String], args: &[Expr]) -> Expr {
    let lookup = |n: &str| {
        params
            .iter()
            .position(|p| p == n)
            .and_then(|i| args.get(i))
            .cloned()
    };
    match e {
        Expr::Variable(n) => lookup(n).unwrap_or_else(|| e.clone()),
        Expr::Member { object, member } => Expr::Member {
            object: Box::new(beta_subst(object, params, args)),
            member: member.clone(),
        },
        Expr::Index { object, index } => Expr::Index {
            object: Box::new(beta_subst(object, params, args)),
            index: Box::new(beta_subst(index, params, args)),
        },
        Expr::Call { function, arguments } => Expr::Call {
            function: Box::new(beta_subst(function, params, args)),
            arguments: arguments.iter().map(|a| beta_subst(a, params, args)).collect(),
        },
        Expr::Constructor { ty, arguments } => Expr::Constructor {
            ty: ty.clone(),
            arguments: arguments.iter().map(|a| beta_subst(a, params, args)).collect(),
        },
        Expr::Assign { lhs, rhs } => Expr::Assign {
            lhs: Box::new(beta_subst(lhs, params, args)),
            rhs: Box::new(beta_subst(rhs, params, args)),
        },
        Expr::Unary { op, expr } => Expr::Unary {
            op: *op,
            expr: Box::new(beta_subst(expr, params, args)),
        },
        Expr::Binary { left, op, right } => Expr::Binary {
            left: Box::new(beta_subst(left, params, args)),
            right: Box::new(beta_subst(right, params, args)),
            op: *op,
        },
        Expr::Conditional { condition, then_branch, else_branch } => Expr::Conditional {
            condition: Box::new(beta_subst(condition, params, args)),
            then_branch: Box::new(beta_subst(then_branch, params, args)),
            else_branch: Box::new(beta_subst(else_branch, params, args)),
        },
        Expr::InitializerList(elems) => Expr::InitializerList(
            elems.iter().map(|a| beta_subst(a, params, args)).collect(),
        ),
        Expr::Lambda { captures, params: inner, body } => {
            // Shadowed names keep the inner binding.
            if inner.iter().any(|p| params.contains(p)) {
                e.clone()
            } else {
                Expr::Lambda {
                    captures: captures.clone(),
                    params: inner.clone(),
                    body: body.clone(),
                }
            }
        }
        Expr::Number(_) | Expr::String(_) | Expr::Bool(_) => e.clone(),
    }
}

/// Rewrites direct calls of usable lambdas in a statement.
fn beta_rewrite_stmt(s: &mut Stmt, usable: &[(String, Vec<String>, Vec<String>, Expr)]) {
    fn rewrite_expr(e: &mut Expr, usable: &[(String, Vec<String>, Vec<String>, Expr)]) {
        // Post-order: reduce inner calls first (fixpoint at the top).
        match e {
            Expr::Member { object, .. } => rewrite_expr(object, usable),
            Expr::Index { object, index } => {
                rewrite_expr(object, usable);
                rewrite_expr(index, usable);
            }
            Expr::Call { function, arguments } => {
                rewrite_expr(function, usable);
                for a in arguments.iter_mut() {
                    rewrite_expr(a, usable);
                }
            }
            Expr::Constructor { arguments, .. } => {
                for a in arguments.iter_mut() {
                    rewrite_expr(a, usable);
                }
            }
            Expr::Assign { lhs, rhs } => {
                rewrite_expr(lhs, usable);
                rewrite_expr(rhs, usable);
            }
            Expr::Unary { expr, .. } => rewrite_expr(expr, usable),
            Expr::Binary { left, right, .. } => {
                rewrite_expr(left, usable);
                rewrite_expr(right, usable);
            }
            Expr::Conditional { condition, then_branch, else_branch } => {
                rewrite_expr(condition, usable);
                rewrite_expr(then_branch, usable);
                rewrite_expr(else_branch, usable);
            }
            Expr::InitializerList(elems) => {
                for x in elems.iter_mut() {
                    rewrite_expr(x, usable);
                }
            }
            Expr::Lambda { .. }
            | Expr::Variable(_)
            | Expr::Number(_)
            | Expr::String(_)
            | Expr::Bool(_) => {}
        }
        if let Expr::Call { function, arguments } = e {
            if let Expr::Variable(n) = function.as_ref() {
                if let Some((_, params, _, ret)) = usable.iter().find(|(u, _, _, _)| u == n) {
                    if arguments.len() == params.len() {
                        *e = beta_subst(ret, params, arguments);
                    }
                }
            }
        }
    }
    match s {
        Stmt::Return(Some(e)) | Stmt::Expr(e) => rewrite_expr(e, usable),
        Stmt::Return(None) => {}
        Stmt::VarDecl { init: Some(e), .. } | Stmt::Destructure { init: Some(e), .. } => {
            rewrite_expr(e, usable)
        }
        Stmt::VarDecl { init: None, .. } | Stmt::Destructure { init: None, .. } => {}
        Stmt::Block(ss) => {
            for x in ss {
                beta_rewrite_stmt(x, usable);
            }
        }
        Stmt::If { condition, then_branch, else_branch } => {
            rewrite_expr(condition, usable);
            for x in then_branch {
                beta_rewrite_stmt(x, usable);
            }
            if let Some(els) = else_branch {
                for x in els {
                    beta_rewrite_stmt(x, usable);
                }
            }
        }
        Stmt::For { init, condition, update, body } => {
            beta_rewrite_stmt(init, usable);
            if let Some(c) = condition {
                rewrite_expr(c, usable);
            }
            if let Some(u) = update {
                rewrite_expr(u, usable);
            }
            for x in body {
                beta_rewrite_stmt(x, usable);
            }
        }
        Stmt::While { condition, body } => {
            rewrite_expr(condition, usable);
            for x in body {
                beta_rewrite_stmt(x, usable);
            }
        }
        Stmt::Switch { scrutinee, cases } => {
            rewrite_expr(scrutinee, usable);
            for c in cases {
                if let Some(label) = &mut c.label {
                    rewrite_expr(label, usable);
                }
                for x in &mut c.body {
                    beta_rewrite_stmt(x, usable);
                }
            }
        }
    }
}

/// Inlines immediately-invoked `ExecuteInPixelOrder` critical sections
/// (`ExecuteInPixelOrder([&]() { ... });`): the lambda literal is both
/// defined and called atomically, so its body statements replace the
/// call in place (captures bind in the same scope, exactly). The
/// raster-ordering guarantee itself has no DXC/Vulkan spelling and is
/// not enforced (documented approximation; validation never executes).
/// Named callables (`ExecuteInPixelOrder(f)`) and lambdas with
/// parameters stay loud (today's path).
pub(crate) fn inline_execute_in_pixel_order(module: &Module) -> Module {
    fn inline_stmt(s: &mut Stmt) -> bool {
        match s {
            Stmt::Expr(Expr::Call { function, arguments }) => {
                if let Expr::Variable(f) = function.as_ref() {
                    if f.eq_ignore_ascii_case("executeinpixelorder") {
                        if let [Expr::Lambda { params, body, .. }] = arguments.as_slice() {
                            if params.is_empty() {
                                *s = Stmt::Block(body.clone());
                                return true;
                            }
                        }
                    }
                }
                false
            }
            Stmt::Block(ss) => {
                let mut changed = false;
                for x in ss.iter_mut() {
                    changed |= inline_stmt(x);
                }
                changed
            }
            Stmt::If { then_branch, else_branch, .. } => {
                let mut changed = false;
                for x in then_branch.iter_mut() {
                    changed |= inline_stmt(x);
                }
                if let Some(els) = else_branch {
                    for x in els.iter_mut() {
                        changed |= inline_stmt(x);
                    }
                }
                changed
            }
            Stmt::For { init, body, .. } => {
                let mut changed = inline_stmt(init);
                for x in body.iter_mut() {
                    changed |= inline_stmt(x);
                }
                changed
            }
            Stmt::While { body, .. } => {
                let mut changed = false;
                for x in body.iter_mut() {
                    changed |= inline_stmt(x);
                }
                changed
            }
            Stmt::Switch { cases, .. } => {
                let mut changed = false;
                for c in cases.iter_mut() {
                    for x in c.body.iter_mut() {
                        changed |= inline_stmt(x);
                    }
                }
                changed
            }
            _ => false,
        }
    }
    let mut module = module.clone();
    for f in &mut module.functions {
        for _ in 0..8 {
            let mut changed = false;
            for s in f.body.iter_mut() {
                changed |= inline_stmt(s);
            }
            if !changed {
                break;
            }
        }
    }
    module
}

/// Desugars integer-texture sampling (`uint v = tex.Sample(smp, uv)`,
/// `x = tex.SampleLevel(smp, uv, lod)` on 2D int/uint textures): integer
/// texels cannot filter, so HLSL rejects `Sample` outright. Each becomes
/// a `GetDimensions` fetch into a function-top `uint2` plus a mip-0 (or
/// explicit-lod) `Load` (`tex.Load(int3(int2(uv * float2(dims)), lod))`).
/// Only top-level statement positions (var-init, assignment rhs, return)
/// rewrite; nested uses stay loud. `auto` targets take the element type.
/// Runs before `auto` deduction so rewritten inits deduce normally.
pub(crate) fn desugar_uint_sample(module: &Module) -> Module {
    // Tables mirror `deduce_auto_locals` (same inference needs).
    let mut structs: HashMap<&str, HashMap<&str, (&Type, Option<u32>)>> = HashMap::new();
    for s in &module.structs {
        structs.insert(
            s.name.as_str(),
            s.fields
                .iter()
                .map(|f| (f.name.as_str(), (&f.ty, f.array_size)))
                .collect(),
        );
    }
    let mut free_rets: HashMap<&str, Vec<((usize, usize), &Type)>> = HashMap::new();
    let mut method_rets: HashMap<(&str, &str), Vec<((usize, usize), &Type)>> = HashMap::new();
    for f in &module.functions {
        if matches!(&f.return_type, Type::Named(n) if n == "auto") {
            continue;
        }
        let total = f.parameters.len();
        let mut required = 0;
        for p in &f.parameters {
            if p.default.is_some() {
                break;
            }
            required += 1;
        }
        let range = (required, total);
        match f.scope.as_deref() {
            Some(s) if structs.contains_key(s) => {
                method_rets
                    .entry((s, f.name.as_str()))
                    .or_default()
                    .push((range, &f.return_type));
            }
            _ => {
                free_rets
                    .entry(f.name.as_str())
                    .or_default()
                    .push((range, &f.return_type));
            }
        }
    }
    let mut roots: HashMap<&str, (Type, Option<u32>)> = HashMap::new();
    for g in &module.globals {
        roots.insert(g.name.as_str(), (g.ty.clone(), g.array_size));
    }
    for r in &module.resources {
        roots.insert(
            r.name.as_str(),
            (
                Type::Resource {
                    kind: r.kind.clone(),
                    access: crate::ast::ResourceAccess::ReadOnly,
                    element: Box::new(r.element_ty.clone().unwrap_or(Type::UInt)),
                },
                None,
            ),
        );
    }

    // Integer 2D texture element (uint/int scalar or vector) or None.
    fn int_tex_element(
        object: &Expr,
        env: &HashMap<String, (Type, Option<u32>)>,
        structs: &HashMap<&str, HashMap<&str, (&Type, Option<u32>)>>,
        free_rets: &HashMap<&str, Vec<((usize, usize), &Type)>>,
        method_rets: &HashMap<(&str, &str), Vec<((usize, usize), &Type)>>,
    ) -> Option<Type> {
        let ty = infer_local(object, env, structs, free_rets, method_rets, 0)?;
        if let Type::Resource { kind, element, .. } = &ty {
            // Kind spellings vary (`Texture2D` from resources,
            // `texture2d` normalized elsewhere).
            if kind.to_ascii_lowercase().replace('_', "") == "texture2d"
                && is_int_element(element)
            {
                return Some(element.as_ref().clone());
            }
        }
        None
    }
    fn is_int_element(ty: &Type) -> bool {
        match ty {
            Type::Int | Type::UInt => true,
            Type::Vector { base, .. } => matches!(
                base,
                crate::ast::ScalarType::Int | crate::ast::ScalarType::UInt
            ),
            _ => false,
        }
    }
    // Builds `tex.Load(int3(int2((uv) * float2(DIMS)), lod))`.
    fn load_expr(tex: &Expr, uv: &Expr, lod: Expr, dims: &str) -> Expr {
        let dims_var = Expr::Variable(dims.to_string());
        let dims_f = Expr::Constructor {
            ty: Type::Vector { base: crate::ast::ScalarType::Float, width: 2 },
            arguments: vec![dims_var],
        };
        let scaled = Expr::Binary {
            left: Box::new(uv.clone()),
            op: crate::ast::BinaryOp::Mul,
            right: Box::new(dims_f),
        };
        let coord = Expr::Constructor {
            ty: Type::Vector { base: crate::ast::ScalarType::Int, width: 2 },
            arguments: vec![scaled],
        };
        let loc = Expr::Constructor {
            ty: Type::Vector { base: crate::ast::ScalarType::Int, width: 3 },
            arguments: vec![coord, lod],
        };
        Expr::Call {
            function: Box::new(Expr::Member {
                object: Box::new(tex.clone()),
                member: "Load".to_string(),
            }),
            arguments: vec![loc],
        }
    }
    // `tex.GetDimensions(DIMS.x, DIMS.y);` statement.
    fn dims_fetch(tex: &Expr, dims: &str) -> Stmt {
        let dims_var = Expr::Variable(dims.to_string());
        let comp = |m: &str| Expr::Member { object: Box::new(dims_var.clone()), member: m.to_string() };
        Stmt::Expr(Expr::Call {
            function: Box::new(Expr::Member {
                object: Box::new(tex.clone()),
                member: "GetDimensions".to_string(),
            }),
            arguments: vec![comp("x"), comp("y")],
        })
    }
    // Tries to rewrite a top-level sample call; returns the replacement
    // init expression plus element type on success.
    // Matches `tex.Sample(sampler, uv)` / `tex.SampleLevel(sampler, uv,
    // lod)` on 2D int/uint textures; returns (tex, uv, lod, element).
    // The caller builds the Load form plus the dims fetch.
    fn try_sample(
        e: &Expr,
        env: &HashMap<String, (Type, Option<u32>)>,
        structs: &HashMap<&str, HashMap<&str, (&Type, Option<u32>)>>,
        free_rets: &HashMap<&str, Vec<((usize, usize), &Type)>>,
        method_rets: &HashMap<(&str, &str), Vec<((usize, usize), &Type)>>,
    ) -> Option<(Expr, Expr, Expr, Type)> {
        let Expr::Call { function, arguments } = e else {
            return None;
        };
        let Expr::Member { object, member } = function.as_ref() else {
            return None;
        };
        let ml = member.to_ascii_lowercase();
        // `Sample(sampler, uv)` (mip 0), `SampleLevel(sampler, uv, lod)`,
        // or PSSL `SampleLOD(sampler, uv, lod)` (mapped to SampleLevel
        // only later in lowering — match it here too).
        let (uv, lod) = match (ml.as_str(), arguments.as_slice()) {
            ("sample", [_, uv]) => (
                uv.clone(),
                Expr::Number("0".to_string()),
            ),
            ("samplelevel", [_, uv, lod]) | ("samplelod", [_, uv, lod]) => {
                (uv.clone(), lod.clone())
            }
            _ => return None,
        };
        let element = int_tex_element(object, env, structs, free_rets, method_rets)?;
        Some((object.as_ref().clone(), uv, lod, element))
    }

    // Declared names in a body (dims-temporary collision check).
    fn collect_declared(stmts: &[Stmt], taken: &mut HashSet<String>) {
        for s in stmts {
            match s {
                Stmt::VarDecl { name, .. } => {
                    taken.insert(name.clone());
                }
                Stmt::Block(ss) => collect_declared(ss, taken),
                Stmt::If { then_branch, else_branch, .. } => {
                    collect_declared(then_branch, taken);
                    if let Some(els) = else_branch {
                        collect_declared(els, taken);
                    }
                }
                Stmt::For { init, body, .. } => {
                    collect_declared(std::slice::from_ref(init), taken);
                    collect_declared(body, taken);
                }
                Stmt::While { body, .. } => collect_declared(body, taken),
                Stmt::Switch { cases, .. } => {
                    for c in cases {
                        collect_declared(&c.body, taken);
                    }
                }
                _ => {}
            }
        }
    }
    // Recursive rewrite of a statement list (locals register in order;
    // nested scopes clone the env). Inserts the dims fetch before each
    // rewritten statement. Returns whether anything hit.
    #[allow(clippy::too_many_arguments)]
    fn process(
        stmts: &mut Vec<Stmt>,
        env: &mut HashMap<String, (Type, Option<u32>)>,
        structs: &HashMap<&str, HashMap<&str, (&Type, Option<u32>)>>,
        free_rets: &HashMap<&str, Vec<((usize, usize), &Type)>>,
        method_rets: &HashMap<(&str, &str), Vec<((usize, usize), &Type)>>,
        dims: &str,
    ) -> bool {
        let mut hit_any = false;
        let mut i = 0;
        while i < stmts.len() {
            if let Stmt::VarDecl { ty, name, array_size, .. } = &stmts[i] {
                env.insert(name.clone(), (ty.clone(), *array_size));
            }
            let hit = match &stmts[i] {
                Stmt::VarDecl { init: Some(e), .. } | Stmt::Return(Some(e)) => {
                    try_sample(e, env, structs, free_rets, method_rets).is_some()
                }
                Stmt::Expr(e) => match e {
                    Expr::Assign { rhs, .. } => {
                        try_sample(rhs, env, structs, free_rets, method_rets).is_some()
                    }
                    _ => false,
                },
                _ => false,
            };
            if hit {
                let (tex, uv, lod, elem) = match &stmts[i] {
                    Stmt::VarDecl { init: Some(e), .. } | Stmt::Return(Some(e)) => {
                        try_sample(e, env, structs, free_rets, method_rets).unwrap()
                    }
                    Stmt::Expr(e) => match e {
                        Expr::Assign { rhs, .. } => {
                            try_sample(rhs, env, structs, free_rets, method_rets).unwrap()
                        }
                        _ => unreachable!("guarded by shape match above"),
                    },
                    _ => unreachable!("guarded by shape match above"),
                };
                let fetch = dims_fetch(&tex, dims);
                let load = load_expr(&tex, &uv, lod, dims);
                // Insert the fetch first (the original shifts to i+1),
                // then rewrite it.
                stmts.insert(i, fetch);
                match &mut stmts[i + 1] {
                    Stmt::VarDecl { ty, init, .. } => {
                        *init = Some(load);
                        if matches!(ty, Type::Named(n) if n == "auto") {
                            *ty = elem;
                        }
                    }
                    Stmt::Return(e) => {
                        *e = Some(load);
                    }
                    Stmt::Expr(e) => match e {
                        Expr::Assign { rhs, .. } => {
                            *rhs = Box::new(load);
                        }
                        _ => unreachable!("guarded by shape match above"),
                    },
                    _ => unreachable!("guarded by shape match above"),
                }
                i += 1;
                hit_any = true;
            } else {
                match &mut stmts[i] {
                    Stmt::Block(ss) => {
                        let mut child = env.clone();
                        if process(ss, &mut child, structs, free_rets, method_rets, dims) {
                            hit_any = true;
                        }
                    }
                    Stmt::If { then_branch, else_branch, .. } => {
                        let mut child = env.clone();
                        if process(then_branch, &mut child, structs, free_rets, method_rets, dims)
                        {
                            hit_any = true;
                        }
                        if let Some(els) = else_branch {
                            let mut child = env.clone();
                            if process(els, &mut child, structs, free_rets, method_rets, dims) {
                                hit_any = true;
                            }
                        }
                    }
                    Stmt::For { body, .. } => {
                        // For-inits cannot host a fetch insert (single
                        // statement, no sibling slot); bodies cover the
                        // realistic shapes.
                        let mut child = env.clone();
                        if process(body, &mut child, structs, free_rets, method_rets, dims) {
                            hit_any = true;
                        }
                    }
                    Stmt::While { body, .. } => {
                        let mut child = env.clone();
                        if process(body, &mut child, structs, free_rets, method_rets, dims) {
                            hit_any = true;
                        }
                    }
                    Stmt::Switch { cases, .. } => {
                        for c in cases.iter_mut() {
                            let mut child = env.clone();
                            if process(&mut c.body, &mut child, structs, free_rets, method_rets, dims)
                            {
                                hit_any = true;
                            }
                        }
                    }
                    _ => {}
                }
            }
            i += 1;
        }
        hit_any
    }

    let mut module = module.clone();
    for f in &mut module.functions {
        let mut env: HashMap<String, (Type, Option<u32>)> = HashMap::new();
        for p in &f.parameters {
            env.insert(p.name.clone(), (p.ty.clone(), p.array_size));
        }
        for (name, (ty, arr)) in &roots {
            env.entry(name.to_string()).or_insert_with(|| (ty.clone(), *arr));
        }
        let mut taken: HashSet<String> = env.keys().cloned().collect();
        collect_declared(&f.body, &mut taken);
        let mut dims = "__sample_dims".to_string();
        let mut n = 0u32;
        while taken.contains(&dims) {
            n += 1;
            dims = format!("__sample_dims_{n}");
        }
        if process(&mut f.body, &mut env, &structs, &free_rets, &method_rets, &dims) {
            f.body.insert(
                0,
                Stmt::VarDecl {
                    storage: None,
                    ty: Type::Vector { base: crate::ast::ScalarType::UInt, width: 2 },
                    name: dims,
                    array_size: None,
                    init: None,
                },
            );
        }
    }
    module
}

/// Monomorphizes template and `auto`-parameter functions.
///
/// A free function with template type params (`template<typename T>`)
/// or `auto` params, whose body avoids `T::` qualified uses, clones per
/// distinct deduced binding: implicit type params (and `auto`) deduce
/// from call argument types, explicit literal template args
/// (`f<true>(x)`, preserved in the callee spelling at parse) bind
/// leading value params positionally. The clone substitutes concrete
/// types (params, return, VarDecls, constructors) and value literals
/// (synthetic value-param locals take the literal init too, so any
/// missed use still reads correctly); calls retarget to it. Unchanged
/// shapes (scoped methods, `T::` bodies, unbound params, arity
/// mismatches, opaque arg types) keep today's loud path. Bounded
/// fixpoint (nested templates resolve as outer clones materialize);
/// fully-retargeted template defs drop (dead stripping would also
/// remove them, but explicit dropping keeps `T::` residue from ever
/// reaching DXC). Mangled callees with no template target rewrite back
/// to the bare name (today's behavior for non-templates).
/// Monomorphizes template structs with all-typename params.
///
/// A use spelling `BcnEncodeSrt<uint2>` (preserved in
/// `Module::template_spellings`) clones the struct with `T` substituted
/// in member types (`RW_Texture2D<T>` → `RWTexture2D<uint2>`); use sites
/// (params for now) retarget to the clone. Only typename args that parse
/// as plain types qualify (scalars/vectors/matrices/bare names; nested
/// template-ids stay `Named` verbatim for union resolution); value args,
/// `T::` members, template bases, and methods mentioning template params
/// keep today's loud path. Runs before function monomorphization so its
/// tables see the clones.
/// Truncates vector/scalar constructors with excess components
/// (`uint2(a, b_vec2)`, `uint2(a, uarr2)`): PSSL takes the first N
/// scalar components (scalar conversion takes `.x`; arrays flatten
/// element-wise), while HLSL rejects the excess outright. Only fires
/// when every argument has an exactly-known scalar/vector/array shape
/// (opaque, matrix, resource, and unknown shapes keep today's loud
/// path); splats and exact counts are untouched.
/// Folds integer texel offsets on texture sampling into UV arithmetic.
///
/// Vulkan SPIR-V only allows the `Offset` image operand on
/// `OpImageGather`; HLSL `Sample`/`SampleLevel` calls carrying an integer
/// offset lower to `Offset` on `OpImageSample*`, which DXC accepts but the
/// Vulkan validator rejects
/// (`VUID-RuntimeSpirv-Offset-10213`). The semantically identical,
/// Vulkan-legal form adds the offset in texels to the UV:
/// `tex.Sample(s, uv, off)` becomes
/// `tex.Sample(s, uv + float2(off)/float2(__dims))` with `__dims` from a
/// function-entry `tex.GetDimensions(__dims.x, __dims.y)` (dims are
/// loop-invariant and `GetDimensions` is pure, so hoisting to function
/// entry is sound). Only fires on exact offset arities (`Sample`=3,
/// `SampleLOD`/`SampleLevel`=4, `SampleCmp`/`SampleCmpLevelZero`=4, offset
/// last); anything else keeps today's loud path. Method spellings are
/// preserved so the `SampleLOD`→`SampleLevel` mapping still fires
/// downstream.
pub(crate) fn fold_sample_offsets(module: &Module) -> Module {
    use crate::ast::{BinaryOp, ScalarType, Stmt};

    // Structural key for texture objects (variables + member chains).
    fn expr_key(e: &Expr) -> Option<String> {
        match e {
            Expr::Variable(n) => Some(n.clone()),
            Expr::Member { object, member } => {
                Some(format!("{}.{}", expr_key(object)?, member))
            }
            _ => None,
        }
    }
    // UV argument index for offsetted samples (offset is always last).
    fn uv_index(member: &str, nargs: usize) -> Option<usize> {
        match member.to_ascii_lowercase().as_str() {
            "sample" if nargs == 3 => Some(1),
            "samplelod" | "samplelevel" if nargs == 4 => Some(1),
            "samplecmp" | "samplecmplevelzero" if nargs == 4 => Some(1),
            _ => None,
        }
    }
    fn float2(e: Expr) -> Expr {
        Expr::Constructor {
            ty: Type::Vector {
                base: ScalarType::Float,
                width: 2,
            },
            arguments: vec![e],
        }
    }
    struct Ctx {
        // tex key (or per-site fallback) -> dims temp name.
        dims_for: HashMap<String, String>,
        // dims temp name -> tex object expr (for the hoisted GetDimensions).
        tex_for_temp: HashMap<String, Expr>,
        // fallback counter for unkeyable tex objects.
        site_counter: usize,
        taken: HashSet<String>,
        counter: usize,
    }
    impl Ctx {
        fn fresh_temp(&mut self) -> String {
            loop {
                let n = format!("__sample_dims{}", self.counter);
                self.counter += 1;
                if !self.taken.contains(&n) {
                    self.taken.insert(n.clone());
                    return n;
                }
            }
        }
        fn dims_temp(&mut self, tex: &Expr) -> String {
            let key = expr_key(tex).unwrap_or_else(|| {
                let k = format!("__site{}", self.site_counter);
                self.site_counter += 1;
                k
            });
            if let Some(t) = self.dims_for.get(&key) {
                return t.clone();
            }
            let temp = self.fresh_temp();
            self.dims_for.insert(key, temp.clone());
            self.tex_for_temp.insert(temp.clone(), tex.clone());
            temp
        }
    }
    fn walk_expr(e: &mut Expr, ctx: &mut Ctx) {
        // Post-order: rewrite children first.
        match e {
            Expr::Call { function, arguments } => {
                walk_expr(function, ctx);
                for a in arguments.iter_mut() {
                    walk_expr(a, ctx);
                }
            }
            Expr::Member { object, .. } => walk_expr(object, ctx),
            Expr::Index { object, index } => {
                walk_expr(object, ctx);
                walk_expr(index, ctx);
            }
            Expr::Constructor { arguments, .. } => {
                for a in arguments.iter_mut() {
                    walk_expr(a, ctx);
                }
            }
            Expr::Assign { lhs, rhs } => {
                walk_expr(lhs, ctx);
                walk_expr(rhs, ctx);
            }
            Expr::Unary { expr, .. } => walk_expr(expr, ctx),
            Expr::Binary { left, right, .. } => {
                walk_expr(left, ctx);
                walk_expr(right, ctx);
            }
            Expr::Conditional { condition, then_branch, else_branch } => {
                walk_expr(condition, ctx);
                walk_expr(then_branch, ctx);
                walk_expr(else_branch, ctx);
            }
            Expr::InitializerList(elems) => {
                for x in elems.iter_mut() {
                    walk_expr(x, ctx);
                }
            }
            Expr::Lambda { body, .. } => {
                for s in body.iter_mut() {
                    walk_stmt(s, ctx);
                }
            }
            Expr::Variable(_) | Expr::Number(_) | Expr::String(_) | Expr::Bool(_) => {}
        }
        // Then this call: offsetted sample -> UV-folded sample.
        if let Expr::Call { function, arguments } = e {
            let (tex, member) = match function.as_ref() {
                Expr::Member { object, member } => (object.as_ref().clone(), member.clone()),
                _ => return,
            };
            let Some(uvi) = uv_index(&member, arguments.len()) else {
                return;
            };
            let offset = arguments.pop().expect("offset is last");
            let uv = std::mem::replace(
                &mut arguments[uvi],
                Expr::Number("0".to_string()),
            );
            let temp = ctx.dims_temp(&tex);
            let folded = Expr::Binary {
                left: Box::new(uv),
                op: BinaryOp::Add,
                right: Box::new(Expr::Binary {
                    left: Box::new(float2(offset)),
                    op: BinaryOp::Div,
                    right: Box::new(float2(Expr::Variable(temp))),
                }),
            };
            arguments[uvi] = folded;
        }
    }
    fn walk_stmt(s: &mut Stmt, ctx: &mut Ctx) {
        match s {
            Stmt::Return(Some(e)) | Stmt::Expr(e) => walk_expr(e, ctx),
            Stmt::Return(None) => {}
            Stmt::VarDecl { init: Some(e), .. } | Stmt::Destructure { init: Some(e), .. } => {
                walk_expr(e, ctx)
            }
            Stmt::VarDecl { init: None, .. } | Stmt::Destructure { init: None, .. } => {}
            Stmt::Block(ss) => {
                for x in ss.iter_mut() {
                    walk_stmt(x, ctx);
                }
            }
            Stmt::If { condition, then_branch, else_branch } => {
                walk_expr(condition, ctx);
                for x in then_branch.iter_mut() {
                    walk_stmt(x, ctx);
                }
                if let Some(els) = else_branch {
                    for x in els.iter_mut() {
                        walk_stmt(x, ctx);
                    }
                }
            }
            Stmt::For { init, condition, update, body } => {
                walk_stmt(init, ctx);
                if let Some(c) = condition {
                    walk_expr(c, ctx);
                }
                if let Some(u) = update {
                    walk_expr(u, ctx);
                }
                for x in body.iter_mut() {
                    walk_stmt(x, ctx);
                }
            }
            Stmt::While { condition, body } => {
                walk_expr(condition, ctx);
                for x in body.iter_mut() {
                    walk_stmt(x, ctx);
                }
            }
            Stmt::Switch { scrutinee, cases } => {
                walk_expr(scrutinee, ctx);
                for c in cases.iter_mut() {
                    if let Some(l) = &mut c.label {
                        walk_expr(l, ctx);
                    }
                    for x in c.body.iter_mut() {
                        walk_stmt(x, ctx);
                    }
                }
            }
        }
    }
    // Dims temps referenced (transitively) by a statement/expression.
    fn temps_in_expr(e: &Expr, temps: &HashSet<String>, out: &mut HashSet<String>) {
        match e {
            Expr::Variable(n) => {
                if temps.contains(n) {
                    out.insert(n.clone());
                }
            }
            Expr::Call { function, arguments } => {
                temps_in_expr(function, temps, out);
                for a in arguments {
                    temps_in_expr(a, temps, out);
                }
            }
            Expr::Member { object, .. } => temps_in_expr(object, temps, out),
            Expr::Index { object, index } => {
                temps_in_expr(object, temps, out);
                temps_in_expr(index, temps, out);
            }
            Expr::Constructor { arguments, .. } => {
                for a in arguments {
                    temps_in_expr(a, temps, out);
                }
            }
            Expr::Assign { lhs, rhs } => {
                temps_in_expr(lhs, temps, out);
                temps_in_expr(rhs, temps, out);
            }
            Expr::Unary { expr, .. } => temps_in_expr(expr, temps, out),
            Expr::Binary { left, right, .. } => {
                temps_in_expr(left, temps, out);
                temps_in_expr(right, temps, out);
            }
            Expr::Conditional { condition, then_branch, else_branch } => {
                temps_in_expr(condition, temps, out);
                temps_in_expr(then_branch, temps, out);
                temps_in_expr(else_branch, temps, out);
            }
            Expr::InitializerList(elems) => {
                for x in elems {
                    temps_in_expr(x, temps, out);
                }
            }
            Expr::Lambda { body, .. } => {
                for s in body {
                    temps_in_stmt(s, temps, out);
                }
            }
            Expr::Number(_) | Expr::String(_) | Expr::Bool(_) => {}
        }
    }
    fn temps_in_stmt(s: &Stmt, temps: &HashSet<String>, out: &mut HashSet<String>) {
        match s {
            Stmt::Return(Some(e)) | Stmt::Expr(e) => temps_in_expr(e, temps, out),
            Stmt::Return(None) => {}
            Stmt::VarDecl { init: Some(e), .. } | Stmt::Destructure { init: Some(e), .. } => {
                temps_in_expr(e, temps, out)
            }
            Stmt::VarDecl { init: None, .. } | Stmt::Destructure { init: None, .. } => {}
            Stmt::Block(ss) => {
                for x in ss {
                    temps_in_stmt(x, temps, out);
                }
            }
            Stmt::If { condition, then_branch, else_branch } => {
                temps_in_expr(condition, temps, out);
                for x in then_branch {
                    temps_in_stmt(x, temps, out);
                }
                if let Some(els) = else_branch {
                    for x in els {
                        temps_in_stmt(x, temps, out);
                    }
                }
            }
            Stmt::For { init, condition, update, body } => {
                temps_in_stmt(init, temps, out);
                if let Some(c) = condition {
                    temps_in_expr(c, temps, out);
                }
                if let Some(u) = update {
                    temps_in_expr(u, temps, out);
                }
                for x in body {
                    temps_in_stmt(x, temps, out);
                }
            }
            Stmt::While { condition, body } => {
                temps_in_expr(condition, temps, out);
                for x in body {
                    temps_in_stmt(x, temps, out);
                }
            }
            Stmt::Switch { scrutinee, cases } => {
                temps_in_expr(scrutinee, temps, out);
                for c in cases {
                    if let Some(l) = &c.label {
                        temps_in_expr(l, temps, out);
                    }
                    for x in &c.body {
                        temps_in_stmt(x, temps, out);
                    }
                }
            }
        }
    }
    fn make_getdims(tex: &Expr, temp: &str) -> Stmt {
        Stmt::Expr(Expr::Call {
            function: Box::new(Expr::Member {
                object: Box::new(tex.clone()),
                member: "GetDimensions".to_string(),
            }),
            arguments: vec![
                Expr::Member {
                    object: Box::new(Expr::Variable(temp.to_string())),
                    member: "x".to_string(),
                },
                Expr::Member {
                    object: Box::new(Expr::Variable(temp.to_string())),
                    member: "y".to_string(),
                },
            ],
        })
    }
    // Child statement-lists of a statement (for scoped placement recursion).
    fn child_blocks(s: &mut Stmt) -> Vec<&mut Vec<Stmt>> {
        match s {
            Stmt::Block(ss) => vec![ss],
            Stmt::If { then_branch, else_branch, .. } => {
                let mut v = vec![then_branch];
                if let Some(els) = else_branch {
                    v.push(els);
                }
                v
            }
            Stmt::For { body, .. } => vec![body],
            Stmt::While { body, .. } => vec![body],
            Stmt::Switch { cases, .. } => cases.iter_mut().map(|c| &mut c.body).collect(),
            _ => Vec::new(),
        }
    }
    // Scope-aware `GetDimensions` placement: walk a block in order;
    // before a statement needing dims for texkeys not yet available in
    // this block prefix, insert the call (same block, so the tex — which
    // the statement references — is in scope). Child blocks inherit
    // availability (outer calls dominate); their insertions stay local.
    // Deterministic: texkeys sorted before inserting.
    fn place_block(
        stmts: &mut Vec<Stmt>,
        temps: &HashSet<String>,
        tex_for_temp: &HashMap<String, Expr>,
        key_for_temp: &HashMap<String, String>,
        available: &mut HashSet<String>,
    ) {
        let mut i = 0;
        while i < stmts.len() {
            let mut need_temps: HashSet<String> = HashSet::new();
            temps_in_stmt(&stmts[i], temps, &mut need_temps);
            let mut need_keys: HashMap<String, (String, Expr)> = HashMap::new();
            for t in need_temps {
                let (Some(k), Some(tex)) = (key_for_temp.get(&t), tex_for_temp.get(&t)) else {
                    continue;
                };
                if available.contains(k) {
                    continue;
                }
                need_keys.entry(k.clone()).or_insert((t.clone(), tex.clone()));
            }
            if !need_keys.is_empty() {
                let mut keys: Vec<String> = need_keys.keys().cloned().collect();
                keys.sort();
                let mut insert = Vec::with_capacity(keys.len());
                for k in &keys {
                    let (t, tex) = &need_keys[k];
                    insert.push(make_getdims(tex, t));
                    available.insert(k.clone());
                }
                stmts.splice(i..i, insert);
                i += keys.len();
            }
            // Recurse into children with inherited availability.
            let mut child_available = available.clone();
            for child in child_blocks(&mut stmts[i]) {
                place_block(child, temps, tex_for_temp, key_for_temp, &mut child_available);
            }
            i += 1;
        }
    }
    fn collect_names(
        stmts: &[Stmt],
        params: &[String],
        out: &mut HashSet<String>,
    ) {
        for p in params {
            out.insert(p.clone());
        }
        fn stmt(s: &Stmt, out: &mut HashSet<String>) {
            match s {
                Stmt::VarDecl { name, .. } => {
                    out.insert(name.clone());
                }
                Stmt::Destructure { names, .. } => {
                    out.extend(names.iter().cloned());
                }
                Stmt::Block(ss) => {
                    for x in ss {
                        stmt(x, out);
                    }
                }
                Stmt::If { then_branch, else_branch, .. } => {
                    for x in then_branch {
                        stmt(x, out);
                    }
                    if let Some(els) = else_branch {
                        for x in els {
                            stmt(x, out);
                        }
                    }
                }
                Stmt::For { init, body, .. } => {
                    stmt(init, out);
                    for x in body {
                        stmt(x, out);
                    }
                }
                Stmt::While { body, .. } => {
                    for x in body {
                        stmt(x, out);
                    }
                }
                Stmt::Switch { cases, .. } => {
                    for c in cases {
                        for x in &c.body {
                            stmt(x, out);
                        }
                    }
                }
                _ => {}
            }
        }
        fn lambda_params_expr(e: &Expr, out: &mut HashSet<String>) {
            match e {
                Expr::Lambda { params, body, .. } => {
                    out.extend(params.iter().cloned());
                    collect_names(body, &[], out);
                }
                Expr::Call { function, arguments } => {
                    lambda_params_expr(function, out);
                    for a in arguments {
                        lambda_params_expr(a, out);
                    }
                }
                Expr::Member { object, .. } => lambda_params_expr(object, out),
                Expr::Index { object, index } => {
                    lambda_params_expr(object, out);
                    lambda_params_expr(index, out);
                }
                Expr::Constructor { arguments, .. } => {
                    for a in arguments {
                        lambda_params_expr(a, out);
                    }
                }
                Expr::Assign { lhs, rhs } => {
                    lambda_params_expr(lhs, out);
                    lambda_params_expr(rhs, out);
                }
                Expr::Unary { expr, .. } => lambda_params_expr(expr, out),
                Expr::Binary { left, right, .. } => {
                    lambda_params_expr(left, out);
                    lambda_params_expr(right, out);
                }
                Expr::Conditional { condition, then_branch, else_branch } => {
                    lambda_params_expr(condition, out);
                    lambda_params_expr(then_branch, out);
                    lambda_params_expr(else_branch, out);
                }
                Expr::InitializerList(elems) => {
                    for x in elems {
                        lambda_params_expr(x, out);
                    }
                }
                _ => {}
            }
        }
        for s in stmts {
            stmt(s, out);
            // Lambda params (nested scopes) must not collide either.
            match s {
                Stmt::Return(Some(e)) | Stmt::Expr(e) => lambda_params_expr(e, out),
                Stmt::VarDecl { init: Some(e), .. }
                | Stmt::Destructure { init: Some(e), .. } => lambda_params_expr(e, out),
                _ => {}
            }
        }
    }

    let mut module = module.clone();
    let mut globals: HashSet<String> = HashSet::new();
    for g in &module.globals {
        globals.insert(g.name.clone());
    }
    for r in &module.resources {
        globals.insert(r.name.clone());
    }
    for f in module.functions.iter_mut() {
        let mut taken = globals.clone();
        collect_names(
            &f.body,
            &f.parameters.iter().map(|p| p.name.clone()).collect::<Vec<_>>(),
            &mut taken,
        );
        let mut ctx = Ctx {
            dims_for: HashMap::new(),
            tex_for_temp: HashMap::new(),
            site_counter: 0,
            taken,
            counter: 0,
        };
        for s in f.body.iter_mut() {
            walk_stmt(s, &mut ctx);
        }
        if ctx.tex_for_temp.is_empty() {
            continue;
        }
        // Temp declarations go to function entry (no tex reference, always
        // safe). `GetDimensions` calls are placed by scoped walk below:
        // immediately before the statement needing them, in the same
        // block (the tex is referenced there, so it is in scope; valid
        // code declares it before use). This never hoists past a texture
        // local's declaration, fixing entry-hoist regressions.
        let mut temps: Vec<String> = ctx.tex_for_temp.keys().cloned().collect();
        temps.sort();
        let mut decls = Vec::with_capacity(temps.len());
        for temp in &temps {
            decls.push(Stmt::VarDecl {
                storage: None,
                ty: Type::Vector {
                    base: ScalarType::UInt,
                    width: 2,
                },
                name: temp.clone(),
                array_size: None,
                init: None,
            });
        }
        let temp_set: HashSet<String> = temps.into_iter().collect();
        // temp -> tex key (for per-block availability dedup).
        let mut key_for_temp: HashMap<String, String> = HashMap::new();
        for (k, t) in &ctx.dims_for {
            key_for_temp.insert(t.clone(), k.clone());
        }
        let mut available: HashSet<String> = HashSet::new();
        place_block(
            &mut f.body,
            &temp_set,
            &ctx.tex_for_temp,
            &key_for_temp,
            &mut available,
        );
        let mut body = std::mem::take(&mut f.body);
        let mut new_body = decls;
        new_body.append(&mut body);
        f.body = new_body;
    }
    module
}

pub(crate) fn truncate_vector_constructors(module: &Module) -> Module {
    use crate::ast::Stmt;
    // Scalar component width of a type, if exactly known.
    fn width_of_ty(ty: &Type) -> Option<usize> {
        match ty {
            Type::Bool | Type::Int | Type::UInt | Type::Float | Type::Half | Type::Double => {
                Some(1)
            }
            Type::Vector { width, .. } => Some(*width as usize),
            _ => None,
        }
    }
    // Component sources for an argument: (expr, width). Arrays expand
    // element-wise (1D scalar/vector arrays of known size only).
    fn expand_arg(
        e: &Expr,
        env: &HashMap<String, (Type, Option<u32>)>,
        structs: &HashMap<&str, HashMap<&str, (&Type, Option<u32>)>>,
        free_rets: &HashMap<&str, Vec<((usize, usize), &Type)>>,
        method_rets: &HashMap<(&str, &str), Vec<((usize, usize), &Type)>>,
    ) -> Option<Vec<(Expr, usize)>> {
        // Whole-array value: expand known-size 1D scalar/vector arrays.
        let as_array = match e {
            Expr::Variable(name) => env.get(name).and_then(|(t, s)| s.map(|n| (t.clone(), n))),
            Expr::Member { object, member } => {
                let obj = infer_local(object, env, structs, free_rets, method_rets, 0)?;
                if let Type::Named(s) = &obj {
                    structs.get(s.as_str()).and_then(|fields| fields.get(member.as_str())).and_then(
                        |(t, s)| s.map(|n| ((*t).clone(), n)),
                    )
                } else {
                    None
                }
            }
            _ => None,
        };
        if let Some((ty, size)) = as_array {
            if size == 0 {
                return None;
            }
            let elem_w = width_of_ty(&ty)?;
            let mut out = Vec::new();
            for i in 0..size {
                out.push((
                    Expr::Index {
                        object: Box::new(e.clone()),
                        index: Box::new(Expr::Number(i.to_string())),
                    },
                    elem_w,
                ));
            }
            return Some(out);
        }
        let ty = infer_local(e, env, structs, free_rets, method_rets, 0)?;
        Some(vec![(e.clone(), width_of_ty(&ty)?)])
    }
    const SWIZZLE: &str = "xyzw";
    fn take_components(arg: &(Expr, usize), n: usize) -> Expr {
        let (e, w) = arg;
        if *w <= n {
            return e.clone();
        }
        Expr::Member {
            object: Box::new(e.clone()),
            member: SWIZZLE[..n].to_string(),
        }
    }
    fn rewrite_expr(
        e: &mut Expr,
        env: &HashMap<String, (Type, Option<u32>)>,
        structs: &HashMap<&str, HashMap<&str, (&Type, Option<u32>)>>,
        free_rets: &HashMap<&str, Vec<((usize, usize), &Type)>>,
        method_rets: &HashMap<(&str, &str), Vec<((usize, usize), &Type)>>,
    ) -> bool {
        let mut changed = false;
        match e {
            Expr::Constructor { ty, arguments } => {
                // Recurse first (inner excess truncates before outer
                // counting, so widths stay accurate).
                for a in arguments.iter_mut() {
                    changed |= rewrite_expr(a, env, structs, free_rets, method_rets);
                }
                let expected = match ty {
                    Type::Vector { width, .. } => Some(*width as usize),
                    Type::Bool
                    | Type::Int
                    | Type::UInt
                    | Type::Float
                    | Type::Half
                    | Type::Double => Some(1),
                    _ => None,
                };
                let Some(expected) = expected else {
                    return false;
                };
                // Widths via inference (constructors report their type).
                let mut parts: Vec<(Expr, usize)> = Vec::new();
                let mut ok = true;
                for a in arguments.iter() {
                    match expand_arg(a, env, structs, free_rets, method_rets) {
                        Some(mut p) => parts.append(&mut p),
                        None => {
                            ok = false;
                            break;
                        }
                    }
                }
                if !ok {
                    return false;
                }
                let supplied: usize = parts.iter().map(|(_, w)| w).sum();
                if supplied <= expected {
                    return false;
                }
                let mut remaining = expected;
                let mut rebuilt = Vec::new();
                for part in &parts {
                    if remaining == 0 {
                        break;
                    }
                    rebuilt.push(take_components(part, remaining.min(part.1)));
                    remaining = remaining.saturating_sub(part.1);
                }
                *arguments = rebuilt;
                changed = true;
            }
            Expr::Call { function, arguments } => {
                rewrite_expr(function, env, structs, free_rets, method_rets);
                for a in arguments.iter_mut() {
                    changed |= rewrite_expr(a, env, structs, free_rets, method_rets);
                }
                return changed;
            }
            Expr::Member { object, .. } => {
                return rewrite_expr(object, env, structs, free_rets, method_rets);
            }
            Expr::Index { object, index } => {
                changed |= rewrite_expr(object, env, structs, free_rets, method_rets);
                changed |= rewrite_expr(index, env, structs, free_rets, method_rets);
                return changed;
            }
            Expr::Assign { lhs, rhs } => {
                changed |= rewrite_expr(lhs, env, structs, free_rets, method_rets);
                changed |= rewrite_expr(rhs, env, structs, free_rets, method_rets);
                return changed;
            }
            Expr::Unary { expr, .. } => {
                return rewrite_expr(expr, env, structs, free_rets, method_rets);
            }
            Expr::Binary { left, right, .. } => {
                changed |= rewrite_expr(left, env, structs, free_rets, method_rets);
                changed |= rewrite_expr(right, env, structs, free_rets, method_rets);
                return changed;
            }
            Expr::Conditional { condition, then_branch, else_branch } => {
                changed |= rewrite_expr(condition, env, structs, free_rets, method_rets);
                changed |= rewrite_expr(then_branch, env, structs, free_rets, method_rets);
                changed |= rewrite_expr(else_branch, env, structs, free_rets, method_rets);
                return changed;
            }
            Expr::InitializerList(elems) => {
                for x in elems.iter_mut() {
                    changed |= rewrite_expr(x, env, structs, free_rets, method_rets);
                }
                return changed;
            }
            Expr::Lambda { body, .. } => {
                for s in body.iter_mut() {
                    changed |= rewrite_stmt(s, env, structs, free_rets, method_rets);
                }
                return changed;
            }
            Expr::Variable(_) | Expr::Number(_) | Expr::String(_) | Expr::Bool(_) => {
                return false;
            }
        }
        changed
    }
    fn rewrite_stmt(
        s: &mut Stmt,
        env: &HashMap<String, (Type, Option<u32>)>,
        structs: &HashMap<&str, HashMap<&str, (&Type, Option<u32>)>>,
        free_rets: &HashMap<&str, Vec<((usize, usize), &Type)>>,
        method_rets: &HashMap<(&str, &str), Vec<((usize, usize), &Type)>>,
    ) -> bool {
        match s {
            Stmt::Return(Some(e)) | Stmt::Expr(e) => {
                rewrite_expr(e, env, structs, free_rets, method_rets)
            }
            Stmt::Return(None) => false,
            Stmt::VarDecl { init: Some(e), .. } | Stmt::Destructure { init: Some(e), .. } => {
                rewrite_expr(e, env, structs, free_rets, method_rets)
            }
            Stmt::VarDecl { init: None, .. } | Stmt::Destructure { init: None, .. } => false,
            Stmt::Block(ss) => {
                let mut c = false;
                for x in ss.iter_mut() {
                    c |= rewrite_stmt(x, env, structs, free_rets, method_rets);
                }
                c
            }
            Stmt::If { condition, then_branch, else_branch } => {
                let mut c = rewrite_expr(condition, env, structs, free_rets, method_rets);
                for x in then_branch.iter_mut() {
                    c |= rewrite_stmt(x, env, structs, free_rets, method_rets);
                }
                if let Some(els) = else_branch {
                    for x in els.iter_mut() {
                        c |= rewrite_stmt(x, env, structs, free_rets, method_rets);
                    }
                }
                c
            }
            Stmt::For { init, condition, update, body } => {
                let mut c = rewrite_stmt(init, env, structs, free_rets, method_rets);
                if let Some(e) = condition {
                    c |= rewrite_expr(e, env, structs, free_rets, method_rets);
                }
                if let Some(e) = update {
                    c |= rewrite_expr(e, env, structs, free_rets, method_rets);
                }
                for x in body.iter_mut() {
                    c |= rewrite_stmt(x, env, structs, free_rets, method_rets);
                }
                c
            }
            Stmt::While { condition, body } => {
                let mut c = rewrite_expr(condition, env, structs, free_rets, method_rets);
                for x in body.iter_mut() {
                    c |= rewrite_stmt(x, env, structs, free_rets, method_rets);
                }
                c
            }
            Stmt::Switch { scrutinee, cases } => {
                let mut c = rewrite_expr(scrutinee, env, structs, free_rets, method_rets);
                for case in cases.iter_mut() {
                    if let Some(l) = &mut case.label {
                        c |= rewrite_expr(l, env, structs, free_rets, method_rets);
                    }
                    for x in case.body.iter_mut() {
                        c |= rewrite_stmt(x, env, structs, free_rets, method_rets);
                    }
                }
                c
            }
        }
    }
    let mut module = module.clone();
    // Owned tables (cloned once; bodies mutate below without borrows).
    let structs: HashMap<String, HashMap<String, (Type, Option<u32>)>> = module
        .structs
        .iter()
        .map(|s| {
            (
                s.name.clone(),
                s.fields.iter().map(|f| (f.name.clone(), (f.ty.clone(), f.array_size))).collect(),
            )
        })
        .collect();
    let mut free_rets: HashMap<String, Vec<((usize, usize), Type)>> = HashMap::new();
    let mut method_rets: HashMap<(String, String), Vec<((usize, usize), Type)>> = HashMap::new();
    for g in &module.functions {
        if matches!(&g.return_type, Type::Named(n) if n == "auto") {
            continue;
        }
        let total = g.parameters.len();
        let mut required = 0;
        for p in &g.parameters {
            if p.default.is_some() {
                break;
            }
            required += 1;
        }
        let range = (required, total);
        match g.scope.as_deref() {
            Some(s) => {
                method_rets
                    .entry((s.to_string(), g.name.clone()))
                    .or_default()
                    .push((range, g.return_type.clone()));
            }
            None => {
                free_rets.entry(g.name.clone()).or_default().push((range, g.return_type.clone()));
            }
        }
    }
    let mut roots: HashMap<String, (Type, Option<u32>)> = HashMap::new();
    for g in &module.globals {
        roots.insert(g.name.clone(), (g.ty.clone(), g.array_size));
    }
    for r in &module.resources {
        roots.insert(
            r.name.clone(),
            (
                Type::Resource {
                    kind: r.kind.clone(),
                    access: crate::ast::ResourceAccess::ReadOnly,
                    element: Box::new(r.element_ty.clone().unwrap_or(Type::UInt)),
                },
                None,
            ),
        );
    }
    for f in module.functions.iter_mut() {
        // Borrowed inference views over the owned tables.
        let mut sf: HashMap<&str, HashMap<&str, (&Type, Option<u32>)>> = HashMap::new();
        for (k, v) in &structs {
            sf.insert(k.as_str(), v.iter().map(|(fk, (ft, fa))| (fk.as_str(), (ft, *fa))).collect());
        }
        let mut fr: HashMap<&str, Vec<((usize, usize), &Type)>> = HashMap::new();
        for (k, v) in &free_rets {
            fr.insert(k.as_str(), v.iter().map(|(r, t)| (*r, t)).collect());
        }
        let mut mr: HashMap<(&str, &str), Vec<((usize, usize), &Type)>> = HashMap::new();
        for ((a, b), v) in &method_rets {
            mr.insert((a.as_str(), b.as_str()), v.iter().map(|(r, t)| (*r, t)).collect());
        }
        // Env: params + roots + top-level locals (nested-block locals
        // invisible — inference misses there stay loud).
        let mut env: HashMap<String, (Type, Option<u32>)> = HashMap::new();
        for p in &f.parameters {
            env.insert(p.name.clone(), (p.ty.clone(), p.array_size));
        }
        for (name, (ty, arr)) in &roots {
            env.entry(name.clone()).or_insert_with(|| (ty.clone(), *arr));
        }
        for s in &f.body {
            if let Stmt::VarDecl { ty, name, array_size, .. } = s {
                env.insert(name.clone(), (ty.clone(), *array_size));
            }
        }
        for s in f.body.iter_mut() {
            rewrite_stmt(s, &env, &sf, &fr, &mr);
        }
    }
    module
}

pub(crate) fn monomorphize_structs(module: &Module) -> Module {
    // Parse a template argument to a Type (None when not plain).
    fn parse_arg(text: &str) -> Option<Type> {
        let lower = text.to_ascii_lowercase();
        let scalar = |name: &str| match name {
            "float" => Some(Type::Float),
            "half" => Some(Type::Half),
            "int" => Some(Type::Int),
            "uint" => Some(Type::UInt),
            "bool" => Some(Type::Bool),
            "double" => Some(Type::Double),
            _ => None,
        };
        if let Some(t) = scalar(&lower) {
            return Some(t);
        }
        // Vectors/matrices (`float3`, `float4x4`, `half2`, ...).
        let (base_name, rest) = match lower.find('x') {
            Some(pos) => {
                let (b, r) = lower.split_at(pos);
                (b, Some(&r[1..]))
            }
            None => (lower.as_str(), None),
        };
        let base = match base_name {
            "float" => crate::ast::ScalarType::Float,
            "half" => crate::ast::ScalarType::Half,
            "int" => crate::ast::ScalarType::Int,
            "uint" => crate::ast::ScalarType::UInt,
            "bool" => crate::ast::ScalarType::Bool,
            "double" => return None,
            _ => {
                // Bare names (structs, nested template-ids verbatim).
                // Literals are values, not types (reject `false`, `0`).
                let lower = text.to_ascii_lowercase();
                if lower == "true" || lower == "false" {
                    return None;
                }
                if !text.is_empty()
                    && text.chars().all(|c| {
                        c.is_alphanumeric() || c == '_' || c == ':' || c == '<' || c == '>' || c == ','
                    })
                    && text.chars().any(|c| c.is_alphabetic() || c == '_' || c == ':')
                {
                    return Some(Type::Named(text.to_string()));
                }
                return None;
            }
        };
        match rest {
            None => None,
            Some(dims) => {
                if let Ok(w) = dims.parse::<u8>() {
                    if (2..=4).contains(&w) {
                        return Some(Type::Vector { base, width: w });
                    }
                    return None;
                }
                if let Some((r, c)) = dims.split_once('x') {
                    if let (Ok(rows), Ok(cols)) = (r.parse::<u8>(), c.parse::<u8>()) {
                        if (1..=4).contains(&rows) && (1..=4).contains(&cols) {
                            return Some(Type::Matrix { base, rows, columns: cols });
                        }
                    }
                }
                None
            }
        }
    }
    // Split `Name<args>` into base + top-level-comma args.
    fn split_template_id(spelling: &str) -> Option<(String, Vec<String>)> {
        let (base, rest) = spelling.split_once('<')?;
        if base.is_empty() || !rest.ends_with('>') {
            return None;
        }
        let inner = &rest[..rest.len() - 1];
        if inner.contains('<') || inner.contains('>') {
            return None;
        }
        let args: Vec<String> =
            inner.split(',').map(|s| s.trim().to_string()).collect();
        if args.is_empty() || args.iter().any(|a| a.is_empty()) {
            return None;
        }
        Some((base.to_string(), args))
    }
    // Substitute template typenames in a member type.
    fn subst_member_ty(ty: &Type, map: &HashMap<String, Type>) -> Type {
        match ty {
            Type::Named(n) => map.get(n).cloned().unwrap_or_else(|| ty.clone()),
            Type::Resource { kind, access, element } => Type::Resource {
                kind: kind.clone(),
                access: *access,
                element: Box::new(subst_member_ty(element, map)),
            },
            _ => ty.clone(),
        }
    }
    // True when a type mentions `T::` (phase-2 shapes stay loud).
    fn ty_qualified(t: &Type, typenames: &HashSet<String>) -> bool {
        match t {
            Type::Named(n) => typenames.iter().any(|p| n.starts_with(&format!("{p}::"))),
            Type::Resource { element, .. } => ty_qualified(element, typenames),
            _ => false,
        }
    }
    let mut module = module.clone();
    // Template structs with all-typename params.
    struct STpl {
        tparams: Vec<String>,
    }
    let mut stemplates: HashMap<String, STpl> = HashMap::new();
    for s in &module.structs {
        if s.template_params.is_empty() {
            continue;
        }
        if s.template_params.iter().any(|(_, is_value)| *is_value) {
            continue;
        }
        if !s.bases.is_empty() {
            continue;
        }
        let typenames: HashSet<String> =
            s.template_params.iter().map(|(n, _)| n.clone()).collect();
        if s.fields.iter().any(|f| ty_qualified(&f.ty, &typenames)) {
            continue;
        }
        stemplates.insert(
            s.name.clone(),
            STpl {
                tparams: s.template_params.iter().map(|(n, _)| n.clone()).collect(),
            },
        );
    }
    if stemplates.is_empty() {
        return module;
    }
    // Methods mentioning template params veto their struct (scope still
    // points at the original; cloning methods is out of scope).
    {
        let mut veto: HashSet<String> = HashSet::new();
        for f in &module.functions {
            let Some(scope) = f.scope.as_deref() else {
                continue;
            };
            let Some(tpl) = stemplates.get(scope) else {
                continue;
            };
            let names: HashSet<String> = tpl.tparams.iter().cloned().collect();
            // Bare uses of a template param anywhere in the method.
            fn mentions(e: &Expr, names: &HashSet<String>) -> bool {
                match e {
                    Expr::Variable(n) => names.contains(n),
                    Expr::Call { function, arguments } => {
                        mentions(function, names)
                            || arguments.iter().any(|a| mentions(a, names))
                    }
                    Expr::Constructor { ty, arguments } => {
                        ty_qualified(ty, names)
                            || arguments.iter().any(|a| mentions(a, names))
                    }
                    Expr::Member { object, .. } => mentions(object, names),
                    Expr::Index { object, index } => {
                        mentions(object, names) || mentions(index, names)
                    }
                    Expr::Assign { lhs, rhs } => mentions(lhs, names) || mentions(rhs, names),
                    Expr::Unary { expr, .. } => mentions(expr, names),
                    Expr::Binary { left, right, .. } => {
                        mentions(left, names) || mentions(right, names)
                    }
                    Expr::Conditional { condition, then_branch, else_branch } => {
                        mentions(condition, names)
                            || mentions(then_branch, names)
                            || mentions(else_branch, names)
                    }
                    Expr::InitializerList(elems) => elems.iter().any(|a| mentions(a, names)),
                    Expr::Lambda { body, .. } => body.iter().any(|s| {
                        fn sm(s: &Stmt, names: &HashSet<String>) -> bool {
                            match s {
                                Stmt::Return(Some(e)) | Stmt::Expr(e) => mentions(e, names),
                                Stmt::Return(None) => false,
                                Stmt::VarDecl { ty, init, .. }
                                | Stmt::Destructure { ty, init, .. } => {
                                    ty_qualified(ty, names)
                                        || init.as_ref().is_some_and(|e| mentions(e, names))
                                }
                                Stmt::Block(ss) => ss.iter().any(|x| sm(x, names)),
                                Stmt::If { condition, then_branch, else_branch } => {
                                    mentions(condition, names)
                                        || then_branch.iter().any(|x| sm(x, names))
                                        || else_branch.iter().flatten().any(|x| sm(x, names))
                                }
                                Stmt::For { init, condition, update, body } => {
                                    sm(init, names)
                                        || condition.as_ref().is_some_and(|c| mentions(c, names))
                                        || update.as_ref().is_some_and(|u| mentions(u, names))
                                        || body.iter().any(|x| sm(x, names))
                                }
                                Stmt::While { condition, body } => {
                                    mentions(condition, names)
                                        || body.iter().any(|x| sm(x, names))
                                }
                                Stmt::Switch { scrutinee, cases } => {
                                    mentions(scrutinee, names)
                                        || cases.iter().any(|c| {
                                            c.label.as_ref().is_some_and(|l| mentions(l, names))
                                                || c.body.iter().any(|x| sm(x, names))
                                        })
                                }
                            }
                        }
                        sm(s, names)
                    }),
                    Expr::Number(_) | Expr::String(_) | Expr::Bool(_) => false,
                }
            }
            let mut hit = f
                .parameters
                .iter()
                .any(|p| ty_qualified(&p.ty, &names) || mentions_param_ty(&p.ty, &names));
            fn mentions_param_ty(ty: &Type, names: &HashSet<String>) -> bool {
                match ty {
                    Type::Named(n) => names.contains(n),
                    Type::Resource { element, .. } => mentions_param_ty(element, names),
                    _ => false,
                }
            }
            if !hit {
                hit = ty_qualified(&f.return_type, &names);
            }
            if !hit {
                hit = f.body.iter().any(|s| {
                    fn sm2(s: &Stmt, names: &HashSet<String>) -> bool {
                        match s {
                            Stmt::Return(Some(e)) | Stmt::Expr(e) => mentions(e, names),
                            Stmt::Return(None) => false,
                            Stmt::VarDecl { ty, init, .. }
                            | Stmt::Destructure { ty, init, .. } => {
                                mentions_param_ty(ty, names)
                                    || init.as_ref().is_some_and(|e| mentions(e, names))
                            }
                            Stmt::Block(ss) => ss.iter().any(|x| sm2(x, names)),
                            Stmt::If { condition, then_branch, else_branch } => {
                                mentions(condition, names)
                                    || then_branch.iter().any(|x| sm2(x, names))
                                    || else_branch.iter().flatten().any(|x| sm2(x, names))
                            }
                            Stmt::For { init, condition, update, body } => {
                                sm2(init, names)
                                    || condition.as_ref().is_some_and(|c| mentions(c, names))
                                    || update.as_ref().is_some_and(|u| mentions(u, names))
                                    || body.iter().any(|x| sm2(x, names))
                            }
                            Stmt::While { condition, body } => {
                                mentions(condition, names)
                                    || body.iter().any(|x| sm2(x, names))
                            }
                            Stmt::Switch { scrutinee, cases } => {
                                mentions(scrutinee, names)
                                    || cases.iter().any(|c| {
                                        c.label.as_ref().is_some_and(|l| mentions(l, names))
                                            || c.body.iter().any(|x| sm2(x, names))
                                    })
                            }
                        }
                    }
                    sm2(s, &names)
                });
            }
            if hit {
                veto.insert(scope.to_string());
            }
        }
        for v in veto {
            stemplates.remove(&v);
        }
    }
    if stemplates.is_empty() {
        return module;
    }
    // Collect distinct (base, args) uses: per-variable declaration
    // spellings plus module-wide observations (covers globals and
    // locals the declaration table misses). Retargeting below applies
    // a clone only under the single-spelling rule per base (several
    // spellings stay loud: per-use disambiguation without full dataflow
    // would guess).
    let mut observed: HashMap<String, HashSet<Vec<String>>> = HashMap::new();
    for (base, args_text) in &module.observed_template_ids {
        if !stemplates.contains_key(base.as_str()) {
            continue;
        }
        // Framework and NGG-compat names never clone (lowering matches
        // them by name: `sce::...` framework mapping, `CullData` NGG).
        if base.starts_with("sce::") || base.ends_with("::CullData") || base == "CullData" {
            continue;
        }
        // Re-split (observation stores raw args text).
        let mut args: Vec<String> = Vec::new();
        let mut depth = 0usize;
        let mut cur = String::new();
        let mut ok_split = true;
        for c in args_text.chars() {
            match c {
                '<' => {
                    depth += 1;
                    cur.push(c);
                }
                '>' => {
                    if depth == 0 {
                        ok_split = false;
                        break;
                    }
                    depth -= 1;
                    cur.push(c);
                }
                ',' if depth == 0 => {
                    args.push(cur.trim().to_string());
                    cur = String::new();
                }
                _ => cur.push(c),
            }
        }
        if !ok_split || depth != 0 {
            continue;
        }
        args.push(cur.trim().to_string());
        if args.iter().any(|a| a.is_empty()) {
            continue;
        }
        observed.entry(base.clone()).or_default().insert(args);
    }
    // Per-variable declaration spellings refine (same single rule).
    for vars in module.template_spellings.values() {
        for spelling in vars.values() {
            let norm: String = spelling.chars().filter(|c| !c.is_whitespace()).collect();
            let Some((base, args)) = split_template_id(&norm) else {
                continue;
            };
            if !stemplates.contains_key(base.as_str()) {
                continue;
            }
            if base.starts_with("sce::") || base.ends_with("::CullData") || base == "CullData" {
                continue;
            }
            observed.entry(base).or_default().insert(args);
        }
    }
    // Single-spelling bases only; also require something to substitute
    // (a clone identical to its template is churn with risk and no gain).
    let mut uses: Vec<(String, Vec<String>)> = Vec::new();
    for (base, argsets) in &observed {
        if argsets.len() != 1 {
            continue;
        }
        let args = argsets.iter().next().unwrap().clone();
        let Some(tpl) = stemplates.get(base.as_str()) else {
            continue;
        };
        if args.len() != tpl.tparams.len() {
            continue;
        }
        // At least one member type must mention a template param.
        let mentions = module
            .structs
            .iter()
            .find(|s| &s.name == base)
            .is_some_and(|decl| {
                decl.fields.iter().any(|f| {
                    tpl.tparams.iter().any(|t| match &f.ty {
                        Type::Named(n) => n == t,
                        Type::Resource { element, .. } => {
                            matches!(element.as_ref(), Type::Named(n) if n == t)
                        }
                        _ => false,
                    })
                })
            });
        if mentions {
            uses.push((base.clone(), args));
        }
    }
    if uses.is_empty() {
        return module;
    }
    // Build clones (arg count must match; all args plain types).
    let mut clones: HashMap<(String, Vec<String>), String> = HashMap::new();
    let mut taken: HashSet<String> = module
        .structs
        .iter()
        .map(|s| s.name.clone())
        .chain(module.functions.iter().map(|f| f.name.clone()))
        .collect();
    let mut counter = 0usize;
    for (base, args) in &uses {
        let Some(tpl) = stemplates.get(base.as_str()) else {
            continue;
        };
        if args.len() != tpl.tparams.len() {
            continue;
        }
        let mut map: HashMap<String, Type> = HashMap::new();
        let mut ok = true;
        for (tname, arg) in tpl.tparams.iter().zip(args.iter()) {
            match parse_arg(arg) {
                Some(ty) => {
                    map.insert(tname.clone(), ty);
                }
                None => {
                    ok = false;
                    break;
                }
            }
        }
        if !ok {
            continue;
        }
        counter += 1;
        let mut cname = format!("{base}__mono{counter}");
        while taken.contains(&cname) {
            counter += 1;
            cname = format!("{base}__mono{counter}");
        }
        taken.insert(cname.clone());
        let Some(decl) = module.structs.iter().find(|s| &s.name == base).cloned() else {
            continue;
        };
        let mut clone = decl;
        clone.name = cname.clone();
        clone.template_params = Vec::new();
        for f in clone.fields.iter_mut() {
            f.ty = subst_member_ty(&f.ty, &map);
        }
        module.structs.push(clone);
        clones.insert((base.clone(), args.clone()), cname);
    }
    if clones.is_empty() {
        return module;
    }
    // Base -> clone (single spelling each by construction above).
    let by_base: HashMap<&String, &String> = clones
        .iter()
        .map(|((base, _), cname)| (base, cname))
        .collect();
    fn retarget_ty(ty: &mut Type, by_base: &HashMap<&String, &String>) {
        if let Type::Named(n) = ty {
            if let Some(cname) = by_base.get(n) {
                *n = (*cname).clone();
            }
        }
    }
    fn retarget_expr(e: &mut Expr, by_base: &HashMap<&String, &String>) {
        match e {
            Expr::Constructor { ty, arguments } => {
                retarget_ty(ty, by_base);
                for a in arguments.iter_mut() {
                    retarget_expr(a, by_base);
                }
            }
            Expr::Call { function, arguments } => {
                retarget_expr(function, by_base);
                for a in arguments.iter_mut() {
                    retarget_expr(a, by_base);
                }
            }
            Expr::Member { object, .. } => retarget_expr(object, by_base),
            Expr::Index { object, index } => {
                retarget_expr(object, by_base);
                retarget_expr(index, by_base);
            }
            Expr::Assign { lhs, rhs } => {
                retarget_expr(lhs, by_base);
                retarget_expr(rhs, by_base);
            }
            Expr::Unary { expr, .. } => retarget_expr(expr, by_base),
            Expr::Binary { left, right, .. } => {
                retarget_expr(left, by_base);
                retarget_expr(right, by_base);
            }
            Expr::Conditional { condition, then_branch, else_branch } => {
                retarget_expr(condition, by_base);
                retarget_expr(then_branch, by_base);
                retarget_expr(else_branch, by_base);
            }
            Expr::InitializerList(elems) => {
                for x in elems.iter_mut() {
                    retarget_expr(x, by_base);
                }
            }
            Expr::Lambda { body, .. } => {
                for s in body.iter_mut() {
                    retarget_stmt(s, by_base);
                }
            }
            Expr::Variable(_) | Expr::Number(_) | Expr::String(_) | Expr::Bool(_) => {}
        }
    }
    fn retarget_stmt(s: &mut Stmt, by_base: &HashMap<&String, &String>) {
        match s {
            Stmt::VarDecl { ty, init, .. } | Stmt::Destructure { ty, init, .. } => {
                retarget_ty(ty, by_base);
                if let Some(e) = init {
                    retarget_expr(e, by_base);
                }
            }
            Stmt::Return(Some(e)) | Stmt::Expr(e) => retarget_expr(e, by_base),
            Stmt::Return(None) => {}
            Stmt::Block(ss) => {
                for x in ss.iter_mut() {
                    retarget_stmt(x, by_base);
                }
            }
            Stmt::If { condition, then_branch, else_branch } => {
                retarget_expr(condition, by_base);
                for x in then_branch.iter_mut() {
                    retarget_stmt(x, by_base);
                }
                if let Some(els) = else_branch {
                    for x in els.iter_mut() {
                        retarget_stmt(x, by_base);
                    }
                }
            }
            Stmt::For { init, condition, update, body } => {
                retarget_stmt(init, by_base);
                if let Some(c) = condition {
                    retarget_expr(c, by_base);
                }
                if let Some(u) = update {
                    retarget_expr(u, by_base);
                }
                for x in body.iter_mut() {
                    retarget_stmt(x, by_base);
                }
            }
            Stmt::While { condition, body } => {
                retarget_expr(condition, by_base);
                for x in body.iter_mut() {
                    retarget_stmt(x, by_base);
                }
            }
            Stmt::Switch { scrutinee, cases } => {
                retarget_expr(scrutinee, by_base);
                for c in cases.iter_mut() {
                    if let Some(l) = &mut c.label {
                        retarget_expr(l, by_base);
                    }
                    for x in c.body.iter_mut() {
                        retarget_stmt(x, by_base);
                    }
                }
            }
        }
    }
    // Params, globals, struct fields, and all bodies.
    for f in module.functions.iter_mut() {
        for p in f.parameters.iter_mut() {
            retarget_ty(&mut p.ty, &by_base);
        }
        if let Type::Named(n) = &f.return_type {
            if by_base.contains_key(n) {
                f.return_type = Type::Named(by_base[n].clone());
            }
        }
        for s in f.body.iter_mut() {
            retarget_stmt(s, &by_base);
        }
    }
    for g in module.globals.iter_mut() {
        retarget_ty(&mut g.ty, &by_base);
    }
    for s in module.structs.iter_mut() {
        for f in s.fields.iter_mut() {
            retarget_ty(&mut f.ty, &by_base);
        }
    }
    // Drop fully-retargeted template defs (a leftover `RW_Texture2D<T>`
    // member would break DXC even when unused). A base still referenced
    // anywhere keeps its def (loud, as today).
    fn ty_mentions(t: &Type, base: &str) -> bool {
        match t {
            Type::Named(n) => n == base,
            Type::Resource { element, .. } => ty_mentions(element, base),
            _ => false,
        }
    }
    fn expr_mentions(e: &Expr, base: &str) -> bool {
        match e {
            Expr::Constructor { ty, arguments } => {
                ty_mentions(ty, base) || arguments.iter().any(|a| expr_mentions(a, base))
            }
            Expr::Call { function, arguments } => {
                expr_mentions(function, base)
                    || arguments.iter().any(|a| expr_mentions(a, base))
            }
            Expr::Member { object, .. } => expr_mentions(object, base),
            Expr::Index { object, index } => {
                expr_mentions(object, base) || expr_mentions(index, base)
            }
            Expr::Assign { lhs, rhs } => expr_mentions(lhs, base) || expr_mentions(rhs, base),
            Expr::Unary { expr, .. } => expr_mentions(expr, base),
            Expr::Binary { left, right, .. } => {
                expr_mentions(left, base) || expr_mentions(right, base)
            }
            Expr::Conditional { condition, then_branch, else_branch } => {
                expr_mentions(condition, base)
                    || expr_mentions(then_branch, base)
                    || expr_mentions(else_branch, base)
            }
            Expr::InitializerList(elems) => elems.iter().any(|a| expr_mentions(a, base)),
            Expr::Lambda { body, .. } => body.iter().any(|s| stmt_mentions(s, base)),
            Expr::Variable(n) => n == base,
            Expr::Number(_) | Expr::String(_) | Expr::Bool(_) => false,
        }
    }
    fn stmt_mentions(s: &Stmt, base: &str) -> bool {
        match s {
            Stmt::Return(Some(e)) | Stmt::Expr(e) => expr_mentions(e, base),
            Stmt::Return(None) => false,
            Stmt::VarDecl { ty, init, .. } | Stmt::Destructure { ty, init, .. } => {
                ty_mentions(ty, base) || init.as_ref().is_some_and(|e| expr_mentions(e, base))
            }
            Stmt::Block(ss) => ss.iter().any(|x| stmt_mentions(x, base)),
            Stmt::If { condition, then_branch, else_branch } => {
                expr_mentions(condition, base)
                    || then_branch.iter().any(|x| stmt_mentions(x, base))
                    || else_branch.iter().flatten().any(|x| stmt_mentions(x, base))
            }
            Stmt::For { init, condition, update, body } => {
                stmt_mentions(init, base)
                    || condition.as_ref().is_some_and(|c| expr_mentions(c, base))
                    || update.as_ref().is_some_and(|u| expr_mentions(u, base))
                    || body.iter().any(|x| stmt_mentions(x, base))
            }
            Stmt::While { condition, body } => {
                expr_mentions(condition, base)
                    || body.iter().any(|x| stmt_mentions(x, base))
            }
            Stmt::Switch { scrutinee, cases } => {
                expr_mentions(scrutinee, base)
                    || cases.iter().any(|c| {
                        c.label.as_ref().is_some_and(|l| expr_mentions(l, base))
                            || c.body.iter().any(|x| stmt_mentions(x, base))
                    })
            }
        }
    }
    let bases: Vec<String> = by_base.keys().map(|s| (*s).clone()).collect();
    for base in bases {
        let mut used = false;
        for f in &module.functions {
            for p in &f.parameters {
                if ty_mentions(&p.ty, &base) {
                    used = true;
                    break;
                }
            }
            if used {
                break;
            }
            if ty_mentions(&f.return_type, &base) {
                used = true;
                break;
            }
            if f.body.iter().any(|s| stmt_mentions(s, &base)) {
                used = true;
                break;
            }
        }
        if !used {
            for g in &module.globals {
                if ty_mentions(&g.ty, &base) {
                    used = true;
                    break;
                }
            }
        }
        if !used {
            for s in &module.structs {
                if s.name == base {
                    continue;
                }
                if s.fields.iter().any(|f| ty_mentions(&f.ty, &base)) {
                    used = true;
                    break;
                }
                if s.bases.iter().any(|b| b == &base) {
                    used = true;
                    break;
                }
            }
        }
        // Methods scoped to the base keep it alive too.
        if !used
            && module.functions.iter().any(|f| f.scope.as_deref() == Some(base.as_str()))
        {
            used = true;
        }
            if !used {
            module.structs.retain(|s| s.name != base);
        }
    }
    module
}

/// Instantiates value-parameter struct templates (`template <uint N>
/// struct Indices {...}`) per concrete use spelling, honoring explicit
/// specializations (`template <> struct Indices<2> {...}`).
///
/// Use sites spell arguments explicitly (`Indices<3> x;`, `out
/// Indices<3> y`, `Indices<2> m;` members); params record in
/// `template_spellings`, locals and fields in
/// `template_site_spellings` (angle lists erase in types). For each
/// (base, args) with all args resolving to integer literals (literals
/// or `module.constants` numbers):
/// * an explicit specialization parsed for those exact args is used
///   (its definition name is already sanitized at parse);
/// * otherwise the primary clones with value params substituted as
///   literals in method bodies (shadowing methods stay loud) and its
///   methods re-scoped to the clone.
/// Use-site types retarget to the resolved struct name. Primary
/// templates with no remaining uses drop with their methods; anything
/// unresolvable (mixed typename/value params, `T::` members,
/// value-dependent array sizes, non-literal args, returns/ctors of
/// template type) keeps today's loud path.
///
/// Only templates with at least one value param and zero typename
/// params qualify (all-typename belongs to `monomorphize_structs`;
/// mixed stays loud).
pub(crate) fn monomorphize_struct_values(module: &Module) -> Module {
    let mut module = module.clone();

    // Primary value-templates: name -> value tparam names, in order.
    let mut primaries: HashMap<String, Vec<String>> = HashMap::new();
    for s in &module.structs {
        if s.template_params.is_empty() || !s.bases.is_empty() {
            continue;
        }
        if s.template_params.iter().any(|(_, is_value)| !is_value) {
            continue;
        }
        primaries.insert(
            s.name.clone(),
            s.template_params.iter().map(|(n, _)| n.clone()).collect(),
        );
    }
    // Explicit specializations: normalized spelling -> definition name,
    // recorded at parse (`@spec` scope).
    let specs: HashMap<String, String> = module
        .template_site_spellings
        .get("@spec")
        .cloned()
        .unwrap_or_default();
    if primaries.is_empty() {
        return module;
    }
    // Resolve an integer value argument to canonical decimal: literals
    // (decimal/hex) or `module.constants` numbers (following chains).
    // Returns (canonical string, literal expr) for substitution.
    fn resolve_arg(arg: &str, constants: &HashMap<String, Expr>) -> Option<(String, Expr)> {
        let mut seen = std::collections::HashSet::new();
        let mut cur = arg.trim().to_string();
        for _ in 0..16 {
            if cur.is_empty() {
                return None;
            }
            // Booleans map to 1/0.
            if cur.eq_ignore_ascii_case("true") {
                return Some(("1".to_string(), Expr::Bool(true)));
            }
            if cur.eq_ignore_ascii_case("false") {
                return Some(("0".to_string(), Expr::Bool(false)));
            }
            // Signed decimal / hex integer literal?
            let t = cur.trim_start_matches(['+', '-']);
            let neg = cur.starts_with('-');
            let digits = t
                .strip_prefix("0x")
                .or_else(|| t.strip_prefix("0X"))
                .map(|h| (16u32, h))
                .unwrap_or((10, t));
            if !digits.1.is_empty()
                && digits
                    .1
                    .chars()
                    .all(|c| c.is_ascii_hexdigit() && (digits.0 == 16 || !c.is_ascii_alphabetic() || ('0'..='9').contains(&c)))
            {
                // Decimal digits only (or hex with prefix): canonicalize.
                let v = if digits.0 == 16 {
                    i64::from_str_radix(digits.1, 16).ok()?
                } else {
                    // Decimal must be all digits.
                    if !digits.1.chars().all(|c| c.is_ascii_digit()) {
                        return None;
                    }
                    digits.1.parse::<i64>().ok()?
                };
                let v = if neg { -v } else { v };
                return Some((v.to_string(), Expr::Number(v.to_string())));
            }
            // Named constant chain.
            if !seen.insert(cur.clone()) {
                return None;
            }
            match constants.get(&cur) {
                Some(Expr::Number(n)) => cur = n.clone(),
                Some(Expr::Bool(b)) => {
                    return Some((
                        if *b { "1".to_string() } else { "0".to_string() },
                        Expr::Bool(*b),
                    ))
                }
                _ => return None,
            }
        }
        None
    }
    // Split a normalized `Base<a,b>` spelling into (base, args).
    // Flat only; nested angles decline.
    fn split_spelling(spelling: &str) -> Option<(String, Vec<String>)> {
        let (base, rest) = spelling.split_once('<')?;
        if base.is_empty() || !rest.ends_with('>') {
            return None;
        }
        let inner = &rest[..rest.len() - 1];
        if inner.contains('<') || inner.contains('>') {
            return None;
        }
        let args: Vec<String> = inner.split(',').map(|a| a.trim().to_string()).collect();
        if args.is_empty() || args.iter().any(|a| a.is_empty()) {
            return None;
        }
        Some((base.to_string(), args))
    }
    // Collect use sites: (scope-kind, scope, var, base, args).
    // Params from `template_spellings`, locals and fields from
    // `template_site_spellings`.
    struct Site {
        scope: String,
        var: String,
        base: String,
        args: Vec<String>,
    }
    let mut sites: Vec<Site> = Vec::new();
    for (scope, vars) in module.template_spellings.iter().chain(module.template_site_spellings.iter()) {
        if scope == "@spec" {
            continue;
        }
        for (var, spelling) in vars {
            let norm: String = spelling.chars().filter(|c| !c.is_whitespace()).collect();
            let Some((base, args)) = split_spelling(&norm) else {
                continue;
            };
            if !primaries.contains_key(base.as_str()) {
                continue;
            }
            sites.push(Site { scope: scope.clone(), var: var.clone(), base, args });
        }
    }
    if sites.is_empty() {
        return module;
    }
    // Resolve each site to a concrete struct name: explicit spec first,
    // else primary instantiation. Groups sites by (base, canonical args).
    let mut taken: std::collections::HashSet<String> = module
        .structs
        .iter()
        .map(|s| s.name.clone())
        .chain(module.functions.iter().map(|f| f.name.clone()))
        .chain(module.globals.iter().map(|g| g.name.clone()))
        .collect();
    // (base, canonical-args-key) -> concrete struct name.
    let mut resolved: HashMap<(String, String), String> = HashMap::new();
    // concrete name -> value bindings (tparam -> literal expr) for clones.
    let mut clone_bindings: HashMap<String, Vec<(String, Expr)>> = HashMap::new();
    for site in &sites {
        let tparams = &primaries[site.base.as_str()];
        if site.args.len() != tparams.len() {
            continue;
        }
        // Canonicalize each arg.
        let mut canon: Vec<String> = Vec::new();
        let mut bindings: Vec<(String, Expr)> = Vec::new();
        let mut ok = true;
        for (tname, arg) in tparams.iter().zip(site.args.iter()) {
            match resolve_arg(arg, &module.constants) {
                Some((s, e)) => {
                    canon.push(s);
                    bindings.push((tname.clone(), e));
                }
                None => {
                    ok = false;
                    break;
                }
            }
        }
        if !ok {
            continue;
        }
        let key = (site.base.clone(), canon.join(","));
        if resolved.contains_key(&key) {
            continue;
        }
        // Raw normalized spelling first (covers `Indices<kTwo>`-style
        // spellings recorded verbatim at parse), then canonical.
        let raw_norm: String = site
            .args
            .iter()
            .map(|a| a.chars().filter(|c| !c.is_whitespace()).collect::<String>())
            .collect::<Vec<_>>()
            .join(",");
        let raw_key = format!("{}<{}>", site.base, raw_norm);
        let canon_key = format!("{}<{}>", site.base, key.1);
        if let Some(spec_name) = specs.get(&raw_key).or_else(|| specs.get(&canon_key)) {
            if module.structs.iter().any(|s| &s.name == spec_name) {
                resolved.insert(key, spec_name.clone());
                continue;
            }
        }
        // Instantiate the primary.
        let mut cname = crate::parser::module::sanitize_template_name(&canon_key);
        if taken.contains(&cname) {
            // Pathological collision with an unrelated declaration:
            // suffix (use sites that would have matched the taken
            // name stay loud).
            let mut n = 0u32;
            loop {
                n += 1;
                let alt = format!("{cname}_{n}");
                if !taken.contains(&alt) {
                    cname = alt;
                    break;
                }
            }
        }
        taken.insert(cname.clone());
        resolved.insert(key, cname.clone());
        clone_bindings.insert(cname, bindings);
    }
    if resolved.is_empty() {
        return module;
    }
    // Build clones: struct + re-scoped methods with values substituted.
    // Method bodies mentioning a shadowed tparam name (param/local) or
    // `T::`-style qualification stay loud (that method is skipped;
    // calls to it fail at DXC as today).
    let mut new_structs: Vec<crate::ast::StructDecl> = Vec::new();
    let mut new_functions: Vec<Function> = Vec::new();
    // (base, canonical-key) list for primary definitions needing clones.
    let mut to_clone: Vec<((String, String), String)> = Vec::new();
    for ((base, key), cname) in &resolved {
        // Skip resolutions pointing at parsed specializations.
        if !clone_bindings.contains_key(cname) {
            continue;
        }
        to_clone.push(((base.clone(), key.clone()), cname.clone()));
    }
    for ((base, _), cname) in to_clone {
        let Some(def) = module.structs.iter().find(|s| s.name == base) else {
            continue;
        };
        let bindings: HashMap<String, Expr> = clone_bindings
            .get(&cname)
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .collect();
        let mut cloned = def.clone();
        cloned.name = cname.clone();
        cloned.template_params = Vec::new();
        new_structs.push(cloned);
        // Clone methods of the primary with the new scope.
        for f in module.functions.iter().filter(|f| f.scope.as_deref() == Some(base.as_str())) {
            if !f.template_params.is_empty() {
                continue;
            }
            // Shadowing veto: a param/local named like a value tparam
            // would capture the substitution.
            if method_shadows(&f.body, &f.parameters, &bindings) {
                continue;
            }
            let mut m = f.clone();
            m.scope = Some(cname.clone());
            subst_values_in_function(&mut m, &bindings);
            new_functions.push(m);
        }
    }
    // Helpers (nested fns): value substitution + shadowing veto.
    fn method_shadows(
        body: &[Stmt],
        params: &[crate::ast::Parameter],
        bindings: &HashMap<String, Expr>,
    ) -> bool {
        for p in params {
            if bindings.contains_key(&p.name) {
                return true;
            }
        }
        fn stmt_declares(s: &Stmt, names: &mut Vec<String>) {
            match s {
                Stmt::VarDecl { name, .. } => names.push(name.clone()),
                Stmt::Destructure { names: ns, .. } => names.extend(ns.clone()),
                Stmt::Block(ss) => {
                    for x in ss {
                        stmt_declares(x, names);
                    }
                }
                Stmt::If { then_branch, else_branch, .. } => {
                    for x in then_branch.iter().chain(else_branch.iter().flatten()) {
                        stmt_declares(x, names);
                    }
                }
                Stmt::For { init, body, .. } => {
                    stmt_declares(init, names);
                    for x in body {
                        stmt_declares(x, names);
                    }
                }
                Stmt::While { body, .. } => {
                    for x in body {
                        stmt_declares(x, names);
                    }
                }
                Stmt::Switch { cases, .. } => {
                    for c in cases {
                        for x in &c.body {
                            stmt_declares(x, names);
                        }
                    }
                }
                _ => {}
            }
        }
        let mut declared = Vec::new();
        for s in body {
            stmt_declares(s, &mut declared);
        }
        // Lambda params also shadow.
        fn lambda_params(e: &Expr, out: &mut Vec<String>) {
            match e {
                Expr::Call { function, arguments } => {
                    lambda_params(function, out);
                    for a in arguments {
                        lambda_params(a, out);
                    }
                }
                Expr::Lambda { params, .. } => out.extend(params.clone()),
                Expr::Member { object, .. } => lambda_params(object, out),
                Expr::Index { object, index } => {
                    lambda_params(object, out);
                    lambda_params(index, out);
                }
                Expr::Constructor { arguments, .. } => {
                    for a in arguments {
                        lambda_params(a, out);
                    }
                }
                Expr::Assign { lhs, rhs } => {
                    lambda_params(lhs, out);
                    lambda_params(rhs, out);
                }
                Expr::Unary { expr, .. } => lambda_params(expr, out),
                Expr::Binary { left, right, .. } => {
                    lambda_params(left, out);
                    lambda_params(right, out);
                }
                Expr::Conditional { condition, then_branch, else_branch } => {
                    lambda_params(condition, out);
                    lambda_params(then_branch, out);
                    lambda_params(else_branch, out);
                }
                Expr::InitializerList(elems) => {
                    for x in elems {
                        lambda_params(x, out);
                    }
                }
                _ => {}
            }
        }
        for s in body {
            let mut tmp = Vec::new();
            match s {
                Stmt::Return(Some(e)) | Stmt::Expr(e) => lambda_params(e, &mut tmp),
                Stmt::VarDecl { init: Some(e), .. }
                | Stmt::Destructure { init: Some(e), .. } => lambda_params(e, &mut tmp),
                _ => {}
            }
            declared.extend(tmp);
        }
        declared.iter().any(|d| bindings.contains_key(d))
    }
    fn subst_values_in_function(f: &mut Function, bindings: &HashMap<String, Expr>) {
        fn expr(e: &mut Expr, bindings: &HashMap<String, Expr>) {
            match e {
                // Bare tparam references substitute; member objects
                // (`x.NumBits`) are different entities and stay.
                Expr::Variable(n) => {
                    if let Some(lit) = bindings.get(n) {
                        *e = lit.clone();
                    }
                }
                Expr::Member { object, .. } => expr(object, bindings),
                Expr::Index { object, index } => {
                    expr(object, bindings);
                    expr(index, bindings);
                }
                Expr::Call { function, arguments } => {
                    expr(function, bindings);
                    for a in arguments.iter_mut() {
                        expr(a, bindings);
                    }
                }
                Expr::Constructor { arguments, .. } => {
                    for a in arguments.iter_mut() {
                        expr(a, bindings);
                    }
                }
                Expr::Assign { lhs, rhs } => {
                    expr(lhs, bindings);
                    expr(rhs, bindings);
                }
                Expr::Unary { expr: inner, .. } => expr(inner, bindings),
                Expr::Binary { left, right, .. } => {
                    expr(left, bindings);
                    expr(right, bindings);
                }
                Expr::Conditional { condition, then_branch, else_branch } => {
                    expr(condition, bindings);
                    expr(then_branch, bindings);
                    expr(else_branch, bindings);
                }
                Expr::InitializerList(elems) => {
                    for x in elems.iter_mut() {
                        expr(x, bindings);
                    }
                }
                Expr::Lambda { body, .. } => {
                    for s in body.iter_mut() {
                        stmt(s, bindings);
                    }
                }
                _ => {}
            }
        }
        fn stmt(s: &mut Stmt, bindings: &HashMap<String, Expr>) {
            match s {
                Stmt::Return(Some(e)) | Stmt::Expr(e) => expr(e, bindings),
                Stmt::Return(None) => {}
                Stmt::VarDecl { init: Some(e), .. }
                | Stmt::Destructure { init: Some(e), .. } => expr(e, bindings),
                Stmt::VarDecl { init: None, .. } | Stmt::Destructure { init: None, .. } => {}
                Stmt::Block(ss) => {
                    for x in ss.iter_mut() {
                        stmt(x, bindings);
                    }
                }
                Stmt::If { condition, then_branch, else_branch } => {
                    expr(condition, bindings);
                    for x in then_branch.iter_mut().chain(else_branch.iter_mut().flatten()) {
                        stmt(x, bindings);
                    }
                }
                Stmt::For { init, condition, update, body } => {
                    stmt(init.as_mut(), bindings);
                    if let Some(c) = condition {
                        expr(c, bindings);
                    }
                    if let Some(u) = update {
                        expr(u, bindings);
                    }
                    for x in body.iter_mut() {
                        stmt(x, bindings);
                    }
                }
                Stmt::While { condition, body } => {
                    expr(condition, bindings);
                    for x in body.iter_mut() {
                        stmt(x, bindings);
                    }
                }
                Stmt::Switch { scrutinee, cases } => {
                    expr(scrutinee, bindings);
                    for c in cases.iter_mut() {
                        if let Some(l) = &mut c.label {
                            expr(l, bindings);
                        }
                        for x in c.body.iter_mut() {
                            stmt(x, bindings);
                        }
                    }
                }
            }
        }
        for p in &mut f.parameters {
            if let Some(d) = &mut p.default {
                expr(d, bindings);
            }
        }
        for s in &mut f.body {
            stmt(s, bindings);
        }
    }
    // Retarget use-site types to resolved concrete names. Only sites
    // whose spelling resolved (same (base, args)) retarget; bare uses
    // keep today's shape.
    // (scope, var) -> concrete name, for sites in `resolved`.
    let mut site_targets: HashMap<(String, String), String> = HashMap::new();
    // Recompute per-site resolution (same rules as above).
    // Params:
    for (scope, vars) in module.template_spellings.iter() {
        for (var, spelling) in vars {
            let norm: String = spelling.chars().filter(|c| !c.is_whitespace()).collect();
            let Some((base, args)) = split_spelling(&norm) else {
                continue;
            };
            if !primaries.contains_key(base.as_str()) {
                continue;
            }
            // Canonicalize args as above.
            let tparams = &primaries[base.as_str()];
            if args.len() != tparams.len() {
                continue;
            }
            let mut canon = Vec::new();
            let mut ok = true;
            for a in &args {
                match resolve_arg(a, &module.constants) {
                    Some((s, _)) => canon.push(s),
                    None => {
                        ok = false;
                        break;
                    }
                }
            }
            if !ok {
                continue;
            }
            let key = (base.clone(), canon.join(","));
            if let Some(concrete) = resolved.get(&key) {
                site_targets.insert((scope.clone(), var.clone()), concrete.clone());
            } else {
                // Raw-key spec match (non-foldable spellings recorded
                // verbatim at parse).
                let raw_key = format!("{base}<{}>", args.join(","));
                if let Some(spec_name) = specs.get(&raw_key) {
                    if module.structs.iter().any(|s| &s.name == spec_name) {
                        site_targets.insert((scope.clone(), var.clone()), spec_name.clone());
                    }
                }
            }
        }
    }
    // Locals + fields:
    for (scope, vars) in module.template_site_spellings.iter() {
        if scope == "@spec" {
            continue;
        }
        for (var, spelling) in vars {
            let norm: String = spelling.chars().filter(|c| !c.is_whitespace()).collect();
            let Some((base, args)) = split_spelling(&norm) else {
                continue;
            };
            if !primaries.contains_key(base.as_str()) {
                continue;
            }
            let tparams = &primaries[base.as_str()];
            if args.len() != tparams.len() {
                continue;
            }
            let mut canon = Vec::new();
            let mut ok = true;
            for a in &args {
                match resolve_arg(a, &module.constants) {
                    Some((s, _)) => canon.push(s),
                    None => {
                        ok = false;
                        break;
                    }
                }
            }
            if !ok {
                continue;
            }
            let key = (base.clone(), canon.join(","));
            if let Some(concrete) = resolved.get(&key) {
                site_targets.insert((scope.clone(), var.clone()), concrete.clone());
            } else {
                let raw_key = format!("{base}<{}>", args.join(","));
                if let Some(spec_name) = specs.get(&raw_key) {
                    if module.structs.iter().any(|s| &s.name == spec_name) {
                        site_targets.insert((scope.clone(), var.clone()), spec_name.clone());
                    }
                }
            }
        }
    }
    if site_targets.is_empty() && new_structs.is_empty() && new_functions.is_empty() {
        return module;
    }
    // Apply retargeting: struct fields, function params, VarDecls.
    for s in &mut module.structs {
        // NOTE: cannot borrow site_targets per struct while iterating
        // structs mutably; collect matches below instead.
        let mut renames: Vec<(String, String)> = Vec::new();
        if let Some(vars) = site_targets_by_scope(&site_targets, &s.name) {
            for f in &s.fields {
                if let crate::ast::Type::Named(n) = &f.ty {
                    if primaries.contains_key(n.as_str()) {
                        if let Some(t) = vars.get(&f.name) {
                            renames.push((f.name.clone(), t.clone()));
                        }
                    }
                }
            }
        }
        for (fname, tname) in renames {
            if let Some(f) = s.fields.iter_mut().find(|f| f.name == fname) {
                f.ty = crate::ast::Type::Named(tname);
            }
        }
    }
    fn site_targets_by_scope(
        site_targets: &HashMap<(String, String), String>,
        scope: &str,
    ) -> Option<HashMap<String, String>> {
        let mut out = HashMap::new();
        for ((s, v), t) in site_targets {
            if s == scope {
                out.insert(v.clone(), t.clone());
            }
        }
        if out.is_empty() {
            None
        } else {
            Some(out)
        }
    }
    for f in &mut module.functions {
        // Params: match by function name; methods (scoped) match by
        // scope+name? Params belong to the function entity: key scope
        // is the function NAME for free functions; for methods the
        // spelling was recorded under... (params drain per function
        // parse with the method's name). Match free by name, methods
        // by name as well (method names are unique per struct here;
        // collisions across structs with same method name keep loud
        // via the scope check below).
        let key_scope = f.name.clone();
        let vars = site_targets_by_scope(&site_targets, &key_scope);
        // Determine the owning scope for locals: free functions use
        // their name; methods use their scope (spellings recorded per
        // parsed function entity — identical string either way here).
        for p in &mut f.parameters {
            if let crate::ast::Type::Named(n) = &p.ty {
                if primaries.contains_key(n.as_str()) {
                    if let Some(t) = vars.as_ref().and_then(|v| v.get(&p.name)) {
                        // Methods: only retarget when the method's scope
                        // matches the spelling scope (avoids cross-struct
                        // same-name collisions).
                        p.ty = crate::ast::Type::Named(t.clone());
                    }
                }
            }
        }
        // Locals: same map (locals recorded per function entity).
        for s in f.body.iter_mut() {
            retarget_vardecls(s, &vars, &primaries);
        }
    }
    fn retarget_vardecls(
        s: &mut Stmt,
        vars: &Option<HashMap<String, String>>,
        primaries: &HashMap<String, Vec<String>>,
    ) {
        // Lambdas bodies skipped (their temps cannot have rewritable
        // uses, mirroring beta reduction).
        match s {
            Stmt::VarDecl { ty, name, .. } => {
                if let crate::ast::Type::Named(n) = ty {
                    if primaries.contains_key(n.as_str()) {
                        if let Some(t) = vars.as_ref().and_then(|v| v.get(name)) {
                            *ty = crate::ast::Type::Named(t.clone());
                        }
                    }
                }
            }
            Stmt::Destructure { .. } => {}
            Stmt::Block(ss) => {
                for x in ss.iter_mut() {
                    retarget_vardecls(x, vars, primaries);
                }
            }
            Stmt::If { then_branch, else_branch, .. } => {
                for x in then_branch.iter_mut().chain(else_branch.iter_mut().flatten()) {
                    retarget_vardecls(x, vars, primaries);
                }
            }
            Stmt::For { init, body, .. } => {
                retarget_vardecls(init.as_mut(), vars, primaries);
                for x in body.iter_mut() {
                    retarget_vardecls(x, vars, primaries);
                }
            }
            Stmt::While { body, .. } => {
                for x in body.iter_mut() {
                    retarget_vardecls(x, vars, primaries);
                }
            }
            Stmt::Switch { cases, .. } => {
                for c in cases.iter_mut() {
                    for x in c.body.iter_mut() {
                        retarget_vardecls(x, vars, primaries);
                    }
                }
            }
            _ => {}
        }
    }
    module.structs.extend(new_structs);
    module.functions.extend(new_functions);
    // Drop primary templates with no remaining uses (bare or spelled):
    // their methods reference unbound value params and would fail.
    // Anything still referenced keeps today's loud shape.
    {
        use crate::ast::Stmt as S_;
        fn mentions_named_ty(
            ty: &crate::ast::Type,
            base: &str,
            out: &mut bool,
        ) {
            if *out {
                return;
            }
            match ty {
                crate::ast::Type::Named(n) if n == base => *out = true,
                crate::ast::Type::Resource { element, .. } => {
                    mentions_named_ty(element, base, out)
                }
                _ => {}
            }
        }
        fn mentions_in_expr(e: &Expr, base: &str, out: &mut bool) {
            if *out {
                return;
            }
            match e {
                Expr::Constructor { ty, .. } => mentions_named_ty(ty, base, out),
                Expr::Call { function, arguments } => {
                    mentions_in_expr(function, base, out);
                    for a in arguments {
                        mentions_in_expr(a, base, out);
                    }
                }
                Expr::Member { object, .. } => mentions_in_expr(object, base, out),
                Expr::Index { object, index } => {
                    mentions_in_expr(object, base, out);
                    mentions_in_expr(index, base, out);
                }
                Expr::Assign { lhs, rhs } => {
                    mentions_in_expr(lhs, base, out);
                    mentions_in_expr(rhs, base, out);
                }
                Expr::Unary { expr, .. } => mentions_in_expr(expr, base, out),
                Expr::Binary { left, right, .. } => {
                    mentions_in_expr(left, base, out);
                    mentions_in_expr(right, base, out);
                }
                Expr::Conditional { condition, then_branch, else_branch } => {
                    mentions_in_expr(condition, base, out);
                    mentions_in_expr(then_branch, base, out);
                    mentions_in_expr(else_branch, base, out);
                }
                Expr::InitializerList(elems) => {
                    for x in elems {
                        mentions_in_expr(x, base, out);
                    }
                }
                Expr::Variable(_) | Expr::Number(_) | Expr::String(_) | Expr::Bool(_) => {}
                Expr::Lambda { .. } => {}
            }
        }
        fn mentions_in_stmt(s: &S_, base: &str, out: &mut bool) {
            if *out {
                return;
            }
            match s {
                S_::Return(Some(e)) | S_::Expr(e) => mentions_in_expr(e, base, out),
                S_::Return(None) => {}
                S_::VarDecl { ty, init, .. } | S_::Destructure { ty, init, .. } => {
                    mentions_named_ty(ty, base, out);
                    if let Some(e) = init {
                        mentions_in_expr(e, base, out);
                    }
                }
                S_::Block(ss) => {
                    for x in ss {
                        mentions_in_stmt(x, base, out);
                    }
                }
                S_::If { condition, then_branch, else_branch } => {
                    mentions_in_expr(condition, base, out);
                    for x in then_branch.iter().chain(else_branch.iter().flatten()) {
                        mentions_in_stmt(x, base, out);
                    }
                }
                S_::For { init, condition, update, body } => {
                    mentions_in_stmt(init, base, out);
                    if let Some(c) = condition {
                        mentions_in_expr(c, base, out);
                    }
                    if let Some(u) = update {
                        mentions_in_expr(u, base, out);
                    }
                    for x in body {
                        mentions_in_stmt(x, base, out);
                    }
                }
                S_::While { condition, body } => {
                    mentions_in_expr(condition, base, out);
                    for x in body {
                        mentions_in_stmt(x, base, out);
                    }
                }
                S_::Switch { scrutinee, cases } => {
                    mentions_in_expr(scrutinee, base, out);
                    for c in cases {
                        for x in &c.body {
                            mentions_in_stmt(x, base, out);
                        }
                    }
                }
            }
        }
        for base in primaries.keys() {
            // Only drop primaries this pass instantiated.
            if !resolved.keys().any(|(b, _)| b == base) {
                continue;
            }
            let mut used = false;
            for s in &module.structs {
                if s.name == *base {
                    continue;
                }
                for f in &s.fields {
                    mentions_named_ty(&f.ty, base, &mut used);
                }
                if used {
                    break;
                }
            }
            if !used {
                for f in &module.functions {
                    for p in &f.parameters {
                        mentions_named_ty(&p.ty, base, &mut used);
                    }
                    mentions_named_ty(&f.return_type, base, &mut used);
                    if used {
                        break;
                    }
                    for s in &f.body {
                        mentions_in_stmt(s, base, &mut used);
                    }
                    if used {
                        break;
                    }
                }
            }
            if !used {
                module.structs.retain(|s| s.name != *base);
                module.functions.retain(|f| f.scope.as_deref() != Some(base.as_str()));
            }
        }
    }
    module
}
/// Inlines functor calls (`isValid(i)` where `isValid` has a struct type
/// with a single-return `operator()`): the operator body substitutes for
/// the call (beta-style, pure-argument gate). Stateless predicate tags
/// like `AllValid` (whose `()` construction HLSL rejects) vanish into
/// constants; stateful functors like `ValidBits` / `InSubset` (whose
/// `operator()` reads struct fields) inline with bare field references
/// rebound to the callee (`m_bits` -> `isValid.m_bits`), matching C++
/// member lookup. Operators calling sibling methods or with non-return
/// bodies stay loud. Bounded fixpoint; runs before tag-param dropping
/// (which removes the then-unused predicate parameters).
pub(crate) fn inline_stateless_functors(module: &Module) -> Module {
    use crate::ast::Stmt;
    // (struct, params, fields, return-expr) operators eligible for inlining.
    struct Op {
        params: Vec<String>,
        fields: HashSet<String>,
        ret: Expr,
    }
    // Purity gate shared with beta-reduction (no duplication or loss of
    // side effects through substitution).
    fn pure(e: &Expr) -> bool {
        match e {
            Expr::Call { .. } | Expr::Assign { .. } | Expr::Lambda { .. } => false,
            Expr::Member { object, .. } => pure(object),
            Expr::Index { object, index } => pure(object) && pure(index),
            Expr::Constructor { arguments, .. } => arguments.iter().all(pure),
            Expr::Unary { expr, .. } => pure(expr),
            Expr::Binary { left, right, .. } => pure(left) && pure(right),
            Expr::Conditional { condition, then_branch, else_branch } => {
                pure(condition) && pure(then_branch) && pure(else_branch)
            }
            Expr::InitializerList(elems) => elems.iter().all(pure),
            Expr::Variable(_) | Expr::Number(_) | Expr::String(_) | Expr::Bool(_) => true,
        }
    }
    // Functor substitution: operator params bind to call args (beta-style)
    // AND bare member references rebind to the callee (`m_bits` becomes
    // `isValid.m_bits`), matching C++ member lookup inside `operator()`.
    // Single pass so arg-introduced names never re-redirect. Bare
    // variables in callee-function position never redirect (a struct
    // field is not callable; nested functors stay loud). Nested lambdas
    // binding a param or field name keep theirs.
    fn subst_functor(
        e: &Expr,
        params: &[String],
        args: &[Expr],
        fields: &HashSet<String>,
        callee: &Expr,
    ) -> Expr {
        fn go(
            e: &Expr,
            params: &[String],
            args: &[Expr],
            fields: &HashSet<String>,
            callee: &Expr,
            in_callee_fn: bool,
        ) -> Expr {
            let lookup = |n: &str| {
                params.iter().position(|p| p == n).and_then(|i| args.get(i)).cloned()
            };
            match e {
                Expr::Variable(n) => {
                    if let Some(a) = lookup(n) {
                        return a;
                    }
                    if !in_callee_fn && fields.contains(n) {
                        return Expr::Member {
                            object: Box::new(callee.clone()),
                            member: n.clone(),
                        };
                    }
                    e.clone()
                }
                Expr::Member { object, member } => Expr::Member {
                    object: Box::new(go(object, params, args, fields, callee, false)),
                    member: member.clone(),
                },
                Expr::Index { object, index } => Expr::Index {
                    object: Box::new(go(object, params, args, fields, callee, false)),
                    index: Box::new(go(index, params, args, fields, callee, false)),
                },
                Expr::Call { function, arguments } => Expr::Call {
                    function: Box::new(go(function, params, args, fields, callee, true)),
                    arguments: arguments
                        .iter()
                        .map(|a| go(a, params, args, fields, callee, false))
                        .collect(),
                },
                Expr::Constructor { ty, arguments } => Expr::Constructor {
                    ty: ty.clone(),
                    arguments: arguments
                        .iter()
                        .map(|a| go(a, params, args, fields, callee, false))
                        .collect(),
                },
                Expr::Assign { lhs, rhs } => Expr::Assign {
                    lhs: Box::new(go(lhs, params, args, fields, callee, false)),
                    rhs: Box::new(go(rhs, params, args, fields, callee, false)),
                },
                Expr::Unary { op, expr } => Expr::Unary {
                    op: *op,
                    expr: Box::new(go(expr, params, args, fields, callee, false)),
                },
                Expr::Binary { left, op, right } => Expr::Binary {
                    left: Box::new(go(left, params, args, fields, callee, false)),
                    right: Box::new(go(right, params, args, fields, callee, false)),
                    op: *op,
                },
                Expr::Conditional { condition, then_branch, else_branch } => Expr::Conditional {
                    condition: Box::new(go(condition, params, args, fields, callee, false)),
                    then_branch: Box::new(go(then_branch, params, args, fields, callee, false)),
                    else_branch: Box::new(go(else_branch, params, args, fields, callee, false)),
                },
                Expr::InitializerList(elems) => Expr::InitializerList(
                    elems
                        .iter()
                        .map(|a| go(a, params, args, fields, callee, false))
                        .collect(),
                ),
                // Nested lambdas binding the same names keep theirs.
                Expr::Lambda { captures, params: inner, body } => {
                    if inner.iter().any(|p| params.contains(p) || fields.contains(p)) {
                        e.clone()
                    } else {
                        Expr::Lambda {
                            captures: captures.clone(),
                            params: inner.clone(),
                            body: body.clone(),
                        }
                    }
                }
                Expr::Number(_) | Expr::String(_) | Expr::Bool(_) => e.clone(),
            }
        }
        go(e, params, args, fields, callee, false)
    }
    let mut module = module.clone();
    for _round in 0..8 {
        // Empty structs (all declarations fieldless).
        // Struct fields (unioned over re-declarations) for member rebinding.
        let mut fields_of: HashMap<String, HashSet<String>> = HashMap::new();
        for s in &module.structs {
            fields_of
                .entry(s.name.clone())
                .or_default()
                .extend(s.fields.iter().map(|fd| fd.name.clone()));
        }
        // Eligible operators: `operator()` with single `return E;`, on any
        // declared struct (stateless tags and stateful functors alike).
        let mut ops: HashMap<(String, usize), Op> = HashMap::new();
        for f in &module.functions {
            let Some(scope) = f.scope.as_deref() else {
                continue;
            };
            if f.name != "operator" {
                continue;
            }
            let Some(fields) = fields_of.get(scope).cloned() else {
                continue;
            };
            if let [Stmt::Return(Some(ret))] = f.body.as_slice() {
                ops.insert(
                    (scope.to_string(), f.parameters.len()),
                    Op {
                        params: f.parameters.iter().map(|p| p.name.clone()).collect(),
                        fields,
                        ret: ret.clone(),
                    },
                );
            }
        }
        if ops.is_empty() {
            break;
        }
        // Sibling methods referenced from operator bodies veto
        // (inlining would unbind them).
        let mut methods_of: HashMap<String, HashSet<String>> = HashMap::new();
        for f in &module.functions {
            if let Some(scope) = f.scope.as_deref() {
                methods_of.entry(scope.to_string()).or_default().insert(f.name.clone());
            }
        }
        // Caller envs need inference tables (minimal: params + roots +
        // top-level locals; nested-block locals missed conservatively).
        let mut roots: HashMap<String, (Type, Option<u32>)> = HashMap::new();
        for g in &module.globals {
            roots.insert(g.name.clone(), (g.ty.clone(), g.array_size));
        }
        for r in &module.resources {
            roots.insert(
                r.name.clone(),
                (
                    Type::Resource {
                        kind: r.kind.clone(),
                        access: crate::ast::ResourceAccess::ReadOnly,
                        element: Box::new(r.element_ty.clone().unwrap_or(Type::UInt)),
                    },
                    None,
                ),
            );
        }
        fn bare_names(e: &Expr, out: &mut HashSet<String>) {
            match e {
                Expr::Variable(n) => {
                    out.insert(n.clone());
                }
                Expr::Call { function, arguments } => {
                    bare_names(function, out);
                    for a in arguments {
                        bare_names(a, out);
                    }
                }
                Expr::Constructor { arguments, .. } => {
                    for a in arguments {
                        bare_names(a, out);
                    }
                }
                Expr::Member { object, .. } => bare_names(object, out),
                Expr::Index { object, index } => {
                    bare_names(object, out);
                    bare_names(index, out);
                }
                Expr::Assign { lhs, rhs } => {
                    bare_names(lhs, out);
                    bare_names(rhs, out);
                }
                Expr::Unary { expr, .. } => bare_names(expr, out),
                Expr::Binary { left, right, .. } => {
                    bare_names(left, out);
                    bare_names(right, out);
                }
                Expr::Conditional { condition, then_branch, else_branch } => {
                    bare_names(condition, out);
                    bare_names(then_branch, out);
                    bare_names(else_branch, out);
                }
                Expr::InitializerList(elems) => {
                    for x in elems {
                        bare_names(x, out);
                    }
                }
                Expr::Lambda { .. } => {}
                Expr::Number(_) | Expr::String(_) | Expr::Bool(_) => {}
            }
        }
        let mut progress = false;
        // Rewrite one expression; true when changed.
        fn rewrite(
            e: &mut Expr,
            env: &HashMap<String, (Type, Option<u32>)>,
            ops: &HashMap<(String, usize), Op>,
            methods_of: &HashMap<String, HashSet<String>>,
            progress: &mut bool,
        ) {
            // Post-order first.
            match e {
                Expr::Call { function, arguments } => {
                    rewrite(function, env, ops, methods_of, progress);
                    for a in arguments.iter_mut() {
                        rewrite(a, env, ops, methods_of, progress);
                    }
                }
                Expr::Member { object, .. } => {
                    rewrite(object, env, ops, methods_of, progress)
                }
                Expr::Index { object, index } => {
                    rewrite(object, env, ops, methods_of, progress);
                    rewrite(index, env, ops, methods_of, progress);
                }
                Expr::Constructor { arguments, .. } => {
                    for a in arguments.iter_mut() {
                        rewrite(a, env, ops, methods_of, progress);
                    }
                }
                Expr::Assign { lhs, rhs } => {
                    rewrite(lhs, env, ops, methods_of, progress);
                    rewrite(rhs, env, ops, methods_of, progress);
                }
                Expr::Unary { expr, .. } => {
                    rewrite(expr, env, ops, methods_of, progress)
                }
                Expr::Binary { left, right, .. } => {
                    rewrite(left, env, ops, methods_of, progress);
                    rewrite(right, env, ops, methods_of, progress);
                }
                Expr::Conditional { condition, then_branch, else_branch } => {
                    rewrite(condition, env, ops, methods_of, progress);
                    rewrite(then_branch, env, ops, methods_of, progress);
                    rewrite(else_branch, env, ops, methods_of, progress);
                }
                Expr::InitializerList(elems) => {
                    for x in elems.iter_mut() {
                        rewrite(x, env, ops, methods_of, progress);
                    }
                }
                _ => {}
            }
            // Then this call: `obj(args)` with obj of functor-struct type.
            // NOTE: free_rets/method_rets are empty here on purpose —
            // only variables (and empty constructors) resolve; calls do
            // not (avoid ordering puzzles with not-yet-deduced autos).
            if let Expr::Call { function, arguments } = e {
                let obj = match function.as_ref() {
                    Expr::Variable(_) => function.as_ref().clone(),
                    _ => return,
                };
                let obj_ty = match &obj {
                    Expr::Variable(n) => env.get(n).map(|(t, _)| t.clone()),
                    Expr::Constructor { ty, arguments } if arguments.is_empty() => Some(ty.clone()),
                    _ => None,
                };
                let Some(Type::Named(s)) = obj_ty else {
                    return;
                };
                let Some(op) = ops.get(&(s.clone(), arguments.len())) else {
                    return;
                };
                if !arguments.iter().all(pure) {
                    return;
                }
                // Sibling-method references would unbind: veto. Field
                // names are NOT methods (they rebind to the callee), so
                // only non-field, non-param names veto.
                let mut names = HashSet::new();
                bare_names(&op.ret, &mut names);
                if let Some(sibs) = methods_of.get(&s) {
                    if names.iter().any(|n| {
                        sibs.contains(n)
                            && !op.params.contains(n)
                            && !op.fields.contains(n)
                    }) {
                        return;
                    }
                }
                *e = subst_functor(&op.ret, &op.params, arguments, &op.fields, &obj);
                *progress = true;
            }
        }
        fn rewrite_stmt(
            s: &mut Stmt,
            env: &HashMap<String, (Type, Option<u32>)>,
            ops: &HashMap<(String, usize), Op>,
            methods_of: &HashMap<String, HashSet<String>>,
            progress: &mut bool,
        ) {
            match s {
                Stmt::Return(Some(e)) | Stmt::Expr(e) => {
                    rewrite(e, env, ops, methods_of, progress)
                }
                Stmt::Return(None) => {}
                Stmt::VarDecl { init: Some(e), .. } | Stmt::Destructure { init: Some(e), .. } => {
                    rewrite(e, env, ops, methods_of, progress)
                }
                Stmt::VarDecl { init: None, .. } | Stmt::Destructure { init: None, .. } => {}
                Stmt::Block(ss) => {
                    for x in ss.iter_mut() {
                        rewrite_stmt(x, env, ops, methods_of, progress);
                    }
                }
                Stmt::If { condition, then_branch, else_branch } => {
                    rewrite(condition, env, ops, methods_of, progress);
                    for x in then_branch.iter_mut() {
                        rewrite_stmt(x, env, ops, methods_of, progress);
                    }
                    if let Some(els) = else_branch {
                        for x in els.iter_mut() {
                            rewrite_stmt(x, env, ops, methods_of, progress);
                        }
                    }
                }
                Stmt::For { init, condition, update, body } => {
                    rewrite_stmt(init, env, ops, methods_of, progress);
                    if let Some(c) = condition {
                        rewrite(c, env, ops, methods_of, progress);
                    }
                    if let Some(u) = update {
                        rewrite(u, env, ops, methods_of, progress);
                    }
                    for x in body.iter_mut() {
                        rewrite_stmt(x, env, ops, methods_of, progress);
                    }
                }
                Stmt::While { condition, body } => {
                    rewrite(condition, env, ops, methods_of, progress);
                    for x in body.iter_mut() {
                        rewrite_stmt(x, env, ops, methods_of, progress);
                    }
                }
                Stmt::Switch { scrutinee, cases } => {
                    rewrite(scrutinee, env, ops, methods_of, progress);
                    for c in cases.iter_mut() {
                        if let Some(l) = &mut c.label {
                            rewrite(l, env, ops, methods_of, progress);
                        }
                        for x in c.body.iter_mut() {
                            rewrite_stmt(x, env, ops, methods_of, progress);
                        }
                    }
                }
            }
        }
        for f in module.functions.iter_mut() {
            let mut env: HashMap<String, (Type, Option<u32>)> = HashMap::new();
            for p in &f.parameters {
                env.insert(p.name.clone(), (p.ty.clone(), p.array_size));
            }
            for (name, (ty, arr)) in &roots {
                env.entry(name.clone()).or_insert_with(|| (ty.clone(), *arr));
            }
            for s in &f.body {
                if let Stmt::VarDecl { ty, name, array_size, .. } = s {
                    env.insert(name.clone(), (ty.clone(), *array_size));
                }
            }
            for s in f.body.iter_mut() {
                rewrite_stmt(s, &env, &ops, &methods_of, &mut progress);
            }
        }
        if !progress {
            break;
        }
    }
    module
}

pub(crate) fn drop_empty_tag_params(module: &Module) -> Module {
    use crate::ast::Stmt;
    // Structs empty in all declarations.
    let mut counts: HashMap<String, (usize, usize)> = HashMap::new();
    for s in &module.structs {
        let e = counts.entry(s.name.clone()).or_insert((0, 0));
        e.0 += 1;
        if s.fields.is_empty() {
            e.1 += 1;
        }
    }
    let empty: HashSet<String> =
        counts.into_iter().filter(|(_, (n, e))| n == e).map(|(s, _)| s).collect();
    if empty.is_empty() {
        return module.clone();
    }
    // Side-effect-free expressions (safe to drop as arguments).
    fn pure(e: &Expr) -> bool {
        match e {
            Expr::Call { .. } | Expr::Assign { .. } | Expr::Lambda { .. } => false,
            Expr::Member { object, .. } => pure(object),
            Expr::Index { object, index } => pure(object) && pure(index),
            Expr::Constructor { arguments, .. } => arguments.iter().all(pure),
            Expr::Unary { expr, .. } => pure(expr),
            Expr::Binary { left, right, .. } => pure(left) && pure(right),
            Expr::Conditional { condition, then_branch, else_branch } => {
                pure(condition) && pure(then_branch) && pure(else_branch)
            }
            Expr::InitializerList(elems) => elems.iter().all(pure),
            Expr::Variable(_) | Expr::Number(_) | Expr::String(_) | Expr::Bool(_) => true,
        }
    }
    fn mentions_var(e: &Expr, name: &str) -> bool {
        match e {
            Expr::Variable(n) => n == name,
            Expr::Call { function, arguments } => {
                mentions_var(function, name) || arguments.iter().any(|a| mentions_var(a, name))
            }
            Expr::Constructor { arguments, .. } => {
                arguments.iter().any(|a| mentions_var(a, name))
            }
            Expr::Member { object, .. } => mentions_var(object, name),
            Expr::Index { object, index } => {
                mentions_var(object, name) || mentions_var(index, name)
            }
            Expr::Assign { lhs, rhs } => mentions_var(lhs, name) || mentions_var(rhs, name),
            Expr::Unary { expr, .. } => mentions_var(expr, name),
            Expr::Binary { left, right, .. } => {
                mentions_var(left, name) || mentions_var(right, name)
            }
            Expr::Conditional { condition, then_branch, else_branch } => {
                mentions_var(condition, name)
                    || mentions_var(then_branch, name)
                    || mentions_var(else_branch, name)
            }
            Expr::InitializerList(elems) => elems.iter().any(|a| mentions_var(a, name)),
            Expr::Lambda { body, .. } => body.iter().any(|s| mentions_var_stmt(s, name)),
            Expr::Number(_) | Expr::String(_) | Expr::Bool(_) => false,
        }
    }
    fn mentions_var_stmt(s: &Stmt, name: &str) -> bool {
        match s {
            Stmt::Return(Some(e)) | Stmt::Expr(e) => mentions_var(e, name),
            Stmt::Return(None) => false,
            Stmt::VarDecl { init: Some(e), .. } | Stmt::Destructure { init: Some(e), .. } => {
                mentions_var(e, name)
            }
            Stmt::VarDecl { init: None, .. } | Stmt::Destructure { init: None, .. } => false,
            Stmt::Block(ss) => ss.iter().any(|x| mentions_var_stmt(x, name)),
            Stmt::If { condition, then_branch, else_branch } => {
                mentions_var(condition, name)
                    || then_branch.iter().any(|x| mentions_var_stmt(x, name))
                    || else_branch.iter().flatten().any(|x| mentions_var_stmt(x, name))
            }
            Stmt::For { init, condition, update, body } => {
                mentions_var_stmt(init, name)
                    || condition.as_ref().is_some_and(|c| mentions_var(c, name))
                    || update.as_ref().is_some_and(|u| mentions_var(u, name))
                    || body.iter().any(|x| mentions_var_stmt(x, name))
            }
            Stmt::While { condition, body } => {
                mentions_var(condition, name)
                    || body.iter().any(|x| mentions_var_stmt(x, name))
            }
            Stmt::Switch { scrutinee, cases } => {
                mentions_var(scrutinee, name)
                    || cases.iter().any(|c| {
                        c.label.as_ref().is_some_and(|l| mentions_var(l, name))
                            || c.body.iter().any(|x| mentions_var_stmt(x, name))
                    })
            }
        }
    }
    // (function) -> (original arity, param index drops).
    let mut drops: HashMap<String, (usize, Vec<usize>)> = HashMap::new();
    for f in &module.functions {
        if f.scope.is_some() {
            continue;
        }
        let mut idxs = Vec::new();
        for (i, p) in f.parameters.iter().enumerate() {
            if let Type::Named(n) = &p.ty {
                if empty.contains(n) && !f.body.iter().any(|s| mentions_var_stmt(s, &p.name)) {
                    idxs.push(i);
                }
            }
        }
        if !idxs.is_empty() {
            drops.insert(f.name.clone(), (f.parameters.len(), idxs));
        }
    }
    if drops.is_empty() {
        return module.clone();
    }
    fn drop_call_args(e: &mut Expr, drops: &HashMap<String, (usize, Vec<usize>)>) {
        match e {
            Expr::Call { function, arguments } => {
                if let Expr::Variable(n) = function.as_ref() {
                    // Calls here are plain (mono already retargeted);
                    // match bare names with the recorded arity so
                    // overloads keep today's path.
                    if let Some((arity, idxs)) = drops.get(n) {
                        // All dropped args must be pure (else keep loud by
                        // leaving this call untouched).
                        if arguments.len() == *arity {
                            let mut ok = true;
                            for &i in idxs {
                                if arguments.get(i).is_none_or(|a| !pure(a)) {
                                    ok = false;
                                    break;
                                }
                            }
                            if ok {
                                let mut keep = Vec::with_capacity(arguments.len());
                                for (i, a) in arguments.drain(..).enumerate() {
                                    if !idxs.contains(&i) {
                                        keep.push(a);
                                    }
                                }
                                *arguments = keep;
                            }
                        }
                    }
                }
                drop_call_args(function, drops);
                for a in arguments.iter_mut() {
                    drop_call_args(a, drops);
                }
            }
            Expr::Member { object, .. } => drop_call_args(object, drops),
            Expr::Index { object, index } => {
                drop_call_args(object, drops);
                drop_call_args(index, drops);
            }
            Expr::Constructor { arguments, .. } => {
                for a in arguments.iter_mut() {
                    drop_call_args(a, drops);
                }
            }
            Expr::Assign { lhs, rhs } => {
                drop_call_args(lhs, drops);
                drop_call_args(rhs, drops);
            }
            Expr::Unary { expr, .. } => drop_call_args(expr, drops),
            Expr::Binary { left, right, .. } => {
                drop_call_args(left, drops);
                drop_call_args(right, drops);
            }
            Expr::Conditional { condition, then_branch, else_branch } => {
                drop_call_args(condition, drops);
                drop_call_args(then_branch, drops);
                drop_call_args(else_branch, drops);
            }
            Expr::InitializerList(elems) => {
                for x in elems.iter_mut() {
                    drop_call_args(x, drops);
                }
            }
            Expr::Lambda { body, .. } => {
                for s in body.iter_mut() {
                    drop_call_args_stmt(s, drops);
                }
            }
            Expr::Variable(_) | Expr::Number(_) | Expr::String(_) | Expr::Bool(_) => {}
        }
    }
    fn drop_call_args_stmt(s: &mut Stmt, drops: &HashMap<String, (usize, Vec<usize>)>) {
        match s {
            Stmt::Return(Some(e)) | Stmt::Expr(e) => drop_call_args(e, drops),
            Stmt::Return(None) => {}
            Stmt::VarDecl { init: Some(e), .. } | Stmt::Destructure { init: Some(e), .. } => {
                drop_call_args(e, drops)
            }
            Stmt::VarDecl { init: None, .. } | Stmt::Destructure { init: None, .. } => {}
            Stmt::Block(ss) => {
                for x in ss.iter_mut() {
                    drop_call_args_stmt(x, drops);
                }
            }
            Stmt::If { condition, then_branch, else_branch } => {
                drop_call_args(condition, drops);
                for x in then_branch.iter_mut() {
                    drop_call_args_stmt(x, drops);
                }
                if let Some(els) = else_branch {
                    for x in els.iter_mut() {
                        drop_call_args_stmt(x, drops);
                    }
                }
            }
            Stmt::For { init, condition, update, body } => {
                drop_call_args_stmt(init, drops);
                if let Some(c) = condition {
                    drop_call_args(c, drops);
                }
                if let Some(u) = update {
                    drop_call_args(u, drops);
                }
                for x in body.iter_mut() {
                    drop_call_args_stmt(x, drops);
                }
            }
            Stmt::While { condition, body } => {
                drop_call_args(condition, drops);
                for x in body.iter_mut() {
                    drop_call_args_stmt(x, drops);
                }
            }
            Stmt::Switch { scrutinee, cases } => {
                drop_call_args(scrutinee, drops);
                for c in cases.iter_mut() {
                    if let Some(l) = &mut c.label {
                        drop_call_args(l, drops);
                    }
                    for x in c.body.iter_mut() {
                        drop_call_args_stmt(x, drops);
                    }
                }
            }
        }
    }
    let mut module = module.clone();
    // Drop params from definitions (arity-guarded like call sites:
    // overloads with other arities keep their params).
    for f in module.functions.iter_mut() {
        if f.scope.is_some() {
            continue;
        }
        if let Some((arity, idxs)) = drops.get(&f.name) {
            if f.parameters.len() != *arity {
                continue;
            }
            let mut kept = Vec::with_capacity(f.parameters.len());
            for (i, p) in f.parameters.drain(..).enumerate() {
                if !idxs.contains(&i) {
                    kept.push(p);
                }
            }
            f.parameters = kept;
        }
    }
    // Drop args at call sites (free-function calls only).
    for f in module.functions.iter_mut() {
        for s in f.body.iter_mut() {
            drop_call_args_stmt(s, &drops);
        }
    }
    module
}

/// Desugars struct construction in declarations (`S x = S(a, b)` →
/// `S x = {a, b}`) for aggregate structs.
///
/// HLSL has no C++-style struct constructors (`S(a, b)` is rejected for
/// non-numeric types); brace-init `S x = {a, b}` is the legal form and,
/// unlike memberwise assignment expansion, initializes `const` members
/// correctly. Resource fields are allowed (brace-init `S x = {texA, …}`
/// and by-value passing verified legal via DXC with valid SPIR-V).
/// Only fires when sound: `S` names a declared struct, the argument count
/// exactly matches the field count, the construction type equals the
/// declared type, and NO user constructor (`S::S`) exists in the model (a
/// present constructor could carry logic/reordering, so those stay loud).
/// Argument-type mismatches stay loud at DXC (no guessing). Runs before
/// monomorphization so clones inherit brace inits.
pub(crate) fn desugar_struct_constructs(module: &Module) -> Module {
    use crate::ast::Stmt;

    let mut module = module.clone();
    // Struct field info (for arity + resource gates).
    let mut struct_fields: HashMap<String, Vec<Type>> = HashMap::new();
    for s in &module.structs {
        struct_fields.insert(
            s.name.clone(),
            s.fields.iter().map(|f| f.ty.clone()).collect(),
        );
    }
    // User-constructor names present in the model (`S` with scope `S`).
    // Any such definition vetoes aggregate treatment for `S`.
    let mut has_ctor: HashSet<String> = HashSet::new();
    for f in &module.functions {
        if let Some(scope) = f.scope.as_deref() {
            if f.name == scope {
                has_ctor.insert(scope.to_string());
            }
        }
    }
    // (Resource fields allowed in brace-init; no resource gate needed.)
    for f in module.functions.iter_mut() {
        // VarDecl inits only (expression-position `S(…)` needs
        // temp-introduction, out of scope; those stay loud).
        fn walk_stmts(
            ss: &mut [Stmt],
            struct_fields: &HashMap<String, Vec<Type>>,
            has_ctor: &HashSet<String>,
        ) {
            for s in ss.iter_mut() {
                match s {
                    Stmt::VarDecl { ty, init: Some(init), .. } => {
                        if let Type::Named(sname) = ty {
                            if let Expr::Constructor { ty: cty, arguments } = init {
                                if let Type::Named(cname) = cty {
                                    if sname == cname {
                                        if let Some(fields) = struct_fields.get(sname) {
                                            // Resource fields ALLOWED:
                                            // brace-init `S x = {texA, …}`
                                            // and by-value passing are legal
                                            // HLSL (DXC-verified with valid
                                            // SPIR-V); direct-brace emission
                                            // (not memberwise assigns)
                                            // keeps resources const-safe.
                                            if !has_ctor.contains(sname)
                                                && fields.len() == arguments.len()
                                            {
                                                *init = Expr::InitializerList(
                                                    std::mem::take(arguments),
                                                );
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        // Recurse into the (possibly rewritten) init.
                        if let Stmt::VarDecl { init: Some(e), .. } = s {
                            walk_expr_inits(e, struct_fields, has_ctor);
                        }
                    }
                    Stmt::VarDecl { init: None, .. } | Stmt::Destructure { .. } => {}
                    Stmt::Block(inner) => walk_stmts(inner, struct_fields, has_ctor),
                    Stmt::If { condition: _, then_branch, else_branch } => {
                        // Conditions/inits with `S(…)` in non-decl position
                        // stay (expression-position out of scope).
                        walk_stmts(then_branch, struct_fields, has_ctor);
                        if let Some(els) = else_branch {
                            walk_stmts(els, struct_fields, has_ctor);
                        }
                    }
                    Stmt::For { init, body, .. } => {
                        walk_stmts(std::slice::from_mut(init.as_mut()), struct_fields, has_ctor);
                        walk_stmts(body, struct_fields, has_ctor);
                    }
                    Stmt::While { body, .. } => {
                        walk_stmts(body, struct_fields, has_ctor);
                    }
                    Stmt::Switch { cases, .. } => {
                        for c in cases.iter_mut() {
                            walk_stmts(&mut c.body, struct_fields, has_ctor);
                        }
                    }
                    Stmt::Return(_) | Stmt::Expr(_) => {}
                }
            }
        }
        fn walk_expr_inits(
            _e: &Expr,
            _struct_fields: &HashMap<String, Vec<Type>>,
            _has_ctor: &HashSet<String>,
        ) {
            // Inits are rewritten at the VarDecl site above; nested
            // `S(…)` inside init expressions (e.g. `S x = f(S(a,b))`)
            // stays loud (would need temp-introduction). No-op.
        }
        walk_stmts(&mut f.body, &struct_fields, &has_ctor);
    }
    module
}

/// Materializes flattened row-slice array arguments for array parameters
/// (`f(arr[x])` where `arr` is a flattened 2D array and `f` takes an
/// array) via temporaries with copy-in/copy-out.
///
/// Row-major flattening (`int3 arr[3][2]` → `int3 arr[6]`, uses
/// `arr[i][j]` → `arr[i*2+j]`) breaks row slices: PSSL `arr[x]` on 2D is
/// a row (`int3[2]`, passable to `out int3[2]` params), but on flattened
/// 1D it is an element (`int3`, mismatched). Since valid PSSL only passes
/// rows where arrays are expected (element-vs-array mismatch would not
/// compile), a detected mismatch provably indicates a flattened
/// row-slice (no dims table needed). The fix copies the row through a
/// temporary (`int3 tmp[2]; tmp[k]=arr[x*P+k]; f(…,tmp,…);` plus
/// write-back `arr[x*P+k]=tmp[k]` for `out`/`inout`), preserving values
/// exactly (stride = param size = row size, by PSSL validity).
///
/// Soundness gates (any failure declines that call site, staying loud):
/// - unique free-function callee (no overloads), not recursive;
/// - param is an array type with known constant size, direction noted
///   (`in`/by-value: copy-in only; `out`: write-back only; `inout`: both);
/// - argument is a single `Index{arr, x}` where `arr` is array-typed
///   (variable or member) and the AST arg type is non-array (proving the
///   flattening mismatch; already-array args pass through untouched);
/// - enclosing statement is not `return` (write-back would be dead).
/// Temporaries splice around the enclosing statement in its block (fresh
/// `__rowsliceN` names). Runs before monomorphization so clones inherit.
/// Materializes flattened row-slice array arguments for array parameters
/// (`f(arr[x])` where `arr` is a flattened 2D array and `f` takes an
/// array) via temporaries with copy-in/copy-out.
///
/// Row-major flattening (`int3 arr[3][2]` → `int3 arr[6]`) breaks row
/// slices: PSSL `arr[x]` on 2D is a row (`int3[2]`, passable to `out
/// int3[2]` params), but on flattened 1D it is an element (`int3`,
/// mismatched). Valid PSSL only passes rows where arrays are expected,
/// so a detected element-vs-array mismatch provably indicates a
/// flattened row-slice. The fix copies through a temporary
/// (`int3 tmp[2]; tmp[k]=arr[x*P+k]; f(…,tmp,…);` plus write-back for
/// `out`/`inout`), preserving values exactly (stride = param size = row
/// size, by PSSL validity).
///
/// Soundness gates (any failure declines that call site, staying loud):
/// unique free-function callee; array param with known size; single
/// `Index{arr,x}` arg on array-typed `arr` whose element matches the
/// param element; direction sets copy-in (`in`/by-value), write-back
/// (`out`), or both (`inout`); at most one site per leaf statement
/// (multi-site ambiguity declines); leaf is `Expr`/`VarDecl` (never
/// `return`, headers, or lambdas — write-back/header placement unsound
/// there). Temporaries splice around the leaf in its block (fresh
/// `__rowsliceN`). Runs before monomorphization.
pub(crate) fn fix_rowslice_args(module: &Module) -> Module {
    use crate::ast::Stmt;

    fn arr_of(
        e: &Expr,
        env: &HashMap<String, (Type, Option<u32>)>,
        struct_fields: &HashMap<String, HashMap<String, (Type, Option<u32>)>>,
    ) -> Option<(Type, u32)> {
        match e {
            Expr::Variable(n) => match env.get(n) {
                Some((t, Some(sz))) if *sz > 0 => Some((t.clone(), *sz)),
                _ => None,
            },
            Expr::Member { object, member } => {
                let sname = match object.as_ref() {
                    Expr::Variable(n) => match env.get(n) {
                        Some((Type::Named(s), _)) => s.clone(),
                        _ => return None,
                    },
                    _ => return None,
                };
                match struct_fields.get(&sname)?.get(member) {
                    Some((t, Some(sz))) if *sz > 0 => Some((t.clone(), *sz)),
                    _ => None,
                }
            }
            _ => None,
        }
    }
    // Find row-slice call args in an expression: (callee, pidx, arr, idx).
    // Only direct `Index{arr,x}` (not nested double) on array-typed arr.
    fn collect(
        e: &Expr,
        env: &HashMap<String, (Type, Option<u32>)>,
        struct_fields: &HashMap<String, HashMap<String, (Type, Option<u32>)>>,
        callee_params: &HashMap<String, Vec<(Type, Option<u32>, Option<String>)>>,
        def_counts: &HashMap<String, usize>,
        out: &mut Vec<(String, usize, Expr, Expr)>,
    ) {
        match e {
            Expr::Call { function, arguments } => {
                if let Expr::Variable(callee) = function.as_ref() {
                    if !callee.contains('<')
                        && def_counts.get(callee).copied().unwrap_or(0) == 1
                    {
                        if let Some(params) = callee_params.get(callee) {
                            for (i, a) in arguments.iter().enumerate() {
                                if let Some((pty, parr, _)) = params.get(i) {
                                    if let Some(psz) = parr {
                                        if *psz > 0 {
                                            if let Expr::Index { object, index } = a {
                                                if !matches!(object.as_ref(), Expr::Index { .. }) {
                                                    if let Some((ety, _)) =
                                                        arr_of(object, env, struct_fields)
                                                    {
                                                        if ety == *pty {
                                                            out.push((
                                                                callee.clone(),
                                                                i,
                                                                object.as_ref().clone(),
                                                                index.as_ref().clone(),
                                                            ));
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                collect(function, env, struct_fields, callee_params, def_counts, out);
                for a in arguments {
                    collect(a, env, struct_fields, callee_params, def_counts, out);
                }
            }
            Expr::Member { object, .. } => {
                collect(object, env, struct_fields, callee_params, def_counts, out)
            }
            Expr::Index { object, index } => {
                collect(object, env, struct_fields, callee_params, def_counts, out);
                collect(index, env, struct_fields, callee_params, def_counts, out);
            }
            Expr::Constructor { arguments, .. } => {
                for a in arguments {
                    collect(a, env, struct_fields, callee_params, def_counts, out);
                }
            }
            Expr::Assign { lhs, rhs } => {
                collect(lhs, env, struct_fields, callee_params, def_counts, out);
                collect(rhs, env, struct_fields, callee_params, def_counts, out);
            }
            Expr::Unary { expr, .. } => {
                collect(expr, env, struct_fields, callee_params, def_counts, out)
            }
            Expr::Binary { left, right, .. } => {
                collect(left, env, struct_fields, callee_params, def_counts, out);
                collect(right, env, struct_fields, callee_params, def_counts, out);
            }
            Expr::Conditional { condition, then_branch, else_branch } => {
                collect(condition, env, struct_fields, callee_params, def_counts, out);
                collect(then_branch, env, struct_fields, callee_params, def_counts, out);
                collect(else_branch, env, struct_fields, callee_params, def_counts, out);
            }
            Expr::InitializerList(elems) => {
                for x in elems {
                    collect(x, env, struct_fields, callee_params, def_counts, out);
                }
            }
            Expr::Lambda { .. } => {}
            Expr::Variable(_) | Expr::Number(_) | Expr::String(_) | Expr::Bool(_) => {}
        }
    }
    // Rewrite the collected site's arg to temp (exact single Index match;
    // idempotent since temps are Variables).
    fn rewrite_arg(s: &mut Stmt, callee: &str, pidx: usize, tmp: &str) -> bool {
        fn rw(e: &mut Expr, callee: &str, pidx: usize, tmp: &str) -> bool {
            match e {
                Expr::Call { function, arguments } => {
                    if let Expr::Variable(c) = function.as_ref() {
                        if c == callee {
                            if let Some(a) = arguments.get_mut(pidx) {
                                if matches!(a, Expr::Index { .. }) {
                                    *a = Expr::Variable(tmp.to_string());
                                    return true;
                                }
                            }
                        }
                    }
                    let mut done = rw(function, callee, pidx, tmp);
                    for a in arguments.iter_mut() {
                        done |= rw(a, callee, pidx, tmp);
                    }
                    done
                }
                Expr::Member { object, .. } => rw(object, callee, pidx, tmp),
                Expr::Index { object, index } => {
                    rw(object, callee, pidx, tmp) | rw(index, callee, pidx, tmp)
                }
                Expr::Constructor { arguments, .. } => {
                    let mut done = false;
                    for a in arguments.iter_mut() {
                        done |= rw(a, callee, pidx, tmp);
                    }
                    done
                }
                Expr::Assign { lhs, rhs } => {
                    rw(lhs, callee, pidx, tmp) | rw(rhs, callee, pidx, tmp)
                }
                Expr::Unary { expr, .. } => rw(expr, callee, pidx, tmp),
                Expr::Binary { left, right, .. } => {
                    rw(left, callee, pidx, tmp) | rw(right, callee, pidx, tmp)
                }
                Expr::Conditional { condition, then_branch, else_branch } => {
                    rw(condition, callee, pidx, tmp)
                        | rw(then_branch, callee, pidx, tmp)
                        | rw(else_branch, callee, pidx, tmp)
                }
                Expr::InitializerList(elems) => {
                    let mut done = false;
                    for x in elems.iter_mut() {
                        done |= rw(x, callee, pidx, tmp);
                    }
                    done
                }
                Expr::Lambda { .. } => false,
                _ => false,
            }
        }
        match s {
            Stmt::Expr(e) => rw(e, callee, pidx, tmp),
            Stmt::VarDecl { init: Some(e), .. }
            | Stmt::Destructure { init: Some(e), .. } => rw(e, callee, pidx, tmp),
            _ => false,
        }
    }

    let mut module = module.clone();
    let mut callee_params: HashMap<String, Vec<(Type, Option<u32>, Option<String>)>> =
        HashMap::new();
    let mut def_counts: HashMap<String, usize> = HashMap::new();
    for f in &module.functions {
        if f.scope.is_none() {
            *def_counts.entry(f.name.clone()).or_insert(0) += 1;
        }
    }
    for f in &module.functions {
        if f.scope.is_none() && def_counts.get(&f.name).copied().unwrap_or(0) == 1 {
            callee_params.insert(
                f.name.clone(),
                f.parameters
                    .iter()
                    .map(|p| {
                        (
                            p.ty.clone(),
                            p.array_size,
                            p.direction.map(|d| d.as_str().to_string()),
                        )
                    })
                    .collect(),
            );
        }
    }
    if callee_params.is_empty() {
        return module;
    }
    let mut struct_fields: HashMap<String, HashMap<String, (Type, Option<u32>)>> =
        HashMap::new();
    for s in &module.structs {
        struct_fields.insert(
            s.name.clone(),
            s.fields.iter().map(|f| (f.name.clone(), (f.ty.clone(), f.array_size))).collect(),
        );
    }
    let mut global_tys: HashMap<String, (Type, Option<u32>)> = HashMap::new();
    for g in &module.globals {
        global_tys.insert(g.name.clone(), (g.ty.clone(), g.array_size));
    }
    for r in &module.resources {
        global_tys.insert(
            r.name.clone(),
            (
                Type::Resource {
                    kind: r.kind.clone(),
                    access: crate::ast::ResourceAccess::ReadOnly,
                    element: Box::new(r.element_ty.clone().unwrap_or(Type::UInt)),
                },
                None,
            ),
        );
    }
    for f in module.functions.iter_mut() {
        let mut env = global_tys.clone();
        for p in &f.parameters {
            env.insert(p.name.clone(), (p.ty.clone(), p.array_size));
        }
        // Top-level AND nested locals (shadowing hazards decline: a name
        // declared with >1 distinct type anywhere in the function is
        // removed, so `arr` resolution never uses a wrong scope's type).
        {
            fn collect_locals(
                ss: &[Stmt],
                seen: &mut HashMap<String, Vec<(Type, Option<u32>)>>,
            ) {
                for s in ss {
                    match s {
                        Stmt::VarDecl { ty, name, array_size, .. } => {
                            seen.entry(name.clone())
                                .or_default()
                                .push((ty.clone(), *array_size));
                        }
                        Stmt::Block(inner) => collect_locals(inner, seen),
                        Stmt::If { then_branch, else_branch, .. } => {
                            collect_locals(then_branch, seen);
                            if let Some(els) = else_branch {
                                collect_locals(els, seen);
                            }
                        }
                        Stmt::For { init, body, .. } => {
                            collect_locals(std::slice::from_ref(init.as_ref()), seen);
                            collect_locals(body, seen);
                        }
                        Stmt::While { body, .. } => collect_locals(body, seen),
                        Stmt::Switch { cases, .. } => {
                            for c in cases {
                                collect_locals(&c.body, seen);
                            }
                        }
                        _ => {}
                    }
                }
            }
            let mut seen: HashMap<String, Vec<(Type, Option<u32>)>> = HashMap::new();
            collect_locals(&f.body, &mut seen);
            for (name, tys) in seen {
                // Deduplicate identical re-declarations (sibling scopes
                // with the same type are sound to merge).
                let mut uniq: Vec<(Type, Option<u32>)> = Vec::new();
                for t in tys {
                    if !uniq.contains(&t) {
                        uniq.push(t);
                    }
                }
                if uniq.len() != 1 {
                    // Shadowed with differing types: remove entirely so
                    // resolution declines (never wrong-scope).
                    env.remove(&name);
                    continue;
                }
                let local_ty = uniq.into_iter().next().unwrap();
                match env.get(&name) {
                    // Unshadowed (new name) or identical re-declaration:
                    // record it.
                    None => {
                        env.insert(name, local_ty);
                    }
                    Some(existing) if *existing == local_ty => {}
                    // Shadows a param/global with a DIFFERENT type:
                    // remove so all uses of this name decline (sound;
                    // concedes fixes in unshadowed scopes to avoid ever
                    // resolving to the wrong scope's type).
                    Some(_) => {
                        env.remove(&name);
                    }
                }
            }
        }
        let mut taken: HashSet<String> = env.keys().cloned().collect();
        let mut counter = 0usize;
        // Recurse blocks; splice leaves with exactly one site.
        fn walk_block(
            ss: &mut Vec<Stmt>,
            callee_params: &HashMap<String, Vec<(Type, Option<u32>, Option<String>)>>,
            def_counts: &HashMap<String, usize>,
            env: &HashMap<String, (Type, Option<u32>)>,
            struct_fields: &HashMap<String, HashMap<String, (Type, Option<u32>)>>,
            taken: &mut HashSet<String>,
            counter: &mut usize,
        ) {
            // Index-based: leaf splices insert temps into THIS Vec around
            // the statement (preserving its scope — declarations stay
            // visible after; Block-wrapping would narrow them and break
            // later uses). Containers recurse (no length change here).
            let mut i = 0;
            while i < ss.len() {
                match &mut ss[i] {
                    Stmt::Block(inner) => {
                        walk_block(inner, callee_params, def_counts, env, struct_fields, taken, counter);
                        i += 1;
                    }
                    Stmt::If { then_branch, else_branch, .. } => {
                        walk_block(then_branch, callee_params, def_counts, env, struct_fields, taken, counter);
                        if let Some(els) = else_branch {
                            walk_block(els, callee_params, def_counts, env, struct_fields, taken, counter);
                        }
                        i += 1;
                    }
                    Stmt::For { init, body, .. } => {
                        // Init is a boxed Stmt; splice via single-element vec.
                        let mut tmp = vec![std::mem::replace(init.as_mut(), Stmt::Block(vec![]))];
                        walk_block(&mut tmp, callee_params, def_counts, env, struct_fields, taken, counter);
                        // (Init splices to at most a Block; restore first,
                        // prepend extras to body to avoid dropping.)
                        let mut it = tmp.into_iter();
                        if let Some(first) = it.next() {
                            **init = first;
                        }
                        let rest: Vec<Stmt> = it.collect();
                        if !rest.is_empty() {
                            let mut nb = rest;
                            nb.extend(std::mem::take(body));
                            *body = nb;
                        }
                        walk_block(body, callee_params, def_counts, env, struct_fields, taken, counter);
                        i += 1;
                    }
                    Stmt::While { body, .. } => {
                        walk_block(body, callee_params, def_counts, env, struct_fields, taken, counter);
                        i += 1;
                    }
                    Stmt::Switch { cases, .. } => {
                        for c in cases.iter_mut() {
                            walk_block(&mut c.body, callee_params, def_counts, env, struct_fields, taken, counter);
                        }
                        i += 1;
                    }
                    Stmt::Return(_) => {
                        // Write-backs dead after return: decline (no splice).
                        i += 1;
                    }
                    Stmt::Expr(_) | Stmt::VarDecl { .. } | Stmt::Destructure { .. } => {
                        // Leaf: collect sites; require EXACTLY ONE
                        // (multi-site ambiguity declines the leaf).
                        let mut sites: Vec<(String, usize, Expr, Expr)> = Vec::new();
                        match &ss[i] {
                            Stmt::Expr(e) => {
                                collect(e, env, struct_fields, callee_params, def_counts, &mut sites)
                            }
                            Stmt::VarDecl { init: Some(e), .. }
                            | Stmt::Destructure { init: Some(e), .. } => {
                                collect(e, env, struct_fields, callee_params, def_counts, &mut sites)
                            }
                            _ => {}
                        }
                        if sites.len() != 1 {
                            i += 1;
                            continue;
                        }
                        let (callee, pidx, arr, idx) = sites.into_iter().next().unwrap();
                        let Some(params) = callee_params.get(&callee) else {
                            i += 1;
                            continue;
                        };
                        let Some((pty, parr, pdir)) = params.get(pidx) else {
                            i += 1;
                            continue;
                        };
                        let Some(psz) = parr else {
                            i += 1;
                            continue;
                        };
                        if *psz == 0 {
                            i += 1;
                            continue;
                        };
                        let Some((arr_elem, _)) = arr_of(&arr, env, struct_fields) else {
                            i += 1;
                            continue;
                        };
                        if arr_elem != *pty {
                            i += 1;
                            continue;
                        }
                        let mut tname = format!("__rowslice{}", counter);
                        *counter += 1;
                        while taken.contains(&tname) {
                            tname = format!("__rowslice{}", counter);
                            *counter += 1;
                        }
                        taken.insert(tname.clone());
                        let (need_in, need_out) = match pdir.as_deref() {
                            Some("out") => (false, true),
                            Some("inout") => (true, true),
                            _ => (true, false),
                        };
                        let at = |k: u32| {
                            Expr::Index {
                                object: Box::new(arr.clone()),
                                index: Box::new(Expr::Binary {
                                    left: Box::new(Expr::Binary {
                                        left: Box::new(idx.clone()),
                                        op: crate::ast::BinaryOp::Mul,
                                        right: Box::new(Expr::Number(psz.to_string())),
                                    }),
                                    op: crate::ast::BinaryOp::Add,
                                    right: Box::new(Expr::Number(k.to_string())),
                                }),
                            }
                        };
                        let mut pre: Vec<Stmt> = vec![Stmt::VarDecl {
                            storage: None,
                            ty: pty.clone(),
                            name: tname.clone(),
                            array_size: Some(*psz),
                            init: None,
                        }];
                        if need_in {
                            for k in 0..*psz {
                                pre.push(Stmt::Expr(Expr::Assign {
                                    lhs: Box::new(Expr::Index {
                                        object: Box::new(Expr::Variable(tname.clone())),
                                        index: Box::new(Expr::Number(k.to_string())),
                                    }),
                                    rhs: Box::new(at(k)),
                                }));
                            }
                        }
                        let mut post: Vec<Stmt> = Vec::new();
                        if need_out {
                            for k in 0..*psz {
                                post.push(Stmt::Expr(Expr::Assign {
                                    lhs: Box::new(at(k)),
                                    rhs: Box::new(Expr::Index {
                                        object: Box::new(Expr::Variable(tname.clone())),
                                        index: Box::new(Expr::Number(k.to_string())),
                                    }),
                                }));
                            }
                        }
                        if rewrite_arg(&mut ss[i], &callee, pidx, &tname) {
                            // Splice into THIS Vec (scope-preserving: the
                            // statement keeps its level, decls stay visible).
                            let orig = std::mem::replace(&mut ss[i], Stmt::Block(vec![]));
                            let n_seq = pre.len() + 1 + post.len();
                            let mut seq = pre;
                            seq.push(orig);
                            seq.extend(post);
                            ss.splice(i..i + 1, seq);
                            i += n_seq;
                        } else {
                            // Rewrite missed; pre/post drop (no dangling
                            // temps, call stays loud). Advance past.
                            i += 1;
                        }
                    }
                }
            }
        }
        walk_block(&mut f.body, &callee_params, &def_counts, &env, &struct_fields, &mut taken, &mut counter);
    }
    module
}

/// Desugars whole-struct zero assignments and declarations (`s = 0`,
/// `S s = 0`) into memberwise zero assignments.
///
/// HLSL rejects assigning a scalar to a struct (`cannot implicitly
/// convert from 'literal int' to 'GS_OUTPUT'`), but PSSL zero-fills
/// aggregates this way (meshlet `primOutput.output = 0`). The desugar
/// expands through plain nested structs; resource-typed or array fields
/// decline (stays loud — no legal zero exists), as do structs with any
/// `operator` method (a user `operator=` would change the meaning).
/// Scalar/vector/matrix/bool/enum leaves keep `= 0` (HLSL splats
/// scalars across all of them). Statement-list positions only (a block
/// cannot appear in for-update/condition slots); re-running is a no-op
/// (expansions contain no struct-typed zero assigns).
pub(crate) fn desugar_struct_zero_assign(module: &Module) -> Module {
    use crate::ast::Stmt;

    fn is_zero_literal(e: &Expr) -> bool {
        match e {
            Expr::Number(text) => {
                let t = text
                    .trim()
                    .trim_end_matches(['f', 'F', 'u', 'U', 'l', 'L', 'h', 'H']);
                if t.is_empty() {
                    return false;
                }
                let neg = t.starts_with('-');
                let digits = t.trim_start_matches(['-', '+']);
                let parsed = match digits
                    .strip_prefix("0x")
                    .or_else(|| digits.strip_prefix("0X"))
                {
                    Some(hex) => {
                        u64::from_str_radix(hex.trim_end_matches(['u', 'U', 'l', 'L']), 16).ok()
                    }
                    None if digits.contains('.') || digits.contains('e') || digits.contains('E') => {
                        digits.parse::<f64>().ok().map(|v| v as u64)
                    }
                    None => digits.parse::<u64>().ok(),
                };
                match parsed {
                    Some(0) => !neg,
                    _ => false,
                }
            }
            _ => false,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn expand_zero(
        lhs: &Expr,
        struct_name: &str,
        layouts: &HashMap<String, Vec<(String, Type)>>,
        has_operator: &HashSet<String>,
        out: &mut Vec<Stmt>,
    ) -> bool {
        if has_operator.contains(struct_name) {
            return false;
        }
        let Some(fields) = layouts.get(struct_name) else {
            return false;
        };
        // Gate the whole expansion first: any illegal leaf vetoes, so a
        // partial memberwise store never miscompiles alongside loud residue.
        fn gate(
            sname: &str,
            layouts: &HashMap<String, Vec<(String, Type)>>,
            has_operator: &HashSet<String>,
        ) -> bool {
            if has_operator.contains(sname) {
                return false;
            }
            let Some(fields) = layouts.get(sname) else {
                return false;
            };
            for (_, fty) in fields {
                match fty {
                    Type::Resource { .. } => return false,
                    Type::Named(s) if layouts.contains_key(s) => {
                        if !gate(s, layouts, has_operator) {
                            return false;
                        }
                    }
                    _ => {}
                }
            }
            true
        }
        // NOTE: array-ness lives beside the type (array_size), invisible
        // here; array fields are conservatively vetoed by checking the
        // declaration layouts with sizes below via a second map. To keep
        // this pass simple the caller passes layouts only for scalar
        // shapes... (arrays handled: see array_fields gate in caller.)
        if !gate(struct_name, layouts, has_operator) {
            return false;
        }
        fn emit_leaves(
            base: &Expr,
            sname: &str,
            layouts: &HashMap<String, Vec<(String, Type)>>,
            out: &mut Vec<Stmt>,
        ) {
            let fields = layouts.get(sname).cloned().unwrap_or_default();
            for (fname, fty) in &fields {
                let target = Expr::Member {
                    object: Box::new(base.clone()),
                    member: fname.clone(),
                };
                match fty {
                    Type::Named(s) if layouts.contains_key(s) => {
                        emit_leaves(&target, s, layouts, out)
                    }
                    _ => out.push(Stmt::Expr(Expr::Assign {
                        lhs: Box::new(target),
                        rhs: Box::new(Expr::Number("0".to_string())),
                    })),
                }
            }
        }
        emit_leaves(lhs, struct_name, layouts, out);
        true
    }

    let mut module = module.clone();
    // Layouts with array flags (array fields veto the expansion: HLSL
    // cannot assign a scalar across an array without an unrolled loop).
    let mut array_fields: HashSet<(String, String)> = HashSet::new();
    let mut layouts: HashMap<String, Vec<(String, Type)>> = HashMap::new();
    for s in &module.structs {
        layouts.insert(
            s.name.clone(),
            s.fields
                .iter()
                .map(|f| (f.name.clone(), f.ty.clone()))
                .collect(),
        );
        for f in &s.fields {
            if f.array_size.is_some() {
                array_fields.insert((s.name.clone(), f.name.clone()));
            }
        }
    }
    let mut has_operator: HashSet<String> = HashSet::new();
    for f in &module.functions {
        if let Some(scope) = f.scope.as_deref() {
            if f.name == "operator" || f.name.starts_with("operator") {
                has_operator.insert(scope.to_string());
            }
        }
    }
    // Free-function operator overloads (`operator==` etc. are free): any
    // struct-typed operand could still customize `= 0`... `operator=`
    // as a free function is impossible in C++ (must be a member), so
    // member scan above suffices. Free conversion operators also cannot
    // affect assignment INTO a struct. No gate needed here.
    for f in &mut module.functions {
        let mut env: HashMap<String, Type> = HashMap::new();
        for p in &f.parameters {
            env.insert(p.name.clone(), p.ty.clone());
        }
        fn collect(stmts: &[Stmt], env: &mut HashMap<String, Type>) {
            for s in stmts {
                match s {
                    Stmt::VarDecl { name, ty, .. } => {
                        env.entry(name.clone()).or_insert(ty.clone());
                    }
                    Stmt::Block(ss) => collect(ss, env),
                    Stmt::If { then_branch, else_branch, .. } => {
                        collect(then_branch, env);
                        if let Some(eb) = else_branch {
                            collect(eb, env);
                        }
                    }
                    Stmt::For { init, body, .. } => {
                        collect(std::slice::from_ref(init), env);
                        collect(body, env);
                    }
                    Stmt::While { body, .. } => collect(body, env),
                    Stmt::Switch { cases, .. } => {
                        for c in cases {
                            collect(&c.body, env);
                        }
                    }
                    Stmt::Return(_) | Stmt::Destructure { .. } | Stmt::Expr(_) => {}
                }
            }
        }
        collect(&f.body, &mut env);
        for g in &module.globals {
            env.entry(g.name.clone()).or_insert(g.ty.clone());
        }
        fn walk_block(
            ss: &mut Vec<Stmt>,
            env: &HashMap<String, Type>,
            layouts: &HashMap<String, Vec<(String, Type)>>,
            array_fields: &HashSet<(String, String)>,
            has_operator: &HashSet<String>,
        ) {
            // Array-field veto needs struct context per path; check during
            // expansion by walking the resolved struct/segments.
            let mut i = 0;
            while i < ss.len() {
                // Splice flat (never a block): declarations must stay
                // visible to later statements, not trapped in a scope.
                let replacement: Option<Vec<Stmt>> = match &mut ss[i] {
                    Stmt::Expr(Expr::Assign { lhs, rhs }) if is_zero_literal(rhs) => {
                        expand_checked(lhs, env, layouts, array_fields, has_operator)
                    }
                    Stmt::VarDecl { ty: Type::Named(s), name, init: Some(init), .. }
                        if is_zero_literal(init) && layouts.contains_key(s) =>
                    {
                        // `S s = 0` → bare decl plus memberwise zeros.
                        let lhs = Expr::Variable(name.clone());
                        expand_checked(&lhs, env, layouts, array_fields, has_operator).map(
                            |mut seq| {
                                let mut out = vec![Stmt::VarDecl {
                                    storage: None,
                                    ty: Type::Named(s.clone()),
                                    name: name.clone(),
                                    array_size: None,
                                    init: None,
                                }];
                                out.append(&mut seq);
                                out
                            },
                        )
                    }
                    _ => None,
                };
                if let Some(seq) = replacement {
                    let n = seq.len();
                    ss.splice(i..=i, seq);
                    i += n;
                    continue;
                }
                // Recurse (into the possibly-new block too).
                match &mut ss[i] {
                    Stmt::Block(inner) => {
                        walk_block(inner, env, layouts, array_fields, has_operator)
                    }
                    Stmt::If { then_branch, else_branch, .. } => {
                        walk_block(then_branch, env, layouts, array_fields, has_operator);
                        if let Some(eb) = else_branch {
                            walk_block(eb, env, layouts, array_fields, has_operator);
                        }
                    }
                    Stmt::For { body, .. } => {
                        walk_block(body, env, layouts, array_fields, has_operator)
                    }
                    Stmt::While { body, .. } => {
                        walk_block(body, env, layouts, array_fields, has_operator)
                    }
                    Stmt::Switch { cases, .. } => {
                        for c in cases.iter_mut() {
                            walk_block(&mut c.body, env, layouts, array_fields, has_operator);
                        }
                    }
                    _ => {}
                }
                i += 1;
            }
            fn expand_checked(
                lhs: &Expr,
                env: &HashMap<String, Type>,
                layouts: &HashMap<String, Vec<(String, Type)>>,
                array_fields: &HashSet<(String, String)>,
                has_operator: &HashSet<String>,
            ) -> Option<Vec<Stmt>> {
                // Collect (struct, field) pairs along the path; the
                // terminal must itself be a struct (else None).
                fn pairs(
                    e: &Expr,
                    env: &HashMap<String, Type>,
                    layouts: &HashMap<String, Vec<(String, Type)>>,
                    acc: &mut Vec<(String, String)>,
                ) -> Option<String> {
                    match e {
                        Expr::Variable(n) => match env.get(n) {
                            Some(Type::Named(s)) if layouts.contains_key(s) => {
                                Some(s.clone())
                            }
                            _ => None,
                        },
                        Expr::Member { object, member } => {
                            let base = pairs(object, env, layouts, acc)?;
                            acc.push((base.clone(), member.clone()));
                            match layouts.get(&base)?.iter().find(|(n, _)| n == member) {
                                Some((_, Type::Named(s))) if layouts.contains_key(s) => {
                                    Some(s.clone())
                                }
                                Some(_) => None,
                                None => None,
                            }
                        }
                        _ => None,
                    }
                }
                let mut path_pairs: Vec<(String, String)> = Vec::new();
                let terminal = pairs(lhs, env, layouts, &mut path_pairs)?;
                for (s, fld) in &path_pairs {
                    if array_fields.contains(&(s.clone(), fld.clone())) {
                        return None;
                    }
                }
                // Also veto arrays INSIDE the terminal struct (gate in
                // expand_zero covers resources/operators; arrays need the
                // sized layouts here).
                fn subtree_has_array(
                    sname: &str,
                    layouts: &HashMap<String, Vec<(String, Type)>>,
                    array_fields: &HashSet<(String, String)>,
                    seen: &mut HashSet<String>,
                ) -> bool {
                    if !seen.insert(sname.to_string()) {
                        return false;
                    }
                    let Some(fields) = layouts.get(sname) else {
                        return false;
                    };
                    for (fname, fty) in fields {
                        if array_fields.contains(&(sname.to_string(), fname.clone())) {
                            return true;
                        }
                        if let Type::Named(s) = fty {
                            if layouts.contains_key(s)
                                && subtree_has_array(s, layouts, array_fields, seen)
                            {
                                return true;
                            }
                        }
                    }
                    false
                }
                if subtree_has_array(&terminal, layouts, array_fields, &mut HashSet::new()) {
                    return None;
                }
                let mut out = Vec::new();
                if expand_zero(lhs, &terminal, layouts, has_operator, &mut out) {
                    Some(out)
                } else {
                    None
                }
            }
        }
        walk_block(&mut f.body, &env, &layouts, &array_fields, &has_operator);
    }
    module
}

/// Hoists expression-position struct constructions to temporaries
/// (`f(S(a, b))` → `S tmp = {a, b}; f(tmp)`).
///
/// HLSL has no struct constructors and (unlike declarations) no inline
/// brace-init in arbitrary expression positions, so a `S(…)` construction
/// nested in a call argument, return, or other expression must materialize
/// via a temporary. The temporary is declared immediately before the
/// containing statement in the same block (so constructor arguments,
/// evaluated there, stay in scope) with brace-init (const-correct, as in
/// `desugar_struct_constructs`). Same soundness gates (declared struct,
/// full arity, no user constructor in the model; resource fields allowed
/// since brace-init + by-value passing verified legal via DXC; VarDecl
/// inits already handled by `desugar_struct_constructs` and skipped here
/// to avoid double-processing). Anything else stays loud. Runs before
/// monomorphization so clones inherit hoisted temporaries.
pub(crate) fn hoist_struct_constructs(module: &Module) -> Module {
    use crate::ast::Stmt;

    // Rewrite expression-position `S(…)` to hoisted temps, collecting
    // `(temp_stmts, …)` per statement. Returns true if hoisted.
    fn hoist_expr(
        e: &mut Expr,
        struct_fields: &HashMap<String, Vec<Type>>,
        has_ctor: &HashSet<String>,
        taken: &mut HashSet<String>,
        counter: &mut usize,
        pending: &mut Vec<Stmt>,
    ) -> bool {
        let mut changed = false;
        // Post-order first (inner constructs hoist before outer uses).
        match e {
            Expr::Call { function, arguments } => {
                changed |= hoist_expr(function, struct_fields, has_ctor, taken, counter, pending);
                for a in arguments.iter_mut() {
                    changed |= hoist_expr(a, struct_fields, has_ctor, taken, counter, pending);
                }
            }
            Expr::Member { object, .. } => {
                changed |= hoist_expr(object, struct_fields, has_ctor, taken, counter, pending);
            }
            Expr::Index { object, index } => {
                changed |= hoist_expr(object, struct_fields, has_ctor, taken, counter, pending);
                changed |= hoist_expr(index, struct_fields, has_ctor, taken, counter, pending);
            }
            Expr::Constructor { ty, arguments } => {
                // Struct construction in expression position (VarDecl
                // inits handled by `desugar_struct_constructs`; but if one
                // reaches here nested, hoist it too for uniformity —
                // EXCEPT direct VarDecl-init position which the caller
                // skips to avoid double-processing; see walk_stmt).
                // Resource fields allowed (brace-init + by-value passing
                // verified legal via DXC with valid SPIR-V).
                let hoist_here = match ty {
                    Type::Named(sname) => struct_fields
                        .get(sname)
                        .is_some_and(|fields| {
                            !has_ctor.contains(sname)
                                && fields.len() == arguments.len()
                        }),
                    _ => false,
                };
                if hoist_here {
                    // Hoist children first (they may themselves hoist).
                    for a in arguments.iter_mut() {
                        changed |= hoist_expr(a, struct_fields, has_ctor, taken, counter, pending);
                    }
                    // Fresh temp + brace-init declaration.
                    let mut tname = format!("__struct_tmp{}", counter);
                    *counter += 1;
                    while taken.contains(&tname) {
                        tname = format!("__struct_tmp{}", counter);
                        *counter += 1;
                    }
                    taken.insert(tname.clone());
                    // Re-collect args (children may have rewritten).
                    // (Arguments already walked above; move them.)
                    let args = std::mem::take(arguments);
                    pending.push(Stmt::VarDecl {
                        storage: None,
                        ty: ty.clone(),
                        name: tname.clone(),
                        array_size: None,
                        init: Some(Expr::InitializerList(args)),
                    });
                    *e = Expr::Variable(tname);
                    return true;
                }
                for a in arguments.iter_mut() {
                    changed |= hoist_expr(a, struct_fields, has_ctor, taken, counter, pending);
                }
            }
            Expr::Assign { lhs, rhs } => {
                changed |= hoist_expr(lhs, struct_fields, has_ctor, taken, counter, pending);
                changed |= hoist_expr(rhs, struct_fields, has_ctor, taken, counter, pending);
            }
            Expr::Unary { expr, .. } => {
                changed |= hoist_expr(expr, struct_fields, has_ctor, taken, counter, pending);
            }
            Expr::Binary { left, right, .. } => {
                changed |= hoist_expr(left, struct_fields, has_ctor, taken, counter, pending);
                changed |= hoist_expr(right, struct_fields, has_ctor, taken, counter, pending);
            }
            Expr::Conditional { condition, then_branch, else_branch } => {
                changed |= hoist_expr(condition, struct_fields, has_ctor, taken, counter, pending);
                changed |= hoist_expr(then_branch, struct_fields, has_ctor, taken, counter, pending);
                changed |= hoist_expr(else_branch, struct_fields, has_ctor, taken, counter, pending);
            }
            Expr::InitializerList(elems) => {
                for x in elems.iter_mut() {
                    changed |= hoist_expr(x, struct_fields, has_ctor, taken, counter, pending);
                }
            }
            Expr::Lambda { body, .. } => {
                for s in body.iter_mut() {
                    // Lambdas get their own scope; hoist within (temps
                    // declared inside the lambda body, before use there).
                    // (Simplification: recurse with fresh pending per
                    // statement via walk_stmt below; here just recurse
                    // expressions and let statement-level insertion
                    // handle placement — actually lambdas need scoped
                    // placement. To stay sound and simple, DO NOT hoist
                    // across lambda boundaries: decline constructs inside
                    // lambdas by not recursing with pending (walk for
                    // nested non-construct shapes only would miss.
                    // Decision: skip lambda bodies entirely (stay loud if
                    // they contain struct constructs; rare).)
                }
            }
            Expr::Variable(_) | Expr::Number(_) | Expr::String(_) | Expr::Bool(_) => {}
        }
        changed
    }
    fn walk_stmt(
        s: &mut Stmt,
        struct_fields: &HashMap<String, Vec<Type>>,
        has_ctor: &HashSet<String>,
        taken: &mut HashSet<String>,
        counter: &mut usize,
    ) {
        // Per-statement pending temps (inserted before this statement).
        // For block-like statements, recurse into children (each child
        // statement gets its own pre-insertion point).
        match s {
            Stmt::Return(Some(e)) | Stmt::Expr(e) => {
                let mut pending = Vec::new();
                hoist_expr(e, struct_fields, has_ctor, taken, counter, &mut pending);
                if !pending.is_empty() {
                    // Splice temps before this statement by converting to
                    // a block: `{ temps...; original; }`. (Caller block
                    // handles flattening; top-level statements become
                    // blocks — scope-safe since temps are fresh.)
                    let orig = std::mem::replace(s, Stmt::Block(vec![]));
                    if let Stmt::Block(ss) = s {
                        ss.extend(pending);
                        ss.push(orig);
                    }
                }
            }
            Stmt::Return(None) => {}
            Stmt::VarDecl { init: Some(e), .. } | Stmt::Destructure { init: Some(e), .. } => {
                // VarDecl direct inits handled by `desugar_struct_constructs`
                // (brace rewrite, no temp needed). Nested constructs deeper
                // inside the init (e.g. `S x = f(S(a,b))`) still hoist.
                let mut pending = Vec::new();
                hoist_expr(e, struct_fields, has_ctor, taken, counter, &mut pending);
                if !pending.is_empty() {
                    let orig = std::mem::replace(s, Stmt::Block(vec![]));
                    if let Stmt::Block(ss) = s {
                        ss.extend(pending);
                        ss.push(orig);
                    }
                }
            }
            Stmt::VarDecl { init: None, .. } | Stmt::Destructure { init: None, .. } => {}
            Stmt::Block(ss) => {
                // Process in place with index-aware insertion (temps go
                // before the needing statement, same block).
                let mut i = 0;
                while i < ss.len() {
                    // Collect pending for ss[i] by walking it into a temp
                    // buffer: walk a CLONE to find hoists, then apply? No
                    // — walk directly with a per-statement pending list,
                    // then splice.
                    // (Implement via recursion into walk_stmt on the single
                    // statement, capturing its pending separately is complex
                    // with the current signatures. Simpler: recurse into
                    // children first (they handle their own pre-insertion
                    // via block-splicing above), then handle direct
                    // expression children of THIS block's statements via
                    // the Return/Expr/VarDecl arms (which splice). For a
                    // bare Block, just recurse into each child (each child
                    // splices itself if needed). No block-level pending
                    // needed here beyond child splicing.)
                    walk_stmt(&mut ss[i], struct_fields, has_ctor, taken, counter);
                    i += 1;
                }
            }
            Stmt::If { condition, then_branch, else_branch } => {
                // Condition constructs hoist before the `if` (same outer
                // block) — but we're inside the `if` statement itself.
                // To keep placement sound, hoist condition temps by
                // splicing THIS `if` (caller block prepends). Since we
                // can't prepend to ourselves here, handle conditions by
                // splicing: convert `if (f(S(..)))` — hmm. Simplest sound
                // approach: recurse into branches (they splice
                // themselves), and for the CONDITION, if it hoists, splice
                // this whole `if` (caller will have already... no.
                // Decision: hoist condition into a pre-block by splicing
                // this statement (like Return/Expr above).
                let mut pending = Vec::new();
                hoist_expr(condition, struct_fields, has_ctor, taken, counter, &mut pending);
                // (Pending for condition would need outer-block insertion;
                // splicing `if` itself doesn't place condition temps
                // before the test. Instead, DECLINE condition hoists by
                // restoring? Complex. Pragmatic: allow it — splice temps
                // + `if` together in a block; temps execute before the
                // `if` test within the block, which dominates correctly
                // and args are in scope (same block as the `if`). SOUND.)
                if !pending.is_empty() {
                    let orig = std::mem::replace(s, Stmt::Block(vec![]));
                    if let Stmt::Block(ssb) = s {
                        ssb.extend(pending);
                        ssb.push(orig);
                    }
                    // Re-walk the spliced original to handle branches
                    // (now inside the block tail).
                    if let Stmt::Block(ssb) = s {
                        if let Some(last) = ssb.last_mut() {
                            walk_stmt(last, struct_fields, has_ctor, taken, counter);
                        }
                    }
                    return;
                }
                for x in then_branch.iter_mut() {
                    walk_stmt(x, struct_fields, has_ctor, taken, counter);
                }
                if let Some(els) = else_branch {
                    for x in els.iter_mut() {
                        walk_stmt(x, struct_fields, has_ctor, taken, counter);
                    }
                }
            }
            Stmt::For { init, condition, update, body } => {
                walk_stmt(init, struct_fields, has_ctor, taken, counter);
                // Condition/update constructs: splice whole `for` like `if`
                // above if they hoist (sound: temps dominate loop in block).
                let mut pending = Vec::new();
                if let Some(c) = condition {
                    hoist_expr(c, struct_fields, has_ctor, taken, counter, &mut pending);
                }
                if let Some(u) = update {
                    hoist_expr(u, struct_fields, has_ctor, taken, counter, &mut pending);
                }
                if !pending.is_empty() {
                    let orig = std::mem::replace(s, Stmt::Block(vec![]));
                    if let Stmt::Block(ssb) = s {
                        ssb.extend(pending);
                        ssb.push(orig);
                    }
                    if let Stmt::Block(ssb) = s {
                        if let Some(last) = ssb.last_mut() {
                            walk_stmt(last, struct_fields, has_ctor, taken, counter);
                        }
                    }
                    return;
                }
                for x in body.iter_mut() {
                    walk_stmt(x, struct_fields, has_ctor, taken, counter);
                }
            }
            Stmt::While { condition, body } => {
                let mut pending = Vec::new();
                hoist_expr(condition, struct_fields, has_ctor, taken, counter, &mut pending);
                if !pending.is_empty() {
                    let orig = std::mem::replace(s, Stmt::Block(vec![]));
                    if let Stmt::Block(ssb) = s {
                        ssb.extend(pending);
                        ssb.push(orig);
                    }
                    if let Stmt::Block(ssb) = s {
                        if let Some(last) = ssb.last_mut() {
                            walk_stmt(last, struct_fields, has_ctor, taken, counter);
                        }
                    }
                    return;
                }
                for x in body.iter_mut() {
                    walk_stmt(x, struct_fields, has_ctor, taken, counter);
                }
            }
            Stmt::Switch { scrutinee, cases } => {
                let mut pending = Vec::new();
                hoist_expr(scrutinee, struct_fields, has_ctor, taken, counter, &mut pending);
                if !pending.is_empty() {
                    let orig = std::mem::replace(s, Stmt::Block(vec![]));
                    if let Stmt::Block(ssb) = s {
                        ssb.extend(pending);
                        ssb.push(orig);
                    }
                    if let Stmt::Block(ssb) = s {
                        if let Some(last) = ssb.last_mut() {
                            walk_stmt(last, struct_fields, has_ctor, taken, counter);
                        }
                    }
                    return;
                }
                for c in cases.iter_mut() {
                    if let Some(l) = &mut c.label {
                        // Labels with hoists: splice case body? Labels must
                        // stay first in case. Decline label hoists by
                        // skipping (stay loud if a construct hides there;
                        // vanishingly rare).
                    }
                    for x in c.body.iter_mut() {
                        walk_stmt(x, struct_fields, has_ctor, taken, counter);
                    }
                }
            }
        }
    }

    let mut module = module.clone();
    let mut struct_fields: HashMap<String, Vec<Type>> = HashMap::new();
    for s in &module.structs {
        struct_fields.insert(
            s.name.clone(),
            s.fields.iter().map(|f| f.ty.clone()).collect(),
        );
    }
    let mut has_ctor: HashSet<String> = HashSet::new();
    for f in &module.functions {
        if let Some(scope) = f.scope.as_deref() {
            if f.name == scope {
                has_ctor.insert(scope.to_string());
            }
        }
    }
    // Taken names per function (params + locals + globals) for fresh temps.
    let mut globals: HashSet<String> = HashSet::new();
    for g in &module.globals {
        globals.insert(g.name.clone());
    }
    for r in &module.resources {
        globals.insert(r.name.clone());
    }
    for f in module.functions.iter_mut() {
        let mut taken = globals.clone();
        for p in &f.parameters {
            taken.insert(p.name.clone());
        }
        // (Nested locals missed; fresh `__struct_tmpN` prefix avoids
        // collisions in practice; `taken` check guards top-level.)
        let mut counter = 0usize;
        // Seed counter past existing `__struct_tmpN` to avoid churn.
        for s in &f.body {
            // (Light scan skipped; while-loop in hoist guards anyway.)
            let _ = s;
        }
        for s in f.body.iter_mut() {
            walk_stmt(s, &struct_fields, &has_ctor, &mut taken, &mut counter);
        }
    }
    module
}

/// Linearizes struct-field 2D array uses (`obj.field[i][j]` →
/// `obj.field[i*D1+j]`, row-major) for fields recorded in
/// `Module::field_multidim` (declarations flattened to the product size
/// at parse). Mirrors the global/local `multidim` design, resolved via
/// struct types (not bare names) so identically-named fields in different
/// structs stay distinct. Only fires on exact 2D constant dims with a
/// directly nested double-index; anything else (3D+, unknown dims,
/// non-nested, unresolvable object struct) keeps today's loud path.
/// Runs before monomorphization so clones inherit linearized indices.
pub(crate) fn linearize_field_indices(module: &Module) -> Module {
    use crate::ast::{BinaryOp, Stmt};

    // Struct of an object expression (Variable via env, or single Member
    // via struct field table). Deeper chains decline (loud, safe).
    fn obj_struct(
        e: &Expr,
        env: &HashMap<String, Type>,
        struct_fields: &HashMap<String, HashMap<String, Type>>,
    ) -> Option<String> {
        match e {
            Expr::Variable(n) => match env.get(n) {
                Some(Type::Named(s)) => Some(s.clone()),
                _ => None,
            },
            Expr::Member { object, member } => {
                let os = obj_struct(object, env, struct_fields)?;
                match struct_fields.get(&os)?.get(member) {
                    Some(Type::Named(s)) => Some(s.clone()),
                    _ => None,
                }
            }
            _ => None,
        }
    }
    fn walk_expr(
        e: &mut Expr,
        env: &HashMap<String, Type>,
        struct_fields: &HashMap<String, HashMap<String, Type>>,
        field_dims: &HashMap<String, Vec<u32>>,
    ) {
        // Post-order.
        match e {
            Expr::Call { function, arguments } => {
                walk_expr(function, env, struct_fields, field_dims);
                for a in arguments.iter_mut() {
                    walk_expr(a, env, struct_fields, field_dims);
                }
            }
            Expr::Member { object, .. } => walk_expr(object, env, struct_fields, field_dims),
            Expr::Index { object, index } => {
                walk_expr(object, env, struct_fields, field_dims);
                walk_expr(index, env, struct_fields, field_dims);
            }
            Expr::Constructor { arguments, .. } => {
                for a in arguments.iter_mut() {
                    walk_expr(a, env, struct_fields, field_dims);
                }
            }
            Expr::Assign { lhs, rhs } => {
                walk_expr(lhs, env, struct_fields, field_dims);
                walk_expr(rhs, env, struct_fields, field_dims);
            }
            Expr::Unary { expr, .. } => walk_expr(expr, env, struct_fields, field_dims),
            Expr::Binary { left, right, .. } => {
                walk_expr(left, env, struct_fields, field_dims);
                walk_expr(right, env, struct_fields, field_dims);
            }
            Expr::Conditional { condition, then_branch, else_branch } => {
                walk_expr(condition, env, struct_fields, field_dims);
                walk_expr(then_branch, env, struct_fields, field_dims);
                walk_expr(else_branch, env, struct_fields, field_dims);
            }
            Expr::InitializerList(elems) => {
                for x in elems.iter_mut() {
                    walk_expr(x, env, struct_fields, field_dims);
                }
            }
            Expr::Lambda { body, .. } => {
                for s in body.iter_mut() {
                    walk_stmt(s, env, struct_fields, field_dims);
                }
            }
            Expr::Variable(_) | Expr::Number(_) | Expr::String(_) | Expr::Bool(_) => {}
        }
        // `obj.field[i][j]` with 2D dims → `obj.field[i*D1+j]`.
        if let Expr::Index { object: outer_obj, index: outer_idx } = e {
            if let Expr::Index { object: inner_obj, index: inner_idx } = outer_obj.as_ref() {
                if let Expr::Member { object: base, member: field } = inner_obj.as_ref() {
                    if let Some(sname) = obj_struct(base, env, struct_fields) {
                        if let Some(dims) = field_dims.get(&format!("{sname}::{field}")) {
                            if dims.len() == 2 {
                                let d1 = dims[1].to_string();
                                let lin = Expr::Binary {
                                    left: Box::new(Expr::Binary {
                                        left: inner_idx.clone(),
                                        op: BinaryOp::Mul,
                                        right: Box::new(Expr::Number(d1)),
                                    }),
                                    op: BinaryOp::Add,
                                    right: outer_idx.clone(),
                                };
                                *e = Expr::Index {
                                    object: Box::new(Expr::Member {
                                        object: base.clone(),
                                        member: field.clone(),
                                    }),
                                    index: Box::new(lin),
                                };
                            }
                        }
                    }
                }
            }
        }
    }
    fn walk_stmt(
        s: &mut Stmt,
        env: &HashMap<String, Type>,
        struct_fields: &HashMap<String, HashMap<String, Type>>,
        field_dims: &HashMap<String, Vec<u32>>,
    ) {
        match s {
            Stmt::Return(Some(e)) | Stmt::Expr(e) => {
                walk_expr(e, env, struct_fields, field_dims)
            }
            Stmt::Return(None) => {}
            Stmt::VarDecl { init: Some(e), .. } | Stmt::Destructure { init: Some(e), .. } => {
                walk_expr(e, env, struct_fields, field_dims)
            }
            Stmt::VarDecl { init: None, .. } | Stmt::Destructure { init: None, .. } => {}
            Stmt::Block(ss) => {
                for x in ss.iter_mut() {
                    walk_stmt(x, env, struct_fields, field_dims);
                }
            }
            Stmt::If { condition, then_branch, else_branch } => {
                walk_expr(condition, env, struct_fields, field_dims);
                for x in then_branch.iter_mut() {
                    walk_stmt(x, env, struct_fields, field_dims);
                }
                if let Some(els) = else_branch {
                    for x in els.iter_mut() {
                        walk_stmt(x, env, struct_fields, field_dims);
                    }
                }
            }
            Stmt::For { init, condition, update, body } => {
                walk_stmt(init, env, struct_fields, field_dims);
                if let Some(c) = condition {
                    walk_expr(c, env, struct_fields, field_dims);
                }
                if let Some(u) = update {
                    walk_expr(u, env, struct_fields, field_dims);
                }
                for x in body.iter_mut() {
                    walk_stmt(x, env, struct_fields, field_dims);
                }
            }
            Stmt::While { condition, body } => {
                walk_expr(condition, env, struct_fields, field_dims);
                for x in body.iter_mut() {
                    walk_stmt(x, env, struct_fields, field_dims);
                }
            }
            Stmt::Switch { scrutinee, cases } => {
                walk_expr(scrutinee, env, struct_fields, field_dims);
                for c in cases.iter_mut() {
                    if let Some(l) = &mut c.label {
                        walk_expr(l, env, struct_fields, field_dims);
                    }
                    for x in c.body.iter_mut() {
                        walk_stmt(x, env, struct_fields, field_dims);
                    }
                }
            }
        }
    }

    let mut module = module.clone();
    if module.field_multidim.is_empty() {
        return module;
    }
    let mut struct_fields: HashMap<String, HashMap<String, Type>> = HashMap::new();
    for s in &module.structs {
        struct_fields.insert(
            s.name.clone(),
            s.fields.iter().map(|f| (f.name.clone(), f.ty.clone())).collect(),
        );
    }
    let mut global_tys: HashMap<String, Type> = HashMap::new();
    for g in &module.globals {
        global_tys.insert(g.name.clone(), g.ty.clone());
    }
    for r in &module.resources {
        global_tys.insert(
            r.name.clone(),
            Type::Resource {
                kind: r.kind.clone(),
                access: crate::ast::ResourceAccess::ReadOnly,
                element: Box::new(r.element_ty.clone().unwrap_or(Type::UInt)),
            },
        );
    }
    for f in module.functions.iter_mut() {
        let mut env = global_tys.clone();
        for p in &f.parameters {
            env.insert(p.name.clone(), p.ty.clone());
        }
        for s in &f.body {
            if let Stmt::VarDecl { ty, name, .. } = s {
                env.insert(name.clone(), ty.clone());
            }
        }
        for s in f.body.iter_mut() {
            walk_stmt(s, &env, &struct_fields, &module.field_multidim);
        }
    }
    module
}

/// Narrows vector-typed params provably holding scalar values to scalars.
///
/// PSSL implicitly splats scalars to vectors at calls (`occlusion(
/// lds_offset(…), …)` passes `int` to an `int2 offset` param, broadcasting
/// to `(s, s)`). When every call site passes a scalar and the param is
/// never used in explicitly-vector syntax, narrowing `int2 → int`
/// preserves every computed value exactly (splat invariance: all lanes
/// equal the scalar; scalar broadcasts back identically; HLSL implicitly
/// splats scalars where vectors are needed). This fixes vector-typed
/// array indices (`buf[base + offset]` with `int2` index is illegal HLSL;
/// scalar index is legal) and similar scalar-position uses.
///
/// Soundness gates (any failure declines that param, staying loud):
/// - free function, unique definition, not the entry, non-recursive,
///   return type not a vector;
/// - param directionless (no `out`/`inout`), of vector type;
/// - ≥1 call site, and EVERY call passes a scalar argument (numeric/bool
///   literal, scalar-typed variable via caller env, or scalar
///   constructor; vectors/unknown decline);
/// - param never assigned, never `P.member` / `P[i]` (explicitly-vector
///   syntax), never a direct call argument (callee could demand a vector
///   or bind `out`), never in an initializer list (aggregate subtleties).
/// All other uses (arithmetic, comparisons, indices, returns) are safe
/// by splat invariance + implicit conversion.
pub(crate) fn narrow_vector_params(module: &Module, entry_name: Option<&str>) -> Module {
    use crate::ast::{ScalarType, Stmt};

    fn scalar_ty(base: &ScalarType) -> Type {
        match base {
            ScalarType::Float => Type::Float,
            ScalarType::Half => Type::Half,
            ScalarType::Int => Type::Int,
            ScalarType::UInt => Type::UInt,
            ScalarType::Bool => Type::Bool,
        }
    }
    fn is_scalar_arg(
        e: &Expr,
        caller_env: &HashMap<String, Type>,
        fn_rets: &HashMap<String, Type>,
    ) -> bool {
        match e {
            Expr::Number(_) | Expr::Bool(_) => true,
            Expr::String(_) => false,
            Expr::Variable(n) => matches!(
                caller_env.get(n),
                Some(Type::Bool)
                    | Some(Type::Int)
                    | Some(Type::UInt)
                    | Some(Type::Float)
                    | Some(Type::Half)
                    | Some(Type::Double)
            ),
            Expr::Constructor { ty, arguments } => {
                matches!(
                    ty,
                    Type::Bool
                        | Type::Int
                        | Type::UInt
                        | Type::Float
                        | Type::Half
                        | Type::Double
                ) && arguments.iter().all(|a| is_scalar_arg(a, caller_env, fn_rets))
            }
            Expr::Call { function, .. } => {
                // Scalar-returning callees (`lds_offset(…)` → `int`) are
                // scalar-valued. Only unambiguous single-def free
                // functions with scalar returns qualify (table below);
                // anything else declines (strict).
                if let Expr::Variable(c) = function.as_ref() {
                    if !c.contains('<') {
                        if let Some(rt) = fn_rets.get(c) {
                            return matches!(
                                rt,
                                Type::Bool
                                    | Type::Int
                                    | Type::UInt
                                    | Type::Float
                                    | Type::Half
                                    | Type::Double
                            );
                        }
                    }
                }
                false
            }
            // Member/index/calls/assigns: value shape unproven (a member
            // could be vector; calls opaque). Decline (strict).
            _ => false,
        }
    }
    // True when param `pname` is used ONLY in scalar-compatible positions
    // (never assigned, dotted, indexed as base, direct-called, or listed).
    fn uses_scalar_only(
        ss: &[Stmt],
        pname: &str,
    ) -> bool {
        fn expr_ok(e: &Expr, pname: &str) -> bool {
            match e {
                Expr::Variable(n) => {
                    // Bare use (arithmetic operand, index, return, …) is
                    // fine by splat invariance.
                    let _ = n;
                    let _ = pname;
                    true
                }
                Expr::Call { function, arguments } => {
                    // Direct `f(P)` could demand a vector or bind `out`:
                    // decline. (P nested inside args, e.g. in an index, is
                    // fine and recursed below.)
                    if let Expr::Variable(n) = function.as_ref() {
                        let _ = n;
                    }
                    // Check function expr + args; direct-arg test below.
                    if !expr_ok(function, pname) {
                        return false;
                    }
                    for a in arguments {
                        // Direct argument that IS the param: decline.
                        if matches!(a, Expr::Variable(n) if n == pname) {
                            return false;
                        }
                        if !expr_ok(a, pname) {
                            return false;
                        }
                    }
                    true
                }
                Expr::Member { object, .. } => {
                    // `P.m` requires a vector: decline. Otherwise recurse
                    // (P may appear deeper, e.g. in the object chain —
                    // but a bare P as ultimate object with a member is
                    // exactly this case; nested P inside object of a
                    // member on something else is fine).
                    if let Expr::Variable(n) = object.as_ref() {
                        if n == pname {
                            return false;
                        }
                    }
                    expr_ok(object, pname)
                }
                Expr::Index { object, index } => {
                    // `P[i]` requires a vector base: decline.
                    if let Expr::Variable(n) = object.as_ref() {
                        if n == pname {
                            return false;
                        }
                    }
                    expr_ok(object, pname) && expr_ok(index, pname)
                }
                Expr::Constructor { arguments, .. } => {
                    for a in arguments {
                        if !expr_ok(a, pname) {
                            return false;
                        }
                    }
                    true
                }
                Expr::Assign { lhs, rhs } => {
                    // `P = …` (assigned): decline (not scalar-valued).
                    if matches!(lhs.as_ref(), Expr::Variable(n) if n == pname) {
                        return false;
                    }
                    expr_ok(lhs, pname) && expr_ok(rhs, pname)
                }
                Expr::Unary { expr, .. } => expr_ok(expr, pname),
                Expr::Binary { left, right, .. } => {
                    expr_ok(left, pname) && expr_ok(right, pname)
                }
                Expr::Conditional { condition, then_branch, else_branch } => {
                    expr_ok(condition, pname)
                        && expr_ok(then_branch, pname)
                        && expr_ok(else_branch, pname)
                }
                Expr::InitializerList(elems) => {
                    // Aggregate subtleties (splat vs element): decline if
                    // P appears at all.
                    for x in elems {
                        let mut found = false;
                        fn mentions(e: &Expr, pname: &str, out: &mut bool) {
                            match e {
                                Expr::Variable(n) if n == pname => *out = true,
                                Expr::Call { function, arguments } => {
                                    mentions(function, pname, out);
                                    for a in arguments {
                                        mentions(a, pname, out);
                                    }
                                }
                                Expr::Member { object, .. } => mentions(object, pname, out),
                                Expr::Index { object, index } => {
                                    mentions(object, pname, out);
                                    mentions(index, pname, out);
                                }
                                Expr::Constructor { arguments, .. } => {
                                    for a in arguments {
                                        mentions(a, pname, out);
                                    }
                                }
                                Expr::Assign { lhs, rhs } => {
                                    mentions(lhs, pname, out);
                                    mentions(rhs, pname, out);
                                }
                                Expr::Unary { expr, .. } => mentions(expr, pname, out),
                                Expr::Binary { left, right, .. } => {
                                    mentions(left, pname, out);
                                    mentions(right, pname, out);
                                }
                                Expr::Conditional { condition, then_branch, else_branch } => {
                                    mentions(condition, pname, out);
                                    mentions(then_branch, pname, out);
                                    mentions(else_branch, pname, out);
                                }
                                Expr::InitializerList(es) => {
                                    for y in es {
                                        mentions(y, pname, out);
                                    }
                                }
                                Expr::Lambda { body, .. } => {
                                    for s in body {
                                        mentions_stmt(s, pname, out);
                                    }
                                }
                                _ => {}
                            }
                        }
                        fn mentions_stmt(s: &Stmt, pname: &str, out: &mut bool) {
                            match s {
                                Stmt::Return(Some(e)) | Stmt::Expr(e) => mentions(e, pname, out),
                                Stmt::Return(None) => {}
                                Stmt::VarDecl { init: Some(e), .. }
                                | Stmt::Destructure { init: Some(e), .. } => {
                                    mentions(e, pname, out)
                                }
                                Stmt::VarDecl { init: None, .. }
                                | Stmt::Destructure { init: None, .. } => {}
                                Stmt::Block(ss) => {
                                    for y in ss {
                                        mentions_stmt(y, pname, out);
                                    }
                                }
                                Stmt::If { condition, then_branch, else_branch } => {
                                    mentions(condition, pname, out);
                                    for y in then_branch {
                                        mentions_stmt(y, pname, out);
                                    }
                                    if let Some(els) = else_branch {
                                        for y in els {
                                            mentions_stmt(y, pname, out);
                                        }
                                    }
                                }
                                Stmt::For { init, condition, update, body } => {
                                    mentions_stmt(init, pname, out);
                                    if let Some(c) = condition {
                                        mentions(c, pname, out);
                                    }
                                    if let Some(u) = update {
                                        mentions(u, pname, out);
                                    }
                                    for y in body {
                                        mentions_stmt(y, pname, out);
                                    }
                                }
                                Stmt::While { condition, body } => {
                                    mentions(condition, pname, out);
                                    for y in body {
                                        mentions_stmt(y, pname, out);
                                    }
                                }
                                Stmt::Switch { scrutinee, cases } => {
                                    mentions(scrutinee, pname, out);
                                    for c in cases {
                                        if let Some(l) = &c.label {
                                            mentions(l, pname, out);
                                        }
                                        for y in &c.body {
                                            mentions_stmt(y, pname, out);
                                        }
                                    }
                                }
                            }
                        }
                        mentions(x, pname, &mut found);
                        if found {
                            return false;
                        }
                    }
                    true
                }
                Expr::Lambda { body, .. } => {
                    for s in body {
                        if !stmt_ok(s, pname) {
                            return false;
                        }
                    }
                    true
                }
                Expr::Number(_) | Expr::String(_) | Expr::Bool(_) => true,
            }
        }
        fn stmt_ok(s: &Stmt, pname: &str) -> bool {
            match s {
                Stmt::Return(Some(e)) | Stmt::Expr(e) => expr_ok(e, pname),
                Stmt::Return(None) => true,
                Stmt::VarDecl { init: Some(e), .. }
                | Stmt::Destructure { init: Some(e), .. } => expr_ok(e, pname),
                Stmt::VarDecl { init: None, .. } | Stmt::Destructure { init: None, .. } => true,
                Stmt::Block(ss) => ss.iter().all(|x| stmt_ok(x, pname)),
                Stmt::If { condition, then_branch, else_branch } => {
                    expr_ok(condition, pname)
                        && then_branch.iter().all(|x| stmt_ok(x, pname))
                        && else_branch.iter().flatten().all(|x| stmt_ok(x, pname))
                }
                Stmt::For { init, condition, update, body } => {
                    stmt_ok(init, pname)
                        && condition.as_ref().is_none_or(|c| expr_ok(c, pname))
                        && update.as_ref().is_none_or(|u| expr_ok(u, pname))
                        && body.iter().all(|x| stmt_ok(x, pname))
                }
                Stmt::While { condition, body } => {
                    expr_ok(condition, pname) && body.iter().all(|x| stmt_ok(x, pname))
                }
                Stmt::Switch { scrutinee, cases } => {
                    expr_ok(scrutinee, pname)
                        && cases.iter().all(|c| {
                            c.label.as_ref().is_none_or(|l| expr_ok(l, pname))
                                && c.body.iter().all(|x| stmt_ok(x, pname))
                        })
                }
            }
        }
        ss.iter().all(|s| stmt_ok(s, pname))
    }
    // Calls to `fname` with the argument at position `pidx`.
    fn call_args(
        ss: &[Stmt],
        fname: &str,
        pidx: usize,
        out: &mut Vec<Expr>,
    ) {
        fn expr_calls(e: &Expr, fname: &str, pidx: usize, out: &mut Vec<Expr>) {
            match e {
                Expr::Call { function, arguments } => {
                    if let Expr::Variable(c) = function.as_ref() {
                        if c == fname && !c.contains('<') {
                            if let Some(a) = arguments.get(pidx) {
                                out.push(a.clone());
                            }
                        }
                    }
                    expr_calls(function, fname, pidx, out);
                    for a in arguments {
                        expr_calls(a, fname, pidx, out);
                    }
                }
                Expr::Member { object, .. } => expr_calls(object, fname, pidx, out),
                Expr::Index { object, index } => {
                    expr_calls(object, fname, pidx, out);
                    expr_calls(index, fname, pidx, out);
                }
                Expr::Constructor { arguments, .. } => {
                    for a in arguments {
                        expr_calls(a, fname, pidx, out);
                    }
                }
                Expr::Assign { lhs, rhs } => {
                    expr_calls(lhs, fname, pidx, out);
                    expr_calls(rhs, fname, pidx, out);
                }
                Expr::Unary { expr, .. } => expr_calls(expr, fname, pidx, out),
                Expr::Binary { left, right, .. } => {
                    expr_calls(left, fname, pidx, out);
                    expr_calls(right, fname, pidx, out);
                }
                Expr::Conditional { condition, then_branch, else_branch } => {
                    expr_calls(condition, fname, pidx, out);
                    expr_calls(then_branch, fname, pidx, out);
                    expr_calls(else_branch, fname, pidx, out);
                }
                Expr::InitializerList(elems) => {
                    for x in elems {
                        expr_calls(x, fname, pidx, out);
                    }
                }
                Expr::Lambda { body, .. } => {
                    for s in body {
                        stmt_calls(s, fname, pidx, out);
                    }
                }
                Expr::Variable(_) | Expr::Number(_) | Expr::String(_) | Expr::Bool(_) => {}
            }
        }
        fn stmt_calls(s: &Stmt, fname: &str, pidx: usize, out: &mut Vec<Expr>) {
            match s {
                Stmt::Return(Some(e)) | Stmt::Expr(e) => expr_calls(e, fname, pidx, out),
                Stmt::Return(None) => {}
                Stmt::VarDecl { init: Some(e), .. }
                | Stmt::Destructure { init: Some(e), .. } => expr_calls(e, fname, pidx, out),
                Stmt::VarDecl { init: None, .. } | Stmt::Destructure { init: None, .. } => {}
                Stmt::Block(s2) => {
                    for x in s2 {
                        stmt_calls(x, fname, pidx, out);
                    }
                }
                Stmt::If { condition, then_branch, else_branch } => {
                    expr_calls(condition, fname, pidx, out);
                    for x in then_branch {
                        stmt_calls(x, fname, pidx, out);
                    }
                    if let Some(els) = else_branch {
                        for x in els {
                            stmt_calls(x, fname, pidx, out);
                        }
                    }
                }
                Stmt::For { init, condition, update, body } => {
                    stmt_calls(init, fname, pidx, out);
                    if let Some(c) = condition {
                        expr_calls(c, fname, pidx, out);
                    }
                    if let Some(u) = update {
                        expr_calls(u, fname, pidx, out);
                    }
                    for x in body {
                        stmt_calls(x, fname, pidx, out);
                    }
                }
                Stmt::While { condition, body } => {
                    expr_calls(condition, fname, pidx, out);
                    for x in body {
                        stmt_calls(x, fname, pidx, out);
                    }
                }
                Stmt::Switch { scrutinee, cases } => {
                    expr_calls(scrutinee, fname, pidx, out);
                    for c in cases {
                        if let Some(l) = &c.label {
                            expr_calls(l, fname, pidx, out);
                        }
                        for x in &c.body {
                            stmt_calls(x, fname, pidx, out);
                        }
                    }
                }
            }
        }
        for s in ss {
            stmt_calls(s, fname, pidx, out);
        }
    }

    let mut module = module.clone();
    // Unique free-function definitions only (overloads stay loud).
    let mut def_counts: HashMap<String, usize> = HashMap::new();
    for f in &module.functions {
        if f.scope.is_none() {
            *def_counts.entry(f.name.clone()).or_insert(0) += 1;
        }
    }
    // Caller envs (params + globals/resources + top-level locals) for
    // scalar-arg verification.
    let mut global_tys: HashMap<String, Type> = HashMap::new();
    for g in &module.globals {
        global_tys.insert(g.name.clone(), g.ty.clone());
    }
    for r in &module.resources {
        global_tys.insert(
            r.name.clone(),
            Type::Resource {
                kind: r.kind.clone(),
                access: crate::ast::ResourceAccess::ReadOnly,
                element: Box::new(r.element_ty.clone().unwrap_or(Type::UInt)),
            },
        );
    }
    // Scalar return types of unambiguous single-def free functions (for
    // scalar-valued call args like `lds_offset(…)` → `int`).
    let mut fn_rets: HashMap<String, Type> = HashMap::new();
    for f in &module.functions {
        if f.scope.is_none() && def_counts.get(&f.name).copied().unwrap_or(0) == 1 {
            fn_rets.insert(f.name.clone(), f.return_type.clone());
        }
    }
    // (func, param) narrow targets with replacement scalar types.
    let mut narrow: Vec<(String, String, Type)> = Vec::new();
    for f in &module.functions {
        if f.scope.is_some() {
            continue;
        }
        if def_counts.get(&f.name).copied().unwrap_or(0) != 1 {
            continue;
        }
        if Some(f.name.as_str()) == entry_name {
            continue;
        }
        // Self-calls need no special case: a self-call passing the
        // vector param fails the scalar-arg gate below (decline, safe);
        // passing scalars is consistent with narrowing (sound).
        // Return type must not be a vector (narrowing must not orphan a
        // vector-typed result derived from the param).
        if matches!(&f.return_type, Type::Vector { .. }) {
            continue;
        }
        for (pidx, p) in f.parameters.iter().enumerate() {
            let Type::Vector { base, .. } = &p.ty else {
                continue;
            };
            // Directionless only (no `out`/`inout` rebinding hazards).
            if p.direction.is_some() {
                continue;
            }
            // Collect every call-site argument at this position.
            let mut args: Vec<Expr> = Vec::new();
            for g in &module.functions {
                call_args(&g.body, &f.name, pidx, &mut args);
            }
            if args.is_empty() {
                continue;
            }
            // Every arg must be scalar-typed (literals, scalar vars,
            // scalar constructors) in its caller env.
            let mut all_scalar = true;
            for g in &module.functions {
                let mut cenv = global_tys.clone();
                for pp in &g.parameters {
                    cenv.insert(pp.name.clone(), pp.ty.clone());
                }
                for s in &g.body {
                    if let Stmt::VarDecl { ty, name, .. } = s {
                        cenv.insert(name.clone(), ty.clone());
                    }
                }
                let mut cargs: Vec<Expr> = Vec::new();
                call_args(&g.body, &f.name, pidx, &mut cargs);
                for a in cargs {
                    if !is_scalar_arg(&a, &cenv, &fn_rets) {
                        all_scalar = false;
                        break;
                    }
                }
                if !all_scalar {
                    break;
                }
            }
            if !all_scalar {
                continue;
            }
            // Every use in-body must be scalar-compatible.
            if !uses_scalar_only(&f.body, &p.name) {
                continue;
            }
            narrow.push((f.name.clone(), p.name.clone(), scalar_ty(base)));
        }
    }
    if narrow.is_empty() {
        return module;
    }
    for f in module.functions.iter_mut() {
        for (fname, pname, sty) in &narrow {
            if &f.name == fname {
                for p in f.parameters.iter_mut() {
                    if &p.name == pname {
                        p.ty = sty.clone();
                    }
                }
            }
        }
    }
    module
}

/// Folds inert literal-zero second arguments on append-buffer `Append`
/// (`buf.Append(v, 0)` → `buf.Append(v)`).
///
/// PSSL `AppendRegularBuffer<T>` spells the append with a trailing
/// literal `0` (`outParticleInstances.Append(instanceData, 0)`); HLSL
/// `AppendStructuredBuffer<T>.Append` takes the value alone. Official
/// ISA confirms the `0` carries no content (single `ds_append` of the
/// value via the GDS counter; no second operand). Dropping a provably
/// zero second argument is therefore semantics-preserving, not
/// information loss. Only fires on literal-zero (`0`/`0u`/`0.0`/`false`
/// spellings); nonzero or dynamic second arguments stay loud (their
/// meaning is not established). Method name `Append` with exactly two
/// arguments is unique to this PSSL append pattern (geometry
/// `TriangleStream.Append` takes one), so no type gate is needed.
pub(crate) fn fold_append_zero(module: &Module) -> Module {
    use crate::ast::Stmt;

    fn is_zero(e: &Expr) -> bool {
        match e {
            Expr::Bool(b) => !b,
            Expr::Number(s) => {
                let t = s.trim_end_matches(|c| {
                    c == 'f' || c == 'F' || c == 'u' || c == 'U' || c == 'l' || c == 'L'
                });
                if let Some(hex) = t.strip_prefix("0x").or_else(|| t.strip_prefix("0X")) {
                    let hex = hex.trim_end_matches(|c| c == 'u' || c == 'U' || c == 'l' || c == 'L');
                    u32::from_str_radix(hex, 16).map(|v| v == 0).unwrap_or(false)
                } else if t.contains('.') || t.contains('e') || t.contains('E') {
                    t.parse::<f64>().map(|v| v == 0.0).unwrap_or(false)
                } else {
                    t.parse::<i64>().map(|v| v == 0).unwrap_or(false)
                }
            }
            _ => false,
        }
    }
    fn walk_expr(e: &mut Expr) {
        match e {
            Expr::Call { function, arguments } => {
                walk_expr(function);
                for a in arguments.iter_mut() {
                    walk_expr(a);
                }
                // Post-order: fold `X.Append(v, 0)` → `X.Append(v)`.
                if let Expr::Member { member, .. } = function.as_ref() {
                    if member.eq_ignore_ascii_case("append")
                        && arguments.len() == 2
                        && is_zero(&arguments[1])
                    {
                        arguments.pop();
                    }
                }
            }
            Expr::Member { object, .. } => walk_expr(object),
            Expr::Index { object, index } => {
                walk_expr(object);
                walk_expr(index);
            }
            Expr::Constructor { arguments, .. } => {
                for a in arguments.iter_mut() {
                    walk_expr(a);
                }
            }
            Expr::Assign { lhs, rhs } => {
                walk_expr(lhs);
                walk_expr(rhs);
            }
            Expr::Unary { expr, .. } => walk_expr(expr),
            Expr::Binary { left, right, .. } => {
                walk_expr(left);
                walk_expr(right);
            }
            Expr::Conditional { condition, then_branch, else_branch } => {
                walk_expr(condition);
                walk_expr(then_branch);
                walk_expr(else_branch);
            }
            Expr::InitializerList(elems) => {
                for x in elems.iter_mut() {
                    walk_expr(x);
                }
            }
            Expr::Lambda { body, .. } => {
                for s in body.iter_mut() {
                    walk_stmt(s);
                }
            }
            Expr::Variable(_) | Expr::Number(_) | Expr::String(_) | Expr::Bool(_) => {}
        }
    }
    fn walk_stmt(s: &mut Stmt) {
        match s {
            Stmt::Return(Some(e)) | Stmt::Expr(e) => walk_expr(e),
            Stmt::Return(None) => {}
            Stmt::VarDecl { init: Some(e), .. } | Stmt::Destructure { init: Some(e), .. } => {
                walk_expr(e)
            }
            Stmt::VarDecl { init: None, .. } | Stmt::Destructure { init: None, .. } => {}
            Stmt::Block(ss) => {
                for x in ss.iter_mut() {
                    walk_stmt(x);
                }
            }
            Stmt::If { condition, then_branch, else_branch } => {
                walk_expr(condition);
                for x in then_branch.iter_mut() {
                    walk_stmt(x);
                }
                if let Some(els) = else_branch {
                    for x in els.iter_mut() {
                        walk_stmt(x);
                    }
                }
            }
            Stmt::For { init, condition, update, body } => {
                walk_stmt(init);
                if let Some(c) = condition {
                    walk_expr(c);
                }
                if let Some(u) = update {
                    walk_expr(u);
                }
                for x in body.iter_mut() {
                    walk_stmt(x);
                }
            }
            Stmt::While { condition, body } => {
                walk_expr(condition);
                for x in body.iter_mut() {
                    walk_stmt(x);
                }
            }
            Stmt::Switch { scrutinee, cases } => {
                walk_expr(scrutinee);
                for c in cases.iter_mut() {
                    if let Some(l) = &mut c.label {
                        walk_expr(l);
                    }
                    for x in c.body.iter_mut() {
                        walk_stmt(x);
                    }
                }
            }
        }
    }

    let mut module = module.clone();
    for f in module.functions.iter_mut() {
        for s in f.body.iter_mut() {
            walk_stmt(s);
        }
    }
    module
}

/// Folds constant-condition `if`s (`if (true/false)`, `if (0>/</== literal)`).
///
/// A statically-known branch never executes the other side, so replacing
/// the `if` with the taken block (or nothing) preserves runtime semantics
/// exactly — including eliminating compile-time-invalid references in the
/// untaken branch (the C++ `if constexpr` model: the discarded branch is
/// not instantiated). Only fires on provably-literal conditions (`Bool`
/// literals; numeric/bool comparisons and `&&`/`||`/`!` over literals);
/// anything with variables, calls, or side effects keeps today's path.
/// Fixpoint (folding reveals nested constants). Required downstream of
/// template/`static-const` resolution, whose literals feed these tests.
pub(crate) fn fold_const_if(module: &Module) -> Module {
    use crate::ast::{BinaryOp, Stmt, UnaryOp};

    // Literal boolean value of an expression, if provably constant.
    fn const_bool(e: &Expr) -> Option<bool> {
        match e {
            Expr::Bool(b) => Some(*b),
            Expr::Number(s) => {
                // `0`/`0.0` is false; any other numeric literal is true
                // (C++ scalar condition semantics). Hex handled.
                let t = s.trim_end_matches(|c| {
                    c == 'f' || c == 'F' || c == 'u' || c == 'U' || c == 'l' || c == 'L'
                });
                if let Some(hex) = t.strip_prefix("0x").or_else(|| t.strip_prefix("0X")) {
                    let hex = hex.trim_end_matches(|c| {
                        c == 'u' || c == 'U' || c == 'l' || c == 'L'
                    });
                    u32::from_str_radix(hex, 16).ok().map(|v| v != 0)
                } else if t.contains('.') || t.contains('e') || t.contains('E') {
                    t.parse::<f64>().ok().map(|v| v != 0.0)
                } else {
                    t.parse::<i64>().ok().map(|v| v != 0)
                }
            }
            Expr::Unary { op, expr } => match op {
                UnaryOp::Not => const_bool(expr).map(|b| !b),
                _ => None,
            },
            Expr::Binary { left, op, right } => {
                // Boolean logic over literal bools.
                if matches!(op, BinaryOp::LogicalAnd | BinaryOp::LogicalOr) {
                    let (a, b) = (const_bool(left)?, const_bool(right)?);
                    return Some(match op {
                        BinaryOp::LogicalAnd => a && b,
                        _ => a || b,
                    });
                }
                // Comparisons over literal numbers/bools.
                let (a, b) = (const_num(left)?, const_num(right)?);
                Some(match op {
                    BinaryOp::Equal => a == b,
                    BinaryOp::NotEqual => a != b,
                    BinaryOp::Less => a < b,
                    BinaryOp::LessEqual => a <= b,
                    BinaryOp::Greater => a > b,
                    BinaryOp::GreaterEqual => a >= b,
                    _ => return None,
                })
            }
            _ => None,
        }
    }
    // Literal numeric value (f64) for comparisons, if provably constant.
    fn const_num(e: &Expr) -> Option<f64> {
        match e {
            Expr::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
            Expr::Number(s) => {
                let t = s.trim_end_matches(|c| {
                    c == 'f' || c == 'F' || c == 'u' || c == 'U' || c == 'l' || c == 'L'
                });
                if let Some(hex) = t.strip_prefix("0x").or_else(|| t.strip_prefix("0X")) {
                    let hex = hex.trim_end_matches(|c| {
                        c == 'u' || c == 'U' || c == 'l' || c == 'L'
                    });
                    u32::from_str_radix(hex, 16).ok().map(|v| v as f64)
                } else {
                    t.parse::<f64>().ok()
                }
            }
            _ => None,
        }
    }
    // Rewrite one statement list; true if anything changed.
    fn fold_block(ss: &mut Vec<Stmt>) -> bool {
        let mut changed = false;
        let mut out: Vec<Stmt> = Vec::with_capacity(ss.len());
        for mut s in std::mem::take(ss) {
            // Recurse first (post-order; nested folds first).
            match &mut s {
                Stmt::Block(inner) => {
                    changed |= fold_block(inner);
                }
                Stmt::If { then_branch, else_branch, .. } => {
                    changed |= fold_block(then_branch);
                    if let Some(els) = else_branch {
                        changed |= fold_block(els);
                    }
                }
                Stmt::For { init, body, .. } => {
                    changed |= fold_stmt(init);
                    changed |= fold_block(body);
                }
                Stmt::While { body, .. } => {
                    changed |= fold_block(body);
                }
                Stmt::Switch { cases, .. } => {
                    for c in cases.iter_mut() {
                        changed |= fold_block(&mut c.body);
                    }
                }
                _ => {}
            }
            // Then fold this `if` if its condition is now literal.
            if let Stmt::If { condition, then_branch, else_branch } = s {
                if let Some(taken) = const_bool(&condition) {
                    changed = true;
                    if taken {
                        out.extend(then_branch);
                    } else if let Some(els) = else_branch {
                        out.extend(els);
                    }
                    // `if (false)` without else vanishes (no-op).
                    continue;
                }
                out.push(Stmt::If { condition, then_branch, else_branch });
            } else {
                out.push(s);
            }
        }
        *ss = out;
        changed
    }
    fn fold_stmt(s: &mut Stmt) -> bool {
        match s {
            Stmt::Block(ss) => fold_block(ss),
            Stmt::If { then_branch, else_branch, .. } => {
                let mut c = fold_block(then_branch);
                if let Some(els) = else_branch {
                    c |= fold_block(els);
                }
                c
            }
            Stmt::For { init, body, .. } => fold_stmt(init) | fold_block(body),
            Stmt::While { body, .. } => fold_block(body),
            Stmt::Switch { cases, .. } => {
                let mut c = false;
                for cs in cases.iter_mut() {
                    c |= fold_block(&mut cs.body);
                }
                c
            }
            _ => false,
        }
    }

    let mut module = module.clone();
    for _ in 0..8 {
        let mut changed = false;
        for f in module.functions.iter_mut() {
            changed |= fold_block(&mut f.body);
        }
        if !changed {
            break;
        }
    }
    module
}

/// Resolves scoped static constants in expressions (`SkinVertex::
/// kDummyBoneIndex` → `65535`).
///
/// `S::m` qualification is only legal on static members, so a recorded
/// literal (`struct_static_consts`, flushed to `Module::constants`) IS the
/// static's value; substituting it preserves semantics exactly. Only
/// fires when `S` names a declared struct and `S::m` has a recorded
/// literal; types, namespace paths (`sce::...`), enum values, and
/// unrecorded shapes keep today's loud path. Runs before lowering so all
/// downstream (comparisons, array sizes, attributes) sees literals.
pub(crate) fn resolve_scoped_consts(module: &Module) -> Module {
    use crate::ast::Stmt;

    fn rewrite_expr(
        e: &mut Expr,
        structs: &std::collections::HashSet<String>,
        consts: &HashMap<String, Expr>,
    ) {
        // Post-order: children first.
        match e {
            Expr::Call { function, arguments } => {
                rewrite_expr(function, structs, consts);
                for a in arguments.iter_mut() {
                    rewrite_expr(a, structs, consts);
                }
            }
            Expr::Member { object, .. } => rewrite_expr(object, structs, consts),
            Expr::Index { object, index } => {
                rewrite_expr(object, structs, consts);
                rewrite_expr(index, structs, consts);
            }
            Expr::Constructor { arguments, .. } => {
                for a in arguments.iter_mut() {
                    rewrite_expr(a, structs, consts);
                }
            }
            Expr::Assign { lhs, rhs } => {
                rewrite_expr(lhs, structs, consts);
                rewrite_expr(rhs, structs, consts);
            }
            Expr::Unary { expr, .. } => rewrite_expr(expr, structs, consts),
            Expr::Binary { left, right, .. } => {
                rewrite_expr(left, structs, consts);
                rewrite_expr(right, structs, consts);
            }
            Expr::Conditional { condition, then_branch, else_branch } => {
                rewrite_expr(condition, structs, consts);
                rewrite_expr(then_branch, structs, consts);
                rewrite_expr(else_branch, structs, consts);
            }
            Expr::InitializerList(elems) => {
                for x in elems.iter_mut() {
                    rewrite_expr(x, structs, consts);
                }
            }
            Expr::Lambda { body, .. } => {
                for s in body.iter_mut() {
                    rewrite_stmt(s, structs, consts);
                }
            }
            Expr::Variable(_) | Expr::Number(_) | Expr::String(_) | Expr::Bool(_) => {}
        }
        // Then this expression: `S::m` value position.
        if let Expr::Member { object, member } = e {
            if let Expr::Variable(s) = object.as_ref() {
                if structs.contains(s) {
                    if let Some(v) = consts.get(&format!("{s}::{member}")) {
                        *e = v.clone();
                        return;
                    }
                }
            }
        }
        if let Expr::Variable(n) = e {
            if n.contains("::") {
                // Exact qualified first (struct statics), then
                // namespace-erased bare trailing (namespaces strip at
                // parse; same collision assumption as the parser).
                if let Some(v) = consts.get(n) {
                    *e = v.clone();
                } else if let Some(bare) = n.rsplit("::").next() {
                    if let Some(v) = consts.get(bare) {
                        *e = v.clone();
                    }
                }
            }
        }
    }
    fn rewrite_stmt(
        s: &mut Stmt,
        structs: &std::collections::HashSet<String>,
        consts: &HashMap<String, Expr>,
    ) {
        match s {
            Stmt::Return(Some(e)) | Stmt::Expr(e) => rewrite_expr(e, structs, consts),
            Stmt::Return(None) => {}
            Stmt::VarDecl { init: Some(e), .. } | Stmt::Destructure { init: Some(e), .. } => {
                rewrite_expr(e, structs, consts)
            }
            Stmt::VarDecl { init: None, .. } | Stmt::Destructure { init: None, .. } => {}
            Stmt::Block(ss) => {
                for x in ss.iter_mut() {
                    rewrite_stmt(x, structs, consts);
                }
            }
            Stmt::If { condition, then_branch, else_branch } => {
                rewrite_expr(condition, structs, consts);
                for x in then_branch.iter_mut() {
                    rewrite_stmt(x, structs, consts);
                }
                if let Some(els) = else_branch {
                    for x in els.iter_mut() {
                        rewrite_stmt(x, structs, consts);
                    }
                }
            }
            Stmt::For { init, condition, update, body } => {
                rewrite_stmt(init, structs, consts);
                if let Some(c) = condition {
                    rewrite_expr(c, structs, consts);
                }
                if let Some(u) = update {
                    rewrite_expr(u, structs, consts);
                }
                for x in body.iter_mut() {
                    rewrite_stmt(x, structs, consts);
                }
            }
            Stmt::While { condition, body } => {
                rewrite_expr(condition, structs, consts);
                for x in body.iter_mut() {
                    rewrite_stmt(x, structs, consts);
                }
            }
            Stmt::Switch { scrutinee, cases } => {
                rewrite_expr(scrutinee, structs, consts);
                for c in cases.iter_mut() {
                    if let Some(l) = &mut c.label {
                        rewrite_expr(l, structs, consts);
                    }
                    for x in c.body.iter_mut() {
                        rewrite_stmt(x, structs, consts);
                    }
                }
            }
        }
    }

    let mut module = module.clone();
    if module.constants.is_empty() {
        return module;
    }
    let structs: std::collections::HashSet<String> =
        module.structs.iter().map(|s| s.name.clone()).collect();
    // Resolves qualified `S::m` (exact) and namespace-erased `a::b::k`
    // (bare trailing). Runs whenever constants exist; rewrites only on
    // match, so unaffected shaders pass through unchanged.
    for f in module.functions.iter_mut() {
        for s in f.body.iter_mut() {
            rewrite_stmt(s, &structs, &module.constants);
        }
    }
    module
}

/// Forces vector-element typed-buffer targets of atomics to structured
/// buffers.
///
/// HLSL/DXC rejects `Interlocked*` on a `RWBuffer<Vector>` element
/// component (`buf[i].x` on `RWBuffer<uint3>`: "texture buffer whose
/// result type is non-scalar"), while `RWStructuredBuffer<Vector>`
/// supports component-select atomics identically (verified via DXC).
/// PSSL `RW_DataBuffer<uint3>` + `AtomicAdd(hist[i].x, 1)` is therefore
/// retargeted: any `(RW_)DataBuffer` with a vector element that feeds an
/// atomic destination becomes `(RW_)StructuredBuffer`, preserving access
/// and element type. Scalar-element buffers are untouched (their
/// `RWBuffer` atomics are legal); non-vector elements and non-atomic
/// uses keep today's path. Only fires on proven `Atomic*`/
/// `Interlocked*` (free or method) uses; struct-field and top-level
/// resource declarations rewrite together so flattening and emission
/// agree.
pub(crate) fn atomic_structured_buffers(module: &Module) -> Module {
    use crate::ast::{ResourceAccess, Stmt};

    fn is_atomic_fn(function: &Expr) -> bool {
        match function {
            Expr::Variable(n) => {
                matches!(
                    n.to_ascii_lowercase().as_str(),
                    "atomicadd" | "atomicor" | "atomicmax" | "atomicmin" | "atomicand"
                        | "atomicxor" | "interlockedadd" | "interlockedor"
                        | "interlockedmax" | "interlockedmin" | "interlockedand"
                        | "interlockedxor"
                )
            }
            Expr::Member { member, .. } => {
                let l = member.to_ascii_lowercase();
                l == "atomicadd"
                    || l == "atomicor"
                    || l == "atomicmax"
                    || l == "atomicmin"
                    || l == "atomicand"
                    || l == "atomicxor"
                    || l.starts_with("interlocked")
            }
            _ => false,
        }
    }
    // Base resource path of an atomic destination: strip Index/Member
    // selects to the underlying variable + struct-field chain.
    // `srt.m_histogram[i].x` -> ("srt", ["m_histogram"]).
    fn dest_base(e: &Expr) -> Option<(String, Vec<String>)> {
        let mut fields = Vec::new();
        let mut cur = e;
        loop {
            match cur {
                Expr::Index { object, .. } => cur = object,
                Expr::Member { object, member } => {
                    fields.push(member.clone());
                    cur = object;
                }
                Expr::Variable(n) => {
                    fields.reverse();
                    return Some((n.clone(), fields));
                }
                _ => return None,
            }
        }
    }
    fn collect_expr(e: &Expr, out: &mut Vec<(String, Vec<String>)>) {
        match e {
            Expr::Call { function, arguments } => {
                if is_atomic_fn(function) && !arguments.is_empty() {
                    if let Some(base) = dest_base(&arguments[0]) {
                        out.push(base);
                    }
                }
                collect_expr(function, out);
                for a in arguments {
                    collect_expr(a, out);
                }
            }
            Expr::Member { object, .. } => collect_expr(object, out),
            Expr::Index { object, index } => {
                collect_expr(object, out);
                collect_expr(index, out);
            }
            Expr::Constructor { arguments, .. } => {
                for a in arguments {
                    collect_expr(a, out);
                }
            }
            Expr::Assign { lhs, rhs } => {
                collect_expr(lhs, out);
                collect_expr(rhs, out);
            }
            Expr::Unary { expr, .. } => collect_expr(expr, out),
            Expr::Binary { left, right, .. } => {
                collect_expr(left, out);
                collect_expr(right, out);
            }
            Expr::Conditional { condition, then_branch, else_branch } => {
                collect_expr(condition, out);
                collect_expr(then_branch, out);
                collect_expr(else_branch, out);
            }
            Expr::InitializerList(elems) => {
                for x in elems {
                    collect_expr(x, out);
                }
            }
            Expr::Lambda { body, .. } => {
                for s in body {
                    collect_stmt(s, out);
                }
            }
            Expr::Variable(_) | Expr::Number(_) | Expr::String(_) | Expr::Bool(_) => {}
        }
    }
    fn collect_stmt(s: &Stmt, out: &mut Vec<(String, Vec<String>)>) {
        match s {
            Stmt::Return(Some(e)) | Stmt::Expr(e) => collect_expr(e, out),
            Stmt::Return(None) => {}
            Stmt::VarDecl { init: Some(e), .. } | Stmt::Destructure { init: Some(e), .. } => {
                collect_expr(e, out)
            }
            Stmt::VarDecl { init: None, .. } | Stmt::Destructure { init: None, .. } => {}
            Stmt::Block(ss) => {
                for x in ss {
                    collect_stmt(x, out);
                }
            }
            Stmt::If { condition, then_branch, else_branch } => {
                collect_expr(condition, out);
                for x in then_branch {
                    collect_stmt(x, out);
                }
                if let Some(els) = else_branch {
                    for x in els {
                        collect_stmt(x, out);
                    }
                }
            }
            Stmt::For { init, condition, update, body } => {
                collect_stmt(init, out);
                if let Some(c) = condition {
                    collect_expr(c, out);
                }
                if let Some(u) = update {
                    collect_expr(u, out);
                }
                for x in body {
                    collect_stmt(x, out);
                }
            }
            Stmt::While { condition, body } => {
                collect_expr(condition, out);
                for x in body {
                    collect_stmt(x, out);
                }
            }
            Stmt::Switch { scrutinee, cases } => {
                collect_expr(scrutinee, out);
                for c in cases {
                    if let Some(l) = &c.label {
                        collect_expr(l, out);
                    }
                    for x in &c.body {
                        collect_stmt(x, out);
                    }
                }
            }
        }
    }
    fn structured_kind(kind: &str) -> Option<(&'static str, ResourceAccess)> {
        match kind {
            "databuffer" => Some(("structuredbuffer", ResourceAccess::ReadOnly)),
            "rwdatabuffer" | "rwbuffer" => {
                Some(("rwstructuredbuffer", ResourceAccess::ReadWrite))
            }
            _ => None,
        }
    }

    let mut module = module.clone();
    // struct name -> fields (for resolving `srt.m_histogram` to a decl).
    let mut struct_fields: HashMap<String, HashMap<String, Type>> = HashMap::new();
    for s in &module.structs {
        struct_fields.insert(
            s.name.clone(),
            s.fields.iter().map(|f| (f.name.clone(), f.ty.clone())).collect(),
        );
    }
    // Top-level resource decls by name.
    let mut resource_tys: HashMap<String, Type> = HashMap::new();
    for g in &module.globals {
        resource_tys.insert(g.name.clone(), g.ty.clone());
    }
    for r in &module.resources {
        resource_tys.insert(
            r.name.clone(),
            Type::Resource {
                kind: r.kind.clone(),
                access: crate::ast::ResourceAccess::ReadOnly,
                element: Box::new(r.element_ty.clone().unwrap_or(Type::UInt)),
            },
        );
    }
    // (struct, field) targets + top-level resource targets to retarget.
    let mut field_targets: HashSet<(String, String)> = HashSet::new();
    let mut res_targets: HashSet<String> = HashSet::new();
    // (func, param) targets retargeted together with linked call-site
    // args (interprocedural phase below).
    let mut param_targets: HashSet<(String, String)> = HashSet::new();
    for f in &module.functions {
        let mut env: HashMap<String, Type> = HashMap::new();
        for p in &f.parameters {
            env.insert(p.name.clone(), p.ty.clone());
        }
        for (n, t) in &resource_tys {
            env.entry(n.clone()).or_insert(t.clone());
        }
        // Top-level locals (nested-block locals missed conservatively).
        for s in &f.body {
            if let Stmt::VarDecl { ty, name, .. } = s {
                env.insert(name.clone(), ty.clone());
            }
        }
        let mut dests = Vec::new();
        for s in &f.body {
            collect_stmt(s, &mut dests);
        }
        for (root, fields) in dests {
            let Some(first) = fields.first() else {
                // Bare resource variable destination.
                if let Some(Type::Resource { kind, element, .. }) = env.get(&root) {
                    if matches!(&**element, Type::Vector { .. })
                        && structured_kind(kind).is_some()
                    {
                        res_targets.insert(root);
                    }
                }
                continue;
            };
            // `root.field` where root is struct-typed: resolve the field.
            let root_ty = env.get(&root).cloned().unwrap_or(Type::Named(root.clone()));
            if let Type::Named(sname) = &root_ty {
                if let Some(ft) = struct_fields.get(sname).and_then(|m| m.get(first)) {
                    if let Type::Resource { kind, element, .. } = ft {
                        if matches!(&**element, Type::Vector { .. })
                            && structured_kind(kind).is_some()
                        {
                            field_targets.insert((sname.clone(), first.clone()));
                        }
                    }
                }
            }
        }
    }
    // Interprocedural: atomic destinations that are bare PARAMS of
    // databuffer-vector type (`accumulateDelta(RWBuffer<int3> dst, …)`
    // with `InterlockedAdd(dst[i].x, …)`). Retarget the param together
    // with every linked call-site argument (direct resource/global or
    // struct-field only), so signatures stay consistent. Declines (stays
    // loud) on chained params, complex args, any `.Load(` byte-method
    // use (Buffer-only; StructuredBuffer uses `[]`), or mismatched
    // call sites. Scalar params and non-atomic uses are untouched.
    {
        // (func, param) atomic-targeted params.
        let mut param_roots: HashSet<(String, String)> = HashSet::new();
        for f in &module.functions {
            let mut env: HashMap<String, Type> = HashMap::new();
            for p in &f.parameters {
                env.insert(p.name.clone(), p.ty.clone());
            }
            let mut dests = Vec::new();
            for s in &f.body {
                collect_stmt(s, &mut dests);
            }
            for (root, fields) in dests {
                // Only bare-param roots (fields handled above). `fields`
                // here are index/component selects, not struct fields
                // (params aren't structs); require the param itself to be
                // the databuffer-vector resource.
                if let Some(Type::Resource { kind, element, .. }) = env.get(&root) {
                    // Confirm `root` is actually a param (not a global /
                    // local that merely shares env).
                    if f.parameters.iter().any(|p| &p.name == &root)
                        && matches!(&**element, Type::Vector { .. })
                        && structured_kind(kind).is_some()
                    {
                        param_roots.insert((f.name.clone(), root));
                    }
                }
            }
        }
        // Buffer-only method uses (`.Load(` byte loads) veto propagation
        // for that resource family (StructuredBuffer has no `.Load`).
        fn uses_byte_load(
            module: &Module,
            names: &HashSet<String>,
        ) -> bool {
            fn expr_uses(e: &Expr, names: &HashSet<String>) -> bool {
                match e {
                    Expr::Call { function, arguments } => {
                        if let Expr::Member { object, member } = function.as_ref() {
                            if member.eq_ignore_ascii_case("load") {
                                // Object root variable name.
                                let mut cur = object.as_ref();
                                loop {
                                    match cur {
                                        Expr::Variable(n) => {
                                            if names.contains(n) {
                                                return true;
                                            }
                                            break;
                                        }
                                        Expr::Member { object: o, .. } => cur = o,
                                        Expr::Index { object: o, .. } => cur = o,
                                        _ => break,
                                    }
                                }
                            }
                        }
                        expr_uses(function, names) || arguments.iter().any(|a| expr_uses(a, names))
                    }
                    Expr::Member { object, .. } => expr_uses(object, names),
                    Expr::Index { object, index } => {
                        expr_uses(object, names) || expr_uses(index, names)
                    }
                    Expr::Constructor { arguments, .. } => {
                        arguments.iter().any(|a| expr_uses(a, names))
                    }
                    Expr::Assign { lhs, rhs } => expr_uses(lhs, names) || expr_uses(rhs, names),
                    Expr::Unary { expr, .. } => expr_uses(expr, names),
                    Expr::Binary { left, right, .. } => {
                        expr_uses(left, names) || expr_uses(right, names)
                    }
                    Expr::Conditional { condition, then_branch, else_branch } => {
                        expr_uses(condition, names)
                            || expr_uses(then_branch, names)
                            || expr_uses(else_branch, names)
                    }
                    Expr::InitializerList(elems) => {
                        elems.iter().any(|a| expr_uses(a, names))
                    }
                    Expr::Lambda { body, .. } => body.iter().any(|s| stmt_uses(s, names)),
                    Expr::Variable(_) | Expr::Number(_) | Expr::String(_) | Expr::Bool(_) => false,
                }
            }
            fn stmt_uses(s: &Stmt, names: &HashSet<String>) -> bool {
                match s {
                    Stmt::Return(Some(e)) | Stmt::Expr(e) => expr_uses(e, names),
                    Stmt::Return(None) => false,
                    Stmt::VarDecl { init: Some(e), .. }
                    | Stmt::Destructure { init: Some(e), .. } => expr_uses(e, names),
                    Stmt::VarDecl { init: None, .. } | Stmt::Destructure { init: None, .. } => false,
                    Stmt::Block(ss) => ss.iter().any(|x| stmt_uses(x, names)),
                    Stmt::If { condition, then_branch, else_branch } => {
                        expr_uses(condition, names)
                            || then_branch.iter().any(|x| stmt_uses(x, names))
                            || else_branch.iter().flatten().any(|x| stmt_uses(x, names))
                    }
                    Stmt::For { init, condition, update, body } => {
                        stmt_uses(init, names)
                            || condition.as_ref().is_some_and(|c| expr_uses(c, names))
                            || update.as_ref().is_some_and(|u| expr_uses(u, names))
                            || body.iter().any(|x| stmt_uses(x, names))
                    }
                    Stmt::While { condition, body } => {
                        expr_uses(condition, names)
                            || body.iter().any(|x| stmt_uses(x, names))
                    }
                    Stmt::Switch { scrutinee, cases } => {
                        expr_uses(scrutinee, names)
                            || cases.iter().any(|c| {
                                c.label.as_ref().is_some_and(|l| expr_uses(l, names))
                                    || c.body.iter().any(|x| stmt_uses(x, names))
                            })
                    }
                }
            }
            module.functions.iter().any(|f| f.body.iter().any(|s| stmt_uses(s, names)))
        }
        // Resolve each param via its call sites.
        // (param_targets declared alongside field/res targets above.)
        for (fname, pname) in param_roots {
            let Some(fdef) = module.functions.iter().find(|f| &f.name == &fname) else {
                continue;
            };
            let Some(pidx) = fdef.parameters.iter().position(|p| &p.name == &pname) else {
                continue;
            };
            // Gather linked arg targets across ALL call sites; decline on
            // any non-clean arg (non-variable/member, chained param,
            // mismatched type, or Buffer-only `.Load` use).
            let mut linked_fields: HashSet<(String, String)> = HashSet::new();
            let mut linked_res: HashSet<String> = HashSet::new();
            let mut ok = true;
            // Names that must avoid `.Load(` (param + all linked args).
            let mut family: HashSet<String> = HashSet::new();
            family.insert(pname.clone());
            for g in &module.functions {
                // Caller env (params + top-level locals + globals/resources).
                let mut cenv: HashMap<String, Type> = HashMap::new();
                for p in &g.parameters {
                    cenv.insert(p.name.clone(), p.ty.clone());
                }
                for (n, t) in &resource_tys {
                    cenv.entry(n.clone()).or_insert(t.clone());
                }
                for s in &g.body {
                    if let Stmt::VarDecl { ty, name, .. } = s {
                        cenv.insert(name.clone(), ty.clone());
                    }
                }
                // Find calls `fname(...)` and inspect arg at pidx.
                fn find_calls(
                    s: &Stmt,
                    fname: &str,
                    pidx: usize,
                    cenv: &HashMap<String, Type>,
                    struct_fields: &HashMap<String, HashMap<String, Type>>,
                    linked_fields: &mut HashSet<(String, String)>,
                    linked_res: &mut HashSet<String>,
                    family: &mut HashSet<String>,
                    ok: &mut bool,
                ) {
                    fn expr_calls(
                        e: &Expr,
                        fname: &str,
                        pidx: usize,
                        cenv: &HashMap<String, Type>,
                        struct_fields: &HashMap<String, HashMap<String, Type>>,
                        linked_fields: &mut HashSet<(String, String)>,
                        linked_res: &mut HashSet<String>,
                        family: &mut HashSet<String>,
                        ok: &mut bool,
                    ) {
                        match e {
                            Expr::Call { function, arguments } => {
                                if let Expr::Variable(c) = function.as_ref() {
                                    if c == fname {
                                        if let Some(arg) = arguments.get(pidx) {
                                            // Arg must be a bare resource/global
                                            // or a struct-field member path
                                            // (single level; deeper stays loud).
                                            match arg {
                                                Expr::Variable(v) => {
                                                    match cenv.get(v) {
                                                        Some(Type::Resource { kind, element, .. })
                                                            if matches!(&**element, Type::Vector { .. })
                                                                && structured_kind(kind).is_some() =>
                                                        {
                                                            linked_res.insert(v.clone());
                                                            family.insert(v.clone());
                                                        }
                                                        _ => {
                                                            *ok = false;
                                                        }
                                                    }
                                                }
                                                Expr::Member { object, member }
                                                    if matches!(object.as_ref(), Expr::Variable(_)) =>
                                                {
                                                    let Expr::Variable(root) =
                                                        object.as_ref()
                                                    else {
                                                        *ok = false;
                                                        return;
                                                    };
                                                    let rt = cenv
                                                        .get(root)
                                                        .cloned()
                                                        .unwrap_or(Type::Named(root.clone()));
                                                    if let Type::Named(sn) = &rt {
                                                        match struct_fields
                                                            .get(sn)
                                                            .and_then(|m| m.get(member))
                                                        {
                                                            Some(Type::Resource { kind, element, .. })
                                                                if matches!(&**element, Type::Vector { .. })
                                                                    && structured_kind(kind).is_some() =>
                                                            {
                                                                linked_fields.insert((
                                                                    sn.clone(),
                                                                    member.clone(),
                                                                ));
                                                                family.insert(format!(
                                                                    "{root}.{member}"
                                                                ));
                                                            }
                                                            _ => {
                                                                *ok = false;
                                                            }
                                                        }
                                                    } else {
                                                        *ok = false;
                                                    }
                                                }
                                                _ => {
                                                    *ok = false;
                                                }
                                            }
                                        }
                                    }
                                }
                                expr_calls(function, fname, pidx, cenv, struct_fields, linked_fields, linked_res, family, ok);
                                for a in arguments {
                                    expr_calls(a, fname, pidx, cenv, struct_fields, linked_fields, linked_res, family, ok);
                                }
                            }
                            Expr::Member { object, .. } => {
                                expr_calls(object, fname, pidx, cenv, struct_fields, linked_fields, linked_res, family, ok)
                            }
                            Expr::Index { object, index } => {
                                expr_calls(object, fname, pidx, cenv, struct_fields, linked_fields, linked_res, family, ok);
                                expr_calls(index, fname, pidx, cenv, struct_fields, linked_fields, linked_res, family, ok);
                            }
                            Expr::Constructor { arguments, .. } => {
                                for a in arguments {
                                    expr_calls(a, fname, pidx, cenv, struct_fields, linked_fields, linked_res, family, ok);
                                }
                            }
                            Expr::Assign { lhs, rhs } => {
                                expr_calls(lhs, fname, pidx, cenv, struct_fields, linked_fields, linked_res, family, ok);
                                expr_calls(rhs, fname, pidx, cenv, struct_fields, linked_fields, linked_res, family, ok);
                            }
                            Expr::Unary { expr, .. } => {
                                expr_calls(expr, fname, pidx, cenv, struct_fields, linked_fields, linked_res, family, ok)
                            }
                            Expr::Binary { left, right, .. } => {
                                expr_calls(left, fname, pidx, cenv, struct_fields, linked_fields, linked_res, family, ok);
                                expr_calls(right, fname, pidx, cenv, struct_fields, linked_fields, linked_res, family, ok);
                            }
                            Expr::Conditional { condition, then_branch, else_branch } => {
                                expr_calls(condition, fname, pidx, cenv, struct_fields, linked_fields, linked_res, family, ok);
                                expr_calls(then_branch, fname, pidx, cenv, struct_fields, linked_fields, linked_res, family, ok);
                                expr_calls(else_branch, fname, pidx, cenv, struct_fields, linked_fields, linked_res, family, ok);
                            }
                            Expr::InitializerList(elems) => {
                                for x in elems {
                                    expr_calls(x, fname, pidx, cenv, struct_fields, linked_fields, linked_res, family, ok);
                                }
                            }
                            Expr::Lambda { body, .. } => {
                                for st in body {
                                    find_calls(st, fname, pidx, cenv, struct_fields, linked_fields, linked_res, family, ok);
                                }
                            }
                            Expr::Variable(_)
                            | Expr::Number(_)
                            | Expr::String(_)
                            | Expr::Bool(_) => {}
                        }
                    }
                    match s {
                        Stmt::Return(Some(e)) | Stmt::Expr(e) => {
                            expr_calls(e, fname, pidx, cenv, struct_fields, linked_fields, linked_res, family, ok)
                        }
                        Stmt::Return(None) => {}
                        Stmt::VarDecl { init: Some(e), .. }
                        | Stmt::Destructure { init: Some(e), .. } => {
                            expr_calls(e, fname, pidx, cenv, struct_fields, linked_fields, linked_res, family, ok)
                        }
                        Stmt::VarDecl { init: None, .. }
                        | Stmt::Destructure { init: None, .. } => {}
                        Stmt::Block(ss) => {
                            for x in ss {
                                find_calls(x, fname, pidx, cenv, struct_fields, linked_fields, linked_res, family, ok);
                            }
                        }
                        Stmt::If { condition, then_branch, else_branch } => {
                            expr_calls(condition, fname, pidx, cenv, struct_fields, linked_fields, linked_res, family, ok);
                            for x in then_branch {
                                find_calls(x, fname, pidx, cenv, struct_fields, linked_fields, linked_res, family, ok);
                            }
                            if let Some(els) = else_branch {
                                for x in els {
                                    find_calls(x, fname, pidx, cenv, struct_fields, linked_fields, linked_res, family, ok);
                                }
                            }
                        }
                        Stmt::For { init, condition, update, body } => {
                            find_calls(init, fname, pidx, cenv, struct_fields, linked_fields, linked_res, family, ok);
                            if let Some(c) = condition {
                                expr_calls(c, fname, pidx, cenv, struct_fields, linked_fields, linked_res, family, ok);
                            }
                            if let Some(u) = update {
                                expr_calls(u, fname, pidx, cenv, struct_fields, linked_fields, linked_res, family, ok);
                            }
                            for x in body {
                                find_calls(x, fname, pidx, cenv, struct_fields, linked_fields, linked_res, family, ok);
                            }
                        }
                        Stmt::While { condition, body } => {
                            expr_calls(condition, fname, pidx, cenv, struct_fields, linked_fields, linked_res, family, ok);
                            for x in body {
                                find_calls(x, fname, pidx, cenv, struct_fields, linked_fields, linked_res, family, ok);
                            }
                        }
                        Stmt::Switch { scrutinee, cases } => {
                            expr_calls(scrutinee, fname, pidx, cenv, struct_fields, linked_fields, linked_res, family, ok);
                            for c in cases {
                                if let Some(l) = &c.label {
                                    expr_calls(l, fname, pidx, cenv, struct_fields, linked_fields, linked_res, family, ok);
                                }
                                for x in &c.body {
                                    find_calls(x, fname, pidx, cenv, struct_fields, linked_fields, linked_res, family, ok);
                                }
                            }
                        }
                    }
                }
                for s in &g.body {
                    find_calls(s, &fname, pidx, &cenv, &struct_fields, &mut linked_fields, &mut linked_res, &mut family, &mut ok);
                }
            }
            // Require at least one linked target (else the param is
            // atomic-targeted but never fed a retargetable buffer;
            // keep loud) and no `.Load(` on the family.
            if !ok || (linked_fields.is_empty() && linked_res.is_empty()) {
                continue;
            }
            if uses_byte_load(&module, &family) {
                continue;
            }
            // Commit: param + all linked args retarget together.
            param_targets.insert((fname, pname));
            field_targets.extend(linked_fields);
            res_targets.extend(linked_res);
        }
    }
    if field_targets.is_empty() && res_targets.is_empty() && param_targets.is_empty() {
        return module;
    }
    for s in module.structs.iter_mut() {
        for fd in s.fields.iter_mut() {
            if field_targets.contains(&(s.name.clone(), fd.name.clone())) {
                if let Type::Resource { kind, element, .. } = &fd.ty {
                    if let Some((new_kind, access)) = structured_kind(kind) {
                        fd.ty = Type::Resource {
                            kind: new_kind.to_string(),
                            access,
                            element: element.clone(),
                        };
                    }
                }
            }
        }
    }
    for r in module.resources.iter_mut() {
        if res_targets.contains(&r.name) {
            if let Some((new_kind, _)) = structured_kind(&r.kind) {
                r.kind = new_kind.to_string();
            }
        }
    }
    // Globals holding resources (rare) retarget by type as well.
    for g in module.globals.iter_mut() {
        if res_targets.contains(&g.name) {
            if let Type::Resource { kind, element, .. } = &g.ty {
                if let Some((new_kind, access)) = structured_kind(kind) {
                    g.ty = Type::Resource {
                        kind: new_kind.to_string(),
                        access,
                        element: element.clone(),
                    };
                }
            }
        }
    }
    // Linked params retarget together with their call-site args (keeps
    // signatures consistent; all args were verified clean above).
    for f in module.functions.iter_mut() {
        for p in f.parameters.iter_mut() {
            if param_targets.contains(&(f.name.clone(), p.name.clone())) {
                if let Type::Resource { kind, element, .. } = &p.ty {
                    if let Some((new_kind, access)) = structured_kind(kind) {
                        p.ty = Type::Resource {
                            kind: new_kind.to_string(),
                            access,
                            element: element.clone(),
                        };
                    }
                }
            }
        }
    }
    module
}

/// Monomorphizes free-function templates via spelling-based `T::`
/// resolution (`phong(V2P<T>)` + `T::kUvCount`, called with
/// `V2P<PackVertex<false,false,0>>`).
///
/// Angle lists erase in `Type`, so implicit deduction cannot see through
/// `V2P<...>` to bind `T`. But declaration spellings survive
/// (`template_spellings`: callee patterns like `V2P<T>`, caller concretes
/// like `V2P<PackVertex<false,false,0>>`). This pass matches them
/// syntactically (parsed angle-trees, exact structural correspondence),
/// binds typenames to concrete template-ids, evaluates `T::kX` from the
/// struct's static initializers under bound value-params (+ recorded
/// defaults for omitted trailing args), substitutes literals, clones per
/// call, retargets, and drops fully-retargeted generics. `fold_const_if`
/// (later) then eliminates the dead `if constexpr` branches.
///
/// Soundness gates (ANY failure declines that call/def, staying loud —
/// never guessing, never wrong):
/// - free functions only (no methods), single definition (no overloads),
///   non-recursive, non-entry;
/// - `T` appears ONLY in `T::` value positions in the body (never in AST
///   param/return types, never in param defaults — those stay with `mono`);
/// - call args forwarding caller params/locals with recorded spellings
///   only (no complex expressions);
/// - patterns bind every typename exactly once, unambiguously
///   (`Base<T>` vs `Base<Concrete>`; non-bare patterns verify exactly);
/// - bound concretes parse as `Struct<literal-args...>` (literals only);
/// - value-params bind positionally + trailing recorded defaults only
///   (missing without default, extras, or non-literals decline);
/// - static inits fold under strict literal/param/pure-op evaluation
///   (anything else declines); results must be `Bool`/`Number` literals.
pub(crate) fn monomorphize_qualified(module: &Module, entry_name: Option<&str>) -> Module {
    use crate::ast::{BinaryOp, Stmt, UnaryOp};

    // Parsed template-id: base + top-level args (opaque strings, may nest).
    fn parse_tid(s: &str) -> Option<(String, Vec<String>)> {
        let s = s.trim();
        let lt = s.find('<')?;
        // Base must be a plain identifier (no nesting/qualification).
        let base = s[..lt].trim();
        if base.is_empty()
            || !base.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == ':')
            || !s.ends_with('>')
        {
            return None;
        }
        let inner = &s[lt + 1..s.len() - 1];
        let mut args = Vec::new();
        let mut depth = 0usize;
        let mut paren = 0usize;
        let mut start = 0usize;
        let chars: Vec<char> = inner.chars().collect();
        let mut i = 0;
        while i <= chars.len() {
            let is_end = i == chars.len();
            if !is_end {
                match chars[i] {
                    '<' => depth += 1,
                    '>' => depth = depth.saturating_sub(1),
                    '(' => paren += 1,
                    ')' => paren = paren.saturating_sub(1),
                    ',' if depth == 0 && paren == 0 => {
                        args.push(inner[start..i].trim().to_string());
                        start = i + 1;
                    }
                    _ => {}
                }
            } else if start < inner.len() || !args.is_empty() {
                // Trailing arg (also handles single-arg with no comma).
                let last = inner[start..].trim().to_string();
                if !last.is_empty() || !args.is_empty() {
                    args.push(last);
                }
            }
            i += 1;
        }
        // Balanced brackets required (depth must return to 0).
        if depth != 0 {
            return None;
        }
        // Reject empty args (`Foo<>`) and empties from `,,`.
        if args.iter().any(|a| a.is_empty()) {
            return None;
        }
        Some((base.to_string(), args))
    }
    fn is_bare_ident(s: &str) -> bool {
        !s.is_empty()
            && s.chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == ':')
            && !s.contains('<')
            && !s.contains('>')
    }
    // Match pattern-args vs concrete-args, binding bare typenames.
    // Returns bindings or None (mismatch/ambiguity/non-bare pattern).
    fn match_args(
        pat_args: &[String],
        conc_args: &[String],
        typenames: &std::collections::HashSet<String>,
    ) -> Option<HashMap<String, String>> {
        if pat_args.len() != conc_args.len() {
            return None;
        }
        let mut binds: HashMap<String, String> = HashMap::new();
        for (p, c) in pat_args.iter().zip(conc_args.iter()) {
            if typenames.contains(p) && is_bare_ident(p) {
                // Bare typename position: bind (must agree on repeat).
                if let Some(prev) = binds.get(p) {
                    if prev != c {
                        return None;
                    }
                } else {
                    binds.insert(p.clone(), c.clone());
                }
            } else if p == c {
                // Exact concrete correspondence (no binding).
                continue;
            } else {
                // Complex pattern (`T*`, `Pack<T>`, ...) or mismatch:
                // decline (strict).
                return None;
            }
        }
        Some(binds)
    }
    // Literal expr from a token string (`true`/`false`/numbers only).
    fn literal_expr(s: &str) -> Option<Expr> {
        let t = s.trim();
        if t.eq_ignore_ascii_case("true") {
            return Some(Expr::Bool(true));
        }
        if t.eq_ignore_ascii_case("false") {
            return Some(Expr::Bool(false));
        }
        // Number (decimal/hex, mirroring const-folding lexical rules).
        let num_ok = if let Some(hex) = t
            .strip_prefix("0x")
            .or_else(|| t.strip_prefix("0X"))
        {
            let hex = hex.trim_end_matches(|c| c == 'u' || c == 'U' || c == 'l' || c == 'L');
            u32::from_str_radix(hex, 16).is_ok() && !hex.is_empty()
        } else {
            let d = t.trim_end_matches(|c| {
                c == 'f' || c == 'F' || c == 'u' || c == 'U' || c == 'l' || c == 'L'
            });
            !d.is_empty()
                && (d.parse::<i64>().is_ok()
                    || d.parse::<f64>().is_ok() && {
                        // Floats must actually look floaty OR be plain
                        // ints (int literals also parse as f64; accept).
                        true
                    })
        };
        if num_ok {
            return Some(Expr::Number(t.to_string()));
        }
        None
    }
    // Strict constexpr evaluation: params → literals, pure ops fold.
    // Returns a literal (Bool/Number) or None (decline).
    fn eval_const(e: &Expr, bindings: &HashMap<String, Expr>) -> Option<Expr> {
        match e {
            Expr::Bool(_) | Expr::Number(_) => Some(e.clone()),
            Expr::Variable(n) => {
                // Bound value-param → its literal (must itself be literal).
                match bindings.get(n) {
                    Some(Expr::Bool(_) | Expr::Number(_)) => bindings.get(n).cloned(),
                    // Unbound or non-literal: decline (no guessing).
                    _ => None,
                }
            }
            Expr::Unary { op, expr } => {
                let v = eval_const(expr, bindings)?;
                match (op, &v) {
                    (UnaryOp::Not, Expr::Bool(b)) => Some(Expr::Bool(!b)),
                    (UnaryOp::Minus, Expr::Number(s)) => {
                        // Negate numeric literal textually when safe.
                        if let Ok(i) = s.parse::<i64>() {
                            Some(Expr::Number((-i).to_string()))
                        } else if let Ok(f) = s.parse::<f64>() {
                            Some(Expr::Number((-f).to_string()))
                        } else {
                            None
                        }
                    }
                    _ => None,
                }
            }
            Expr::Binary { left, op, right } => {
                let (a, b) = (eval_const(left, bindings)?, eval_const(right, bindings)?);
                // Boolean logic.
                match (&a, op, &b) {
                    (Expr::Bool(x), BinaryOp::LogicalAnd, Expr::Bool(y)) => {
                        return Some(Expr::Bool(*x && *y));
                    }
                    (Expr::Bool(x), BinaryOp::LogicalOr, Expr::Bool(y)) => {
                        return Some(Expr::Bool(*x || *y));
                    }
                    _ => {}
                }
                // Numeric/bool comparisons and arithmetic via f64/i64.
                // (Mirrors `fold_const_if` shapes; strict.)
                fn num(e: &Expr) -> Option<f64> {
                    match e {
                        Expr::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
                        Expr::Number(s) => {
                            let t = s.trim_end_matches(|c| {
                                c == 'f'
                                    || c == 'F'
                                    || c == 'u'
                                    || c == 'U'
                                    || c == 'l'
                                    || c == 'L'
                            });
                            if let Some(hex) =
                                t.strip_prefix("0x").or_else(|| t.strip_prefix("0X"))
                            {
                                let hex = hex.trim_end_matches(|c| {
                                    c == 'u' || c == 'U' || c == 'l' || c == 'L'
                                });
                                u32::from_str_radix(hex, 16).ok().map(|v| v as f64)
                            } else {
                                t.parse::<f64>().ok()
                            }
                        }
                        _ => None,
                    }
                }
                let (x, y) = (num(&a)?, num(&b)?);
                match op {
                    BinaryOp::Equal => Some(Expr::Bool(x == y)),
                    BinaryOp::NotEqual => Some(Expr::Bool(x != y)),
                    BinaryOp::Less => Some(Expr::Bool(x < y)),
                    BinaryOp::LessEqual => Some(Expr::Bool(x <= y)),
                    BinaryOp::Greater => Some(Expr::Bool(x > y)),
                    BinaryOp::GreaterEqual => Some(Expr::Bool(x >= y)),
                    BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul => {
                        // Integer-preserving when both integral.
                        let v = match op {
                            BinaryOp::Add => x + y,
                            BinaryOp::Sub => x - y,
                            _ => x * y,
                        };
                        if v.fract() == 0.0 && v.abs() < 9e15 {
                            Some(Expr::Number((v as i64).to_string()))
                        } else {
                            Some(Expr::Number(v.to_string()))
                        }
                    }
                    _ => None,
                }
            }
            // Constructors/calls/members/index/assign/conditional/lambda:
            // not constexpr-foldable here (strict decline).
            _ => None,
        }
    }
    // `T::member` uses in an expression (value positions).
    fn qualified_uses(e: &Expr, typenames: &std::collections::HashSet<String>, out: &mut Vec<(String, String)>) {
        match e {
            Expr::Member { object, member } => {
                if let Expr::Variable(t) = object.as_ref() {
                    if typenames.contains(t) {
                        out.push((t.clone(), member.clone()));
                    }
                }
                qualified_uses(object, typenames, out);
            }
            Expr::Call { function, arguments } => {
                // `T::member` spelled as a single variable (`T::kX`).
                if let Expr::Variable(n) = function.as_ref() {
                    if let Some((t, m)) = n.split_once("::") {
                        if typenames.contains(t) && !m.contains(':') {
                            out.push((t.to_string(), m.to_string()));
                        }
                    }
                }
                qualified_uses(function, typenames, out);
                for a in arguments {
                    qualified_uses(a, typenames, out);
                }
            }
            Expr::Variable(n) => {
                if let Some((t, m)) = n.split_once("::") {
                    if typenames.contains(t) && !m.contains(':') {
                        out.push((t.to_string(), m.to_string()));
                    }
                }
            }
            Expr::Index { object, index } => {
                qualified_uses(object, typenames, out);
                qualified_uses(index, typenames, out);
            }
            Expr::Constructor { arguments, .. } => {
                for a in arguments {
                    qualified_uses(a, typenames, out);
                }
            }
            Expr::Assign { lhs, rhs } => {
                qualified_uses(lhs, typenames, out);
                qualified_uses(rhs, typenames, out);
            }
            Expr::Unary { expr, .. } => qualified_uses(expr, typenames, out),
            Expr::Binary { left, right, .. } => {
                qualified_uses(left, typenames, out);
                qualified_uses(right, typenames, out);
            }
            Expr::Conditional { condition, then_branch, else_branch } => {
                qualified_uses(condition, typenames, out);
                qualified_uses(then_branch, typenames, out);
                qualified_uses(else_branch, typenames, out);
            }
            Expr::InitializerList(elems) => {
                for x in elems {
                    qualified_uses(x, typenames, out);
                }
            }
            Expr::Lambda { body, .. } => {
                for s in body {
                    qualified_uses_stmt(s, typenames, out);
                }
            }
            Expr::Number(_) | Expr::String(_) | Expr::Bool(_) => {}
        }
    }
    fn qualified_uses_stmt(s: &Stmt, typenames: &std::collections::HashSet<String>, out: &mut Vec<(String, String)>) {
        match s {
            Stmt::Return(Some(e)) | Stmt::Expr(e) => qualified_uses(e, typenames, out),
            Stmt::Return(None) => {}
            Stmt::VarDecl { init: Some(e), .. } | Stmt::Destructure { init: Some(e), .. } => {
                qualified_uses(e, typenames, out)
            }
            Stmt::VarDecl { init: None, .. } | Stmt::Destructure { init: None, .. } => {}
            Stmt::Block(ss) => {
                for x in ss {
                    qualified_uses_stmt(x, typenames, out);
                }
            }
            Stmt::If { condition, then_branch, else_branch } => {
                qualified_uses(condition, typenames, out);
                for x in then_branch {
                    qualified_uses_stmt(x, typenames, out);
                }
                if let Some(els) = else_branch {
                    for x in els {
                        qualified_uses_stmt(x, typenames, out);
                    }
                }
            }
            Stmt::For { init, condition, update, body } => {
                qualified_uses_stmt(init, typenames, out);
                if let Some(c) = condition {
                    qualified_uses(c, typenames, out);
                }
                if let Some(u) = update {
                    qualified_uses(u, typenames, out);
                }
                for x in body {
                    qualified_uses_stmt(x, typenames, out);
                }
            }
            Stmt::While { condition, body } => {
                qualified_uses(condition, typenames, out);
                for x in body {
                    qualified_uses_stmt(x, typenames, out);
                }
            }
            Stmt::Switch { scrutinee, cases } => {
                qualified_uses(scrutinee, typenames, out);
                for c in cases {
                    if let Some(l) = &c.label {
                        qualified_uses(l, typenames, out);
                    }
                    for x in &c.body {
                        qualified_uses_stmt(x, typenames, out);
                    }
                }
            }
        }
    }
    // Substitute resolved `T::member` → literal in an expression.
    fn subst_qualified(
        e: &mut Expr,
        resolved: &HashMap<(String, String), Expr>,
    ) -> bool {
        let mut changed = false;
        match e {
            Expr::Call { function, arguments } => {
                changed |= subst_qualified(function, resolved);
                for a in arguments.iter_mut() {
                    changed |= subst_qualified(a, resolved);
                }
            }
            Expr::Member { object, .. } => {
                changed |= subst_qualified(object, resolved);
            }
            Expr::Index { object, index } => {
                changed |= subst_qualified(object, resolved);
                changed |= subst_qualified(index, resolved);
            }
            Expr::Constructor { arguments, .. } => {
                for a in arguments.iter_mut() {
                    changed |= subst_qualified(a, resolved);
                }
            }
            Expr::Assign { lhs, rhs } => {
                changed |= subst_qualified(lhs, resolved);
                changed |= subst_qualified(rhs, resolved);
            }
            Expr::Unary { expr, .. } => {
                changed |= subst_qualified(expr, resolved);
            }
            Expr::Binary { left, right, .. } => {
                changed |= subst_qualified(left, resolved);
                changed |= subst_qualified(right, resolved);
            }
            Expr::Conditional { condition, then_branch, else_branch } => {
                changed |= subst_qualified(condition, resolved);
                changed |= subst_qualified(then_branch, resolved);
                changed |= subst_qualified(else_branch, resolved);
            }
            Expr::InitializerList(elems) => {
                for x in elems.iter_mut() {
                    changed |= subst_qualified(x, resolved);
                }
            }
            Expr::Lambda { body, .. } => {
                for s in body.iter_mut() {
                    changed |= subst_qualified_stmt(s, resolved);
                }
            }
            Expr::Variable(_) | Expr::Number(_) | Expr::String(_) | Expr::Bool(_) => {}
        }
        // This node itself: `T::m` (member or single-variable spelling).
        if let Expr::Member { object, member } = &*e {
            if let Expr::Variable(t) = object.as_ref() {
                if let Some(v) = resolved.get(&(t.clone(), member.clone())) {
                    *e = v.clone();
                    return true;
                }
            }
        }
        if let Expr::Variable(n) = &*e {
            if let Some((t, m)) = n.split_once("::") {
                if let Some(v) = resolved.get(&(t.to_string(), m.to_string())) {
                    *e = v.clone();
                    return true;
                }
            }
        }
        changed
    }
    fn subst_qualified_stmt(s: &mut Stmt, resolved: &HashMap<(String, String), Expr>) -> bool {
        let mut changed = false;
        match s {
            Stmt::Return(Some(e)) | Stmt::Expr(e) => {
                changed |= subst_qualified(e, resolved)
            }
            Stmt::Return(None) => {}
            Stmt::VarDecl { init: Some(e), .. } | Stmt::Destructure { init: Some(e), .. } => {
                changed |= subst_qualified(e, resolved)
            }
            Stmt::VarDecl { init: None, .. } | Stmt::Destructure { init: None, .. } => {}
            Stmt::Block(ss) => {
                for x in ss.iter_mut() {
                    changed |= subst_qualified_stmt(x, resolved);
                }
            }
            Stmt::If { condition, then_branch, else_branch } => {
                changed |= subst_qualified(condition, resolved);
                for x in then_branch.iter_mut() {
                    changed |= subst_qualified_stmt(x, resolved);
                }
                if let Some(els) = else_branch {
                    for x in els.iter_mut() {
                        changed |= subst_qualified_stmt(x, resolved);
                    }
                }
            }
            Stmt::For { init, condition, update, body } => {
                changed |= subst_qualified_stmt(init, resolved);
                if let Some(c) = condition {
                    changed |= subst_qualified(c, resolved);
                }
                if let Some(u) = update {
                    changed |= subst_qualified(u, resolved);
                }
                for x in body.iter_mut() {
                    changed |= subst_qualified_stmt(x, resolved);
                }
            }
            Stmt::While { condition, body } => {
                changed |= subst_qualified(condition, resolved);
                for x in body.iter_mut() {
                    changed |= subst_qualified_stmt(x, resolved);
                }
            }
            Stmt::Switch { scrutinee, cases } => {
                changed |= subst_qualified(scrutinee, resolved);
                for c in cases.iter_mut() {
                    if let Some(l) = &mut c.label {
                        changed |= subst_qualified(l, resolved);
                    }
                    for x in c.body.iter_mut() {
                        changed |= subst_qualified_stmt(x, resolved);
                    }
                }
            }
        }
        changed
    }
    // Callee names referenced anywhere in a body (for drop + recursion).
    fn callee_names_stmt(s: &Stmt, out: &mut std::collections::HashSet<String>) {
        match s {
            Stmt::Return(Some(e)) | Stmt::Expr(e) => callee_names_expr(e, out),
            Stmt::Return(None) => {}
            Stmt::VarDecl { init: Some(e), .. } | Stmt::Destructure { init: Some(e), .. } => {
                callee_names_expr(e, out)
            }
            Stmt::VarDecl { init: None, .. } | Stmt::Destructure { init: None, .. } => {}
            Stmt::Block(ss) => {
                for x in ss {
                    callee_names_stmt(x, out);
                }
            }
            Stmt::If { condition, then_branch, else_branch } => {
                callee_names_expr(condition, out);
                for x in then_branch {
                    callee_names_stmt(x, out);
                }
                if let Some(els) = else_branch {
                    for x in els {
                        callee_names_stmt(x, out);
                    }
                }
            }
            Stmt::For { init, condition, update, body } => {
                callee_names_stmt(init, out);
                if let Some(c) = condition {
                    callee_names_expr(c, out);
                }
                if let Some(u) = update {
                    callee_names_expr(u, out);
                }
                for x in body {
                    callee_names_stmt(x, out);
                }
            }
            Stmt::While { condition, body } => {
                callee_names_expr(condition, out);
                for x in body {
                    callee_names_stmt(x, out);
                }
            }
            Stmt::Switch { scrutinee, cases } => {
                callee_names_expr(scrutinee, out);
                for c in cases {
                    if let Some(l) = &c.label {
                        callee_names_expr(l, out);
                    }
                    for x in &c.body {
                        callee_names_stmt(x, out);
                    }
                }
            }
        }
    }
    fn callee_names_expr(e: &Expr, out: &mut std::collections::HashSet<String>) {
        match e {
            Expr::Call { function, arguments } => {
                if let Expr::Variable(n) = function.as_ref() {
                    // Strip explicit args (`f<T>`) to the base name.
                    let base = n.split('<').next().unwrap_or(n).to_string();
                    out.insert(base);
                }
                callee_names_expr(function, out);
                for a in arguments {
                    callee_names_expr(a, out);
                }
            }
            Expr::Member { object, .. } => callee_names_expr(object, out),
            Expr::Index { object, index } => {
                callee_names_expr(object, out);
                callee_names_expr(index, out);
            }
            Expr::Constructor { arguments, .. } => {
                for a in arguments {
                    callee_names_expr(a, out);
                }
            }
            Expr::Assign { lhs, rhs } => {
                callee_names_expr(lhs, out);
                callee_names_expr(rhs, out);
            }
            Expr::Unary { expr, .. } => callee_names_expr(expr, out),
            Expr::Binary { left, right, .. } => {
                callee_names_expr(left, out);
                callee_names_expr(right, out);
            }
            Expr::Conditional { condition, then_branch, else_branch } => {
                callee_names_expr(condition, out);
                callee_names_expr(then_branch, out);
                callee_names_expr(else_branch, out);
            }
            Expr::InitializerList(elems) => {
                for x in elems {
                    callee_names_expr(x, out);
                }
            }
            Expr::Lambda { body, .. } => {
                for s in body {
                    callee_names_stmt(s, out);
                }
            }
            Expr::Variable(_) | Expr::Number(_) | Expr::String(_) | Expr::Bool(_) => {}
        }
    }

    let mut module = module.clone();
    // Candidate templates: free, single-def, non-entry, non-recursive
    // functions with typenames used ONLY as `T::` in body exprs (never
    // in AST param/return types — those stay with `mono`).
    struct Cand {
        idx: usize,
        typenames: std::collections::HashSet<String>,
    }
    let mut cands: HashMap<String, Cand> = HashMap::new();
    {
        let mut def_counts: HashMap<String, usize> = HashMap::new();
        for f in &module.functions {
            if f.scope.is_none() {
                *def_counts.entry(f.name.clone()).or_insert(0) += 1;
            }
        }
        for (idx, f) in module.functions.iter().enumerate() {
            if f.scope.is_some() {
                continue;
            }
            if def_counts.get(&f.name).copied().unwrap_or(0) != 1 {
                continue;
            }
            let typenames: std::collections::HashSet<String> = f
                .template_params
                .iter()
                .filter(|(_, v)| !v)
                .map(|(n, _)| n.clone())
                .collect();
            if typenames.is_empty() {
                continue;
            }
            // T must not appear in AST param/return types (else `mono`
            // owns it; we handle pure `T::`-value templates only).
            fn ty_mentions(ty: &Type, names: &std::collections::HashSet<String>) -> bool {
                match ty {
                    Type::Named(n) => names.contains(n),
                    Type::Resource { element, .. } => ty_mentions(element, names),
                    _ => false,
                }
            }
            if f.parameters.iter().any(|p| ty_mentions(&p.ty, &typenames))
                || ty_mentions(&f.return_type, &typenames)
            {
                continue;
            }
            // Must use `T::` in the body (else nothing to resolve).
            let mut uses = Vec::new();
            for s in &f.body {
                qualified_uses_stmt(s, &typenames, &mut uses);
            }
            if uses.is_empty() {
                continue;
            }
            // Non-recursive (self-calls would need fixpoint binding).
            let mut callees = std::collections::HashSet::new();
            for s in &f.body {
                callee_names_stmt(s, &mut callees);
            }
            if callees.contains(&f.name) {
                continue;
            }
            // `T::` only in body exprs (decline if in param defaults —
            // those aren't walked above, so separately gate defaults).
            if f.parameters.iter().any(|p| p.default.is_some()) {
                // Defaults could hide `T::`; conservatively decline when
                // any default exists alongside qualified uses. (phong has
                // `bool` defaults without `T::`; refine only if needed.)
                let mut def_uses = Vec::new();
                for p in &f.parameters {
                    if let Some(d) = &p.default {
                        qualified_uses(d, &typenames, &mut def_uses);
                    }
                }
                if !def_uses.is_empty() {
                    continue;
                }
            }
            cands.insert(f.name.clone(), Cand { idx, typenames });
        }
    }
    if cands.is_empty() {
        return module;
    }
    // Struct template value-params + defaults for constexpr binding.
    let mut struct_tparams: HashMap<String, Vec<(String, bool)>> = HashMap::new();
    for s in &module.structs {
        if !s.template_params.is_empty() {
            struct_tparams.insert(s.name.clone(), s.template_params.clone());
        }
    }
    // Per-caller param/local spellings for concrete arg lookup.
    // (template_spellings: fn -> param -> spelling;
    //  template_site_spellings: scope -> local/field -> spelling.)
    let mut new_functions: Vec<Function> = Vec::new();
    let mut clones: HashMap<(String, String), String> = HashMap::new();
    let taken: std::collections::HashSet<String> =
        module.functions.iter().map(|f| f.name.clone()).collect();
    let mut retargeted: std::collections::HashSet<String> = std::collections::HashSet::new();
    // Walk callers with owned bodies (no borrow conflict with the
    // immutable module lookups below). All functions, including
    // candidates calling others; never retarget calls inside the callee
    // itself (recursion declined at candidacy; pre-clone bodies only).
    let caller_names: Vec<String> = module.functions.iter().map(|f| f.name.clone()).collect();
    for caller in &caller_names {
        let cidx = module.functions.iter().position(|f| &f.name == caller).unwrap();
        let mut body = std::mem::take(&mut module.functions[cidx].body);
        for s in body.iter_mut() {
            walk_call_stmt(s, caller, &module, &cands, &struct_tparams, &mut clones, &mut new_functions, &taken, &mut retargeted);
        }
        module.functions[cidx].body = body;
        // Recursive expression walker retargeting candidate calls.
        fn walk_call_expr(
            e: &mut Expr,
            caller: &str,
            module: &Module,
            cands: &HashMap<String, Cand>,
            struct_tparams: &HashMap<String, Vec<(String, bool)>>,
            clones: &mut HashMap<(String, String), String>,
            new_functions: &mut Vec<Function>,
            taken: &std::collections::HashSet<String>,
            retargeted: &mut std::collections::HashSet<String>,
        ) -> bool {
            let mut progress = false;
            // Post-order.
            match e {
                Expr::Call { function, arguments } => {
                    walk_call_expr(function, caller, module, cands, struct_tparams, clones, new_functions, taken, retargeted);
                    for a in arguments.iter_mut() {
                        progress |= walk_call_expr(a, caller, module, cands, struct_tparams, clones, new_functions, taken, retargeted);
                    }
                }
                Expr::Member { object, .. } => {
                    progress |= walk_call_expr(object, caller, module, cands, struct_tparams, clones, new_functions, taken, retargeted);
                }
                Expr::Index { object, index } => {
                    progress |= walk_call_expr(object, caller, module, cands, struct_tparams, clones, new_functions, taken, retargeted);
                    progress |= walk_call_expr(index, caller, module, cands, struct_tparams, clones, new_functions, taken, retargeted);
                }
                Expr::Constructor { arguments, .. } => {
                    for a in arguments.iter_mut() {
                        progress |= walk_call_expr(a, caller, module, cands, struct_tparams, clones, new_functions, taken, retargeted);
                    }
                }
                Expr::Assign { lhs, rhs } => {
                    progress |= walk_call_expr(lhs, caller, module, cands, struct_tparams, clones, new_functions, taken, retargeted);
                    progress |= walk_call_expr(rhs, caller, module, cands, struct_tparams, clones, new_functions, taken, retargeted);
                }
                Expr::Unary { expr, .. } => {
                    progress |= walk_call_expr(expr, caller, module, cands, struct_tparams, clones, new_functions, taken, retargeted);
                }
                Expr::Binary { left, right, .. } => {
                    progress |= walk_call_expr(left, caller, module, cands, struct_tparams, clones, new_functions, taken, retargeted);
                    progress |= walk_call_expr(right, caller, module, cands, struct_tparams, clones, new_functions, taken, retargeted);
                }
                Expr::Conditional { condition, then_branch, else_branch } => {
                    progress |= walk_call_expr(condition, caller, module, cands, struct_tparams, clones, new_functions, taken, retargeted);
                    progress |= walk_call_expr(then_branch, caller, module, cands, struct_tparams, clones, new_functions, taken, retargeted);
                    progress |= walk_call_expr(else_branch, caller, module, cands, struct_tparams, clones, new_functions, taken, retargeted);
                }
                Expr::InitializerList(elems) => {
                    for x in elems.iter_mut() {
                        progress |= walk_call_expr(x, caller, module, cands, struct_tparams, clones, new_functions, taken, retargeted);
                    }
                }
                Expr::Lambda { body, .. } => {
                    for s in body.iter_mut() {
                        progress |= walk_call_stmt(s, caller, module, cands, struct_tparams, clones, new_functions, taken, retargeted);
                    }
                }
                Expr::Variable(_) | Expr::Number(_) | Expr::String(_) | Expr::Bool(_) => {}
            }
            // This call: bare `F(args)` to a candidate (never inside F
            // itself — callers are pre-clone bodies; F's own body is
            // walked only when F is someone's caller for OTHER callees,
            // and self-calls were declined at candidacy).
            if let Expr::Call { function, arguments } = e {
                if let Expr::Variable(callee) = function.as_ref() {
                    // Bare name only (no explicit `<>`; those stay loud).
                    if !callee.contains('<') && !callee.contains('>') {
                        if let Some(cand) = cands.get(callee) {
                            if caller != callee {
                                if let Some(cname) = try_retarget(
                                    callee,
                                    cand,
                                    arguments,
                                    caller,
                                    module,
                                    struct_tparams,
                                    clones,
                                    new_functions,
                                    taken,
                                    retargeted,
                                ) {
                                    // Retarget to the resolved clone.
                                    *function = Box::new(Expr::Variable(cname));
                                    progress = true;
                                }
                            }
                        }
                    }
                }
            }
            progress
        }
        fn walk_call_stmt(
            s: &mut Stmt,
            caller: &str,
            module: &Module,
            cands: &HashMap<String, Cand>,
            struct_tparams: &HashMap<String, Vec<(String, bool)>>,
            clones: &mut HashMap<(String, String), String>,
            new_functions: &mut Vec<Function>,
            taken: &std::collections::HashSet<String>,
            retargeted: &mut std::collections::HashSet<String>,
        ) -> bool {
            let mut progress = false;
            match s {
                Stmt::Return(Some(e)) | Stmt::Expr(e) => {
                    progress |= walk_call_expr(e, caller, module, cands, struct_tparams, clones, new_functions, taken, retargeted)
                }
                Stmt::Return(None) => {}
                Stmt::VarDecl { init: Some(e), .. } | Stmt::Destructure { init: Some(e), .. } => {
                    progress |= walk_call_expr(e, caller, module, cands, struct_tparams, clones, new_functions, taken, retargeted)
                }
                Stmt::VarDecl { init: None, .. } | Stmt::Destructure { init: None, .. } => {}
                Stmt::Block(ss) => {
                    for x in ss.iter_mut() {
                        progress |= walk_call_stmt(x, caller, module, cands, struct_tparams, clones, new_functions, taken, retargeted);
                    }
                }
                Stmt::If { condition, then_branch, else_branch } => {
                    progress |= walk_call_expr(condition, caller, module, cands, struct_tparams, clones, new_functions, taken, retargeted);
                    for x in then_branch.iter_mut() {
                        progress |= walk_call_stmt(x, caller, module, cands, struct_tparams, clones, new_functions, taken, retargeted);
                    }
                    if let Some(els) = else_branch {
                        for x in els.iter_mut() {
                            progress |= walk_call_stmt(x, caller, module, cands, struct_tparams, clones, new_functions, taken, retargeted);
                        }
                    }
                }
                Stmt::For { init, condition, update, body } => {
                    progress |= walk_call_stmt(init, caller, module, cands, struct_tparams, clones, new_functions, taken, retargeted);
                    if let Some(c) = condition {
                        progress |= walk_call_expr(c, caller, module, cands, struct_tparams, clones, new_functions, taken, retargeted);
                    }
                    if let Some(u) = update {
                        progress |= walk_call_expr(u, caller, module, cands, struct_tparams, clones, new_functions, taken, retargeted);
                    }
                    for x in body.iter_mut() {
                        progress |= walk_call_stmt(x, caller, module, cands, struct_tparams, clones, new_functions, taken, retargeted);
                    }
                }
                Stmt::While { condition, body } => {
                    progress |= walk_call_expr(condition, caller, module, cands, struct_tparams, clones, new_functions, taken, retargeted);
                    for x in body.iter_mut() {
                        progress |= walk_call_stmt(x, caller, module, cands, struct_tparams, clones, new_functions, taken, retargeted);
                    }
                }
                Stmt::Switch { scrutinee, cases } => {
                    progress |= walk_call_expr(scrutinee, caller, module, cands, struct_tparams, clones, new_functions, taken, retargeted);
                    for c in cases.iter_mut() {
                        if let Some(l) = &mut c.label {
                            progress |= walk_call_expr(l, caller, module, cands, struct_tparams, clones, new_functions, taken, retargeted);
                        }
                        for x in c.body.iter_mut() {
                            progress |= walk_call_stmt(x, caller, module, cands, struct_tparams, clones, new_functions, taken, retargeted);
                        }
                    }
                }
            }
            progress
        }
        #[allow(clippy::too_many_arguments)]
        fn try_retarget(
            callee: &str,
            cand: &Cand,
            arguments: &[Expr],
            caller: &str,
            module: &Module,
            struct_tparams: &HashMap<String, Vec<(String, bool)>>,
            clones: &mut HashMap<(String, String), String>,
            new_functions: &mut Vec<Function>,
            taken: &std::collections::HashSet<String>,
            retargeted: &mut std::collections::HashSet<String>,
        ) -> Option<String> {
            let def = module.functions.get(cand.idx)?.clone();
            // Callee patterns (param -> spelling like `V2P<T>`).
            let pats = module.template_spellings.get(callee)?;
            // Caller concretes (param/local -> spelling).
            let caller_pats = module.template_spellings.get(caller);
            let caller_sites = module.template_site_spellings.get(caller);
            // Bind typenames via pattern/concrete matching.
            let mut binds: HashMap<String, String> = HashMap::new();
            for (i, arg) in arguments.iter().enumerate() {
                let Some(param) = def.parameters.get(i) else {
                    // Arity mismatch (extra args): decline.
                    return None;
                };
                let Some(pat) = pats.get(&param.name) else {
                    // No pattern for this param (non-template): skip.
                    continue;
                };
                // Concrete spelling only for forwarded params/locals.
                let conc = match arg {
                    Expr::Variable(v) => caller_pats
                        .and_then(|m| m.get(v))
                        .or_else(|| caller_sites.and_then(|m| m.get(v)))
                        .cloned(),
                    _ => None,
                };
                let Some(conc) = conc else {
                    continue;
                };
                // Parse both; match structurally.
                let (pbase, pargs) = parse_tid(pat)?;
                let (cbase, cargs) = parse_tid(&conc)?;
                if pbase != cbase || pargs.len() != cargs.len() {
                    return None;
                }
                let m = match_args(&pargs, &cargs, &cand.typenames)?;
                for (t, v) in m {
                    if let Some(prev) = binds.get(&t) {
                        if prev != &v {
                            return None;
                        }
                    } else {
                        binds.insert(t, v);
                    }
                }
            }
            // Every typename must bind (else some `T::` unresolvable).
            if cand.typenames.iter().any(|t| !binds.contains_key(t)) {
                return None;
            }
            // Resolve every distinct `T::member` in the body.
            let mut uses: Vec<(String, String)> = Vec::new();
            for s in &def.body {
                qualified_uses_stmt(s, &cand.typenames, &mut uses);
            }
            uses.sort();
            uses.dedup();
            if uses.is_empty() {
                // No `T::` (shouldn't happen post-candidacy, but safe).
                return None;
            }
            let mut resolved: HashMap<(String, String), Expr> = HashMap::new();
            // Value bindings per bound struct (struct -> param -> literal).
            let mut struct_binds: HashMap<String, HashMap<String, Expr>> = HashMap::new();
            for (t, m) in &uses {
                let concrete = binds.get(t)?;
                // Concrete must be `Struct<lit-args...>` (literals only).
                let (sbase, sargs) = parse_tid(concrete)?;
                // Struct template value-params.
                let tps = struct_tparams.get(&sbase)?;
                // Bind positionally + trailing defaults. All value-params
                // must bind; extras decline; non-literals decline.
                let vb = struct_binds.entry(sbase.clone()).or_insert_with(|| {
                    let mut map: HashMap<String, Expr> = HashMap::new();
                    // (Filled below; entry API needs the closure. Rework:)
                    map
                });
                // (Re)build deterministically per (struct, args) — but
                // args are fixed per T binding, so build once per struct.
                // To keep it simple and sound, build fresh each use and
                // require identical results (they will be; deterministic).
                let mut fresh: HashMap<String, Expr> = HashMap::new();
                for (j, (pname, is_value)) in tps.iter().enumerate() {
                    if !is_value {
                        // Typename param in a value position: concrete
                        // must not need it (only literal args supported).
                        // If this param is referenced by the needed static,
                        // eval declines below. Record nothing; the static
                        // lookup will fail-safe. Actually strict: decline
                        // the whole resolution if any typename param exists
                        // (concrete can't supply types here).
                        return None;
                    }
                    if let Some(a) = sargs.get(j) {
                        let lit = literal_expr(a)?;
                        fresh.insert(pname.clone(), lit);
                    } else {
                        // Trailing default only.
                        let def = module
                            .template_param_defaults
                            .get(&format!("{sbase}::{pname}"))?;
                        let lit = literal_expr(def)?;
                        fresh.insert(pname.clone(), lit);
                    }
                }
                if sargs.len() > tps.len() {
                    return None;
                }
                // Merge (must agree; deterministic so always does).
                for (k, v) in fresh {
                    if let Some(prev) = vb.get(&k) {
                        if prev != &v {
                            return None;
                        }
                    } else {
                        vb.insert(k, v);
                    }
                }
                let bindings = struct_binds.get(&sbase)?;
                // Static init expr.
                let init = module.template_static_inits.get(&format!("{sbase}::{m}"))?;
                let lit = eval_const(init, bindings)?;
                // Must be Bool/Number literal.
                match &lit {
                    Expr::Bool(_) | Expr::Number(_) => {}
                    _ => return None,
                }
                resolved.insert((t.clone(), m.clone()), lit);
            }
            // All `T::` resolved: clone, substitute, cache, retarget.
            let mut key = String::new();
            let mut bs: Vec<(&String, &String)> = binds.iter().collect();
            bs.sort();
            for (t, v) in bs {
                key.push_str(&format!("{t}={v};"));
            }
            let cache_key = (callee.to_string(), key);
            if let Some(existing) = clones.get(&cache_key) {
                return Some(existing.clone());
            }
            let mut cname = format!("{callee}__qual{}", clones.len());
            while taken.contains(&cname)
                || clones.values().any(|n| n == &cname)
                || new_functions.iter().any(|f| f.name == cname)
            {
                cname.push('_');
            }
            let mut clone = def.clone();
            clone.name = cname.clone();
            clone.template_params = Vec::new();
            for s in clone.body.iter_mut() {
                subst_qualified_stmt(s, &resolved);
            }
            clones.insert(cache_key, cname.clone());
            new_functions.push(clone);
            retargeted.insert(callee.to_string());
            Some(cname)
        }
        let _ = (cidx, walk_call_expr, walk_call_stmt, try_retarget);
    }
    module.functions.extend(new_functions);
    // Drop fully-retargeted generics (zero uses remain, not the entry).
    // Partially-retargeted templates stay for their declined callers
    // (loud, correct). Uncalled templates with `T::` drop too (dead).
    {
        let mut uses: std::collections::HashSet<String> = std::collections::HashSet::new();
        for f in &module.functions {
            for s in &f.body {
                callee_names_stmt(s, &mut uses);
            }
        }
        // (Bare + mangled spellings pin defs; mirror mono's drop.)
        let drop: Vec<String> = cands
            .keys()
            .filter(|name| {
                Some((*name).as_str()) != entry_name
                    && !uses.contains(*name)
            })
            .cloned()
            .collect();
        for name in drop {
            if let Some(pos) = module.functions.iter().position(|f| &f.name == &name) {
                module.functions.remove(pos);
            }
        }
    }
    module
}

/// Caller-scope constant evaluation for explicit VALUE template arguments.
///
/// C++ evaluates template arguments as constant expressions in the caller's
/// scope (`run<flags>` with `const flags = 16` means `run<16>`), but the
/// parser keeps the spelling verbatim and monomorphization only binds
/// literals — so identifier arguments never instantiate and the callee
/// leaks to HLSL undeclared (`run`, and every `Closest<optimizedFlags>`
/// nested inside it). This pass folds them: per function, collect
/// single-definition locals whose inits fold to integers through module
/// constants + already-folded locals (fixpoint, so chains like
/// `optimizedFlags = flags | kNoAABBGeometryInBvh | kNoAABBInstanceInBvh`
/// resolve), then respell `f<ident>` arguments to decimal literals.
/// Anything unresolvable (params, reassigned locals, non-constant inits)
/// stays untouched — today's loud path. Runs every monomorphization round
/// so fresh clones resolve their own nested template-ids.
pub(crate) fn resolve_mangled_value_args(module: &mut Module) -> bool {
    use crate::ast::{BinaryOp, UnaryOp};
    // Module-constant pool (evaluated enumerators, constexprs: `Number`).
    let mut pool: HashMap<String, i64> = HashMap::new();
    for (name, expr) in &module.constants {
        if let Expr::Number(text) = expr {
            if let Some(v) = parse_int_literal(text) {
                pool.insert(name.clone(), v);
            }
        }
    }
    fn parse_int_literal(text: &str) -> Option<i64> {
        let t = text.trim().trim_end_matches(['u', 'U', 'l', 'L']);
        if t.is_empty() {
            return None;
        }
        let neg = t.starts_with('-');
        let digits = t.trim_start_matches(['-', '+']);
        let (digits, radix) = match digits.strip_prefix("0x").or_else(|| digits.strip_prefix("0X")) {
            Some(hex) => (hex.trim_end_matches(['u', 'U', 'l', 'L']), 16),
            None => (digits, 10),
        };
        // Digit separators (`1'000`) never survive lexing; reject junk.
        if digits.is_empty() || !digits.chars().all(|c| c.is_ascii_hexdigit() || radix == 10 && c.is_ascii_digit()) {
            return None;
        }
        if radix == 16 && !digits.chars().all(|c| c.is_ascii_hexdigit()) {
            return None;
        }
        let v = i64::from_str_radix(digits, radix).ok()?;
        Some(if neg { v.checked_neg()? } else { v })
    }
    fn fold_expr(e: &Expr, scope: &HashMap<String, i64>) -> Option<i64> {
        match e {
            Expr::Number(text) => parse_int_literal(text),
            Expr::Bool(b) => Some(i64::from(*b)),
            Expr::Variable(name) => scope.get(name).copied(),
            Expr::Unary { op, expr } => {
                let v = fold_expr(expr, scope)?;
                match op {
                    UnaryOp::Plus => Some(v),
                    UnaryOp::Minus => v.checked_neg(),
                    UnaryOp::BitNot => Some(!v),
                    UnaryOp::Not => Some(i64::from(v == 0)),
                }
            }
            Expr::Binary { left, op, right } => {
                let (l, r) = (fold_expr(left, scope)?, fold_expr(right, scope)?);
                match op {
                    BinaryOp::Add => l.checked_add(r),
                    BinaryOp::Sub => l.checked_sub(r),
                    BinaryOp::Mul => l.checked_mul(r),
                    BinaryOp::Div => l.checked_div(r),
                    BinaryOp::Mod => l.checked_rem(r),
                    BinaryOp::BitAnd => Some(l & r),
                    BinaryOp::BitOr => Some(l | r),
                    BinaryOp::BitXor => Some(l ^ r),
                    BinaryOp::ShiftLeft => l.checked_shl(r as u32).map(|v| v as i64),
                    BinaryOp::ShiftRight => l.checked_shr(r as u32).map(|v| v as i64),
                    _ => None,
                }
            }
            _ => None,
        }
    }
    // Statement walkers: declaration counting (+= params, lambda params,
    // nested VarDecls — repeated names veto flat folding), assignment
    // targets (any `x = ...` vetoes `x`), and call-site respelling.
    fn count_decls(stmts: &[Stmt], counts: &mut HashMap<String, usize>) {
        for s in stmts {
            match s {
                Stmt::VarDecl { name, init, .. } => {
                    *counts.entry(name.clone()).or_insert(0) += 1;
                    if let Some(e) = init {
                        count_lambda_decls(e, counts);
                    }
                }
                Stmt::Destructure { names, init, .. } => {
                    for n in names {
                        *counts.entry(n.clone()).or_insert(0) += 1;
                    }
                    if let Some(e) = init {
                        count_lambda_decls(e, counts);
                    }
                }
                Stmt::Return(e) => {
                    if let Some(e) = e {
                        count_lambda_decls(e, counts);
                    }
                }
                Stmt::Expr(e) => count_lambda_decls(e, counts),
                Stmt::Block(ss) => count_decls(ss, counts),
                Stmt::If { condition, then_branch, else_branch } => {
                    count_lambda_decls(condition, counts);
                    count_decls(then_branch, counts);
                    if let Some(eb) = else_branch {
                        count_decls(eb, counts);
                    }
                }
                Stmt::For { init, condition, update, body } => {
                    count_decls(std::slice::from_ref(init), counts);
                    if let Some(c) = condition {
                        count_lambda_decls(c, counts);
                    }
                    if let Some(u) = update {
                        count_lambda_decls(u, counts);
                    }
                    count_decls(body, counts);
                }
                Stmt::While { condition, body } => {
                    count_lambda_decls(condition, counts);
                    count_decls(body, counts);
                }
                Stmt::Switch { scrutinee, cases } => {
                    count_lambda_decls(scrutinee, counts);
                    for c in cases {
                        if let Some(l) = &c.label {
                            count_lambda_decls(l, counts);
                        }
                        count_decls(&c.body, counts);
                    }
                }
            }
        }
    }
    fn count_lambda_decls(e: &Expr, counts: &mut HashMap<String, usize>) {
        match e {
            Expr::Lambda { params, body, .. } => {
                for p in params {
                    *counts.entry(p.clone()).or_insert(0) += 1;
                }
                count_decls(body, counts);
            }
            Expr::Call { function, arguments } => {
                count_lambda_decls(function, counts);
                for a in arguments {
                    count_lambda_decls(a, counts);
                }
            }
            Expr::Member { object, .. } => count_lambda_decls(object, counts),
            Expr::Index { object, index } => {
                count_lambda_decls(object, counts);
                count_lambda_decls(index, counts);
            }
            Expr::Assign { lhs, rhs } => {
                count_lambda_decls(lhs, counts);
                count_lambda_decls(rhs, counts);
            }
            Expr::Unary { expr, .. } => count_lambda_decls(expr, counts),
            Expr::Binary { left, right, .. } => {
                count_lambda_decls(left, counts);
                count_lambda_decls(right, counts);
            }
            Expr::Conditional { condition, then_branch, else_branch } => {
                count_lambda_decls(condition, counts);
                count_lambda_decls(then_branch, counts);
                count_lambda_decls(else_branch, counts);
            }
            Expr::Constructor { arguments, .. } => {
                for a in arguments {
                    count_lambda_decls(a, counts);
                }
            }
            Expr::InitializerList(items) => {
                for i in items {
                    count_lambda_decls(i, counts);
                }
            }
            Expr::Number(_) | Expr::String(_) | Expr::Bool(_) | Expr::Variable(_) => {}
        }
    }
    fn mark_assigned_expr(e: &Expr, assigned: &mut HashSet<String>) {
        match e {
            Expr::Assign { lhs, rhs } => {
                if let Expr::Variable(n) = lhs.as_ref() {
                    assigned.insert(n.clone());
                }
                mark_assigned_expr(lhs, assigned);
                mark_assigned_expr(rhs, assigned);
            }
            Expr::Call { function, arguments } => {
                mark_assigned_expr(function, assigned);
                for a in arguments {
                    mark_assigned_expr(a, assigned);
                }
            }
            Expr::Member { object, .. } => mark_assigned_expr(object, assigned),
            Expr::Index { object, index } => {
                mark_assigned_expr(object, assigned);
                mark_assigned_expr(index, assigned);
            }
            Expr::Unary { expr, .. } => mark_assigned_expr(expr, assigned),
            Expr::Binary { left, right, .. } => {
                mark_assigned_expr(left, assigned);
                mark_assigned_expr(right, assigned);
            }
            Expr::Conditional { condition, then_branch, else_branch } => {
                mark_assigned_expr(condition, assigned);
                mark_assigned_expr(then_branch, assigned);
                mark_assigned_expr(else_branch, assigned);
            }
            Expr::Constructor { arguments, .. } => {
                for a in arguments {
                    mark_assigned_expr(a, assigned);
                }
            }
            Expr::InitializerList(items) => {
                for i in items {
                    mark_assigned_expr(i, assigned);
                }
            }
            Expr::Lambda { body, .. } => {
                for s in body {
                    mark_assigned_stmt(s, assigned);
                }
            }
            Expr::Number(_) | Expr::String(_) | Expr::Bool(_) | Expr::Variable(_) => {}
        }
    }
    fn mark_assigned_stmt(s: &Stmt, assigned: &mut HashSet<String>) {
        match s {
            Stmt::Return(e) => {
                if let Some(e) = e {
                    mark_assigned_expr(e, assigned);
                }
            }
            Stmt::VarDecl { init, .. } => {
                if let Some(e) = init {
                    mark_assigned_expr(e, assigned);
                }
            }
            Stmt::Destructure { init, .. } => {
                if let Some(e) = init {
                    mark_assigned_expr(e, assigned);
                }
            }
            Stmt::Expr(e) => mark_assigned_expr(e, assigned),
            Stmt::Block(ss) => {
                for x in ss {
                    mark_assigned_stmt(x, assigned);
                }
            }
            Stmt::If { condition, then_branch, else_branch } => {
                mark_assigned_expr(condition, assigned);
                for x in then_branch {
                    mark_assigned_stmt(x, assigned);
                }
                if let Some(eb) = else_branch {
                    for x in eb {
                        mark_assigned_stmt(x, assigned);
                    }
                }
            }
            Stmt::For { init, condition, update, body } => {
                mark_assigned_stmt(init, assigned);
                if let Some(c) = condition {
                    mark_assigned_expr(c, assigned);
                }
                if let Some(u) = update {
                    mark_assigned_expr(u, assigned);
                }
                for x in body {
                    mark_assigned_stmt(x, assigned);
                }
            }
            Stmt::While { condition, body } => {
                mark_assigned_expr(condition, assigned);
                for x in body {
                    mark_assigned_stmt(x, assigned);
                }
            }
            Stmt::Switch { scrutinee, cases } => {
                mark_assigned_expr(scrutinee, assigned);
                for c in cases {
                    if let Some(l) = &c.label {
                        mark_assigned_expr(l, assigned);
                    }
                    for x in &c.body {
                        mark_assigned_stmt(x, assigned);
                    }
                }
            }
        }
    }
    fn is_bare_ident(s: &str) -> bool {
        !s.is_empty()
            && s.chars().next().is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
            && s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
    }
    // Top-level comma split respecting `<...>`/`(...)`/`[...]`/`{...}`.
    fn split_args(inner: &str) -> Option<Vec<String>> {
        let mut parts = Vec::new();
        let mut depth_angle = 0usize;
        let mut depth_paren = 0usize;
        let mut depth_brack = 0usize;
        let mut depth_brace = 0usize;
        let mut start = 0usize;
        let bytes = inner.as_bytes();
        let mut i = 0usize;
        while i < bytes.len() {
            match bytes[i] {
                b'<' => depth_angle += 1,
                b'>' => depth_angle = depth_angle.saturating_sub(1),
                b'(' => depth_paren += 1,
                b')' => depth_paren = depth_paren.saturating_sub(1),
                b'[' => depth_brack += 1,
                b']' => depth_brack = depth_brack.saturating_sub(1),
                b'{' => depth_brace += 1,
                b'}' => depth_brace = depth_brace.saturating_sub(1),
                b',' if depth_angle == 0 && depth_paren == 0 && depth_brack == 0 && depth_brace == 0 => {
                    parts.push(inner[start..i].to_string());
                    start = i + 1;
                }
                _ => {}
            }
            i += 1;
        }
        parts.push(inner[start..].to_string());
        Some(parts)
    }
    fn respell_expr(e: &mut Expr, scope: &HashMap<String, i64>, progress: &mut bool) {
        match e {
            Expr::Call { function, arguments } => {
                if let Expr::Variable(callee) = function.as_mut() {
                    if let Some((base, rest)) = callee.split_once('<') {
                        if rest.ends_with('>') && !base.is_empty() {
                            let inner = &rest[..rest.len() - 1];
                            if !inner.contains('<') && !inner.contains('>') {
                                if let Some(parts) = split_args(inner) {
                                    let mut changed = false;
                                    let out: Vec<String> = parts
                                        .iter()
                                        .map(|p| {
                                            let t = p.trim();
                                            // Bare constant (`flags`) folds.
                                            if is_bare_ident(t) {
                                                if let Some(v) = scope.get(t) {
                                                    changed = true;
                                                    return v.to_string();
                                                }
                                                return t.to_string();
                                            }
                                            // Scoped enum constant
                                            // (`RenderingMode::kLoop`): the
                                            // enumerator is registered bare
                                            // (`kLoop`) in the module
                                            // constants, so fold the tail.
                                            // Framework enumerators are `k…`; other
                                            // scoped spellings stay for the
                                            // type-mungled paths downstream.
                                            if let Some((_, tail)) = t.rsplit_once("::") {
                                                if is_bare_ident(tail) && tail.starts_with('k') {
                                                    if let Some(v) = scope.get(tail) {
                                                        changed = true;
                                                        return v.to_string();
                                                    }
                                                }
                                            }
                                            t.to_string()
                                        })
                                        .collect();
                                    if changed {
                                        *callee = format!("{base}<{}>", out.join(","));
                                        *progress = true;
                                    }
                                }
                            }
                        }
                    }
                }
                respell_expr(function, scope, progress);
                for a in arguments.iter_mut() {
                    respell_expr(a, scope, progress);
                }
            }
            Expr::Member { object, .. } => respell_expr(object, scope, progress),
            Expr::Index { object, index } => {
                respell_expr(object, scope, progress);
                respell_expr(index, scope, progress);
            }
            Expr::Assign { lhs, rhs } => {
                respell_expr(lhs, scope, progress);
                respell_expr(rhs, scope, progress);
            }
            Expr::Unary { expr, .. } => respell_expr(expr, scope, progress),
            Expr::Binary { left, right, .. } => {
                respell_expr(left, scope, progress);
                respell_expr(right, scope, progress);
            }
            Expr::Conditional { condition, then_branch, else_branch } => {
                respell_expr(condition, scope, progress);
                respell_expr(then_branch, scope, progress);
                respell_expr(else_branch, scope, progress);
            }
            Expr::Constructor { arguments, .. } => {
                for a in arguments.iter_mut() {
                    respell_expr(a, scope, progress);
                }
            }
            Expr::InitializerList(items) => {
                for i in items.iter_mut() {
                    respell_expr(i, scope, progress);
                }
            }
            Expr::Lambda { body, .. } => {
                for s in body.iter_mut() {
                    respell_stmt(s, scope, progress);
                }
            }
            Expr::Number(_) | Expr::String(_) | Expr::Bool(_) | Expr::Variable(_) => {}
        }
    }
    fn respell_stmt(s: &mut Stmt, scope: &HashMap<String, i64>, progress: &mut bool) {
        match s {
            Stmt::Return(e) => {
                if let Some(e) = e {
                    respell_expr(e, scope, progress);
                }
            }
            Stmt::VarDecl { init, .. } => {
                if let Some(e) = init {
                    respell_expr(e, scope, progress);
                }
            }
            Stmt::Destructure { init, .. } => {
                if let Some(e) = init {
                    respell_expr(e, scope, progress);
                }
            }
            Stmt::Expr(e) => respell_expr(e, scope, progress),
            Stmt::Block(ss) => {
                for x in ss.iter_mut() {
                    respell_stmt(x, scope, progress);
                }
            }
            Stmt::If { condition, then_branch, else_branch } => {
                respell_expr(condition, scope, progress);
                for x in then_branch.iter_mut() {
                    respell_stmt(x, scope, progress);
                }
                if let Some(eb) = else_branch {
                    for x in eb.iter_mut() {
                        respell_stmt(x, scope, progress);
                    }
                }
            }
            Stmt::For { init, condition, update, body } => {
                respell_stmt(init, scope, progress);
                if let Some(c) = condition {
                    respell_expr(c, scope, progress);
                }
                if let Some(u) = update {
                    respell_expr(u, scope, progress);
                }
                for x in body.iter_mut() {
                    respell_stmt(x, scope, progress);
                }
            }
            Stmt::While { condition, body } => {
                respell_expr(condition, scope, progress);
                for x in body.iter_mut() {
                    respell_stmt(x, scope, progress);
                }
            }
            Stmt::Switch { scrutinee, cases } => {
                respell_expr(scrutinee, scope, progress);
                for c in cases.iter_mut() {
                    if let Some(l) = c.label.as_mut() {
                        respell_expr(l, scope, progress);
                    }
                    for x in c.body.iter_mut() {
                        respell_stmt(x, scope, progress);
                    }
                }
            }
        }
    }
    fn collect_inits(stmts: &[Stmt], inits: &mut Vec<(String, Expr)>) {
        for s in stmts {
            match s {
                Stmt::VarDecl { name, init: Some(e), .. } => {
                    inits.push((name.clone(), e.clone()));
                }
                Stmt::Block(ss) => collect_inits(ss, inits),
                Stmt::If { then_branch, else_branch, .. } => {
                    collect_inits(then_branch, inits);
                    if let Some(eb) = else_branch {
                        collect_inits(eb, inits);
                    }
                }
                Stmt::For { init, body, .. } => {
                    collect_inits(std::slice::from_ref(init), inits);
                    collect_inits(body, inits);
                }
                Stmt::While { body, .. } => collect_inits(body, inits),
                Stmt::Switch { cases, .. } => {
                    for c in cases {
                        collect_inits(&c.body, inits);
                    }
                }
                Stmt::Return(_) | Stmt::Expr(_) | Stmt::VarDecl { init: None, .. } | Stmt::Destructure { .. } => {}
            }
        }
    }
    let mut progress = false;
    for f in module.functions.iter_mut() {
        // Declaration census: params + all nested VarDecls/lambda params.
        // Repeated names veto flat folding (shadowing would be unsound).
        let mut counts: HashMap<String, usize> = HashMap::new();
        for p in &f.parameters {
            *counts.entry(p.name.clone()).or_insert(0) += 1;
        }
        count_decls(&f.body, &mut counts);
        let mut assigned: HashSet<String> = HashSet::new();
        for s in &f.body {
            mark_assigned_stmt(s, &mut assigned);
        }
        let mut scope: HashMap<String, i64> = pool.clone();
        // Soundness: function-declared names (params, locals) shadow
        // module constants. Evict every declared name from the scope so
        // a runtime param (`void f(uint kNone) { g<kNone>(); }`) never
        // folds to an unrelated global; only single-definition foldable
        // locals are added back below.
        for name in counts.keys() {
            scope.remove(name);
        }
        // Foldable locals, fixpoint for init chains (`b = a | 3`).
        let mut inits: Vec<(String, Expr)> = Vec::new();
        collect_inits(&f.body, &mut inits);
        let mut changed = true;
        while changed {
            changed = false;
            for (name, init) in &inits {
                if scope.contains_key(name)
                    || counts.get(name).copied().unwrap_or(0) != 1
                    || assigned.contains(name)
                {
                    continue;
                }
                if let Some(v) = fold_expr(init, &scope) {
                    scope.insert(name.clone(), v);
                    changed = true;
                }
            }
        }
        // Drop pool-only entries the function never declares? No — the
        // callee args may name module constants directly (`Closest<f>`
        // never does, but `run<kFlag>` could). Keep the full scope.
        for s in f.body.iter_mut() {
            respell_stmt(s, &scope, &mut progress);
        }
    }
    progress
}

pub(crate) fn monomorphize(module: &Module, entry_name: Option<&str>) -> Module {
    use crate::ast::Stmt;
    // Template registry entry (owned; bodies mutate without borrows).
    // Per-overload data now reads directly off candidate `Function`s;
    // the entry itself is only an existence gate (at least one
    // qualifying overload), so its fields may go unread.
    #[allow(dead_code)]
    struct TInfo {
        tparams: Vec<(String, bool)>,
        auto_positions: Vec<usize>,
    }
    // Substitute a type (Named T/auto only; resources recurse).
    fn subst_ty(ty: &Type, map: &HashMap<String, Type>) -> Type {
        match ty {
            Type::Named(n) => map.get(n).cloned().unwrap_or_else(|| ty.clone()),
            Type::Resource { kind, access, element } => Type::Resource {
                kind: kind.clone(),
                access: *access,
                element: Box::new(subst_ty(element, map)),
            },
            _ => ty.clone(),
        }
    }
    // Substitute types in an expression (constructors only carry types).
    fn subst_expr(e: &mut Expr, map: &HashMap<String, Type>) {
        match e {
            Expr::Constructor { ty, arguments } => {
                *ty = subst_ty(ty, map);
                for a in arguments.iter_mut() {
                    subst_expr(a, map);
                }
            }
            Expr::Call { function, arguments } => {
                subst_expr(function, map);
                for a in arguments.iter_mut() {
                    subst_expr(a, map);
                }
            }
            Expr::Member { object, .. } => subst_expr(object, map),
            Expr::Index { object, index } => {
                subst_expr(object, map);
                subst_expr(index, map);
            }
            Expr::Assign { lhs, rhs } => {
                subst_expr(lhs, map);
                subst_expr(rhs, map);
            }
            Expr::Unary { expr, .. } => subst_expr(expr, map),
            Expr::Binary { left, right, .. } => {
                subst_expr(left, map);
                subst_expr(right, map);
            }
            Expr::Conditional { condition, then_branch, else_branch } => {
                subst_expr(condition, map);
                subst_expr(then_branch, map);
                subst_expr(else_branch, map);
            }
            Expr::InitializerList(elems) => {
                for x in elems.iter_mut() {
                    subst_expr(x, map);
                }
            }
            Expr::Lambda { body, .. } => {
                for s in body.iter_mut() {
                    subst_stmt(s, map);
                }
            }
            Expr::Variable(_) | Expr::Number(_) | Expr::String(_) | Expr::Bool(_) => {}
        }
    }
    fn subst_stmt(s: &mut Stmt, map: &HashMap<String, Type>) {
        match s {
            Stmt::VarDecl { ty, init, .. } => {
                *ty = subst_ty(ty, map);
                if let Some(e) = init {
                    subst_expr(e, map);
                }
            }
            Stmt::Destructure { ty, init, .. } => {
                *ty = subst_ty(ty, map);
                if let Some(e) = init {
                    subst_expr(e, map);
                }
            }
            Stmt::Return(Some(e)) | Stmt::Expr(e) => subst_expr(e, map),
            Stmt::Return(None) => {}
            Stmt::Block(ss) => {
                for x in ss.iter_mut() {
                    subst_stmt(x, map);
                }
            }
            Stmt::If { condition, then_branch, else_branch } => {
                subst_expr(condition, map);
                for x in then_branch.iter_mut() {
                    subst_stmt(x, map);
                }
                if let Some(els) = else_branch {
                    for x in els.iter_mut() {
                        subst_stmt(x, map);
                    }
                }
            }
            Stmt::For { init, condition, update, body } => {
                subst_stmt(init, map);
                if let Some(c) = condition {
                    subst_expr(c, map);
                }
                if let Some(u) = update {
                    subst_expr(u, map);
                }
                for x in body.iter_mut() {
                    subst_stmt(x, map);
                }
            }
            Stmt::While { condition, body } => {
                subst_expr(condition, map);
                for x in body.iter_mut() {
                    subst_stmt(x, map);
                }
            }
            Stmt::Switch { scrutinee, cases } => {
                subst_expr(scrutinee, map);
                for c in cases.iter_mut() {
                    if let Some(l) = &mut c.label {
                        subst_expr(l, map);
                    }
                    for x in c.body.iter_mut() {
                        subst_stmt(x, map);
                    }
                }
            }
        }
    }
    // True when `T::` (T a template typename) appears anywhere (types
    // and expressions): phase-2 shapes stay loud. Bare `T` in TYPE
    // positions (params, returns, `MS_Texture2D<type>` carriers,
    // `type x[4]` locals, `T(...)` constructors) are substitution
    // sites the clone concretizes — only `T::` spellings veto.
    // Value-position bare `T` (a variable literally named `T`, e.g.
    // inside `sizeof(T)`) still vetoes: substitution cannot rewrite it.
    fn uses_qualified_param(f: &Function, typenames: &HashSet<String>) -> bool {
        if typenames.is_empty() {
            return false;
        }
        fn ty_uses(t: &Type, names: &HashSet<String>) -> bool {
            match t {
                Type::Named(n) => names.iter().any(|p| n.starts_with(&format!("{p}::"))),
                Type::Resource { element, .. } => ty_uses(element, names),
                _ => false,
            }
        }
        fn expr_uses(e: &Expr, names: &HashSet<String>) -> bool {
            match e {
                Expr::Variable(n) => names.iter().any(|p| n == p || n.starts_with(&format!("{p}::"))),
                Expr::Constructor { ty, arguments } => {
                    ty_uses(ty, names) || arguments.iter().any(|a| expr_uses(a, names))
                }
                Expr::Call { function, arguments } => {
                    expr_uses(function, names) || arguments.iter().any(|a| expr_uses(a, names))
                }
                Expr::Member { object, .. } => expr_uses(object, names),
                Expr::Index { object, index } => {
                    expr_uses(object, names) || expr_uses(index, names)
                }
                Expr::Assign { lhs, rhs } => expr_uses(lhs, names) || expr_uses(rhs, names),
                Expr::Unary { expr, .. } => expr_uses(expr, names),
                Expr::Binary { left, right, .. } => {
                    expr_uses(left, names) || expr_uses(right, names)
                }
                Expr::Conditional { condition, then_branch, else_branch } => {
                    expr_uses(condition, names)
                        || expr_uses(then_branch, names)
                        || expr_uses(else_branch, names)
                }
                Expr::InitializerList(elems) => elems.iter().any(|a| expr_uses(a, names)),
                Expr::Lambda { body, .. } => body.iter().any(|s| stmt_uses(s, names)),
                Expr::Number(_) | Expr::String(_) | Expr::Bool(_) => false,
            }
        }
        fn stmt_uses(s: &Stmt, names: &HashSet<String>) -> bool {
            match s {
                Stmt::Return(Some(e)) | Stmt::Expr(e) => expr_uses(e, names),
                Stmt::Return(None) => false,
                Stmt::VarDecl { ty, init, .. } | Stmt::Destructure { ty, init, .. } => {
                    ty_uses(ty, names)
                        || init.as_ref().is_some_and(|e| expr_uses(e, names))
                }
                Stmt::Block(ss) => ss.iter().any(|x| stmt_uses(x, names)),
                Stmt::If { condition, then_branch, else_branch } => {
                    expr_uses(condition, names)
                        || then_branch.iter().any(|x| stmt_uses(x, names))
                        || else_branch.iter().flatten().any(|x| stmt_uses(x, names))
                }
                Stmt::For { init, condition, update, body } => {
                    stmt_uses(init, names)
                        || condition.as_ref().is_some_and(|c| expr_uses(c, names))
                        || update.as_ref().is_some_and(|u| expr_uses(u, names))
                        || body.iter().any(|x| stmt_uses(x, names))
                }
                Stmt::While { condition, body } => {
                    expr_uses(condition, names)
                        || body.iter().any(|x| stmt_uses(x, names))
                }
                Stmt::Switch { scrutinee, cases } => {
                    expr_uses(scrutinee, names)
                        || cases.iter().any(|c| {
                            c.label.as_ref().is_some_and(|l| expr_uses(l, names))
                                || c.body.iter().any(|x| stmt_uses(x, names))
                        })
                }
            }
        }
        if ty_uses(&f.return_type, typenames) {
            return true;
        }
        for p in &f.parameters {
            if ty_uses(&p.ty, typenames) {
                return true;
            }
        }
        f.body.iter().any(|s| stmt_uses(s, typenames))
    }

    // Split a mangled callee (`name<lit,...>`) into bare name + literal
    // args; None when not mangled-shaped.
    fn split_mangled(callee: &str) -> Option<(String, Vec<Expr>)> {
        let (base, rest) = callee.split_once('<')?;
        if base.is_empty() || !rest.ends_with('>') {
            return None;
        }
        let inner = &rest[..rest.len() - 1];
        if inner.contains('<') || inner.contains('>') {
            return None;
        }
        let mut args = Vec::new();
        for part in inner.split(',') {
            let part = part.trim();
            if part.eq_ignore_ascii_case("true") {
                args.push(Expr::Bool(true));
            } else if part.eq_ignore_ascii_case("false") {
                args.push(Expr::Bool(false));
            } else if !part.is_empty()
                && part.chars().all(|c| {
                    c.is_ascii_hexdigit()
                        || matches!(c, 'x' | 'X' | 'u' | 'U' | 'l' | 'L' | '\'' | '+' | '-')
                })
            {
                args.push(Expr::Number(part.to_string()));
            } else {
                return None;
            }
        }
        if args.is_empty() {
            return None;
        }
        Some((base.to_string(), args))
    }
    // Split a mangled callee with TYPE args (`name<T1, T2>`) into bare
    // name + parsed types; None when not type-mangled-shaped. Only flat
    // spellings qualify (no nesting); each arg parses via
    // `parse_named_type` (scalars/vectors/matrices/aliases/bare names —
    // validity is checked by `concrete_ty` at binding time, mirroring
    // how literal args defer to use sites). Runs only when
    // `split_mangled` (literal/value args) declines, so `f<true>` keeps
    // its value-arg path.
    fn split_mangled_ty(callee: &str) -> Option<(String, Vec<Type>)> {
        let (base, rest) = callee.split_once('<')?;
        if base.is_empty() || !rest.ends_with('>') {
            return None;
        }
        let inner = &rest[..rest.len() - 1];
        if inner.contains('<') || inner.contains('>') {
            return None;
        }
        let mut args = Vec::new();
        for part in inner.split(',') {
            let part = part.trim();
            if part.is_empty()
                || !part.chars().all(|c| {
                    c.is_ascii_alphanumeric() || matches!(c, '_' | ':')
                })
            {
                return None;
            }
            // Pure scalar spellings (`float`, `half`) parse as `Named`
            // via `parse_named_type`; resolve through `builtin_type`
            // (mirrors the clone-body respelling guard below) so they
            // bind as concrete scalars instead of declining.
            let ty = crate::parser::types::parse_named_type(part);
            let ty = match &ty {
                Type::Named(n) => {
                    crate::parser::types::builtin_type(n).unwrap_or(ty)
                }
                _ => ty,
            };
            args.push(ty);
        }
        if args.is_empty() {
            return None;
        }
        Some((base.to_string(), args))
    }
    /// One explicit template argument that is either a type or a value.
    enum MixedArg {
        Ty(Type),
        Val(Expr),
    }
    // Split a mangled callee with MIXED args (`name<T, false>`) into bare
    // name + per-arg kinds. Value classification mirrors `split_mangled`
    // exactly (bool words, number charset), so `f<face>` stays a value
    // there as here; anything else identifier-shaped parses as a type
    // exactly like `split_mangled_ty`. Runs when both of those decline
    // (each only accepts uniform lists), so uniform spellings keep their
    // existing paths. None when not flat-mangled-shaped or any arg fits
    // neither class.
    fn split_mangled_mixed(callee: &str) -> Option<(String, Vec<MixedArg>)> {
        let (base, rest) = callee.split_once('<')?;
        if base.is_empty() || !rest.ends_with('>') {
            return None;
        }
        let inner = &rest[..rest.len() - 1];
        if inner.contains('<') || inner.contains('>') {
            return None;
        }
        let mut args = Vec::new();
        for part in inner.split(',') {
            let part = part.trim();
            if part.is_empty() {
                return None;
            }
            if part.eq_ignore_ascii_case("true") {
                args.push(MixedArg::Val(Expr::Bool(true)));
            } else if part.eq_ignore_ascii_case("false") {
                args.push(MixedArg::Val(Expr::Bool(false)));
            } else if part.chars().all(|c| {
                c.is_ascii_hexdigit()
                    || matches!(c, 'x' | 'X' | 'u' | 'U' | 'l' | 'L' | '\'' | '+' | '-')
            }) {
                args.push(MixedArg::Val(Expr::Number(part.to_string())));
            } else if part.chars().all(|c| {
                c.is_ascii_alphanumeric() || matches!(c, '_' | ':')
            }) {
                let ty = crate::parser::types::parse_named_type(part);
                let ty = match &ty {
                    Type::Named(n) => {
                        crate::parser::types::builtin_type(n).unwrap_or(ty)
                    }
                    _ => ty,
                };
                args.push(MixedArg::Ty(ty));
            } else {
                return None;
            }
        }
        if args.is_empty() {
            return None;
        }
        Some((base.to_string(), args))
    }
    // Concrete emittable deduction (not auto, opaque, or still-generic).
    fn concrete_ty(t: &Type, structs: &HashSet<String>) -> bool {
        match t {
            Type::Void => false,
            Type::Bool | Type::Int | Type::UInt | Type::Float | Type::Half | Type::Double
            | Type::Vector { .. } | Type::Matrix { .. } => true,
            Type::Resource { .. } => true,
            Type::Named(n) => {
                structs.contains(n.as_str()) || crate::framework::is_sampler_type_name(n)
            }
        }
    }
    // Rewrites explicit type args inside a fresh clone using the outer
    // binding map (`Load4SamplesUsingFMask<type>` with {type: float3}
    // becomes `<float3>`): explicit args evaluate in caller scope, so
    // outer-bound names substitute (C++ [temp.deduct.call]). Only
    // bare-identifier args present in the map rewrite, only when the
    // base names a template (else the walk's loud path owns the
    // spelling), and only when the respelling re-parses to the same
    // types (display round-trip guard).
    fn rewrite_inner_targs(
        stmts: &mut [Stmt],
        map: &HashMap<String, Type>,
        registry: &HashMap<String, TInfo>,
    ) {
        fn ri_expr(e: &mut Expr, map: &HashMap<String, Type>, registry: &HashMap<String, TInfo>) {
            match e {
                Expr::Call { function, arguments } => {
                    if let Expr::Variable(callee) = function.as_mut() {
                        if let Some((base, rest)) = callee.split_once('<') {
                            if rest.ends_with('>') && registry.contains_key(base) {
                                let inner = &rest[..rest.len() - 1];
                                if !inner.contains('<') && !inner.contains('>') {
                                    let mut parts = Vec::new();
                                    let mut changed = false;
                                    let mut ok = true;
                                    for part in inner.split(',') {
                                        let part = part.trim();
                                        if let Some(ty) = map.get(part) {
                                            let s = ty.display();
                                            // Round-trip guard: the
                                            // respelling must parse back
                                            // to the same type (scalars
                                            // resolve via `builtin_type`,
                                            // composites via
                                            // `parse_named_type`).
                                            let round_trips = crate::parser::types::parse_named_type(
                                                &s,
                                            ) == *ty
                                                || crate::parser::types::builtin_type(&s).as_ref()
                                                    == Some(ty);
                                            if round_trips {
                                                parts.push(s);
                                                changed = true;
                                                continue;
                                            }
                                            ok = false;
                                            break;
                                        }
                                        parts.push(part.to_string());
                                    }
                                    if ok && changed {
                                        *callee = format!("{base}<{}>", parts.join(","));
                                    }
                                }
                            }
                        }
                    }
                    ri_expr(function, map, registry);
                    for a in arguments.iter_mut() {
                        ri_expr(a, map, registry);
                    }
                }
                Expr::Member { object, .. } => ri_expr(object, map, registry),
                Expr::Index { object, index } => {
                    ri_expr(object, map, registry);
                    ri_expr(index, map, registry);
                }
                Expr::Constructor { arguments, .. } => {
                    for a in arguments.iter_mut() {
                        ri_expr(a, map, registry);
                    }
                }
                Expr::Assign { lhs, rhs } => {
                    ri_expr(lhs, map, registry);
                    ri_expr(rhs, map, registry);
                }
                Expr::Unary { expr: inner, .. } => ri_expr(inner, map, registry),
                Expr::Binary { left, right, .. } => {
                    ri_expr(left, map, registry);
                    ri_expr(right, map, registry);
                }
                Expr::Conditional { condition, then_branch, else_branch } => {
                    ri_expr(condition, map, registry);
                    ri_expr(then_branch, map, registry);
                    ri_expr(else_branch, map, registry);
                }
                Expr::InitializerList(elems) => {
                    for x in elems.iter_mut() {
                        ri_expr(x, map, registry);
                    }
                }
                Expr::Lambda { body, .. } => ri_stmts(body, map, registry),
                _ => {}
            }
        }
        fn ri_stmts(ss: &mut [Stmt], map: &HashMap<String, Type>, registry: &HashMap<String, TInfo>) {
            for s in ss.iter_mut() {
                match s {
                    Stmt::Return(Some(e)) | Stmt::Expr(e) => ri_expr(e, map, registry),
                    Stmt::Return(None) => {}
                    Stmt::VarDecl { init: Some(e), .. }
                    | Stmt::Destructure { init: Some(e), .. } => ri_expr(e, map, registry),
                    Stmt::VarDecl { init: None, .. } | Stmt::Destructure { init: None, .. } => {}
                    Stmt::Block(inner) => ri_stmts(inner, map, registry),
                    Stmt::If { condition, then_branch, else_branch } => {
                        ri_expr(condition, map, registry);
                        ri_stmts(then_branch, map, registry);
                        if let Some(els) = else_branch {
                            ri_stmts(els, map, registry);
                        }
                    }
                    Stmt::For { init, condition, update, body } => {
                        ri_stmts(std::slice::from_mut(init.as_mut()), map, registry);
                        if let Some(c) = condition {
                            ri_expr(c, map, registry);
                        }
                        if let Some(u) = update {
                            ri_expr(u, map, registry);
                        }
                        ri_stmts(body, map, registry);
                    }
                    Stmt::While { condition, body } => {
                        ri_expr(condition, map, registry);
                        ri_stmts(body, map, registry);
                    }
                    Stmt::Switch { scrutinee, cases } => {
                        ri_expr(scrutinee, map, registry);
                        for c in cases.iter_mut() {
                            if let Some(l) = &mut c.label {
                                ri_expr(l, map, registry);
                            }
                            ri_stmts(&mut c.body, map, registry);
                        }
                    }
                }
            }
        }
        ri_stmts(stmts, map, registry);
    }
    // Try one call (callee spelling + args): build/reuse the clone and
    // rewrite the callee in place. Returns true on retarget. Mangled
    // callees with no template target restore the bare name.
    #[allow(clippy::too_many_arguments)]
    fn try_call(
        callee: &mut String,
        args: &[Expr],
        env: &HashMap<String, (Type, Option<u32>)>,
        struct_fields: &HashMap<String, HashMap<String, (Type, Option<u32>)>>,
        free_rets: &HashMap<String, Vec<((usize, usize), Type)>>,
        method_rets: &HashMap<(String, String), Vec<((usize, usize), Type)>>,
        registry: &HashMap<String, TInfo>,
        caller_tnames: &HashSet<String>,
        structs: &HashSet<String>,
        tpl_defs: &HashMap<String, Vec<Function>>,
        taken: &HashSet<String>,
        clones: &mut HashMap<(String, String), String>,
        new_functions: &mut Vec<Function>,
    ) -> bool {
        if std::env::var("PSSL_MONO_DBG").or_else(|_| std::env::var("PS5_MONO_DBG")).is_ok() && callee.contains("projectPixels3") {
            eprintln!("DBG try_call {callee} nargs={}", args.len());
        }
        let (base, explicit) = match split_mangled(callee) {
            Some((b, lits)) => (b, lits),
            None => (callee.clone(), Vec::new()),
        };
        // Explicit TYPE args (`minT<uint2>(...)`): `split_mangled` only
        // parses literal/value args, so these callees miss the registry
        // and degrade to bare calls (the emitter's generic fallback then
        // lowers ternaries with unknown `T` widths). Parse the type
        // spellings separately for leading-typename binding below.
        // MIXED args (`updateAccumBuffer<FastTraceSampleSrt, false>`):
        // neither uniform splitter accepts them, so split per-arg kinds
        // (value classes mirror `split_mangled`, types mirror
        // `split_mangled_ty`) and bind positionally below with kind
        // agreement (a type arg never binds a value param and vice
        // versa, exactly C++ [temp.deduct.call] for explicit lists).
        let (base, explicit_ty, explicit_mixed) = if explicit.is_empty() {
            match split_mangled_mixed(callee) {
                Some((b, mixed)) => (b, Vec::new(), mixed),
                None => match split_mangled_ty(callee) {
                    Some((b, tys)) => (b, tys, Vec::new()),
                    None => (base, Vec::new(), Vec::new()),
                },
            }
        } else {
            (base, Vec::new(), Vec::new())
        };
        let Some(_) = registry.get(&base) else {
            if base != *callee {
                *callee = base;
            }
            return false;
        };
        let Some(targets) = tpl_defs.get(&base) else {
            return false;
        };
        // Implicit-deduction infer tables (reference-keyed) from owned
        // clones. Built before overload selection: compatibility checks
        // infer call-arg types.
        let mut fr: HashMap<&str, Vec<((usize, usize), &Type)>> = HashMap::new();
        for (k, v) in free_rets {
            fr.insert(k.as_str(), v.iter().map(|(r, t)| (*r, t)).collect());
        }
        let mut mr: HashMap<(&str, &str), Vec<((usize, usize), &Type)>> = HashMap::new();
        for ((a, b), v) in method_rets {
            mr.insert((a.as_str(), b.as_str()), v.iter().map(|(r, t)| (*r, t)).collect());
        }
        let mut sf: HashMap<&str, HashMap<&str, (&Type, Option<u32>)>> = HashMap::new();
        for (k, v) in struct_fields {
            sf.insert(k.as_str(), v.iter().map(|(fk, (ft, fa))| (fk.as_str(), (ft, *fa))).collect());
        }
        // Whether a type mentions a template typename param (bare or
        // inside a resource carrier like `MS_Texture2D<type>`).
        fn mentions_tparam(ty: &Type, tparams: &[(String, bool)]) -> bool {
            match ty {
                Type::Named(n) => tparams.iter().any(|(t, v)| !v && t == n),
                Type::Resource { element, .. } => mentions_tparam(element, tparams),
                _ => false,
            }
        }
        // Overload selection: first definition with matching arity whose
        // concrete (non-`T`) params accept the call args (up to implicit
        // conversion). `T` positions bind during deduction below.
        // Uninferrable args skip their pair (the deduction loop
        // decides); proven mismatches reject the overload. Sole
        // overloads skip filtering (today's unchecked behavior can only
        // decline to redirect, never regress a unique candidate).
        fn overload_accepts(
            cand: &Function,
            args: &[Expr],
            env: &HashMap<String, (Type, Option<u32>)>,
            sf: &HashMap<&str, HashMap<&str, (&Type, Option<u32>)>>,
            fr: &HashMap<&str, Vec<((usize, usize), &Type)>>,
            mr: &HashMap<(&str, &str), Vec<((usize, usize), &Type)>>,
            tparams: &[(String, bool)],
        ) -> bool {
            for (i, p) in cand.parameters.iter().enumerate() {
                if matches!(&p.ty, Type::Named(n) if n == "auto") {
                    continue;
                }
                if mentions_tparam(&p.ty, tparams) {
                    continue;
                }
                let Some(arg) = args.get(i) else {
                    return false;
                };
                let Some(aty) = infer_local(arg, env, sf, fr, mr, 0) else {
                    continue;
                };
                if unify_arith(&aty, &p.ty).is_none() {
                    return false;
                }
            }
            true
        }
        let mut selected: Option<&Function> = None;
        if targets.len() == 1 {
            if args.len() == targets[0].parameters.len() {
                selected = Some(&targets[0]);
            }
        } else {
            for cand in targets {
                if args.len() != cand.parameters.len() {
                    continue;
                }
                if overload_accepts(
                    cand,
                    args,
                    env,
                    &sf,
                    &fr,
                    &mr,
                    &cand.template_params,
                ) {
                    selected = Some(cand);
                    break;
                }
            }
        }
        let Some(target) = selected else {
            return false;
        };
        let tpl_tparams = target.template_params.clone();
        // Explicit literals bind leading value params positionally.
        let mut value_bindings: Vec<(String, Expr)> = Vec::new();
        for (i, lit) in explicit.iter().enumerate() {
            let Some((pname, is_value)) = tpl_tparams.get(i) else {
                return false;
            };
            if !is_value {
                return false;
            }
            value_bindings.push((pname.clone(), lit.clone()));
        }
        let mut type_bindings: Vec<(String, Type)> = Vec::new();
        // Explicit type args (`minT<uint2>(...)`) bind leading typename
        // params positionally (mirrors value bindings above). The
        // implicit `need` loop below must agree (its consistency check
        // rejects conflicts, e.g. an arg that does not convert to the
        // named type).
        for (i, ty) in explicit_ty.iter().enumerate() {
            let Some((pname, is_value)) = tpl_tparams.get(i) else {
                return false;
            };
            if *is_value {
                return false;
            }
            if !concrete_ty(ty, structs) {
                return false;
            }
            type_bindings.push((pname.clone(), ty.clone()));
        }
        // Explicit MIXED args (`f<T, false>`) bind positionally with kind
        // agreement: a type arg binds a typename param (concrete only), a
        // value arg binds a value param; any crossing or overrun declines.
        for (i, marg) in explicit_mixed.iter().enumerate() {
            let Some((pname, is_value)) = tpl_tparams.get(i) else {
                return false;
            };
            match marg {
                MixedArg::Ty(ty) => {
                    if *is_value {
                        return false;
                    }
                    if !concrete_ty(ty, structs) {
                        return false;
                    }
                    type_bindings.push((pname.clone(), ty.clone()));
                }
                MixedArg::Val(lit) => {
                    if !is_value {
                        return false;
                    }
                    value_bindings.push((pname.clone(), lit.clone()));
                }
            }
        }
        // Implicit: typename params + auto positions from arg types.
        // (Infer tables were built above for overload selection.)
        // Deduce-me slots: (arg index, key). Typename params mentioned
        // bare in param types + auto positions. Slots already bound by
        // explicit type args above hit the consistency check below
        // (conflicts reject the call).
        let mut need: Vec<(usize, String)> = Vec::new();
        for (i, p) in target.parameters.iter().enumerate() {
            if let Type::Named(n) = &p.ty {
                if n == "auto" {
                    need.push((i, format!("auto@{i}")));
                    continue;
                }
                if tpl_tparams.iter().any(|(t, v)| !v && t == n) {
                    need.push((i, n.clone()));
                }
            }
        }
        for (i, tname) in need {
            let Some(arg) = args.get(i) else {
                return false;
            };
            let Some(ty) = infer_local(arg, env, &sf, &fr, &mr, 0) else {
                return false;
            };
            if tname == "auto" || tname.starts_with("auto@") {
                if !concrete_ty(&ty, structs) {
                    return false;
                }
                type_bindings.push((tname, ty));
                continue;
            }
            if let Type::Named(n) = &ty {
                if n == "auto" || caller_tnames.contains(n) {
                    return false;
                }
            }
            if !concrete_ty(&ty, structs) {
                return false;
            }
            if let Some(prev) = type_bindings
                .iter()
                .position(|(n, _)| n == &tname)
            {
                // Repeated typename slots must agree up to implicit
                // conversion (`minT(float2_expr, uint2_expr)` deduces
                // `T = float2`, mirroring PSSL's conversion): unify
                // instead of requiring strict equality. Unconvertible
                // pairs reject the call.
                let Some(unified) = unify_arith(&type_bindings[prev].1, &ty) else {
                    return false;
                };
                type_bindings[prev].1 = unified;
                continue;
            }
            type_bindings.push((tname, ty));
        }
        if type_bindings.is_empty() && value_bindings.is_empty() {
            if std::env::var("PSSL_MONO_DBG").or_else(|_| std::env::var("PS5_MONO_DBG")).is_ok() && callee.contains("projectPixels3") {
                eprintln!("DBG projectPixels3 no bindings");
            }
            return false;
        }
        // Canonical key. Includes the overload signature: two overloads
        // with identical bindings must not share a clone.
        let mut key = String::new();
        key.push_str(&format!("{:?}|", target.parameters));
        let mut tb = type_bindings.clone();
        tb.sort_by(|a, b| a.0.cmp(&b.0));
        for (n, t) in &tb {
            key.push_str(n);
            key.push('=');
            key.push_str(&format!("{t:?}"));
            key.push(';');
        }
        let mut vb = value_bindings.clone();
        vb.sort_by(|a, b| a.0.cmp(&b.0));
        for (n, e) in &vb {
            key.push_str(n);
            key.push('=');
            key.push_str(&format!("{e:?}"));
            key.push(';');
        }
        let cache_key = (base.clone(), key);
        if let Some(existing) = clones.get(&cache_key) {
            *callee = existing.clone();
            return true;
        }
        // Build the clone.
        let mut counter = clones.len();
        let mut cname = format!("{base}__mono{counter}");
        while taken.contains(&cname) || clones.values().any(|n| n == &cname) {
            counter += 1;
            cname = format!("{base}__mono{counter}");
        }
        let mut type_map: HashMap<String, Type> = HashMap::new();
        for (tname, ty) in &tb {
            if let Some(n) = tname.strip_prefix("auto@") {
                if let Ok(i) = n.parse::<usize>() {
                    type_map.insert(format!("auto@{i}"), ty.clone());
                    continue;
                }
            }
            type_map.insert(tname.clone(), ty.clone());
        }
        let mut type_map: HashMap<String, Type> = HashMap::new();
        for (tname, ty) in &tb {
            type_map.insert(tname.clone(), ty.clone());
        }
        let mut clone = target.clone();
        clone.name = cname.clone();
        clone.template_params = Vec::new();
        for (i, p) in clone.parameters.iter_mut().enumerate() {
            if let Type::Named(n) = &p.ty {
                if n == "auto" {
                    if let Some((_, ty)) = tb.iter().find(|(nn, _)| nn == &format!("auto@{i}")) {
                        p.ty = ty.clone();
                    }
                    continue;
                }
            }
            p.ty = subst_ty(&p.ty, &type_map);
        }
        clone.return_type = subst_ty(&clone.return_type, &type_map);
        for s in clone.body.iter_mut() {
            subst_stmt(s, &type_map);
        }
        // Propagate outer bindings into nested explicit type args
        // (template chains: a clone calling `inner<T>` with outer-bound
        // `T` respells to the concrete args for later rounds).
        rewrite_inner_targs(&mut clone.body, &type_map, registry);
        // Value bindings: synthetic value-param locals take the literal
        // init (first top-level UInt match per name).
        for (vname, lit) in &vb {
            for s in clone.body.iter_mut() {
                if let Stmt::VarDecl { ty: Type::UInt, name, init, .. } = s {
                    if name == vname {
                        *init = Some(lit.clone());
                        break;
                    }
                }
            }
            // Bound value-params are compile-time constants: substitute
            // the literal into `if` conditions (`if constexpr
            // (updateSeedBuffer)` becomes `if constexpr (false)`), so
            // the post-monomorphization const-if fold discards the
            // untaken branch. Only bare-Variable conditions naming the
            // bound param rewrite — every other use keeps the local
            // (still declared above), so nothing can dangle.
            fn subst_cond_value(
                stmts: &mut [Stmt],
                vname: &str,
                lit: &Expr,
            ) {
                for s in stmts.iter_mut() {
                    match s {
                        Stmt::If { condition, then_branch, else_branch } => {
                            if matches!(condition.as_ref(), Expr::Variable(n) if n == vname) {
                                *condition = Box::new(lit.clone());
                            }
                            subst_cond_value(then_branch, vname, lit);
                            if let Some(eb) = else_branch {
                                subst_cond_value(eb, vname, lit);
                            }
                        }
                        Stmt::Block(ss) => subst_cond_value(ss, vname, lit),
                        Stmt::For { body, .. } | Stmt::While { body, .. } => {
                            subst_cond_value(body, vname, lit)
                        }
                        Stmt::Switch { cases, .. } => {
                            for c in cases.iter_mut() {
                                subst_cond_value(&mut c.body, vname, lit);
                            }
                        }
                        Stmt::Return(_)
                        | Stmt::VarDecl { .. }
                        | Stmt::Destructure { .. }
                        | Stmt::Expr(_) => {}
                    }
                }
            }
            subst_cond_value(&mut clone.body, vname, lit);
        }
        new_functions.push(clone);
        clones.insert(cache_key, cname.clone());
        *callee = cname;
        true
    }
    #[allow(clippy::too_many_arguments)]
    fn walk_expr(
        e: &mut Expr,
        env: &HashMap<String, (Type, Option<u32>)>,
        struct_fields: &HashMap<String, HashMap<String, (Type, Option<u32>)>>,
        free_rets: &HashMap<String, Vec<((usize, usize), Type)>>,
        method_rets: &HashMap<(String, String), Vec<((usize, usize), Type)>>,
        registry: &HashMap<String, TInfo>,
        caller_tnames: &HashSet<String>,
        structs: &HashSet<String>,
        tpl_defs: &HashMap<String, Vec<Function>>,
        taken: &HashSet<String>,
        clones: &mut HashMap<(String, String), String>,
        new_functions: &mut Vec<Function>,
        progress: &mut bool,
    ) {
        match e {
            Expr::Call { function, arguments } => {
                walk_expr(function, env, struct_fields, free_rets, method_rets, registry, caller_tnames, structs, tpl_defs, taken, clones, new_functions, progress);
                for a in arguments.iter_mut() {
                    walk_expr(a, env, struct_fields, free_rets, method_rets, registry, caller_tnames, structs, tpl_defs, taken, clones, new_functions, progress);
                }
                if let Expr::Variable(callee) = function.as_mut() {
                    if try_call(callee, arguments, env, struct_fields, free_rets, method_rets, registry, caller_tnames, structs, tpl_defs, taken, clones, new_functions) {
                        *progress = true;
                    } else if callee.contains('<') {
                        if let Some((base, _)) = split_mangled(callee) {
                            if registry.contains_key(&base) || taken.contains(&base) {
                                *callee = base;
                            }
                        }
                    }
                }
            }
            Expr::Member { object, .. } => walk_expr(object, env, struct_fields, free_rets, method_rets, registry, caller_tnames, structs, tpl_defs, taken, clones, new_functions, progress),
            Expr::Index { object, index } => {
                walk_expr(object, env, struct_fields, free_rets, method_rets, registry, caller_tnames, structs, tpl_defs, taken, clones, new_functions, progress);
                walk_expr(index, env, struct_fields, free_rets, method_rets, registry, caller_tnames, structs, tpl_defs, taken, clones, new_functions, progress);
            }
            Expr::Constructor { arguments, .. } => {
                for a in arguments.iter_mut() {
                    walk_expr(a, env, struct_fields, free_rets, method_rets, registry, caller_tnames, structs, tpl_defs, taken, clones, new_functions, progress);
                }
            }
            Expr::Assign { lhs, rhs } => {
                walk_expr(lhs, env, struct_fields, free_rets, method_rets, registry, caller_tnames, structs, tpl_defs, taken, clones, new_functions, progress);
                walk_expr(rhs, env, struct_fields, free_rets, method_rets, registry, caller_tnames, structs, tpl_defs, taken, clones, new_functions, progress);
            }
            Expr::Unary { expr, .. } => walk_expr(expr, env, struct_fields, free_rets, method_rets, registry, caller_tnames, structs, tpl_defs, taken, clones, new_functions, progress),
            Expr::Binary { left, right, .. } => {
                walk_expr(left, env, struct_fields, free_rets, method_rets, registry, caller_tnames, structs, tpl_defs, taken, clones, new_functions, progress);
                walk_expr(right, env, struct_fields, free_rets, method_rets, registry, caller_tnames, structs, tpl_defs, taken, clones, new_functions, progress);
            }
            Expr::Conditional { condition, then_branch, else_branch } => {
                walk_expr(condition, env, struct_fields, free_rets, method_rets, registry, caller_tnames, structs, tpl_defs, taken, clones, new_functions, progress);
                walk_expr(then_branch, env, struct_fields, free_rets, method_rets, registry, caller_tnames, structs, tpl_defs, taken, clones, new_functions, progress);
                walk_expr(else_branch, env, struct_fields, free_rets, method_rets, registry, caller_tnames, structs, tpl_defs, taken, clones, new_functions, progress);
            }
            Expr::InitializerList(elems) => {
                for x in elems.iter_mut() {
                    walk_expr(x, env, struct_fields, free_rets, method_rets, registry, caller_tnames, structs, tpl_defs, taken, clones, new_functions, progress);
                }
            }
            Expr::Lambda { body, .. } => {
                for s in body.iter_mut() {
                    walk_stmt_top(s, env, struct_fields, free_rets, method_rets, registry, caller_tnames, structs, tpl_defs, taken, clones, new_functions, progress);
                }
            }
            Expr::Variable(_) | Expr::Number(_) | Expr::String(_) | Expr::Bool(_) => {}
        }
    }
    #[allow(clippy::too_many_arguments)]
    fn walk_stmt_top(
        s: &mut Stmt,
        env: &HashMap<String, (Type, Option<u32>)>,
        struct_fields: &HashMap<String, HashMap<String, (Type, Option<u32>)>>,
        free_rets: &HashMap<String, Vec<((usize, usize), Type)>>,
        method_rets: &HashMap<(String, String), Vec<((usize, usize), Type)>>,
        registry: &HashMap<String, TInfo>,
        caller_tnames: &HashSet<String>,
        structs: &HashSet<String>,
        tpl_defs: &HashMap<String, Vec<Function>>,
        taken: &HashSet<String>,
        clones: &mut HashMap<(String, String), String>,
        new_functions: &mut Vec<Function>,
        progress: &mut bool,
    ) {
        match s {
            Stmt::Return(Some(e)) | Stmt::Expr(e) => walk_expr(e, env, struct_fields, free_rets, method_rets, registry, caller_tnames, structs, tpl_defs, taken, clones, new_functions, progress),
            Stmt::Return(None) => {}
            Stmt::VarDecl { init: Some(e), .. }
            | Stmt::Destructure { init: Some(e), .. } => walk_expr(e, env, struct_fields, free_rets, method_rets, registry, caller_tnames, structs, tpl_defs, taken, clones, new_functions, progress),
            Stmt::VarDecl { init: None, .. } | Stmt::Destructure { init: None, .. } => {}
            Stmt::Block(ss) => {
                for x in ss.iter_mut() {
                    walk_stmt_top(x, env, struct_fields, free_rets, method_rets, registry, caller_tnames, structs, tpl_defs, taken, clones, new_functions, progress);
                }
            }
            Stmt::If { condition, then_branch, else_branch } => {
                walk_expr(condition, env, struct_fields, free_rets, method_rets, registry, caller_tnames, structs, tpl_defs, taken, clones, new_functions, progress);
                for x in then_branch.iter_mut() {
                    walk_stmt_top(x, env, struct_fields, free_rets, method_rets, registry, caller_tnames, structs, tpl_defs, taken, clones, new_functions, progress);
                }
                if let Some(els) = else_branch {
                    for x in els.iter_mut() {
                        walk_stmt_top(x, env, struct_fields, free_rets, method_rets, registry, caller_tnames, structs, tpl_defs, taken, clones, new_functions, progress);
                    }
                }
            }
            Stmt::For { init, condition, update, body } => {
                walk_stmt_top(init, env, struct_fields, free_rets, method_rets, registry, caller_tnames, structs, tpl_defs, taken, clones, new_functions, progress);
                if let Some(c) = condition {
                    walk_expr(c, env, struct_fields, free_rets, method_rets, registry, caller_tnames, structs, tpl_defs, taken, clones, new_functions, progress);
                }
                if let Some(u) = update {
                    walk_expr(u, env, struct_fields, free_rets, method_rets, registry, caller_tnames, structs, tpl_defs, taken, clones, new_functions, progress);
                }
                for x in body.iter_mut() {
                    walk_stmt_top(x, env, struct_fields, free_rets, method_rets, registry, caller_tnames, structs, tpl_defs, taken, clones, new_functions, progress);
                }
            }
            Stmt::While { condition, body } => {
                walk_expr(condition, env, struct_fields, free_rets, method_rets, registry, caller_tnames, structs, tpl_defs, taken, clones, new_functions, progress);
                for x in body.iter_mut() {
                    walk_stmt_top(x, env, struct_fields, free_rets, method_rets, registry, caller_tnames, structs, tpl_defs, taken, clones, new_functions, progress);
                }
            }
            Stmt::Switch { scrutinee, cases } => {
                walk_expr(scrutinee, env, struct_fields, free_rets, method_rets, registry, caller_tnames, structs, tpl_defs, taken, clones, new_functions, progress);
                for c in cases.iter_mut() {
                    if let Some(l) = &mut c.label {
                        walk_expr(l, env, struct_fields, free_rets, method_rets, registry, caller_tnames, structs, tpl_defs, taken, clones, new_functions, progress);
                    }
                    for x in c.body.iter_mut() {
                        walk_stmt_top(x, env, struct_fields, free_rets, method_rets, registry, caller_tnames, structs, tpl_defs, taken, clones, new_functions, progress);
                    }
                }
            }
        }
    }

    let mut module = module.clone();
    // Clone cache persists across rounds: a mangled call retargeted in
    // round N must hit the same clone in round N+1 (per-round caches
    // rebuild identical clones under new names, duplicating definitions
    // and orphaning inner-call chains). `progress` stays per-round.
    let mut clones: HashMap<(String, String), String> = HashMap::new();
    let mut new_functions: Vec<Function> = Vec::new();
    for _round in 0..8 {
        // Caller-scope constant evaluation for identifier template args
        // (`run<flags>` -> `run<16>`) before registry/try_call: fresh
        // clones from prior rounds resolve their own nested template-ids
        // (`Closest<optimizedFlags>` inside `run__mono0`) here.
        resolve_mangled_value_args(&mut module);
        let structs: HashSet<String> = module.structs.iter().map(|s| s.name.clone()).collect();
        let mut registry: HashMap<String, TInfo> = HashMap::new();
        for f in module.functions.iter() {
            if f.scope.is_some() {
                continue;
            }
            let auto_positions: Vec<usize> = f
                .parameters
                .iter()
                .enumerate()
                .filter(|(_, p)| matches!(&p.ty, Type::Named(n) if n == "auto"))
                .map(|(i, _)| i)
                .collect();
            if f.template_params.is_empty() && auto_positions.is_empty() {
                continue;
            }
            let typenames: HashSet<String> = f
                .template_params
                .iter()
                .filter(|(_, is_value)| !is_value)
                .map(|(n, _)| n.clone())
                .collect();
            if uses_qualified_param(f, &typenames) {
                continue;
            }
            registry.insert(
                f.name.clone(),
                TInfo { tparams: f.template_params.clone(), auto_positions },
            );
        }
        if registry.is_empty() {
            break;
        }
        let mut roots: HashMap<String, (Type, Option<u32>)> = HashMap::new();
        for g in &module.globals {
            roots.insert(g.name.clone(), (g.ty.clone(), g.array_size));
        }
        for r in &module.resources {
            roots.insert(
                r.name.clone(),
                (
                    Type::Resource {
                        kind: r.kind.clone(),
                        access: crate::ast::ResourceAccess::ReadOnly,
                        element: Box::new(r.element_ty.clone().unwrap_or(Type::UInt)),
                    },
                    None,
                ),
            );
        }
        let mut struct_fields: HashMap<String, HashMap<String, (Type, Option<u32>)>> =
            HashMap::new();
        for s in &module.structs {
            struct_fields.insert(
                s.name.clone(),
                s.fields.iter().map(|f| (f.name.clone(), (f.ty.clone(), f.array_size))).collect(),
            );
        }
        let mut free_rets: HashMap<String, Vec<((usize, usize), Type)>> = HashMap::new();
        let mut method_rets: HashMap<(String, String), Vec<((usize, usize), Type)>> =
            HashMap::new();
        for f in &module.functions {
            if matches!(&f.return_type, Type::Named(n) if n == "auto") {
                continue;
            }
            let total = f.parameters.len();
            let mut required = 0;
            for p in &f.parameters {
                if p.default.is_some() {
                    break;
                }
                required += 1;
            }
            let range = (required, total);
            match f.scope.as_deref() {
                Some(s) if structs.contains(s) => {
                    method_rets
                        .entry((s.to_string(), f.name.clone()))
                        .or_default()
                        .push((range, f.return_type.clone()));
                }
                _ => {
                    free_rets
                        .entry(f.name.clone())
                        .or_default()
                        .push((range, f.return_type.clone()));
                }
            }
        }
        let tpl_defs: HashMap<String, Vec<Function>> = registry
            .keys()
            .map(|name| {
                (
                    name.clone(),
                    module
                        .functions
                        .iter()
                        .filter(|f| &f.name == name)
                        .cloned()
                        .collect(),
                )
            })
            .collect();
        let taken: HashSet<String> =
            module.functions.iter().map(|f| f.name.clone()).collect();
        let caller_tnames: HashMap<String, HashSet<String>> = module
            .functions
            .iter()
            .map(|f| {
                (
                    f.name.clone(),
                    f.template_params
                        .iter()
                        .filter(|(_, v)| !v)
                        .map(|(n, _)| n.clone())
                        .collect(),
                )
            })
            .collect();
        let mut progress = false;
        for f in module.functions.iter_mut() {
            let tnames: HashSet<String> =
                caller_tnames.get(&f.name).cloned().unwrap_or_default();
            let params = f.parameters.clone();
            let mut env: HashMap<String, (Type, Option<u32>)> = HashMap::new();
            for p in &params {
                env.insert(p.name.clone(), (p.ty.clone(), p.array_size));
            }
            for g in &module.globals {
                env.entry(g.name.clone()).or_insert((g.ty.clone(), g.array_size));
            }
            for r in &module.resources {
                env.entry(r.name.clone()).or_insert((
                    Type::Resource {
                        kind: r.kind.clone(),
                        access: crate::ast::ResourceAccess::ReadOnly,
                        element: Box::new(r.element_ty.clone().unwrap_or(Type::UInt)),
                    },
                    None,
                ));
            }
            // Nested-block locals (`float3 opaqueSamples[4];` inside an
            // `if`): template-arg inference misses them with top-level
            // collection alone. Collect recursively; multiply-declared
            // names veto (flat shadowing would be unsound).
            {
                let mut counts: HashMap<String, usize> = HashMap::new();
                let mut tys: HashMap<String, (Type, Option<u32>)> = HashMap::new();
                // Every declared type per name (params + nested): names
                // declared multiple times with DIFFERENT types veto (a
                // flat map cannot scope shadowing); identical
                // re-declarations in sibling scopes (`const auto isValid
                // = InSubset(...)` per loop) map soundly to the shared
                // type.
                let mut seen_tys: HashMap<String, Vec<Type>> = HashMap::new();
                let mut note = |name: String, ty: Type, arr: Option<u32>,
                                counts: &mut HashMap<String, usize>,
                                tys: &mut HashMap<String, (Type, Option<u32>)>,
                                seen_tys: &mut HashMap<String, Vec<Type>>| {
                    *counts.entry(name.clone()).or_insert(0) += 1;
                    tys.entry(name.clone())
                        .or_insert_with(|| (ty.clone(), arr));
                    let entry = seen_tys.entry(name).or_default();
                    if !entry.contains(&ty) {
                        entry.push(ty);
                    }
                };
                for p in &params {
                    note(
                        p.name.clone(),
                        p.ty.clone(),
                        p.array_size,
                        &mut counts,
                        &mut tys,
                        &mut seen_tys,
                    );
                }
                fn collect_nested2(
                    stmts: &[Stmt],
                    counts: &mut HashMap<String, usize>,
                    tys: &mut HashMap<String, (Type, Option<u32>)>,
                    seen_tys: &mut HashMap<String, Vec<Type>>,
                ) {
                    for s in stmts {
                        match s {
                            Stmt::VarDecl { ty, name, array_size, .. } => {
                                *counts.entry(name.clone()).or_insert(0) += 1;
                                tys.entry(name.clone())
                                    .or_insert_with(|| (ty.clone(), *array_size));
                                let entry = seen_tys.entry(name.clone()).or_default();
                                if !entry.contains(ty) {
                                    entry.push(ty.clone());
                                }
                            }
                            Stmt::Destructure { ty, names, .. } => {
                                for name in names {
                                    *counts.entry(name.clone()).or_insert(0) += 1;
                                    tys.entry(name.clone())
                                        .or_insert_with(|| (ty.clone(), None));
                                    let entry = seen_tys.entry(name.clone()).or_default();
                                    if !entry.contains(ty) {
                                        entry.push(ty.clone());
                                    }
                                }
                            }
                            Stmt::Block(ss) => collect_nested2(ss, counts, tys, seen_tys),
                            Stmt::If { then_branch, else_branch, .. } => {
                                collect_nested2(then_branch, counts, tys, seen_tys);
                                if let Some(els) = else_branch {
                                    collect_nested2(els, counts, tys, seen_tys);
                                }
                            }
                            Stmt::For { init, body, .. } => {
                                collect_nested2(
                                    std::slice::from_ref(init.as_ref()),
                                    counts,
                                    tys,
                                    seen_tys,
                                );
                                collect_nested2(body, counts, tys, seen_tys);
                            }
                            Stmt::While { body, .. } => {
                                collect_nested2(body, counts, tys, seen_tys)
                            }
                            Stmt::Switch { cases, .. } => {
                                for c in cases {
                                    collect_nested2(&c.body, counts, tys, seen_tys);
                                }
                            }
                            _ => {}
                        }
                    }
                }
                collect_nested2(&f.body, &mut counts, &mut tys, &mut seen_tys);
                for name in counts.keys() {
                    let distinct = seen_tys.get(name).map(|v| v.len()).unwrap_or(0);
                    if distinct != 1 {
                        // Zero (unreachable) or ambiguous: drop any
                        // earlier (param/global) entry.
                        env.remove(name);
                    }
                }
                for (name, (ty, arr)) in tys {
                    if seen_tys.get(&name).map(|v| v.len()).unwrap_or(0) == 1 {
                        env.insert(name, (ty, arr));
                    }
                }
            }
            for s in f.body.iter_mut() {
                walk_stmt_top(
                    s, &env, &struct_fields, &free_rets, &method_rets, &registry,
                    &tnames, &structs, &tpl_defs, &taken, &mut clones, &mut new_functions,
                    &mut progress,
                );
            }
        }
        module.functions.extend(new_functions.drain(..));
        // Drop fully-retargeted template defs (no bare or mangled calls
        // remain anywhere; never the entry).
        {
            let mut uses: HashSet<String> = HashSet::new();
            for f in &module.functions {
                for s in &f.body {
                    collect_callee_names(s, &mut uses);
                }
            }
            fn collect_callee_names(s: &Stmt, out: &mut HashSet<String>) {
                match s {
                    Stmt::Return(Some(e)) | Stmt::Expr(e) => {
                        collect_callee_names_expr(e, out)
                    }
                    Stmt::Return(None) => {}
                    Stmt::VarDecl { init: Some(e), .. }
                    | Stmt::Destructure { init: Some(e), .. } => {
                        collect_callee_names_expr(e, out)
                    }
                    Stmt::VarDecl { init: None, .. }
                    | Stmt::Destructure { init: None, .. } => {}
                    Stmt::Block(ss) => {
                        for x in ss {
                            collect_callee_names(x, out);
                        }
                    }
                    Stmt::If { condition, then_branch, else_branch } => {
                        collect_callee_names_expr(condition, out);
                        for x in then_branch {
                            collect_callee_names(x, out);
                        }
                        if let Some(els) = else_branch {
                            for x in els {
                                collect_callee_names(x, out);
                            }
                        }
                    }
                    Stmt::For { init, condition, update, body } => {
                        collect_callee_names(init, out);
                        if let Some(c) = condition {
                            collect_callee_names_expr(c, out);
                        }
                        if let Some(u) = update {
                            collect_callee_names_expr(u, out);
                        }
                        for x in body {
                            collect_callee_names(x, out);
                        }
                    }
                    Stmt::While { condition, body } => {
                        collect_callee_names_expr(condition, out);
                        for x in body {
                            collect_callee_names(x, out);
                        }
                    }
                    Stmt::Switch { scrutinee, cases } => {
                        collect_callee_names_expr(scrutinee, out);
                        for c in cases {
                            if let Some(l) = &c.label {
                                collect_callee_names_expr(l, out);
                            }
                            for x in &c.body {
                                collect_callee_names(x, out);
                            }
                        }
                    }
                }
            }
            fn collect_callee_names_expr(e: &Expr, out: &mut HashSet<String>) {
                match e {
                    Expr::Call { function, arguments } => {
                        if let Expr::Variable(n) = function.as_ref() {
                            out.insert(n.clone());
                            if let Some((base, _)) = split_mangled(n) {
                                out.insert(base);
                            } else if let Some((base, _)) = split_mangled_ty(n) {
                                // Type-mangled uses (`minT<uint2>`) pin the
                                // template def exactly like literal-mangled
                                // ones; otherwise the def drops while a
                                // clone created pre-retarget still calls it.
                                out.insert(base);
                            }
                        }
                        collect_callee_names_expr(function, out);
                        for a in arguments {
                            collect_callee_names_expr(a, out);
                        }
                    }
                    Expr::Member { object, .. } => collect_callee_names_expr(object, out),
                    Expr::Index { object, index } => {
                        collect_callee_names_expr(index, out);
                        collect_callee_names_expr(object, out);
                    }
                    Expr::Constructor { arguments, .. } => {
                        for a in arguments {
                            collect_callee_names_expr(a, out);
                        }
                    }
                    Expr::Assign { lhs, rhs } => {
                        collect_callee_names_expr(lhs, out);
                        collect_callee_names_expr(rhs, out);
                    }
                    Expr::Unary { expr, .. } => collect_callee_names_expr(expr, out),
                    Expr::Binary { left, right, .. } => {
                        collect_callee_names_expr(left, out);
                        collect_callee_names_expr(right, out);
                    }
                    Expr::Conditional { condition, then_branch, else_branch } => {
                        collect_callee_names_expr(condition, out);
                        collect_callee_names_expr(then_branch, out);
                        collect_callee_names_expr(else_branch, out);
                    }
                    Expr::InitializerList(elems) => {
                        for x in elems {
                            collect_callee_names_expr(x, out);
                        }
                    }
                    Expr::Lambda { body, .. } => {
                        for s in body {
                            collect_callee_names(s, out);
                        }
                    }
                    Expr::Variable(_)
                    | Expr::Number(_)
                    | Expr::String(_)
                    | Expr::Bool(_) => {}
                }
            }
            let entry = entry_name.map(|s| s.to_string());
            let drop: Vec<String> = if std::env::var("PSSL_NO_DROP").or_else(|_| std::env::var("PS5_NO_DROP")).is_ok() {
                vec![]
            } else {
            registry
                .keys()
                .filter(|name| {
                    Some(*name) != entry.as_ref()
                        && !uses.contains(*name)
                        && !uses.iter().any(|u| {
                            split_mangled(u).is_some_and(|(b, _)| &b == *name)
                                || split_mangled_ty(u).is_some_and(|(b, _)| &b == *name)
                        })
                })
                .cloned()
                .collect()
            };
            for name in drop {
                if std::env::var("PSSL_MONO_DBG").or_else(|_| std::env::var("PS5_MONO_DBG")).is_ok() && name.contains("rojectPixels3") {
                    eprintln!("DBG drop {name}");
                }
                if let Some(pos) = module.functions.iter().position(|f| &f.name == &name) {
                    module.functions.remove(pos);
                }
            }
        }
        if !progress {
            break;
        }
    }
    module
}

/// Infers an expression type where cheap and unambiguous; `None` otherwise.
fn infer(e: &Expr, vars: &HashMap<String, Type>) -> Option<Type> {
    match e {
        Expr::Variable(name) => vars.get(name).cloned(),
        Expr::Constructor { ty, .. } => Some(ty.clone()),
        Expr::Number(value) => {
            // Mirrors `semantic`: `0` -> Int, `0.0`/`0f` -> Float.
            let is_float = value.contains('.')
                || value.contains('e')
                || value.contains('E')
                || value.ends_with('f')
                || value.ends_with('F');
            Some(if is_float { Type::Float } else { Type::Int })
        }
        Expr::Bool(_) => Some(Type::Bool),
        _ => None,
    }
}
