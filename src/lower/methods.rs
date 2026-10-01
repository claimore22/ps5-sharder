//! Struct-method reachability stripping.
//!
//! Only methods reachable from entry + free functions survive lowering.
//! Uncalled methods stay dropped (as before struct methods existed):
//! their bodies would otherwise emit dead code that regresses passing
//! shaders on latent issues, and their references pollute hoisting
//! decisions (pinning resource fields that must be stripped).
//!
//! Only methods in *declared* structs are gated; undeclared-scope
//! functions keep free behavior below, exactly as before.

use std::collections::{HashMap, HashSet};

use crate::ir::{IrExpr, IrStmt, IrType, ShaderModule};

/// Removes struct methods unreachable from entry + free functions.
pub(crate) fn strip_unreachable_methods(ir: &mut ShaderModule) {
    let declared: HashSet<&str> = ir.structs.iter().map(|s| s.name.as_str()).collect();
    let reachable = crate::ir::reachable_methods(ir);
    ir.functions.retain(|f| match f.scope.as_deref() {
        Some(s) if declared.contains(s) => reachable.contains(&(s.to_string(), f.name.clone())),
        _ => true,
    });
}

/// Removes free functions unreachable from the entry point.
///
/// Uncalled header baggage (vectormath double-precision helpers over
/// `vec_ullongN`, unused encoder variants) would otherwise emit dead
/// definitions whose latent body issues fail DXC — the same principle as
/// method stripping above. Reachability is name-based over the call
/// graph (fixpoint from entry; overloads share a name and are kept
/// together). Call sources are live-named free functions plus every
/// surviving scoped method (survivors are reachable by construction —
/// a method-only call chain like `main → obj.decodeNormal() →
/// decodeSnorm()` must keep its free tail). Scoped methods are exempt
/// from removal here (gated by `strip_unreachable_methods` instead);
/// unscoped and unknown-scope functions share free behavior (mirrors
/// the emitter). Without an entry (broken shaders) everything is kept:
/// stripping must never mask the real failure with an emptied module.
///
/// Must run in a fixpoint loop with `strip_unreachable_methods` (see
/// `lower_module`): each pass shrinks the other's roots until neither
/// removes anything.
pub(crate) fn strip_unreachable_free_functions(ir: &mut ShaderModule) {
    let Some(entry) = ir.entry_point.clone() else {
        return;
    };
    let declared: HashSet<&str> = ir.structs.iter().map(|s| s.name.as_str()).collect();
    let is_free = |f: &crate::ir::Function| {
        f.scope.is_none() || f.scope.as_deref().is_some_and(|s| !declared.contains(s))
    };
    let free_names: HashSet<&str> = ir
        .functions
        .iter()
        .filter(|f| is_free(f))
        .map(|f| f.name.as_str())
        .collect();
    // Callee names that resolve to a free function: bare calls, plus
    // `Scope::f` where Scope is not a declared struct (a method call
    // otherwise). Dotted `obj.m` calls are always member calls.
    let mark = |target: &str, live: &mut HashSet<String>| {
        if let Some(pos) = target.rfind("::") {
            let (scope, name) = target.split_at(pos);
            let name = &name[2..];
            if !declared.contains(scope) && free_names.contains(name) {
                live.insert(name.to_string());
            }
        } else if !target.contains('.') && free_names.contains(target) {
            live.insert(target.to_string());
        }
    };
    let mut live: HashSet<String> = HashSet::from([entry.clone()]);
    // Hull patch-constant functions are referenced only by the entry's
    // `[patchconstantfunc("F")]` attribute string (never called): pin
    // them live or the entry dangles (same class as entry itself).
    for f in &ir.functions {
        if f.name == entry {
            if let Some(patch) = f.hull_patch_func.as_deref() {
                if free_names.contains(patch) {
                    live.insert(patch.to_string());
                }
            }
        }
    }
    let mut changed = true;
    while changed {
        changed = false;
        for f in &ir.functions {
            // Live-named free functions plus every surviving scoped
            // method (reachable by construction — see above).
            let source = match f.scope.as_deref() {
                None => live.contains(&f.name),
                Some(s) if !declared.contains(s) => live.contains(&f.name),
                Some(_) => true,
            };
            if !source {
                continue;
            }
            for target in crate::ir::function_calls(f) {
                let before = live.len();
                mark(&target, &mut live);
                changed |= live.len() != before;
            }
        }
    }
    ir.functions.retain(|f| {
        if !is_free(f) {
            return true;
        }
        live.contains(&f.name)
    });
}

/// Appends implicit binding parameters to struct methods.
///
/// A method body may reference module bindings (hoisted `__hoist_`
/// globals, plain globals, cbuffer fields, resources) that emit *after*
/// structs, so `obj.m()` would read undeclared identifiers. Each such
/// reference becomes an extra parameter (shadowing the global with the
/// same value passed at call sites, where bindings are visible).
/// Locals, params, and the scope struct's own members never become
/// extras. Runs after hoisting (final global set), before emission.
pub(crate) fn method_binding_params(ir: &mut ShaderModule) {
    // Name -> type for globals, cbuffer fields (first wins on collision;
    // ambiguous names are skipped), and resources.
    let mut binding_tys: HashMap<String, IrType> = HashMap::new();
    let mut ambiguous: HashSet<String> = HashSet::new();
    let mut declare = |name: String, ty: IrType| {
        if binding_tys.contains_key(&name) {
            ambiguous.insert(name);
        } else {
            binding_tys.insert(name, ty);
        }
    };
    for g in &ir.globals {
        declare(g.name.clone(), g.ty.clone());
    }
    for r in &ir.resources {
        if r.resource_type == crate::ir::ResourceType::ConstantBuffer {
            for f in &r.fields {
                declare(f.name.clone(), f.ty.clone());
            }
        } else {
            let ty = IrType::Resource {
                kind: r.kind.clone(),
                access: match r.resource_type {
                    crate::ir::ResourceType::RWBuffer | crate::ir::ResourceType::RWTexture => {
                        crate::ir::IrAccess::ReadWrite
                    }
                    _ => crate::ir::IrAccess::ReadOnly,
                },
                element: Box::new(
                    r.element_type
                        .clone()
                        .unwrap_or(IrType::Named("unknown".to_string())),
                ),
            };
            declare(r.name.clone(), ty);
        }
    }
    for name in std::mem::take(&mut ambiguous) {
        binding_tys.remove(&name);
    }

    // Struct member names per struct (excluded: `this`-relative).
    let mut members: HashMap<&str, HashSet<&str>> = HashMap::new();
    for s in &ir.structs {
        for f in &s.fields {
            members.entry(s.name.as_str()).or_default().insert(f.name.as_str());
        }
    }

    // Per-function type envs (params + local VarDecls; globals consulted
    // separately so shadowing stays correct).
    let tenvs: Vec<HashMap<String, IrType>> = ir
        .functions
        .iter()
        .map(|f| {
            let mut env: HashMap<String, IrType> = HashMap::new();
            for p in &f.parameters {
                env.insert(p.name.clone(), p.ty.clone());
            }
            collect_local_tys(&f.body, &mut env);
            env
        })
        .collect();

    // Direct extras per method index (struct methods only).
    let mut extras: HashMap<usize, Vec<String>> = HashMap::new();
    for (idx, f) in ir.functions.iter().enumerate() {
        let Some(scope) = f.scope.as_deref() else {
            continue;
        };
        let mut names = Vec::new();
        collect_variables(&f.body, &mut names);
        let mut seen = HashSet::new();
        for name in names {
            if !binding_tys.contains_key(&name) {
                continue;
            }
            if tenvs[idx].contains_key(&name) {
                continue;
            }
            if members
                .get(scope)
                .is_some_and(|fields| fields.contains(name.as_str()))
            {
                continue;
            }
            if seen.insert(name.clone()) {
                extras.entry(idx).or_default().push(name);
            }
        }
    }

    // Fixpoint: methods also need the extras of the methods they call
    // (to pass them along; globals are visible at the call site).
    // Only methods carry extras; free functions forward without declaring.
    loop {
        let mut changed = false;
        // (struct, name, pre-extras arity) -> extras snapshot.
        let table = extras_table(ir, &extras);
        for (idx, f) in ir.functions.iter().enumerate() {
            if f.scope.is_none() {
                continue;
            }
            for (scope, name, arity) in method_callees(f, ir, &tenvs[idx]) {
                if let Some(extra) = table.get(&(scope, name, arity)) {
                    for g in extra.clone() {
                        if !extras.entry(idx).or_default().contains(&g) {
                            extras.entry(idx).or_default().push(g);
                            changed = true;
                        }
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }

    // Drop extras whose types would not survive HLSL emission (unmapped
    // `sce::` names): such a param creates an earlier, louder error than
    // the forward reference it was meant to fix — and could break a
    // passing shader. Skipped extras behave as pre-extras (loud).
    for names in extras.values_mut() {
        names.retain(|g| {
            binding_tys
                .get(g)
                .is_some_and(|ty| emittable_type(ty))
        });
    }

    // Rewrite calls: dotted `obj.m(args)` with a resolved struct and bare
    // sibling calls gain the callee's extras. Unresolvable shapes and
    // `S::m` stay untouched (loud). Slices (not `&ir`) so the mutable
    // function walk below compiles.
    let table = extras_table(ir, &extras);
    for (idx, f) in ir.functions.iter_mut().enumerate() {
        let scope = f.scope.clone();
        let env = tenvs[idx].clone();
        for stmt in &mut f.body {
            rewrite_method_calls(
                stmt,
                scope.as_deref(),
                &env,
                &ir.structs,
                &ir.globals,
                &table,
            );
        }
    }

    // Append the params (shadowing the globals with passed values).
    let table = extras_table(ir, &extras);
    for (idx, f) in ir.functions.iter_mut().enumerate() {
        if f.scope.is_none() {
            continue;
        }
        if let Some(names) = extras.get(&idx) {
            for gname in names {
                if f.parameters.iter().any(|p| &p.name == gname) {
                    continue;
                }
                if let Some(ty) = binding_tys.get(gname).cloned() {
                    f.parameters.push(crate::ir::function::Parameter {
                        ty,
                        name: gname.clone(),
                        semantic: None,
                        array_size: None,
                        direction: None,
                        gs_primitive: None,
                        default: None,
                        is_uniform: false,
                        mesh_object: None,
                    });
                }
            }
        }
    }
    let _ = table;
}

/// Whether a binding type survives HLSL emission without leaking
/// `sce::` framework names (modulo the mapped ones: `SamplerState`,
/// `uint`-backed enums via `framework_type`).
fn emittable_type(ty: &IrType) -> bool {
    match ty {
        IrType::Named(n) => {
            !n.starts_with("sce::")
                || n == "sce::Agc::Core::Sampler"
                || crate::framework::framework_type(n).is_some()
        }
        IrType::Resource { element, .. } => emittable_type(element),
        _ => true,
    }
}

/// (struct, method, pre-extras arity) -> extras snapshot helper.
fn extras_table(
    ir: &ShaderModule,
    extras: &HashMap<usize, Vec<String>>,
) -> HashMap<(String, String, usize), Vec<String>> {
    let mut table = HashMap::new();
    for (&idx, names) in extras {
        if let Some(f) = ir.functions.get(idx) {
            if let Some(scope) = f.scope.as_deref() {
                // Pre-extras arity = current params minus extras already
                // appended (none appended yet at fixpoint time; at rewrite
                // time extras are still unappended — same count).
                table.insert(
                    (scope.to_string(), f.name.clone(), f.parameters.len()),
                    names.clone(),
                );
            }
        }
    }
    table
}

/// Collects every `Variable` name in statements.
fn collect_variables(stmts: &[IrStmt], out: &mut Vec<String>) {
    fn expr(e: &IrExpr, out: &mut Vec<String>) {
        match e {
            IrExpr::Variable(v) => out.push(v.clone()),
            IrExpr::Literal(_) => {}
            IrExpr::Member { object, .. } => expr(object, out),
            IrExpr::Index { object, index } | IrExpr::ResourceIndex { resource: object, index } => {
                expr(object, out);
                expr(index, out);
            }
            IrExpr::ResourceLoad {
                resource,
                coordinates,
                sample,
            } => {
                expr(resource, out);
                expr(coordinates, out);
                if let Some(s) = sample {
                    expr(s, out);
                }
            }
            IrExpr::Call { arguments, .. } => {
                for a in arguments {
                    expr(a, out);
                }
            }
            IrExpr::Construct { arguments, .. } => {
                for a in arguments {
                    expr(a, out);
                }
            }
            IrExpr::Unary { expr: inner, .. } => expr(inner, out),
            IrExpr::Binary { left, right, .. } => {
                expr(left, out);
                expr(right, out);
            }
            IrExpr::Conditional {
                condition,
                then_branch,
                else_branch,
            } => {
                expr(condition, out);
                expr(then_branch, out);
                expr(else_branch, out);
            }
            IrExpr::Assign { lhs, rhs } => {
                expr(lhs, out);
                expr(rhs, out);
            }
            IrExpr::InitializerList(elems) => {
                for x in elems {
                    expr(x, out);
                }
            }
        }
    }
    fn stmt(s: &IrStmt, out: &mut Vec<String>) {
        match s {
            IrStmt::Return(Some(e)) | IrStmt::Expr(e) => expr(e, out),
            IrStmt::Return(None) => {}
            IrStmt::VarDecl { init: Some(e), .. } => expr(e, out),
            IrStmt::VarDecl { init: None, .. } => {}
            IrStmt::If {
                condition,
                then_branch,
                else_branch,
            } => {
                expr(condition, out);
                for s in then_branch {
                    stmt(s, out);
                }
                if let Some(b) = else_branch {
                    for s in b {
                        stmt(s, out);
                    }
                }
            }
            IrStmt::For {
                init,
                condition,
                update,
                body,
            } => {
                stmt(init, out);
                if let Some(c) = condition {
                    expr(c, out);
                }
                if let Some(u) = update {
                    expr(u, out);
                }
                for s in body {
                    stmt(s, out);
                }
            }
            IrStmt::While { condition, body } => {
                expr(condition, out);
                for s in body {
                    stmt(s, out);
                }
            }
            IrStmt::Switch { scrutinee, cases } => {
                expr(scrutinee, out);
                for c in cases {
                    if let Some(label) = &c.label {
                        expr(label, out);
                    }
                    for s in &c.body {
                        stmt(s, out);
                    }
                }
            }
            IrStmt::Block(ss) => {
                for s in ss {
                    stmt(s, out);
                }
            }
        }
    }
    for s in stmts {
        stmt(s, out);
    }
}

/// Collects `(name, type)` of every `VarDecl` in statements.
fn collect_local_tys(stmts: &[IrStmt], out: &mut HashMap<String, IrType>) {
    fn stmt(s: &IrStmt, out: &mut HashMap<String, IrType>) {
        match s {
            IrStmt::VarDecl { ty, name, .. } => {
                out.entry(name.clone()).or_insert_with(|| ty.clone());
            }
            IrStmt::Block(ss) => {
                for s in ss {
                    stmt(s, out);
                }
            }
            IrStmt::If {
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
            IrStmt::For { init, body, .. } => {
                stmt(init, out);
                for s in body {
                    stmt(s, out);
                }
            }
            IrStmt::While { body, .. } => {
                for s in body {
                    stmt(s, out);
                }
            }
            IrStmt::Switch { cases, .. } => {
                for c in cases {
                    for s in &c.body {
                        stmt(s, out);
                    }
                }
            }
            _ => {}
        }
    }
    for s in stmts {
        stmt(s, out);
    }
}

/// (overloads by type are rare; arity distinguishes the rest).
fn find_method(ir: &ShaderModule, scope: &str, name: &str, arity: usize) -> Option<usize> {
    ir.functions.iter().position(|f| {
        f.scope.as_deref() == Some(scope) && f.name == name && f.parameters.len() == arity
    })
}

/// Resolved method calls in a function body: (struct, name, arity).
/// Dotted calls resolve the object struct; bare calls in methods resolve
/// to siblings; `S::m` resolves exactly; bare calls in free functions and
/// unresolvable shapes resolve to nothing (loud).
fn method_callees(
    f: &crate::ir::Function,
    ir: &ShaderModule,
    env: &HashMap<String, IrType>,
) -> Vec<(String, String, usize)> {
    fn expr(
        e: &IrExpr,
        scope: Option<&str>,
        env: &HashMap<String, IrType>,
        ir: &ShaderModule,
        out: &mut Vec<(String, String, usize)>,
    ) {
        match e {
            IrExpr::Call { function, arguments } => {
                if let Some(dot) = function.rfind('.') {
                    let method = &function[dot + 1..];
                    if !method.is_empty() {
                        let obj_str = &function[..dot];
                        let mut segs = obj_str.split('.');
                        let root = segs.next().unwrap_or("");
                        let mut cur = env.get(root).cloned().or_else(|| {
                            ir.globals.iter().find(|g| g.name == root).map(|g| g.ty.clone())
                        });
                        for seg in segs {
                            let seg = seg.trim_end_matches(|c: char| {
                                c == ']' || c == '[' || c.is_numeric()
                            });
                            let Some(IrType::Named(s)) = cur else {
                                cur = None;
                                break;
                            };
                            cur = ir
                                .structs
                                .iter()
                                .find(|d| d.name == s)
                                .and_then(|d| d.fields.iter().find(|f| f.name == seg))
                                .map(|f| f.ty.clone());
                        }
                        if let Some(IrType::Named(s)) = cur {
                            if find_method(ir, &s, method, arguments.len()).is_some() {
                                out.push((s, method.to_string(), arguments.len()));
                            }
                        }
                    }
                } else if let Some(pos) = function.rfind("::") {
                    let (sc, m) = function.split_at(pos);
                    let m = &m[2..];
                    if find_method(ir, sc, m, arguments.len()).is_some() {
                        out.push((sc.to_string(), m.to_string(), arguments.len()));
                    }
                } else if let Some(scope) = scope {
                    if find_method(ir, scope, function, arguments.len()).is_some() {
                        out.push((scope.to_string(), function.clone(), arguments.len()));
                    }
                }
                for a in arguments {
                    expr(a, scope, env, ir, out);
                }
            }
            IrExpr::Member { object, .. } => expr(object, scope, env, ir, out),
            IrExpr::Index { object, index } | IrExpr::ResourceIndex { resource: object, index } => {
                expr(object, scope, env, ir, out);
                expr(index, scope, env, ir, out);
            }
            IrExpr::ResourceLoad {
                resource,
                coordinates,
                sample,
            } => {
                expr(resource, scope, env, ir, out);
                expr(coordinates, scope, env, ir, out);
                if let Some(s) = sample {
                    expr(s, scope, env, ir, out);
                }
            }
            IrExpr::Construct { arguments, .. } => {
                for a in arguments {
                    expr(a, scope, env, ir, out);
                }
            }
            IrExpr::Unary { expr: inner, .. } => expr(inner, scope, env, ir, out),
            IrExpr::Binary { left, right, .. } => {
                expr(left, scope, env, ir, out);
                expr(right, scope, env, ir, out);
            }
            IrExpr::Conditional {
                condition,
                then_branch,
                else_branch,
            } => {
                expr(condition, scope, env, ir, out);
                expr(then_branch, scope, env, ir, out);
                expr(else_branch, scope, env, ir, out);
            }
            IrExpr::Assign { lhs, rhs } => {
                expr(lhs, scope, env, ir, out);
                expr(rhs, scope, env, ir, out);
            }
            IrExpr::InitializerList(elems) => {
                for x in elems {
                    expr(x, scope, env, ir, out);
                }
            }
            IrExpr::Variable(_) | IrExpr::Literal(_) => {}
        }
    }
    fn stmt(
        s: &IrStmt,
        scope: Option<&str>,
        env: &HashMap<String, IrType>,
        ir: &ShaderModule,
        out: &mut Vec<(String, String, usize)>,
    ) {
        match s {
            IrStmt::Return(Some(e)) | IrStmt::Expr(e) => expr(e, scope, env, ir, out),
            IrStmt::Return(None) => {}
            IrStmt::VarDecl { init: Some(e), .. } => expr(e, scope, env, ir, out),
            IrStmt::VarDecl { init: None, .. } => {}
            IrStmt::If {
                condition,
                then_branch,
                else_branch,
            } => {
                expr(condition, scope, env, ir, out);
                for s in then_branch {
                    stmt(s, scope, env, ir, out);
                }
                if let Some(b) = else_branch {
                    for s in b {
                        stmt(s, scope, env, ir, out);
                    }
                }
            }
            IrStmt::For {
                init,
                condition,
                update,
                body,
            } => {
                stmt(init, scope, env, ir, out);
                if let Some(c) = condition {
                    expr(c, scope, env, ir, out);
                }
                if let Some(u) = update {
                    expr(u, scope, env, ir, out);
                }
                for s in body {
                    stmt(s, scope, env, ir, out);
                }
            }
            IrStmt::While { condition, body } => {
                expr(condition, scope, env, ir, out);
                for s in body {
                    stmt(s, scope, env, ir, out);
                }
            }
            IrStmt::Switch { scrutinee, cases } => {
                expr(scrutinee, scope, env, ir, out);
                for c in cases {
                    if let Some(label) = &c.label {
                        expr(label, scope, env, ir, out);
                    }
                    for s in &c.body {
                        stmt(s, scope, env, ir, out);
                    }
                }
            }
            IrStmt::Block(ss) => {
                for s in ss {
                    stmt(s, scope, env, ir, out);
                }
            }
        }
    }
    let mut out = Vec::new();
    for s in &f.body {
        stmt(s, f.scope.as_deref(), env, ir, &mut out);
    }
    out
}

/// Rewrites method calls in statements, appending callee extras:
/// dotted `obj.m(args)` with a resolved struct and bare sibling calls.
/// Runs after extras are final.
#[allow(clippy::too_many_arguments)]
fn rewrite_method_calls(
    root: &mut IrStmt,
    scope: Option<&str>,
    env: &HashMap<String, IrType>,
    structs: &[crate::ir::StructDecl],
    globals: &[crate::ir::GlobalVar],
    table: &HashMap<(String, String, usize), Vec<String>>,
) {
    fn expr(
        e: &mut IrExpr,
        scope: Option<&str>,
        env: &HashMap<String, IrType>,
        structs: &[crate::ir::StructDecl],
        globals: &[crate::ir::GlobalVar],
        table: &HashMap<(String, String, usize), Vec<String>>,
    ) {
        // Post-order: arguments first.
        match e {
            IrExpr::Call { arguments, .. } => {
                for a in arguments.iter_mut() {
                    expr(a, scope, env, structs, globals, table);
                }
            }
            IrExpr::Member { object, .. } => expr(object, scope, env, structs, globals, table),
            IrExpr::Index { object, index } | IrExpr::ResourceIndex { resource: object, index } => {
                expr(object, scope, env, structs, globals, table);
                expr(index, scope, env, structs, globals, table);
            }
            IrExpr::ResourceLoad {
                resource,
                coordinates,
                sample,
            } => {
                expr(resource, scope, env, structs, globals, table);
                expr(coordinates, scope, env, structs, globals, table);
                if let Some(s) = sample {
                    expr(s, scope, env, structs, globals, table);
                }
            }
            IrExpr::Construct { arguments, .. } => {
                for a in arguments {
                    expr(a, scope, env, structs, globals, table);
                }
            }
            IrExpr::Unary { expr: inner, .. } => expr(inner, scope, env, structs, globals, table),
            IrExpr::Binary { left, right, .. } => {
                expr(left, scope, env, structs, globals, table);
                expr(right, scope, env, structs, globals, table);
            }
            IrExpr::Conditional {
                condition,
                then_branch,
                else_branch,
            } => {
                expr(condition, scope, env, structs, globals, table);
                expr(then_branch, scope, env, structs, globals, table);
                expr(else_branch, scope, env, structs, globals, table);
            }
            IrExpr::Assign { lhs, rhs } => {
                expr(lhs, scope, env, structs, globals, table);
                expr(rhs, scope, env, structs, globals, table);
            }
            IrExpr::InitializerList(elems) => {
                for x in elems {
                    expr(x, scope, env, structs, globals, table);
                }
            }
            IrExpr::Variable(_) | IrExpr::Literal(_) => {}
        }
        // Then the call itself.
        let (callee, arity) = match e {
            IrExpr::Call { function, arguments } => (function.clone(), arguments.len()),
            _ => return,
        };
        let resolved: Option<(String, String)> = if let Some(dot) = callee.rfind('.') {
            let method = callee[dot + 1..].to_string();
            if method.is_empty() {
                None
            } else {
                let obj_str = callee[..dot].to_string();
                let mut segs = obj_str.split('.');
                let root = segs.next().unwrap_or("").to_string();
                if root.is_empty() {
                    None
                } else {
                    let mut cur = env.get(&root).cloned().or_else(|| {
                        globals.iter().find(|g| g.name == root).map(|g| g.ty.clone())
                    });
                    let mut ok = true;
                    for seg in segs {
                        let seg = seg.trim_end_matches(|c: char| {
                            c == ']' || c == '[' || c.is_numeric()
                        });
                        match cur {
                            Some(IrType::Named(s)) => {
                                cur = structs
                                    .iter()
                                    .find(|d| d.name == s)
                                    .and_then(|d| d.fields.iter().find(|f| f.name == seg))
                                    .map(|f| f.ty.clone());
                            }
                            _ => {
                                ok = false;
                                break;
                            }
                        }
                    }
                    match cur {
                        Some(IrType::Named(s)) if ok => Some((s, method)),
                        _ => None,
                    }
                }
            }
        } else if callee.contains("::") {
            None
        } else if let Some(scope) = scope {
            Some((scope.to_string(), callee))
        } else {
            None
        };
        let Some((sname, mname)) = resolved else {
            return;
        };
        if let IrExpr::Call { arguments, .. } = e {
            if let Some(extra) = table.get(&(sname, mname, arguments.len())) {
                for g in extra {
                    arguments.push(IrExpr::Variable(g.clone()));
                }
            }
        }
    }
    fn stmt(
        s: &mut IrStmt,
        scope: Option<&str>,
        env: &HashMap<String, IrType>,
        structs: &[crate::ir::StructDecl],
        globals: &[crate::ir::GlobalVar],
        table: &HashMap<(String, String, usize), Vec<String>>,
    ) {
        match s {
            IrStmt::Return(Some(e)) | IrStmt::Expr(e) => expr(e, scope, env, structs, globals, table),
            IrStmt::Return(None) => {}
            IrStmt::VarDecl { init: Some(e), .. } => expr(e, scope, env, structs, globals, table),
            IrStmt::VarDecl { init: None, .. } => {}
            IrStmt::If {
                condition,
                then_branch,
                else_branch,
            } => {
                expr(condition, scope, env, structs, globals, table);
                for s in then_branch {
                    stmt(s, scope, env, structs, globals, table);
                }
                if let Some(b) = else_branch {
                    for s in b {
                        stmt(s, scope, env, structs, globals, table);
                    }
                }
            }
            IrStmt::For {
                init,
                condition,
                update,
                body,
            } => {
                stmt(init, scope, env, structs, globals, table);
                if let Some(c) = condition {
                    expr(c, scope, env, structs, globals, table);
                }
                if let Some(u) = update {
                    expr(u, scope, env, structs, globals, table);
                }
                for s in body {
                    stmt(s, scope, env, structs, globals, table);
                }
            }
            IrStmt::While { condition, body } => {
                expr(condition, scope, env, structs, globals, table);
                for s in body {
                    stmt(s, scope, env, structs, globals, table);
                }
            }
            IrStmt::Switch { scrutinee, cases } => {
                expr(scrutinee, scope, env, structs, globals, table);
                for c in cases {
                    if let Some(label) = &mut c.label {
                        expr(label, scope, env, structs, globals, table);
                    }
                    for s in &mut c.body {
                        stmt(s, scope, env, structs, globals, table);
                    }
                }
            }
            IrStmt::Block(ss) => {
                for s in ss {
                    stmt(s, scope, env, structs, globals, table);
                }
            }
        }
    }
    stmt(root, scope, env, structs, globals, table);
}

/// Inlines single-return struct methods called on removed roots.
///
/// SRT flattening elides helper params (`RenderMeshSRT* pSrt`) and rewrites
/// `pSrt.member` paths to flattened bindings, but `pSrt.getX()` method calls
/// keep a dangling object: the param is gone, so `rewrite_method_calls`
/// cannot resolve the struct and never appends the callee's binding extras.
/// The call stays `pSrt.getTransform()` with 0 args while the method needs 1,
/// emitting `use of undeclared identifier 'pSrt'`.
///
/// Such calls currently fail loudly at DXC, so replacing them is
/// zero-regression by construction (passing shaders have no dangling roots
/// and are untouched). Only safe shapes inline:
/// * the object root is undeclared in the caller (not a param/local/global/
 /// resource/cbuffer field),
/// * exactly one declared struct provides the method name (no ambiguity),
/// * the method body is a single `return expr` (getters like
///   `getGeometry`/`getTransform`/`getMaterial`),
/// * the call supplies the original params and every extra param names a
///   module binding (extras shadow globals, so the inlined body references
///   the globals directly).
/// Multi-statement methods (e.g. `GeometrySRT::getVertex`) stay loud.
pub(crate) fn inline_dangling_root_methods(ir: &mut ShaderModule) {
    use std::collections::{HashMap, HashSet};

    // Module bindings visible as bare names: globals, resources, cbuffer fields.
    let mut bindings: HashSet<String> = HashSet::new();
    for g in &ir.globals {
        bindings.insert(g.name.clone());
    }
    for r in &ir.resources {
        bindings.insert(r.name.clone());
        if r.resource_type == crate::ir::ResourceType::ConstantBuffer {
            for f in &r.fields {
                bindings.insert(f.name.clone());
            }
        }
    }

    // Method name -> declaring scopes (for uniqueness).
    let mut by_name: HashMap<String, Vec<(String, usize)>> = HashMap::new();
    for (idx, f) in ir.functions.iter().enumerate() {
        if let Some(scope) = f.scope.as_deref() {
            by_name
                .entry(f.name.clone())
                .or_default()
                .push((scope.to_string(), idx));
        }
    }

    // Snapshot method bodies/params (indices stay stable; we only rewrite
    // free-function and helper bodies, never add/remove functions here).
    for fi in 0..ir.functions.len() {
        // Declared names in this caller: params + locals + bindings.
        let mut declared: HashSet<String> = bindings.clone();
        {
            let f = &ir.functions[fi];
            // Methods themselves keep `this`-scope; only free/helper bodies
            // can hold dangling removed roots. Scoped bodies reference
            // members bare, which would false-positive as dangling.
            if f.scope.is_some() {
                continue;
            }
            for p in &f.parameters {
                declared.insert(p.name.clone());
            }
            let mut env: HashMap<String, IrType> = HashMap::new();
            collect_local_tys(&f.body, &mut env);
            for k in env.keys() {
                declared.insert(k.clone());
            }
        }
        // Collect candidate calls via mutable walk (post-order, fixpoint for
        // chains like `pSrt.getGeometry()` feeding `geometry.getVertex()` —
        // the latter is not dangling, so one pass suffices, but loop anyway).
        for _ in 0..4 {
            let changed = inline_dangling_in_fn(ir, fi, &declared, &by_name, &bindings);
            if !changed {
                break;
            }
        }
    }
}

fn inline_dangling_in_fn(
    ir: &mut ShaderModule,
    fi: usize,
    declared: &std::collections::HashSet<String>,
    by_name: &std::collections::HashMap<String, Vec<(String, usize)>>,
    bindings: &std::collections::HashSet<String>,
) -> bool {
    // Clone the method table snapshot for this pass (function bodies change
    // below, but method definitions themselves are only read).
    let methods: Vec<crate::ir::Function> = ir.functions.clone();
    let mut changed = false;
    // Locals can shadow: refresh declared with any new VarDecls? Bodies only
    // shrink (calls -> exprs), no new decls, so the outer set stays valid.
    let bodies = std::mem::take(&mut ir.functions[fi].body);
    let mut bodies = bodies;
    for stmt in &mut bodies {
        changed |= inline_dangling_in_stmt(stmt, declared, by_name, bindings, &methods);
    }
    ir.functions[fi].body = bodies;
    changed
}

fn inline_dangling_in_stmt(
    stmt: &mut IrStmt,
    declared: &std::collections::HashSet<String>,
    by_name: &std::collections::HashMap<String, Vec<(String, usize)>>,
    bindings: &std::collections::HashSet<String>,
    methods: &[crate::ir::Function],
) -> bool {
    match stmt {
        IrStmt::Return(Some(e)) | IrStmt::Expr(e) => {
            inline_dangling_in_expr(e, declared, by_name, bindings, methods)
        }
        IrStmt::Return(None) => false,
        IrStmt::VarDecl { init: Some(e), .. } => {
            inline_dangling_in_expr(e, declared, by_name, bindings, methods)
        }
        IrStmt::VarDecl { init: None, .. } => false,
        IrStmt::If {
            condition,
            then_branch,
            else_branch,
        } => {
            let mut c = inline_dangling_in_expr(condition, declared, by_name, bindings, methods);
            for s in then_branch {
                c |= inline_dangling_in_stmt(s, declared, by_name, bindings, methods);
            }
            if let Some(b) = else_branch {
                for s in b {
                    c |= inline_dangling_in_stmt(s, declared, by_name, bindings, methods);
                }
            }
            c
        }
        IrStmt::For {
            init,
            condition,
            update,
            body,
        } => {
            let mut c = inline_dangling_in_stmt(init, declared, by_name, bindings, methods);
            if let Some(e) = condition {
                c |= inline_dangling_in_expr(e, declared, by_name, bindings, methods);
            }
            if let Some(e) = update {
                c |= inline_dangling_in_expr(e, declared, by_name, bindings, methods);
            }
            for s in body {
                c |= inline_dangling_in_stmt(s, declared, by_name, bindings, methods);
            }
            c
        }
        IrStmt::While { condition, body } => {
            let mut c = inline_dangling_in_expr(condition, declared, by_name, bindings, methods);
            for s in body {
                c |= inline_dangling_in_stmt(s, declared, by_name, bindings, methods);
            }
            c
        }
        IrStmt::Switch { scrutinee, cases } => {
            let mut c = inline_dangling_in_expr(scrutinee, declared, by_name, bindings, methods);
            for case in cases {
                if let Some(l) = &mut case.label {
                    c |= inline_dangling_in_expr(l, declared, by_name, bindings, methods);
                }
                for s in &mut case.body {
                    c |= inline_dangling_in_stmt(s, declared, by_name, bindings, methods);
                }
            }
            c
        }
        IrStmt::Block(ss) => {
            let mut c = false;
            for s in ss {
                c |= inline_dangling_in_stmt(s, declared, by_name, bindings, methods);
            }
            c
        }
    }
}

fn inline_dangling_in_expr(
    e: &mut IrExpr,
    declared: &std::collections::HashSet<String>,
    by_name: &std::collections::HashMap<String, Vec<(String, usize)>>,
    bindings: &std::collections::HashSet<String>,
    methods: &[crate::ir::Function],
) -> bool {
    // Post-order: arguments/children first.
    let mut changed = false;
    match e {
        IrExpr::Call { arguments, .. } => {
            for a in arguments.iter_mut() {
                changed |= inline_dangling_in_expr(a, declared, by_name, bindings, methods);
            }
        }
        IrExpr::Member { object, .. } => {
            changed |= inline_dangling_in_expr(object, declared, by_name, bindings, methods);
        }
        IrExpr::Index { object, index } | IrExpr::ResourceIndex { resource: object, index } => {
            changed |= inline_dangling_in_expr(object, declared, by_name, bindings, methods);
            changed |= inline_dangling_in_expr(index, declared, by_name, bindings, methods);
        }
        IrExpr::ResourceLoad {
            resource,
            coordinates,
            sample,
        } => {
            changed |= inline_dangling_in_expr(resource, declared, by_name, bindings, methods);
            changed |= inline_dangling_in_expr(coordinates, declared, by_name, bindings, methods);
            if let Some(s) = sample {
                changed |= inline_dangling_in_expr(s, declared, by_name, bindings, methods);
            }
        }
        IrExpr::Construct { arguments, .. } => {
            for a in arguments {
                changed |= inline_dangling_in_expr(a, declared, by_name, bindings, methods);
            }
        }
        IrExpr::Unary { expr: inner, .. } => {
            changed |= inline_dangling_in_expr(inner, declared, by_name, bindings, methods);
        }
        IrExpr::Binary { left, right, .. } => {
            changed |= inline_dangling_in_expr(left, declared, by_name, bindings, methods);
            changed |= inline_dangling_in_expr(right, declared, by_name, bindings, methods);
        }
        IrExpr::Conditional {
            condition,
            then_branch,
            else_branch,
        } => {
            changed |= inline_dangling_in_expr(condition, declared, by_name, bindings, methods);
            changed |= inline_dangling_in_expr(then_branch, declared, by_name, bindings, methods);
            changed |= inline_dangling_in_expr(else_branch, declared, by_name, bindings, methods);
        }
        IrExpr::Assign { lhs, rhs } => {
            changed |= inline_dangling_in_expr(lhs, declared, by_name, bindings, methods);
            changed |= inline_dangling_in_expr(rhs, declared, by_name, bindings, methods);
        }
        IrExpr::InitializerList(elems) => {
            for x in elems {
                changed |= inline_dangling_in_expr(x, declared, by_name, bindings, methods);
            }
        }
        IrExpr::Variable(_) | IrExpr::Literal(_) => {}
    }
    // Then this call itself, if dangling.
    let (callee, arity) = match e {
        IrExpr::Call { function, arguments } => (function.clone(), arguments.len()),
        _ => return changed,
    };
    let dot = match callee.rfind('.') {
        Some(d) => d,
        None => return changed,
    };
    let method = callee[dot + 1..].to_string();
    if method.is_empty() || method.contains('[') || method.contains(']') {
        return changed;
    }
    let obj_str = callee[..dot].to_string();
    let root = obj_str
        .split('.')
        .next()
        .unwrap_or("")
        .split('[')
        .next()
        .unwrap_or("")
        .to_string();
    if root.is_empty() || declared.contains(&root) {
        return changed;
    }
    // Unique declaring struct only (ambiguous names stay loud) — unless
    // every same-named method is identical (same parameters and body):
    // inlining any one then yields the same code, so the call is
    // unambiguous in effect (e.g. `RenderMeshSRT::getMaterial` and
    // `ForwardRenderSRT::getMaterial` share one getter body; the bare
    // member/index references resolve in the caller either way).
    let candidates = match by_name.get(&method) {
        Some(v) if v.len() == 1 => v.clone(),
        Some(v) if !v.is_empty() => {
            let first = match methods.get(v[0].1) {
                Some(f) => f,
                None => return changed,
            };
            let all_same = v[1..].iter().all(|(_, idx)| match methods.get(*idx) {
                Some(f) => f.parameters == first.parameters && f.body == first.body,
                None => false,
            });
            if !all_same {
                return changed;
            }
            vec![v[0].clone()]
        }
        _ => return changed,
    };
    let (_, midx) = candidates.into_iter().next().unwrap();
    let callee_fn = match methods.get(midx) {
        Some(f) => f.clone(),
        None => return changed,
    };
    // Single `return expr;` only.
    if callee_fn.body.len() != 1 {
        return changed;
    }
    let ret_expr = match &callee_fn.body[0] {
        IrStmt::Return(Some(r)) => r.clone(),
        _ => return changed,
    };
    // No recursion.
    if crate::ir::function_calls(&callee_fn)
        .iter()
        .any(|c| c.ends_with(&format!(".{method}")) || c == &method)
    {
        return changed;
    }
    if callee_fn.parameters.len() < arity {
        return changed;
    }
    // Extras must all be module bindings (they shadow globals).
    for p in callee_fn.parameters.iter().skip(arity) {
        if !bindings.contains(&p.name) {
            return changed;
        }
    }
    // Substitute original params with call args; extras stay bare (globals).
    let mut sub: std::collections::HashMap<String, IrExpr> = std::collections::HashMap::new();
    if let IrExpr::Call { arguments, .. } = e.clone() {
        for (i, arg) in arguments.into_iter().enumerate() {
            if let Some(p) = callee_fn.parameters.get(i) {
                sub.insert(p.name.clone(), arg);
            }
        }
    }
    let mut inlined = ret_expr;
    substitute_vars(&mut inlined, &sub);
    *e = inlined;
    true
}

fn substitute_vars(e: &mut IrExpr, sub: &std::collections::HashMap<String, IrExpr>) {
    match e {
        IrExpr::Variable(v) => {
            if let Some(r) = sub.get(v) {
                *e = r.clone();
            }
        }
        IrExpr::Member { object, .. } => substitute_vars(object, sub),
        IrExpr::Index { object, index } | IrExpr::ResourceIndex { resource: object, index } => {
            substitute_vars(object, sub);
            substitute_vars(index, sub);
        }
        IrExpr::ResourceLoad {
            resource,
            coordinates,
            sample,
        } => {
            substitute_vars(resource, sub);
            substitute_vars(coordinates, sub);
            if let Some(s) = sample {
                substitute_vars(s, sub);
            }
        }
        IrExpr::Call { arguments, .. } | IrExpr::Construct { arguments, .. } => {
            for a in arguments {
                substitute_vars(a, sub);
            }
        }
        IrExpr::Unary { expr: inner, .. } => substitute_vars(inner, sub),
        IrExpr::Binary { left, right, .. } => {
            substitute_vars(left, sub);
            substitute_vars(right, sub);
        }
        IrExpr::Conditional {
            condition,
            then_branch,
            else_branch,
        } => {
            substitute_vars(condition, sub);
            substitute_vars(then_branch, sub);
            substitute_vars(else_branch, sub);
        }
        IrExpr::Assign { lhs, rhs } => {
            substitute_vars(lhs, sub);
            substitute_vars(rhs, sub);
        }
        IrExpr::InitializerList(elems) => {
            for x in elems {
                substitute_vars(x, sub);
            }
        }
        IrExpr::Literal(_) => {}
    }
}

