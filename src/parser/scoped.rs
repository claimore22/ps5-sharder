//! Scoped-name resolution for parsed modules.
//!
//! The parser flattens C++ namespaces: `namespace sce { namespace Got {
//! struct BasicDrawSRT { ... }; } }` registers a bare `BasicDrawSRT`, but
//! uses keep their scopes (`sce::Got::BasicDrawSRT *srt`, `sce::Got::Matrices
//! mat`). This pass rewrites a scoped `Type::Named("a::b::C")` to the bare
//! `C` when (and only when) `C` names a struct declared in this module and
//! the full scoped name does not. Unresolvable scopes (framework classes
//! like `sce::Agc::Core::Sampler`, enums like `sce::Agc::VrsRate`) are left
//! untouched so they keep failing loudly at DXC instead of being silently
//! misresolved.

use std::collections::{HashMap, HashSet};

use crate::ast::{Expr, Function, Module, Stmt, Type};

/// Flattens struct inheritance (`struct Derived : Base`): base fields
/// prepend (derived declarations win on name clash, matching C++
/// unqualified lookup), recursively for chains. Unknown or cyclic bases
/// stay loud (today's behavior — the members simply don't resolve).
pub(crate) fn flatten_struct_bases(module: &mut Module) {
    use std::collections::HashSet;
    fn base_fields(
        name: &str,
        module: &Module,
        visiting: &mut HashSet<String>,
    ) -> Option<Vec<crate::ast::FieldDecl>> {
        if !visiting.insert(name.to_string()) {
            return None;
        }
        let decl = module.structs.iter().find(|s| s.name == name)?;
        let mut out = Vec::new();
        for base in &decl.bases {
            if let Some(mut inherited) = base_fields(base, module, visiting) {
                out.append(&mut inherited);
            }
        }
        out.extend(decl.fields.clone());
        visiting.remove(name);
        Some(out)
    }
    // Snapshot base lists first (flattening mutates field lists).
    let work: Vec<(String, Vec<String>)> = module
        .structs
        .iter()
        .map(|s| (s.name.clone(), s.bases.clone()))
        .collect();
    for (name, bases) in &work {
        if bases.is_empty() {
            continue;
        }
        let mut visiting = HashSet::new();
        visiting.insert(name.clone());
        let mut merged: Vec<crate::ast::FieldDecl> = Vec::new();
        let mut ok = true;
        for base in bases {
            match base_fields(base, module, &mut visiting) {
                Some(mut inherited) => merged.append(&mut inherited),
                None => {
                    ok = false;
                    break;
                }
            }
        }
        if !ok {
            continue;
        }
        if let Some(decl) = module.structs.iter_mut().find(|s| &s.name == name) {
            // Derived declarations win: drop inherited fields shadowed by
            // an own field of the same name.
            let own: HashSet<&str> = decl.fields.iter().map(|f| f.name.as_str()).collect();
            merged.retain(|f| !own.contains(f.name.as_str()));
            let mut fields = std::mem::take(&mut decl.fields);
            merged.append(&mut fields);
            decl.fields = merged;
        }
    }
}

/// Rewrites scoped struct references in `module` to their declared names.
pub(crate) fn resolve_scoped_struct_types(module: &mut Module) {
    let declared: HashSet<String> =
        module.structs.iter().map(|s| s.name.clone()).collect();
    // Bare global names (also namespace-flattened namespace variables like
    // `namespace Lds { thread_group_memory uint batchID; }` → `batchID`):
    // scoped `Lds::batchID` uses fold to the bare identifier — unless the
    // bare name also declares a free function (see below).
    let mut global_names: HashSet<String> =
        module.globals.iter().map(|g| g.name.clone()).collect();
    // Free functions as HLSL sees them: methods scoped to a declared
    // struct emit inside it, but unscoped and unknown-scope functions emit
    // as globals (mirrors the emitter), so they share the namespace with
    // flattened variables.
    let free_functions: HashSet<String> = module
        .functions
        .iter()
        .filter(|f| {
            f.scope
                .as_deref()
                .is_none_or(|s| !declared.contains(s))
        })
        .map(|f| f.name.clone())
        .collect();
    // A flattened namespace variable sharing its bare name with a free
    // function (`namespace Lds { ... uint batchID; }` plus
    // `uint batchID(uint dispatchThreadID)` from another header) can
    // neither fold to the bare name (DXC `redefinition of 'batchID' as
    // different kind of symbol`) nor stay scoped (DXC `undeclared
    // identifier 'Lds'`). Worse, the bare fold silently misbinds: inside
    // `findBatch`, `batchID = Lds::batchID` would become a self-assignment
    // to the `inout` parameter instead of a read of the LDS global.
    // Disambiguate by qualifying the GLOBAL with its namespace
    // (`Lds_batchID`) and folding scoped uses to the qualified name.
    // Bare uses (parameters, calls) keep resolving to the parameter or
    // function, exactly as C++ scopes them. Shaders without such a
    // collision keep the bare fold exactly as before.
    // Map each colliding scoped use to its qualified global
    // (`Lds::batchID` → `Lds_batchID`); empty when nothing collides.
    let renames = collision_renames(module, &declared, &global_names, &free_functions);
    if !renames.is_empty() {
        let mut renamed_bare: HashSet<&str> = HashSet::new();
        for (scoped, qualified) in &renames {
            if let Some(bare) = scoped.rsplit("::").next() {
                renamed_bare.insert(bare);
            }
            global_names.insert(qualified.clone());
        }
        for g in &mut module.globals {
            for (scoped, qualified) in &renames {
                if let Some(bare) = scoped.rsplit("::").next()
                    && g.name == bare
                {
                    g.name = qualified.clone();
                }
            }
        }
        // The bare name now denotes the FUNCTION: other qualifiers'
        // uses (`M::batchID`) must stay loud rather than fold onto it.
        for bare in renamed_bare {
            global_names.remove(bare);
        }
    }
    for s in &mut module.structs {
        for f in &mut s.fields {
            resolve_type(&mut f.ty, &declared);
        }
    }
    for g in &mut module.globals {
        resolve_type(&mut g.ty, &declared);
    }
    for r in &mut module.resources {
        if let Some(ty) = r.element_ty.as_mut() {
            resolve_type(ty, &declared);
        }
        for f in &mut r.fields {
            resolve_type(&mut f.ty, &declared);
        }
    }
    // Free-function arities for scoped static-call folding below
    // (`sce::Got::Blnd::Gpu::umToCm(x)` → `umToCm(x)`): overloads share
    // an entry; HLSL resolves the call. Each entry is a
    // (required, total) range: trailing defaulted params
    // (`heatmap(float s, bool b = false)`) accept any arity in between,
    // with the lowering default-arg fill completing the call — so a
    // scoped `heatmap(x)` folds exactly like an unscoped one.
    let mut free_arity: HashMap<String, Vec<(usize, usize)>> = HashMap::new();
    for f in &module.functions {
        let is_free = f.scope.as_deref().is_none_or(|s| !declared.contains(s));
        if is_free {
            let total = f.parameters.len();
            let mut required = 0;
            for p in &f.parameters {
                if p.default.is_some() {
                    break;
                }
                required += 1;
            }
            free_arity
                .entry(f.name.clone())
                .or_default()
                .push((required, total));
        }
    }
    for f in &mut module.functions {
        resolve_function(f, &declared, &global_names, &renames, &free_arity);
    }
}

/// Scoped uses (`Ns::x`) that must fold to a namespace-qualified global
/// (`Ns_x`) because the bare name collides with a free function.
/// Returns scoped→qualified pairs; empty when nothing collides.
///
/// Per distinct qualifier `Ns` of bare `x`: a global `x` must exist, a
/// free function `x` must exist, nothing may already be named `Ns_x`,
/// and `Ns` must not be a declared struct (struct scopes belong to the
/// enum lowering). Zero or several qualifiers for the same `x` stay loud.
fn collision_renames(
    module: &Module,
    declared: &HashSet<String>,
    global_names: &HashSet<String>,
    free_functions: &HashSet<String>,
) -> HashMap<String, String> {
    // Exact scoped spellings (`Lds::batchID`) from every function body,
    // grouped by bare name, each with its qualifier prefix.
    let mut uses: HashSet<String> = HashSet::new();
    for f in &module.functions {
        for s in &f.body {
            collect_scoped_uses_stmt(s, &mut uses);
        }
    }
    // Bare name → qualifiers using it.
    let mut qualifiers: HashMap<&str, HashSet<&str>> = HashMap::new();
    for use_ in &uses {
        if let Some(bare) = use_.rsplit("::").next()
            && use_.len() > bare.len()
        {
            let ns = use_[..use_.len() - bare.len()].trim_end_matches(':');
            qualifiers.entry(bare).or_default().insert(ns);
        }
    }
    let mut out = HashMap::new();
    for (bare, quals) in &qualifiers {
        if quals.len() != 1 {
            continue;
        }
        let ns = quals.iter().next().unwrap();
        let head = ns.split("::").next().unwrap_or("");
        if declared.contains(head) {
            continue;
        }
        if !global_names.contains(*bare) {
            continue;
        }
        if !free_functions.contains(*bare) {
            continue;
        }
        let qualified = format!("{}_{bare}", ns.replace("::", "_"));
        if global_names.contains(&qualified) || free_functions.contains(&qualified) {
            continue;
        }
        for use_ in &uses {
            if let Some(use_bare) = use_.rsplit("::").next()
                && use_bare == *bare
                && use_.len() > use_bare.len()
                && use_[..use_.len() - use_bare.len()].trim_end_matches(':') == *ns
            {
                out.insert(use_.clone(), qualified.clone());
            }
        }
    }
    out
}

/// Collects exact scoped-variable spellings (`Lds::batchID`) from a
/// statement for collision detection (see [`collision_renames`]).
fn collect_scoped_uses_stmt(s: &Stmt, out: &mut HashSet<String>) {
    match s {
        Stmt::Return(e) => {
            if let Some(e) = e {
                collect_scoped_uses_expr(e, out);
            }
        }
        Stmt::Block(ss) => {
            for s in ss {
                collect_scoped_uses_stmt(s, out);
            }
        }
        Stmt::If {
            condition,
            then_branch,
            else_branch,
        } => {
            collect_scoped_uses_expr(condition, out);
            for s in then_branch {
                collect_scoped_uses_stmt(s, out);
            }
            if let Some(b) = else_branch {
                for s in b {
                    collect_scoped_uses_stmt(s, out);
                }
            }
        }
        Stmt::For {
            init,
            condition,
            update,
            body,
        } => {
            collect_scoped_uses_stmt(init, out);
            if let Some(c) = condition {
                collect_scoped_uses_expr(c, out);
            }
            if let Some(u) = update {
                collect_scoped_uses_expr(u, out);
            }
            for s in body {
                collect_scoped_uses_stmt(s, out);
            }
        }
        Stmt::While { condition, body } => {
            collect_scoped_uses_expr(condition, out);
            for s in body {
                collect_scoped_uses_stmt(s, out);
            }
        }
        Stmt::Switch { scrutinee, cases } => {
            collect_scoped_uses_expr(scrutinee, out);
            for c in cases {
                if let Some(label) = &c.label {
                    collect_scoped_uses_expr(label, out);
                }
                for s in &c.body {
                    collect_scoped_uses_stmt(s, out);
                }
            }
        }
        Stmt::VarDecl { init, .. } | Stmt::Destructure { init, .. } => {
            if let Some(e) = init {
                collect_scoped_uses_expr(e, out);
            }
        }
        Stmt::Expr(e) => collect_scoped_uses_expr(e, out),
    }
}

fn collect_scoped_uses_expr(e: &Expr, out: &mut HashSet<String>) {
    match e {
        Expr::Variable(name) => {
            if name.contains("::") {
                out.insert(name.clone());
            }
        }
        Expr::Member { object, .. } => collect_scoped_uses_expr(object, out),
        Expr::Index { object, index } => {
            collect_scoped_uses_expr(object, out);
            collect_scoped_uses_expr(index, out);
        }
        Expr::Call {
            function,
            arguments,
        } => {
            collect_scoped_uses_expr(function, out);
            for a in arguments {
                collect_scoped_uses_expr(a, out);
            }
        }
        Expr::Constructor { arguments, .. } => {
            for a in arguments {
                collect_scoped_uses_expr(a, out);
            }
        }
        Expr::Assign { lhs, rhs } => {
            collect_scoped_uses_expr(lhs, out);
            collect_scoped_uses_expr(rhs, out);
        }
        Expr::Unary { expr, .. } => collect_scoped_uses_expr(expr, out),
        Expr::Binary { left, right, .. } => {
            collect_scoped_uses_expr(left, out);
            collect_scoped_uses_expr(right, out);
        }
        Expr::Conditional {
            condition,
            then_branch,
            else_branch,
        } => {
            collect_scoped_uses_expr(condition, out);
            collect_scoped_uses_expr(then_branch, out);
            collect_scoped_uses_expr(else_branch, out);
        }
        Expr::InitializerList(elems) => {
            for x in elems {
                collect_scoped_uses_expr(x, out);
            }
        }
        // Lambda bodies keep their scopes (beta-reduction inlines them
        // later; scoped names inside stay loud).
        Expr::Lambda { .. } => {}
        Expr::Number(_) | Expr::String(_) | Expr::Bool(_) => {}
    }
}

fn resolve_type(ty: &mut Type, declared: &HashSet<String>) {
    match ty {
        Type::Named(name) => {
            if name.contains("::") && !declared.contains(name) {
                if let Some(bare) = name.rsplit("::").next() {
                    if declared.contains(bare) {
                        *name = bare.to_string();
                    }
                }
            }
        }
        Type::Resource { element, .. } => resolve_type(element, declared),
        _ => {}
    }
}

fn resolve_function(
    f: &mut Function,
    declared: &HashSet<String>,
    global_names: &HashSet<String>,
    renames: &HashMap<String, String>,
    free_arity: &HashMap<String, Vec<(usize, usize)>>,
) {
    resolve_type(&mut f.return_type, declared);
    for p in &mut f.parameters {
        resolve_type(&mut p.ty, declared);
    }
    for s in &mut f.body {
        resolve_stmt(s, declared, global_names, renames, free_arity);
    }
}

fn resolve_stmt(
    s: &mut Stmt,
    declared: &HashSet<String>,
    global_names: &HashSet<String>,
    renames: &HashMap<String, String>,
    free_arity: &HashMap<String, Vec<(usize, usize)>>,
) {
    match s {
        Stmt::Return(e) => {
            if let Some(e) = e {
                resolve_expr(e, declared, global_names, renames, free_arity);
            }
        }
        Stmt::Block(ss) => {
            for s in ss {
                resolve_stmt(s, declared, global_names, renames, free_arity);
            }
        }
        Stmt::If {
            condition,
            then_branch,
            else_branch,
        } => {
            resolve_expr(condition, declared, global_names, renames, free_arity);
            for s in then_branch {
                resolve_stmt(s, declared, global_names, renames, free_arity);
            }
            if let Some(b) = else_branch {
                for s in b {
                    resolve_stmt(s, declared, global_names, renames, free_arity);
                }
            }
        }
        Stmt::For {
            init,
            condition,
            update,
            body,
        } => {
            resolve_stmt(init, declared, global_names, renames, free_arity);
            if let Some(c) = condition {
                resolve_expr(c, declared, global_names, renames, free_arity);
            }
            if let Some(u) = update {
                resolve_expr(u, declared, global_names, renames, free_arity);
            }
            for s in body {
                resolve_stmt(s, declared, global_names, renames, free_arity);
            }
        }
        Stmt::While { condition, body } => {
            resolve_expr(condition, declared, global_names, renames, free_arity);
            for s in body {
                resolve_stmt(s, declared, global_names, renames, free_arity);
            }
        }
        Stmt::Switch { scrutinee, cases } => {
            resolve_expr(scrutinee, declared, global_names, renames, free_arity);
            for c in cases {
                if let Some(label) = c.label.as_mut() {
                    resolve_expr(label, declared, global_names, renames, free_arity);
                }
                for s in &mut c.body {
                    resolve_stmt(s, declared, global_names, renames, free_arity);
                }
            }
        }
        Stmt::VarDecl { ty, init, .. } => {
            resolve_type(ty, declared);
            if let Some(e) = init {
                resolve_expr(e, declared, global_names, renames, free_arity);
            }
        }
        Stmt::Destructure { ty, init, .. } => {
            resolve_type(ty, declared);
            if let Some(e) = init {
                resolve_expr(e, declared, global_names, renames, free_arity);
            }
        }
        Stmt::Expr(e) => resolve_expr(e, declared, global_names, renames, free_arity),
    }
}

fn resolve_expr(
    e: &mut Expr,
    declared: &HashSet<String>,
    global_names: &HashSet<String>,
    renames: &HashMap<String, String>,
    free_arity: &HashMap<String, Vec<(usize, usize)>>,
) {
    match e {
        Expr::Number(_) | Expr::String(_) | Expr::Bool(_) => {}
        Expr::Variable(name) => {
            // Collision-disambiguated uses (`Lds::batchID` → `Lds_batchID`
            // when bare `batchID` also names a free function) rewrite first.
            // Otherwise, namespace-flattened globals: `Lds::batchID` →
            // `batchID` when `batchID` is a declared global (the namespace
            // body registered a bare `groupshared uint batchID`).
            // Struct-scoped enumerators (`M::Transparent`, head `M` a
            // declared struct) are left for the shadow-aware
            // `fold_scoped_enum_uses` lowering pass, which respects locals
            // shadowing the bare name. Framework `sce::...` paths never
            // match a bare global and stay untouched.
            if let Some(qualified) = renames.get(name) {
                *name = qualified.clone();
            } else if name.contains("::") && !global_names.contains(name) {
                let head = name.split("::").next().unwrap_or("");
                if !declared.contains(head) {
                    if let Some(bare) = name.rsplit("::").next() {
                        if global_names.contains(bare) {
                            *name = bare.to_string();
                        }
                    }
                }
            }
        }
        Expr::Member { object, .. } => resolve_expr(object, declared, global_names, renames, free_arity),
        Expr::Index { object, index } => {
            resolve_expr(object, declared, global_names, renames, free_arity);
            resolve_expr(index, declared, global_names, renames, free_arity);
        }
        Expr::Call {
            function,
            arguments,
        } => {
            // Scoped static calls (`sce::Got::Blnd::Gpu::umToCm(x)`) to
            // flattened free functions: fold to the bare name when it
            // names free function(s) whose (required, total) arity range
            // admits the call (HLSL resolves the overload, exactly as
            // PSSL would; defaulted trailing params fill in lowering).
            // Skipped when the head is a declared struct (struct-scoped
            // methods belong to the method machinery), when the bare name
            // is a declared struct (a constructor call — handled just
            // below), when the exact spelling is already claimed above,
            // or when no overload admits the arity — those keep today's
            // loud path.
            if let Expr::Variable(callee) = function.as_mut() {
                if callee.contains("::") && !renames.contains_key(callee) {
                    let head = callee.split("::").next().unwrap_or("");
                    if !declared.contains(head) {
                        if let Some(bare) = callee.rsplit("::").next() {
                            if !declared.contains(bare) {
                                if let Some(arities) = free_arity.get(bare) {
                                    let n = arguments.len();
                                    if arities.iter().any(|(lo, hi)| *lo <= n && n <= *hi) {
                                        *callee = bare.to_string();
                                    }
                                }
                            }
                        }
                    }
                }
            }
            // Scoped struct construction (`shader::MaterialData(...)`
            // with bare `MaterialData` declared): the parser builds a
            // Call (scoped names aren't known structs at parse), but
            // downstream construction handling needs a Constructor.
            // Rewrite unless a free function of the same name admits the
            // arity (genuine call ambiguity keeps today's loud path).
            let rewrite_ctor = match function.as_ref() {
                Expr::Variable(callee) => {
                    if !callee.contains("::") {
                        false
                    } else {
                        let head = callee.split("::").next().unwrap_or("");
                        match callee.rsplit("::").next() {
                            Some(bare)
                                if !declared.contains(head) && declared.contains(bare) =>
                            {
                                let n = arguments.len();
                                !free_arity.get(bare).is_some_and(|arities| {
                                    arities.iter().any(|(lo, hi)| *lo <= n && n <= *hi)
                                })
                            }
                            _ => false,
                        }
                    }
                }
                _ => false,
            };
            if rewrite_ctor {
                if let Expr::Variable(callee) = function.as_ref() {
                    let bare = callee.rsplit("::").next().unwrap_or(callee).to_string();
                    let args = std::mem::take(arguments);
                    *e = Expr::Constructor {
                        ty: Type::Named(bare),
                        arguments: args,
                    };
                }
            }
            // NOTE: `e` may now be a Constructor (resolve accordingly).
            match e {
                Expr::Constructor { ty, arguments } => {
                    resolve_type(ty, declared);
                    for a in arguments {
                        resolve_expr(a, declared, global_names, renames, free_arity);
                    }
                }
                Expr::Call { function, arguments } => {
                    resolve_expr(function, declared, global_names, renames, free_arity);
                    for a in arguments {
                        resolve_expr(a, declared, global_names, renames, free_arity);
                    }
                }
                _ => unreachable!("guarded by outer Call match"),
            }
        }
        Expr::Constructor { ty, arguments } => {
            resolve_type(ty, declared);
            for a in arguments {
                resolve_expr(a, declared, global_names, renames, free_arity);
            }
        }
        Expr::Assign { lhs, rhs } => {
            resolve_expr(lhs, declared, global_names, renames, free_arity);
            resolve_expr(rhs, declared, global_names, renames, free_arity);
        }
        Expr::Unary { expr, .. } => resolve_expr(expr, declared, global_names, renames, free_arity),
        Expr::Binary { left, right, .. } => {
            resolve_expr(left, declared, global_names, renames, free_arity);
            resolve_expr(right, declared, global_names, renames, free_arity);
        }
        Expr::Conditional {
            condition,
            then_branch,
            else_branch,
        } => {
            resolve_expr(condition, declared, global_names, renames, free_arity);
            resolve_expr(then_branch, declared, global_names, renames, free_arity);
            resolve_expr(else_branch, declared, global_names, renames, free_arity);
        }
        Expr::InitializerList(elems) => {
            for x in elems {
                resolve_expr(x, declared, global_names, renames, free_arity);
            }
        }
        // Lambda bodies keep their scopes (see collect above).
        Expr::Lambda { .. } => {}
        Expr::Number(_) | Expr::String(_) | Expr::Bool(_) => {}
    }
}
