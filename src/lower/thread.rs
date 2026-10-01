//! Threads pixel-position values into 0-arg position builtins.
//!
//! The target platform exposes 0-arg position queries (`GetViVjLinearCentroid()` — the
//! current pixel's linear centroid, a compiler builtin with no in-tree
//! definition anywhere in the framework headers) that have no HLSL spelling: DXC
//! rejects guesses like `GetRenderTargetSamplePosition` for Vulkan
//! targets, and HLSL can only source per-pixel positions from an
//! `SV_Position` input. This pass gives every function calling such a
//! builtin a trailing `float2` parameter (the builtin calls become the
//! parameter) and appends the caller's position value at call sites:
//! inside the entry, an `SV_Position` input (`.xy`-adapted) or a fresh
//! `SV_Position` entry parameter when the entry has none; inside already
//! threaded functions, their own position parameter (chains compose).
//! Calls from unthreaded non-entry functions stay loud (no such chain
//! exists in-tree). Shaders without these builtins are untouched.

use std::collections::{HashMap, HashSet};

use crate::ast::{Expr, Function, Module, Semantic, Stmt, Type};

/// 0-arg position builtins (lowercase) handled by this pass: the
/// linear centroid plus the perspective center/centroid siblings (same
/// compiler-builtin family, all float2 pixel positions).
const POSITION_BUILTINS: &[&str] = &[
    "getvivjlinearcentroid",
    "getvivjperspcenter",
    "getvivjperspcentroid",
];

/// Param name carrying a threaded builtin (`__pos_<builtin>`), suffixed
/// while it collides with an existing parameter.
fn threaded_param_name(base: &str, taken: &HashSet<String>) -> String {
    let mut name = format!("__pos_{base}");
    let mut n = 0;
    while taken.contains(&name) {
        n += 1;
        name = format!("__pos_{base}_{n}");
    }
    name
}

/// True for a 0-arg call to a handled builtin.
fn is_position_builtin_call(e: &Expr) -> Option<String> {
    if let Expr::Call { function, arguments } = e {
        if arguments.is_empty() {
            if let Expr::Variable(name) = function.as_ref() {
                let lower = name.to_ascii_lowercase();
                // Bare or `ns::`-qualified spelling (`wave::inverse` style
                // namespaces never define these; the bare tail decides).
                let bare = lower.rsplit("::").next().unwrap_or(&lower);
                if POSITION_BUILTINS.contains(&bare) {
                    return Some(bare.to_string());
                }
            }
        }
    }
    None
}

/// Collects handled builtins called in an expression (any depth).
fn collect_builtin_calls(e: &Expr, out: &mut HashSet<String>) {
    if let Some(b) = is_position_builtin_call(e) {
        out.insert(b);
    }
    match e {
        Expr::Call { function, arguments } => {
            collect_builtin_calls(function, out);
            for a in arguments {
                collect_builtin_calls(a, out);
            }
        }
        Expr::Member { object, .. } => collect_builtin_calls(object, out),
        Expr::Index { object, index } => {
            collect_builtin_calls(object, out);
            collect_builtin_calls(index, out);
        }
        Expr::Constructor { arguments, .. } => {
            for a in arguments {
                collect_builtin_calls(a, out);
            }
        }
        Expr::Assign { lhs, rhs } => {
            collect_builtin_calls(lhs, out);
            collect_builtin_calls(rhs, out);
        }
        Expr::Unary { expr, .. } => collect_builtin_calls(expr, out),
        Expr::Binary { left, right, .. } => {
            collect_builtin_calls(left, out);
            collect_builtin_calls(right, out);
        }
        Expr::Conditional { condition, then_branch, else_branch } => {
            collect_builtin_calls(condition, out);
            collect_builtin_calls(then_branch, out);
            collect_builtin_calls(else_branch, out);
        }
        Expr::InitializerList(elems) => {
            for x in elems {
                collect_builtin_calls(x, out);
            }
        }
        // Lambda bodies keep their scopes (beta-reduction inlines them
        // later; scoped names inside stay loud — documented limitation).
        Expr::Lambda { .. } => {}
        Expr::Variable(_) | Expr::Number(_) | Expr::String(_) | Expr::Bool(_) => {}
    }
}

/// Collects handled builtins called in a statement (any depth).
fn collect_builtin_calls_stmt(s: &Stmt, out: &mut HashSet<String>) {
    match s {
        Stmt::Return(Some(e)) | Stmt::Expr(e) => collect_builtin_calls(e, out),
        Stmt::Return(None) => {}
        Stmt::VarDecl { init: Some(e), .. } | Stmt::Destructure { init: Some(e), .. } => {
            collect_builtin_calls(e, out)
        }
        Stmt::VarDecl { init: None, .. } | Stmt::Destructure { init: None, .. } => {}
        Stmt::Block(ss) => {
            for x in ss {
                collect_builtin_calls_stmt(x, out);
            }
        }
        Stmt::If { condition, then_branch, else_branch } => {
            collect_builtin_calls(condition, out);
            for x in then_branch {
                collect_builtin_calls_stmt(x, out);
            }
            if let Some(els) = else_branch {
                for x in els {
                    collect_builtin_calls_stmt(x, out);
                }
            }
        }
        Stmt::For { init, condition, update, body } => {
            collect_builtin_calls_stmt(init, out);
            if let Some(c) = condition {
                collect_builtin_calls(c, out);
            }
            if let Some(u) = update {
                collect_builtin_calls(u, out);
            }
            for x in body {
                collect_builtin_calls_stmt(x, out);
            }
        }
        Stmt::While { condition, body } => {
            collect_builtin_calls(condition, out);
            for x in body {
                collect_builtin_calls_stmt(x, out);
            }
        }
        Stmt::Switch { scrutinee, cases } => {
            collect_builtin_calls(scrutinee, out);
            for c in cases {
                if let Some(label) = &c.label {
                    collect_builtin_calls(label, out);
                }
                for x in &c.body {
                    collect_builtin_calls_stmt(x, out);
                }
            }
        }
    }
}

/// Collects call-target names (bare tail of scoped spellings) in an
/// expression, to test whether the entry calls a threaded function.
fn collect_call_targets_expr(e: &Expr, out: &mut HashSet<String>) {
    match e {
        Expr::Call { function, arguments } => {
            if let Expr::Variable(n) = function.as_ref() {
                out.insert(n.rsplit("::").next().unwrap_or(n).to_string());
            }
            collect_call_targets_expr(function, out);
            for a in arguments {
                collect_call_targets_expr(a, out);
            }
        }
        Expr::Member { object, .. } => collect_call_targets_expr(object, out),
        Expr::Index { object, index } => {
            collect_call_targets_expr(object, out);
            collect_call_targets_expr(index, out);
        }
        Expr::Constructor { arguments, .. } => {
            for a in arguments {
                collect_call_targets_expr(a, out);
            }
        }
        Expr::Assign { lhs, rhs } => {
            collect_call_targets_expr(lhs, out);
            collect_call_targets_expr(rhs, out);
        }
        Expr::Unary { expr, .. } => collect_call_targets_expr(expr, out),
        Expr::Binary { left, right, .. } => {
            collect_call_targets_expr(left, out);
            collect_call_targets_expr(right, out);
        }
        Expr::Conditional { condition, then_branch, else_branch } => {
            collect_call_targets_expr(condition, out);
            collect_call_targets_expr(then_branch, out);
            collect_call_targets_expr(else_branch, out);
        }
        Expr::InitializerList(elems) => {
            for x in elems {
                collect_call_targets_expr(x, out);
            }
        }
        // Lambda bodies keep their scopes (beta-reduction inlines them
        // later; scoped names inside stay loud — documented limitation).
        Expr::Lambda { .. } => {}
        Expr::Variable(_) | Expr::Number(_) | Expr::String(_) | Expr::Bool(_) => {}
    }
}

/// Collects call-target names in a statement.
fn collect_call_targets_stmt(s: &Stmt, out: &mut HashSet<String>) {
    match s {
        Stmt::Return(Some(e)) | Stmt::Expr(e) => collect_call_targets_expr(e, out),
        Stmt::Return(None) => {}
        Stmt::VarDecl { init: Some(e), .. } | Stmt::Destructure { init: Some(e), .. } => {
            collect_call_targets_expr(e, out)
        }
        Stmt::VarDecl { init: None, .. } | Stmt::Destructure { init: None, .. } => {}
        Stmt::Block(ss) => {
            for x in ss {
                collect_call_targets_stmt(x, out);
            }
        }
        Stmt::If { condition, then_branch, else_branch } => {
            collect_call_targets_expr(condition, out);
            for x in then_branch {
                collect_call_targets_stmt(x, out);
            }
            if let Some(els) = else_branch {
                for x in els {
                    collect_call_targets_stmt(x, out);
                }
            }
        }
        Stmt::For { init, condition, update, body } => {
            collect_call_targets_stmt(init, out);
            if let Some(c) = condition {
                collect_call_targets_expr(c, out);
            }
            if let Some(u) = update {
                collect_call_targets_expr(u, out);
            }
            for x in body {
                collect_call_targets_stmt(x, out);
            }
        }
        Stmt::While { condition, body } => {
            collect_call_targets_expr(condition, out);
            for x in body {
                collect_call_targets_stmt(x, out);
            }
        }
        Stmt::Switch { scrutinee, cases } => {
            collect_call_targets_expr(scrutinee, out);
            for c in cases {
                if let Some(label) = &c.label {
                    collect_call_targets_expr(label, out);
                }
                for x in &c.body {
                    collect_call_targets_stmt(x, out);
                }
            }
        }
    }
}

/// Rewrites 0-arg builtin calls to `replacement` in an expression.
fn rewrite_builtin_call_expr(e: &mut Expr, builtin: &str, replacement: &Expr) {
    if let Some(b) = is_position_builtin_call(e) {
        if b == builtin {
            *e = replacement.clone();
            return;
        }
    }
    match e {
        Expr::Call { function, arguments } => {
            rewrite_builtin_call_expr(function, builtin, replacement);
            for a in arguments {
                rewrite_builtin_call_expr(a, builtin, replacement);
            }
        }
        Expr::Member { object, .. } => rewrite_builtin_call_expr(object, builtin, replacement),
        Expr::Index { object, index } => {
            rewrite_builtin_call_expr(object, builtin, replacement);
            rewrite_builtin_call_expr(index, builtin, replacement);
        }
        Expr::Constructor { arguments, .. } => {
            for a in arguments {
                rewrite_builtin_call_expr(a, builtin, replacement);
            }
        }
        Expr::Assign { lhs, rhs } => {
            rewrite_builtin_call_expr(lhs, builtin, replacement);
            rewrite_builtin_call_expr(rhs, builtin, replacement);
        }
        Expr::Unary { expr, .. } => rewrite_builtin_call_expr(expr, builtin, replacement),
        Expr::Binary { left, right, .. } => {
            rewrite_builtin_call_expr(left, builtin, replacement);
            rewrite_builtin_call_expr(right, builtin, replacement);
        }
        Expr::Conditional { condition, then_branch, else_branch } => {
            rewrite_builtin_call_expr(condition, builtin, replacement);
            rewrite_builtin_call_expr(then_branch, builtin, replacement);
            rewrite_builtin_call_expr(else_branch, builtin, replacement);
        }
        Expr::InitializerList(elems) => {
            for x in elems {
                rewrite_builtin_call_expr(x, builtin, replacement);
            }
        }
        // Lambda bodies keep their scopes (beta-reduction inlines them
        // later; scoped names inside stay loud — documented limitation).
        Expr::Lambda { .. } => {}
        Expr::Variable(_) | Expr::Number(_) | Expr::String(_) | Expr::Bool(_) => {}
    }
}

/// Rewrites 0-arg builtin calls to `Variable(param)` in a statement.
fn rewrite_builtin_call_stmt(s: &mut Stmt, builtin: &str, replacement: &Expr) {
    match s {
        Stmt::Return(Some(e)) | Stmt::Expr(e) => rewrite_builtin_call_expr(e, builtin, replacement),
        Stmt::Return(None) => {}
        Stmt::VarDecl { init: Some(e), .. } | Stmt::Destructure { init: Some(e), .. } => {
            rewrite_builtin_call_expr(e, builtin, replacement)
        }
        Stmt::VarDecl { init: None, .. } | Stmt::Destructure { init: None, .. } => {}
        Stmt::Block(ss) => {
            for x in ss {
                rewrite_builtin_call_stmt(x, builtin, replacement);
            }
        }
        Stmt::If { condition, then_branch, else_branch } => {
            rewrite_builtin_call_expr(condition, builtin, replacement);
            for x in then_branch {
                rewrite_builtin_call_stmt(x, builtin, replacement);
            }
            if let Some(els) = else_branch {
                for x in els {
                    rewrite_builtin_call_stmt(x, builtin, replacement);
                }
            }
        }
        Stmt::For { init, condition, update, body } => {
            rewrite_builtin_call_stmt(init, builtin, replacement);
            if let Some(c) = condition {
                rewrite_builtin_call_expr(c, builtin, replacement);
            }
            if let Some(u) = update {
                rewrite_builtin_call_expr(u, builtin, replacement);
            }
            for x in body {
                rewrite_builtin_call_stmt(x, builtin, replacement);
            }
        }
        Stmt::While { condition, body } => {
            rewrite_builtin_call_expr(condition, builtin, replacement);
            for x in body {
                rewrite_builtin_call_stmt(x, builtin, replacement);
            }
        }
        Stmt::Switch { scrutinee, cases } => {
            rewrite_builtin_call_expr(scrutinee, builtin, replacement);
            for c in cases {
                if let Some(label) = &mut c.label {
                    rewrite_builtin_call_expr(label, builtin, replacement);
                }
                for x in &mut c.body {
                    rewrite_builtin_call_stmt(x, builtin, replacement);
                }
            }
        }
    }
}

/// Appends `arg` to calls of `target` (bare or scoped spelling) in an
/// expression, recursing everywhere (conditions, inits, nested calls).
fn append_call_arg_expr(e: &mut Expr, target: &str, arg: &Expr) {
    // Bare or scoped callee spelling (`wireframeScale`, `sce::Got::f`).
    let matches = |f: &Expr| match f {
        Expr::Variable(n) => n == target || n.rsplit("::").next().unwrap_or(n) == target,
        _ => false,
    };
    match e {
        Expr::Call { function, arguments } => {
            if matches(function) {
                arguments.push(arg.clone());
            }
            append_call_arg_expr(function, target, arg);
            // The appended arg is a plain value (variable/member path):
            // recursing into it terminates.
            for a in arguments.iter_mut() {
                append_call_arg_expr(a, target, arg);
            }
        }
        Expr::Member { object, .. } => append_call_arg_expr(object, target, arg),
        Expr::Index { object, index } => {
            append_call_arg_expr(object, target, arg);
            append_call_arg_expr(index, target, arg);
        }
        Expr::Constructor { arguments, .. } => {
            for a in arguments {
                append_call_arg_expr(a, target, arg);
            }
        }
        Expr::Assign { lhs, rhs } => {
            append_call_arg_expr(lhs, target, arg);
            append_call_arg_expr(rhs, target, arg);
        }
        Expr::Unary { expr, .. } => append_call_arg_expr(expr, target, arg),
        Expr::Binary { left, right, .. } => {
            append_call_arg_expr(left, target, arg);
            append_call_arg_expr(right, target, arg);
        }
        Expr::Conditional { condition, then_branch, else_branch } => {
            append_call_arg_expr(condition, target, arg);
            append_call_arg_expr(then_branch, target, arg);
            append_call_arg_expr(else_branch, target, arg);
        }
        Expr::InitializerList(elems) => {
            for x in elems {
                append_call_arg_expr(x, target, arg);
            }
        }
        // Lambda bodies keep their scopes (beta-reduction inlines them
        // later; scoped names inside stay loud — documented limitation).
        Expr::Lambda { .. } => {}
        Expr::Variable(_) | Expr::Number(_) | Expr::String(_) | Expr::Bool(_) => {}
    }
}

/// Appends `arg` to calls of `target` in a statement.
fn append_call_arg_stmt(s: &mut Stmt, target: &str, arg: &Expr) {
    match s {
        Stmt::Return(Some(e)) | Stmt::Expr(e) => append_call_arg_expr(e, target, arg),
        Stmt::Return(None) => {}
        Stmt::VarDecl { init: Some(e), .. } | Stmt::Destructure { init: Some(e), .. } => {
            append_call_arg_expr(e, target, arg)
        }
        Stmt::VarDecl { init: None, .. } | Stmt::Destructure { init: None, .. } => {}
        Stmt::Block(ss) => {
            for x in ss {
                append_call_arg_stmt(x, target, arg);
            }
        }
        Stmt::If { condition, then_branch, else_branch } => {
            append_call_arg_expr(condition, target, arg);
            for x in then_branch {
                append_call_arg_stmt(x, target, arg);
            }
            if let Some(els) = else_branch {
                for x in els {
                    append_call_arg_stmt(x, target, arg);
                }
            }
        }
        Stmt::For { init, condition, update, body } => {
            append_call_arg_stmt(init, target, arg);
            if let Some(c) = condition {
                append_call_arg_expr(c, target, arg);
            }
            if let Some(u) = update {
                append_call_arg_expr(u, target, arg);
            }
            for x in body {
                append_call_arg_stmt(x, target, arg);
            }
        }
        Stmt::While { condition, body } => {
            append_call_arg_expr(condition, target, arg);
            for x in body {
                append_call_arg_stmt(x, target, arg);
            }
        }
        Stmt::Switch { scrutinee, cases } => {
            append_call_arg_expr(scrutinee, target, arg);
            for c in cases {
                if let Some(label) = &mut c.label {
                    append_call_arg_expr(label, target, arg);
                }
                for x in &mut c.body {
                    append_call_arg_stmt(x, target, arg);
                }
            }
        }
    }
}

/// Finds the entry's pixel-position value: an `SV_Position`/`S_POSITION`
/// entry input (`.xy`-adapted to float2), else `None` (the caller adds a
/// fresh `SV_Position` entry parameter).
fn entry_position_value(entry: &Function, module: &Module) -> Option<Expr> {
    for p in &entry.parameters {
        let is_pos = p
            .semantic
            .as_ref()
            .is_some_and(|s| s.name() == "S_POSITION" || s.name() == "SV_Position");
        if !is_pos {
            continue;
        }
        match &p.ty {
            Type::Vector { width: 2, .. } => return Some(Expr::Variable(p.name.clone())),
            Type::Vector { .. } => {
                return Some(Expr::Member {
                    object: Box::new(Expr::Variable(p.name.clone())),
                    member: "xy".to_string(),
                })
            }
            _ => continue,
        }
    }
    // Struct inputs (e.g. `BasicGsOut input` with `float4 pos :
    // S_POSITION`, which maps to `SV_Position`): `input.pos.xy`.
    for p in &entry.parameters {
        if let Type::Named(s) = &p.ty {
            if let Some(decl) = module.structs.iter().find(|d| &d.name == s) {
                for f in &decl.fields {
                    let is_pos = f
                        .semantic
                        .as_ref()
                        .is_some_and(|s| s.name() == "S_POSITION" || s.name() == "SV_Position");
                    if !is_pos {
                        continue;
                    }
                    let base = Expr::Member {
                        object: Box::new(Expr::Variable(p.name.clone())),
                        member: f.name.clone(),
                    };
                    match &f.ty {
                        Type::Vector { width: 2, .. } => return Some(base),
                        Type::Vector { .. } => {
                            return Some(Expr::Member {
                                object: Box::new(base),
                                member: "xy".to_string(),
                            })
                        }
                        _ => continue,
                    }
                }
            }
        }
    }
    None
}

/// Threads pixel-position values into 0-arg position builtins; see the
/// module docs. Returns the (possibly cloned) module.
pub(crate) fn thread_position_builtins(module: &Module, entry_name: Option<&str>) -> Module {
    let Some(entry_name) = entry_name else {
        return module.clone();
    };
    // Non-entry functions calling a handled builtin, per builtin, plus
    // the builtins the entry itself calls (rewritten to the position
    // value directly — entries feed DXC, so no plain params allowed).
    let mut needy: HashMap<String, Vec<String>> = HashMap::new();
    let mut entry_builtins: HashSet<String> = HashSet::new();
    for f in &module.functions {
        let mut found = HashSet::new();
        for s in &f.body {
            collect_builtin_calls_stmt(s, &mut found);
        }
        if f.name == entry_name {
            entry_builtins.extend(found);
        } else {
            for b in found {
                needy.entry(b).or_default().push(f.name.clone());
            }
        }
    }
    if needy.is_empty() && entry_builtins.is_empty() {
        return module.clone();
    }
    let mut module = module.clone();
    // Function name -> threaded param name (per builtin).
    let mut threaded: HashMap<(String, String), String> = HashMap::new();
    for (builtin, names) in &needy {
        for name in names {
            let Some(f) = module.functions.iter_mut().find(|f| &f.name == name) else {
                continue;
            };
            let taken: HashSet<String> = f.parameters.iter().map(|p| p.name.clone()).collect();
            let param = threaded_param_name(builtin, &taken);
            let replacement = Expr::Variable(param.clone());
            for s in &mut f.body {
                rewrite_builtin_call_stmt(s, builtin, &replacement);
            }
            f.parameters.push(crate::ast::Parameter {
                ty: Type::Vector { base: crate::ast::ScalarType::Float, width: 2 },
                name: param.clone(),
                direction: None,
                semantic: None,
                array_size: None,
                gs_primitive: None,
                default: None,
                is_uniform: false,
                span: crate::error::Span::default(),
            });
            threaded.insert((name.clone(), builtin.clone()), param);
        }
    }
    // Bind call sites. In the entry: the SV_Position value (or a fresh
    // SV_Position entry param). In threaded functions: their own param
    // (chains compose).
    let entry_idx = module.functions.iter().position(|f| f.name == entry_name);
    let Some(entry_idx) = entry_idx else {
        return module;
    };
    // Collect (function, param) pairs first (borrow split).
    let triples: Vec<(String, String)> = threaded
        .iter()
        .map(|((f, _b), p)| (f.clone(), p.clone()))
        .collect();
    // The entry needs a position value only when it calls the builtin
    // directly or calls a threaded function; otherwise no entry changes
    // (threaded defs survive only if reachable, like any helper).
    let mut entry_targets = HashSet::new();
    for s in &module.functions[entry_idx].body {
        collect_call_targets_stmt(s, &mut entry_targets);
    }
    let entry_needs_pos = !entry_builtins.is_empty()
        || triples.iter().any(|(f, _)| entry_targets.contains(f));
    if !entry_needs_pos {
        return module;
    }
    let pos_value = entry_position_value(&module.functions[entry_idx], &module);
    let pos_expr = match pos_value {
        Some(e) => e,
        None => {
            let entry = &mut module.functions[entry_idx];
            let taken: HashSet<String> =
                entry.parameters.iter().map(|p| p.name.clone()).collect();
            let mut pname = "__thread_svpos".to_string();
            let mut n = 0;
            while taken.contains(&pname) {
                n += 1;
                pname = format!("__thread_svpos_{n}");
            }
            entry.parameters.push(crate::ast::Parameter {
                ty: Type::Vector { base: crate::ast::ScalarType::Float, width: 4 },
                name: pname.clone(),
                direction: None,
                semantic: Some(Semantic::Name("SV_Position".to_string())),
                array_size: None,
                gs_primitive: None,
                default: None,
                is_uniform: false,
                span: crate::error::Span::default(),
            });
            Expr::Member {
                object: Box::new(Expr::Variable(pname)),
                member: "xy".to_string(),
            }
        }
    };
    // Entry-level builtin calls rewrite to the position value directly
    // (entries feed DXC: no plain params allowed).
    for builtin in &entry_builtins {
        let entry = &mut module.functions[entry_idx];
        for s in &mut entry.body {
            rewrite_builtin_call_stmt(s, builtin, &pos_expr);
        }
    }
    for (fname, param) in triples {
        // Entry call sites take the position value...
        let entry = &mut module.functions[entry_idx];
        for s in &mut entry.body {
            append_call_arg_stmt(s, &fname, &pos_expr);
        }
        // ...threaded-function call sites take their own param (chains).
        for f in module.functions.iter_mut() {
            if f.name == entry_name {
                continue;
            }
            let arg = Expr::Variable(param.clone());
            for s in &mut f.body {
                append_call_arg_stmt(s, &fname, &arg);
            }
        }
    }
    module
}
