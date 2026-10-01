//! Hoist resource members out of IR structs into top-level globals.
//!
//! PSSL Shader Resource Tables freely nest GPU resources inside structs:
//! a struct field may itself be a buffer of textures
//! (`RegularBuffer<Texture2D<float4>> m_albedoTexture`), and a buffer element
//! type may be a struct that itself contains buffers
//! (`RegularBuffer<GeometrySRT>` where `GeometrySRT` has a
//! `RegularBuffer<float3> m_positions` member). DXC rejects both shapes:
//! resources are not allowed as builtin template arguments
//! (`StructuredBuffer<Texture2D<float4>>`, `StructuredBuffer<GeometrySRT>`
//! whose members are buffers) nor as plain global-struct members.
//!
//! The correct translation is descriptor hoisting: each nested resource
//! member `(Struct, member)` becomes a top-level binding, and every access
//! path through it is rewritten:
//!
//! * array-like member, static (index-free) base:
//!   `srt.mat.albedo[e]` → `__hoist_Mat_albedo[e]`
//!   (global keeps the member's buffer-of-textures type);
//! * single buffer/texture/sampler member, indexed base:
//!   `geos[i].positions[j]` → `__hoist_Geo_positions[i][j]`
//!   (global is an array of the member type; the outer index transfers,
//!   preserving the original indexing structure);
//! * single member, static base:
//!   `srt.inner.buf[j]` → `__hoist_Inner_buf[j]`
//!   (one backing store for one static path).
//!
//! Anything else bails out untouched: indexed bases into array-like members
//! would need arrays-of-arrays; materialized struct temporaries (locals),
//! ambiguous call results, and constructed/initializer-list-built values
//! have no statically known backing store. Parameters alias caller tables
//! (same convention as `flatten_srt`).
//!
//! A struct member is stripped from its definition only when every module
//! reference to it was rewritten (or there are none at all); otherwise the
//! definition is kept exactly as before, so no passing shader can regress.
//! Globals are allocated lazily on the first successful rewrite.
//!
//! This runs as a post-lowering IR transform, after `flatten_srt` (which
//! keeps handling root globals that mix resources with scalars).

use std::collections::{HashMap, HashSet};

use crate::ir::*;

/// Buffer-like resource kinds: integer-indexable arrays of elements.
fn is_buffer_kind(kind: &str) -> bool {
    matches!(
        kind,
        "regularbuffer"
            | "structuredbuffer"
            | "databuffer"
            | "rwregularbuffer"
            | "rwstructuredbuffer"
            | "rwdatabuffer"
            | "rwbuffer"
            | "buffer"
            | "appendregularbuffer"
            | "appendstructuredbuffer"
            | "consumeregularbuffer"
            | "consumestructuredbuffer"
            | "bytebuffer"
            | "byteaddressbuffer"
            | "rwbytebuffer"
            | "rwbyteaddressbuffer"
    )
}

pub fn hoist_nested_resources(ir: &mut ShaderModule) {
    // Struct name -> fields.
    let structs: HashMap<String, Vec<(String, IrType)>> = ir
        .structs
        .iter()
        .map(|s| {
            (
                s.name.clone(),
                s.fields.iter().map(|f| (f.name.clone(), f.ty.clone())).collect(),
            )
        })
        .collect();

    // Structs built by value (`Construct` or brace-initialized `VarDecl`)
    // keep all members so initializers keep working.
    let mut pinned: HashSet<String> = HashSet::new();
    for func in &ir.functions {
        for stmt in &func.body {
            collect_pinned_stmt(stmt, &mut pinned);
        }
    }

    // Hoist set: (struct, member) with a direct Resource-typed member,
    // plus resource-like framework objects (`sce::Agc::Core::Texture`,
    // `sce::Psr::TopLevelBvhDescriptor`), which lower to the same
    // top-level bindings and must rewrite/strip exactly like resources
    // (otherwise `srt.m_bvh` uses dangle while the binding exists).
    let mut member_ty: HashMap<(String, String), IrType> = HashMap::new();
    for (sname, fields) in &structs {
        if pinned.contains(sname) {
            continue;
        }
        for (fname, fty) in fields {
            if matches!(fty, IrType::Resource { .. }) {
                member_ty.insert((sname.clone(), fname.clone()), fty.clone());
            } else if let IrType::Named(n) = fty {
                if let Some(res_ty) = crate::framework::framework_resource_ir(n) {
                    member_ty.insert((sname.clone(), fname.clone()), res_ty);
                }
            }
        }
    }
    if member_ty.is_empty() {
        return;
    }
    // Global variable types for path resolution. Resources live in a
    // separate list from plain globals: rebuild their IR types so
    // buffer-typed roots (`RegularBuffer<Geo> geos`) resolve.
    let mut global_tys: HashMap<String, IrType> = ir
        .globals
        .iter()
        .map(|g| (g.name.clone(), g.ty.clone()))
        .collect();
    for r in &ir.resources {
        let access = match r.resource_type {
            ResourceType::RWBuffer | ResourceType::RWTexture => IrAccess::ReadWrite,
            _ => IrAccess::ReadOnly,
        };
        global_tys.insert(
            r.name.clone(),
            IrType::Resource {
                kind: r.kind.clone(),
                access,
                element: Box::new(
                    r.element_type
                        .clone()
                        .unwrap_or(IrType::Named("unknown".to_string())),
                ),
            },
        );
    }

    // Function return types (None = overloaded with differing returns).
    let mut returns: HashMap<String, Option<IrType>> = HashMap::new();
    for func in &ir.functions {
        let ty = func.return_type.clone();
        match returns.get(&func.name) {
            None => {
                returns.insert(func.name.clone(), Some(ty));
            }
            Some(Some(prev)) if *prev != ty => {
                returns.insert(func.name.clone(), None);
            }
            _ => {}
        }
    }

    let mut ctx = Ctx {
        structs,
        member_ty,
        global_tys,
        returns,
        created: HashMap::new(),
        refs: HashMap::new(),
        string_seen: HashSet::new(),
    };

    for func in &mut ir.functions {
        // Locals holding struct values are materialized copies with no
        // statically known backing store: ban them as path roots.
        // Parameters alias caller tables (same convention as flatten_srt).
        let mut var_tys: HashMap<String, IrType> = HashMap::new();
        for p in &func.parameters {
            var_tys.insert(p.name.clone(), p.ty.clone());
        }
        let mut banned: HashSet<String> = HashSet::new();
        collect_locals(&func.body, &mut var_tys, &mut banned);
        let fctx = FnCtx {
            var_tys,
            banned,
            scope: func.scope.clone(),
        };
        for stmt in &mut func.body {
            rewrite_stmt(stmt, &mut ctx, &fctx);
        }
    }

    // Strip a hoisted member from its struct definition only when every
    // module reference to it was rewritten (or there are none and the name
    // never appears in a surviving unparsed method-call string). Dotted
    // call strings (`a.b.Method()`) are not resolved as member uses: a
    // mention there vetoes stripping even with rewritten uses elsewhere,
    // since the mentioned use may be unrewritable (local base) and would
    // dangle. The veto set is recomputed from the FINAL bodies (below):
    // calls already rewritten to hoisted globals no longer mention the
    // member, so only genuinely opaque survivors veto.
    collect_string_seen(ir, &mut ctx.string_seen);
    let mut stripped: HashSet<(String, String)> = HashSet::new();
    for (key, (rewritten, bailed)) in &ctx.refs {
        if *bailed == 0 && *rewritten > 0 && !ctx.string_seen.contains(&key.1) {
            stripped.insert((key.0.clone(), key.1.clone()));
        }
    }
    // Members with zero references of any kind are invisible: strip them
    // unless the name appears in an unparsed call string.
    for key in ctx.member_ty.keys() {
        if !ctx.refs.contains_key(key) && !ctx.string_seen.contains(&key.1) {
            stripped.insert(key.clone());
        }
    }
    for s in &mut ir.structs {
        let sname = s.name.clone();
        s.fields
            .retain(|f| !stripped.contains(&(sname.clone(), f.name.clone())));
    }

    // Emit lazily-created globals for rewritten uses.
    let mut created: Vec<((String, String, bool), String)> = ctx.created.into_iter().collect();
    created.sort_by(|a, b| a.1.cmp(&b.1));
    for ((sname, mname, is_array), gname) in created {
        let used = ctx
            .refs
            .get(&(sname.clone(), mname.clone()))
            .map(|(r, _)| *r)
            .unwrap_or(0);
        if used == 0 {
            continue;
        }
        let mty = match ctx.member_ty.get(&(sname, mname)) {
            Some(t) => t.clone(),
            None => continue,
        };
        let (gty, arr) = hoisted_global_ty(&mty, is_array);
        ir.globals.push(GlobalVar {
            ty: gty,
            name: gname,
            array_size: arr,
            const_value: None,
            shared: false,
        });
    }
}

/// Global type/array-size for a hoisted member. `as_array` is true when the
/// outer struct index transfers onto the global (rule B-indexed).
fn hoisted_global_ty(mty: &IrType, as_array: bool) -> (IrType, Option<u32>) {
    if as_array {
        return (mty.clone(), Some(0));
    }
    match mty {
        IrType::Resource { element, .. } if is_texture_buffer_element(element) => {
            (element.as_ref().clone(), Some(0))
        }
        _ => (mty.clone(), None),
    }
}

/// True when a buffer element is itself a texture object type, i.e. the
/// member is a buffer-of-textures (array-like) rather than a single buffer.
fn is_texture_buffer_element(element: &IrType) -> bool {
    match element {
        IrType::Resource { kind, .. } => {
            matches!(
                kind.as_str(),
                "texture1d"
                    | "texture2d"
                    | "texture2darray"
                    | "texturecube"
                    | "texturecubearray"
                    | "texture3d"
            )
        }
        IrType::Named(n) => {
            matches!(
                n.to_ascii_lowercase().as_str(),
                "texture1d" | "texture2d" | "texturecube" | "texture3d"
            )
        }
        _ => false,
    }
}

struct Ctx {
    structs: HashMap<String, Vec<(String, IrType)>>,
    member_ty: HashMap<(String, String), IrType>,
    global_tys: HashMap<String, IrType>,
    returns: HashMap<String, Option<IrType>>,
    /// (struct, member, as_array) -> global name (created lazily).
    created: HashMap<(String, String, bool), String>,
    /// (struct, member) -> (rewritten, bailed) reference counts.
    refs: HashMap<(String, String), (u32, u32)>,
    /// Identifier segments seen inside dotted method-call strings that were
    /// not parsed (conservative veto against stripping).
    string_seen: HashSet<String>,
}

struct FnCtx {
    var_tys: HashMap<String, IrType>,
    banned: HashSet<String>,
    /// In-struct method scope (`Ret Scope::method`, methods declared inside
    /// `struct Scope`): bare member names resolve through this struct's
    /// fields (C++ member scope: locals/params, then members, then globals).
    /// Without it, `this`-relative paths in method bodies (`m_globals.m_x`)
    /// stay invisible to hoisting while their fields get stripped —
    /// dangling references (DXC ICEs instead of clean errors).
    scope: Option<String>,
}

/// Member type for a bare name inside a method body (`None` when unscoped
/// or not a member), consulted after locals/params but before globals.
fn scope_field_ty(ctx: &Ctx, fctx: &FnCtx, name: &str) -> Option<IrType> {
    let fields = ctx.structs.get(fctx.scope.as_ref()?)?;
    fields.iter().find(|(n, _)| n == name).map(|(_, t)| t.clone())
}

fn mark_ref(ctx: &mut Ctx, sname: &str, mname: &str, rewritten: bool) {
    let e = ctx
        .refs
        .entry((sname.to_string(), mname.to_string()))
        .or_insert((0, 0));
    if rewritten {
        e.0 += 1;
    } else {
        e.1 += 1;
    }
}

fn collect_pinned_stmt(stmt: &IrStmt, out: &mut HashSet<String>) {
    match stmt {
        IrStmt::VarDecl { ty, init, .. } => {
            // Brace-initialized struct values are positional: keep members.
            if let (IrType::Named(s), Some(IrExpr::InitializerList(_))) = (ty, init) {
                out.insert(s.clone());
            }
            if let Some(e) = init {
                collect_pinned_expr(e, out);
            }
        }
        IrStmt::Return(Some(e)) | IrStmt::Expr(e) => collect_pinned_expr(e, out),
        IrStmt::Return(None) => {}
        IrStmt::If {
            condition,
            then_branch,
            else_branch,
        } => {
            collect_pinned_expr(condition, out);
            for s in then_branch {
                collect_pinned_stmt(s, out);
            }
            if let Some(eb) = else_branch {
                for s in eb {
                    collect_pinned_stmt(s, out);
                }
            }
        }
        IrStmt::For {
            init,
            condition,
            update,
            body,
        } => {
            collect_pinned_stmt(init, out);
            if let Some(c) = condition {
                collect_pinned_expr(c, out);
            }
            if let Some(u) = update {
                collect_pinned_expr(u, out);
            }
            for s in body {
                collect_pinned_stmt(s, out);
            }
        }
        IrStmt::While { condition, body } => {
            collect_pinned_expr(condition, out);
            for s in body {
                collect_pinned_stmt(s, out);
            }
        }
        IrStmt::Switch { scrutinee, cases } => {
            collect_pinned_expr(scrutinee, out);
            for case in cases {
                if let Some(label) = &case.label {
                    collect_pinned_expr(label, out);
                }
                for s in &case.body {
                    collect_pinned_stmt(s, out);
                }
            }
        }
        IrStmt::Block(stmts) => {
            for s in stmts {
                collect_pinned_stmt(s, out);
            }
        }
    }
}

fn collect_pinned_expr(e: &IrExpr, out: &mut HashSet<String>) {
    match e {
        IrExpr::Construct { ty: IrType::Named(s), .. } => {
            out.insert(s.clone());
        }
        IrExpr::Variable(_) | IrExpr::Literal(_) => {}
        IrExpr::Member { object, .. } => collect_pinned_expr(object, out),
        IrExpr::Index { object, index } | IrExpr::ResourceIndex { resource: object, index } => {
            collect_pinned_expr(object, out);
            collect_pinned_expr(index, out);
        }
        IrExpr::ResourceLoad {
            resource,
            coordinates,
            sample,
        } => {
            collect_pinned_expr(resource, out);
            collect_pinned_expr(coordinates, out);
            if let Some(s) = sample {
                collect_pinned_expr(s, out);
            }
        }
        IrExpr::Call { arguments, .. } => {
            for a in arguments {
                collect_pinned_expr(a, out);
            }
        }
        IrExpr::Construct { arguments, .. } => {
            for a in arguments {
                collect_pinned_expr(a, out);
            }
        }
        IrExpr::Unary { expr: inner, .. } => collect_pinned_expr(inner, out),
        IrExpr::Binary { left, right, .. } => {
            collect_pinned_expr(left, out);
            collect_pinned_expr(right, out);
        }
        IrExpr::Conditional {
            condition,
            then_branch,
            else_branch,
        } => {
            collect_pinned_expr(condition, out);
            collect_pinned_expr(then_branch, out);
            collect_pinned_expr(else_branch, out);
        }
        IrExpr::Assign { lhs, rhs } => {
            collect_pinned_expr(lhs, out);
            collect_pinned_expr(rhs, out);
        }
        IrExpr::InitializerList(elems) => {
            for x in elems {
                collect_pinned_expr(x, out);
            }
        }
    }
}

/// Collect local variable types; ban struct-typed locals as path roots
/// (materialized copies with no static backing store).
fn collect_locals(
    stmts: &[IrStmt],
    tys: &mut HashMap<String, IrType>,
    banned: &mut HashSet<String>,
) {
    for stmt in stmts {
        match stmt {
            IrStmt::VarDecl { ty, name, .. } => {
                tys.insert(name.clone(), ty.clone());
                if matches!(ty, IrType::Named(_)) {
                    banned.insert(name.clone());
                }
            }
            IrStmt::Return(_) | IrStmt::Expr(_) => {}
            IrStmt::If {
                then_branch,
                else_branch,
                ..
            } => {
                collect_locals(then_branch, tys, banned);
                if let Some(eb) = else_branch {
                    collect_locals(eb, tys, banned);
                }
            }
            IrStmt::For { init, body, .. } => {
                collect_locals(std::slice::from_ref(init.as_ref()), tys, banned);
                collect_locals(body, tys, banned);
            }
            IrStmt::While { body, .. } => collect_locals(body, tys, banned),
            IrStmt::Switch { cases, .. } => {
                for case in cases {
                    collect_locals(&case.body, tys, banned);
                }
            }
            IrStmt::Block(inner) => collect_locals(inner, tys, banned),
        }
    }
}

fn rewrite_stmt(stmt: &mut IrStmt, ctx: &mut Ctx, fctx: &FnCtx) {
    match stmt {
        IrStmt::Return(Some(e)) => {
            rewrite_expr(e, ctx, fctx);
        }
        IrStmt::Return(None) => {}
        IrStmt::VarDecl { init, .. } => {
            if let Some(e) = init {
                rewrite_expr(e, ctx, fctx);
            }
        }
        IrStmt::If {
            condition,
            then_branch,
            else_branch,
        } => {
            rewrite_expr(condition, ctx, fctx);
            for s in then_branch {
                rewrite_stmt(s, ctx, fctx);
            }
            if let Some(eb) = else_branch {
                for s in eb {
                    rewrite_stmt(s, ctx, fctx);
                }
            }
        }
        IrStmt::For {
            init,
            condition,
            update,
            body,
        } => {
            rewrite_stmt(init, ctx, fctx);
            if let Some(c) = condition {
                rewrite_expr(c, ctx, fctx);
            }
            if let Some(u) = update {
                rewrite_expr(u, ctx, fctx);
            }
            for s in body {
                rewrite_stmt(s, ctx, fctx);
            }
        }
        IrStmt::While { condition, body } => {
            rewrite_expr(condition, ctx, fctx);
            for s in body {
                rewrite_stmt(s, ctx, fctx);
            }
        }
        IrStmt::Switch {
            scrutinee,
            cases,
        } => {
            rewrite_expr(scrutinee, ctx, fctx);
            for case in cases {
                if let Some(label) = &mut case.label {
                    rewrite_expr(label, ctx, fctx);
                }
                for s in &mut case.body {
                    rewrite_stmt(s, ctx, fctx);
                }
            }
        }
        IrStmt::Block(inner) => {
            for s in inner {
                rewrite_stmt(s, ctx, fctx);
            }
        }
        IrStmt::Expr(e) => {
            rewrite_expr(e, ctx, fctx);
        }
    }
}

/// Resolve the struct type of a static (index-free, call-free) path root
/// expression: a Variable or Member chain over params/globals.
fn static_path_struct(e: &IrExpr, ctx: &Ctx, fctx: &FnCtx) -> Option<String> {
    match e {
        IrExpr::Variable(v) => {
            if fctx.banned.contains(v) {
                return None;
            }
            // C++ member scope: locals/params, then members, then globals.
            if let Some(ty) = fctx.var_tys.get(v) {
                return match ty {
                    IrType::Named(s) => Some(s.clone()),
                    _ => None,
                };
            }
            if let Some(IrType::Named(s)) = scope_field_ty(ctx, fctx, v) {
                return Some(s);
            }
            match ctx.global_tys.get(v) {
                Some(IrType::Named(s)) => Some(s.clone()),
                _ => None,
            }
        }
        IrExpr::Member { object, member } => {
            let base = static_path_struct(object, ctx, fctx)?;
            let fields = ctx.structs.get(&base)?;
            let fty = fields.iter().find(|(n, _)| n == member).map(|(_, t)| t)?;
            match fty {
                IrType::Named(s) => Some(s.clone()),
                _ => None,
            }
        }
        _ => None,
    }
}

/// True when the subtree is a static path: only Variable/Member nodes.
fn is_static_path(e: &IrExpr) -> bool {
    match e {
        IrExpr::Variable(_) => true,
        IrExpr::Member { object, .. } => is_static_path(object),
        _ => false,
    }
}

/// Resolve an indexable base to its buffer element struct.
fn indexed_base_struct(e: &IrExpr, ctx: &Ctx, fctx: &FnCtx) -> Option<String> {
    let ty = path_type(e, ctx, fctx)?;
    match ty {
        IrType::Resource {
            kind, element, ..
        } => {
            if !is_buffer_kind(&kind) {
                return None;
            }
            match element.as_ref() {
                IrType::Named(s) => Some(s.clone()),
                _ => None,
            }
        }
        _ => None,
    }
}

/// Resolve the IR type of a Variable/Member path expression. Call results
/// resolve through unambiguous function return types.
fn path_type(e: &IrExpr, ctx: &Ctx, fctx: &FnCtx) -> Option<IrType> {
    match e {
        IrExpr::Variable(v) => {
            if fctx.banned.contains(v) {
                return None;
            }
            // C++ member scope: locals/params, then members, then globals.
            if let Some(ty) = fctx.var_tys.get(v) {
                return Some(ty.clone());
            }
            if let Some(ty) = scope_field_ty(ctx, fctx, v) {
                return Some(ty);
            }
            ctx.global_tys.get(v).cloned()
        }
        IrExpr::Member { object, member } => {
            let base = path_type(object, ctx, fctx)?;
            match base {
                IrType::Named(s) => {
                    let fields = ctx.structs.get(&s)?;
                    fields.iter().find(|(n, _)| n == member).map(|(_, t)| t.clone())
                }
                _ => None,
            }
        }
        IrExpr::Call { function, .. } => {
            // Plain function name (no dots): look up the return type.
            // Dotted method paths are not return-typed here.
            if function.contains('.') {
                return None;
            }
            match ctx.returns.get(function) {
                Some(Some(ty)) => Some(ty.clone()),
                _ => None,
            }
        }
        _ => None,
    }
}

fn member_is_array_like(ctx: &Ctx, sname: &str, mname: &str) -> Option<bool> {
    let mty = ctx.member_ty.get(&(sname.to_string(), mname.to_string()))?;
    match mty {
        IrType::Resource { kind, .. } => Some(is_buffer_kind(kind)),
        _ => None,
    }
}

/// True when the member is a buffer whose element is itself a texture
/// object (buffer-of-textures). Such members are already array-like AND
/// their hoisted form is an array; stacking another outer index would need
/// arrays-of-arrays (illegal), so indexed bases bail out for these.
fn member_is_texture_array(ctx: &Ctx, sname: &str, mname: &str) -> bool {
    match ctx.member_ty.get(&(sname.to_string(), mname.to_string())) {
        Some(IrType::Resource { element, .. }) => is_texture_buffer_element(element),
        _ => false,
    }
}

fn hoist_global_name(sname: &str, mname: &str, as_array: bool) -> String {
    if as_array {
        format!("__hoist_{sname}_{mname}_arr")
    } else {
        format!("__hoist_{sname}_{mname}")
    }
}

/// Allocate (lazily) the hoisted global for `(struct, member, form)` and
/// return its name.
fn hoisted(ctx: &mut Ctx, sname: &str, mname: &str, as_array: bool) -> String {
    let key = (sname.to_string(), mname.to_string(), as_array);
    if let Some(g) = ctx.created.get(&key) {
        return g.clone();
    }
    let g = hoist_global_name(sname, mname, as_array);
    ctx.created.insert(key, g.clone());
    g
}

fn rewrite_expr(e: &mut IrExpr, ctx: &mut Ctx, fctx: &FnCtx) {
    match e {
        IrExpr::Literal(_) => {}
        IrExpr::Variable(v) => {
            // Bare resource members in method bodies (`buf[i]` where `buf`
            // is a member of the method's scope struct): the field is
            // stripped when hoisted, so rewrite to the hoisted global.
            // Locals/params win (C++ shadowing); like members, they beat
            // globals. Non-resource members, unknown names, and globals
            // stay untouched.
            if fctx.var_tys.contains_key(v) || fctx.banned.contains(v) {
                return;
            }
            let Some(scope) = fctx.scope.as_ref() else {
                return;
            };
            if !ctx.member_ty.contains_key(&(scope.clone(), v.clone())) {
                return;
            }
            let g = hoisted(ctx, scope, v, false);
            mark_ref(ctx, scope, v, true);
            *e = IrExpr::Variable(g);
        }
        IrExpr::Member { object, member } => {
            // Bottom-up: rewrite the object first.
            rewrite_expr(object, ctx, fctx);
            // Rule B-indexed: `Base[idx].member` (single member).
            // Clone the index first so the assignment below has no
            // live borrows.
            // Rule B-indexed (single member) and rule A-indexed
            // (array-like non-texture member): transfer the outer index
            // onto the array global. Texture-array members would need
            // arrays-of-arrays: bail out for those.
            let idx_hit: Option<(String, IrExpr)> = match object.as_ref() {
                IrExpr::Index { object: base, index } if is_static_path(base) => {
                    match indexed_base_struct(base, ctx, fctx) {
                        Some(sname)
                            if member_is_array_like(ctx, &sname, member) == Some(false) =>
                        {
                            Some((sname, index.as_ref().clone()))
                        }
                        Some(sname)
                            if member_is_array_like(ctx, &sname, member) == Some(true)
                                && !member_is_texture_array(ctx, &sname, member) =>
                        {
                            Some((sname, index.as_ref().clone()))
                        }
                        Some(sname)
                            if member_is_array_like(ctx, &sname, member).is_some() =>
                        {
                            let b = base.as_ref().clone();
                            mark_bailed_if_hoisted(ctx, fctx, &b, &sname, member);
                            None
                        }
                        _ => None,
                    }
                }
                IrExpr::Index { .. } => {
                    return;
                }
                _ => None,
            };
            if let Some((sname, idx)) = idx_hit {
                let g = hoisted(ctx, &sname, member, true);
                mark_ref(ctx, &sname, member, true);
                *e = IrExpr::Index {
                    object: Box::new(IrExpr::Variable(g)),
                    index: Box::new(idx),
                };
                return;
            }
            if matches!(object.as_ref(), IrExpr::Index { .. }) {
                return;
            }
            // Rule B-indexed through a resource-indexed base
            // (`srt.buf[i].member`). Clone the index first so the
            // assignment below has no live borrows.
            let ri_hit: Option<(String, IrExpr)> = match object.as_ref() {
                IrExpr::ResourceIndex { resource: base, index }
                    if is_static_path(base) =>
                {
                    match indexed_base_struct(base, ctx, fctx) {
                        Some(sname)
                            if member_is_array_like(ctx, &sname, member) == Some(false) =>
                        {
                            Some((sname, index.as_ref().clone()))
                        }
                        Some(sname)
                            if member_is_array_like(ctx, &sname, member) == Some(true)
                                && !member_is_texture_array(ctx, &sname, member) =>
                        {
                            Some((sname, index.as_ref().clone()))
                        }
                        Some(sname)
                            if member_is_array_like(ctx, &sname, member).is_some() =>
                        {
                            let b = base.as_ref().clone();
                            mark_bailed_if_hoisted(ctx, fctx, &b, &sname, member);
                            None
                        }
                        _ => None,
                    }
                }
                IrExpr::ResourceIndex { .. } => {
                    return;
                }
                _ => None,
            };
            if let Some((sname, idx)) = ri_hit {
                let g = hoisted(ctx, &sname, member, true);
                mark_ref(ctx, &sname, member, true);
                *e = IrExpr::Index {
                    object: Box::new(IrExpr::Variable(g)),
                    index: Box::new(idx),
                };
                return;
            }
            if matches!(object.as_ref(), IrExpr::ResourceIndex { .. }) {
                return;
            }
            // Rules A / B-static: static base, no index.
            if !is_static_path(object) {
                mark_bailed_call_base(ctx, object, member);
                return;
            }
            let Some(sname) = static_path_struct(object, ctx, fctx) else {
                // Unresolvable base (banned materialized local, untyped
                // root, ...): the use can never rewrite, so it vetoes
                // stripping exactly like a non-static base. Without this,
                // the use stays invisible to refcounting while other
                // (rewritten) uses strip the member out from under it.
                mark_bailed_call_base(ctx, object, member);
                return;
            };
            if member_is_array_like(ctx, &sname, member).is_none() {
                return;
            }
            let g = hoisted(ctx, &sname, member, false);
            mark_ref(ctx, &sname, member, true);
            *e = IrExpr::Variable(g);
        }
        IrExpr::Index { object, index } => {
            rewrite_expr(object, ctx, fctx);
            rewrite_expr(index, ctx, fctx);
        }
        IrExpr::ResourceIndex { resource, index } => {
            rewrite_expr(resource, ctx, fctx);
            rewrite_expr(index, ctx, fctx);
        }
        IrExpr::ResourceLoad {
            resource,
            coordinates,
            sample,
        } => {
            rewrite_expr(resource, ctx, fctx);
            rewrite_expr(coordinates, ctx, fctx);
            if let Some(s) = sample {
                rewrite_expr(s, ctx, fctx);
            }
        }
        IrExpr::Call { function, arguments } => {
            for a in arguments {
                rewrite_expr(a, ctx, fctx);
            }
            rewrite_call_function(function, ctx, fctx);
        }
        IrExpr::Construct { arguments, .. } => {
            for a in arguments {
                rewrite_expr(a, ctx, fctx);
            }
        }
        IrExpr::Unary { expr: inner, .. } => rewrite_expr(inner, ctx, fctx),
        IrExpr::Binary { left, right, .. } => {
            rewrite_expr(left, ctx, fctx);
            rewrite_expr(right, ctx, fctx);
        }
        IrExpr::Conditional {
            condition,
            then_branch,
            else_branch,
        } => {
            rewrite_expr(condition, ctx, fctx);
            rewrite_expr(then_branch, ctx, fctx);
            rewrite_expr(else_branch, ctx, fctx);
        }
        IrExpr::Assign { lhs, rhs } => {
            rewrite_expr(lhs, ctx, fctx);
            rewrite_expr(rhs, ctx, fctx);
        }
        IrExpr::InitializerList(elems) => {
            for x in elems {
                rewrite_expr(x, ctx, fctx);
            }
        }
    }
}

/// Mark a bailed reference when the base resolves to a struct with this
/// hoisted member (keeps the definition so dangling references cannot occur).
fn mark_bailed_if_hoisted(
    ctx: &mut Ctx,
    fctx: &FnCtx,
    base: &IrExpr,
    sname: &str,
    member: &str,
) {
    let _ = fctx;
    let _ = base;
    mark_ref(ctx, sname, member, false);
}

/// Mark a bailed reference for a non-static (call/materialized) base when
/// the member name is hoisted in any struct.
fn mark_bailed_call_base(ctx: &mut Ctx, object: &IrExpr, member: &str) {
    let _ = object;
    // Conservative: any struct hoisting this member name pins its definition.
    let structs: Vec<String> = ctx
        .member_ty
        .keys()
        .filter(|(_, m)| m == member)
        .map(|(s, _)| s.clone())
        .collect();
    for s in structs {
        mark_ref(ctx, &s, member, false);
    }
}

/// Recompute the string-seen veto set from final bodies: every dotted
/// identifier segment of a surviving method-call string. Calls already
/// rewritten to hoisted globals (`__hoist_S_m[i].Sample`) contribute
/// only their global/index/method segments, so successfully absorbed
/// members no longer veto their own stripping.
fn collect_string_seen(ir: &ShaderModule, out: &mut HashSet<String>) {
    fn expr(e: &IrExpr, out: &mut HashSet<String>) {
        match e {
            IrExpr::Call { function, arguments } => {
                if function.contains('.') {
                    for seg in function.split(|c| c == '.' || c == '[' || c == ']' || c == '(' || c == ')') {
                        if !seg.is_empty() {
                            out.insert(seg.to_string());
                        }
                    }
                }
                for a in arguments {
                    expr(a, out);
                }
            }
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
                if let Some(sample) = sample {
                    expr(sample, out);
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
            IrExpr::Variable(_) | IrExpr::Literal(_) => {}
        }
    }
    fn stmt(s: &IrStmt, out: &mut HashSet<String>) {
        match s {
            IrStmt::Return(e) => {
                if let Some(e) = e {
                    expr(e, out);
                }
            }
            IrStmt::VarDecl { init, .. } => {
                if let Some(init) = init {
                    expr(init, out);
                }
            }
            IrStmt::Block(inner) => {
                for s in inner {
                    stmt(s, out);
                }
            }
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
                if let Some(condition) = condition {
                    expr(condition, out);
                }
                if let Some(update) = update {
                    expr(update, out);
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
            IrStmt::Expr(e) => expr(e, out),
        }
    }
    for f in &ir.functions {
        for s in &f.body {
            stmt(s, out);
        }
    }
}

/// Rewrite dotted method-call paths (`a.b[i].m.Load`, `a.b.m[e].Sample`).
fn rewrite_call_function(function: &mut String, ctx: &mut Ctx, fctx: &FnCtx) {
    if !function.contains('.') {
        return;
    }
    // Split trailing `.Method`.
    let dot = match function.rfind('.') {
        Some(d) => d,
        None => return,
    };
    let (obj, method) = (function[..dot].to_string(), function[dot + 1..].to_string());
    if method.is_empty() || method.contains('[') || method.contains(']') {
        return;
    }
    // Optional trailing `[idx]` (belongs to the member for rule A).
    let mut trailing_idx: Option<String> = None;
    let mut obj2 = obj.clone();
    if obj2.ends_with(']') {
        let open = match obj2.rfind('[') {
            Some(o) => o,
            None => return,
        };
        trailing_idx = Some(obj2[open..].to_string());
        obj2 = obj2[..open].to_string();
    }
    // Now `obj2` is `P.member` or `P[idx].member`.
    let mdot = match obj2.rfind('.') {
        Some(d) => d,
        None => return,
    };
    let (base, member) = (obj2[..mdot].to_string(), obj2[mdot + 1..].to_string());
    if member.is_empty() || member.contains('[') || member.contains(']') {
        return;
    }
    // Optional `[idx]` on the base (rule B-indexed).
    let mut base_idx: Option<String> = None;
    let mut base_path = base.clone();
    if base_path.ends_with(']') {
        let open = match base_path.rfind('[') {
            Some(o) => o,
            None => return,
        };
        base_idx = Some(base_path[open..].to_string());
        base_path = base_path[..open].to_string();
    }
    if base_path.contains('[') || base_path.contains(']') || base_path.contains('(') {
        return;
    }
    // Resolve the base path type segment by segment.
    let Some(base_ty) = resolve_string_type(&base_path, ctx, fctx) else {
        return;
    };
    let (sname, base_is_buffer) = match &base_ty {
        IrType::Named(s) => (s.clone(), false),
        IrType::Resource {
            kind, element, ..
        } if is_buffer_kind(kind) => match element.as_ref() {
            IrType::Named(s) => (s.clone(), true),
            _ => return,
        },
        _ => return,
    };
    // Rule B-indexed requires a buffer base; rules A/B-static require a
    // plain struct base.
    if base_idx.is_some() != base_is_buffer {
        mark_ref(ctx, &sname, &member, false);
        return;
    }
    let is_arr = member_is_array_like(ctx, &sname, &member);
    let tail = trailing_idx.unwrap_or_default();
    // Rule A-indexed needs a non-texture array-like member (texture
    // arrays would need arrays-of-arrays).
    let a_indexed = is_arr == Some(true)
        && base_idx.is_some()
        && !member_is_texture_array(ctx, &sname, &member);
    let base_idx_str = base_idx.clone().unwrap_or_default();
    match (is_arr, base_idx) {
        // Rule A: array-like member, index-free base, index after member.
        (Some(true), None) => {
            let g = hoisted(ctx, &sname, &member, false);
            mark_ref(ctx, &sname, &member, true);
            *function = format!("{g}{tail}.{method}");
        }
        // Rule A-indexed: array-like non-texture member, indexed base.
        _ if a_indexed => {
            let g = hoisted(ctx, &sname, &member, true);
            mark_ref(ctx, &sname, &member, true);
            *function = format!("{g}{base_idx_str}{tail}.{method}");
        }
        // Rule B-indexed: single member, indexed base.
        (Some(false), Some(idx)) => {
            let g = hoisted(ctx, &sname, &member, true);
            mark_ref(ctx, &sname, &member, true);
            *function = format!("{g}{idx}{tail}.{method}");
        }
        // Rule B-static: single member, static base.
        (Some(false), None) => {
            let g = hoisted(ctx, &sname, &member, false);
            mark_ref(ctx, &sname, &member, true);
            *function = format!("{g}{tail}.{method}");
        }
        _ => {
            mark_ref(ctx, &sname, &member, false);
        }
    }
}

/// Resolve a bracket-free dotted path (`a.b.c`) to its IR type by walking
/// variables and struct fields.
fn resolve_string_type(path: &str, ctx: &Ctx, fctx: &FnCtx) -> Option<IrType> {
    let mut segs = path.split('.');
    let root = segs.next()?;
    if root.is_empty() || fctx.banned.contains(root) {
        return None;
    }
    // C++ member scope: locals/params, then members, then globals.
    let mut cur = fctx
        .var_tys
        .get(root)
        .cloned()
        .or_else(|| scope_field_ty(ctx, fctx, root))
        .or_else(|| ctx.global_tys.get(root).cloned())?;
    for seg in segs {
        if seg.is_empty() {
            return None;
        }
        let IrType::Named(s) = cur else {
            return None;
        };
        let fields = ctx.structs.get(&s)?;
        let fty = fields.iter().find(|(n, _)| n == seg).map(|(_, t)| t)?;
        cur = fty.clone();
    }
    Some(cur)
}

/// Splits buffers of structs with texture/sampler members.
///
/// `StructuredBuffer<ShaderMaterial> buf` (where `ShaderMaterial` has
/// `Texture2D` members) is illegal HLSL/DXC ("object ... is not allowed
/// in builtin template parameters"). The Vulkan translation splits the
/// storage while keeping every use shape working:
///
/// * the buffer keeps its name but takes a generated scalars-only
///   element struct (`ShaderMaterialScalars`), so scalar member uses
///   (`buf[i].k`) pass through untouched;
/// * each texture/sampler member `m` becomes an unbounded array
///   (`Texture2D<E> buf_m_arr[]`), and `buf[i].m` rewrites to it;
/// * whole-struct values (`S x = buf[i]`, `f(buf[i])`) materialize
///   full-`S` locals from both pieces (HLSL struct locals and opaque
///   assignments are legal; struct *construction* is not).
///
/// Only direct texture/sampler members qualify (nested structs must be
/// all-scalar and stay verbatim). Helpers taking split buffers by value
/// (`f(StructuredBuffer<S> buf, ...)` with `S v = buf[i]` uses) are
/// handled afterwards by [`split_helper_buffer_params`], which re-points
/// them at the caller's buffer and drops the parameter.
pub fn split_textured_struct_buffers(ir: &mut ShaderModule) {
    use std::collections::{BTreeMap, BTreeSet};

    // Struct name -> (field name, field type, array size).
    let structs: HashMap<String, Vec<(String, IrType, Option<u32>)>> = ir
        .structs
        .iter()
        .map(|s| {
            (
                s.name.clone(),
                s.fields
                    .iter()
                    .map(|f| (f.name.clone(), f.ty.clone(), f.array_size))
                    .collect(),
            )
        })
        .collect();

    fn is_texture_member(fty: &IrType) -> bool {
        match fty {
            IrType::Resource { kind, .. } => {
                let k = kind.as_str();
                k.starts_with("texture") || k == "mstexture2d"
            }
            // Framework texture objects (`sce::Agc::Core::Texture`) lower
            // to `Texture2D<float4>` bindings and must split exactly like
            // ordinary texture members (else the element struct nests a
            // resource: DXC `object ... is not allowed in builtin template
            // parameters`). Precedent: `is_sampler_member` below.
            IrType::Named(n) => {
                crate::framework::framework_resource_kind(n) == Some("texture2d")
            }
            _ => false,
        }
    }
    fn is_sampler_member(fty: &IrType) -> bool {
        match fty {
            IrType::Named(n) => crate::framework::is_sampler_type_name(n),
            _ => false,
        }
    }
    // Direct buffer members (`RegularBuffer<float3> m_positions`,
    // `DataBuffer<uint> m_indexBuffer`): PSSL pointers-in-structs with
    // no HLSL spelling, exactly like texture members. Each splits to
    // an unbounded array of same-typed buffers
    // (`StructuredBuffer<float3> buf_m_positions_arr[]`), with
    // `buf[i].m[j]` rewriting to `buf_m_arr[i][j]` through the shared
    // index-transfer machinery.
    fn is_buffer_member(fty: &IrType) -> bool {
        match fty {
            IrType::Resource { kind, .. } => {
                matches!(
                    kind.as_str(),
                    "regularbuffer"
                        | "structuredbuffer"
                        | "databuffer"
                        | "buffer"
                        | "bytebuffer"
                        | "byteaddressbuffer"
                        | "rwregularbuffer"
                        | "rwstructuredbuffer"
                        | "rwdatabuffer"
                        | "rwbuffer"
                        | "rwbytebuffer"
                        | "rwbyteaddressbuffer"
                )
            }
            _ => false,
        }
    }
    // True when a buffer member's element can live in a split binding:
    // scalars/vectors/matrices, or structs with no resource members at
    // any depth (a resourceful element would re-nest the illegality the
    // split removes).
    fn buffer_element_ok(
        ty: &IrType,
        structs: &HashMap<String, Vec<(String, IrType, Option<u32>)>>,
        depth: usize,
    ) -> bool {
        if depth > 8 {
            return false;
        }
        match ty {
            IrType::Bool
            | IrType::Int
            | IrType::UInt
            | IrType::Float
            | IrType::Half
            | IrType::Double
            | IrType::Vector { .. }
            | IrType::Matrix { .. } => true,
            IrType::Named(n) => structs.get(n).is_some_and(|fields| {
                fields.iter().all(|(_, fty, _)| match fty {
                    IrType::Bool
                    | IrType::Int
                    | IrType::UInt
                    | IrType::Float
                    | IrType::Half
                    | IrType::Double
                    | IrType::Vector { .. }
                    | IrType::Matrix { .. } => true,
                    IrType::Named(m) => {
                        buffer_element_ok(&IrType::Named(m.clone()), structs, depth + 1)
                    }
                    _ => false,
                })
            }),
            _ => false,
        }
    }
    // True when the struct is all scalars/vectors/matrices (recursively),
    // i.e. safe to keep verbatim inside a buffer element twin.
    fn all_scalar(
        sname: &str,
        structs: &HashMap<String, Vec<(String, IrType, Option<u32>)>>,
        depth: usize,
    ) -> bool {
        if depth > 8 {
            return false;
        }
        let Some(fields) = structs.get(sname) else {
            return false;
        };
        fields.iter().all(|(_, fty, _)| match fty {
            IrType::Bool
            | IrType::Int
            | IrType::UInt
            | IrType::Float
            | IrType::Half
            | IrType::Double
            | IrType::Vector { .. }
            | IrType::Matrix { .. } => true,
            IrType::Named(n) => all_scalar(n, structs, depth + 1),
            _ => false,
        })
    }
    // Splittable: at least one direct texture/sampler/buffer member, at
    // least one scalar member, and nothing else but scalars / all-scalar
    // nested structs. Scalar arrays are fine (kept in the twin,
    // unrolled on materialize); texture/sampler/buffer arrays and nested
    // resourceful structs disqualify (buffer elements must themselves be
    // resource-free, else the split binding re-nests the illegality).
    fn splittable(
        sname: &str,
        structs: &HashMap<String, Vec<(String, IrType, Option<u32>)>>,
    ) -> bool {
        let Some(fields) = structs.get(sname) else {
            return false;
        };
        let mut has_split = false;
        let mut has_scalar = false;
        for (_, fty, array_size) in fields {
            if is_texture_member(fty) || is_sampler_member(fty) {
                if array_size.is_some() {
                    return false;
                }
                has_split = true;
            } else if is_buffer_member(fty) {
                if array_size.is_some() {
                    return false;
                }
                if let IrType::Resource { element, .. } = fty {
                    if !buffer_element_ok(element, structs, 0) {
                        return false;
                    }
                } else {
                    return false;
                }
                has_split = true;
            } else if let IrType::Named(n) = fty {
                if !all_scalar(n, structs, 0) {
                    return false;
                }
                has_scalar = true;
            } else if matches!(
                fty,
                IrType::Bool
                    | IrType::Int
                    | IrType::UInt
                    | IrType::Float
                    | IrType::Half
                    | IrType::Double
                    | IrType::Vector { .. }
                    | IrType::Matrix { .. }
            ) {
                has_scalar = true;
            } else {
                return false;
            }
        }
        has_split && has_scalar
    }

    // Element structs used as any function parameter type cannot always
    // split (callers construct full values). The veto narrows to the
    // shapes that actually need a whole-`S` buffer element:
    // * structs with sampler members (splitting those is historically
    //   fragile: shared-array reuse across buffers), or
    // * structs with a call site feeding a whole buffer element
    //   (`f(buf[i])`) directly.
    // Texture-only and buffer-only structs whose call sites all pass
    // locals/values split safely: `S x = buf[i]` materializes from the
    // twin + arrays (buffer assignments between struct members are
    // plain opaque copies, exactly like textures), and scalar member
    // uses read the twin in place. Buffer bindings are per-buffer (no
    // sharing), so the sampler fragility does not apply to them.
    let mut param_structs: HashSet<String> = HashSet::new();
    for f in &ir.functions {
        for p in &f.parameters {
            if let IrType::Named(s) = &p.ty {
                param_structs.insert(s.clone());
            }
        }
    }

    // Buffer index -> element struct, for splittable read-only structured
    // buffers of splittable structs. Hoist-created top-level buffer
    // globals (`__hoist_S_m` for a whole single buffer member `m`) nest
    // element structs the same way (`StructuredBuffer<S-with-textures>`
    // is just as illegal from a global), so they split too — but only
    // whole single buffers (`array_size.is_none()`, not member arrays,
    // whose index chains have no member path to rewrite) and never
    // shared-memory globals.
    let mut splits: BTreeMap<usize, String> = BTreeMap::new();
    let mut gsplits: BTreeMap<usize, String> = BTreeMap::new();
    for (i, r) in ir.resources.iter().enumerate() {
        if !matches!(r.resource_type, ResourceType::StructuredBuffer) {
            continue;
        }
        if !matches!(
            r.kind.as_str(),
            "regularbuffer" | "structuredbuffer" | "databuffer" | "buffer"
        ) {
            continue;
        }
        let Some(IrType::Named(s)) = r.element_type.as_ref() else {
            continue;
        };
        // NOTE: the parameter veto is applied narrowed below (after all
        // candidates are known); do NOT `continue` here.
        if splittable(s, &structs) {
            splits.insert(i, s.clone());
        }
    }
    for (i, g) in ir.globals.iter().enumerate() {
        if g.shared || g.array_size.is_some() {
            continue;
        }
        let IrType::Resource { kind, element, .. } = &g.ty else {
            continue;
        };
        if !matches!(
            kind.as_str(),
            "regularbuffer" | "structuredbuffer" | "databuffer" | "buffer"
        ) {
            continue;
        }
        let IrType::Named(s) = element.as_ref() else {
            continue;
        };
        if splittable(s, &structs) {
            gsplits.insert(i, s.clone());
        }
    }
    // Narrowed parameter veto: drop candidates whose struct is taken by
    // value while needing a whole-`S` buffer element (sampler members,
    // or a call site feeding `buf[i]` directly). Texture-only structs
    // passed as locals/values stay split.
    if (!splits.is_empty() || !gsplits.is_empty()) && !param_structs.is_empty() {
        let buf_struct: HashMap<String, String> = splits
            .iter()
            .map(|(i, s)| (ir.resources[*i].name.clone(), s.clone()))
            .chain(
                gsplits
                    .iter()
                    .map(|(i, s)| (ir.globals[*i].name.clone(), s.clone())),
            )
            .collect();
        let fed_directly = buffer_index_call_args(ir, &buf_struct);
        let veto = |s: &String| {
            if !param_structs.contains(s) {
                return false;
            }
            let has_sampler = structs
                .get(s)
                .is_some_and(|fields| fields.iter().any(|(_, fty, _)| is_sampler_member(fty)));
            if has_sampler {
                return true;
            }
            fed_directly.contains(s)
        };
        splits.retain(|_, s| !veto(s));
        gsplits.retain(|_, s| !veto(s));
    }
    if splits.is_empty() && gsplits.is_empty() {
        return;
    }

    // Scalars-only twin per element struct (shared across buffers).
    let mut twins: HashMap<String, String> = HashMap::new();
    for s in splits.values().chain(gsplits.values()) {
        if twins.contains_key(s) {
            continue;
        }
        let mut twin = format!("{s}Scalars");
        let mut n = 0u32;
        while ir.structs.iter().any(|d| d.name == twin) {
            n += 1;
            twin = format!("{s}Scalars{n}");
        }
        twins.insert(s.clone(), twin);
    }
    for (s, twin) in &twins {
        let fields = structs.get(s).cloned().unwrap_or_default();
        ir.structs.push(StructDecl {
            name: twin.clone(),
            fields: fields
                .iter()
                .filter_map(|(fname, fty, array_size)| {
                    if is_texture_member(fty) || is_sampler_member(fty) || is_buffer_member(fty) {
                        return None;
                    }
                    Some(FieldDecl {
                        ty: fty.clone(),
                        name: fname.clone(),
                        semantic: None,
                        array_size: *array_size,
                    })
                })
                .collect(),
        });
    }
    // Point buffers at their twins.
    for (i, s) in &splits {
        if let Some(twin) = twins.get(s) {
            if let Some(r) = ir.resources.get_mut(*i) {
                r.element_type = Some(IrType::Named(twin.clone()));
            }
        }
    }
    for (i, s) in &gsplits {
        if let Some(twin) = twins.get(s) {
            if let Some(g) = ir.globals.get_mut(*i) {
                if let IrType::Resource { element, .. } = &mut g.ty {
                    *element = Box::new(IrType::Named(twin.clone()));
                }
            }
        }
    }

    // Per-buffer texture/sampler array bindings. When hoisting already
    // created its shared array for this (struct, member) — for direct
    // global-path uses it rewrote — reuse it instead of allocating a
    // duplicate, but only when this is the sole split buffer of the
    // struct (shared content guaranteed); otherwise per-buffer arrays
    // stay correct each.
    let mut buf_count: HashMap<&String, usize> = HashMap::new();
    for s in splits.values().chain(gsplits.values()) {
        *buf_count.entry(s).or_default() += 1;
    }
    // (buffer name, element struct) for every split buffer, resource or
    // hoisted global alike.
    let split_bufs: Vec<(String, String)> = splits
        .iter()
        .map(|(i, s)| (ir.resources[*i].name.clone(), s.clone()))
        .chain(
            gsplits
                .iter()
                .map(|(i, s)| (ir.globals[*i].name.clone(), s.clone())),
        )
        .collect();
    let mut bindings: HashMap<(String, String), String> = HashMap::new();
    for (buf, s) in &split_bufs {
        let fields = structs.get(s).cloned().unwrap_or_default();
        for (fname, fty, _) in &fields {
            // Binding element type for splittable texture/sampler/buffer
            // members; None (skipped) otherwise.
            enum SplitMember {
                Texture(IrType),
                Sampler,
                Buffer { kind: String, access: IrAccess, element: IrType },
            }
            let split = match fty {
                IrType::Resource { element, .. } if is_texture_member(fty) => {
                    Some(SplitMember::Texture(element.as_ref().clone()))
                }
                // Framework texture objects lower to `Texture2D<float4>`.
                _ if is_texture_member(fty) => Some(SplitMember::Texture(IrType::Vector {
                    base: IrScalar::Float,
                    width: 4,
                })),
                _ if is_sampler_member(fty) => Some(SplitMember::Sampler),
                // Buffer members keep kind/access/element: the binding is
                // an unbounded array of same-typed buffers.
                IrType::Resource { kind, access, element } if is_buffer_member(fty) => {
                    Some(SplitMember::Buffer {
                        kind: kind.clone(),
                        access: *access,
                        element: element.as_ref().clone(),
                    })
                }
                _ => None,
            };
            let Some(split) = split else {
                continue;
            };
            let hoisted_name = format!("__hoist_{s}_{fname}_arr");
            let sole = buf_count.get(s).copied().unwrap_or(0) == 1;
            if sole
                && ir.globals.iter().any(|g| {
                    g.name == hoisted_name && g.array_size.is_some() && !g.shared
                })
            {
                bindings.insert((buf.clone(), fname.clone()), hoisted_name);
                continue;
            }
            let mut binding = format!("{buf}_{fname}_arr");
            let mut n = 0u32;
            while ir.globals.iter().any(|g| g.name == binding)
                || ir.resources.iter().any(|r| r.name == binding)
            {
                n += 1;
                binding = format!("{buf}_{fname}_arr{n}");
            }
            bindings.insert((buf.clone(), fname.clone()), binding.clone());
            match split {
                // Sampler binding: plain SamplerState array (probed valid).
                SplitMember::Sampler => ir.globals.push(GlobalVar {
                    ty: IrType::Named("SamplerState".to_string()),
                    name: binding,
                    array_size: Some(0),
                    const_value: None,
                    shared: false,
                }),
                SplitMember::Texture(elem) => ir.globals.push(GlobalVar {
                    ty: IrType::Resource {
                        kind: match fty {
                            IrType::Resource { kind, .. } => kind.clone(),
                            _ => "texture2d".to_string(),
                        },
                        access: IrAccess::ReadOnly,
                        element: Box::new(elem),
                    },
                    name: binding,
                    array_size: Some(0),
                    const_value: None,
                    shared: false,
                }),
                // Buffer binding: same kind/access/element, unbounded.
                SplitMember::Buffer { kind, access, element } => ir.globals.push(GlobalVar {
                    ty: IrType::Resource {
                        kind,
                        access,
                        element: Box::new(element),
                    },
                    name: binding,
                    array_size: Some(0),
                    const_value: None,
                    shared: false,
                }),
            }
        }
    }

    // Rewrite uses in every function body (skipping functions where a
    // split buffer name is shadowed by a parameter or local: those
    // refer to a different object). Buffer arrays (`StructuredBuffer<S>
    // bufs[N]`, e.g. per-LOD mesh tables) stay out of the map: `bufs[i]`
    // is a BUFFER, never a struct value, and the twin binding already
    // carries the dimension — so double-index uses (`bufs[i][j].m`) are
    // valid untouched, while whole-value materialization would mistreat
    // `bufs[i]` as a struct (member-on-buffer errors). Resource-member
    // paths through array buffers keep today's loud path.
    let mut buf_struct: HashMap<String, String> = HashMap::new();
    for (buf, s) in &split_bufs {
        let is_array = ir
            .resources
            .iter()
            .any(|r| &r.name == buf && r.array_size.is_some())
            || ir
                .globals
                .iter()
                .any(|g| &g.name == buf && g.array_size.is_some());
        if is_array {
            continue;
        }
        buf_struct.insert(buf.clone(), s.clone());
    }
    for function in &mut ir.functions {
        if split_name_shadowed(&function.body, &function.parameters, &buf_struct) {
            continue;
        }
        let mut counter = 0u32;
        rewrite_split_uses(
            &mut function.body,
            &buf_struct,
            &bindings,
            &structs,
            &mut counter,
        );
    }

    // Dead stores into split members (`globals.m_mats = twinbuf`) dangle
    // once the member keeps its template-nesting declaration: drop them
    // and strip the member (see below).
    drop_dead_split_member_stores(ir, &structs, &twins);

    // Helpers taking split buffers by value (`IblResult
    // f(StructuredBuffer<S> buf, uint idx, ...)` with `S v = buf[idx]`
    // materializations): re-point the body at the caller's buffer and
    // drop the parameter (a by-value struct parameter cannot be split
    // at call sites — the limitation noted above). The existing
    // materializer then expands the element from the twin + arrays.
    split_helper_buffer_params(ir, &structs, &twins, &buf_struct, &bindings);
}

/// Re-points helper functions that take a split buffer by value at the
/// caller's buffer and drops the parameter, so the existing
/// whole-struct materialization handles the element.
///
/// For a free function `f` (never the entry: driver-facing signatures
/// stay) with a `StructuredBuffer<S>`-shaped parameter `B` (`S`
/// splittable with a twin) where every `B` use materializes an element
/// (`S v = B[<any>]`), and every call site passes the same known-split
/// buffer `BUF` for `B`: drop `B`, rewrite body `B` uses to `BUF`, drop
/// the `BUF` argument at calls, and run the split materializer over
/// the body. Anything else (multiple buffers, non-materialization
/// uses, whole-`B` stores, scoped methods, overload ambiguity, name
/// collisions) keeps today's loud path.
fn split_helper_buffer_params(
    ir: &mut ShaderModule,
    structs: &HashMap<String, Vec<(String, IrType, Option<u32>)>>,
    twins: &HashMap<String, String>,
    buf_struct: &HashMap<String, String>,
    bindings: &HashMap<(String, String), String>,
) {
    use std::collections::HashSet;

    fn buffer_kinds(kind: &str) -> bool {
        matches!(
            kind,
            "regularbuffer" | "structuredbuffer" | "databuffer" | "buffer"
        )
    }

    // Whole-`B` stores (`B = x`): the single unsound shape for the
    // re-pointing below (rebinding a parameter vs a global differ).
    // Member/index writes share the buffer object either way.
    fn has_whole_store(body: &[IrStmt], name: &str) -> bool {
        fn expr(e: &IrExpr, name: &str) -> bool {
            match e {
                IrExpr::Assign { lhs, rhs } => {
                    matches!(lhs.as_ref(), IrExpr::Variable(n) if n == name)
                        || expr(rhs, name)
                }
                IrExpr::Variable(_) | IrExpr::Literal(_) => false,
                IrExpr::Member { object, .. } => expr(object, name),
                IrExpr::Index { object, index }
                | IrExpr::ResourceIndex { resource: object, index } => {
                    expr(object, name) || expr(index, name)
                }
                IrExpr::ResourceLoad {
                    resource,
                    coordinates,
                    sample,
                } => {
                    expr(resource, name)
                        || expr(coordinates, name)
                        || sample.as_ref().is_some_and(|s| expr(s, name))
                }
                IrExpr::Call { arguments, .. } => {
                    arguments.iter().any(|a| expr(a, name))
                }
                IrExpr::Construct { arguments, .. } => {
                    arguments.iter().any(|a| expr(a, name))
                }
                IrExpr::Unary { expr: inner, .. } => expr(inner, name),
                IrExpr::Binary { left, right, .. } => {
                    expr(left, name) || expr(right, name)
                }
                IrExpr::Conditional {
                    condition,
                    then_branch,
                    else_branch,
                } => {
                    expr(condition, name)
                        || expr(then_branch, name)
                        || expr(else_branch, name)
                }
                IrExpr::InitializerList(elems) => elems.iter().any(|x| expr(x, name)),
            }
        }
        fn stmt(s: &IrStmt, name: &str) -> bool {
            match s {
                IrStmt::Return(e) => e.as_ref().is_some_and(|e| expr(e, name)),
                IrStmt::Block(ss) => ss.iter().any(|s| stmt(s, name)),
                IrStmt::If {
                    condition,
                    then_branch,
                    else_branch,
                } => {
                    expr(condition, name)
                        || then_branch.iter().any(|s| stmt(s, name))
                        || else_branch.as_ref().is_some_and(|b| b.iter().any(|s| stmt(s, name)))
                }
                IrStmt::For {
                    init,
                    condition,
                    update,
                    body,
                } => {
                    stmt(init, name)
                        || condition.as_ref().is_some_and(|c| expr(c, name))
                        || update.as_ref().is_some_and(|u| expr(u, name))
                        || body.iter().any(|s| stmt(s, name))
                }
                IrStmt::While { condition, body } => {
                    expr(condition, name)
                        || body.iter().any(|s| stmt(s, name))
                }
                IrStmt::Switch { scrutinee, cases } => {
                    expr(scrutinee, name)
                        || cases.iter().any(|c| {
                            c.label.as_ref().is_some_and(|l| expr(l, name))
                                || c.body.iter().any(|s| stmt(s, name))
                        })
                }
                IrStmt::VarDecl { init, .. } => {
                    init.as_ref().is_some_and(|e| expr(e, name))
                }
                IrStmt::Expr(e) => expr(e, name),
            }
        }
        body.iter().any(|s| stmt(s, name))
    }

    // Locally declared names in a body (for shadowing checks).
    fn declared_names(body: &[IrStmt], out: &mut HashSet<String>) {
        fn stmt(s: &IrStmt, out: &mut HashSet<String>) {
            match s {
                IrStmt::VarDecl { name, .. } => {
                    out.insert(name.clone());
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
        for s in body {
            stmt(s, out);
        }
    }

    // Rewrites every `Variable(from)` to `Variable(to)` in a body.
    fn rename_var_stmt(s: &mut IrStmt, from: &str, to: &str) {
        fn expr(e: &mut IrExpr, from: &str, to: &str) {
            match e {
                IrExpr::Variable(n) => {
                    if n == from {
                        *n = to.to_string();
                    }
                }
                IrExpr::Member { object, .. } => expr(object, from, to),
                IrExpr::Index { object, index }
                | IrExpr::ResourceIndex { resource: object, index } => {
                    expr(object, from, to);
                    expr(index, from, to);
                }
                IrExpr::ResourceLoad {
                    resource,
                    coordinates,
                    sample,
                } => {
                    expr(resource, from, to);
                    expr(coordinates, from, to);
                    if let Some(s) = sample {
                        expr(s, from, to);
                    }
                }
                IrExpr::Call { arguments, .. } => {
                    for a in arguments {
                        expr(a, from, to);
                    }
                }
                IrExpr::Construct { arguments, .. } => {
                    for a in arguments {
                        expr(a, from, to);
                    }
                }
                IrExpr::Unary { expr: inner, .. } => expr(inner, from, to),
                IrExpr::Binary { left, right, .. } => {
                    expr(left, from, to);
                    expr(right, from, to);
                }
                IrExpr::Conditional {
                    condition,
                    then_branch,
                    else_branch,
                } => {
                    expr(condition, from, to);
                    expr(then_branch, from, to);
                    expr(else_branch, from, to);
                }
                IrExpr::Assign { lhs, rhs } => {
                    expr(lhs, from, to);
                    expr(rhs, from, to);
                }
                IrExpr::InitializerList(elems) => {
                    for x in elems {
                        expr(x, from, to);
                    }
                }
                IrExpr::Literal(_) => {}
            }
        }
        match s {
            IrStmt::Return(e) => {
                if let Some(e) = e {
                    expr(e, from, to);
                }
            }
            IrStmt::Block(ss) => {
                for s in ss {
                    rename_var_stmt(s, from, to);
                }
            }
            IrStmt::If {
                condition,
                then_branch,
                else_branch,
            } => {
                expr(condition, from, to);
                for s in then_branch {
                    rename_var_stmt(s, from, to);
                }
                if let Some(b) = else_branch {
                    for s in b {
                        rename_var_stmt(s, from, to);
                    }
                }
            }
            IrStmt::For {
                init,
                condition,
                update,
                body,
            } => {
                rename_var_stmt(init, from, to);
                if let Some(c) = condition {
                    expr(c, from, to);
                }
                if let Some(u) = update {
                    expr(u, from, to);
                }
                for s in body {
                    rename_var_stmt(s, from, to);
                }
            }
            IrStmt::While { condition, body } => {
                expr(condition, from, to);
                for s in body {
                    rename_var_stmt(s, from, to);
                }
            }
            IrStmt::Switch { scrutinee, cases } => {
                expr(scrutinee, from, to);
                for c in cases {
                    if let Some(label) = &mut c.label {
                        expr(label, from, to);
                    }
                    for s in &mut c.body {
                        rename_var_stmt(s, from, to);
                    }
                }
            }
            IrStmt::VarDecl { init, .. } => {
                if let Some(e) = init {
                    expr(e, from, to);
                }
            }
            IrStmt::Expr(e) => {
                expr(e, from, to);
            }
        }
    }

    // Calls to `fname` with exactly `arity` args: caller index + the
    // buffer-slot argument (must be a bare variable), if every match
    // qualifies.
    fn collect_calls(
        ir: &ShaderModule,
        fname: &str,
        arity: usize,
        bi: usize,
        elem_struct: &str,
        buf_struct: &HashMap<String, String>,
    ) -> Option<Vec<(usize, String)>> {
        fn expr(
            e: &IrExpr,
            fname: &str,
            arity: usize,
            bi: usize,
            elem_struct: &str,
            buf_struct: &HashMap<String, String>,
            out: &mut Vec<(usize, String)>,
            ok: &mut bool,
            fi: usize,
        ) {
            match e {
                IrExpr::Call { function, arguments } => {
                    if function == fname && arguments.len() == arity {
                        match arguments.get(bi) {
                            Some(IrExpr::Variable(buf))
                                if buf_struct.get(buf).is_some_and(|s| s == elem_struct) =>
                            {
                                out.push((fi, buf.clone()));
                            }
                            _ => {
                                *ok = false;
                            }
                        }
                    }
                    for a in arguments {
                        expr(a, fname, arity, bi, elem_struct, buf_struct, out, ok, fi);
                    }
                }
                IrExpr::Member { object, .. } => {
                    expr(object, fname, arity, bi, elem_struct, buf_struct, out, ok, fi)
                }
                IrExpr::Index { object, index }
                | IrExpr::ResourceIndex { resource: object, index } => {
                    expr(object, fname, arity, bi, elem_struct, buf_struct, out, ok, fi);
                    expr(index, fname, arity, bi, elem_struct, buf_struct, out, ok, fi);
                }
                IrExpr::ResourceLoad {
                    resource,
                    coordinates,
                    sample,
                } => {
                    expr(resource, fname, arity, bi, elem_struct, buf_struct, out, ok, fi);
                    expr(
                        coordinates,
                        fname,
                        arity,
                        bi,
                        elem_struct,
                        buf_struct,
                        out,
                        ok,
                        fi,
                    );
                    if let Some(s) = sample {
                        expr(s, fname, arity, bi, elem_struct, buf_struct, out, ok, fi);
                    }
                }
                IrExpr::Construct { arguments, .. } => {
                    for a in arguments {
                        expr(a, fname, arity, bi, elem_struct, buf_struct, out, ok, fi);
                    }
                }
                IrExpr::Unary { expr: inner, .. } => {
                    expr(inner, fname, arity, bi, elem_struct, buf_struct, out, ok, fi)
                }
                IrExpr::Binary { left, right, .. } => {
                    expr(left, fname, arity, bi, elem_struct, buf_struct, out, ok, fi);
                    expr(right, fname, arity, bi, elem_struct, buf_struct, out, ok, fi);
                }
                IrExpr::Conditional {
                    condition,
                    then_branch,
                    else_branch,
                } => {
                    expr(condition, fname, arity, bi, elem_struct, buf_struct, out, ok, fi);
                    expr(then_branch, fname, arity, bi, elem_struct, buf_struct, out, ok, fi);
                    expr(else_branch, fname, arity, bi, elem_struct, buf_struct, out, ok, fi);
                }
                IrExpr::Assign { lhs, rhs } => {
                    expr(lhs, fname, arity, bi, elem_struct, buf_struct, out, ok, fi);
                    expr(rhs, fname, arity, bi, elem_struct, buf_struct, out, ok, fi);
                }
                IrExpr::InitializerList(elems) => {
                    for x in elems {
                        expr(x, fname, arity, bi, elem_struct, buf_struct, out, ok, fi);
                    }
                }
                IrExpr::Variable(_) | IrExpr::Literal(_) => {}
            }
        }
        fn stmt(
            s: &IrStmt,
            fname: &str,
            arity: usize,
            bi: usize,
            elem_struct: &str,
            buf_struct: &HashMap<String, String>,
            out: &mut Vec<(usize, String)>,
            ok: &mut bool,
            fi: usize,
        ) {
            match s {
                IrStmt::Return(e) => {
                    if let Some(e) = e {
                        expr(e, fname, arity, bi, elem_struct, buf_struct, out, ok, fi);
                    }
                }
                IrStmt::Block(ss) => {
                    for s in ss {
                        stmt(s, fname, arity, bi, elem_struct, buf_struct, out, ok, fi);
                    }
                }
                IrStmt::If {
                    condition,
                    then_branch,
                    else_branch,
                } => {
                    expr(condition, fname, arity, bi, elem_struct, buf_struct, out, ok, fi);
                    for s in then_branch {
                        stmt(s, fname, arity, bi, elem_struct, buf_struct, out, ok, fi);
                    }
                    if let Some(b) = else_branch {
                        for s in b {
                            stmt(s, fname, arity, bi, elem_struct, buf_struct, out, ok, fi);
                        }
                    }
                }
                IrStmt::For {
                    init,
                    condition,
                    update,
                    body,
                } => {
                    stmt(init, fname, arity, bi, elem_struct, buf_struct, out, ok, fi);
                    if let Some(c) = condition {
                        expr(c, fname, arity, bi, elem_struct, buf_struct, out, ok, fi);
                    }
                    if let Some(u) = update {
                        expr(u, fname, arity, bi, elem_struct, buf_struct, out, ok, fi);
                    }
                    for s in body {
                        stmt(s, fname, arity, bi, elem_struct, buf_struct, out, ok, fi);
                    }
                }
                IrStmt::While { condition, body } => {
                    expr(condition, fname, arity, bi, elem_struct, buf_struct, out, ok, fi);
                    for s in body {
                        stmt(s, fname, arity, bi, elem_struct, buf_struct, out, ok, fi);
                    }
                }
                IrStmt::Switch { scrutinee, cases } => {
                    expr(scrutinee, fname, arity, bi, elem_struct, buf_struct, out, ok, fi);
                    for c in cases {
                        if let Some(label) = &c.label {
                            expr(label, fname, arity, bi, elem_struct, buf_struct, out, ok, fi);
                        }
                        for s in &c.body {
                            stmt(s, fname, arity, bi, elem_struct, buf_struct, out, ok, fi);
                        }
                    }
                }
                IrStmt::VarDecl { init, .. } => {
                    if let Some(e) = init {
                        expr(e, fname, arity, bi, elem_struct, buf_struct, out, ok, fi);
                    }
                }
                IrStmt::Expr(e) => {
                    expr(e, fname, arity, bi, elem_struct, buf_struct, out, ok, fi);
                }
            }
        }
        let mut out = Vec::new();
        let mut ok = true;
        for (fi, f) in ir.functions.iter().enumerate() {
            for s in &f.body {
                stmt(s, fname, arity, bi, elem_struct, buf_struct, &mut out, &mut ok, fi);
            }
        }
        if !ok || out.is_empty() {
            return None;
        }
        Some(out)
    }

    // Drops argument `slot` from calls to (`fname`, `arity`) in a body.
    fn drop_call_slot(body: &mut [IrStmt], fname: &str, arity: usize, slot: usize) {
        fn expr(e: &mut IrExpr, fname: &str, arity: usize, slot: usize) {
            match e {
                IrExpr::Call { function, arguments } => {
                    if function == fname && arguments.len() == arity && slot < arguments.len() {
                        arguments.remove(slot);
                    }
                    for a in arguments {
                        expr(a, fname, arity, slot);
                    }
                }
                IrExpr::Member { object, .. } => expr(object, fname, arity, slot),
                IrExpr::Index { object, index }
                | IrExpr::ResourceIndex { resource: object, index } => {
                    expr(object, fname, arity, slot);
                    expr(index, fname, arity, slot);
                }
                IrExpr::ResourceLoad {
                    resource,
                    coordinates,
                    sample,
                } => {
                    expr(resource, fname, arity, slot);
                    expr(coordinates, fname, arity, slot);
                    if let Some(s) = sample {
                        expr(s, fname, arity, slot);
                    }
                }
                IrExpr::Construct { arguments, .. } => {
                    for a in arguments {
                        expr(a, fname, arity, slot);
                    }
                }
                IrExpr::Unary { expr: inner, .. } => expr(inner, fname, arity, slot),
                IrExpr::Binary { left, right, .. } => {
                    expr(left, fname, arity, slot);
                    expr(right, fname, arity, slot);
                }
                IrExpr::Conditional {
                    condition,
                    then_branch,
                    else_branch,
                } => {
                    expr(condition, fname, arity, slot);
                    expr(then_branch, fname, arity, slot);
                    expr(else_branch, fname, arity, slot);
                }
                IrExpr::Assign { lhs, rhs } => {
                    expr(lhs, fname, arity, slot);
                    expr(rhs, fname, arity, slot);
                }
                IrExpr::InitializerList(elems) => {
                    for x in elems {
                        expr(x, fname, arity, slot);
                    }
                }
                IrExpr::Variable(_) | IrExpr::Literal(_) => {}
            }
        }
        for s in body.iter_mut() {
            match s {
                IrStmt::Return(e) => {
                    if let Some(e) = e {
                        expr(e, fname, arity, slot);
                    }
                }
                IrStmt::Block(ss) => drop_call_slot(ss, fname, arity, slot),
                IrStmt::If {
                    condition,
                    then_branch,
                    else_branch,
                } => {
                    expr(condition, fname, arity, slot);
                    drop_call_slot(then_branch, fname, arity, slot);
                    if let Some(b) = else_branch {
                        drop_call_slot(b, fname, arity, slot);
                    }
                }
                IrStmt::For {
                    init,
                    condition,
                    update,
                    body,
                } => {
                    drop_call_slot(std::slice::from_mut(init), fname, arity, slot);
                    if let Some(c) = condition {
                        expr(c, fname, arity, slot);
                    }
                    if let Some(u) = update {
                        expr(u, fname, arity, slot);
                    }
                    drop_call_slot(body, fname, arity, slot);
                }
                IrStmt::While { condition, body } => {
                    expr(condition, fname, arity, slot);
                    drop_call_slot(body, fname, arity, slot);
                }
                IrStmt::Switch { scrutinee, cases } => {
                    expr(scrutinee, fname, arity, slot);
                    for c in cases {
                        if let Some(label) = &mut c.label {
                            expr(label, fname, arity, slot);
                        }
                        drop_call_slot(&mut c.body, fname, arity, slot);
                    }
                }
                IrStmt::VarDecl { init, .. } => {
                    if let Some(e) = init {
                        expr(e, fname, arity, slot);
                    }
                }
                IrStmt::Expr(e) => {
                    expr(e, fname, arity, slot);
                }
            }
        }
    }

    let entry_name = ir.entry_point.clone();
    // One buffer parameter per function in this pass (multiple
    // independent pairs stay loud).
    let mut candidates: Vec<(usize, usize, String)> = Vec::new();
    for (fi, f) in ir.functions.iter().enumerate() {
        if f.scope.is_some() {
            continue;
        }
        if entry_name.as_deref() == Some(f.name.as_str()) {
            continue;
        }
        if ir.functions.iter().filter(|g| g.name == f.name).count() != 1 {
            continue;
        }
        let mut found: Vec<(usize, String)> = Vec::new();
        for (bi, p) in f.parameters.iter().enumerate() {
            if p.array_size.is_some() {
                continue;
            }
            if let IrType::Resource { kind, element, .. } = &p.ty {
                if !buffer_kinds(kind) {
                    continue;
                }
                if let IrType::Named(s) = element.as_ref() {
                    if twins.contains_key(s) {
                        found.push((bi, s.clone()));
                    }
                }
            }
        }
        if found.len() == 1 {
            candidates.push((fi, found[0].0, found[0].1.clone()));
        }
    }
    for (fi, bi, elem_struct) in candidates {
        let fname = ir.functions[fi].name.clone();
        let arity = ir.functions[fi].parameters.len();
        let bname = ir.functions[fi].parameters[bi].name.clone();
        // No whole rebinds; no shadowing locals (the parameter itself
        // is being removed); the shared buffer must not collide either.
        if has_whole_store(&ir.functions[fi].body, &bname) {
            continue;
        }
        // Every call site converts: bare calls with matching arity whose
        // buffer slot holds the same known-split buffer. Exactly one
        // distinct buffer across all sites.
        let sites = match collect_calls(
            ir,
            &fname,
            arity,
            bi,
            &elem_struct,
            buf_struct,
        ) {
            Some(sites) => sites,
            None => continue,
        };
        let mut bufs: HashSet<String> = HashSet::new();
        for (_, b) in &sites {
            bufs.insert(b.clone());
        }
        if bufs.len() != 1 {
            continue;
        }
        let buf = bufs.into_iter().next().unwrap();
        if buf == bname {
            continue;
        }
        {
            let f = &ir.functions[fi];
            if f.parameters.iter().any(|p| p.name == buf) {
                continue;
            }
            let mut declared = HashSet::new();
            declared_names(&f.body, &mut declared);
            if declared.contains(&bname) || declared.contains(&buf) {
                continue;
            }
        }
        // Shadow guard for the rewrite below (mirrors the split pass's
        // own rule).
        if split_name_shadowed(
            &ir.functions[fi].body,
            &ir.functions[fi].parameters,
            buf_struct,
        ) {
            continue;
        }
        // Transform: drop `B`, rename uses to `BUF`, drop the argument
        // at calls, then materialize from the known-split buffer.
        ir.functions[fi].parameters.remove(bi);
        for s in &mut ir.functions[fi].body {
            rename_var_stmt(s, &bname, &buf);
        }
        for g in &mut ir.functions {
            drop_call_slot(&mut g.body, &fname, arity, bi);
        }
        {
            let f = &mut ir.functions[fi];
            if split_name_shadowed(&f.body, &f.parameters, buf_struct) {
                continue;
            }
            let mut counter = 0u32;
            rewrite_split_uses(&mut f.body, buf_struct, bindings, structs, &mut counter);
        }
    }
}

/// Drops dead stores into split members and strips those members.
///
/// After splitting, whole-buffer stores into a split member slot
/// (`globals.m_instanceIdToMaterials = forwardSRT_…_materials`, now
/// twin-typed) mismatch the member's full-typed declaration — and the
/// declaration itself (`StructuredBuffer<MaterialSRT>` inside the
/// struct) is a template-nesting error. Both vanish together when the
/// member is never read: the stores are side-effect-free (purity
/// gated), and stripping the member lets hoisting's zero-reference
/// rule... (hoisting already ran; the definition edit below is
/// direct).
///
/// Sound by construction: a slot `(struct S, member m)` qualifies only
/// when `m` is a buffer of a twinned element struct `T`, no
/// `.m` member-read occurs module-wide (buffer-index reads were
/// already rewritten to per-member arrays, which carry different
/// names), `S` is never positionally constructed (brace/constructor
/// initializers would misalign), and every dropped store is call-free
/// (right-hand side and index expressions). Only failing shaders reach
/// here (splittable buffers never compile un-split), so the worst case
/// is staying loud — passing shaders are untouched.
fn drop_dead_split_member_stores(
    ir: &mut ShaderModule,
    structs: &HashMap<String, Vec<(String, IrType, Option<u32>)>>,
    twins: &HashMap<String, String>,
) {
    use std::collections::HashSet;

    fn is_split_buffer_member(fty: &IrType, twins: &HashMap<String, String>) -> bool {
        match fty {
            IrType::Resource { kind, element, .. } => {
                matches!(
                    kind.as_str(),
                    "regularbuffer" | "structuredbuffer" | "databuffer" | "buffer"
                ) && matches!(element.as_ref(), IrType::Named(t) if twins.contains_key(t))
            }
            _ => false,
        }
    }

    // Candidate slots: (struct, member) with a buffer-of-twinned member.
    let mut slots: Vec<(String, String)> = Vec::new();
    for (sname, fields) in structs {
        for (fname, fty, _) in fields {
            if is_split_buffer_member(fty, twins)
                && !slots.iter().any(|(s, m)| s == sname && m == fname)
            {
                slots.push((sname.clone(), fname.clone()));
            }
        }
    }
    if slots.is_empty() {
        return;
    }

    // Member names read anywhere (write positions = outermost member of
    // an assignment-target root chain — excluded).
    fn collect_read_members(e: &IrExpr, out: &mut HashSet<String>, lhs_root: bool) {
        match e {
            IrExpr::Member { object, member } => {
                if !lhs_root {
                    out.insert(member.clone());
                }
                collect_read_members(object, out, false);
            }
            IrExpr::Index { object, index } | IrExpr::ResourceIndex { resource: object, index } => {
                collect_read_members(object, out, lhs_root);
                collect_read_members(index, out, false);
            }
            IrExpr::Assign { lhs, rhs } => {
                collect_read_members(lhs, out, true);
                collect_read_members(rhs, out, false);
            }
            IrExpr::Call { function, arguments } => {
                // Dotted callee (`buf[i].m.Sample(...)`): every segment
                // but the trailing method is a read (indices evaluate,
                // member paths resolve). Over-marking the method name is
                // harmless (never a member slot).
                if function.contains('.') {
                    let mut segs: Vec<&str> = function
                        .split(|c| c == '.' || c == '[' || c == ']' || c == '(' || c == ')')
                        .filter(|s| !s.is_empty())
                        .collect();
                    segs.pop();
                    for seg in segs {
                        out.insert(seg.to_string());
                    }
                }
                for a in arguments {
                    collect_read_members(a, out, false);
                }
            }
            IrExpr::Construct { arguments, .. } => {
                for a in arguments {
                    collect_read_members(a, out, false);
                }
            }
            IrExpr::ResourceLoad {
                resource,
                coordinates,
                sample,
            } => {
                collect_read_members(resource, out, false);
                collect_read_members(coordinates, out, false);
                if let Some(sample) = sample {
                    collect_read_members(sample, out, false);
                }
            }
            IrExpr::Unary { expr: inner, .. } => collect_read_members(inner, out, false),
            IrExpr::Binary { left, right, .. } => {
                collect_read_members(left, out, false);
                collect_read_members(right, out, false);
            }
            IrExpr::Conditional {
                condition,
                then_branch,
                else_branch,
            } => {
                collect_read_members(condition, out, false);
                collect_read_members(then_branch, out, false);
                collect_read_members(else_branch, out, false);
            }
            IrExpr::InitializerList(elems) => {
                for x in elems {
                    collect_read_members(x, out, false);
                }
            }
            IrExpr::Variable(_) | IrExpr::Literal(_) => {}
        }
    }
    fn collect_stmt_reads(s: &IrStmt, out: &mut HashSet<String>) {
        match s {
            IrStmt::Return(e) => {
                if let Some(e) = e {
                    collect_read_members(e, out, false);
                }
            }
            IrStmt::VarDecl { init, .. } => {
                if let Some(init) = init {
                    collect_read_members(init, out, false);
                }
            }
            IrStmt::Block(inner) => {
                for s in inner {
                    collect_stmt_reads(s, out);
                }
            }
            IrStmt::If {
                condition,
                then_branch,
                else_branch,
            } => {
                collect_read_members(condition, out, false);
                for s in then_branch {
                    collect_stmt_reads(s, out);
                }
                if let Some(else_branch) = else_branch {
                    for s in else_branch {
                        collect_stmt_reads(s, out);
                    }
                }
            }
            IrStmt::For {
                init,
                condition,
                update,
                body,
            } => {
                collect_stmt_reads(init, out);
                if let Some(condition) = condition {
                    collect_read_members(condition, out, false);
                }
                if let Some(update) = update {
                    collect_read_members(update, out, false);
                }
                for s in body {
                    collect_stmt_reads(s, out);
                }
            }
            IrStmt::While { condition, body } => {
                collect_read_members(condition, out, false);
                for s in body {
                    collect_stmt_reads(s, out);
                }
            }
            IrStmt::Switch { scrutinee, cases } => {
                collect_read_members(scrutinee, out, false);
                for c in cases {
                    if let Some(label) = &c.label {
                        collect_read_members(label, out, false);
                    }
                    for s in &c.body {
                        collect_stmt_reads(s, out);
                    }
                }
            }
            IrStmt::Expr(e) => collect_read_members(e, out, false),
        }
    }
    let mut read_members: HashSet<String> = HashSet::new();
    for f in &ir.functions {
        for s in &f.body {
            collect_stmt_reads(s, &mut read_members);
        }
    }

    // Structs built positionally (constructor calls or brace
    // initializers): member removal would misalign arguments.
    fn struct_built_positionally(s: &IrStmt, out: &mut HashSet<String>) {
        match s {
            IrStmt::VarDecl { ty, init, .. } => {
                if let IrType::Named(n) = ty {
                    if matches!(init, Some(IrExpr::InitializerList(_))) {
                        out.insert(n.clone());
                    }
                }
                if let Some(IrExpr::Construct { ty, .. }) = init.as_ref() {
                    if let IrType::Named(n) = ty {
                        out.insert(n.clone());
                    }
                }
            }
            IrStmt::Return(e) => {
                if let Some(IrExpr::Construct { ty, .. }) = e.as_ref() {
                    if let IrType::Named(n) = ty {
                        out.insert(n.clone());
                    }
                }
            }
            IrStmt::Block(inner) => {
                for s in inner {
                    struct_built_positionally(s, out);
                }
            }
            IrStmt::If {
                then_branch,
                else_branch,
                ..
            } => {
                for s in then_branch {
                    struct_built_positionally(s, out);
                }
                if let Some(b) = else_branch {
                    for s in b {
                        struct_built_positionally(s, out);
                    }
                }
            }
            IrStmt::For { init, body, .. } => {
                struct_built_positionally(init, out);
                for s in body {
                    struct_built_positionally(s, out);
                }
            }
            IrStmt::While { body, .. } => {
                for s in body {
                    struct_built_positionally(s, out);
                }
            }
            IrStmt::Switch { cases, .. } => {
                for c in cases {
                    for s in &c.body {
                        struct_built_positionally(s, out);
                    }
                }
            }
            IrStmt::Expr(e) => {
                fn find_construct(e: &IrExpr, out: &mut HashSet<String>) {
                    match e {
                        IrExpr::Construct { ty, arguments } => {
                            if let IrType::Named(n) = ty {
                                out.insert(n.clone());
                            }
                            for a in arguments {
                                find_construct(a, out);
                            }
                        }
                        IrExpr::Call { arguments, .. } => {
                            for a in arguments {
                                find_construct(a, out);
                            }
                        }
                        IrExpr::Member { object, .. } => find_construct(object, out),
                        IrExpr::Index { object, index }
                        | IrExpr::ResourceIndex { resource: object, index } => {
                            find_construct(object, out);
                            find_construct(index, out);
                        }
                        IrExpr::ResourceLoad {
                            resource,
                            coordinates,
                            sample,
                        } => {
                            find_construct(resource, out);
                            find_construct(coordinates, out);
                            if let Some(sample) = sample {
                                find_construct(sample, out);
                            }
                        }
                        IrExpr::Unary { expr: inner, .. } => find_construct(inner, out),
                        IrExpr::Binary { left, right, .. } => {
                            find_construct(left, out);
                            find_construct(right, out);
                        }
                        IrExpr::Conditional {
                            condition,
                            then_branch,
                            else_branch,
                        } => {
                            find_construct(condition, out);
                            find_construct(then_branch, out);
                            find_construct(else_branch, out);
                        }
                        IrExpr::Assign { lhs, rhs } => {
                            find_construct(lhs, out);
                            find_construct(rhs, out);
                        }
                        IrExpr::InitializerList(elems) => {
                            for x in elems {
                                find_construct(x, out);
                            }
                        }
                        IrExpr::Variable(_) | IrExpr::Literal(_) => {}
                    }
                }
                find_construct(e, out);
            }
        }
    }
    let mut constructed: HashSet<String> = HashSet::new();
    for f in &ir.functions {
        for s in &f.body {
            struct_built_positionally(s, &mut constructed);
        }
    }

    // Outermost member of an assignment-target root chain (`x.m`,
    // `x.m[i]` both write `m`).
    fn target_outermost_member(e: &IrExpr) -> Option<&str> {
        match e {
            IrExpr::Member { member, .. } => Some(member.as_str()),
            IrExpr::Index { object, .. } | IrExpr::ResourceIndex { resource: object, .. } => {
                target_outermost_member(object)
            }
            _ => None,
        }
    }
    fn target_has_call(e: &IrExpr) -> bool {
        match e {
            IrExpr::Variable(_) => false,
            IrExpr::Member { object, .. } => target_has_call(object),
            IrExpr::Index { object, index } | IrExpr::ResourceIndex { resource: object, index } => {
                target_has_call(object) || expr_has_call(index)
            }
            other => expr_has_call(other),
        }
    }
    fn expr_has_call(e: &IrExpr) -> bool {
        match e {
            IrExpr::Call { .. } => true,
            IrExpr::Variable(_) | IrExpr::Literal(_) => false,
            IrExpr::Member { object, .. } => expr_has_call(object),
            IrExpr::Index { object, index } | IrExpr::ResourceIndex { resource: object, index } => {
                expr_has_call(object) || expr_has_call(index)
            }
            IrExpr::ResourceLoad {
                resource,
                coordinates,
                sample,
            } => {
                expr_has_call(resource)
                    || expr_has_call(coordinates)
                    || sample.as_ref().is_some_and(|s| expr_has_call(s))
            }
            IrExpr::Construct { arguments, .. } => arguments.iter().any(expr_has_call),
            IrExpr::Unary { expr: inner, .. } => expr_has_call(inner),
            IrExpr::Binary { left, right, .. } => expr_has_call(left) || expr_has_call(right),
            IrExpr::Conditional {
                condition,
                then_branch,
                else_branch,
            } => {
                expr_has_call(condition) || expr_has_call(then_branch) || expr_has_call(else_branch)
            }
            IrExpr::Assign { lhs, rhs } => expr_has_call(lhs) || expr_has_call(rhs),
            IrExpr::InitializerList(elems) => elems.iter().any(expr_has_call),
        }
    }

    // Live slots first: any read of the member name keeps every slot
    // with that name (name-shared across structs stays conservative).
    let mut dead: Vec<(String, String)> = slots
        .into_iter()
        .filter(|(s, m)| !read_members.contains(m) && !constructed.contains(s))
        .collect();
    if dead.is_empty() {
        return;
    }

    // Drop pure stores into dead slots; a slot with an impure store
    // stays (loud) — remove it from the strip set. Sub-expression
    // stores (`f(x.m = v)`, conditions, initializers) can never drop
    // (their value feeds the enclosing expression), so any such store
    // disqualifies its slot up front.
    //
    // Three phases over all functions (shared `dead` set): flag
    // sub-expression stores, drop pure top-level stores (recording
    // impure survivors), then strip.
    fn flag_subexpr_stores(stmts: &[IrStmt], dead_names: &HashSet<String>, bad: &mut HashSet<String>) {
        fn scan_expr(e: &IrExpr, dead_names: &HashSet<String>, bad: &mut HashSet<String>, top: bool) {
            if let IrExpr::Assign { lhs, rhs } = e {
                if !top {
                    if let Some(m) = target_outermost_member(lhs) {
                        if dead_names.contains(m) {
                            bad.insert(m.to_string());
                        }
                    }
                }
                scan_expr(lhs, dead_names, bad, false);
                scan_expr(rhs, dead_names, bad, false);
                return;
            }
            match e {
                IrExpr::Member { object, .. } => scan_expr(object, dead_names, bad, false),
                IrExpr::Index { object, index }
                | IrExpr::ResourceIndex { resource: object, index } => {
                    scan_expr(object, dead_names, bad, false);
                    scan_expr(index, dead_names, bad, false);
                }
                IrExpr::ResourceLoad {
                    resource,
                    coordinates,
                    sample,
                } => {
                    scan_expr(resource, dead_names, bad, false);
                    scan_expr(coordinates, dead_names, bad, false);
                    if let Some(sample) = sample {
                        scan_expr(sample, dead_names, bad, false);
                    }
                }
                IrExpr::Call { arguments, .. } | IrExpr::Construct { arguments, .. } => {
                    for a in arguments {
                        scan_expr(a, dead_names, bad, false);
                    }
                }
                IrExpr::Unary { expr: inner, .. } => scan_expr(inner, dead_names, bad, false),
                IrExpr::Binary { left, right, .. } => {
                    scan_expr(left, dead_names, bad, false);
                    scan_expr(right, dead_names, bad, false);
                }
                IrExpr::Conditional {
                    condition,
                    then_branch,
                    else_branch,
                } => {
                    scan_expr(condition, dead_names, bad, false);
                    scan_expr(then_branch, dead_names, bad, false);
                    scan_expr(else_branch, dead_names, bad, false);
                }
                IrExpr::InitializerList(elems) => {
                    for x in elems {
                        scan_expr(x, dead_names, bad, false);
                    }
                }
                IrExpr::Variable(_) | IrExpr::Literal(_) | IrExpr::Assign { .. } => {}
            }
        }
        fn scan_stmt(s: &IrStmt, dead_names: &HashSet<String>, bad: &mut HashSet<String>) {
            match s {
                // Top-level expression statements drop safely; only
                // nested positions disqualify here.
                IrStmt::Expr(e) => scan_expr(e, dead_names, bad, true),
                IrStmt::Return(e) => {
                    if let Some(e) = e {
                        scan_expr(e, dead_names, bad, false);
                    }
                }
                IrStmt::VarDecl { init, .. } => {
                    if let Some(init) = init {
                        scan_expr(init, dead_names, bad, false);
                    }
                }
                IrStmt::Block(inner) => {
                    for s in inner {
                        scan_stmt(s, dead_names, bad);
                    }
                }
                IrStmt::If {
                    condition,
                    then_branch,
                    else_branch,
                } => {
                    scan_expr(condition, dead_names, bad, false);
                    for s in then_branch {
                        scan_stmt(s, dead_names, bad);
                    }
                    if let Some(b) = else_branch {
                        for s in b {
                            scan_stmt(s, dead_names, bad);
                        }
                    }
                }
                IrStmt::For {
                    init,
                    condition,
                    update,
                    body,
                } => {
                    scan_stmt(init, dead_names, bad);
                    if let Some(condition) = condition {
                        scan_expr(condition, dead_names, bad, false);
                    }
                    if let Some(update) = update {
                        scan_expr(update, dead_names, bad, false);
                    }
                    for s in body {
                        scan_stmt(s, dead_names, bad);
                    }
                }
                IrStmt::While { condition, body } => {
                    scan_expr(condition, dead_names, bad, false);
                    for s in body {
                        scan_stmt(s, dead_names, bad);
                    }
                }
                IrStmt::Switch { scrutinee, cases } => {
                    scan_expr(scrutinee, dead_names, bad, false);
                    for c in cases {
                        if let Some(label) = &c.label {
                            scan_expr(label, dead_names, bad, false);
                        }
                        for s in &c.body {
                            scan_stmt(s, dead_names, bad);
                        }
                    }
                }
            }
        }
        for s in stmts {
            scan_stmt(s, dead_names, bad);
        }
    }
    fn drop_stores(stmts: &mut Vec<IrStmt>, dead: &mut Vec<(String, String)>) {
        // Member names currently dead (any struct).
        let dead_names: HashSet<String> = dead.iter().map(|(_, m)| m.clone()).collect();
        stmts.retain(|s| {
            if let IrStmt::Expr(e) = s {
                if let IrExpr::Assign { lhs, rhs } = e {
                    if let Some(m) = target_outermost_member(lhs) {
                        if dead_names.contains(m) {
                            // Keep impure stores (loud) and disqualify.
                            return expr_has_call(rhs) || target_has_call(lhs);
                        }
                    }
                }
            }
            true
        });
        // Disqualify slots whose impure store survived.
        let mut impure: HashSet<String> = HashSet::new();
        fn scan(s: &IrStmt, dead_names: &HashSet<String>, impure: &mut HashSet<String>) {
            match s {
                IrStmt::Block(inner) => {
                    for s in inner {
                        scan(s, dead_names, impure);
                    }
                }
                IrStmt::If {
                    then_branch,
                    else_branch,
                    ..
                } => {
                    for s in then_branch {
                        scan(s, dead_names, impure);
                    }
                    if let Some(b) = else_branch {
                        for s in b {
                            scan(s, dead_names, impure);
                        }
                    }
                }
                IrStmt::For { init, body, .. } => {
                    scan(init, dead_names, impure);
                    for s in body {
                        scan(s, dead_names, impure);
                    }
                }
                IrStmt::While { body, .. } => {
                    for s in body {
                        scan(s, dead_names, impure);
                    }
                }
                IrStmt::Switch { cases, .. } => {
                    for c in cases {
                        for s in &c.body {
                            scan(s, dead_names, impure);
                        }
                    }
                }
                IrStmt::Expr(e) => {
                    if let IrExpr::Assign { lhs, rhs } = e {
                        if let Some(m) = target_outermost_member(lhs) {
                            if dead_names.contains(m)
                                && (expr_has_call(rhs) || target_has_call(lhs))
                            {
                                impure.insert(m.to_string());
                            }
                        }
                    }
                }
                _ => {}
            }
        }
        for s in stmts.iter() {
            scan(s, &dead_names, &mut impure);
        }
        dead.retain(|(_, m)| !impure.contains(m));
        for s in stmts.iter_mut() {
            match s {
                IrStmt::Block(inner) => drop_stores(inner, dead),
                IrStmt::If {
                    then_branch,
                    else_branch,
                    ..
                } => {
                    drop_stores(then_branch, dead);
                    if let Some(b) = else_branch {
                        drop_stores(b, dead);
                    }
                }
                IrStmt::For { body, .. } => {
                    // For-init is loop machinery: never drop there (the
                    // disqualification scan above still sees it).
                    drop_stores(body, dead);
                }
                IrStmt::While { body, .. } => drop_stores(body, dead),
                IrStmt::Switch { cases, .. } => {
                    for c in cases {
                        drop_stores(&mut c.body, dead);
                    }
                }
                _ => {}
            }
        }
    }
    for func in &mut ir.functions {
        drop_stores(&mut func.body, &mut dead);
    }
    // Disqualify slots with sub-expression stores (kept above by
    // position, still dangling if stripped).
    {
        let dead_names: HashSet<String> = dead.iter().map(|(_, m)| m.clone()).collect();
        let mut bad: HashSet<String> = HashSet::new();
        for func in &ir.functions {
            flag_subexpr_stores(&func.body, &dead_names, &mut bad);
        }
        dead.retain(|(_, m)| !bad.contains(m));
    }
    if dead.is_empty() {
        return;
    }

    // Strip the dead slots from struct definitions (buffer-typed field
    // only; same-named scalars elsewhere are untouched).
    for st in &mut ir.structs {
        st.fields.retain(|f| {
            !(dead.iter().any(|(s, m)| s == &st.name && m == &f.name)
                && matches!(&f.ty, IrType::Resource { .. }))
        });
    }
}

/// Structs with a call site feeding a whole split-candidate buffer
/// element (`f(buf[i])`) directly as a call argument. Such calls need
/// the buffer's element type to stay whole-`S`, so the struct keeps the
/// parameter veto. Every other whole-value position (`S x = buf[i]`,
/// `return buf[i]`, nested expressions) materializes through the twin +
/// per-member arrays, locals pass by value untouched, and scalar member
/// uses read the twin in place — none of them need the veto.
fn buffer_index_call_args(
    ir: &ShaderModule,
    buf_struct: &HashMap<String, String>,
) -> HashSet<String> {
    fn expr_args(e: &IrExpr, buf_struct: &HashMap<String, String>, out: &mut HashSet<String>) {
        if let IrExpr::Call { arguments, .. } = e {
            for a in arguments {
                if let Some((object, _)) = as_buffer_index(a) {
                    if let IrExpr::Variable(b) = object {
                        if let Some(s) = buf_struct.get(b) {
                            out.insert(s.clone());
                        }
                    }
                }
            }
        }
        match e {
            IrExpr::Call { arguments, .. } | IrExpr::Construct { arguments, .. } => {
                for a in arguments {
                    expr_args(a, buf_struct, out);
                }
            }
            IrExpr::Member { object, .. } => expr_args(object, buf_struct, out),
            IrExpr::Index { object, index } | IrExpr::ResourceIndex { resource: object, index } => {
                expr_args(object, buf_struct, out);
                expr_args(index, buf_struct, out);
            }
            IrExpr::ResourceLoad {
                resource,
                coordinates,
                sample,
            } => {
                expr_args(resource, buf_struct, out);
                expr_args(coordinates, buf_struct, out);
                if let Some(sample) = sample {
                    expr_args(sample, buf_struct, out);
                }
            }
            IrExpr::Unary { expr: inner, .. } => expr_args(inner, buf_struct, out),
            IrExpr::Binary { left, right, .. } => {
                expr_args(left, buf_struct, out);
                expr_args(right, buf_struct, out);
            }
            IrExpr::Conditional {
                condition,
                then_branch,
                else_branch,
            } => {
                expr_args(condition, buf_struct, out);
                expr_args(then_branch, buf_struct, out);
                expr_args(else_branch, buf_struct, out);
            }
            IrExpr::Assign { lhs, rhs } => {
                expr_args(lhs, buf_struct, out);
                expr_args(rhs, buf_struct, out);
            }
            IrExpr::InitializerList(elems) => {
                for x in elems {
                    expr_args(x, buf_struct, out);
                }
            }
            IrExpr::Variable(_) | IrExpr::Literal(_) => {}
        }
    }
    fn stmt_args(s: &IrStmt, buf_struct: &HashMap<String, String>, out: &mut HashSet<String>) {
        match s {
            IrStmt::Return(e) => {
                if let Some(e) = e {
                    expr_args(e, buf_struct, out);
                }
            }
            IrStmt::Block(inner) => {
                for s in inner {
                    stmt_args(s, buf_struct, out);
                }
            }
            IrStmt::VarDecl { init, .. } => {
                if let Some(init) = init {
                    expr_args(init, buf_struct, out);
                }
            }
            IrStmt::If {
                condition,
                then_branch,
                else_branch,
            } => {
                expr_args(condition, buf_struct, out);
                for s in then_branch {
                    stmt_args(s, buf_struct, out);
                }
                if let Some(else_branch) = else_branch {
                    for s in else_branch {
                        stmt_args(s, buf_struct, out);
                    }
                }
            }
            IrStmt::For {
                init,
                condition,
                update,
                body,
            } => {
                stmt_args(init, buf_struct, out);
                if let Some(condition) = condition {
                    expr_args(condition, buf_struct, out);
                }
                if let Some(update) = update {
                    expr_args(update, buf_struct, out);
                }
                for s in body {
                    stmt_args(s, buf_struct, out);
                }
            }
            IrStmt::While { condition, body } => {
                expr_args(condition, buf_struct, out);
                for s in body {
                    stmt_args(s, buf_struct, out);
                }
            }
            IrStmt::Switch { scrutinee, cases } => {
                expr_args(scrutinee, buf_struct, out);
                for c in cases {
                    if let Some(label) = &c.label {
                        expr_args(label, buf_struct, out);
                    }
                    for s in &c.body {
                        stmt_args(s, buf_struct, out);
                    }
                }
            }
            IrStmt::Expr(e) => expr_args(e, buf_struct, out),
        }
    }
    let mut out = HashSet::new();
    for f in &ir.functions {
        for s in &f.body {
            stmt_args(s, buf_struct, &mut out);
        }
    }
    out
}

/// True when any split buffer name is shadowed in the function by a
/// parameter or local: such references mean a different object, so the
/// function keeps today's loud path.
fn split_name_shadowed(
    body: &[IrStmt],
    parameters: &[crate::ir::function::Parameter],
    buf_struct: &HashMap<String, String>,
) -> bool {
    if parameters.iter().any(|p| buf_struct.contains_key(&p.name)) {
        return true;
    }
    fn has_local(stmts: &[IrStmt], bufs: &HashMap<String, String>) -> bool {
        stmts.iter().any(|s| match s {
            IrStmt::VarDecl { name, .. } => bufs.contains_key(name),
            IrStmt::Block(inner) => has_local(inner, bufs),
            IrStmt::If {
                then_branch,
                else_branch,
                ..
            } => {
                has_local(then_branch, bufs)
                    || else_branch.as_ref().is_some_and(|b| has_local(b, bufs))
            }
            IrStmt::For { init, body, .. } => {
                has_local(std::slice::from_ref(init.as_ref()), bufs) || has_local(body, bufs)
            }
            IrStmt::While { body, .. } => has_local(body, bufs),
            IrStmt::Switch { cases, .. } => cases.iter().any(|c| has_local(&c.body, bufs)),
            _ => false,
        })
    }
    has_local(body, buf_struct)
}

/// Builds `[bare S name; name.f = ...; ...]` materializing a whole
/// `buf[idx]` value, or `None` when any member cannot complete (arrays,
/// unknown shapes stay loud).
fn materialize_buffer_value(
    name: &str,
    struct_name: &str,
    buf: &str,
    index: &IrExpr,
    buf_struct: &HashMap<String, String>,
    bindings: &HashMap<(String, String), String>,
    structs: &HashMap<String, Vec<(String, IrType, Option<u32>)>>,
) -> Option<Vec<IrStmt>> {
    fn leaves(
        sname: &str,
        buf: &str,
        index: &IrExpr,
        rel: &mut Vec<SubSeg>,
        buf_struct: &HashMap<String, String>,
        bindings: &HashMap<(String, String), String>,
        structs: &HashMap<String, Vec<(String, IrType, Option<u32>)>>,
        out: &mut Vec<(Vec<SubSeg>, IrExpr)>,
    ) -> bool {
        let _ = buf_struct;
        let Some(fields) = structs.get(sname) else {
            return false;
        };
        for (fname, fty, array_size) in fields {
            match fty {
                IrType::Named(n) if structs.contains_key(n) => {
                    // Nested structs materialize recursively; arrays of
                    // structs stay loud.
                    if array_size.is_some() {
                        return false;
                    }
                    rel.push(SubSeg::Field(fname.clone()));
                    if !leaves(n, buf, index, rel, buf_struct, bindings, structs, out) {
                        return false;
                    }
                    rel.pop();
                }
                _ => {
                    // Texture/sampler members resolve through their
                    // per-buffer arrays (scalar arrays unroll below);
                    // everything else (scalars and unknown-named enums)
                    // reads through the twin buffer, which keeps the name
                    // and layout. Anything else (buffers, byte address,
                    // array textures) stays loud.
                    if let Some(binding) =
                        bindings.get(&(buf.to_string(), fname.clone()))
                    {
                        if array_size.is_some() {
                            return false;
                        }
                        rel.push(SubSeg::Field(fname.clone()));
                        out.push((
                            rel.clone(),
                            IrExpr::Index {
                                object: Box::new(IrExpr::Variable(binding.clone())),
                                index: Box::new(index.clone()),
                            },
                        ));
                        rel.pop();
                    } else {
                        match fty {
                            IrType::Bool
                            | IrType::Int
                            | IrType::UInt
                            | IrType::Float
                            | IrType::Half
                            | IrType::Double
                            | IrType::Vector { .. }
                            | IrType::Matrix { .. }
                            | IrType::Named(_) => {}
                            _ => return false,
                        }
                        // Scalar arrays (kept verbatim in the twin) unroll
                        // per element; plain scalars assign directly.
                        match array_size {
                            Some(n) if *n > 0 => {
                                rel.push(SubSeg::Field(fname.clone()));
                                for i in 0..*n {
                                    let mut erel = rel.clone();
                                    erel.push(SubSeg::Index(i));
                                    let mut target = IrExpr::Index {
                                        object: Box::new(IrExpr::Variable(buf.to_string())),
                                        index: Box::new(index.clone()),
                                    };
                                    for seg in erel.iter() {
                                        target = match seg {
                                            SubSeg::Field(member) => IrExpr::Member {
                                                object: Box::new(target),
                                                member: member.clone(),
                                            },
                                            SubSeg::Index(idx) => IrExpr::Index {
                                                object: Box::new(target),
                                                index: Box::new(IrExpr::Literal(
                                                    IrLiteral::Number(idx.to_string()),
                                                )),
                                            },
                                        };
                                    }
                                    out.push((erel, target));
                                }
                                rel.pop();
                            }
                            Some(_) => return false,
                            None => {
                                let mut target = IrExpr::Index {
                                    object: Box::new(IrExpr::Variable(buf.to_string())),
                                    index: Box::new(index.clone()),
                                };
                                for seg in
                                    rel.iter().chain(std::iter::once(&SubSeg::Field(fname.clone())))
                                {
                                    target = match seg {
                                        SubSeg::Field(member) => IrExpr::Member {
                                            object: Box::new(target),
                                            member: member.clone(),
                                        },
                                        SubSeg::Index(idx) => IrExpr::Index {
                                            object: Box::new(target),
                                            index: Box::new(IrExpr::Literal(
                                                IrLiteral::Number(idx.to_string()),
                                            )),
                                        },
                                    };
                                }
                                rel.push(SubSeg::Field(fname.clone()));
                                out.push((rel.clone(), target));
                                rel.pop();
                            }
                        }
                    }
                }
            }
        }
        true
    }
    let mut collected: Vec<(Vec<SubSeg>, IrExpr)> = Vec::new();
    let mut rel: Vec<SubSeg> = Vec::new();
    if !leaves(
        struct_name,
        buf,
        index,
        &mut rel,
        buf_struct,
        bindings,
        structs,
        &mut collected,
    ) {
        return None;
    }
    let mut expanded = vec![IrStmt::VarDecl {
        ty: IrType::Named(struct_name.to_string()),
        name: name.to_string(),
        array_size: None,
        init: None,
    }];
    for (rel, value) in collected {
        let mut target = IrExpr::Variable(name.to_string());
        for seg in rel {
            target = match seg {
                SubSeg::Field(member) => IrExpr::Member {
                    object: Box::new(target),
                    member,
                },
                SubSeg::Index(idx) => IrExpr::Index {
                    object: Box::new(target),
                    index: Box::new(IrExpr::Literal(IrLiteral::Number(idx.to_string()))),
                },
            };
        }
        expanded.push(IrStmt::Expr(IrExpr::Assign {
            lhs: Box::new(target),
            rhs: Box::new(value),
        }));
    }
    Some(expanded)
}

/// Hoists UAV mipmap views (`tex.MipMaps[m][c]` reads/writes through a
/// texture's mip levels): a single HLSL image view cannot address mips,
/// so each rooted texture gains an unbounded array of same-typed views
/// (`RWTexture2DArray<float4> __mips_tex[]`) and `MipMaps` members
/// rewrite to it (`tex.MipMaps[m][c]` → `__mips_tex[m][c]`; int coords
/// convert implicitly). Textures keep their kind/access/element, so
/// reads and writes both work; unresolvable roots stay loud. Shaders
/// without `MipMaps` uses are untouched.
pub fn hoist_mipmap_views(ir: &mut ShaderModule) {
    // Root expression -> resource type for bare names (locals per
    // function are collected separately below).
    let mut root_tys: HashMap<String, IrType> = HashMap::new();
    for g in &ir.globals {
        root_tys.insert(g.name.clone(), g.ty.clone());
    }
    for r in &ir.resources {
        root_tys.insert(
            r.name.clone(),
            IrType::Resource {
                kind: r.kind.clone(),
                access: match r.resource_type {
                    ResourceType::RWBuffer | ResourceType::RWTexture => IrAccess::ReadWrite,
                    _ => IrAccess::ReadOnly,
                },
                element: Box::new(
                    r.element_type.clone().unwrap_or(IrType::Named("unknown".to_string())),
                ),
            },
        );
    }
    // Struct fields for dotted member paths (`srt.tex`).
    let structs: HashMap<String, HashMap<String, IrType>> = ir
        .structs
        .iter()
        .map(|s| {
            (
                s.name.clone(),
                s.fields.iter().map(|f| (f.name.clone(), f.ty.clone())).collect(),
            )
        })
        .collect();
    // Flatten a member path to (root, segments), looking through index
    // levels (`a.b[0].c` -> root `a`).
    fn path_of(e: &IrExpr) -> Option<(String, Vec<String>)> {
        match e {
            IrExpr::Variable(n) => Some((n.clone(), Vec::new())),
            IrExpr::Member { object, member } => {
                let (root, mut segs) = path_of(object)?;
                segs.push(member.clone());
                Some((root, segs))
            }
            IrExpr::Index { object, .. } | IrExpr::ResourceIndex { resource: object, .. } => {
                path_of(object)
            }
            _ => None,
        }
    }
    fn is_texture_view_kind(kind: &str) -> bool {
        matches!(
            kind,
            "texture1d"
                | "texture2d"
                | "texture2darray"
                | "texture3d"
                | "texturecube"
                | "texturecubearray"
                | "rwtexture1d"
                | "rwtexture2d"
                | "rwtexture2darray"
                | "rwtexture3d"
                | "mstexture2d"
                | "rwmstexture2d"
        )
    }
    // Resolve a rooted path to a texture-like (kind, access, element).
    fn resolve_tex_ty(
        root: &str,
        segs: &[String],
        root_tys: &HashMap<String, IrType>,
        locals: &HashMap<String, IrType>,
        structs: &HashMap<String, HashMap<String, IrType>>,
    ) -> Option<(String, IrAccess, IrType)> {
        let mut ty = locals
            .get(root)
            .or_else(|| root_tys.get(root))?
            .clone();
        for seg in segs {
            match ty {
                IrType::Named(s) => {
                    ty = structs.get(&s)?.get(seg)?.clone();
                }
                _ => return None,
            }
        }
        match ty {
            IrType::Resource { kind, access, element } if is_texture_view_kind(&kind) => {
                Some((kind, access, *element))
            }
            _ => None,
        }
    }
    struct Use {
        path: String,
        kind: String,
        access: IrAccess,
        element: IrType,
    }
    fn collect_expr(
        e: &IrExpr,
        locals: &HashMap<String, IrType>,
        root_tys: &HashMap<String, IrType>,
        structs: &HashMap<String, HashMap<String, IrType>>,
        out: &mut Vec<Use>,
    ) {
        if let IrExpr::Member { object, member } = e {
            if member.eq_ignore_ascii_case("mipmaps") {
                if let Some((root, segs)) = path_of(object) {
                    let path = if segs.is_empty() {
                        root.clone()
                    } else {
                        format!("{root}.{}", segs.join("."))
                    };
                    if let Some((kind, access, element)) =
                        resolve_tex_ty(&root, &segs, root_tys, locals, structs)
                    {
                        out.push(Use { path, kind, access, element });
                    }
                }
            }
        }
        match e {
            IrExpr::Call { arguments, .. } => {
                for a in arguments {
                    collect_expr(a, locals, root_tys, structs, out);
                }
            }
            IrExpr::Construct { arguments, .. } => {
                for a in arguments {
                    collect_expr(a, locals, root_tys, structs, out);
                }
            }
            IrExpr::Member { object, .. } => collect_expr(object, locals, root_tys, structs, out),
            IrExpr::Index { object, index } | IrExpr::ResourceIndex { resource: object, index } => {
                collect_expr(object, locals, root_tys, structs, out);
                collect_expr(index, locals, root_tys, structs, out);
            }
            IrExpr::ResourceLoad { resource, coordinates, sample } => {
                collect_expr(resource, locals, root_tys, structs, out);
                collect_expr(coordinates, locals, root_tys, structs, out);
                if let Some(s) = sample {
                    collect_expr(s, locals, root_tys, structs, out);
                }
            }
            IrExpr::Assign { lhs, rhs } => {
                collect_expr(lhs, locals, root_tys, structs, out);
                collect_expr(rhs, locals, root_tys, structs, out);
            }
            IrExpr::Unary { expr, .. } => collect_expr(expr, locals, root_tys, structs, out),
            IrExpr::Binary { left, right, .. } => {
                collect_expr(left, locals, root_tys, structs, out);
                collect_expr(right, locals, root_tys, structs, out);
            }
            IrExpr::Conditional { condition, then_branch, else_branch } => {
                collect_expr(condition, locals, root_tys, structs, out);
                collect_expr(then_branch, locals, root_tys, structs, out);
                collect_expr(else_branch, locals, root_tys, structs, out);
            }
            IrExpr::InitializerList(elems) => {
                for x in elems {
                    collect_expr(x, locals, root_tys, structs, out);
                }
            }
            _ => {}
        }
    }
    fn collect_locals(stmts: &[IrStmt], locals: &mut HashMap<String, IrType>) {
        for s in stmts {
            match s {
                IrStmt::VarDecl { name, ty, .. } => {
                    locals.insert(name.clone(), ty.clone());
                }
                IrStmt::Block(ss) => collect_locals(ss, locals),
                IrStmt::If { then_branch, else_branch, .. } => {
                    collect_locals(then_branch, locals);
                    for x in else_branch.iter().flatten() {
                        collect_locals(std::slice::from_ref(x), locals);
                    }
                }
                IrStmt::For { init, body, .. } => {
                    collect_locals(std::slice::from_ref(init.as_ref()), locals);
                    collect_locals(body, locals);
                }
                IrStmt::While { body, .. } => collect_locals(body, locals),
                IrStmt::Switch { cases, .. } => {
                    for c in cases {
                        collect_locals(&c.body, locals);
                    }
                }
                _ => {}
            }
        }
    }
    fn collect_in_stmt(
        s: &IrStmt,
        locals: &HashMap<String, IrType>,
        root_tys: &HashMap<String, IrType>,
        structs: &HashMap<String, HashMap<String, IrType>>,
        out: &mut Vec<Use>,
    ) {
        match s {
            IrStmt::Return(Some(e)) | IrStmt::Expr(e) => {
                collect_expr(e, locals, root_tys, structs, out)
            }
            IrStmt::Return(None) => {}
            IrStmt::VarDecl { init: Some(e), .. } => {
                collect_expr(e, locals, root_tys, structs, out)
            }
            IrStmt::VarDecl { init: None, .. } => {}
            IrStmt::Block(ss) => {
                for x in ss {
                    collect_in_stmt(x, locals, root_tys, structs, out);
                }
            }
            IrStmt::If { condition, then_branch, else_branch } => {
                collect_expr(condition, locals, root_tys, structs, out);
                for x in then_branch {
                    collect_in_stmt(x, locals, root_tys, structs, out);
                }
                for x in else_branch.iter().flatten() {
                    collect_in_stmt(x, locals, root_tys, structs, out);
                }
            }
            IrStmt::For { init, condition, update, body } => {
                collect_in_stmt(init, locals, root_tys, structs, out);
                if let Some(c) = condition {
                    collect_expr(c, locals, root_tys, structs, out);
                }
                if let Some(u) = update {
                    collect_expr(u, locals, root_tys, structs, out);
                }
                for x in body {
                    collect_in_stmt(x, locals, root_tys, structs, out);
                }
            }
            IrStmt::While { condition, body } => {
                collect_expr(condition, locals, root_tys, structs, out);
                for x in body {
                    collect_in_stmt(x, locals, root_tys, structs, out);
                }
            }
            IrStmt::Switch { scrutinee, cases } => {
                collect_expr(scrutinee, locals, root_tys, structs, out);
                for c in cases {
                    if let Some(label) = &c.label {
                        collect_expr(label, locals, root_tys, structs, out);
                    }
                    for x in &c.body {
                        collect_in_stmt(x, locals, root_tys, structs, out);
                    }
                }
            }
            _ => {}
        }
    }
    fn rewrite_expr(e: &mut IrExpr, bindings: &HashMap<String, String>) {
        if let IrExpr::Member { object, member } = e {
            if member.eq_ignore_ascii_case("mipmaps") {
                if let Some((root, segs)) = path_of(object) {
                    let path = if segs.is_empty() {
                        root
                    } else {
                        format!("{root}.{}", segs.join("."))
                    };
                    if let Some(binding) = bindings.get(&path) {
                        *e = IrExpr::Variable(binding.clone());
                        return;
                    }
                }
            }
        }
        match e {
            IrExpr::Call { arguments, .. } => {
                for a in arguments {
                    rewrite_expr(a, bindings);
                }
            }
            IrExpr::Construct { arguments, .. } => {
                for a in arguments {
                    rewrite_expr(a, bindings);
                }
            }
            IrExpr::Member { object, .. } => rewrite_expr(object, bindings),
            IrExpr::Index { object, index } | IrExpr::ResourceIndex { resource: object, index } => {
                rewrite_expr(object, bindings);
                rewrite_expr(index, bindings);
            }
            IrExpr::ResourceLoad { resource, coordinates, sample } => {
                rewrite_expr(resource, bindings);
                rewrite_expr(coordinates, bindings);
                if let Some(s) = sample {
                    rewrite_expr(s, bindings);
                }
            }
            IrExpr::Assign { lhs, rhs } => {
                rewrite_expr(lhs, bindings);
                rewrite_expr(rhs, bindings);
            }
            IrExpr::Unary { expr, .. } => rewrite_expr(expr, bindings),
            IrExpr::Binary { left, right, .. } => {
                rewrite_expr(left, bindings);
                rewrite_expr(right, bindings);
            }
            IrExpr::Conditional { condition, then_branch, else_branch } => {
                rewrite_expr(condition, bindings);
                rewrite_expr(then_branch, bindings);
                rewrite_expr(else_branch, bindings);
            }
            IrExpr::InitializerList(elems) => {
                for x in elems {
                    rewrite_expr(x, bindings);
                }
            }
            _ => {}
        }
    }
    fn rewrite_stmt(s: &mut IrStmt, bindings: &HashMap<String, String>) {
        match s {
            IrStmt::Return(Some(e)) | IrStmt::Expr(e) => rewrite_expr(e, bindings),
            IrStmt::Return(None) => {}
            IrStmt::VarDecl { init: Some(e), .. } => rewrite_expr(e, bindings),
            IrStmt::VarDecl { init: None, .. } => {}
            IrStmt::Block(ss) => {
                for x in ss {
                    rewrite_stmt(x, bindings);
                }
            }
            IrStmt::If { condition, then_branch, else_branch } => {
                rewrite_expr(condition, bindings);
                for x in then_branch {
                    rewrite_stmt(x, bindings);
                }
                for x in else_branch.iter_mut().flatten() {
                    rewrite_stmt(x, bindings);
                }
            }
            IrStmt::For { init, condition, update, body } => {
                rewrite_stmt(init, bindings);
                if let Some(c) = condition {
                    rewrite_expr(c, bindings);
                }
                if let Some(u) = update {
                    rewrite_expr(u, bindings);
                }
                for x in body {
                    rewrite_stmt(x, bindings);
                }
            }
            IrStmt::While { condition, body } => {
                rewrite_expr(condition, bindings);
                for x in body {
                    rewrite_stmt(x, bindings);
                }
            }
            IrStmt::Switch { scrutinee, cases } => {
                rewrite_expr(scrutinee, bindings);
                for c in cases {
                    if let Some(label) = &mut c.label {
                        rewrite_expr(label, bindings);
                    }
                    for x in &mut c.body {
                        rewrite_stmt(x, bindings);
                    }
                }
            }
            _ => {}
        }
    }
    // Gather uses with per-function locals (params seed).
    let mut uses: Vec<Use> = Vec::new();
    for f in &ir.functions {
        let mut locals: HashMap<String, IrType> = HashMap::new();
        for p in &f.parameters {
            locals.insert(p.name.clone(), p.ty.clone());
        }
        collect_locals(&f.body, &mut locals);
        for s in &f.body {
            collect_in_stmt(s, &locals, &root_tys, &structs, &mut uses);
        }
    }
    if uses.is_empty() {
        return;
    }
    // One binding per root path (consistent kind/access/element required;
    // mismatches keep no binding and stay loud).
    let mut bindings: HashMap<String, String> = HashMap::new();
    let mut seen: HashMap<String, (String, IrAccess, IrType)> = HashMap::new();
    for u in &uses {
        match seen.get(&u.path) {
            Some((k, a, e)) => {
                if k != &u.kind || a != &u.access || e != &u.element {
                    continue;
                }
            }
            None => {
                seen.insert(u.path.clone(), (u.kind.clone(), u.access, u.element.clone()));
            }
        }
        if bindings.contains_key(&u.path) {
            continue;
        }
        let mut binding = format!("__mips_{}", u.path.replace('.', "_"));
        let mut n = 0u32;
        while ir.globals.iter().any(|g| g.name == binding)
            || ir.resources.iter().any(|r| r.name == binding)
        {
            n += 1;
            binding = format!("__mips_{}_{n}", u.path.replace('.', "_"));
        }
        bindings.insert(u.path.clone(), binding.clone());
        ir.globals.push(GlobalVar {
            ty: IrType::Resource {
                kind: u.kind.clone(),
                access: u.access,
                element: Box::new(u.element.clone()),
            },
            name: binding,
            array_size: Some(0),
            const_value: None,
            shared: false,
        });
    }
    if bindings.is_empty() {
        return;
    }
    for f in &mut ir.functions {
        for s in &mut f.body {
            rewrite_stmt(s, &bindings);
        }
    }
}

/// A step in a materialized-local access path: struct field or array
/// element.
#[derive(Clone)]
enum SubSeg {
    Field(String),
    Index(u32),
}

/// Rewrites split-buffer uses in every statement list:
/// - `buf[i].m` (texture/sampler member) → `buf_m_arr[i]`;
/// - whole values (`S x = buf[i]`, `f(buf[i])`, `return buf[i]`) →
///   materialized full-`S` locals (spliced temporaries for value
///   positions, in-place expansion for declarations).
/// Scalar member uses pass through untouched (same buffer name).
fn rewrite_split_uses(
    stmts: &mut Vec<IrStmt>,
    buf_struct: &HashMap<String, String>,
    bindings: &HashMap<(String, String), String>,
    structs: &HashMap<String, Vec<(String, IrType, Option<u32>)>>,
    counter: &mut u32,
) {
    let mut i = 0;
    while i < stmts.len() {
        // Whole-struct copies materialize in place (tighter than a
        // temporary plus copy); spliced at list level so later uses
        // see the local (never wrapped in a block).
        if let Some(expanded) = expand_split_decl(&stmts[i], buf_struct, bindings, structs) {
            let n = expanded.len();
            stmts.splice(i..=i, expanded);
            i += n;
            continue;
        }
        let mut preamble: Vec<IrStmt> = Vec::new();
        rewrite_split_in_stmt(&mut stmts[i], buf_struct, bindings, structs, counter, &mut preamble);
        if preamble.is_empty() {
            i += 1;
        } else {
            let n = preamble.len();
            stmts.splice(i..i, preamble);
            i += n + 1;
        }
    }
}

/// Expands `S x = buf[i]` (or `auto`) into a bare local plus memberwise
/// assigns, or `None` for any other statement.
fn expand_split_decl(
    stmt: &IrStmt,
    buf_struct: &HashMap<String, String>,
    bindings: &HashMap<(String, String), String>,
    structs: &HashMap<String, Vec<(String, IrType, Option<u32>)>>,
) -> Option<Vec<IrStmt>> {
    let IrStmt::VarDecl { ty, name, init, .. } = stmt else {
        return None;
    };
    let init_expr = init.as_ref()?;
    let (object, index) = as_buffer_index(init_expr)?;
    let buf = buffer_of(object, buf_struct)?;
    let s = buf_struct.get(&buf).cloned()?;
    if !matches!(ty, IrType::Named(n) if n == "auto" || n == &s) {
        return None;
    }
    materialize_buffer_value(name, &s, &buf, index, buf_struct, bindings, structs)
}

fn rewrite_split_in_stmt(
    stmt: &mut IrStmt,
    buf_struct: &HashMap<String, String>,
    bindings: &HashMap<(String, String), String>,
    structs: &HashMap<String, Vec<(String, IrType, Option<u32>)>>,
    counter: &mut u32,
    preamble: &mut Vec<IrStmt>,
) {
    match stmt {
        IrStmt::Block(inner) => rewrite_split_uses(inner, buf_struct, bindings, structs, counter),
        IrStmt::If {
            condition,
            then_branch,
            else_branch,
        } => {
            rewrite_split_in_expr(condition, buf_struct, bindings, structs, counter, preamble, false);
            rewrite_split_uses(then_branch, buf_struct, bindings, structs, counter);
            if let Some(else_branch) = else_branch {
                rewrite_split_uses(else_branch, buf_struct, bindings, structs, counter);
            }
        }
        IrStmt::For {
            init,
            condition,
            update,
            body,
        } => {
            // Lone init: probe on a clone so an unplaceable expansion
            // leaves the original loud instead of dangling a temporary.
            let mut probe = init.as_ref().clone();
            let mut probe_pre: Vec<IrStmt> = Vec::new();
            rewrite_split_in_stmt(&mut probe, buf_struct, bindings, structs, counter, &mut probe_pre);
            if probe_pre.is_empty() {
                **init = probe;
            }
            if let Some(condition) = condition {
                rewrite_split_in_expr(condition, buf_struct, bindings, structs, counter, preamble, false);
            }
            if let Some(update) = update {
                rewrite_split_in_expr(update, buf_struct, bindings, structs, counter, preamble, false);
            }
            rewrite_split_uses(body, buf_struct, bindings, structs, counter);
        }
        IrStmt::While { condition, body } => {
            rewrite_split_in_expr(condition, buf_struct, bindings, structs, counter, preamble, false);
            rewrite_split_uses(body, buf_struct, bindings, structs, counter);
        }
        IrStmt::Switch { scrutinee, cases } => {
            rewrite_split_in_expr(scrutinee, buf_struct, bindings, structs, counter, preamble, false);
            for case in cases {
                if let Some(label) = &mut case.label {
                    rewrite_split_in_expr(label, buf_struct, bindings, structs, counter, preamble, false);
                }
                rewrite_split_uses(&mut case.body, buf_struct, bindings, structs, counter);
            }
        }
        IrStmt::Return(e) => {
            if let Some(e) = e {
                rewrite_split_in_expr(e, buf_struct, bindings, structs, counter, preamble, false);
            }
        }
        IrStmt::VarDecl { init, .. } => {
            // Whole-struct copies expand at list level (see above);
            // still descend (e.g. nested calls in the initializer).
            if let Some(init_expr) = init {
                rewrite_split_in_expr(init_expr, buf_struct, bindings, structs, counter, preamble, false);
            }
        }
        IrStmt::Expr(e) => {
            rewrite_split_in_expr(e, buf_struct, bindings, structs, counter, preamble, false)
        }
    }
}

/// Rewrites a dotted method callee through a split buffer:
/// `buf[idx].member.Method` → `buf_member_arr[idx].Method`.
/// Anything else (locals, other bases, malformed shapes) stays loud.
fn rewrite_split_call_function(
    function: &mut String,
    buf_struct: &HashMap<String, String>,
    bindings: &HashMap<(String, String), String>,
) {
    let dot = match function.rfind('.') {
        Some(d) => d,
        None => return,
    };
    let (obj, method) = (function[..dot].to_string(), function[dot + 1..].to_string());
    if method.is_empty() {
        return;
    }
    // obj = `buf[idx].member`: split off the trailing member first.
    let mdot = match obj.rfind('.') {
        Some(d) => d,
        None => return,
    };
    let (base_idx, member) = (
        obj[..mdot].to_string(),
        obj[mdot + 1..].to_string(),
    );
    if !base_idx.ends_with(']') {
        return;
    }
    let open = match base_idx.rfind('[') {
        Some(o) => o,
        None => return,
    };
    let (buf, idx) = (
        base_idx[..open].to_string(),
        base_idx[open + 1..base_idx.len() - 1].to_string(),
    );
    if buf.is_empty()
        || member.is_empty()
        || buf.contains(|c| c == '.' || c == '[' || c == ']')
        || member.contains(|c| c == '.' || c == '[' || c == ']')
        || !buf_struct.contains_key(&buf)
    {
        return;
    }
    if let Some(binding) = bindings.get(&(buf, member)) {
        *function = format!("{binding}[{idx}].{method}");
    }
}

/// Buffer name behind an expression if it is a plain split-buffer
/// variable, else `None`.
fn buffer_of(expr: &IrExpr, buf_struct: &HashMap<String, String>) -> Option<String> {
    if let IrExpr::Variable(b) = expr {
        if buf_struct.contains_key(b) {
            return Some(b.clone());
        }
    }
    None
}

/// Views an index-like node (`Index` or `ResourceIndex`) as base + index.
/// SRT-flattened buffers lower to `ResourceIndex` (resource-typed struct
/// member) while plain globals lower to `Index`; both spell the same
/// `buf[i]` access.
fn as_buffer_index(expr: &IrExpr) -> Option<(&IrExpr, &IrExpr)> {
    match expr {
        IrExpr::Index { object, index } => Some((object, index)),
        IrExpr::ResourceIndex { resource, index } => Some((resource, index)),
        _ => None,
    }
}

fn rewrite_split_in_expr(
    expr: &mut IrExpr,
    buf_struct: &HashMap<String, String>,
    bindings: &HashMap<(String, String), String>,
    structs: &HashMap<String, Vec<(String, IrType, Option<u32>)>>,
    counter: &mut u32,
    preamble: &mut Vec<IrStmt>,
    writable: bool,
) {
    // Texture/sampler member access through a split buffer.
    if let IrExpr::Member { object, member } = expr {
        if let Some((base, _)) = as_buffer_index(object.as_ref()) {
            if let Some(buf) = buffer_of(base, buf_struct) {
                if let Some(binding) = bindings.get(&(buf.clone(), member.clone())) {
                    // `buf[i].m` → `buf_m_arr[i]` (index transfers).
                    let index = match object.as_ref() {
                        IrExpr::Index { index, .. } | IrExpr::ResourceIndex { index, .. } => {
                            index.as_ref().clone()
                        }
                        _ => unreachable!("matched index-like above"),
                    };
                    *expr = IrExpr::Index {
                        object: Box::new(IrExpr::Variable(binding.clone())),
                        index: Box::new(index),
                    };
                    return;
                }
            }
        }
    }
    // Whole-struct buffer values in readable positions materialize a
    // same-scope temporary (overload-safe: identical static type).
    // Writable positions keep today's loud path.
    if !writable {
        if let Some((object, index)) = as_buffer_index(expr) {
            if let Some(buf) = buffer_of(object, buf_struct) {
                if let Some(s) = buf_struct.get(&buf).cloned() {
                    if let Some(pieces) =
                        materialize_buffer_value(&format!("__splitarg_{counter}"), &s, &buf, index, buf_struct, bindings, structs)
                    {
                        *counter += 1;
                        let temp = match &pieces[0] {
                            IrStmt::VarDecl { name, .. } => name.clone(),
                            _ => unreachable!("materialize starts with a declaration"),
                        };
                        preamble.extend(pieces);
                        *expr = IrExpr::Variable(temp);
                        return;
                    }
                }
            }
        }
    }
    match expr {
        IrExpr::Call { function, arguments } => {
            rewrite_split_call_function(function, buf_struct, bindings);
            for a in arguments {
                rewrite_split_in_expr(a, buf_struct, bindings, structs, counter, preamble, false);
            }
        }
        IrExpr::Construct { arguments, .. } => {
            for a in arguments {
                rewrite_split_in_expr(a, buf_struct, bindings, structs, counter, preamble, false);
            }
        }
        IrExpr::Member { object, .. } => {
            rewrite_split_in_expr(object, buf_struct, bindings, structs, counter, preamble, false);
        }
        IrExpr::Index { object, index } | IrExpr::ResourceIndex { resource: object, index } => {
            rewrite_split_in_expr(object, buf_struct, bindings, structs, counter, preamble, writable);
            rewrite_split_in_expr(index, buf_struct, bindings, structs, counter, preamble, false);
        }
        IrExpr::ResourceLoad {
            resource,
            coordinates,
            sample,
        } => {
            rewrite_split_in_expr(resource, buf_struct, bindings, structs, counter, preamble, writable);
            rewrite_split_in_expr(coordinates, buf_struct, bindings, structs, counter, preamble, false);
            if let Some(sample) = sample {
                rewrite_split_in_expr(sample, buf_struct, bindings, structs, counter, preamble, false);
            }
        }
        IrExpr::Unary { expr: inner, .. } => {
            rewrite_split_in_expr(inner, buf_struct, bindings, structs, counter, preamble, writable)
        }
        IrExpr::Binary { left, right, .. } => {
            rewrite_split_in_expr(left, buf_struct, bindings, structs, counter, preamble, writable);
            rewrite_split_in_expr(right, buf_struct, bindings, structs, counter, preamble, writable);
        }
        IrExpr::Conditional {
            condition,
            then_branch,
            else_branch,
        } => {
            rewrite_split_in_expr(condition, buf_struct, bindings, structs, counter, preamble, false);
            rewrite_split_in_expr(then_branch, buf_struct, bindings, structs, counter, preamble, writable);
            rewrite_split_in_expr(else_branch, buf_struct, bindings, structs, counter, preamble, writable);
        }
        IrExpr::Assign { lhs, rhs } => {
            rewrite_split_in_expr(lhs, buf_struct, bindings, structs, counter, preamble, true);
            rewrite_split_in_expr(rhs, buf_struct, bindings, structs, counter, preamble, false);
        }
        IrExpr::InitializerList(elems) => {
            for e in elems {
                rewrite_split_in_expr(e, buf_struct, bindings, structs, counter, preamble, writable);
            }
        }
        _ => {}
    }
}
