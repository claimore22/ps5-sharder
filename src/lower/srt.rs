//! SRT (Shader Resource Table) flattening.
//!
//! PSSL often bundles a whole resource table into a single global struct whose
//! members mix GPU resources (textures, buffers) with plain scalar uniforms,
//! e.g.:
//!
//! ```text
//! struct BloomParamsTx {
//!     Texture2D<float3> src;
//!     RW_Texture2D<float3> dst;
//! };
//! struct BloomParams {
//!     BloomParamsTx* tx;
//!     DataBuffer<float> exposure_stats;
//!     float2 uv_scale;
//!     float intensity;
//!     float threshold;
//! };
//! BloomParams srt : S_SRT_DATA;
//! ```
//!
//! DXC does not accept a *global* struct that mixes resources and non-resources
//! ("global structures containing both resources and non-resources are not
//! supported"). To emit valid Vulkan HLSL we flatten such a root global into:
//!
//! * one top-level HLSL resource binding per resource member reachable from the
//!   root (textures, rw-textures, buffers), and
//! * a single `cbuffer` holding the scalar/uniform members.
//!
//! Every access to the original struct members is rewritten to reference the
//! flattened bindings. This runs as a post-lowering IR transform, so it works on
//! the HLSL-agnostic IR and only fires for globals that actually mix resources
//! with non-resources (leaving ordinary shaders untouched).

use std::collections::HashMap;

use crate::ir::*;

/// Struct name -> ordered `(field name, field type)` map, derived from the IR
/// module's struct declarations.
type StructFields = HashMap<String, Vec<(String, IrType)>>;

/// Flattens any top-level SRT globals that mix resources with non-resources.
pub fn flatten_srt(ir: &mut ShaderModule) {
    let fields = build_struct_map(ir);

    // Root struct-typed globals to inspect.
    let roots: Vec<(String, String)> = ir
        .globals
        .iter()
        .filter_map(|g| match &g.ty {
            IrType::Named(s) => Some((g.name.clone(), s.clone())),
            _ => None,
        })
        .collect();

    // Cbuffer member names claimed by earlier roots. HLSL injects cbuffer
    // members into global scope, so two flattened roots sharing a member
    // name (an SRT param global `srt` plus its `s_srt = srt;` mirror, both
    // carrying `proj2View`) would collide (DXC `redefinition of
    // 'proj2View'`). Later roots take the parent-suffixed form via the
    // per-root collision logic below, seeded with this history.
    let mut cross_cbuffer_members: std::collections::HashSet<String> =
        std::collections::HashSet::new();

    for (root, root_struct) in roots {
        // Only flatten when the struct is actually declared in the IR (a real
        // composite) and not a scalar-like named type.
        if !fields.contains_key(&root_struct) {
            continue;
        }

        let mut res: Vec<(String, IrType)> = Vec::new();
        let mut scal: Vec<(String, IrType)> = Vec::new();
        let mut visited: Vec<String> = Vec::new();
        walk_struct(
            &fields,
            &root_struct,
            &mut vec![root.clone()],
            &mut res,
            &mut scal,
            &mut visited,
        );

        // Mixed (resources + scalars) is the only case DXC rejects.
        if res.is_empty() || scal.is_empty() {
            continue;
        }

        let mut resource_map: HashMap<String, String> = HashMap::new();
        let mut new_resources: Vec<Resource> = Vec::new();
        // Leaf paths through a sized intermediate struct array, mapped to
        // the array field's segment index (root = 0): use rewrites consult
        // this to transfer the index inward only when it sits exactly at
        // the array segment (any other placement selects a different
        // object and stays loud).
        let mut array_at: HashMap<String, usize> = HashMap::new();

        for (path, ty) in &res {
            let resource_name = path.replace('.', "_");
            // Resource member arrays (`RW_RegularBuffer<uint>
            // indirect_buffers[2]`): the flattened binding keeps the
            // dimension (`...[2]`), so indexed uses stay valid. Unknown
            // sizes keep today's scalar shape. Paths through a sized
            // intermediate struct array (`srt_data.buffers[2].tile`) take
            // the intermediate dimension instead (uses transfer inward).
            let (member_array, member_at) =
                subobject_array_size(&fields, &ir.structs, &root_struct, path, &root);
            if let Some(at) = member_at {
                array_at.insert(path.clone(), at);
            }
            match ty {
                IrType::Resource { kind, access, element } => {
                    resource_map.insert(path.clone(), resource_name.clone());
                    let mut r = resource_resource(kind, *access, element, &resource_name);
                    r.array_size = member_array;
                    new_resources.push(r);
                }
                // Sampler members (`SamplerState`, framework samplers):
                // Vulkan samplers are externally-bound top-level bindings,
                // never cbuffer fields.
                IrType::Named(n) if crate::framework::is_sampler_type_name(n) => {
                    resource_map.insert(path.clone(), resource_name.clone());
                    let mut r = sampler_resource(&resource_name);
                    r.array_size = member_array;
                    new_resources.push(r);
                }
                // Resource-like framework objects (`sce::Agc::Core::Texture`,
                // `sce::Psr::TopLevelBvhDescriptor`): externally-bound
                // top-level bindings, never cbuffer fields.
                IrType::Named(n)
                    if crate::framework::framework_resource_kind(n).is_some() =>
                {
                    resource_map.insert(path.clone(), resource_name.clone());
                    let mut r = framework_object_resource(n, &resource_name);
                    r.array_size = member_array;
                    new_resources.push(r);
                }
                // Bare texture spellings (`Texture2D tex`, no element):
                // sampled textures default float4 (HLSL's own default,
                // matching use sites like `tex.Sample(smp, uv).rgb`).
                IrType::Named(n)
                    if matches!(
                        n.to_ascii_lowercase().as_str(),
                        "texture1d" | "texture2d" | "texturecube" | "texture3d"
                    ) =>
                {
                    resource_map.insert(path.clone(), resource_name.clone());
                    let mut r = resource_resource(
                        &n.to_ascii_lowercase(),
                        IrAccess::ReadOnly,
                        &IrType::Vector {
                            base: IrScalar::Float,
                            width: 4,
                        },
                        &resource_name,
                    );
                    r.array_size = member_array;
                    new_resources.push(r);
                }
                _ => {}
            }
        }

        let scalar_field_name = format!("{root}_constants");
        // Assign a unique cbuffer field name per scalar path. Two different
        // SRT structs can both carry a member with the same bare name
        // (e.g. `EnvInfo.mipScale` and `DeferredOptions.mipScale`); the bare
        // name would collide in the one cbuffer (DXC: redefinition). On
        // collision, suffix the field with its parent segment.
        //
        // Cbuffer members also share global scope with free functions: a
        // field whose bare name matches a function name (e.g.
        // `DeferredOptions.tonemap` vs `tonemap()`) is a genuine HLSL
        // conflict (DXC: redefinition as different kind of symbol), so it
        // takes the parent-suffixed name too. A numeric suffix breaks any
        // remaining tie. Shaders without such collisions are untouched.
        let func_names: std::collections::HashSet<String> =
            ir.functions.iter().map(|f| f.name.clone()).collect();
        // Seed with earlier roots' members (cross-cbuffer history above):
        // first root keeps bare names, later roots disambiguate.
        let mut used_names: std::collections::HashSet<String> = cross_cbuffer_members.clone();
        let parent_of = |path: &str| -> String {
            path.rsplitn(2, '.')
                .nth(1)
                .map(|p| p.rsplit('.').next().unwrap_or(""))
                .unwrap_or("")
                .to_string()
        };
        let mut scalar_field_names: HashMap<String, String> = HashMap::new();
        for (path, _) in &scal {
            let bare = path.rsplit('.').next().unwrap_or(path).to_string();
            let mut candidate = if used_names.contains(&bare) {
                let parent = parent_of(path);
                format!("{parent}_{bare}")
            } else {
                bare.clone()
            };
            if !used_names.contains(&candidate) && func_names.contains(&candidate) {
                candidate = format!("{}_{bare}", parent_of(path));
            }
            if candidate.is_empty() {
                candidate = bare.clone();
            }
            let stem = candidate.clone();
            let mut n = 0u32;
            while used_names.contains(&candidate) || func_names.contains(&candidate) {
                n += 1;
                candidate = format!("{stem}_{n}");
            }
            used_names.insert(candidate.clone());
            scalar_field_names.insert(path.clone(), candidate);
        }
        // Publish this root's member names for later roots (see
        // `cross_cbuffer_members` above).
        cross_cbuffer_members.extend(used_names.iter().cloned());
        // Sized array dimensions by scalar path: array members keep their
        // size in the cbuffer so indexed uses stay valid (leaf arrays as
        // before; intermediate struct-array paths take the intermediate
        // dimension, with uses transferring the index inward).
        let mut scal_arrays: HashMap<String, u32> = HashMap::new();
        for (path, _) in &scal {
            let (dim, at) = subobject_array_size(&fields, &ir.structs, &root_struct, path, &root);
            if let Some(n) = dim {
                scal_arrays.insert(path.clone(), n);
            }
            if let Some(pos) = at {
                array_at.insert(path.clone(), pos);
            }
        }
        let cb_resource =
            scalar_cbuffer(&scalar_field_name, &scal, &scalar_field_names, &scal_arrays);
        let mut scalar_map: HashMap<String, String> = HashMap::new();
        for (path, _) in &scal {
            // HLSL cbuffer members are accessed by bare name, not
            // `cbuffer_name.member`, so rewrite scalar paths to the member name.
            if let Some(name) = scalar_field_names.get(path) {
                scalar_map.insert(path.clone(), name.clone());
            } else {
                let last = path.rsplit('.').next().unwrap_or(path).to_string();
                scalar_map.insert(path.clone(), last);
            }
        }

        // Drop the original mixed global.
        ir.globals.retain(|g| g.name != root);
        // Drop whole-variable stores to the flattened root
        // (`cbStaticCommon = srt.cbStaticCommon;` rebinding a static
        // view): the global is gone and reads already resolve to the
        // flattened bindings, so the store can never execute
        // meaningfully. Pure right-hand sides drop entirely; RHS with
        // calls keeps the effects as a bare statement. Stores in
        // non-statement positions keep today's loud path (no such
        // shapes occur in SRT-view rebinding).
        fn expr_has_call(e: &IrExpr) -> bool {
            match e {
                IrExpr::Call { .. } => true,
                IrExpr::Variable(_) | IrExpr::Literal(_) => false,
                IrExpr::Member { object, .. } => expr_has_call(object),
                IrExpr::Index { object, index }
                | IrExpr::ResourceIndex { resource: object, index } => {
                    expr_has_call(object) || expr_has_call(index)
                }
                IrExpr::ResourceLoad { resource, coordinates, sample } => {
                    expr_has_call(resource)
                        || expr_has_call(coordinates)
                        || sample.as_ref().is_some_and(|s| expr_has_call(s))
                }
                IrExpr::Construct { arguments, .. } => arguments.iter().any(expr_has_call),
                IrExpr::Unary { expr: inner, .. } => expr_has_call(inner),
                IrExpr::Binary { left, right, .. } => {
                    expr_has_call(left) || expr_has_call(right)
                }
                IrExpr::Conditional { condition, then_branch, else_branch } => {
                    expr_has_call(condition)
                        || expr_has_call(then_branch)
                        || expr_has_call(else_branch)
                }
                IrExpr::Assign { lhs, rhs } => expr_has_call(lhs) || expr_has_call(rhs),
                IrExpr::InitializerList(elems) => elems.iter().any(expr_has_call),
            }
        }
        fn drop_root_stores(stmts: &mut Vec<IrStmt>, root: &str) {
            let mut i = 0;
            while i < stmts.len() {
                let drop_this = match &stmts[i] {
                    IrStmt::Expr(IrExpr::Assign { lhs, rhs }) => {
                        if matches!(lhs.as_ref(), IrExpr::Variable(v) if v == root) {
                            Some(expr_has_call(rhs))
                        } else {
                            None
                        }
                    }
                    _ => None,
                };
                if let Some(has_call) = drop_this {
                    if has_call {
                        if let IrStmt::Expr(IrExpr::Assign { rhs, .. }) =
                            std::mem::replace(&mut stmts[i], IrStmt::Return(None))
                        {
                            stmts[i] = IrStmt::Expr((*rhs).clone());
                        }
                    } else {
                        stmts.remove(i);
                        continue;
                    }
                }
                match &mut stmts[i] {
                    IrStmt::Block(inner) => drop_root_stores(inner, root),
                    IrStmt::If { then_branch, else_branch, .. } => {
                        drop_root_stores(then_branch, root);
                        if let Some(els) = else_branch {
                            drop_root_stores(els, root);
                        }
                    }
                    IrStmt::For { body, .. } | IrStmt::While { body, .. } => {
                        drop_root_stores(body, root);
                    }
                    IrStmt::Switch { cases, .. } => {
                        for c in cases {
                            drop_root_stores(&mut c.body, root);
                        }
                    }
                    _ => {}
                }
                i += 1;
            }
        }
        for func in &mut ir.functions {
            drop_root_stores(&mut func.body, &root);
        }

        // Add flattened resources.
        ir.resources.extend(new_resources);
        ir.resources.push(cb_resource);

        // Header-owned `srt` flattening is shader-owned; helpers that take
        // `DofBlurParams param` must be rewritten to use flattened bindings.
        // Detect helpers with a `root_struct` param and alias their `param.*`
        // paths to the same flattened resources.
        // Per-function-index (not per-name): overloads share a name but have
        // distinct arities, so name-keyed elision double-strips the first
        // overload and leaves the second dangling (one overload lost its
        // `VertexInput` while keeping `pSrt`). `pre_arity` disambiguates call
        // sites (5-arg calls target the 5-param overload, 2-arg the 2-param).
        let mut helpers: Vec<(usize, String, usize, String, usize)> = Vec::new(); // (func_idx, func_name, param_idx, param_name, pre_arity)
        for (fi, func) in ir.functions.iter().enumerate() {
            for (idx, p) in func.parameters.iter().enumerate() {
                if let IrType::Named(n) = &p.ty {
                    if n == &root_struct {
                        tracing::trace!(
                            target: "pssl_shader::lower::srt",
                            func = %func.name,
                            param = %p.name,
                            "helper takes SRT struct by value – will be flattened"
                        );
                        helpers.push((
                            fi,
                            func.name.clone(),
                            idx,
                            p.name.clone(),
                            func.parameters.len(),
                        ));
                        break;
                    }
                }
            }
        }
        // Alias param.* → srt_* for rewriting helper bodies.
        let mut param_resource_map = HashMap::new();
        let mut param_scalar_map = HashMap::new();
        for (_, _, _, param_name, _) in &helpers {
            for (path, flat) in &resource_map {
                // `srt.textures.coc` → `param.textures.coc`
                let suffix = path.strip_prefix(&format!("{root}.")).unwrap_or(path);
                let alias = format!("{param_name}.{suffix}");
                param_resource_map.insert(alias, flat.clone());
            }
            for (path, flat) in &scalar_map {
                let suffix = path.strip_prefix(&format!("{root}.")).unwrap_or(path);
                let alias = format!("{param_name}.{suffix}");
                param_scalar_map.insert(alias, flat.clone());
            }
        }

        // Remove the `DofBlurParams param` parameter from helpers (call sites
        // that pass `srt` will be fixed below). Per-index: each overload
        // elides exactly its own param.
        for (fi, helper_name, idx, _, _) in &helpers {
            if let Some(func) = ir.functions.get_mut(*fi) {
                if *idx >= func.parameters.len() {
                    tracing::warn!(
                        target: "pssl_shader::lower::srt",
                        func = %helper_name,
                        idx = %idx,
                        params = func.parameters.len(),
                        "stale SRT param index; skipping elision",
                    );
                    continue;
                }
                tracing::info!(
                    target: "pssl_shader::lower::srt",
                    func = %helper_name,
                    idx = %idx,
                    "eliding SRT struct param from helper"
                );
                func.parameters.remove(*idx);
            }
        }

        // Merge param aliases into combined maps for single-pass rewrite.
        let mut combined_res = resource_map.clone();
        combined_res.extend(param_resource_map.clone());
        let mut combined_scal = scalar_map.clone();
        combined_scal.extend(param_scalar_map.clone());

        // Subobject-param helpers (`float3 ambient_contribution(SceneData
        // scene_data, Surface s)` called as `ambient_contribution(
        // srt_data.scene_data, s)`): elide the parameter and rebind
        // `param.*` to the flattened leaves (same mechanism as root
        // params, with the subpath prefix). Strictly gated per (helper,
        // param, subpath); anything unverified keeps today's loud path.
        // See `elide_subobject_params` for the soundness argument.
        // Snapshot pre-elision parameter lists first: elision removes
        // the very params later passes need for param-rooted path
        // resolution.
        let pre_elision_params: Vec<Vec<crate::ir::function::Parameter>> =
            ir.functions.iter().map(|f| f.parameters.clone()).collect();
        let (sub_elided, sub_bindings) = elide_subobject_params(
            ir,
            root.as_str(),
            root_struct.as_str(),
            &fields,
            &helpers,
            &resource_map,
            &scalar_map,
            &array_at,
            &mut combined_res,
            &mut combined_scal,
        );
        // Elided subobject params stay OUT of `helpers`: the
        // arity-heuristic call rewriter below would strip arguments from
        // calls whose arity happens to match another overload's
        // pre-elision arity (stacked overloads double-strip). D1 call
        // sites are stripped strictly inside `elide_subobject_params`;
        // only the param names join the roots below.
        let sub_param_roots: Vec<String> = {
            let mut names: Vec<String> = Vec::new();
            for (_, _, _, param_name, _) in &sub_elided {
                if !names.iter().any(|n| n == param_name) {
                    names.push(param_name.clone());
                }
            }
            names
        };
        // Locals aliasing whole SRT subobjects (`EnvInfo& x = *srt.y`,
        // `auto x = srt.y`) cannot rewrite by path maps (only leaves map):
        // materialize them from the flattened pieces first (bare local +
        // memberwise assigns), so every later use resolves.
        {
            let param_roots: Vec<String> = helpers
                .iter()
                .map(|(_, _, _, p, _)| p.clone())
                .chain(sub_param_roots.iter().cloned())
                .collect();
            let ctx = SubobjectCtx {
                root: root.as_str(),
                root_struct: root_struct.as_str(),
                fields: &fields,
                param_roots: &param_roots,
                resource_map: &combined_res,
                scalar_map: &combined_scal,
                res_tys: &res,
                structs: &ir.structs,
            };
            for function in &mut ir.functions {
                if srt_root_shadowed(&function.body, &function.parameters, root.as_str(), &param_roots) {
                    continue;
                }
                materialize_in_stmts(&mut function.body, &ctx);
            }
        }

        // Whole-subobject paths in value positions (`f(*srt.y)`) become
        // same-scope materialized temporaries (overload-safe: the
        // temporary carries the path's struct type).
        {
            let param_roots: Vec<String> = helpers
                .iter()
                .map(|(_, _, _, p, _)| p.clone())
                .chain(sub_param_roots.iter().cloned())
                .collect();
            let ctx = SubobjectCtx {
                root: root.as_str(),
                root_struct: root_struct.as_str(),
                fields: &fields,
                param_roots: &param_roots,
                resource_map: &combined_res,
                scalar_map: &combined_scal,
                res_tys: &res,
                structs: &ir.structs,
            };
            let mut counter = max_srtarg_suffix(ir).map_or(0, |m| m.saturating_add(1));
            for function in &mut ir.functions {
                if srt_root_shadowed(&function.body, &function.parameters, root.as_str(), &param_roots) {
                    continue;
                }
                splice_subobject_temps(&mut function.body, &ctx, &mut counter);
            }
        }

        // Struct-typed members used through an index (`srt.buf[i].x`):
        // the member is a host-sized array (PSSL pointers desugar away),
        // so hoist it as an unbounded StructuredBuffer and rewrite indexed
        // uses to it. Plain/member uses of the same path keep working
        // through the leaf maps and temporaries. Members whose struct
        // transitively contains resources stay loud (no legal binding).
        {
            let param_roots: Vec<String> = helpers
                .iter()
                .map(|(_, _, _, p, _)| p.clone())
                .chain(sub_param_roots.iter().cloned())
                .collect();
            let mut indexed: std::collections::BTreeSet<(String, String)> =
                std::collections::BTreeSet::new();
            for function in &ir.functions {
                if srt_root_shadowed(&function.body, &function.parameters, root.as_str(), &param_roots) {
                    continue;
                }
                collect_indexed_subobjects(
                    &function.body,
                    root.as_str(),
                    &root_struct,
                    &fields,
                    &param_roots,
                    &ir.structs,
                    &mut indexed,
                );
            }
            let mut index_map: HashMap<String, String> = HashMap::new();
            for (path, struct_name) in &indexed {
                if struct_contains_resource(struct_name, &fields, 0) {
                    continue;
                }
                let mut binding = path.replace('.', "_");
                let mut n = 0u32;
                while ir.globals.iter().any(|g| g.name == binding)
                    || ir.resources.iter().any(|r| r.name == binding)
                {
                    n += 1;
                    binding = format!("{}_{n}", path.replace('.', "_"));
                }
                index_map.insert(path.clone(), binding.clone());
                ir.resources.push(Resource {
                    name: binding,
                    resource_type: ResourceType::StructuredBuffer,
                    kind: "structuredbuffer".to_string(),
                    element_type: Some(IrType::Named(struct_name.clone())),
                    descriptor_set: Some(0),
                    binding: None,
                    register: None,
                    space: None,
                    array_size: None,
                    fields: Vec::new(),
                });
            }
            for function in &mut ir.functions {
                if srt_root_shadowed(&function.body, &function.parameters, root.as_str(), &param_roots) {
                    continue;
                }
                rewrite_indexed_uses(&mut function.body, &index_map);
            }
        }

        // Rewrite every function body.
        // Indexed struct-array members first (`srt_data.buffers[k].tile`,
        // param roots alike): resolve whole-element aliases, then transfer
        // the index inward ahead of the plain path rewrite below.
        {
            let param_roots: Vec<String> = helpers
                .iter()
                .map(|(_, _, _, p, _)| p.clone())
                .chain(sub_param_roots.iter().cloned())
                .collect();
            // array_at covers root paths; root-bound helper params alias
            // `param.rest...` for the same leaves, so prefix-swap those
            // entries (positions carry over: a root-bound param renames
            // the root segment 1:1). Subpath-bound (D1) params swap with
            // a coordinate shift (the array segment sits shallower in
            // parameter coordinates); conflicting keys across params drop
            // the entry (loud-safe: neither transfer fires).
            let root_param_roots: Vec<String> = helpers
                .iter()
                .map(|(_, _, _, p, _)| p.clone())
                .collect();
            let mut combined_at = array_at.clone();
            let mut poisoned: std::collections::HashSet<String> = std::collections::HashSet::new();
            // (param name, shift): root-bound params shift 0, D1 params
            // shift by subpath depth.
            let mut shifts: Vec<(String, usize)> =
                root_param_roots.iter().map(|p| (p.clone(), 0)).collect();
            for (pname, subpath) in &sub_bindings {
                let shift = subpath.split('.').count().saturating_sub(1);
                if !shifts.iter().any(|(n, s)| n == pname && s == &shift) {
                    shifts.push((pname.clone(), shift));
                }
            }
            for (path, at) in &array_at {
                if let Some(rest) = path.strip_prefix(&format!("{root}.")) {
                    for (p, shift) in &shifts {
                        if poisoned.contains(&format!("{p}.{rest}")) {
                            continue;
                        }
                        let Some(key_at) = at.checked_sub(*shift) else {
                            continue;
                        };
                        if key_at == 0 {
                            continue;
                        }
                        let key = format!("{p}.{rest}");
                        match combined_at.get(&key) {
                            Some(existing) if *existing != key_at => {
                                combined_at.remove(&key);
                                poisoned.insert(key);
                            }
                            Some(_) => {}
                            None => {
                                combined_at.insert(key, key_at);
                            }
                        }
                    }
                }
            }
            for (fi, function) in ir.functions.iter_mut().enumerate() {
                if srt_root_shadowed(&function.body, &function.parameters, root.as_str(), &param_roots) {
                    continue;
                }
                // Pre-elision parameter lists: elision removes the very
                // params param-rooted path resolution needs for types.
                let params = pre_elision_params.get(fi).cloned().unwrap_or_default();
                // Indexed whole-element copies of all-scalar structs
                // first (they consume the same declaration shape the
                // alias pass would otherwise claim).
                materialize_indexed_copies(
                    &mut function.body,
                    root.as_str(),
                    &param_roots,
                    &params,
                    &fields,
                    &ir.structs,
                    root_struct.as_str(),
                    &combined_scal,
                );
                resolve_srt_aliases(
                    &mut function.body,
                    root.as_str(),
                    &param_roots,
                    &params,
                    &fields,
                    &ir.structs,
                    root_struct.as_str(),
                );
                transfer_indexed_uses(
                    &mut function.body,
                    &combined_res,
                    &combined_scal,
                    &combined_at,
                );
            }
        }
        for function in &mut ir.functions {
            for stmt in &mut function.body {
                rewrite_stmt(stmt, root.as_str(), &combined_res, &combined_scal);
            }
        }

        // Rewrite call sites that pass `srt` as the helper's SRT argument.
        if !helpers.is_empty() {
            tracing::trace!(
                target: "pssl_shader::lower::srt",
                helpers = ?helpers,
                "rewriting call sites for SRT helpers"
            );
        }
        for func in &mut ir.functions {
            for stmt in &mut func.body {
                rewrite_calls_for_helpers(stmt, &helpers);
            }
        }

        // Keep all struct declarations: `DofBlurParams` is still used as a
        // helper-function parameter type before elision, but after elision the
        // struct remains for debugging. Retaining is harmless.
        let _ = visited;
    }
}

/// Resolves a root-anchored path to its struct when every segment is a
/// non-array member of declared struct type (`None` otherwise, including
/// through arrays: array elements need an index no helper signature can
/// supply).
fn resolve_path_struct(
    fields: &StructFields,
    structs: &[StructDecl],
    root_struct: &str,
    segs: &[&str],
) -> Option<String> {
    let mut cur = root_struct.to_string();
    for seg in segs {
        let fty = fields
            .get(&cur)?
            .iter()
            .find(|(n, _)| n == seg)
            .map(|(_, t)| t.clone())?;
        if field_array_size(structs, &cur, seg).is_some() {
            return None;
        }
        match fty {
            IrType::Named(s) if fields.contains_key(&s) => cur = s,
            _ => return None,
        }
    }
    Some(cur)
}

/// True when a subobject parameter is never stored through, redeclared,
/// or read bare outside call arguments: member-rooted reads rebind
/// (verified by the coverage dry run), and bare uses under call
/// arguments compose with callee elision (or dangle loudly without it:
/// the post-elision check). Stores through the parameter would corrupt
/// globals (local copy semantics), so any assignment mentioning it
/// vetoes.
fn subparam_body_ok(body: &[IrStmt], param: &str) -> bool {
    fn expr_ok(e: &IrExpr, param: &str, in_call_args: bool) -> bool {
        match e {
            IrExpr::Variable(v) => v != param || in_call_args,
            // Member-rooted reads rebind (coverage decides); bare
            // parameters under them are not bare uses.
            IrExpr::Member { object, .. } => expr_ok(object, param, true),
            IrExpr::Index { object, index } | IrExpr::ResourceIndex { resource: object, index } => {
                expr_ok(object, param, in_call_args) && expr_ok(index, param, in_call_args)
            }
            IrExpr::ResourceLoad { resource, coordinates, sample } => {
                expr_ok(resource, param, in_call_args)
                    && expr_ok(coordinates, param, in_call_args)
                    && sample.as_ref().is_none_or(|s| expr_ok(s, param, in_call_args))
            }
            IrExpr::Call { arguments, .. } => {
                arguments.iter().all(|a| expr_ok(a, param, true))
            }
            IrExpr::Construct { arguments, .. } => {
                arguments.iter().all(|a| expr_ok(a, param, in_call_args))
            }
            IrExpr::Unary { expr: inner, .. } => expr_ok(inner, param, in_call_args),
            IrExpr::Binary { left, right, .. } => {
                expr_ok(left, param, in_call_args) && expr_ok(right, param, in_call_args)
            }
            IrExpr::Conditional { condition, then_branch, else_branch } => {
                expr_ok(condition, param, in_call_args)
                    && expr_ok(then_branch, param, in_call_args)
                    && expr_ok(else_branch, param, in_call_args)
            }
            IrExpr::Assign { lhs, rhs } => {
                !mentions_var(lhs, param) && expr_ok(rhs, param, in_call_args)
            }
            IrExpr::InitializerList(elems) => {
                elems.iter().all(|x| expr_ok(x, param, in_call_args))
            }
            IrExpr::Literal(_) => true,
        }
    }
    fn stmt_ok(s: &IrStmt, param: &str) -> bool {
        match s {
            IrStmt::Return(e) => e.as_ref().is_none_or(|e| expr_ok(e, param, false)),
            IrStmt::VarDecl { name, init, .. } => {
                if name == param {
                    return false;
                }
                init.as_ref().is_none_or(|e| expr_ok(e, param, false))
            }
            IrStmt::If { condition, then_branch, else_branch } => {
                expr_ok(condition, param, false)
                    && then_branch.iter().all(|x| stmt_ok(x, param))
                    && else_branch.as_ref().is_none_or(|eb| eb.iter().all(|x| stmt_ok(x, param)))
            }
            IrStmt::For { init, condition, update, body } => {
                stmt_ok(init, param)
                    && condition.as_ref().is_none_or(|c| expr_ok(c, param, false))
                    && update.as_ref().is_none_or(|u| expr_ok(u, param, false))
                    && body.iter().all(|x| stmt_ok(x, param))
            }
            IrStmt::While { condition, body } => {
                expr_ok(condition, param, false) && body.iter().all(|x| stmt_ok(x, param))
            }
            IrStmt::Switch { scrutinee, cases } => {
                expr_ok(scrutinee, param, false)
                    && cases.iter().all(|c| {
                        c.label.as_ref().is_none_or(|l| expr_ok(l, param, false))
                            && c.body.iter().all(|x| stmt_ok(x, param))
                    })
            }
            IrStmt::Block(ss) => ss.iter().all(|x| stmt_ok(x, param)),
            IrStmt::Expr(e) => expr_ok(e, param, false),
        }
    }
    body.iter().all(|s| stmt_ok(s, param))
}

/// Elides helper parameters taking flattened SRT subobjects
/// (`float3 ambient_contribution(SceneData scene_data, Surface s)`
/// called as `ambient_contribution(srt_data.scene_data, s)`): the
/// parameter drops and `param.*` uses rebind to the flattened leaves
/// (returned alias maps, merged by the caller).
///
/// Soundness: elision is exact exactly when every removed argument
/// evaluates the subpath (directly, or through an already-elided
/// parameter of identical binding — induction grounded in direct
/// calls), and the body never stores through the parameter or reads it
/// outside call arguments and member paths (see `subparam_body_ok`).
/// `out`/`inout` parameters never elide (stores would corrupt globals);
/// anything unverified keeps today's loud path; call sites that survive
/// with stale arity fail loudly at DXC (never silently rebound).
/// Forwarding chains (`ambient` passing `scene_data` to an elided
/// `compute_env_factor`) converge through the fixpoint bound; cycles
/// without a direct base never verify. Entry helpers never elide
/// (driver-facing signatures stay).
///
/// Returns `(function index, helper name, param index, param name,
/// pre-elision arity)` per elided parameter for the caller's
/// helper bookkeeping, plus `(param name, subpath)` bindings for the
/// index-transfer coordinate swap.
#[allow(clippy::too_many_arguments)]
fn elide_subobject_params(
    ir: &mut ShaderModule,
    root: &str,
    root_struct: &str,
    fields: &StructFields,
    helpers: &[(usize, String, usize, String, usize)],
    resource_map: &HashMap<String, String>,
    scalar_map: &HashMap<String, String>,
    array_at: &HashMap<String, usize>,
    combined_res: &mut HashMap<String, String>,
    combined_scal: &mut HashMap<String, String>,
) -> (
    Vec<(usize, String, usize, String, usize)>,
    Vec<(String, String)>,
) {
    // Candidate subpaths per struct: proper array-free prefixes of
    // flattened leaf paths.
    let mut struct_subpaths: HashMap<String, Vec<String>> = HashMap::new();
    for key in resource_map.keys().chain(scalar_map.keys()) {
        let segs: Vec<&str> = key.split('.').collect();
        if segs.first().is_none_or(|r| *r != root) {
            continue;
        }
        for len in 2..segs.len() {
            let prefix = segs[..len].join(".");
            let Some(resolved) = resolve_path_struct(fields, &ir.structs, root_struct, &segs[1..len])
            else {
                continue;
            };
            struct_subpaths.entry(resolved).or_default().push(prefix);
        }
    }
    for paths in struct_subpaths.values_mut() {
        paths.sort();
        paths.dedup();
    }
    // (function index, param name) -> subpath, grown to fixpoint.
    // Seeded from the already-elided root params (whole-root binding).
    let mut bound: HashMap<(usize, String), String> = HashMap::new();
    for (fi, _, _, pname, _) in helpers {
        bound.insert((*fi, pname.clone()), root.to_string());
    }
    // Plain-name calls collected per function for verification.
    fn visit_calls(
        e: &IrExpr,
        out: &mut Vec<(String, usize, Vec<IrExpr>)>,
    ) {
        match e {
            IrExpr::Call { function, arguments } => {
                if !function.contains('.') {
                    out.push((function.clone(), arguments.len(), arguments.clone()));
                }
                for a in arguments {
                    visit_calls(a, out);
                }
            }
            IrExpr::Member { object, .. } => visit_calls(object, out),
            IrExpr::Index { object, index } | IrExpr::ResourceIndex { resource: object, index } => {
                visit_calls(object, out);
                visit_calls(index, out);
            }
            IrExpr::ResourceLoad { resource, coordinates, sample } => {
                visit_calls(resource, out);
                visit_calls(coordinates, out);
                if let Some(s) = sample {
                    visit_calls(s, out);
                }
            }
            IrExpr::Construct { arguments, .. } => {
                for a in arguments {
                    visit_calls(a, out);
                }
            }
            IrExpr::Unary { expr: inner, .. } => visit_calls(inner, out),
            IrExpr::Binary { left, right, .. } => {
                visit_calls(left, out);
                visit_calls(right, out);
            }
            IrExpr::Conditional { condition, then_branch, else_branch } => {
                visit_calls(condition, out);
                visit_calls(then_branch, out);
                visit_calls(else_branch, out);
            }
            IrExpr::Assign { lhs, rhs } => {
                visit_calls(lhs, out);
                visit_calls(rhs, out);
            }
            IrExpr::InitializerList(elems) => {
                for x in elems {
                    visit_calls(x, out);
                }
            }
            IrExpr::Variable(_) | IrExpr::Literal(_) => {}
        }
    }
    fn body_calls(body: &[IrStmt], out: &mut Vec<(String, usize, Vec<IrExpr>)>) {
        for s in body {
            match s {
                IrStmt::Return(e) => {
                    if let Some(e) = e {
                        visit_calls(e, out);
                    }
                }
                IrStmt::VarDecl { init, .. } => {
                    if let Some(e) = init {
                        visit_calls(e, out);
                    }
                }
                IrStmt::If { condition, then_branch, else_branch } => {
                    visit_calls(condition, out);
                    body_calls(then_branch, out);
                    if let Some(eb) = else_branch {
                        body_calls(eb, out);
                    }
                }
                IrStmt::For { init, condition, update, body } => {
                    body_calls(std::slice::from_ref(init), out);
                    if let Some(c) = condition {
                        visit_calls(c, out);
                    }
                    if let Some(u) = update {
                        visit_calls(u, out);
                    }
                    body_calls(body, out);
                }
                IrStmt::While { condition, body } => {
                    visit_calls(condition, out);
                    body_calls(body, out);
                }
                IrStmt::Switch { scrutinee, cases } => {
                    visit_calls(scrutinee, out);
                    for c in cases {
                        if let Some(label) = &c.label {
                            visit_calls(label, out);
                        }
                        body_calls(&c.body, out);
                    }
                }
                IrStmt::Block(ss) => body_calls(ss, out),
                IrStmt::Expr(e) => visit_calls(e, out),
            }
        }
    }
    let entry_name = ir.entry_point.clone();
    // Local names per function (shadowing guard for bound expansion).
    let all_locals: Vec<std::collections::HashSet<String>> = ir
        .functions
        .iter()
        .map(|f| {
            let mut set = std::collections::HashSet::new();
            collect_local_names(&f.body, &mut set);
            set
        })
        .collect();
    // Fixpoint detection (arities intact throughout: no removal yet).
    // (function index, param index, subpath), in verification order.
    let mut elide: Vec<(usize, usize, String)> = Vec::new();
    loop {
        let mut progress = false;
        for fi in 0..ir.functions.len() {
            if entry_name.as_deref() == Some(ir.functions[fi].name.as_str()) {
                continue;
            }
            // Same-name overloads share detection: each (fi) verifies
            // independently by arity below. Indices are positions in
            // the real parameter list (a filtered enumeration would
            // drift removal onto neighboring parameters).
            let params: Vec<(usize, String, String)> = ir.functions[fi]
                .parameters
                .iter()
                .enumerate()
                .filter_map(|(idx, p)| match &p.ty {
                    IrType::Named(s)
                        if p.array_size.is_none()
                            && !matches!(
                                p.direction.as_deref(),
                                Some("out") | Some("inout")
                            ) =>
                    {
                        Some((idx, p.name.clone(), s.clone()))
                    }
                    _ => None,
                })
                .collect();
            for (idx, pname, stype) in params.iter() {
                if pname == root {
                    continue;
                }
                if bound.contains_key(&(fi, pname.clone())) {
                    continue;
                }
                let Some(subpaths) = struct_subpaths.get(stype) else {
                    continue;
                };
                // Elide only under unanimity: every verifying call must
                // agree on one subpath (polymorphic use stays loud).
                let mut agreeing: Option<String> = None;
                let mut ok = true;
                // Body must rebind cleanly regardless of subpath.
                if !subparam_body_ok(&ir.functions[fi].body, pname) {
                    continue;
                }
                for subpath in subpaths {
                    let mut calls_ok = true;
                    for (ci, f) in ir.functions.iter().enumerate() {
                        let mut calls = Vec::new();
                        body_calls(&f.body, &mut calls);
                        for (cname, arity, args) in &calls {
                            if cname != &ir.functions[fi].name || *arity != args.len() {
                                continue;
                            }
                            // Attribute the call to this overload by
                            // arity: amphibolous arity (two overloads
                            // sharing name+arity) cannot attribute the
                            // call, so it vetoes (it might pass the
                            // elided slot; C++ overload resolution is
                            // out of scope).
                            let overloads = ir
                                .functions
                                .iter()
                                .filter(|g| {
                                    g.name == *cname && g.parameters.len() == *arity
                                })
                                .count();
                            if overloads != 1 {
                                calls_ok = false;
                                break;
                            }
                            if *arity != ir.functions[fi].parameters.len() {
                                continue;
                            }
                            let Some(arg) = args.get(*idx) else {
                                calls_ok = false;
                                break;
                            };
                            // Shadowed roots cannot resolve: any call in
                            // a shadowed function vetoes.
                            if srt_root_shadowed(&f.body, &f.parameters, root, &[]) {
                                calls_ok = false;
                                break;
                            }
                            let expanded = expanded_arg_subpath(
                                arg,
                                ci,
                                &f.parameters,
                                &all_locals[ci],
                                &bound,
                                root,
                            );
                            if expanded.as_deref() != Some(subpath.as_str()) {
                                calls_ok = false;
                                break;
                            }
                        }
                        if !calls_ok {
                            break;
                        }
                    }
                    // Constructive coverage: every member-rooted use must
                    // rebind through the downstream passes (dry run on a
                    // clone); leftovers veto loudly instead of dangling.
                    if ok
                        && calls_ok
                        && !subobject_uses_covered(
                            &ir.functions[fi].body,
                            pname,
                            subpath,
                            &ir.functions[fi].parameters,
                            fields,
                            &ir.structs,
                            root_struct,
                            root,
                            resource_map,
                            scalar_map,
                            array_at,
                        )
                    {
                        calls_ok = false;
                    }
                    if calls_ok && agreeing.replace(subpath.clone()).is_some() {
                        ok = false;
                        break;
                    }
                }
                if ok {
                    if let Some(subpath) = agreeing {
                        bound.insert((fi, pname.clone()), subpath);
                        elide.push((fi, *idx, pname.clone()));
                        progress = true;
                    }
                }
            }
        }
        if !progress {
            break;
        }
    }
    // Postcheck (fixpoint): bare parameter uses under call arguments
    // must target elided callees with the identical binding (else they
    // dangle after elision). Vetoes cascade: dropping one binding can
    // strand its forwarders.
    loop {
        let mut progress = false;
        // Snapshot of currently-bound entries for this round.
        let live: Vec<(usize, String, String)> = elide
            .iter()
            .filter_map(|(fi, _, pname)| {
                bound
                    .get(&(*fi, pname.clone()))
                    .map(|p| (*fi, pname.clone(), p.clone()))
            })
            .collect();
        for (fi, pname, subpath) in &live {
            let mut calls = Vec::new();
            body_calls(&ir.functions[*fi].body, &mut calls);
            let mut ok = true;
            for (cname, arity, args) in &calls {
                for (j, arg) in args.iter().enumerate() {
                    // Only direct bare uses need a bound callee; member
                    // paths were covered constructively, deeper nesting
                    // resolves at its own call level.
                    if !matches!(arg, IrExpr::Variable(v) if v == pname) {
                        continue;
                    }
                    // Resolve the callee overload by current arity
                    // (pre-removal: arities intact throughout).
                    let mut target: Option<(usize, String)> = None;
                    let mut count = 0;
                    for (gi, g) in ir.functions.iter().enumerate() {
                        if g.name == *cname && g.parameters.len() == *arity {
                            count += 1;
                            target = Some((gi, g.parameters[j].name.clone()));
                        }
                    }
                    match target {
                        Some((gi, qname)) if count == 1 => {
                            if bound.get(&(gi, qname)).is_none_or(|b| b != subpath) {
                                ok = false;
                                break;
                            }
                        }
                        _ => {
                            ok = false;
                            break;
                        }
                    }
                }
                if !ok {
                    break;
                }
            }
            if !ok {
                bound.remove(&(*fi, pname.clone()));
                elide.retain(|(efi, _, epname)| !(*efi == *fi && epname == pname));
                progress = true;
            }
        }
        if !progress {
            break;
        }
    }
    // Removal (descending indices per function) + strict call-site
    // stripping (re-verified against the final bound: anything else
    // keeps stale arity and fails loudly at DXC).
    let mut elided: Vec<(usize, String, usize, String, usize)> = Vec::new();
    {
        let mut by_fi: HashMap<usize, Vec<(usize, String, String)>> = HashMap::new();
        for (fi, idx, pname) in &elide {
            let subpath = bound
                .get(&(*fi, pname.clone()))
                .cloned()
                .unwrap_or_default();
            by_fi.entry(*fi).or_default().push((*idx, pname.clone(), subpath));
        }
        for (fi, mut idxs) in by_fi {
            idxs.sort_by(|a, b| b.0.cmp(&a.0));
            let Some(func) = ir.functions.get_mut(fi) else {
                continue;
            };
            let pre_arity = func.parameters.len();
            let fname = func.name.clone();
            for (idx, pname, subpath) in &idxs {
                if *idx >= func.parameters.len() {
                    continue;
                }
                // Alias `param.rest...` to the flattened leaves for the
                // subpath prefix (collisions across helpers take a
                // helper-suffixed name; values are identical-or-disjoint
                // per subpath, so distinct names stay sound).
                for (path, flat) in resource_map.iter().chain(scalar_map.iter()) {
                    let is_res = resource_map.contains_key(path);
                    if let Some(rest) = path.strip_prefix(&format!("{subpath}.")) {
                        let mut key = format!("{pname}.{rest}");
                        let target = if is_res { &mut *combined_res } else { &mut *combined_scal };
                        if let Some(existing) = target.get(&key) {
                            if existing != flat {
                                let stem = key.clone();
                                let mut n = 0u32;
                                key = format!("{stem}_{fname}");
                                while target.contains_key(&key) && target.get(&key) != Some(flat) {
                                    n += 1;
                                    key = format!("{stem}_{fname}_{n}");
                                }
                            }
                        }
                        target.insert(key, flat.clone());
                    }
                }
                func.parameters.remove(*idx);
                elided.push((fi, fname.clone(), *idx, pname.clone(), pre_arity));
            }
        }
    }
    // Strip verified call arguments (re-checked: only direct-subpath or
    // finally-bound bare uses go; anything else keeps stale arity and
    // fails loudly).
    {
        // (helper name, pre-arity, param index, subpath) for stripping.
        let mut strips: Vec<(String, usize, usize, String)> = Vec::new();
        for (fi, _, idx, pname, pre_arity) in &elided {
            if let Some(subpath) = bound.get(&(*fi, pname.clone())) {
                if let Some(func) = ir.functions.get(*fi) {
                    strips.push((func.name.clone(), *pre_arity, *idx, subpath.clone()));
                }
            }
        }
        // Re-resolve bound per enclosing call for the final check.
        let mut final_bound: HashMap<(usize, String), String> = HashMap::new();
        for (fi, _, _, pname, _) in &elided {
            if let Some(subpath) = bound.get(&(*fi, pname.clone())) {
                final_bound.insert((*fi, pname.clone()), subpath.clone());
            }
        }
        let fn_info: Vec<(
            Vec<crate::ir::function::Parameter>,
            std::collections::HashSet<String>,
        )> = ir
            .functions
            .iter()
            .map(|f| {
                let mut locals = std::collections::HashSet::new();
                collect_local_names(&f.body, &mut locals);
                (f.parameters.clone(), locals)
            })
            .collect();
        for (ci, (params, locals)) in fn_info.iter().enumerate() {
            for stmt in &mut ir.functions[ci].body {
                strip_calls_in_stmt(
                    stmt, ci, params, locals, &strips, &final_bound, root,
                );
            }
        }
    }
    // (param name, subpath) bindings for the index-transfer coordinate
    // swap (duplicate names with distinct subpaths all apply; the swap
    // drops conflicting keys loudly-safe).
    let mut bindings_out: Vec<(String, String)> = Vec::new();
    for (fi, _, _, pname, _) in &elided {
        if let Some(subpath) = bound.get(&(*fi, pname.clone())) {
            if !bindings_out.iter().any(|(n, s)| n == pname && s == subpath) {
                bindings_out.push((pname.clone(), subpath.clone()));
            }
        }
    }
    (elided, bindings_out)
}

/// Expands a call argument to its SRT subpath when provable: direct
/// root paths stand as-is; a bare variable (or a member path rooted at
/// one) takes the enclosing parameter's bound binding. Indexed
/// selections are values, never subobjects. A same-named local in the
/// enclosing body shadows the parameter and vetoes the expansion.
fn expanded_arg_subpath(
    arg: &IrExpr,
    enclosing: usize,
    enclosing_params: &[crate::ir::function::Parameter],
    enclosing_locals: &std::collections::HashSet<String>,
    bound: &HashMap<(usize, String), String>,
    root: &str,
) -> Option<String> {
    let (segs, idxs) = split_indexed(arg)?;
    if !idxs.is_empty() || segs.is_empty() {
        return None;
    }
    if segs[0] == root {
        return Some(segs.join("."));
    }
    let v = &segs[0];
    if !enclosing_params.iter().any(|p| &p.name == v) {
        return None;
    }
    if enclosing_locals.contains(v) {
        return None;
    }
    let base = bound.get(&(enclosing, v.clone()))?;
    Some(if segs.len() == 1 {
        base.clone()
    } else {
        format!("{base}.{}", segs[1..].join("."))
    })
}

/// Local variable names declared anywhere in a body (shadowing guard
/// for parameter-rooted path expansion).
fn collect_local_names(body: &[IrStmt], out: &mut std::collections::HashSet<String>) {
    for s in body {
        match s {
            IrStmt::VarDecl { name, .. } => {
                out.insert(name.clone());
            }
            IrStmt::Block(ss) => collect_local_names(ss, out),
            IrStmt::If { then_branch, else_branch, .. } => {
                collect_local_names(then_branch, out);
                if let Some(eb) = else_branch {
                    collect_local_names(eb, out);
                }
            }
            IrStmt::For { init, body, .. } => {
                collect_local_names(std::slice::from_ref(init), out);
                collect_local_names(body, out);
            }
            IrStmt::While { body, .. } => collect_local_names(body, out),
            IrStmt::Switch { cases, .. } => {
                for c in cases {
                    collect_local_names(&c.body, out);
                }
            }
            _ => {}
        }
    }
}

/// Strips verified subobject arguments at calls to elided helpers
/// (re-checked against the final bound: anything else keeps stale arity
/// and fails loudly at DXC).
fn strip_calls_in_stmt(
    stmt: &mut IrStmt,
    enclosing: usize,
    enclosing_params: &[crate::ir::function::Parameter],
    enclosing_locals: &std::collections::HashSet<String>,
    strips: &[(String, usize, usize, String)],
    bound: &HashMap<(usize, String), String>,
    root: &str,
) {
    fn expr(
        e: &mut IrExpr,
        enclosing: usize,
        enclosing_params: &[crate::ir::function::Parameter],
        enclosing_locals: &std::collections::HashSet<String>,
        strips: &[(String, usize, usize, String)],
        bound: &HashMap<(usize, String), String>,
        root: &str,
    ) {
        if let IrExpr::Call { function, arguments } = e {
            if let Some((_, _, idx, subpath)) = strips
                .iter()
                .find(|(n, pre, _, _)| n == function && arguments.len() == *pre)
            {
                let qualifies = match arguments.get(*idx) {
                    // Bare uses strip exactly when finally bound to the
                    // identical subpath. Consult the bound directly: the
                    // parameter is already removed, so membership tests
                    // against current params would miss. A same-named
                    // local in the enclosing body shadows (the use reads
                    // it, not the parameter): never strip those.
                    Some(IrExpr::Variable(v)) => {
                        !enclosing_locals.contains(v)
                            && bound.get(&(enclosing, v.clone())).as_deref() == Some(subpath)
                    }
                    Some(arg) => {
                        expanded_arg_subpath(
                            arg,
                            enclosing,
                            enclosing_params,
                            enclosing_locals,
                            bound,
                            root,
                        )
                        .as_deref()
                            == Some(subpath.as_str())
                    }
                    None => false,
                };
                if qualifies {
                    arguments.remove(*idx);
                }
            }
        }
        match e {
            IrExpr::Member { object, .. } => {
                expr(object, enclosing, enclosing_params, enclosing_locals, strips, bound, root)
            }
            IrExpr::Index { object, index } | IrExpr::ResourceIndex { resource: object, index } => {
                expr(object, enclosing, enclosing_params, enclosing_locals, strips, bound, root);
                expr(index, enclosing, enclosing_params, enclosing_locals, strips, bound, root);
            }
            IrExpr::ResourceLoad { resource, coordinates, sample } => {
                expr(resource, enclosing, enclosing_params, enclosing_locals, strips, bound, root);
                expr(coordinates, enclosing, enclosing_params, enclosing_locals, strips, bound, root);
                if let Some(s) = sample {
                    expr(s, enclosing, enclosing_params, enclosing_locals, strips, bound, root);
                }
            }
            IrExpr::Call { arguments, .. } => {
                for a in arguments {
                    expr(a, enclosing, enclosing_params, enclosing_locals, strips, bound, root);
                }
            }
            IrExpr::Construct { arguments, .. } => {
                for a in arguments {
                    expr(a, enclosing, enclosing_params, enclosing_locals, strips, bound, root);
                }
            }
            IrExpr::Unary { expr: inner, .. } => {
                expr(inner, enclosing, enclosing_params, enclosing_locals, strips, bound, root)
            }
            IrExpr::Binary { left, right, .. } => {
                expr(left, enclosing, enclosing_params, enclosing_locals, strips, bound, root);
                expr(right, enclosing, enclosing_params, enclosing_locals, strips, bound, root);
            }
            IrExpr::Conditional { condition, then_branch, else_branch } => {
                expr(condition, enclosing, enclosing_params, enclosing_locals, strips, bound, root);
                expr(then_branch, enclosing, enclosing_params, enclosing_locals, strips, bound, root);
                expr(else_branch, enclosing, enclosing_params, enclosing_locals, strips, bound, root);
            }
            IrExpr::Assign { lhs, rhs } => {
                expr(lhs, enclosing, enclosing_params, enclosing_locals, strips, bound, root);
                expr(rhs, enclosing, enclosing_params, enclosing_locals, strips, bound, root);
            }
            IrExpr::InitializerList(elems) => {
                for x in elems {
                    expr(x, enclosing, enclosing_params, enclosing_locals, strips, bound, root);
                }
            }
            IrExpr::Variable(_) | IrExpr::Literal(_) => {}
        }
    }
    match stmt {
        IrStmt::Return(e) => {
            if let Some(e) = e {
                expr(e, enclosing, enclosing_params, enclosing_locals, strips, bound, root);
            }
        }
        IrStmt::VarDecl { init, .. } => {
            if let Some(e) = init {
                expr(e, enclosing, enclosing_params, enclosing_locals, strips, bound, root);
            }
        }
        IrStmt::If { condition, then_branch, else_branch } => {
            expr(condition, enclosing, enclosing_params, enclosing_locals, strips, bound, root);
            for x in then_branch {
                strip_calls_in_stmt(x, enclosing, enclosing_params, enclosing_locals, strips, bound, root);
            }
            if let Some(eb) = else_branch {
                for x in eb {
                    strip_calls_in_stmt(x, enclosing, enclosing_params, enclosing_locals, strips, bound, root);
                }
            }
        }
        IrStmt::For { init, condition, update, body } => {
            strip_calls_in_stmt(init, enclosing, enclosing_params, enclosing_locals, strips, bound, root);
            if let Some(c) = condition {
                expr(c, enclosing, enclosing_params, enclosing_locals, strips, bound, root);
            }
            if let Some(u) = update {
                expr(u, enclosing, enclosing_params, enclosing_locals, strips, bound, root);
            }
            for x in body {
                strip_calls_in_stmt(x, enclosing, enclosing_params, enclosing_locals, strips, bound, root);
            }
        }
        IrStmt::While { condition, body } => {
            expr(condition, enclosing, enclosing_params, enclosing_locals, strips, bound, root);
            for x in body {
                strip_calls_in_stmt(x, enclosing, enclosing_params, enclosing_locals, strips, bound, root);
            }
        }
        IrStmt::Switch { scrutinee, cases } => {
            expr(scrutinee, enclosing, enclosing_params, enclosing_locals, strips, bound, root);
            for c in cases {
                if let Some(label) = &mut c.label {
                    expr(label, enclosing, enclosing_params, enclosing_locals, strips, bound, root);
                }
                for x in &mut c.body {
                    strip_calls_in_stmt(x, enclosing, enclosing_params, enclosing_locals, strips, bound, root);
                }
            }
        }
        IrStmt::Block(ss) => {
            for x in ss {
                strip_calls_in_stmt(x, enclosing, enclosing_params, enclosing_locals, strips, bound, root);
            }
        }
        IrStmt::Expr(e) => {
            expr(e, enclosing, enclosing_params, enclosing_locals, strips, bound, root)
        }
    }
}

/// Prefix-swapped alias maps for one elided subobject parameter:
/// `pname.rest...` for every flattened leaf under `subpath`, plus array
/// positions shifted into parameter coordinates.
///
/// Coordinate note: `array_at` positions count segments from the root
/// (`srt_data.buffers.tile` → 1), but a use through `pname` (bound to
/// `subpath = srt_data.scene_data`) counts from `pname`. The array
/// segment sits `shift = len(subpath) - 1` segments deeper in root
/// coordinates, so swapped positions subtract the shift (dropping to
/// nothing when the array is at or above the binding: no use through
/// the parameter can select it).
fn swapped_param_maps(
    resource_map: &HashMap<String, String>,
    scalar_map: &HashMap<String, String>,
    array_at: &HashMap<String, usize>,
    subpath: &str,
    pname: &str,
) -> (
    HashMap<String, String>,
    HashMap<String, String>,
    HashMap<String, usize>,
) {
    let shift = subpath.split('.').count().saturating_sub(1);
    let mut res = HashMap::new();
    let mut scal = HashMap::new();
    for (path, flat) in resource_map.iter().chain(scalar_map.iter()) {
        let is_res = resource_map.contains_key(path);
        if let Some(rest) = path.strip_prefix(&format!("{subpath}.")) {
            let key = format!("{pname}.{rest}");
            if is_res {
                res.insert(key, flat.clone());
            } else {
                scal.insert(key, flat.clone());
            }
        }
    }
    let mut at = HashMap::new();
    for (path, pos) in array_at {
        if let Some(rest) = path.strip_prefix(&format!("{subpath}.")) {
            if let Some(newpos) = pos.checked_sub(shift) {
                if newpos > 0 {
                    at.insert(format!("{pname}.{rest}"), newpos);
                }
            }
        }
    }
    (res, scal, at)
}

/// True when a use of `param` survives the downstream passes
/// uncovered: any `param` variable outside a direct call argument, or
/// any encoded callee mentioning it. Direct call arguments are the
/// post-elision check's job (callee elision), not this scan's.
fn param_use_uncovered(stmts: &[IrStmt], param: &str) -> bool {
    fn expr(e: &IrExpr, param: &str, direct_arg: bool) -> bool {
        match e {
            IrExpr::Variable(v) => v == param && !direct_arg,
            IrExpr::Member { object, .. } => expr(object, param, false),
            IrExpr::Index { object, index } | IrExpr::ResourceIndex { resource: object, index } => {
                expr(object, param, false) || expr(index, param, false)
            }
            IrExpr::ResourceLoad { resource, coordinates, sample } => {
                expr(resource, param, false)
                    || expr(coordinates, param, false)
                    || sample.as_ref().is_some_and(|s| expr(s, param, false))
            }
            IrExpr::Call { function, arguments } => {
                if callee_mentions_ident(function, param) {
                    return true;
                }
                arguments.iter().any(|a| expr(a, param, true))
            }
            IrExpr::Construct { arguments, .. } => {
                arguments.iter().any(|a| expr(a, param, false))
            }
            IrExpr::Unary { expr: inner, .. } => expr(inner, param, false),
            IrExpr::Binary { left, right, .. } => {
                expr(left, param, false) || expr(right, param, false)
            }
            IrExpr::Conditional { condition, then_branch, else_branch } => {
                expr(condition, param, false)
                    || expr(then_branch, param, false)
                    || expr(else_branch, param, false)
            }
            IrExpr::Assign { lhs, rhs } => {
                expr(lhs, param, false) || expr(rhs, param, false)
            }
            IrExpr::InitializerList(elems) => {
                elems.iter().any(|x| expr(x, param, false))
            }
            IrExpr::Literal(_) => false,
        }
    }
    fn stmt(s: &IrStmt, param: &str) -> bool {
        match s {
            IrStmt::Return(e) => e.as_ref().is_some_and(|e| expr(e, param, false)),
            IrStmt::VarDecl { init, .. } => {
                init.as_ref().is_some_and(|e| expr(e, param, false))
            }
            IrStmt::If { condition, then_branch, else_branch } => {
                expr(condition, param, false)
                    || then_branch.iter().any(|x| stmt(x, param))
                    || else_branch.as_ref().is_some_and(|eb| eb.iter().any(|x| stmt(x, param)))
            }
            IrStmt::For { init, condition, update, body } => {
                stmt(init, param)
                    || condition.as_ref().is_some_and(|c| expr(c, param, false))
                    || update.as_ref().is_some_and(|u| expr(u, param, false))
                    || body.iter().any(|x| stmt(x, param))
            }
            IrStmt::While { condition, body } => {
                expr(condition, param, false) || body.iter().any(|x| stmt(x, param))
            }
            IrStmt::Switch { scrutinee, cases } => {
                expr(scrutinee, param, false)
                    || cases.iter().any(|c| {
                        c.label.as_ref().is_some_and(|l| expr(l, param, false))
                            || c.body.iter().any(|x| stmt(x, param))
                    })
            }
            IrStmt::Block(ss) => ss.iter().any(|x| stmt(x, param)),
            IrStmt::Expr(e) => expr(e, param, false),
        }
    }
    stmts.iter().any(|s| stmt(s, param))
}

/// True when every `param`-rooted use in `body` rebinds through the
/// downstream passes: clones the body and runs the indexed-copy, alias,
/// transfer, and plain rewrites with temporary param-swapped maps, then
/// scans for leftovers (see `param_use_uncovered`). Direct call
/// arguments are exempt (the post-elision check assigns them to elided
/// callees). This is the constructive coverage gate: whatever the
/// passes cannot rewrite stays loud by vetoed elision instead of
/// dangling after parameter removal.
#[allow(clippy::too_many_arguments)]
fn subobject_uses_covered(
    body: &[IrStmt],
    param: &str,
    subpath: &str,
    params: &[crate::ir::function::Parameter],
    fields: &StructFields,
    structs: &[StructDecl],
    root_struct: &str,
    root: &str,
    resource_map: &HashMap<String, String>,
    scalar_map: &HashMap<String, String>,
    array_at: &HashMap<String, usize>,
) -> bool {
    let (tmp_res, tmp_scal, tmp_at) =
        swapped_param_maps(resource_map, scalar_map, array_at, subpath, param);
    let mut tmp_res_full = resource_map.clone();
    tmp_res_full.extend(tmp_res);
    let mut tmp_scal_full = scalar_map.clone();
    tmp_scal_full.extend(tmp_scal);
    let mut tmp_at_full = array_at.clone();
    for (k, v) in tmp_at {
        tmp_at_full.entry(k).or_insert(v);
    }
    let param_roots = vec![param.to_string()];
    let mut probe = body.to_vec();
    materialize_indexed_copies(
        &mut probe,
        root,
        &param_roots,
        params,
        fields,
        structs,
        root_struct,
        &tmp_scal_full,
    );
    resolve_srt_aliases(&mut probe, root, &param_roots, params, fields, structs, root_struct);
    transfer_indexed_uses(&mut probe, &tmp_res_full, &tmp_scal_full, &tmp_at_full);
    for stmt in &mut probe {
        rewrite_stmt(stmt, root, &tmp_res_full, &tmp_scal_full);
    }
    !param_use_uncovered(&probe, param)
}

/// Builds a name -> fields map for every struct declaration in the IR.
fn build_struct_map(ir: &ShaderModule) -> StructFields {
    ir.structs
        .iter()
        .map(|s| {
            let flds = s
                .fields
                .iter()
                .map(|f| (f.name.clone(), f.ty.clone()))
                .collect();
            (s.name.clone(), flds)
        })
        .collect()
}

/// Recursively walks a struct's fields, recording full dotted paths (rooted at
/// the SRT variable) that are GPU resources vs. scalar uniforms.
#[allow(clippy::too_many_arguments)]
fn walk_struct(
    fields: &StructFields,
    struct_name: &str,
    path: &mut Vec<String>,
    res: &mut Vec<(String, IrType)>,
    scal: &mut Vec<(String, IrType)>,
    visited: &mut Vec<String>,
) {
    let Some(field_list) = fields.get(struct_name) else {
        return;
    };
    visited.push(struct_name.to_string());
    for (fname, fty) in field_list {
        path.push(fname.clone());
        let full = path.join(".");
        match fty {
            IrType::Resource { .. } => res.push((full.clone(), fty.clone())),
            IrType::Named(n) if fields.contains_key(n) => {
                walk_struct(fields, n, path, res, scal, visited);
            }
            // Sampler members are resources (top-level bindings), never
            // cbuffer scalars.
            IrType::Named(n) if crate::framework::is_sampler_type_name(n) => {
                res.push((full.clone(), fty.clone()))
            }
            // Resource-like framework objects (`sce::Agc::Core::Texture`,
            // `sce::Psr::TopLevelBvhDescriptor`): externally-bound
            // top-level bindings, never cbuffer scalars (a `Texture2D<float4>`
            // / acceleration structure inside a cbuffer is not valid HLSL).
            IrType::Named(n)
                if crate::framework::framework_resource_kind(n).is_some() =>
            {
                res.push((full.clone(), fty.clone()))
            }
            // Bare texture spellings (`Texture2D tex`): same treatment
            // (sampled default float4); see the binding arm above.
            IrType::Named(n)
                if matches!(
                    n.to_ascii_lowercase().as_str(),
                    "texture1d" | "texture2d" | "texturecube" | "texture3d"
                ) =>
            {
                res.push((full.clone(), fty.clone()))
            }
            // Scalar / vector / matrix / unknown-named: a uniform member.
            _ => scal.push((full.clone(), fty.clone())),
        }
        path.pop();
    }
}

/// Builds a top-level binding for a resource-like framework object
/// (`sce::Agc::Core::Texture` / `sce::Agc::Core::Buffer`,
/// `sce::Psr::{Top,Bottom}LevelBvhDescriptor`).
/// Textures reuse the ordinary texture path (`Texture2D<float4>`); the
/// untyped kinds (BVH descriptor, byte-address buffer) emit bare
/// (`RaytracingAccelerationStructure name;`, `ByteAddressBuffer name;` —
/// no element suffix, see `resource_kind_name`).
fn framework_object_resource(name: &str, resource_name: &str) -> Resource {
    let kind = crate::framework::framework_resource_kind(name).unwrap_or("texture2d");
    if kind == "texture2d" {
        let element = IrType::Vector {
            base: IrScalar::Float,
            width: 4,
        };
        resource_resource("texture2d", IrAccess::ReadOnly, &element, resource_name)
    } else {
        Resource {
            name: resource_name.to_string(),
            resource_type: ResourceType::Texture,
            kind: kind.to_string(),
            element_type: None,
            descriptor_set: Some(0),
            binding: None,
            register: None,
            space: None,
            array_size: None,
            fields: Vec::new(),
        }
    }
}

/// Builds a top-level `SamplerState` binding for a hoisted sampler member
/// (emitted as a plain `SamplerState name;` — Vulkan samplers are
/// externally bound).
fn sampler_resource(name: &str) -> Resource {
    Resource {
        name: name.to_string(),
        resource_type: ResourceType::Sampler,
        kind: "samplerstate".to_string(),
        element_type: None,
        descriptor_set: Some(0),
        binding: None,
        register: None,
        space: None,
        array_size: None,
        fields: Vec::new(),
    }
}

/// Builds a `Resource` for a texturing/data buffer from an IR resource type.
fn resource_resource(kind: &str, access: IrAccess, element: &IrType, name: &str) -> Resource {
    let resource_type = match kind {
        "texture1d" | "texture2d" | "texture2darray" | "texturecube" | "texturecubearray"
        | "texture3d" | "mstexture2d" => ResourceType::Texture,
        "rwtexture2d" | "rwtexture2darray" | "rwtexture3d" | "rwtexture1d" => {
            ResourceType::RWTexture
        }
        "databuffer"
            | "regularbuffer"
            | "structuredbuffer"
            | "bytebuffer"
            | "byteaddressbuffer" => ResourceType::StructuredBuffer,
        "rwdatabuffer"
            | "rwregularbuffer"
            | "rwstructuredbuffer"
            | "rwbuffer"
            | "rwbytebuffer"
            | "rwbyteaddressbuffer"
            | "appendregularbuffer"
            | "appendstructuredbuffer"
            | "consumeregularbuffer"
            | "consumestructuredbuffer" => ResourceType::RWBuffer,
        _ => {
            if matches!(access, IrAccess::ReadWrite) {
                ResourceType::RWTexture
            } else {
                ResourceType::Texture
            }
        }
    };
    Resource {
        name: name.to_string(),
        resource_type,
        kind: kind.to_string(),
        element_type: Some(element.clone()),
        descriptor_set: Some(0),
        binding: None,
        register: None,
        space: None,
        array_size: None,
        fields: Vec::new(),
    }
}

/// Builds a single ConstantBuffer resource holding all scalar/uniform members.
/// `field_names` maps each scalar path to its (deduplicated) cbuffer field
/// name, so colliding bare member names across SRT structs stay distinct.
/// `arrays` carries sized array dimensions by path: cbuffer array members
/// keep their size so indexed uses (`srt.a[i]` → `a[i]`) stay valid.
/// Unsized/unknown arrays keep today's scalar shape.
fn scalar_cbuffer(
    name: &str,
    scal: &[(String, IrType)],
    field_names: &std::collections::HashMap<String, String>,
    arrays: &std::collections::HashMap<String, u32>,
) -> Resource {
    let mut fields = Vec::new();
    for (path, ty) in scal {
        let fname = field_names
            .get(path)
            .cloned()
            .unwrap_or_else(|| path.rsplit('.').next().unwrap_or(path).to_string());
        fields.push(FieldDecl {
            ty: ty.clone(),
            name: fname,
            semantic: None,
            array_size: arrays.get(path).copied(),
        });
    }
    Resource {
        name: name.to_string(),
        resource_type: ResourceType::ConstantBuffer,
        kind: "constantbuffer".to_string(),
        element_type: None,
        descriptor_set: Some(0),
        binding: None,
        register: None,
        space: None,
        array_size: None,
        fields,
    }
}

/// Rewrites every resource/uniform access within a single statement.
fn rewrite_stmt(
    stmt: &mut IrStmt,
    root: &str,
    resource_map: &HashMap<String, String>,
    scalar_map: &HashMap<String, String>,
) {
    match stmt {
        IrStmt::Return(e) => {
            if let Some(e) = e {
                rewrite_expr(e, root, resource_map, scalar_map);
            }
        }
        IrStmt::VarDecl { init, .. } => {
            if let Some(init) = init {
                rewrite_expr(init, root, resource_map, scalar_map);
            }
        }
        IrStmt::If {
            condition,
            then_branch,
            else_branch,
        } => {
            rewrite_expr(condition, root, resource_map, scalar_map);
            for s in then_branch {
                rewrite_stmt(s, root, resource_map, scalar_map);
            }
            if let Some(eb) = else_branch {
                for s in eb {
                    rewrite_stmt(s, root, resource_map, scalar_map);
                }
            }
        }
        IrStmt::For {
            init,
            condition,
            update,
            body,
        } => {
            rewrite_stmt(init, root, resource_map, scalar_map);
            if let Some(cond) = condition {
                rewrite_expr(cond, root, resource_map, scalar_map);
            }
            if let Some(upd) = update {
                rewrite_expr(upd, root, resource_map, scalar_map);
            }
            for s in body {
                rewrite_stmt(s, root, resource_map, scalar_map);
            }
        }
        IrStmt::While { condition, body } => {
            rewrite_expr(condition, root, resource_map, scalar_map);
            for s in body {
                rewrite_stmt(s, root, resource_map, scalar_map);
            }
        }
        IrStmt::Switch { scrutinee, cases } => {
            rewrite_expr(scrutinee, root, resource_map, scalar_map);
            for case in cases {
                if let Some(label) = &mut case.label {
                    rewrite_expr(label, root, resource_map, scalar_map);
                }
                for s in &mut case.body {
                    rewrite_stmt(s, root, resource_map, scalar_map);
                }
            }
        }
        IrStmt::Block(stmts) => {
            for s in stmts {
                rewrite_stmt(s, root, resource_map, scalar_map);
            }
        }
        IrStmt::Expr(e) => {
            rewrite_expr(e, root, resource_map, scalar_map);
        }
    }
}

/// Rewrites a single expression in place.
#[allow(clippy::only_used_in_recursion)]
fn rewrite_expr(
    expr: &mut IrExpr,
    root: &str,
    resource_map: &HashMap<String, String>,
    scalar_map: &HashMap<String, String>,
) -> bool {
    match expr {
        IrExpr::Variable(_) | IrExpr::Literal(_) => false,

        IrExpr::Member { object, member } => {
            let candidate = IrExpr::Member {
                object: object.clone(),
                member: member.clone(),
            };
            if let Some(path) = path_of(&candidate) {
                if let Some(res) = resource_map.get(&path) {
                    *expr = IrExpr::Variable(res.clone());
                    return true;
                }
                if let Some(last) = scalar_map.get(&path) {
                    *expr = IrExpr::Variable(last.clone());
                    return true;
                }
            }
            rewrite_expr(object, root, resource_map, scalar_map)
        }

        IrExpr::Index { object, index } => {
            let object_replaced = rewrite_expr(object, root, resource_map, scalar_map);
            rewrite_expr(index, root, resource_map, scalar_map);
            object_replaced
        }

        IrExpr::ResourceIndex { resource, index } => {
            let r = rewrite_expr(resource, root, resource_map, scalar_map);
            rewrite_expr(index, root, resource_map, scalar_map);
            r
        }

        IrExpr::ResourceLoad {
            resource,
            coordinates,
            sample,
        } => {
            let r = rewrite_expr(resource, root, resource_map, scalar_map);
            rewrite_expr(coordinates, root, resource_map, scalar_map);
            if let Some(s) = sample {
                rewrite_expr(s, root, resource_map, scalar_map);
            }
            r
        }

        IrExpr::Call { function, arguments } => {
            // A resource-method object may be encoded in the dotted function
            // name (e.g. `srt.tx.src.SampleLevel`). Buffer members add an
            // index between path and method
            // (`srt.mats.materials[i].albedo.Sample`): match that spelling
            // too, keeping the index suffix verbatim for downstream
            // (hoist/split) buffer-index rewrites.
            for (path, res_name) in resource_map {
                let rest = if let Some(stripped) = function.strip_prefix(&format!("{path}.")) {
                    Some(stripped)
                } else if let Some(stripped) = function.strip_prefix(path.as_str()) {
                    if stripped.starts_with('[') {
                        Some(stripped)
                    } else {
                        None
                    }
                } else {
                    None
                };
                if let Some(rest) = rest {
                    if rest.starts_with('[') {
                        *function = format!("{res_name}{rest}");
                    } else {
                        *function = format!("{res_name}.{rest}");
                    }
                    break;
                }
            }
            for a in arguments {
                rewrite_expr(a, root, resource_map, scalar_map);
            }
            true
        }

        IrExpr::Construct { arguments, .. } => {
            for a in arguments {
                rewrite_expr(a, root, resource_map, scalar_map);
            }
            false
        }

        IrExpr::Unary { expr: inner, .. } => rewrite_expr(inner, root, resource_map, scalar_map),

        IrExpr::Binary { left, right, .. } => {
            let l = rewrite_expr(left, root, resource_map, scalar_map);
            let r = rewrite_expr(right, root, resource_map, scalar_map);
            l || r
        }

        IrExpr::Conditional {
            condition,
            then_branch,
            else_branch,
        } => {
            let a = rewrite_expr(condition, root, resource_map, scalar_map);
            let b = rewrite_expr(then_branch, root, resource_map, scalar_map);
            let c = rewrite_expr(else_branch, root, resource_map, scalar_map);
            a || b || c
        }

        IrExpr::Assign { lhs, rhs } => {
            let l = rewrite_expr(lhs, root, resource_map, scalar_map);
            let r = rewrite_expr(rhs, root, resource_map, scalar_map);
            l || r
        }

        IrExpr::InitializerList(elems) => {
            for e in elems {
                rewrite_expr(e, root, resource_map, scalar_map);
            }
            false
        }
    }
}

/// Flattens a `Variable`/`Member` expression into a dotted path string, if
/// possible (e.g. `Variable("srt")` -> `"srt"`, `srt.uv_scale` -> `"srt.uv_scale"`).
fn path_of(expr: &IrExpr) -> Option<String> {
    match expr {
        IrExpr::Variable(v) => Some(v.clone()),
        IrExpr::Member { object, member } => path_of(object).map(|p| format!("{p}.{member}")),
        _ => None,
    }
}

/// Root variable of a member/index path (`srt.a[k].b` -> `srt`), if the
/// base is a bare variable.
fn path_root(expr: &IrExpr) -> Option<String> {
    match expr {
        IrExpr::Variable(v) => Some(v.clone()),
        IrExpr::Member { object, .. } => path_root(object),
        IrExpr::Index { object, .. } => path_root(object),
        IrExpr::ResourceIndex { resource, .. } => path_root(resource),
        _ => None,
    }
}

/// Decomposes a member/index chain into root-first segments plus the
/// index selections with the segment position they sit after and whether
/// they select through a resource (`srt_data.buffers[k].tile[j]` ->
/// segments `[srt_data, buffers, tile]`, indices `[(1, plain, k),
/// (2, plain, j)]`). Anything else yields `None`.
fn split_indexed(e: &IrExpr) -> Option<(Vec<String>, Vec<(usize, bool, IrExpr)>)> {
    match e {
        IrExpr::Variable(v) => Some((vec![v.clone()], vec![])),
        IrExpr::Member { object, member } => {
            let (mut segs, idxs) = split_indexed(object)?;
            segs.push(member.clone());
            Some((segs, idxs))
        }
        IrExpr::Index { object, index } => {
            let (segs, mut idxs) = split_indexed(object)?;
            if segs.is_empty() {
                return None;
            }
            idxs.push((segs.len() - 1, false, (**index).clone()));
            Some((segs, idxs))
        }
        IrExpr::ResourceIndex { resource, index } => {
            let (segs, mut idxs) = split_indexed(resource)?;
            if segs.is_empty() {
                return None;
            }
            idxs.push((segs.len() - 1, true, (**index).clone()));
            Some((segs, idxs))
        }
        _ => None,
    }
}

/// Transfers a single intermediate-struct-array index inward
/// (`root.arr[k].leaf` -> `BINDING[k]`, trailing selections preserved:
/// `root.arr[k].leaf[j]` -> `BINDING[k][j]`).
///
/// Only fires when the de-indexed path is a dimensioned leaf (`array_at`)
/// and the index sits exactly at the array segment: any other placement
/// selects a different object and stays loud. Scalar leaves reject
/// trailing selections (indexing a scalar is invalid); resource leaves
/// keep them (buffer-element selects). Returns whether anything changed.
fn transfer_indexed(
    expr: &mut IrExpr,
    resource_map: &HashMap<String, String>,
    scalar_map: &HashMap<String, String>,
    array_at: &HashMap<String, usize>,
) -> bool {
    let (segs, idxs) = match split_indexed(expr) {
        Some(v) => v,
        None => return false,
    };
    if segs.len() < 3 {
        return false;
    }
    let depath = segs.join(".");
    let at = match array_at.get(&depath) {
        Some(at) => *at,
        None => return false,
    };
    // Exactly one index at the array segment; nothing between it and the
    // leaf (indexing into a struct is invalid); trailing selections only
    // past the leaf.
    let n = segs.len() - 1;
    let mut at_idx: Option<IrExpr> = None;
    for (pos, _, idx) in &idxs {
        if *pos == at {
            if at_idx.is_some() {
                return false;
            }
            at_idx = Some(idx.clone());
        } else if *pos < n {
            return false;
        }
    }
    let at_idx = match at_idx {
        Some(e) => e,
        None => return false,
    };
    let trailing: Vec<(bool, IrExpr)> = idxs
        .iter()
        .filter(|(p, _, _)| *p == n)
        .map(|(_, res, e)| (*res, e.clone()))
        .collect();
    if let Some(binding) = resource_map.get(&depath) {
        let mut out = IrExpr::Index {
            object: Box::new(IrExpr::Variable(binding.clone())),
            index: Box::new(at_idx),
        };
        for (is_res, idx) in trailing {
            if is_res {
                out = IrExpr::ResourceIndex {
                    resource: Box::new(out),
                    index: Box::new(idx),
                };
            } else {
                out = IrExpr::Index {
                    object: Box::new(out),
                    index: Box::new(idx),
                };
            }
        }
        *expr = out;
        return true;
    }
    if let Some(name) = scalar_map.get(&depath) {
        if !trailing.is_empty() {
            return false;
        }
        *expr = IrExpr::Index {
            object: Box::new(IrExpr::Variable(name.clone())),
            index: Box::new(at_idx),
        };
        return true;
    }
    false
}

/// Pre-pass transferring struct-array indices inward
/// (`root.arr[k].leaf` -> `BINDING[k]`, see `transfer_indexed`) in every
/// statement, ahead of the plain path rewrite (which then sees ordinary
/// indexed bindings). Same statement/expression coverage as
/// `rewrite_stmt`/`rewrite_expr`; non-matching shapes pass through for
/// the later passes (fail-closed).
fn transfer_indexed_uses(
    stmts: &mut [IrStmt],
    resource_map: &HashMap<String, String>,
    scalar_map: &HashMap<String, String>,
    array_at: &HashMap<String, usize>,
) {
    fn stmt(
        s: &mut IrStmt,
        resource_map: &HashMap<String, String>,
        scalar_map: &HashMap<String, String>,
        array_at: &HashMap<String, usize>,
    ) {
        match s {
            IrStmt::Return(e) => {
                if let Some(e) = e {
                    expr(e, resource_map, scalar_map, array_at);
                }
            }
            IrStmt::VarDecl { init, .. } => {
                if let Some(init) = init {
                    expr(init, resource_map, scalar_map, array_at);
                }
            }
            IrStmt::If { condition, then_branch, else_branch } => {
                expr(condition, resource_map, scalar_map, array_at);
                transfer_indexed_uses(then_branch, resource_map, scalar_map, array_at);
                if let Some(eb) = else_branch {
                    transfer_indexed_uses(eb, resource_map, scalar_map, array_at);
                }
            }
            IrStmt::For { init, condition, update, body } => {
                stmt(init, resource_map, scalar_map, array_at);
                if let Some(c) = condition {
                    expr(c, resource_map, scalar_map, array_at);
                }
                if let Some(u) = update {
                    expr(u, resource_map, scalar_map, array_at);
                }
                transfer_indexed_uses(body, resource_map, scalar_map, array_at);
            }
            IrStmt::While { condition, body } => {
                expr(condition, resource_map, scalar_map, array_at);
                transfer_indexed_uses(body, resource_map, scalar_map, array_at);
            }
            IrStmt::Switch { scrutinee, cases } => {
                expr(scrutinee, resource_map, scalar_map, array_at);
                for c in cases {
                    if let Some(label) = &mut c.label {
                        expr(label, resource_map, scalar_map, array_at);
                    }
                    transfer_indexed_uses(&mut c.body, resource_map, scalar_map, array_at);
                }
            }
            IrStmt::Block(inner) => {
                transfer_indexed_uses(inner, resource_map, scalar_map, array_at)
            }
            IrStmt::Expr(e) => expr(e, resource_map, scalar_map, array_at),
        }
    }
    fn expr(
        e: &mut IrExpr,
        resource_map: &HashMap<String, String>,
        scalar_map: &HashMap<String, String>,
        array_at: &HashMap<String, usize>,
    ) {
        match e {
            IrExpr::Member { .. } | IrExpr::Index { .. } | IrExpr::ResourceIndex { .. } => {
                if !transfer_indexed(e, resource_map, scalar_map, array_at) {
                    match e {
                        IrExpr::Member { object, .. } => {
                            expr(object, resource_map, scalar_map, array_at)
                        }
                        IrExpr::Index { object, index }
                        | IrExpr::ResourceIndex { resource: object, index } => {
                            expr(object, resource_map, scalar_map, array_at);
                            expr(index, resource_map, scalar_map, array_at);
                        }
                        _ => unreachable!("guarded by outer match"),
                    }
                } else {
                    // Rebuilt bindings embed the cloned index: recurse for
                    // nested uses inside it.
                    expr(e, resource_map, scalar_map, array_at);
                }
            }
            IrExpr::ResourceLoad { resource, coordinates, sample } => {
                expr(resource, resource_map, scalar_map, array_at);
                expr(coordinates, resource_map, scalar_map, array_at);
                if let Some(s) = sample {
                    expr(s, resource_map, scalar_map, array_at);
                }
            }
            // Call callees keep the encoded-name surgery of the main
            // rewrite (untouched here); arguments still transfer.
            IrExpr::Call { arguments, .. } => {
                for a in arguments {
                    expr(a, resource_map, scalar_map, array_at);
                }
            }
            IrExpr::Construct { arguments, .. } => {
                for a in arguments {
                    expr(a, resource_map, scalar_map, array_at);
                }
            }
            IrExpr::Unary { expr: inner, .. } => expr(inner, resource_map, scalar_map, array_at),
            IrExpr::Binary { left, right, .. } => {
                expr(left, resource_map, scalar_map, array_at);
                expr(right, resource_map, scalar_map, array_at);
            }
            IrExpr::Conditional { condition, then_branch, else_branch } => {
                expr(condition, resource_map, scalar_map, array_at);
                expr(then_branch, resource_map, scalar_map, array_at);
                expr(else_branch, resource_map, scalar_map, array_at);
            }
            IrExpr::Assign { lhs, rhs } => {
                expr(lhs, resource_map, scalar_map, array_at);
                expr(rhs, resource_map, scalar_map, array_at);
            }
            IrExpr::InitializerList(elems) => {
                for x in elems {
                    expr(x, resource_map, scalar_map, array_at);
                }
            }
            IrExpr::Variable(_) | IrExpr::Literal(_) => {}
        }
    }
    for s in stmts.iter_mut() {
        stmt(s, resource_map, scalar_map, array_at);
    }
}

/// True when `e` is side-effect-free to duplicate (`Assign` and `Call`
/// nodes may run effects or re-execute).
fn index_pure(e: &IrExpr) -> bool {
    match e {
        IrExpr::Call { .. } => false,
        IrExpr::Assign { .. } => false,
        IrExpr::Variable(_) | IrExpr::Literal(_) => true,
        IrExpr::Member { object, .. } => index_pure(object),
        IrExpr::Index { object, index } => index_pure(object) && index_pure(index),
        IrExpr::ResourceIndex { resource, index } => {
            index_pure(resource) && index_pure(index)
        }
        IrExpr::ResourceLoad { resource, coordinates, sample } => {
            index_pure(resource)
                && index_pure(coordinates)
                && sample.as_ref().is_none_or(|s| index_pure(s))
        }
        IrExpr::Construct { arguments, .. } => arguments.iter().all(index_pure),
        IrExpr::Unary { expr: inner, .. } => index_pure(inner),
        IrExpr::Binary { left, right, .. } => index_pure(left) && index_pure(right),
        IrExpr::Conditional { condition, then_branch, else_branch } => {
            index_pure(condition) && index_pure(then_branch) && index_pure(else_branch)
        }
        IrExpr::InitializerList(elems) => elems.iter().all(index_pure),
    }
}

/// A local aliasing one element of an SRT struct array
/// (`ClassifySrtBuffers buffers = srt_data.buffers[k];`).
struct SrtAlias {
    name: String,
    init: IrExpr,
    /// Element struct name.
    elem: String,
    /// De-indexed path (`srt_data.buffers`).
    depath: String,
    /// The index expression (pure: duplicated per use).
    index: IrExpr,
}

/// True when `e` mentions the variable `name` anywhere (any position).
fn mentions_var(e: &IrExpr, name: &str) -> bool {
    match e {
        IrExpr::Variable(v) => v == name,
        IrExpr::Member { object, .. } => mentions_var(object, name),
        IrExpr::Index { object, index } => {
            mentions_var(object, name) || mentions_var(index, name)
        }
        IrExpr::ResourceIndex { resource, index } => {
            mentions_var(resource, name) || mentions_var(index, name)
        }
        IrExpr::ResourceLoad { resource, coordinates, sample } => {
            mentions_var(resource, name)
                || mentions_var(coordinates, name)
                || sample.as_ref().is_some_and(|s| mentions_var(s, name))
        }
        IrExpr::Call { function, arguments } => {
            callee_mentions_ident(function, name) || arguments.iter().any(|a| mentions_var(a, name))
        }
        IrExpr::Construct { arguments, .. } => arguments.iter().any(|a| mentions_var(a, name)),
        IrExpr::Unary { expr: inner, .. } => mentions_var(inner, name),
        IrExpr::Binary { left, right, .. } => {
            mentions_var(left, name) || mentions_var(right, name)
        }
        IrExpr::Conditional { condition, then_branch, else_branch } => {
            mentions_var(condition, name)
                || mentions_var(then_branch, name)
                || mentions_var(else_branch, name)
        }
        IrExpr::Assign { lhs, rhs } => mentions_var(lhs, name) || mentions_var(rhs, name),
        IrExpr::InitializerList(elems) => elems.iter().any(|x| mentions_var(x, name)),
        IrExpr::Literal(_) => false,
    }
}

/// True when the encoded call-callee string references the identifier
/// `name` as a whole token (`buffers.tile.Load` counts; `mybuffers.x`
/// does not). Such calls escape member-root substitution.
fn callee_mentions_ident(callee: &str, name: &str) -> bool {
    callee
        .split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
        .any(|tok| tok == name)
}

/// True when `alias` escapes alias substitution anywhere in `stmts`
/// (the owning declaration's suffix scope) and the alias must stay
/// loud: reassigned or redeclared, used bare (outside a member-rooted
/// path: whole-value call args, returns, callee positions, encoded
/// method callees), or mentioned in its own initializer. Member-rooted
/// uses are exactly what substitution rewrites, so they are fine.
fn alias_violated(body: &[IrStmt], alias: &str, init: &IrExpr) -> bool {
    if mentions_var(init, alias) {
        return true;
    }
    fn expr_bad(e: &IrExpr, alias: &str, under_member: bool, bad: &mut bool) {
        if *bad {
            return;
        }
        match e {
            IrExpr::Variable(v) => {
                if v == alias && !under_member {
                    *bad = true;
                }
            }
            IrExpr::Member { object, .. } => expr_bad(object, alias, true, bad),
            IrExpr::Index { object, index } | IrExpr::ResourceIndex { resource: object, index } => {
                expr_bad(object, alias, false, bad);
                expr_bad(index, alias, false, bad);
            }
            IrExpr::ResourceLoad { resource, coordinates, sample } => {
                expr_bad(resource, alias, false, bad);
                expr_bad(coordinates, alias, false, bad);
                if let Some(s) = sample {
                    expr_bad(s, alias, false, bad);
                }
            }
            IrExpr::Call { function, arguments } => {
                if callee_mentions_ident(function, alias) {
                    *bad = true;
                    return;
                }
                for a in arguments {
                    expr_bad(a, alias, false, bad);
                }
            }
            IrExpr::Construct { arguments, .. } => {
                for a in arguments {
                    expr_bad(a, alias, false, bad);
                }
            }
            IrExpr::Unary { expr: inner, .. } => expr_bad(inner, alias, false, bad),
            IrExpr::Binary { left, right, .. } => {
                expr_bad(left, alias, false, bad);
                expr_bad(right, alias, false, bad);
            }
            IrExpr::Conditional { condition, then_branch, else_branch } => {
                expr_bad(condition, alias, false, bad);
                expr_bad(then_branch, alias, false, bad);
                expr_bad(else_branch, alias, false, bad);
            }
            IrExpr::Assign { lhs, rhs } => {
                expr_bad(lhs, alias, false, bad);
                expr_bad(rhs, alias, false, bad);
            }
            IrExpr::InitializerList(elems) => {
                for x in elems {
                    expr_bad(x, alias, false, bad);
                }
            }
            IrExpr::Literal(_) => {}
        }
    }
    // Whole-value stores to the alias itself (`buffers = ...`) rebind
    // it: always a violation (substitution assumes one binding).
    fn stmt_binds(s: &IrStmt, alias: &str, decls: &mut u32, bad: &mut bool) {
        match s {
            IrStmt::VarDecl { name, init, .. } => {
                if name == alias {
                    *decls += 1;
                }
                if let Some(e) = init {
                    expr_bad(e, alias, false, bad);
                }
            }
            IrStmt::Return(e) => {
                if let Some(e) = e {
                    expr_bad(e, alias, false, bad);
                }
            }
            IrStmt::Expr(e) => {
                if let IrExpr::Assign { lhs, .. } = e {
                    if matches!(lhs.as_ref(), IrExpr::Variable(v) if v == alias) {
                        *bad = true;
                        return;
                    }
                }
                expr_bad(e, alias, false, bad);
            }
            IrStmt::Block(ss) => {
                for x in ss {
                    stmt_binds(x, alias, decls, bad);
                }
            }
            IrStmt::If { condition, then_branch, else_branch } => {
                expr_bad(condition, alias, false, bad);
                for x in then_branch {
                    stmt_binds(x, alias, decls, bad);
                }
                if let Some(eb) = else_branch {
                    for x in eb {
                        stmt_binds(x, alias, decls, bad);
                    }
                }
            }
            IrStmt::For { init, condition, update, body } => {
                stmt_binds(init, alias, decls, bad);
                if let Some(c) = condition {
                    expr_bad(c, alias, false, bad);
                }
                if let Some(u) = update {
                    expr_bad(u, alias, false, bad);
                }
                for x in body {
                    stmt_binds(x, alias, decls, bad);
                }
            }
            IrStmt::While { condition, body } => {
                expr_bad(condition, alias, false, bad);
                for x in body {
                    stmt_binds(x, alias, decls, bad);
                }
            }
            IrStmt::Switch { scrutinee, cases } => {
                expr_bad(scrutinee, alias, false, bad);
                for c in cases {
                    if let Some(label) = &c.label {
                        expr_bad(label, alias, false, bad);
                    }
                    for x in &c.body {
                        stmt_binds(x, alias, decls, bad);
                    }
                }
            }
        }
    }
    let mut decls = 0u32;
    let mut bad = false;
    for s in body {
        stmt_binds(s, alias, &mut decls, &mut bad);
    }
    // The owning declaration was already removed: any same-name
    // declaration left in the suffix shadows (or is invalid) and vetoes.
    bad || decls != 0
}

/// Substitutes `init` for member-rooted `alias` uses throughout (bare
/// uses were rejected by `alias_violated`, so the `Variable` arm is a
/// no-op: fail-closed).
fn subst_alias_in_stmts(stmts: &mut [IrStmt], alias: &SrtAlias) {
    fn stmt(s: &mut IrStmt, alias: &SrtAlias) {
        match s {
            IrStmt::Return(e) => {
                if let Some(e) = e {
                    expr(e, alias);
                }
            }
            IrStmt::VarDecl { init, .. } => {
                if let Some(e) = init {
                    expr(e, alias);
                }
            }
            IrStmt::If { condition, then_branch, else_branch } => {
                expr(condition, alias);
                subst_alias_in_stmts(then_branch, alias);
                if let Some(eb) = else_branch {
                    subst_alias_in_stmts(eb, alias);
                }
            }
            IrStmt::For { init, condition, update, body } => {
                stmt(init, alias);
                if let Some(c) = condition {
                    expr(c, alias);
                }
                if let Some(u) = update {
                    expr(u, alias);
                }
                subst_alias_in_stmts(body, alias);
            }
            IrStmt::While { condition, body } => {
                expr(condition, alias);
                subst_alias_in_stmts(body, alias);
            }
            IrStmt::Switch { scrutinee, cases } => {
                expr(scrutinee, alias);
                for c in cases {
                    if let Some(label) = &mut c.label {
                        expr(label, alias);
                    }
                    subst_alias_in_stmts(&mut c.body, alias);
                }
            }
            IrStmt::Block(inner) => subst_alias_in_stmts(inner, alias),
            IrStmt::Expr(e) => expr(e, alias),
        }
    }
    fn expr(e: &mut IrExpr, alias: &SrtAlias) {
        match e {
            IrExpr::Member { object, .. } => {
                if let IrExpr::Variable(v) = object.as_ref() {
                    if v == &alias.name {
                        *object = Box::new(alias.init.clone());
                        return;
                    }
                }
                expr(object, alias);
            }
            IrExpr::Index { object, index } | IrExpr::ResourceIndex { resource: object, index } => {
                expr(object, alias);
                expr(index, alias);
            }
            IrExpr::ResourceLoad { resource, coordinates, sample } => {
                expr(resource, alias);
                expr(coordinates, alias);
                if let Some(s) = sample {
                    expr(s, alias);
                }
            }
            // Call callees are encoded strings (never alias variables);
            // arguments still substitute.
            IrExpr::Call { arguments, .. } => {
                for a in arguments {
                    expr(a, alias);
                }
            }
            IrExpr::Construct { arguments, .. } => {
                for a in arguments {
                    expr(a, alias);
                }
            }
            IrExpr::Unary { expr: inner, .. } => expr(inner, alias),
            IrExpr::Binary { left, right, .. } => {
                expr(left, alias);
                expr(right, alias);
            }
            IrExpr::Conditional { condition, then_branch, else_branch } => {
                expr(condition, alias);
                expr(then_branch, alias);
                expr(else_branch, alias);
            }
            IrExpr::Assign { lhs, rhs } => {
                expr(lhs, alias);
                expr(rhs, alias);
            }
            IrExpr::InitializerList(elems) => {
                for x in elems {
                    expr(x, alias);
                }
            }
            IrExpr::Variable(_) | IrExpr::Literal(_) => {}
        }
    }
    for s in stmts.iter_mut() {
        stmt(s, alias);
    }
}

/// Recognizes `T name = root.a[k]` (or param-rooted) alias declarations:
/// `T` names the array's element struct (or `auto`), the index is pure,
/// and no nested arrays intervene. Anything else yields `None` (the
/// existing loud paths handle it).
#[allow(clippy::too_many_arguments)]
fn as_srt_alias(
    ty: &IrType,
    name: &str,
    init: &IrExpr,
    root: &str,
    param_roots: &[String],
    params: &[crate::ir::function::Parameter],
    fields: &StructFields,
    structs: &[StructDecl],
    root_struct: &str,
) -> Option<SrtAlias> {
    let (object, index) = match init {
        IrExpr::Index { object, index } => (object.as_ref(), index.as_ref()),
        IrExpr::ResourceIndex { resource, index } => (resource.as_ref(), index.as_ref()),
        _ => return None,
    };
    if !index_pure(index) {
        return None;
    }
    let full = path_of(object)?;
    let segs: Vec<&str> = full.split('.').collect();
    let base_struct = if segs[0] == root {
        root_struct.to_string()
    } else if param_roots.iter().any(|p| p == segs[0]) {
        let pty = params.iter().find(|p| p.name == segs[0]).map(|p| &p.ty)?;
        match pty {
            IrType::Named(s) => s.clone(),
            _ => return None,
        }
    } else {
        return None;
    };
    if segs.len() < 2 {
        return None;
    }
    let mut cur = base_struct;
    for seg in &segs[1..segs.len() - 1] {
        let fty = fields
            .get(&cur)?
            .iter()
            .find(|(n, _)| n == seg)
            .map(|(_, t)| t.clone())?;
        if field_array_size(structs, &cur, seg).is_some() {
            return None;
        }
        match fty {
            IrType::Named(s) if fields.contains_key(&s) => cur = s,
            _ => return None,
        }
    }
    let last = segs[segs.len() - 1];
    let fty = fields
        .get(&cur)?
        .iter()
        .find(|(n, _)| n == last)
        .map(|(_, t)| t.clone())?;
    if field_array_size(structs, &cur, last).filter(|&n| n > 0).is_none() {
        return None;
    }
    let IrType::Named(elem) = fty else {
        return None;
    };
    if !fields.contains_key(&elem) {
        return None;
    }
    match ty {
        IrType::Named(t) if t == "auto" || t == &elem => {}
        _ => return None,
    }
    Some(SrtAlias {
        name: name.to_string(),
        init: init.clone(),
        elem,
        depath: full,
        index: index.clone(),
    })
}

/// Scalar leaves of an all-scalar struct as (relative member segments,
/// optional array dimension): `None` on any resource (buffers, textures,
/// samplers, framework objects), unsized arrays, struct arrays, or more
/// than one array level below the indexed element (whole-element copies
/// linearize a single 2D index; deeper needs a backend).
fn scalar_copy_leaves(
    elem: &str,
    fields: &StructFields,
    structs: &[StructDecl],
) -> Option<Vec<(Vec<String>, Option<u32>)>> {
    fn scalar_ty(ty: &IrType) -> bool {
        matches!(
            ty,
            IrType::Bool
                | IrType::Int
                | IrType::UInt
                | IrType::Float
                | IrType::Half
                | IrType::Double
                | IrType::Vector { .. }
                | IrType::Matrix { .. }
        )
    }
    fn walk(
        sname: &str,
        fields: &StructFields,
        structs: &[StructDecl],
        rel: &mut Vec<String>,
        out: &mut Vec<(Vec<String>, Option<u32>)>,
    ) -> bool {
        let Some(flist) = fields.get(sname) else {
            return false;
        };
        for (fname, fty) in flist {
            if scalar_ty(fty) {
                match field_array_size(structs, sname, fname).filter(|&n| n > 0) {
                    None => {
                        let mut r = rel.clone();
                        r.push(fname.clone());
                        out.push((r, None));
                    }
                    Some(n) => {
                        let mut r = rel.clone();
                        r.push(fname.clone());
                        out.push((r, Some(n)));
                    }
                }
            } else if let IrType::Named(s) = fty {
                // Nested arrays (struct arrays or arrays of structs)
                // exceed the single-2D copy model.
                if field_array_size(structs, sname, fname).is_some() {
                    return false;
                }
                if !fields.contains_key(s) {
                    return false;
                }
                rel.push(fname.clone());
                if !walk(s, fields, structs, rel, out) {
                    return false;
                }
                rel.pop();
            } else {
                return false;
            }
        }
        true
    }
    let mut out = Vec::new();
    if !walk(elem, fields, structs, &mut Vec::new(), &mut out) {
        return None;
    }
    if out.is_empty() {
        return None;
    }
    Some(out)
}

/// Splices indexed whole-element copies of all-scalar structs
/// (`LightData light = scene_data.light_table[i];`) into a bare local
/// plus memberwise reads from the dimensioned cbuffer arrays (scalar
/// arrays unrolled with a linearized `FLAT[k*N+j]` index). Resource
/// bearing elements keep the alias path (loud if unresolvable); the
/// index duplicates per leaf, so impure indices stay loud (checked by
/// the alias recognizer).
#[allow(clippy::too_many_arguments)]
fn materialize_indexed_copies(
    list: &mut Vec<IrStmt>,
    root: &str,
    param_roots: &[String],
    params: &[crate::ir::function::Parameter],
    fields: &StructFields,
    structs: &[StructDecl],
    root_struct: &str,
    scalar_map: &HashMap<String, String>,
) {
    fn member_chain(base: IrExpr, rel: &[String]) -> IrExpr {
        let mut out = base;
        for seg in rel {
            out = IrExpr::Member { object: Box::new(out), member: seg.clone() };
        }
        out
    }
    let mut i = 0;
    while i < list.len() {
        let expansion = match &list[i] {
            IrStmt::VarDecl { ty, name, array_size: None, init: Some(init) } => {
                let alias = as_srt_alias(
                    ty, name, init, root, param_roots, params, fields, structs, root_struct,
                );
                let Some(alias) = alias else {
                    i += 1;
                    continue;
                };
                // Exact element type only (`auto` stays on the alias
                // path, whose uses transfer the same way).
                if !matches!(ty, IrType::Named(t) if t == &alias.elem) {
                    i += 1;
                    continue;
                }
                let Some(leaves) = scalar_copy_leaves(&alias.elem, fields, structs) else {
                    i += 1;
                    continue;
                };
                // Every leaf must map (partially-mapped structs decline:
                // the copy cannot complete).
                let mut pieces: Vec<(Vec<String>, Option<u32>, String)> = Vec::new();
                let mut mapped = true;
                for (rel, dim) in &leaves {
                    let key = format!("{}.{}", alias.depath, rel.join("."));
                    match scalar_map.get(&key) {
                        Some(flat) => pieces.push((rel.clone(), *dim, flat.clone())),
                        None => {
                            mapped = false;
                            break;
                        }
                    }
                }
                if !mapped {
                    i += 1;
                    continue;
                }
                let mut seq = vec![IrStmt::VarDecl {
                    ty: ty.clone(),
                    name: name.clone(),
                    array_size: None,
                    init: None,
                }];
                for (rel, dim, flat) in &pieces {
                    match dim {
                        None => seq.push(IrStmt::Expr(IrExpr::Assign {
                            lhs: Box::new(member_chain(
                                IrExpr::Variable(name.clone()),
                                rel,
                            )),
                            rhs: Box::new(IrExpr::Index {
                                object: Box::new(IrExpr::Variable(flat.clone())),
                                index: Box::new(alias.index.clone()),
                            }),
                        })),
                        Some(n) => {
                            for j in 0..*n {
                                let at = IrExpr::Binary {
                                    left: Box::new(IrExpr::Binary {
                                        left: Box::new(alias.index.clone()),
                                        op: IrOp::Mul,
                                        right: Box::new(IrExpr::Literal(IrLiteral::Number(
                                            n.to_string(),
                                        ))),
                                    }),
                                    op: IrOp::Add,
                                    right: Box::new(IrExpr::Literal(IrLiteral::Number(
                                        j.to_string(),
                                    ))),
                                };
                                seq.push(IrStmt::Expr(IrExpr::Assign {
                                    lhs: Box::new(IrExpr::Index {
                                        object: Box::new(member_chain(
                                            IrExpr::Variable(name.clone()),
                                            rel,
                                        )),
                                        index: Box::new(IrExpr::Literal(IrLiteral::Number(
                                            j.to_string(),
                                        ))),
                                    }),
                                    rhs: Box::new(IrExpr::Index {
                                        object: Box::new(IrExpr::Variable(flat.clone())),
                                        index: Box::new(at),
                                    }),
                                }));
                            }
                        }
                    }
                }
                Some(seq)
            }
            _ => None,
        };
        if let Some(seq) = expansion {
            let n = seq.len();
            list.splice(i..=i, seq);
            i += n;
            continue;
        }
        match &mut list[i] {
            IrStmt::Block(inner) => materialize_indexed_copies(
                inner, root, param_roots, params, fields, structs, root_struct, scalar_map,
            ),
            IrStmt::If { then_branch, else_branch, .. } => {
                materialize_indexed_copies(
                    then_branch, root, param_roots, params, fields, structs, root_struct, scalar_map,
                );
                if let Some(eb) = else_branch {
                    materialize_indexed_copies(
                        eb, root, param_roots, params, fields, structs, root_struct, scalar_map,
                    );
                }
            }
            IrStmt::For { body, .. } | IrStmt::While { body, .. } => materialize_indexed_copies(
                body, root, param_roots, params, fields, structs, root_struct, scalar_map,
            ),
            IrStmt::Switch { cases, .. } => {
                for c in cases {
                    materialize_indexed_copies(
                        &mut c.body, root, param_roots, params, fields, structs, root_struct,
                        scalar_map,
                    );
                }
            }
            _ => {}
        }
        i += 1;
    }
}
/// (`ClassifySrtBuffers buffers = srt_data.buffers[k];` + member uses
/// become `srt_data.buffers[k].member` for the index-transfer pre-pass).
///
/// Operates per statement list with suffix substitution: a declaration
/// at index `i` owns the list suffix (its scope), so same-name
/// declarations in sibling branches are independent aliases, and outer
/// same-name bindings (shadowed) are untouched. Each iteration
/// validates, removes, and substitutes exactly one alias.
fn resolve_srt_aliases(
    body: &mut Vec<IrStmt>,
    root: &str,
    param_roots: &[String],
    params: &[crate::ir::function::Parameter],
    fields: &StructFields,
    structs: &[StructDecl],
    root_struct: &str,
) {
    resolve_srt_alias_list(body, root, param_roots, params, fields, structs, root_struct);
}

/// Per-list driver: direct-child aliases to fixpoint, then nested lists.
#[allow(clippy::too_many_arguments)]
fn resolve_srt_alias_list(
    list: &mut Vec<IrStmt>,
    root: &str,
    param_roots: &[String],
    params: &[crate::ir::function::Parameter],
    fields: &StructFields,
    structs: &[StructDecl],
    root_struct: &str,
) {
    let mut skip: std::collections::HashSet<String> = std::collections::HashSet::new();
    loop {
        let mut found: Option<(usize, SrtAlias)> = None;
        for (i, s) in list.iter().enumerate() {
            if let IrStmt::VarDecl { ty, name, array_size: None, init: Some(init) } = s {
                if skip.contains(name) {
                    continue;
                }
                if let Some(alias) = as_srt_alias(
                    ty, name, init, root, param_roots, params, fields, structs, root_struct,
                ) {
                    found = Some((i, alias));
                    break;
                }
            }
        }
        let Some((i, alias)) = found else {
            break;
        };
        // The declaration owns the suffix: any same-name declaration or
        // violation there vetoes (fail-closed); the prefix belongs to
        // outer scopes (use-before-decl is invalid PSSL and stays loud).
        if alias_violated(&list[i + 1..], &alias.name, &alias.init) {
            skip.insert(alias.name);
            continue;
        }
        list.remove(i);
        subst_alias_in_stmts(&mut list[i..], &alias);
    }
    for s in list.iter_mut() {
        match s {
            IrStmt::Block(inner) => {
                resolve_srt_alias_list(inner, root, param_roots, params, fields, structs, root_struct)
            }
            IrStmt::If { then_branch, else_branch, .. } => {
                resolve_srt_alias_list(then_branch, root, param_roots, params, fields, structs, root_struct);
                if let Some(eb) = else_branch {
                    resolve_srt_alias_list(eb, root, param_roots, params, fields, structs, root_struct);
                }
            }
            IrStmt::For { body, .. } | IrStmt::While { body, .. } => {
                resolve_srt_alias_list(body, root, param_roots, params, fields, structs, root_struct);
            }
            IrStmt::Switch { cases, .. } => {
                for c in cases {
                    resolve_srt_alias_list(&mut c.body, root, param_roots, params, fields, structs, root_struct);
                }
            }
            _ => {}
        }
    }
}
/// Removes the `srt` argument from calls to helpers that had their
/// `DofBlurParams` param elided. Arity-gated: only calls whose argument
/// count matches the helper's pre-elision arity lose their SRT argument,
/// so overloads (`f(srt, x)` vs `f(srt, a, b, c, d)`) resolve correctly.
fn rewrite_calls_for_helpers(stmt: &mut IrStmt, helpers: &[(usize, String, usize, String, usize)]) {
    match stmt {
        IrStmt::Return(e) => {
            if let Some(e) = e {
                rewrite_calls_in_expr(e, helpers);
            }
        }
        IrStmt::VarDecl { init, .. } => {
            if let Some(init) = init {
                rewrite_calls_in_expr(init, helpers);
            }
        }
        IrStmt::If {
            condition,
            then_branch,
            else_branch,
        } => {
            rewrite_calls_in_expr(condition, helpers);
            for s in then_branch {
                rewrite_calls_for_helpers(s, helpers);
            }
            if let Some(eb) = else_branch {
                for s in eb {
                    rewrite_calls_for_helpers(s, helpers);
                }
            }
        }
        IrStmt::For {
            init,
            condition,
            update,
            body,
        } => {
            rewrite_calls_for_helpers(init, helpers);
            if let Some(cond) = condition {
                rewrite_calls_in_expr(cond, helpers);
            }
            if let Some(upd) = update {
                rewrite_calls_in_expr(upd, helpers);
            }
            for s in body {
                rewrite_calls_for_helpers(s, helpers);
            }
        }
        IrStmt::While { condition, body } => {
            rewrite_calls_in_expr(condition, helpers);
            for s in body {
                rewrite_calls_for_helpers(s, helpers);
            }
        }
        IrStmt::Switch { scrutinee, cases } => {
            rewrite_calls_in_expr(scrutinee, helpers);
            for case in cases {
                if let Some(label) = &mut case.label {
                    rewrite_calls_in_expr(label, helpers);
                }
                for s in &mut case.body {
                    rewrite_calls_for_helpers(s, helpers);
                }
            }
        }
        IrStmt::Block(stmts) => {
            for s in stmts {
                rewrite_calls_for_helpers(s, helpers);
            }
        }
        IrStmt::Expr(e) => rewrite_calls_in_expr(e, helpers),
    }
}

fn rewrite_calls_in_expr(expr: &mut IrExpr, helpers: &[(usize, String, usize, String, usize)]) {
    match expr {
        IrExpr::Call { function, arguments } => {
            // Check if this call targets a helper whose SRT param was elided.
            for (_, helper_name, idx, _, pre_arity) in helpers {
                if function == helper_name
                    && arguments.len() == *pre_arity
                    && *idx < arguments.len()
                {
                    // Heuristic: the SRT argument is a bare `srt` variable or
                    // any `IrExpr::Variable` that looks like the flattened root.
                    // Remove it so arity matches the new helper signature.
                    tracing::trace!(
                        target: "pssl_shader::lower::srt",
                        helper = %helper_name,
                        idx = %idx,
                        "eliding SRT argument from call"
                    );
                    arguments.remove(*idx);
                    break;
                }
            }
            for a in arguments {
                rewrite_calls_in_expr(a, helpers);
            }
        }
        IrExpr::Member { object, .. } => rewrite_calls_in_expr(object, helpers),
        IrExpr::Index { object, index } => {
            rewrite_calls_in_expr(object, helpers);
            rewrite_calls_in_expr(index, helpers);
        }
        IrExpr::ResourceIndex { resource, index } => {
            rewrite_calls_in_expr(resource, helpers);
            rewrite_calls_in_expr(index, helpers);
        }
        IrExpr::ResourceLoad {
            resource,
            coordinates,
            sample,
        } => {
            rewrite_calls_in_expr(resource, helpers);
            rewrite_calls_in_expr(coordinates, helpers);
            if let Some(s) = sample {
                rewrite_calls_in_expr(s, helpers);
            }
        }
        IrExpr::Construct { arguments, .. } => {
            for a in arguments {
                rewrite_calls_in_expr(a, helpers);
            }
        }
        IrExpr::Unary { expr: inner, .. } => rewrite_calls_in_expr(inner, helpers),
        IrExpr::Binary { left, right, .. } => {
            rewrite_calls_in_expr(left, helpers);
            rewrite_calls_in_expr(right, helpers);
        }
        IrExpr::Conditional {
            condition,
            then_branch,
            else_branch,
        } => {
            rewrite_calls_in_expr(condition, helpers);
            rewrite_calls_in_expr(then_branch, helpers);
            rewrite_calls_in_expr(else_branch, helpers);
        }
        IrExpr::Assign { lhs, rhs } => {
            rewrite_calls_in_expr(lhs, helpers);
            rewrite_calls_in_expr(rhs, helpers);
        }
        IrExpr::InitializerList(elems) => {
            for e in elems {
                rewrite_calls_in_expr(e, helpers);
            }
        }
        IrExpr::Variable(_) | IrExpr::Literal(_) => {}
    }
}

/// Context for SRT-subobject materialization (locals aliasing whole
/// `srt.*` subtrees, e.g. `EnvInfo& x = *srt.y`).
struct SubobjectCtx<'a> {
    root: &'a str,
    root_struct: &'a str,
    fields: &'a StructFields,
    param_roots: &'a [String],
    resource_map: &'a HashMap<String, String>,
    scalar_map: &'a HashMap<String, String>,
    res_tys: &'a [(String, IrType)],
    structs: &'a [StructDecl],
}

/// Array size of a struct field, if any (`None` for scalars).
fn field_array_size(structs: &[StructDecl], struct_name: &str, field: &str) -> Option<u32> {
    structs
        .iter()
        .find(|s| s.name == struct_name)
        .and_then(|s| s.fields.iter().find(|f| f.name == field))
        .and_then(|f| f.array_size)
}

/// Whether `root` (or an elided helper-param name) is shadowed inside
/// `function` by a parameter or local of the same name: uses then refer
/// to shadowing declaration, never the flattened global root (e.g.
/// `resolveDeferred(DeferredSrt srt, ...)` vs a global `Srt srt`).
/// Skipping is always safe (unprocessed uses keep today's loud path).
fn srt_root_shadowed(
    body: &[IrStmt],
    parameters: &[crate::ir::function::Parameter],
    root: &str,
    param_roots: &[String],
) -> bool {
    if parameters
        .iter()
        .any(|p| p.name == root || param_roots.iter().any(|r| r == &p.name))
    {
        return true;
    }
    fn has_local(stmts: &[IrStmt], root: &str) -> bool {
        stmts.iter().any(|s| match s {
            IrStmt::VarDecl { name, .. } => name == root,
            IrStmt::Block(inner) => has_local(inner, root),
            IrStmt::If {
                then_branch,
                else_branch,
                ..
            } => {
                has_local(then_branch, root)
                    || else_branch.as_ref().is_some_and(|b| has_local(b, root))
            }
            IrStmt::For { init, body, .. } => {
                has_local(std::slice::from_ref(init.as_ref()), root) || has_local(body, root)
            }
            IrStmt::While { body, .. } => has_local(body, root),
            IrStmt::Switch { cases, .. } => cases.iter().any(|c| has_local(&c.body, root)),
            _ => false,
        })
    }
    has_local(body, root)
}

/// Array size of the terminal field of a flattened scalar path
/// (`srt.a.b`), if it is a sized array. Segments resolve from the root
/// struct; paths through arrays or unknown segments yield nothing (those
/// keep today's scalar shape).
/// Array dimension for a leaf path plus the segment index (root = 0)
/// of the transferable intermediate struct-array field, if any.
///
/// The dimension is the product of every sized array segment along the
/// path (leaf arrays as before; intermediate struct arrays contribute
/// their size; a 2D `root.arr[k].leaf[j]` leaf sizes its binding with
/// the product for whole-element copies). Any unsized segment yields no
/// dimension (loud).
///
/// `transfer_pos` is set only when the intermediate array is the SOLE
/// array segment: uses then transfer the index inward
/// (`root.arr[k].leaf` -> `BINDING[k]`); it also verifies placement (an
/// index anywhere else selects a different object and stays loud).
/// Leaf-array paths (`srt.x[j]` -> `BINDING[j]`) and 2D paths keep no
/// transfer position (loud unless copied whole).
fn subobject_array_size(
    fields: &StructFields,
    structs: &[StructDecl],
    root_struct: &str,
    path: &str,
    root: &str,
) -> (Option<u32>, Option<usize>) {
    let mut segs = path.split('.');
    if segs.next().is_none_or(|r| r != root) {
        return (None, None);
    }
    let segs: Vec<&str> = segs.collect();
    if segs.is_empty() {
        return (None, None);
    }
    let mut product: Option<u32> = None;
    let mut first_at: Option<usize> = None;
    let mut count = 0u32;
    let mut cur = root_struct.to_string();
    for (i, seg) in segs.iter().enumerate() {
        let field = fields
            .get(&cur)
            .and_then(|fs| fs.iter().find(|(n, _)| n == seg))
            .map(|(_, t)| t.clone());
        let Some(field) = field else {
            return (None, None);
        };
        if let Some(n) = field_array_size(structs, &cur, seg).filter(|&n| n > 0) {
            if i + 1 < segs.len() && first_at.is_none() {
                // Segment index counts the root (segs[0] is the first
                // field, so this field sits at i + 1).
                first_at = Some(i + 1);
            }
            count += 1;
            product = Some(product.map_or(n, |p| p.saturating_mul(n)));
        } else if field_array_size(structs, &cur, seg).is_some() {
            return (None, None);
        }
        if i + 1 == segs.len() {
            break;
        }
        let IrType::Named(s) = field else {
            return (None, None);
        };
        if !fields.contains_key(&s) {
            return (None, None);
        }
        cur = s;
    }
    // Transfer only for exactly one array segment that is intermediate
    // (not the leaf itself).
    let last_is_array = field_array_size(structs, &cur, segs[segs.len() - 1])
        .is_some_and(|n| n > 0);
    let at = match (count, last_is_array, first_at) {
        (1, false, Some(at)) => Some(at),
        _ => None,
    };
    (product, at)
}

/// Materializes locals aliasing whole SRT subobjects in every statement
/// list (spliced in place: wrapping them in a block would scope the
/// variable away from its later uses).
fn materialize_in_stmts(stmts: &mut Vec<IrStmt>, ctx: &SubobjectCtx) {
    let mut i = 0;
    while i < stmts.len() {
        match expand_subobject_decl(&stmts[i], ctx) {
            Some(expanded) => {
                let n = expanded.len();
                stmts.splice(i..=i, expanded);
                i += n;
            }
            None => {
                materialize_in_stmt(&mut stmts[i], ctx);
                i += 1;
            }
        }
    }
}

fn materialize_in_stmt(stmt: &mut IrStmt, ctx: &SubobjectCtx) {
    match stmt {
        IrStmt::Block(inner) => materialize_in_stmts(inner, ctx),
        IrStmt::If {
            then_branch,
            else_branch,
            ..
        } => {
            materialize_in_stmts(then_branch, ctx);
            if let Some(else_branch) = else_branch {
                materialize_in_stmts(else_branch, ctx);
            }
        }
        // A `for` init is a lone statement (no list to splice into): a
        // subobject declaration there keeps today's loud path.
        IrStmt::For { body, .. } => materialize_in_stmts(body, ctx),
        IrStmt::While { body, .. } => materialize_in_stmts(body, ctx),
        IrStmt::Switch { cases, .. } => {
            for case in cases {
                materialize_in_stmts(&mut case.body, ctx);
            }
        }
        _ => {}
    }
}

/// Synthesizes materialized temporaries for whole-subobject SRT paths in
/// value positions (`f(*srt.y)`, `return srt.y`): each becomes a
/// same-scope local populated from the flattened pieces, and the use
/// becomes the temporary. The temporary carries exactly the path's struct
/// type, so overload resolution is unaffected. Paths in writable
/// positions (assignment targets) keep today's loud path: scattering back
/// into flattened pieces is not supported.
/// Highest `__srtarg_N` suffix already present in the module, if any.
/// AST-level SRT-arg materialization numbers its own temporaries from 0
/// and splice temps share the `__srtarg_` prefix, so the splice counter
/// must start past the max to avoid declaring the same name twice in one
/// function (DXC `redefinition`).
fn max_srtarg_suffix(ir: &ShaderModule) -> Option<u32> {
    fn scan_stmts(stmts: &[IrStmt], max: &mut Option<u32>) {
        for s in stmts {
            match s {
                IrStmt::VarDecl { name, .. } => {
                    if let Some(rest) = name.strip_prefix("__srtarg_") {
                        if let Ok(n) = rest.parse::<u32>() {
                            *max = Some(max.map_or(n, |m: u32| m.max(n)));
                        }
                    }
                }
                IrStmt::Block(inner) => scan_stmts(inner, max),
                IrStmt::If { then_branch, else_branch, .. } => {
                    scan_stmts(then_branch, max);
                    if let Some(eb) = else_branch {
                        scan_stmts(eb, max);
                    }
                }
                IrStmt::For { init, body, .. } => {
                    scan_stmts(std::slice::from_ref(init), max);
                    scan_stmts(body, max);
                }
                IrStmt::While { body, .. } => scan_stmts(body, max),
                IrStmt::Switch { cases, .. } => {
                    for c in cases {
                        scan_stmts(&c.body, max);
                    }
                }
                IrStmt::Return(_) | IrStmt::Expr(_) => {}
            }
        }
    }
    let mut max: Option<u32> = None;
    for f in &ir.functions {
        for p in &f.parameters {
            if let Some(rest) = p.name.strip_prefix("__srtarg_") {
                if let Ok(n) = rest.parse::<u32>() {
                    max = Some(max.map_or(n, |m: u32| m.max(n)));
                }
            }
        }
        scan_stmts(&f.body, &mut max);
    }
    max
}

fn splice_subobject_temps(stmts: &mut Vec<IrStmt>, ctx: &SubobjectCtx, counter: &mut u32) {
    let mut i = 0;
    while i < stmts.len() {
        let mut preamble: Vec<IrStmt> = Vec::new();
        splice_temps_in_stmt(&mut stmts[i], ctx, counter, &mut preamble);
        if preamble.is_empty() {
            i += 1;
        } else {
            let n = preamble.len();
            stmts.splice(i..i, preamble);
            i += n + 1;
        }
    }
}

/// Block-top splice for LEAF statements only (control flow never reaches
/// here — the caller routes it to the list-local path): evaluates into
/// the outer preamble so temps hoist before the block.
fn splice_block_top(
    stmt: &mut IrStmt,
    ctx: &SubobjectCtx,
    counter: &mut u32,
    preamble: &mut Vec<IrStmt>,
) {
    match stmt {
        IrStmt::Return(e) => {
            if let Some(e) = e {
                splice_temps_in_expr(e, ctx, counter, preamble, false);
            }
        }
        IrStmt::VarDecl { init, .. } => {
            if let Some(init) = init {
                splice_temps_in_expr(init, ctx, counter, preamble, false);
            }
        }
        IrStmt::Expr(e) => splice_temps_in_expr(e, ctx, counter, preamble, false),
        _ => unreachable!("splice_block_top routes only leaf statements"),
    }
}

fn splice_temps_in_stmt(
    stmt: &mut IrStmt,
    ctx: &SubobjectCtx,
    counter: &mut u32,
    preamble: &mut Vec<IrStmt>,
) {
    match stmt {
        IrStmt::Block(inner) => {
            // Thread the OUTER preamble through the block's top-level
            // leaf statements instead of splicing fresh ones inside:
            // block contents are an unconditional sequence, and
            // materialized temps are pure reads of flattened bindings,
            // so hoisting them before the block preserves order and
            // semantics while keeping desugared-destructure outputs
            // (`org`/`dir`) outside any new scope. (Splicing inside
            // would trap them: the emitter braces blocks holding
            // assignments, hiding the outputs from later uses.)
            // Nested control flow keeps list-local splicing via the
            // normal path.
            let mut i = 0;
            while i < inner.len() {
                let is_leaf = matches!(
                    inner[i],
                    IrStmt::Return(_) | IrStmt::VarDecl { .. } | IrStmt::Expr(_)
                );
                if is_leaf {
                    splice_block_top(&mut inner[i], ctx, counter, preamble);
                    i += 1;
                } else {
                    let mut local: Vec<IrStmt> = Vec::new();
                    splice_temps_in_stmt(&mut inner[i], ctx, counter, &mut local);
                    if local.is_empty() {
                        i += 1;
                    } else {
                        let n = local.len();
                        inner.splice(i..i, local);
                        i += n + 1;
                    }
                }
            }
        }
        IrStmt::If {
            condition,
            then_branch,
            else_branch,
        } => {
            splice_temps_in_expr(condition, ctx, counter, preamble, false);
            splice_subobject_temps(then_branch, ctx, counter);
            if let Some(else_branch) = else_branch {
                splice_subobject_temps(else_branch, ctx, counter);
            }
        }
        IrStmt::For {
            init,
            condition,
            update,
            body,
        } => {
            // Lone init: no list to splice into; still rewrite inside it.
            splice_temps_single(init, ctx, counter);
            if let Some(condition) = condition {
                splice_temps_in_expr(condition, ctx, counter, preamble, false);
            }
            if let Some(update) = update {
                splice_temps_in_expr(update, ctx, counter, preamble, false);
            }
            splice_subobject_temps(body, ctx, counter);
        }
        IrStmt::While { condition, body } => {
            splice_temps_in_expr(condition, ctx, counter, preamble, false);
            splice_subobject_temps(body, ctx, counter);
        }
        IrStmt::Switch { scrutinee, cases } => {
            splice_temps_in_expr(scrutinee, ctx, counter, preamble, false);
            for case in cases {
                if let Some(label) = &mut case.label {
                    splice_temps_in_expr(label, ctx, counter, preamble, false);
                }
                splice_subobject_temps(&mut case.body, ctx, counter);
            }
        }
        IrStmt::Return(e) => {
            if let Some(e) = e {
                splice_temps_in_expr(e, ctx, counter, preamble, false);
            }
        }
        IrStmt::VarDecl { init, .. } => {
            // Declarations are materialized in place (tighter codegen);
            // still descend (e.g. nested calls in the initializer).
            if let Some(init) = init {
                splice_temps_in_expr(init, ctx, counter, preamble, false);
            }
        }
        IrStmt::Expr(e) => splice_temps_in_expr(e, ctx, counter, preamble, false),
    }
}

/// Variant for lone statements (no parent list): rewrites in place,
/// never splicing. Probes on a clone so an unplaceable expansion leaves
/// the original loud instead of dangling a temporary.
fn splice_temps_single(stmt: &mut IrStmt, ctx: &SubobjectCtx, counter: &mut u32) {
    let mut probe = stmt.clone();
    let mut preamble: Vec<IrStmt> = Vec::new();
    splice_temps_in_stmt(&mut probe, ctx, counter, &mut preamble);
    if preamble.is_empty() {
        *stmt = probe;
    }
}

/// Walks an expression, synthesizing materialized temporaries for
/// whole-subobject SRT paths in readable positions. `writable`
/// disables expansion (assignment targets scatter nowhere); nested
/// index expressions always stay readable.
fn splice_temps_in_expr(
    expr: &mut IrExpr,
    ctx: &SubobjectCtx,
    counter: &mut u32,
    preamble: &mut Vec<IrStmt>,
    writable: bool,
) {
    if !writable {
        if let Some(path) = path_of(expr) {
            let segs: Vec<&str> = path.split('.').collect();
            if !segs.is_empty()
                && (segs[0] == ctx.root || ctx.param_roots.iter().any(|p| p == segs[0]))
                && let Some(struct_name) = resolve_subobject_struct(&segs, ctx)
                && let Some(pieces) =
                    materialize_pieces(&format!("__srtarg_{counter}"), &struct_name, &path, ctx)
            {
                *counter += 1;
                let temp = match &pieces[0] {
                    IrStmt::VarDecl { name, .. } => name.clone(),
                    _ => unreachable!("materialize_pieces starts with a declaration"),
                };
                preamble.extend(pieces);
                *expr = IrExpr::Variable(temp);
                return;
            }
        }
    }
    match expr {
        IrExpr::Call { function, arguments } => {
            // Method calls on flattened subobjects (`srt.m_camera.generate`
            // after helper-param elision): `function` is a dotted string,
            // so the whole-expression `path_of` above never fires and the
            // receiver would dangle. When all-but-last segments form an
            // srt-rooted path resolving to a struct, materialize it and
            // retarget the receiver (`__srtarg_N.generate`). Resource/scalar
            // leaf prefixes resolve to no struct and stay loud for the
            // dedicated resource-method paths; partially-mapped structs
            // decline via `materialize_pieces`.
            if let Some(dot) = function.rfind('.') {
                let (prefix, method) = (&function[..dot], &function[dot + 1..]);
                let segs: Vec<&str> = prefix.split('.').collect();
                // Strictly a SUBOBJECT path (2+ segments): the bare root
                // itself (`p.getId()` on an elided helper param) belongs
                // to the dangling-root method inliner, which substitutes
                // the getter body — materializing the root here would
                // fabricate a value for a parameter that has none.
                if segs.len() >= 2
                    && !method.is_empty()
                    && (segs[0] == ctx.root || ctx.param_roots.iter().any(|p| p == segs[0]))
                    && let Some(struct_name) = resolve_subobject_struct(&segs, ctx)
                    && let Some(pieces) = materialize_pieces(
                        &format!("__srtarg_{counter}"),
                        &struct_name,
                        prefix,
                        ctx,
                    )
                {
                    *counter += 1;
                    let temp = match &pieces[0] {
                        IrStmt::VarDecl { name, .. } => name.clone(),
                        _ => unreachable!("materialize_pieces starts with a declaration"),
                    };
                    preamble.extend(pieces);
                    *function = format!("{temp}.{method}");
                }
            }
            for a in arguments {
                splice_temps_in_expr(a, ctx, counter, preamble, false);
            }
        }
        IrExpr::Construct { arguments, .. } => {
            for a in arguments {
                splice_temps_in_expr(a, ctx, counter, preamble, false);
            }
        }
        IrExpr::Member { object, .. } => {
            // Never descend into an srt-rooted object: the full path was
            // already tried above, and expanding a prefix would
            // materialize a foreign temporary (`srt.w` must rewrite to
            // the cbuffer, not `__t.w`; unresolvable longer paths must
            // stay loud). Non-rooted objects recurse normally.
            let rooted = path_of(object).is_some_and(|p| {
                let seg0 = p.split('.').next().unwrap_or("");
                seg0 == ctx.root || ctx.param_roots.iter().any(|r| r == seg0)
            });
            if !rooted {
                splice_temps_in_expr(object, ctx, counter, preamble, false);
            }
        }
        IrExpr::Index { object, index } | IrExpr::ResourceIndex { resource: object, index } => {
            // An srt-rooted struct object under an index belongs to the
            // indexed-buffer rule, not temporaries: expanding it here
            // would index into a scalar temporary (invalid). Leaves and
            // unresolvable paths recurse normally.
            let struct_object = path_of(object).is_some_and(|p| {
                let segs: Vec<&str> = p.split('.').collect();
                !segs.is_empty()
                    && (segs[0] == ctx.root || ctx.param_roots.iter().any(|r| r == segs[0]))
                    && resolve_subobject_struct(&segs, ctx).is_some()
            });
            if !struct_object {
                splice_temps_in_expr(object, ctx, counter, preamble, writable);
            }
            splice_temps_in_expr(index, ctx, counter, preamble, false);
        }
        IrExpr::ResourceLoad {
            resource,
            coordinates,
            sample,
        } => {
            splice_temps_in_expr(resource, ctx, counter, preamble, writable);
            splice_temps_in_expr(coordinates, ctx, counter, preamble, false);
            if let Some(sample) = sample {
                splice_temps_in_expr(sample, ctx, counter, preamble, false);
            }
        }
        IrExpr::Unary { expr: inner, .. } => {
            splice_temps_in_expr(inner, ctx, counter, preamble, writable)
        }
        IrExpr::Binary { left, right, .. } => {
            splice_temps_in_expr(left, ctx, counter, preamble, writable);
            splice_temps_in_expr(right, ctx, counter, preamble, writable);
        }
        IrExpr::Conditional {
            condition,
            then_branch,
            else_branch,
        } => {
            splice_temps_in_expr(condition, ctx, counter, preamble, false);
            splice_temps_in_expr(then_branch, ctx, counter, preamble, writable);
            splice_temps_in_expr(else_branch, ctx, counter, preamble, writable);
        }
        IrExpr::Assign { lhs, rhs } => {
            splice_temps_in_expr(lhs, ctx, counter, preamble, true);
            splice_temps_in_expr(rhs, ctx, counter, preamble, false);
        }
        IrExpr::InitializerList(elems) => {
            for e in elems {
                splice_temps_in_expr(e, ctx, counter, preamble, writable);
            }
        }
        _ => {}
    }
}

/// Expands `T x = srt.path;` (or `*srt.path`; `&`/`*` desugar away) where
/// `path` is rooted at the flattened root (or an elided helper param of
/// the root's struct):
/// - leaf in a flattened map → rebind the initializer to the binding
///   (`auto x = srt.k` → `auto x = k_cbuf;`). Resource leaves only for
///   `auto` declarations (the binding carries the type); other resource
///   leaves keep today's loud path (opaque-type copies are not valid
///   HLSL everywhere).
/// - struct-typed path with every leaf mapped → bare local plus
///   memberwise assignments from the flattened pieces, so every later
///   use (member access or whole value) resolves.
/// `None` for any other statement.
fn expand_subobject_decl(stmt: &IrStmt, ctx: &SubobjectCtx) -> Option<Vec<IrStmt>> {
    let IrStmt::VarDecl {
        ty,
        name,
        array_size: None,
        init: Some(init),
    } = stmt
    else {
        return None;
    };
    let path = path_of(init)?;
    let segs: Vec<&str> = path.split('.').collect();
    if segs.is_empty() || (segs[0] != ctx.root && !ctx.param_roots.iter().any(|p| p == segs[0])) {
        return None;
    }
    // Leaf rebinding.
    if let Some(binding) = ctx.resource_map.get(&path) {
        if matches!(ty, IrType::Named(n) if n == "auto") {
            let (_, bty) = ctx.res_tys.iter().find(|(p, _)| p == &path)?;
            return Some(vec![IrStmt::VarDecl {
                ty: bty.clone(),
                name: name.clone(),
                array_size: None,
                init: Some(IrExpr::Variable(binding.clone())),
            }]);
        }
        return None;
    }
    if let Some(field) = ctx.scalar_map.get(&path) {
        return Some(vec![IrStmt::VarDecl {
            ty: ty.clone(),
            name: name.clone(),
            array_size: None,
            init: Some(IrExpr::Variable(field.clone())),
        }]);
    }
    // Struct-typed path: resolve through the layout from the root struct.
    let cur = resolve_subobject_struct(&segs, ctx)?;
    if !matches!(ty, IrType::Named(n) if n == "auto" || n == &cur) {
        return None;
    }
    // Every leaf must map; otherwise the local cannot be completed.
    materialize_pieces(name, &cur, &path, ctx)
}

/// Resolves an srt-rooted path to its struct type (`None` for leaves,
/// arrays, or unknown segments).
fn resolve_subobject_struct(segs: &[&str], ctx: &SubobjectCtx) -> Option<String> {
    let mut cur = ctx.root_struct.to_string();
    for seg in &segs[1..] {
        if field_array_size(ctx.structs, &cur, seg).is_some() {
            return None;
        }
        let next = ctx
            .fields
            .get(&cur)?
            .iter()
            .find(|(n, _)| n == seg)
            .map(|(_, t)| t.clone())?;
        let IrType::Named(s) = next else {
            return None;
        };
        if !ctx.fields.contains_key(&s) {
            return None;
        }
        cur = s;
    }
    Some(cur)
}

/// A step in a materialized-local access path: struct field or array
/// element.
#[derive(Clone)]
enum SubSeg {
    Field(String),
    Index(u32),
}

/// Builds `[bare T name; name.f = <flattened>; ...]` for the struct at
/// `path`, or `None` when any leaf is unmapped.
fn materialize_pieces(
    name: &str,
    struct_name: &str,
    path: &str,
    ctx: &SubobjectCtx,
) -> Option<Vec<IrStmt>> {
    let mut leaves: Vec<(Vec<SubSeg>, IrExpr)> = Vec::new();
    let mut rel: Vec<SubSeg> = Vec::new();
    if !collect_subobject_leaves(struct_name, path, &mut rel, ctx, &mut leaves) {
        return None;
    }
    let mut expanded = vec![IrStmt::VarDecl {
        ty: IrType::Named(struct_name.to_string()),
        name: name.to_string(),
        array_size: None,
        init: None,
    }];
    for (rel, value) in leaves {
        let mut target = IrExpr::Variable(name.to_string());
        for seg in rel {
            target = match seg {
                SubSeg::Field(member) => IrExpr::Member {
                    object: Box::new(target),
                    member,
                },
                SubSeg::Index(index) => IrExpr::Index {
                    object: Box::new(target),
                    index: Box::new(IrExpr::Literal(IrLiteral::Number(index.to_string()))),
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

/// Collects `(relative segments, flattened value)` for every leaf under
/// `struct_name`, whose full dotted path is `abs_prefix` (e.g.
/// `srt.envInfo`). Relative segments address the materialized local.
/// Sized scalar arrays unroll per element (their cbuffer bindings keep
/// array shape); resource arrays, unsized arrays, and unmapped leaves
/// return false (unsupported shapes stay loud).
fn collect_subobject_leaves(
    struct_name: &str,
    abs_prefix: &str,
    rel: &mut Vec<SubSeg>,
    ctx: &SubobjectCtx,
    out: &mut Vec<(Vec<SubSeg>, IrExpr)>,
) -> bool {
    let Some(field_list) = ctx.fields.get(struct_name) else {
        return false;
    };
    for (fname, fty) in field_list {
        let full = format!("{abs_prefix}.{fname}");
        let array_size = field_array_size(ctx.structs, struct_name, fname);
        match fty {
            IrType::Named(s) if ctx.fields.contains_key(s) => {
                // Arrays of structs cannot materialize elementwise here.
                if array_size.is_some() {
                    return false;
                }
                rel.push(SubSeg::Field(fname.clone()));
                if !collect_subobject_leaves(s, &full, rel, ctx, out) {
                    return false;
                }
                rel.pop();
            }
            _ => {
                match array_size.filter(|&n| n > 0) {
                    Some(n) => {
                        // Array leaf: only scalar-mapped arrays unroll
                        // (their binding is array-shaped); resource arrays
                        // stay loud.
                        if ctx.resource_map.contains_key(&full) {
                            return false;
                        }
                        let binding = match ctx.scalar_map.get(&full) {
                            Some(b) => b.clone(),
                            None => return false,
                        };
                        rel.push(SubSeg::Field(fname.clone()));
                        for i in 0..n {
                            let mut erel = rel.clone();
                            erel.push(SubSeg::Index(i));
                            out.push((
                                erel,
                                IrExpr::Index {
                                    object: Box::new(IrExpr::Variable(binding.clone())),
                                    index: Box::new(IrExpr::Literal(IrLiteral::Number(
                                        i.to_string(),
                                    ))),
                                },
                            ));
                        }
                        rel.pop();
                    }
                    None if array_size.is_some() => return false,
                    None => {
                        let value = if let Some(b) = ctx.resource_map.get(&full) {
                            IrExpr::Variable(b.clone())
                        } else if let Some(f) = ctx.scalar_map.get(&full) {
                            IrExpr::Variable(f.clone())
                        } else {
                            return false;
                        };
                        rel.push(SubSeg::Field(fname.clone()));
                        out.push((rel.clone(), value));
                        rel.pop();
                    }
                }
            }
        }
    }
    true
}

/// Whether the struct transitively contains a resource (or sampler)
/// member: such structs have no legal buffer binding.
fn struct_contains_resource(struct_name: &str, fields: &StructFields, depth: usize) -> bool {
    if depth > 8 {
        return true;
    }
    let Some(field_list) = fields.get(struct_name) else {
        return false;
    };
    field_list.iter().any(|(_, fty)| match fty {
        IrType::Resource { .. } => true,
        IrType::Named(n) if crate::framework::is_sampler_type_name(n) => true,
        IrType::Named(n) if fields.contains_key(n) => {
            struct_contains_resource(n, fields, depth + 1)
        }
        _ => false,
    })
}

/// Collects `(path, struct)` for srt-rooted struct member paths used
/// through an index (`srt.buf[i]...`).
fn collect_indexed_subobjects(
    stmts: &[IrStmt],
    root: &str,
    root_struct: &str,
    fields: &StructFields,
    param_roots: &[String],
    structs: &[StructDecl],
    out: &mut std::collections::BTreeSet<(String, String)>,
) {
    fn expr(
        e: &IrExpr,
        root: &str,
        root_struct: &str,
        fields: &StructFields,
        param_roots: &[String],
        structs: &[StructDecl],
        out: &mut std::collections::BTreeSet<(String, String)>,
    ) {
        if let IrExpr::Index { object, index } = e {
            if let Some(path) = path_of(object) {
                let segs: Vec<&str> = path.split('.').collect();
                if !segs.is_empty()
                    && (segs[0] == root || param_roots.iter().any(|p| p == segs[0]))
                    && let Some(struct_name) = resolve_indexed_struct(&segs, root_struct, fields, structs)
                {
                    out.insert((path, struct_name));
                }
            }
            expr(index, root, root_struct, fields, param_roots, structs, out);
            expr(object, root, root_struct, fields, param_roots, structs, out);
            return;
        }
        match e {
            IrExpr::Call { arguments, .. } | IrExpr::Construct { arguments, .. } => {
                for a in arguments {
                    expr(a, root, root_struct, fields, param_roots, structs, out);
                }
            }
            IrExpr::Member { object, .. } => {
                expr(object, root, root_struct, fields, param_roots, structs, out)
            }
            IrExpr::ResourceIndex { resource, index } => {
                expr(resource, root, root_struct, fields, param_roots, structs, out);
                expr(index, root, root_struct, fields, param_roots, structs, out);
            }
            IrExpr::ResourceLoad {
                resource,
                coordinates,
                sample,
            } => {
                expr(resource, root, root_struct, fields, param_roots, structs, out);
                expr(coordinates, root, root_struct, fields, param_roots, structs, out);
                if let Some(sample) = sample {
                    expr(sample, root, root_struct, fields, param_roots, structs, out);
                }
            }
            IrExpr::Unary { expr: inner, .. } => {
                expr(inner, root, root_struct, fields, param_roots, structs, out)
            }
            IrExpr::Binary { left, right, .. } => {
                expr(left, root, root_struct, fields, param_roots, structs, out);
                expr(right, root, root_struct, fields, param_roots, structs, out);
            }
            IrExpr::Conditional {
                condition,
                then_branch,
                else_branch,
            } => {
                expr(condition, root, root_struct, fields, param_roots, structs, out);
                expr(then_branch, root, root_struct, fields, param_roots, structs, out);
                expr(else_branch, root, root_struct, fields, param_roots, structs, out);
            }
            IrExpr::Assign { lhs, rhs } => {
                expr(lhs, root, root_struct, fields, param_roots, structs, out);
                expr(rhs, root, root_struct, fields, param_roots, structs, out);
            }
            IrExpr::InitializerList(elems) => {
                for e in elems {
                    expr(e, root, root_struct, fields, param_roots, structs, out);
                }
            }
            _ => {}
        }
    }
    fn stmt(
        s: &IrStmt,
        root: &str,
        root_struct: &str,
        fields: &StructFields,
        param_roots: &[String],
        structs: &[StructDecl],
        out: &mut std::collections::BTreeSet<(String, String)>,
    ) {
        match s {
            IrStmt::Return(e) => {
                if let Some(e) = e {
                    expr(e, root, root_struct, fields, param_roots, structs, out);
                }
            }
            IrStmt::VarDecl { init, .. } => {
                if let Some(init) = init {
                    expr(init, root, root_struct, fields, param_roots, structs, out);
                }
            }
            IrStmt::If {
                condition,
                then_branch,
                else_branch,
            } => {
                expr(condition, root, root_struct, fields, param_roots, structs, out);
                collect_indexed_subobjects(
                    then_branch,
                    root,
                    root_struct,
                    fields,
                    param_roots,
                    structs,
                    out,
                );
                if let Some(else_branch) = else_branch {
                    collect_indexed_subobjects(
                        else_branch,
                        root,
                        root_struct,
                        fields,
                        param_roots,
                        structs,
                        out,
                    );
                }
            }
            IrStmt::For {
                init,
                condition,
                update,
                body,
            } => {
                stmt(init, root, root_struct, fields, param_roots, structs, out);
                if let Some(condition) = condition {
                    expr(condition, root, root_struct, fields, param_roots, structs, out);
                }
                if let Some(update) = update {
                    expr(update, root, root_struct, fields, param_roots, structs, out);
                }
                collect_indexed_subobjects(body, root, root_struct, fields, param_roots, structs, out);
            }
            IrStmt::While { condition, body } => {
                expr(condition, root, root_struct, fields, param_roots, structs, out);
                collect_indexed_subobjects(body, root, root_struct, fields, param_roots, structs, out);
            }
            IrStmt::Switch { scrutinee, cases } => {
                expr(scrutinee, root, root_struct, fields, param_roots, structs, out);
                for case in cases {
                    if let Some(label) = &case.label {
                        expr(label, root, root_struct, fields, param_roots, structs, out);
                    }
                    collect_indexed_subobjects(
                        &case.body,
                        root,
                        root_struct,
                        fields,
                        param_roots,
                        structs,
                        out,
                    );
                }
            }
            IrStmt::Block(inner) => {
                collect_indexed_subobjects(inner, root, root_struct, fields, param_roots, structs, out)
            }
            IrStmt::Expr(e) => expr(e, root, root_struct, fields, param_roots, structs, out),
        }
    }
    for s in stmts {
        stmt(s, root, root_struct, fields, param_roots, structs, out);
    }
}

/// Resolves an indexed base path to its struct type through the layout.
/// Unlike the materialize path, the terminal segment may itself carry an
/// array shape in the source (pointers desugar away); arrays on the path
/// still disqualify.
fn resolve_indexed_struct(
    segs: &[&str],
    root_struct: &str,
    fields: &StructFields,
    structs: &[StructDecl],
) -> Option<String> {
    // The terminal segment may itself carry an array shape in the source
    // (an indexed use implies array semantics; pointers desugar away).
    // Intermediate arrays still disqualify (no static layout below them).
    let mut cur = root_struct.to_string();
    let tail = segs.len().saturating_sub(1);
    for (i, seg) in segs[1..].iter().enumerate() {
        if i + 1 < tail && field_array_size(structs, &cur, seg).is_some() {
            return None;
        }
        let next = fields
            .get(&cur)?
            .iter()
            .find(|(n, _)| n == seg)
            .map(|(_, t)| t.clone())?;
        let IrType::Named(s) = next else {
            return None;
        };
        if !fields.contains_key(&s) {
            return None;
        }
        cur = s;
    }
    Some(cur)
}

/// Rewrites indexed uses of hoisted subobject buffers
/// (`srt.buf[i]` → `srt_buf[i]`). Writable positions keep today's loud
/// path (the binding is read-only).
fn rewrite_indexed_uses(stmts: &mut Vec<IrStmt>, index_map: &HashMap<String, String>) {
    fn expr(e: &mut IrExpr, index_map: &HashMap<String, String>, writable: bool) {
        match e {
            IrExpr::Index { object, index } => {
                if !writable {
                    if let Some(path) = path_of(object) {
                        if let Some(binding) = index_map.get(&path) {
                            *object = Box::new(IrExpr::Variable(binding.clone()));
                        }
                    }
                }
                expr(object, index_map, writable);
                expr(index, index_map, false);
            }
            IrExpr::Call { arguments, .. } | IrExpr::Construct { arguments, .. } => {
                for a in arguments {
                    expr(a, index_map, false);
                }
            }
            IrExpr::Member { object, .. } => expr(object, index_map, false),
            IrExpr::ResourceIndex { resource, index } => {
                expr(resource, index_map, writable);
                expr(index, index_map, false);
            }
            IrExpr::ResourceLoad {
                resource,
                coordinates,
                sample,
            } => {
                expr(resource, index_map, writable);
                expr(coordinates, index_map, false);
                if let Some(sample) = sample {
                    expr(sample, index_map, false);
                }
            }
            IrExpr::Unary { expr: inner, .. } => expr(inner, index_map, writable),
            IrExpr::Binary { left, right, .. } => {
                expr(left, index_map, writable);
                expr(right, index_map, writable);
            }
            IrExpr::Assign { lhs, rhs } => {
                expr(lhs, index_map, true);
                expr(rhs, index_map, false);
            }
            IrExpr::Conditional {
                condition,
                then_branch,
                else_branch,
            } => {
                expr(condition, index_map, false);
                expr(then_branch, index_map, writable);
                expr(else_branch, index_map, writable);
            }
            IrExpr::InitializerList(elems) => {
                for e in elems {
                    expr(e, index_map, writable);
                }
            }
            _ => {}
        }
    }
    fn stmt(s: &mut IrStmt, index_map: &HashMap<String, String>) {
        match s {
            IrStmt::Return(e) => {
                if let Some(e) = e {
                    expr(e, index_map, false);
                }
            }
            IrStmt::VarDecl { init, .. } => {
                if let Some(init) = init {
                    expr(init, index_map, false);
                }
            }
            IrStmt::If {
                condition,
                then_branch,
                else_branch,
            } => {
                expr(condition, index_map, false);
                rewrite_indexed_uses(then_branch, index_map);
                if let Some(else_branch) = else_branch {
                    rewrite_indexed_uses(else_branch, index_map);
                }
            }
            IrStmt::For {
                init,
                condition,
                update,
                body,
            } => {
                stmt(init, index_map);
                if let Some(condition) = condition {
                    expr(condition, index_map, false);
                }
                if let Some(update) = update {
                    expr(update, index_map, false);
                }
                rewrite_indexed_uses(body, index_map);
            }
            IrStmt::While { condition, body } => {
                expr(condition, index_map, false);
                rewrite_indexed_uses(body, index_map);
            }
            IrStmt::Switch { scrutinee, cases } => {
                expr(scrutinee, index_map, false);
                for case in cases {
                    if let Some(label) = &mut case.label {
                        expr(label, index_map, false);
                    }
                    rewrite_indexed_uses(&mut case.body, index_map);
                }
            }
            IrStmt::Block(inner) => rewrite_indexed_uses(inner, index_map),
            IrStmt::Expr(e) => expr(e, index_map, false),
        }
    }
    for s in stmts {
        stmt(s, index_map);
    }
}

/// Drops provably-dead locals: bare-declared variables that are never
/// read, only memberwise-assigned from call-free expressions.
///
/// Subobject materialization above builds whole-struct locals
/// (`ViewGlobalsSRT globals; globals.m_x = …; …`,
/// `ForwardRenderSRT __srtarg_0; …`) for every elided SRT param, even
/// when the body never reads them (lighting moved into callees, method
/// calls later inlined). The residue is dozens of dead stores per
/// function — and after buffer splitting, stores of twin-typed buffers
/// into full-typed members (`globals.m_instanceIdToMaterials =
/// forwardSRT_…`) actively mismatch. (Hoisting also counts those
/// stores as member references, pinning template-nesting members in
/// struct definitions.)
///
/// Sound by construction: a local is dropped only when it is declared
/// exactly once (never a parameter, never shadow-redeclared), never
/// read (every occurrence is an assignment target), its initializer
/// (if any) is call-free, and every dropped store's right-hand side
/// and index expressions are call-free. Dropping pure reads changes
/// nothing observable; passing shaders without dead locals are
/// byte-identical.
pub fn drop_dead_locals(ir: &mut ShaderModule) {
    use std::collections::HashSet;

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

    /// Root variable written by an assignment target (`x`, `x.m`,
    /// `x[i].m` all write `x`; index subexpressions are reads).
    fn assign_root(e: &IrExpr) -> Option<&str> {
        match e {
            IrExpr::Variable(v) => Some(v.as_str()),
            IrExpr::Member { object, .. } => assign_root(object),
            IrExpr::Index { object, .. } | IrExpr::ResourceIndex { resource: object, .. } => {
                assign_root(object)
            }
            _ => None,
        }
    }

    /// Collect declarations, reads, and candidate stores per function.
    struct Usage {
        /// Local name -> declaration count (params count as declarations
        /// elsewhere; shadowing poisons by counting twice).
        decls: HashMap<String, usize>,
        /// Names read anywhere (including call args, RHS, indices).
        reads: HashSet<String>,
        /// `out`/`inout` parameters: stores through them are
        /// caller-visible and never dead, even when the name is never
        /// read in-body (helpers like `f(out uint o, ...)` with
        /// `o = buf[i]` had their only store dropped, silently
        /// miscompiling to an empty body).
        out_params: HashSet<String>,
    }

    fn walk_expr_reads(e: &IrExpr, u: &mut Usage) {
        match e {
            IrExpr::Variable(v) => {
                u.reads.insert(v.clone());
            }
            IrExpr::Assign { lhs, rhs } => {
                // Target root is written, not read; index subexpressions
                // and the RHS are reads.
                walk_target_reads(lhs, u);
                walk_expr_reads(rhs, u);
            }
            IrExpr::Member { object, .. } => walk_expr_reads(object, u),
            IrExpr::Index { object, index } | IrExpr::ResourceIndex { resource: object, index } => {
                walk_expr_reads(object, u);
                walk_expr_reads(index, u);
            }
            IrExpr::ResourceLoad {
                resource,
                coordinates,
                sample,
            } => {
                walk_expr_reads(resource, u);
                walk_expr_reads(coordinates, u);
                if let Some(sample) = sample {
                    walk_expr_reads(sample, u);
                }
            }
            IrExpr::Call { function, arguments } => {
                // Dotted callee (`loc.Append(...)`, `buf[i].Sample(...)`):
                // the object root is read (callee strings carry no
                // expression for it) — as is every identifier inside
                // index segments (`arr[materialId].Sample`: `materialId`
                // evaluates at the call). Method-name segments may
                // over-mark (a local sharing the name stays), which is
                // the safe direction.
                if function.contains('.') {
                    for seg in function.split(|c| {
                        c == '.' || c == '[' || c == ']' || c == '(' || c == ')'
                    }) {
                        if !seg.is_empty() {
                            u.reads.insert(seg.to_string());
                        }
                    }
                }
                for a in arguments {
                    walk_expr_reads(a, u);
                }
            }
            IrExpr::Construct { arguments, .. } => {
                for a in arguments {
                    walk_expr_reads(a, u);
                }
            }
            IrExpr::Unary { expr: inner, .. } => walk_expr_reads(inner, u),
            IrExpr::Binary { left, right, .. } => {
                walk_expr_reads(left, u);
                walk_expr_reads(right, u);
            }
            IrExpr::Conditional {
                condition,
                then_branch,
                else_branch,
            } => {
                walk_expr_reads(condition, u);
                walk_expr_reads(then_branch, u);
                walk_expr_reads(else_branch, u);
            }
            IrExpr::InitializerList(elems) => {
                for x in elems {
                    walk_expr_reads(x, u);
                }
            }
            IrExpr::Literal(_) => {}
        }
    }

    /// Reads inside an assignment target: everything except the written
    /// root (index/coordinate subexpressions still evaluate).
    fn walk_target_reads(e: &IrExpr, u: &mut Usage) {
        match e {
            IrExpr::Variable(_) => {}
            IrExpr::Member { object, .. } => walk_target_reads(object, u),
            IrExpr::Index { object, index } | IrExpr::ResourceIndex { resource: object, index } => {
                walk_target_reads(object, u);
                walk_expr_reads(index, u);
            }
            // Non-path targets (constructs, calls): be conservative and
            // count every variable as read.
            other => walk_expr_reads(other, u),
        }
    }

    fn walk_stmt(s: &IrStmt, u: &mut Usage, in_for_init: bool) {
        match s {
            IrStmt::Return(e) => {
                if let Some(e) = e {
                    walk_expr_reads(e, u);
                }
            }
            IrStmt::VarDecl { name, init, .. } => {
                // For-loop initializers are loop machinery, never drop
                // candidates (poison the name).
                *u.decls.entry(name.clone()).or_insert(0) += if in_for_init { 2 } else { 1 };
                if let Some(init) = init {
                    walk_expr_reads(init, u);
                }
            }
            IrStmt::Block(inner) => {
                for s in inner {
                    walk_stmt(s, u, false);
                }
            }
            IrStmt::If {
                condition,
                then_branch,
                else_branch,
            } => {
                walk_expr_reads(condition, u);
                for s in then_branch {
                    walk_stmt(s, u, false);
                }
                if let Some(else_branch) = else_branch {
                    for s in else_branch {
                        walk_stmt(s, u, false);
                    }
                }
            }
            IrStmt::For {
                init,
                condition,
                update,
                body,
            } => {
                walk_stmt(init, u, true);
                if let Some(condition) = condition {
                    walk_expr_reads(condition, u);
                }
                if let Some(update) = update {
                    walk_expr_reads(update, u);
                }
                for s in body {
                    walk_stmt(s, u, false);
                }
            }
            IrStmt::While { condition, body } => {
                walk_expr_reads(condition, u);
                for s in body {
                    walk_stmt(s, u, false);
                }
            }
            IrStmt::Switch { scrutinee, cases } => {
                walk_expr_reads(scrutinee, u);
                for c in cases {
                    if let Some(label) = &c.label {
                        walk_expr_reads(label, u);
                    }
                    for s in &c.body {
                        walk_stmt(s, u, false);
                    }
                }
            }
            IrStmt::Expr(e) => walk_expr_reads(e, u),
        }
    }

    /// True when removing `s` is safe for a dead local `name`: the
    /// declaration itself, or a pure store into it.
    fn droppable_for(s: &IrStmt, name: &str) -> bool {
        match s {
            IrStmt::VarDecl {
                name: n, init, ..
            } if n == name => match init {
                None => true,
                Some(e) => !expr_has_call(e),
            },
            IrStmt::Expr(e) => match e {
                IrExpr::Assign { lhs, rhs } => {
                    assign_root(lhs) == Some(name) && !expr_has_call(rhs) && !target_has_call(lhs)
                }
                _ => false,
            },
            _ => false,
        }
    }

    /// Calls hiding in an assignment target's index expressions (the
    /// written root itself is free).
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

    fn drop_in_list(stmts: &mut Vec<IrStmt>, dead: &HashSet<String>) {
        stmts.retain(|s| {
            if let IrStmt::VarDecl { name, .. } = s {
                if dead.contains(name) {
                    return false;
                }
            }
            if let IrStmt::Expr(e) = s {
                        if let IrExpr::Assign { lhs, .. } = e {
                    if let Some(root) = assign_root(lhs) {
                        if dead.contains(root) {
                            return false;
                        }
                    }
                }
            }
            true
        });
        for s in stmts.iter_mut() {
            match s {
                IrStmt::Block(inner) => drop_in_list(inner, dead),
                IrStmt::If {
                    then_branch,
                    else_branch,
                    ..
                } => {
                    drop_in_list(then_branch, dead);
                    if let Some(else_branch) = else_branch {
                        drop_in_list(else_branch, dead);
                    }
                }
                IrStmt::For { body, .. } => drop_in_list(body, dead),
                IrStmt::While { body, .. } => drop_in_list(body, dead),
                IrStmt::Switch { cases, .. } => {
                    for c in cases {
                        drop_in_list(&mut c.body, dead);
                    }
                }
                _ => {}
            }
        }
    }

    for func in &mut ir.functions {
        let mut u = Usage {
            decls: HashMap::new(),
            reads: HashSet::new(),
            out_params: HashSet::new(),
        };
        for p in &func.parameters {
            *u.decls.entry(p.name.clone()).or_insert(0) += 1;
            if p.direction.as_deref().is_some_and(|d| {
                d.eq_ignore_ascii_case("out") || d.eq_ignore_ascii_case("inout")
            }) {
                u.out_params.insert(p.name.clone());
            }
        }
        for s in &func.body {
            walk_stmt(s, &mut u, false);
        }
        // Candidates: declared exactly once, never read, and never an
        // out/inout parameter.
        let candidates: HashSet<String> = u
            .decls
            .iter()
            .filter(|(n, c)| **c == 1 && !u.reads.contains(*n) && !u.out_params.contains(*n))
            .map(|(n, _)| n.clone())
            .collect();
        if candidates.is_empty() {
            continue;
        }
        // Keep only locals whose declaration and every store are
        // droppable (pure). (Reads were precomputed above, so only
        // stores/declarations of dead names can appear here.)
        fn all_droppable(stmts: &[IrStmt], dead: &HashSet<String>) -> HashSet<String> {
            let mut ok: HashSet<String> = dead.clone();
            fn check(s: &IrStmt, dead: &HashSet<String>, ok: &mut HashSet<String>) {
                match s {
                    IrStmt::Block(inner) => {
                        for s in inner {
                            check(s, dead, ok);
                        }
                    }
                    IrStmt::If {
                        then_branch,
                        else_branch,
                        ..
                    } => {
                        for s in then_branch {
                            check(s, dead, ok);
                        }
                        if let Some(else_branch) = else_branch {
                            for s in else_branch {
                                check(s, dead, ok);
                            }
                        }
                    }
                    IrStmt::For { init, body, .. } => {
                        check(init, dead, ok);
                        for s in body {
                            check(s, dead, ok);
                        }
                    }
                    IrStmt::While { body, .. } => {
                        for s in body {
                            check(s, dead, ok);
                        }
                    }
                    IrStmt::Switch { cases, .. } => {
                        for c in cases {
                            for s in &c.body {
                                check(s, dead, ok);
                            }
                        }
                    }
                    IrStmt::VarDecl { name, .. } if dead.contains(name) => {
                        if !droppable_for(s, name) {
                            ok.remove(name);
                        }
                    }
                    IrStmt::Expr(e) => {
                        if let IrExpr::Assign { lhs, .. } = e {
                            if let Some(root) = assign_root(lhs) {
                                if dead.contains(root) && !droppable_for(s, root) {
                                    ok.remove(root);
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
            for s in stmts {
                check(s, &dead, &mut ok);
            }
            ok
        }
        let dead = all_droppable(&func.body, &candidates);
        if dead.is_empty() {
            continue;
        }
        drop_in_list(&mut func.body, &dead);
    }
}

/// Erases single-assignment resource aliases: resource-typed globals
/// stored exactly once from a pure value (`PbrPreintegratedDFG =
/// lookupTexturesSRT.m_preIntegratedDFG;`) and only read afterwards.
///
/// PSSL lets shaders alias texture handles through `static` globals;
/// SPIR-V forbids storing opaque types to any global storage
/// (`OpStore … storage class is read-only`), and HLSL `static` does not
/// help (images cannot live in private storage either). The store's
/// value, resolved through single-assignment locals to a module-scope
/// expression (bindings, cbuffer fields, member/index paths over
/// them), replaces every read of the alias; the global and its store
/// are dropped. Module-scope values are valid at every read site —
/// strictly more defined than the alias (which dangles before its
/// store executes).
///
/// Sound by construction: only resource-typed (or sampler-typed)
/// globals with exactly one pure store qualify; resolution aborts
/// (loud as today) on parameters, multi-assigned or impure locals,
/// calls anywhere in the chain, and unreadable shapes. Passing shaders
/// without resource aliases are untouched.
pub fn erase_resource_aliases(ir: &mut ShaderModule) {
    use std::collections::HashSet;

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

    fn is_aliasable_ty(ty: &IrType) -> bool {
        match ty {
            IrType::Resource { .. } => true,
            IrType::Named(n) => crate::framework::is_sampler_type_name(n),
            _ => false,
        }
    }

    // Module-scope roots usable at any read site: globals, resources,
    // and cbuffer fields (all host-set, always valid).
    let mut roots: HashSet<String> = HashSet::new();
    for g in &ir.globals {
        roots.insert(g.name.clone());
    }
    for r in &ir.resources {
        roots.insert(r.name.clone());
        if r.resource_type == ResourceType::ConstantBuffer {
            for f in &r.fields {
                roots.insert(f.name.clone());
            }
        }
    }

    // Locals with exactly one pure assignment/init each, plus leaf
    // stores; see FuncFacts below (per-function precise maps).
    // Calls hiding in an assignment target's index expressions (the
    // written root itself is free). Local copy: `drop_dead_locals`
    // owns the canonical one; duplicated here to keep this pass
    // self-contained.
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
    // NOTE: `single`/`multi` are function-scoped in principle, but the
    // resolved value must be module-scoped to replace reads anywhere;
    // locals resolving through same-named locals in other functions
    // would mis-resolve. Key by function index to stay precise.
    // Leaf stores (member/index paths like `globals.m_lut.m_tex`) are
    // likewise per-function: materialization residue assigns leaves,
    // never whole subobjects.
    struct FuncFacts {
        single: HashMap<String, IrExpr>,
        multi: HashSet<String>,
        leaves: HashMap<String, IrExpr>,
        leaf_multi: HashSet<String>,
    }
    // Canonical dotted path for variable-rooted member/literal-index
    // chains (`a.b[0].c`); `None` for anything else (calls, variable
    // indices, ...).
    fn path_of(e: &IrExpr) -> Option<String> {
        match e {
            IrExpr::Variable(v) => Some(v.clone()),
            IrExpr::Member { object, member } => {
                Some(format!("{}.{member}", path_of(object)?))
            }
            IrExpr::Index { object, index } => {
                if let IrExpr::Literal(lit) = index.as_ref() {
                    Some(format!("{}[{lit:?}]", path_of(object)?))
                } else {
                    None
                }
            }
            _ => None,
        }
    }
    let mut per_func: Vec<FuncFacts> = Vec::new();
    for func in &ir.functions {
        let mut facts = FuncFacts {
            single: HashMap::new(),
            multi: HashSet::new(),
            leaves: HashMap::new(),
            leaf_multi: HashSet::new(),
        };
        for p in &func.parameters {
            facts.multi.insert(p.name.clone());
        }
        fn record(
            s: &IrStmt,
            facts: &mut FuncFacts,
        ) {
            match s {
                IrStmt::VarDecl { name, init, .. } => {
                    if let Some(init) = init {
                        if expr_has_call(init) {
                            facts.multi.insert(name.clone());
                        } else if facts.multi.contains(name) {
                        } else if facts.single.contains_key(name) {
                            facts.multi.insert(name.clone());
                            facts.single.remove(name);
                        } else {
                            facts.single.insert(name.clone(), init.clone());
                        }
                    }
                }
                IrStmt::Block(inner) => {
                    for s in inner {
                        record(s, facts);
                    }
                }
                IrStmt::If {
                    then_branch,
                    else_branch,
                    ..
                } => {
                    for s in then_branch {
                        record(s, facts);
                    }
                    if let Some(b) = else_branch {
                        for s in b {
                            record(s, facts);
                        }
                    }
                }
                IrStmt::For { init, body, .. } => {
                    record(init, facts);
                    for s in body {
                        record(s, facts);
                    }
                }
                IrStmt::While { body, .. } => {
                    for s in body {
                        record(s, facts);
                    }
                }
                IrStmt::Switch { cases, .. } => {
                    for c in cases {
                        for s in &c.body {
                            record(s, facts);
                        }
                    }
                }
                IrStmt::Expr(e) => {
                    if let IrExpr::Assign { lhs, rhs } = e {
                        if expr_has_call(rhs) || target_has_call(lhs) {
                            // Impure store: disqualify every path it
                            // could alias (bare root and full path).
                            if let IrExpr::Variable(name) = lhs.as_ref() {
                                facts.multi.insert(name.clone());
                            }
                            if let Some(path) = path_of(lhs) {
                                facts.leaf_multi.insert(path);
                            }
                            return;
                        }
                        if let IrExpr::Variable(name) = lhs.as_ref() {
                            if facts.multi.contains(name) {
                            } else if facts.single.contains_key(name) {
                                facts.multi.insert(name.clone());
                                facts.single.remove(name);
                            } else {
                                facts.single.insert(name.clone(), rhs.as_ref().clone());
                            }
                        }
                        if let Some(path) = path_of(lhs) {
                            if facts.leaf_multi.contains(&path) {
                            } else if facts.leaves.contains_key(&path) {
                                facts.leaf_multi.insert(path.clone());
                                facts.leaves.remove(&path);
                            } else {
                                facts.leaves.insert(path, rhs.as_ref().clone());
                            }
                        }
                    }
                }
                IrStmt::Return(_) => {}
            }
        }
        for s in &func.body {
            record(s, &mut facts);
        }
        per_func.push(facts);
    }

    // Resolve an expression to module scope. Returns the resolved
    // expression, or `None` to abort (loud). Locals unfold through
    // their single pure assignment; member paths resolve through
    // single pure leaf stores (`globals.m_lut.m_tex`); anything else
    // must already be module-rooted (bindings, cbuffer fields,
    // literals). Unresolvable variable-rooted paths propagate as
    // dotted strings so outer member accesses can still hit a leaf.
    fn resolve(
        e: &IrExpr,
        func_idx: usize,
        per_func: &[FuncFacts],
        roots: &HashSet<String>,
        alias_raw: &HashMap<String, (usize, IrExpr)>,
        depth: usize,
    ) -> Option<IrExpr> {
        // (path, expr): variable-rooted dotted path still needing a
        // leaf, or a fully module-scoped expression.
        fn go(
            e: &IrExpr,
            func_idx: usize,
            per_func: &[FuncFacts],
            roots: &HashSet<String>,
            alias_raw: &HashMap<String, (usize, IrExpr)>,
            depth: usize,
        ) -> Option<(Option<String>, IrExpr)> {
            if depth > 16 {
                return None;
            }
            match e {
                IrExpr::Variable(v) => {
                    if roots.contains(v) {
                        return Some((None, e.clone()));
                    }
                    // Chained aliases resolve through each other; only
                    // fully module-scoped results cross functions
                    // (leftover paths stay home — abort). Cycles die by
                    // depth.
                    if let Some((afi, araw)) = alias_raw.get(v) {
                        let (p, r) = go(araw, *afi, per_func, roots, alias_raw, depth + 1)?;
                        if p.is_some() {
                            return None;
                        }
                        return Some((None, r));
                    }
                    let facts = per_func.get(func_idx)?;
                    if facts.multi.contains(v) {
                        return None;
                    }
                    if let Some(value) = facts.single.get(v) {
                        return go(value, func_idx, per_func, roots, alias_raw, depth + 1);
                    }
                    // Bare-declared local (materialization residue): no
                    // whole value, but member paths may hit leaves.
                    Some((Some(v.clone()), e.clone()))
                }
                IrExpr::Member { object, member } => {
                    let (path, o) = go(object, func_idx, per_func, roots, alias_raw, depth + 1)?;
                    let full = match path {
                        Some(p) => format!("{p}.{member}"),
                        None => return Some((
                            None,
                            IrExpr::Member {
                                object: Box::new(o),
                                member: member.clone(),
                            },
                        )),
                    };
                    let facts = per_func.get(func_idx)?;
                    if facts.leaf_multi.contains(&full) {
                        return None;
                    }
                    if let Some(value) = facts.leaves.get(&full) {
                        return go(value, func_idx, per_func, roots, alias_raw, depth + 1);
                    }
                    Some((Some(full), IrExpr::Member {
                        object: Box::new(o),
                        member: member.clone(),
                    }))
                }
                IrExpr::Index { object, index } => {
                    let (path, o) = go(object, func_idx, per_func, roots, alias_raw, depth + 1)?;
                    let (_, i) = go(index, func_idx, per_func, roots, alias_raw, depth + 1)?;
                    // Indices must be module-scoped (literals, fields);
                    // variable indices stay only if module-rooted.
                    let elem = match path {
                        Some(p) => {
                            if let IrExpr::Literal(_) = i {
                                format!("{p}[{i:?}]")
                            } else {
                                return None;
                            }
                        }
                        None => {
                            return Some((
                                None,
                                IrExpr::Index {
                                    object: Box::new(o),
                                    index: Box::new(i),
                                },
                            ))
                        }
                    };
                    let facts = per_func.get(func_idx)?;
                    if facts.leaf_multi.contains(&elem) {
                        return None;
                    }
                    if let Some(value) = facts.leaves.get(&elem) {
                        return go(value, func_idx, per_func, roots, alias_raw, depth + 1);
                    }
                    Some((Some(elem), IrExpr::Index {
                        object: Box::new(o),
                        index: Box::new(i),
                    }))
                }
                IrExpr::ResourceIndex { resource, index } => {
                    let (path, o) = go(resource, func_idx, per_func, roots, alias_raw, depth + 1)?;
                    if path.is_some() {
                        // Reads through locals never resolve to module
                        // scope here (buffer handles stay loud).
                        return None;
                    }
                    let (_, i) = go(index, func_idx, per_func, roots, alias_raw, depth + 1)?;
                    Some((
                        None,
                        IrExpr::ResourceIndex {
                            resource: Box::new(o),
                            index: Box::new(i),
                        },
                    ))
                }
                IrExpr::Literal(_) => Some((None, e.clone())),
                // Calls, loads, constructs, arithmetic: not alias values.
                _ => None,
            }
        }
        let (path, expr) = go(e, func_idx, per_func, roots, alias_raw, depth)?;
        if path.is_some() {
            // Leftover variable-rooted path (no leaf hit): not module
            // scope — abort.
            return None;
        }
        Some(expr)
    }

    // Alias candidates: resource/sampler globals AND non-cbuffer
    // resources with exactly one pure bare store (any function).
    // Chained aliases resolve through each other below.
    struct Alias {
        name: String,
        value: IrExpr,
    }
    let mut stores: HashMap<String, Vec<(usize, IrExpr)>> = HashMap::new();
    for (fi, func) in ir.functions.iter().enumerate() {
        fn find_stores(
            s: &IrStmt,
            fi: usize,
            stores: &mut HashMap<String, Vec<(usize, IrExpr)>>,
        ) {
            match s {
                IrStmt::Block(inner) => {
                    for s in inner {
                        find_stores(s, fi, stores);
                    }
                }
                IrStmt::If {
                    then_branch,
                    else_branch,
                    ..
                } => {
                    for s in then_branch {
                        find_stores(s, fi, stores);
                    }
                    if let Some(b) = else_branch {
                        for s in b {
                            find_stores(s, fi, stores);
                        }
                    }
                }
                IrStmt::For { init, body, .. } => {
                    find_stores(init, fi, stores);
                    for s in body {
                        find_stores(s, fi, stores);
                    }
                }
                IrStmt::While { body, .. } => {
                    for s in body {
                        find_stores(s, fi, stores);
                    }
                }
                IrStmt::Switch { cases, .. } => {
                    for c in cases {
                        for s in &c.body {
                            find_stores(s, fi, stores);
                        }
                    }
                }
                IrStmt::Expr(e) => {
                    if let IrExpr::Assign { lhs, rhs } = e {
                        if let IrExpr::Variable(name) = lhs.as_ref() {
                            stores
                                .entry(name.clone())
                                .or_default()
                                .push((fi, rhs.as_ref().clone()));
                        }
                    }
                }
                _ => {}
            }
        }
        for s in &func.body {
            find_stores(s, fi, &mut stores);
        }
    }
    let mut aliases: Vec<Alias> = Vec::new();
    // Raw single-store values by name (resolution may chain).
    let mut alias_raw: HashMap<String, (usize, IrExpr)> = HashMap::new();
    for g in &ir.globals {
        if g.shared || g.const_value.is_some() || g.array_size.is_some() {
            continue;
        }
        if !is_aliasable_ty(&g.ty) {
            continue;
        }
        let Some(entries) = stores.get(&g.name) else {
            continue;
        };
        if entries.len() != 1 {
            continue;
        }
        let (fi, value) = entries[0].clone();
        if expr_has_call(&value) {
            continue;
        }
        alias_raw.insert(g.name.clone(), (fi, value));
    }
    for r in &ir.resources {
        if r.resource_type == ResourceType::ConstantBuffer {
            continue;
        }
        if r.array_size.is_some() {
            continue;
        }
        let Some(entries) = stores.get(&r.name) else {
            continue;
        };
        if entries.len() != 1 {
            continue;
        }
        let (fi, value) = entries[0].clone();
        if expr_has_call(&value) {
            continue;
        }
        alias_raw.insert(r.name.clone(), (fi, value));
    }
    if alias_raw.is_empty() {
        return;
    }
    // Shadowing guard: a parameter/local sharing the alias name in ANY
    // function would be corrupted by module-wide rewriting — drop such
    // candidates (loud as today).
    {
        let mut declared: HashSet<String> = HashSet::new();
        fn collect_declared(s: &IrStmt, out: &mut HashSet<String>) {
            match s {
                IrStmt::VarDecl { name, .. } => {
                    out.insert(name.clone());
                }
                IrStmt::Block(inner) => {
                    for s in inner {
                        collect_declared(s, out);
                    }
                }
                IrStmt::If {
                    then_branch,
                    else_branch,
                    ..
                } => {
                    for s in then_branch {
                        collect_declared(s, out);
                    }
                    if let Some(b) = else_branch {
                        for s in b {
                            collect_declared(s, out);
                        }
                    }
                }
                IrStmt::For { init, body, .. } => {
                    collect_declared(init, out);
                    for s in body {
                        collect_declared(s, out);
                    }
                }
                IrStmt::While { body, .. } => {
                    for s in body {
                        collect_declared(s, out);
                    }
                }
                IrStmt::Switch { cases, .. } => {
                    for c in cases {
                        for s in &c.body {
                            collect_declared(s, out);
                        }
                    }
                }
                _ => {}
            }
        }
        for func in &ir.functions {
            for p in &func.parameters {
                declared.insert(p.name.clone());
            }
            for s in &func.body {
                collect_declared(s, &mut declared);
            }
        }
        alias_raw.retain(|name, _| !declared.contains(name));
    }
    if alias_raw.is_empty() {
        return;
    }
    // Module roots usable as replacement values, minus the aliases
    // themselves (chained aliases resolve through `alias_raw` above,
    // never short-circuit to another alias name).
    let resolve_roots: HashSet<String> = roots
        .iter()
        .filter(|r| !alias_raw.contains_key(*r))
        .cloned()
        .collect();
    for (name, (fi, value)) in &alias_raw {
        // Resolve through the storing function's locals to module scope.
        if let Some(resolved) = resolve(value, *fi, &per_func, &resolve_roots, &alias_raw, 0) {
            aliases.push(Alias {
                name: name.clone(),
                value: resolved,
            });
        }
    }
    if aliases.is_empty() {
        return;
    }

    // Dotted-callee uses (`PbrLtcMatTex.SampleLevel(...)` embed the
    // object in the call-function string, not as an expression) rewrite
    // through the value's path spelling. Values that cannot spell as a
    // path (calls, arithmetic, ...) keep aliases with such uses loud.
    fn render_path(e: &IrExpr) -> Option<String> {
        match e {
            IrExpr::Variable(v) => Some(v.clone()),
            IrExpr::Member { object, member } => {
                Some(format!("{}.{member}", render_path(object)?))
            }
            IrExpr::Index { object, index } => {
                Some(format!("{}[{}]", render_path(object)?, render_path(index)?))
            }
            IrExpr::ResourceIndex { resource, index } => {
                Some(format!("{}[{}]", render_path(resource)?, render_path(index)?))
            }
            IrExpr::Literal(lit) => match lit {
                crate::ir::IrLiteral::Number(s) => Some(s.clone()),
                crate::ir::IrLiteral::Bool(b) => Some(b.to_string()),
                crate::ir::IrLiteral::String(_) => None,
            },
            _ => None,
        }
    }
    fn callee_uses(s: &IrStmt, name: &str, out: &mut bool) {
        fn expr(e: &IrExpr, name: &str, out: &mut bool) {
            match e {
                IrExpr::Call { function, arguments } => {
                    if function == name
                        || function.starts_with(&format!("{name}."))
                        || function.starts_with(&format!("{name}["))
                    {
                        *out = true;
                    }
                    for a in arguments {
                        expr(a, name, out);
                    }
                }
                IrExpr::Member { object, .. } => expr(object, name, out),
                IrExpr::Index { object, index }
                | IrExpr::ResourceIndex { resource: object, index } => {
                    expr(object, name, out);
                    expr(index, name, out);
                }
                IrExpr::ResourceLoad {
                    resource,
                    coordinates,
                    sample,
                } => {
                    expr(resource, name, out);
                    expr(coordinates, name, out);
                    if let Some(sample) = sample {
                        expr(sample, name, out);
                    }
                }
                IrExpr::Construct { arguments, .. } => {
                    for a in arguments {
                        expr(a, name, out);
                    }
                }
                IrExpr::Unary { expr: inner, .. } => expr(inner, name, out),
                IrExpr::Binary { left, right, .. } => {
                    expr(left, name, out);
                    expr(right, name, out);
                }
                IrExpr::Conditional {
                    condition,
                    then_branch,
                    else_branch,
                } => {
                    expr(condition, name, out);
                    expr(then_branch, name, out);
                    expr(else_branch, name, out);
                }
                IrExpr::Assign { lhs, rhs } => {
                    expr(lhs, name, out);
                    expr(rhs, name, out);
                }
                IrExpr::InitializerList(elems) => {
                    for x in elems {
                        expr(x, name, out);
                    }
                }
                IrExpr::Variable(_) | IrExpr::Literal(_) => {}
            }
        }
        match s {
            IrStmt::Return(e) => {
                if let Some(e) = e {
                    expr(e, name, out);
                }
            }
            IrStmt::VarDecl { init, .. } => {
                if let Some(init) = init {
                    expr(init, name, out);
                }
            }
            IrStmt::Block(inner) => {
                for s in inner {
                    callee_uses(s, name, out);
                }
            }
            IrStmt::If {
                condition,
                then_branch,
                else_branch,
            } => {
                expr(condition, name, out);
                for s in then_branch {
                    callee_uses(s, name, out);
                }
                if let Some(b) = else_branch {
                    for s in b {
                        callee_uses(s, name, out);
                    }
                }
            }
            IrStmt::For {
                init,
                condition,
                update,
                body,
            } => {
                callee_uses(init, name, out);
                if let Some(condition) = condition {
                    expr(condition, name, out);
                }
                if let Some(update) = update {
                    expr(update, name, out);
                }
                for s in body {
                    callee_uses(s, name, out);
                }
            }
            IrStmt::While { condition, body } => {
                expr(condition, name, out);
                for s in body {
                    callee_uses(s, name, out);
                }
            }
            IrStmt::Switch { scrutinee, cases } => {
                expr(scrutinee, name, out);
                for c in cases {
                    if let Some(label) = &c.label {
                        expr(label, name, out);
                    }
                    for s in &c.body {
                        callee_uses(s, name, out);
                    }
                }
            }
            IrStmt::Expr(e) => expr(e, name, out),
        }
    }
    // path per alias; aliases with callee uses but no path stay loud.
    let mut paths: HashMap<String, String> = HashMap::new();
    aliases.retain(|a| {
        let mut used = false;
        for func in &ir.functions {
            for s in &func.body {
                callee_uses(s, &a.name, &mut used);
            }
        }
        if !used {
            return true;
        }
        match render_path(&a.value) {
            Some(p) => {
                paths.insert(a.name.clone(), p);
                true
            }
            None => false,
        }
    });
    if aliases.is_empty() {
        return;
    }

    // Rewrite reads (everywhere except the alias's own store target),
    // drop the store statements and the globals.
    fn rewrite_reads(
        s: &mut IrStmt,
        name: &str,
        value: &IrExpr,
        paths: &HashMap<String, String>,
    ) {
        fn expr(
            e: &mut IrExpr,
            name: &str,
            value: &IrExpr,
            paths: &HashMap<String, String>,
        ) {
            match e {
                IrExpr::Variable(v) => {
                    if v == name {
                        *e = value.clone();
                    }
                }
                IrExpr::Member { object, .. } => expr(object, name, value, paths),
                IrExpr::Index { object, index }
                | IrExpr::ResourceIndex { resource: object, index } => {
                    expr(object, name, value, paths);
                    expr(index, name, value, paths);
                }
                IrExpr::ResourceLoad {
                    resource,
                    coordinates,
                    sample,
                } => {
                    expr(resource, name, value, paths);
                    expr(coordinates, name, value, paths);
                    if let Some(sample) = sample {
                        expr(sample, name, value, paths);
                    }
                }
                IrExpr::Call { function, arguments } => {
                    // Dotted-callee object (`PbrLtcMatTex.SampleLevel`):
                    // re-root through the value's path spelling.
                    if let Some(path) = paths.get(name) {
                        if let Some(rest) = function.strip_prefix(&format!("{name}.")) {
                            *function = format!("{path}.{rest}");
                        } else if function == name {
                            *function = path.clone();
                        }
                    }
                    for a in arguments {
                        expr(a, name, value, paths);
                    }
                }
                IrExpr::Construct { arguments, .. } => {
                    for a in arguments {
                        expr(a, name, value, paths);
                    }
                }
                IrExpr::Unary { expr: inner, .. } => expr(inner, name, value, paths),
                IrExpr::Binary { left, right, .. } => {
                    expr(left, name, value, paths);
                    expr(right, name, value, paths);
                }
                IrExpr::Conditional {
                    condition,
                    then_branch,
                    else_branch,
                } => {
                    expr(condition, name, value, paths);
                    expr(then_branch, name, value, paths);
                    expr(else_branch, name, value, paths);
                }
                IrExpr::Assign { lhs, rhs } => {
                    // The alias's own store target is removed with its
                    // statement (below); never rewrite a target into a
                    // value (would create `binding = x`).
                    if !matches!(lhs.as_ref(), IrExpr::Variable(v) if v == name) {
                        expr(lhs, name, value, paths);
                    }
                    expr(rhs, name, value, paths);
                }
                IrExpr::InitializerList(elems) => {
                    for x in elems {
                        expr(x, name, value, paths);
                    }
                }
                IrExpr::Literal(_) => {}
            }
        }
        match s {
            IrStmt::Return(e) => {
                if let Some(e) = e {
                    expr(e, name, value, paths);
                }
            }
            IrStmt::VarDecl { init, .. } => {
                if let Some(init) = init {
                    expr(init, name, value, paths);
                }
            }
            IrStmt::Block(inner) => {
                for s in inner {
                    rewrite_reads(s, name, value, paths);
                }
            }
            IrStmt::If {
                condition,
                then_branch,
                else_branch,
            } => {
                expr(condition, name, value, paths);
                for s in then_branch {
                    rewrite_reads(s, name, value, paths);
                }
                if let Some(b) = else_branch {
                    for s in b {
                        rewrite_reads(s, name, value, paths);
                    }
                }
            }
            IrStmt::For {
                init,
                condition,
                update,
                body,
            } => {
                rewrite_reads(init, name, value, paths);
                if let Some(condition) = condition {
                    expr(condition, name, value, paths);
                }
                if let Some(update) = update {
                    expr(update, name, value, paths);
                }
                for s in body {
                    rewrite_reads(s, name, value, paths);
                }
            }
            IrStmt::While { condition, body } => {
                expr(condition, name, value, paths);
                for s in body {
                    rewrite_reads(s, name, value, paths);
                }
            }
            IrStmt::Switch { scrutinee, cases } => {
                expr(scrutinee, name, value, paths);
                for c in cases {
                    if let Some(label) = &mut c.label {
                        expr(label, name, value, paths);
                    }
                    for s in &mut c.body {
                        rewrite_reads(s, name, value, paths);
                    }
                }
            }
            IrStmt::Expr(e) => expr(e, name, value, paths),
        }
    }
    for alias in &aliases {
        for func in &mut ir.functions {
            for s in &mut func.body {
                rewrite_reads(s, &alias.name, &alias.value, &paths);
            }
        }
    }
    // Drop bare `alias = value;` store statements, then erase globals
    // whose every store dropped. Stores in non-statement positions
    // (conditions, initializers, for-init) cannot drop safely; the
    // final rescan below retains those globals (loud as today — such
    // shapes do not occur in materialization residue).
    fn drop_store_list(stmts: &mut Vec<IrStmt>, name: &str) {
        stmts.retain(|s| {
            if let IrStmt::Expr(e) = s {
                if let IrExpr::Assign { lhs, .. } = e {
                    if matches!(lhs.as_ref(), IrExpr::Variable(v) if v == name) {
                        return false;
                    }
                }
            }
            true
        });
        for s in stmts.iter_mut() {
            match s {
                IrStmt::Block(inner) => drop_store_list(inner, name),
                IrStmt::If {
                    then_branch,
                    else_branch,
                    ..
                } => {
                    drop_store_list(then_branch, name);
                    if let Some(b) = else_branch {
                        drop_store_list(b, name);
                    }
                }
                // For-init is loop machinery: never drop there (a store
                // there keeps its global loud via the rescan below).
                IrStmt::For { body, .. } => drop_store_list(body, name),
                IrStmt::While { body, .. } => drop_store_list(body, name),
                IrStmt::Switch { cases, .. } => {
                    for c in cases {
                        drop_store_list(&mut c.body, name);
                    }
                }
                _ => {}
            }
        }
    }
    for alias in &aliases {
        for func in &mut ir.functions {
            drop_store_list(&mut func.body, &alias.name);
        }
    }
    // Erase globals/resources with no remaining stores (the rescan
    // below also catches nested/condition stores the drop above
    // cannot take).
    let erase: Vec<String> = aliases
        .iter()
        .filter(|a| !complex_alias(&a.name, ir))
        .map(|a| a.name.clone())
        .collect();
    ir.globals.retain(|g| !erase.contains(&g.name));
    ir.resources.retain(|r| !erase.contains(&r.name));
    fn complex_alias(name: &str, ir: &ShaderModule) -> bool {
        // Re-scan: any remaining store to `name` (nested or otherwise)
        // means erasure was incomplete.
        fn has_store_stmt(s: &IrStmt, name: &str) -> bool {
            fn has_store(e: &IrExpr, name: &str) -> bool {
                match e {
                    IrExpr::Assign { lhs, rhs } => {
                        matches!(lhs.as_ref(), IrExpr::Variable(v) if v == name)
                            || has_store(lhs, name)
                            || has_store(rhs, name)
                    }
                    IrExpr::Member { object, .. } => has_store(object, name),
                    IrExpr::Index { object, index }
                    | IrExpr::ResourceIndex { resource: object, index } => {
                        has_store(object, name) || has_store(index, name)
                    }
                    IrExpr::ResourceLoad {
                        resource,
                        coordinates,
                        sample,
                    } => {
                        has_store(resource, name)
                            || has_store(coordinates, name)
                            || sample.as_ref().is_some_and(|s| has_store(s, name))
                    }
                    IrExpr::Call { arguments, .. } | IrExpr::Construct { arguments, .. } => {
                        arguments.iter().any(|a| has_store(a, name))
                    }
                    IrExpr::Unary { expr: inner, .. } => has_store(inner, name),
                    IrExpr::Binary { left, right, .. } => {
                        has_store(left, name) || has_store(right, name)
                    }
                    IrExpr::Conditional {
                        condition,
                        then_branch,
                        else_branch,
                    } => {
                        has_store(condition, name)
                            || has_store(then_branch, name)
                            || has_store(else_branch, name)
                    }
                    IrExpr::InitializerList(elems) => {
                        elems.iter().any(|x| has_store(x, name))
                    }
                    IrExpr::Variable(_) | IrExpr::Literal(_) => false,
                }
            }
            match s {
                IrStmt::Return(e) => e.as_ref().is_some_and(|e| has_store(e, name)),
                IrStmt::VarDecl { init, .. } => {
                    init.as_ref().is_some_and(|e| has_store(e, name))
                }
                IrStmt::Block(inner) => inner.iter().any(|s| has_store_stmt(s, name)),
                IrStmt::If {
                    condition,
                    then_branch,
                    else_branch,
                } => {
                    has_store(condition, name)
                        || then_branch.iter().any(|s| has_store_stmt(s, name))
                        || else_branch
                            .as_ref()
                            .is_some_and(|b| b.iter().any(|s| has_store_stmt(s, name)))
                }
                IrStmt::For {
                    init,
                    condition,
                    update,
                    body,
                } => {
                    has_store_stmt(init, name)
                        || condition.as_ref().is_some_and(|e| has_store(e, name))
                        || update.as_ref().is_some_and(|e| has_store(e, name))
                        || body.iter().any(|s| has_store_stmt(s, name))
                }
                IrStmt::While { condition, body } => {
                    has_store(condition, name)
                        || body.iter().any(|s| has_store_stmt(s, name))
                }
                IrStmt::Switch { scrutinee, cases } => {
                    has_store(scrutinee, name)
                        || cases.iter().any(|c| {
                            c.label.as_ref().is_some_and(|e| has_store(e, name))
                                || c.body.iter().any(|s| has_store_stmt(s, name))
                        })
                }
                IrStmt::Expr(e) => has_store(e, name),
            }
        }
        ir.functions.iter().any(|f| {
            f.body
                .iter()
                .any(|s| has_store_stmt(s, name))
        })
    }
}
