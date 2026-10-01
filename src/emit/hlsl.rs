//! Vulkan HLSL emitter: Shader IR -> HLSL.
//!
//! This is a deterministic, pure function of the IR: the same IR always
//! produces identical HLSL. Resources/bindings and semantics are emitted with
//! explicit Vulkan-oriented annotations.

use std::fmt::Write;

use crate::ast::Semantic;
use crate::ir::*;
use crate::semantic::bindings::map_semantic;

/// Emits Vulkan-oriented HLSL for the given IR module.
pub fn emit_module(module: &crate::ir::ShaderModule) -> String {
    let mut out = String::new();

    // Intrinsic shims first: they are self-contained (builtins only),
    // while struct methods (emitted inside structs below) and global
    // initializers can both reference them (`M_PI` inside a struct
    // method, `__wave_inverse` anywhere). HLSL needs declarations
    // before use, so everything else follows.
    emit_intrinsic_shims(&mut out, module);

    // Emit structs first – resources like `StructuredBuffer<PackModelInstanceUserData>`
    // require the struct type to be defined before use (HLSL forward-reference
    // is not allowed). Previously resources were emitted first, causing
    // `use of undeclared identifier` errors for structs used by resources.
    //
    // Re-inlined headers (and SFINAE helpers like `enable_if` picked up as
    // empty structs) can register the same struct name more than once. Emit
    // each name a single time: DXC rejects `redefinition of '...'`.
    let mut emitted_structs: std::collections::HashSet<&str> =
        std::collections::HashSet::new();
    // Declared struct names, used to resolve Named field types: a Named
    // field type that is NOT a declared struct is an enum (PSSL enums are
    // int-backed) and must emit as `int` (DXC: unknown type name).
    let declared: std::collections::HashSet<&str> = module
        .structs
        .iter()
        .map(|s| s.name.as_str())
        .collect();
    // Struct definition order: PSSL headers forward-reference structs
    // (`TransformedNodeData` holds `MeshData*` before `MeshData` is
    // defined), but HLSL needs dependencies first. Stable topological
    // order (original index breaks ties, so already-ordered modules are
    // byte-identical); cycles keep original order and stay loud.
    let struct_order = sort_structs_for_emit(module);
    // Free-function prototypes, gated on actually-emitted struct methods
    // (otherwise the HLSL below is byte-identical to before): methods see
    // the whole class scope in C++ and may call helpers defined later.
    // Builtin-signature prototypes come before structs (methods need
    // them); struct-typed ones after (they name struct types).
    //
    // Only reachable methods survive lowering (unreachable ones are
    // stripped there); emit survivors scoped to this struct.
    let reachable = crate::ir::reachable_methods(module);
    let has_methods = !reachable.is_empty();
    // Free-function forward calls (e.g. a 2-arg overload wrapper calling
    // the 4-arg overload defined later in the module, as PSSL headers
    // order them): HLSL needs declarations before use. Previously
    // prototypes were only emitted when struct methods existed, so a
    // methodless module with forward calls emitted a callee defined
    // later and DXC failed (`no matching function for call`). Extend
    // the gate: any bare call to a free function defined later in
    // module order needs prototypes. (Self-recursion needs none: the
    // function's own signature precedes its body.)
    let needs_prototypes = has_methods || {
        // Module indices of free functions (emitted at top level below),
        // by bare name.
        let mut by_name: std::collections::HashMap<&str, Vec<usize>> =
            std::collections::HashMap::new();
        for (idx, f) in module.functions.iter().enumerate() {
            if f.scope.as_deref().is_some_and(|s| declared.contains(s)) {
                continue;
            }
            by_name.entry(f.name.as_str()).or_default().push(idx);
        }
        module.functions.iter().enumerate().any(|(i, f)| {
            if f.scope.as_deref().is_some_and(|s| declared.contains(s)) {
                return false;
            }
            crate::ir::function::function_calls(f).iter().any(|c| {
                // Bare free calls appear as `foo`; member calls as
                // `obj.method` and scoped calls as `Scope::method` —
                // neither is a free-function forward reference.
                if c.contains('.') || c.contains("::") {
                    return false;
                }
                by_name
                    .get(c.as_str())
                    .is_some_and(|idxs| idxs.iter().any(|&j| j > i))
            })
        })
    };
    if needs_prototypes {
        emit_prototypes(&mut out, module, &declared, true);
    }
    // Globals assigned anywhere in-shader (mutable state like RNG seeds)
    // must emit `static` (see the globals loop below).
    let assigned = assigned_globals(module);
    // Statically-called instance-clean methods emit `static` (native
    // `S::m` resolution); computed once for all structs.
    let statics = static_methods(module);
    if has_methods {
        emit_prototypes(&mut out, module, &declared, true);
    }
    for structure in &struct_order {
        if !emitted_structs.insert(structure.name.as_str()) {
            continue;
        }
        // Methods scoped to this struct emit inside its braces as HLSL
        // struct methods, so `obj.method()` calls resolve natively.
        let methods: Vec<&Function> = module
            .functions
            .iter()
            .filter(|f| {
                f.scope.as_deref() == Some(structure.name.as_str())
                    && reachable.contains(&(structure.name.clone(), f.name.clone()))
            })
            .collect();
        emit_struct(&mut out, structure, &declared, &methods, &module.structs, &statics);
        out.push('\n');
    }

    // Struct-typed prototypes (naming struct types declared above).
    if needs_prototypes {
        emit_prototypes(&mut out, module, &declared, false);
    }

    for resource in &module.resources {
        emit_resource(&mut out, resource, &assigned);
    }
    if !module.resources.is_empty() {
        out.push('\n');
    }

    let mut seen_globals: std::collections::HashSet<String> = std::collections::HashSet::new();
    for global in &module.globals {
        if !seen_globals.insert(global.name.clone()) {
            continue;
        }
        // Skip HLSL-invalid `VertexLayout` global – handled via `strip_vertex_layout_helpers`
        // but may still appear as `VertexLayout vertexLayout;` if the header was not
        // fully elided. It's not a valid HLSL type and the `[vertexlayout]` attribute
        // is PSSL-only, so we drop it entirely.
        if global.ty == crate::ir::IrType::Named("VertexLayout".to_string()) {
            continue;
        }
        let array = global
            .array_size
            .map(|n| if n == 0 { "[]".to_string() } else { format!("[{n}]") })
            .unwrap_or_default();
        // Compile-time constants (e.g. `static constexpr uint k_size = 8;`) are
        // rendered as `static const uint k_size = 8;` rather than an uninitialized
        // global — DXC (and Vulkan) require an initializer for such constants.
        if let Some(value) = &global.const_value {
            // An unsuffixed decimal literal over int32 range is not a
            // valid HLSL `uint` initializer (`static const uint k =
            // 4294967295;` errors: the literal itself is parsed as int).
            // `ulong`-backed enumerators (`InvalidIntersection::kValue`)
            // land here, so suffix exactly those.
            let rendered = match &global.ty {
                crate::ir::IrType::UInt
                    if value
                        .trim()
                        .parse::<i64>()
                        .is_ok_and(|v| v > i32::MAX as i64) =>
                {
                    format!("{}u", value.trim())
                }
                _ => value.clone(),
            };
            let _ = writeln!(
                out,
                "static const {} {} = {};",
                hlsl_type(&global.ty),
                hlsl_ident(&global.name),
                rendered
            );
        } else {
            // Per-workgroup LDS (`thread_group_memory`): HLSL `groupshared`
            // (writable). A plain global would be read-only under SPIR-V.
            //
            // Shader-mutated globals (e.g. an xorshift `uint m_state;`
            // assigned inside a function) also cannot stay plain: a
            // plain global lands in the implicit `$Globals` cbuffer and
            // is const-qualified at DXC (`cannot assign to variable ...
            // with const-qualified type`). They emit `static` (mutable
            // private state, never host-bound — host-bound uniforms are
            // never assigned in-shader, so they keep today's shape).
            // Resource-typed globals keep today's shape (rebinding a
            // resource stays loud).
            let mutated = !global.shared
                && !matches!(global.ty, crate::ir::IrType::Resource { .. })
                && assigned.contains(global.name.as_str());
            let prefix = if global.shared {
                "groupshared "
            } else if mutated {
                "static "
            } else {
                ""
            };
            let _ = writeln!(
                out,
                "{prefix}{} {}{};",
                hlsl_type(&global.ty),
                hlsl_ident(&global.name),
                array
            );
        }
    }
    if !module.globals.is_empty() {
        out.push('\n');
    }

    for function in &module.functions {
        // Scoped methods emit inside their struct (above), not as free
        // functions — unless the scope names no declared struct (e.g.
        // out-of-class definitions into unknown scopes), which keep
        // today's free-function behavior.
        if let Some(scope) = function.scope.as_deref() {
            if declared.contains(scope) {
                continue;
            }
        }
        // Generic `T` from `template<class T>` – e.g. `T tex2D_bicubic(Texture2D<T> tex, ...)`
        // DXC has no `T`; emit concrete overloads for the half family used in
        // this shader (half, half2, half3, half4). The call sites deduce `T`
        // from the texture element type, so overload resolution picks the right one.
        if is_generic_t_function(function) {
            tracing::trace!(
                target: "pssl_shader::emit::generic",
                func = %function.name,
                "emitting T overloads for half/float family"
            );
            // Emit concrete overloads for both the half and float families
            // (the call site deduces T from the texture element type, e.g.
            // `Texture2D<float4>` -> float4). The body is also substituted
            // so `T` locals/casts (`T output; T(1.f)`) become concrete.
            for base in [crate::ir::IrScalar::Half, crate::ir::IrScalar::Float] {
                for width in [1u8, 2, 3, 4] {
                    let concrete = if width == 1 {
                        match base {
                            crate::ir::IrScalar::Half => crate::ir::IrType::Half,
                            _ => crate::ir::IrType::Float,
                        }
                    } else {
                        crate::ir::IrType::Vector { base, width }
                    };
                    let mut cloned = function.clone();
                    replace_t_with(&mut cloned.return_type, &concrete);
                    for param in &mut cloned.parameters {
                        replace_t_with(&mut param.ty, &concrete);
                    }
                    for stmt in &mut cloned.body {
                        replace_t_in_stmt(stmt, &concrete);
                    }
                    emit_function(&mut out, &cloned, &module.structs);
                    out.push('\n');
                }
            }
            continue;
        }
        emit_function(&mut out, function, &module.structs);
        out.push('\n');
    }

    out
}

fn is_generic_t_function(func: &Function) -> bool {
    let has_t = |ty: &IrType| match ty {
        IrType::Named(n) if n == "T" => true,
        IrType::Resource { element, .. } if **element == IrType::Named("T".to_string()) => true,
        _ => false,
    };
    if has_t(&func.return_type) {
        return true;
    }
    for p in &func.parameters {
        if has_t(&p.ty) {
            return true;
        }
    }
    false
}

fn replace_t_with(ty: &mut IrType, concrete: &IrType) {
    match ty {
        IrType::Named(n) if n == "T" => *ty = concrete.clone(),
        IrType::Resource { element, .. } if **element == IrType::Named("T".to_string()) => {
            *element = Box::new(concrete.clone());
        }
        _ => {}
    }
}

/// Substitute `T` in a function body statement (local `T` variables, casts).
fn replace_t_in_stmt(stmt: &mut crate::ir::function::IrStmt, concrete: &IrType) {
    use crate::ir::function::IrStmt;
    match stmt {
        IrStmt::VarDecl { ty, init, .. } => {
            replace_t_with(ty, concrete);
            if let Some(e) = init {
                replace_t_in_expr(e, concrete);
            }
        }
        IrStmt::Return(Some(e)) | IrStmt::Expr(e) => replace_t_in_expr(e, concrete),
        IrStmt::Return(None) => {}
        IrStmt::If {
            condition,
            then_branch,
            else_branch,
        } => {
            replace_t_in_expr(condition, concrete);
            for s in then_branch {
                replace_t_in_stmt(s, concrete);
            }
            if let Some(eb) = else_branch {
                for s in eb {
                    replace_t_in_stmt(s, concrete);
                }
            }
        }
        IrStmt::For {
            init,
            condition,
            update,
            body,
        } => {
            replace_t_in_stmt(init, concrete);
            if let Some(c) = condition {
                replace_t_in_expr(c, concrete);
            }
            if let Some(u) = update {
                replace_t_in_expr(u, concrete);
            }
            for s in body {
                replace_t_in_stmt(s, concrete);
            }
        }
        IrStmt::While { condition, body } => {
            replace_t_in_expr(condition, concrete);
            for s in body {
                replace_t_in_stmt(s, concrete);
            }
        }
        IrStmt::Switch { scrutinee, cases } => {
            replace_t_in_expr(scrutinee, concrete);
            for case in cases {
                if let Some(label) = &mut case.label {
                    replace_t_in_expr(label, concrete);
                }
                for s in &mut case.body {
                    replace_t_in_stmt(s, concrete);
                }
            }
        }
        IrStmt::Block(stmts) => {
            for s in stmts {
                replace_t_in_stmt(s, concrete);
            }
        }
    }
}

/// Substitute `T` in an expression, including `Construct { ty: T }` casts.
fn replace_t_in_expr(e: &mut crate::ir::IrExpr, concrete: &IrType) {
    use crate::ir::IrExpr;
    match e {
        IrExpr::Construct { ty, arguments } => {
            replace_t_with(ty, concrete);
            for a in arguments {
                replace_t_in_expr(a, concrete);
            }
        }
        IrExpr::Variable(_) | IrExpr::Literal(_) => {}
        IrExpr::Member { object, .. } => replace_t_in_expr(object, concrete),
        IrExpr::Index { object, index } | IrExpr::ResourceIndex { resource: object, index } => {
            replace_t_in_expr(object, concrete);
            replace_t_in_expr(index, concrete);
        }
        IrExpr::ResourceLoad {
            resource,
            coordinates,
            sample,
        } => {
            replace_t_in_expr(resource, concrete);
            replace_t_in_expr(coordinates, concrete);
            if let Some(s) = sample {
                replace_t_in_expr(s, concrete);
            }
        }
        IrExpr::Call { arguments, .. } => {
            for a in arguments {
                replace_t_in_expr(a, concrete);
            }
        }
        IrExpr::Unary { expr: inner, .. } => replace_t_in_expr(inner, concrete),
        IrExpr::Binary { left, right, .. } => {
            replace_t_in_expr(left, concrete);
            replace_t_in_expr(right, concrete);
        }
        IrExpr::Conditional {
            condition,
            then_branch,
            else_branch,
        } => {
            replace_t_in_expr(condition, concrete);
            replace_t_in_expr(then_branch, concrete);
            replace_t_in_expr(else_branch, concrete);
        }
        IrExpr::Assign { lhs, rhs } => {
            replace_t_in_expr(lhs, concrete);
            replace_t_in_expr(rhs, concrete);
        }
        IrExpr::InitializerList(elems) => {
            for x in elems {
                replace_t_in_expr(x, concrete);
            }
        }
    }
}

/// Emits fallback definitions for PSSL-specific intrinsics that have no direct
/// HLSL/DXC equivalent. These are declared as identity shims so the generated
/// HLSL compiles and validates; faithfully reproducing the platform quad-lane
/// semantics is not possible with portable HLSL controls.
fn emit_intrinsic_shims(out: &mut String, module: &crate::ir::ShaderModule) {
    tracing::trace!(target: "pssl_shader::emit::shims", "emitting Vulkan sampler shims");
    let mut shims = String::from(
        r#"
SamplerState g_point_sampler;
SamplerState g_bilinear_sampler;
SamplerState g_trilinear_sampler;

float QuadSwizzle(float v, int o0, int o1, int o2, int o3) { return v; }
float2 QuadSwizzle(float2 v, int o0, int o1, int o2, int o3) { return v; }
float3 QuadSwizzle(float3 v, int o0, int o1, int o2, int o3) { return v; }
float4 QuadSwizzle(float4 v, int o0, int o1, int o2, int o3) { return v; }
"#,
    );
    // M_PI is provided by virtual headers when needed; also emit here only if
    // the module actually references it and doesn't already define it, to keep
    // already-valid HLSL unchanged.
    let has_m_pi_global = module.globals.iter().any(|g| g.name == "M_PI");
    let uses_m_pi = !has_m_pi_global && module_uses_m_pi(module);
    if uses_m_pi {
        shims.push_str("\n#ifndef M_PI\nstatic const float M_PI = 3.14159265358979323846;\n#endif\n");
    }
    // `MaskBitCnt(m)` (AMD mbcnt lowered in `lower/expr.rs`) needs a helper:
    // popcount of the mask bits strictly below the current lane. PSSL
    // `ulong` is 32-bit, so one dword suffices; lane 0 yields 0 and lanes
    // >= 32 see the whole word. Emitted only when used.
    if module_uses_maskbitcnt(module, "__maskbitcnt_u32") {
        shims.push_str("\nuint __maskbitcnt_u32(uint m) { uint lane = WaveGetLaneIndex(); uint below = lane >= 32u ? 0xffffffffu : ((1u << lane) - 1u); return countbits(m & below); }\n");
    }
    // Raw `__v_mbcnt_hi_u32_b32` (upper 32 lanes below the current lane;
    // lanes below 32 contribute nothing). Pairs with the `_lo` shim above
    // for the wave64 lane-index idiom. Emitted only when used.
    if module_uses_maskbitcnt(module, "__maskbitcnt_hi_u32") {
        shims.push_str("\nuint __maskbitcnt_hi_u32(uint m) { uint lane = WaveGetLaneIndex(); uint rel = lane >= 32u ? (lane - 32u) : 0u; uint below = rel == 0u ? 0u : ((1u << rel) - 1u); return countbits(m & below); }\n");
    }
    // GCN `V_PERM_B32(a, b, sel)` (byte permute across the concatenated
    // `(b, a)` pair; selectors < 8 pick source bytes, 12 broadcasts
    // 0xff, anything else yields 0): no HLSL spelling, so a gated shim
    // with dynamic shifts (upscale mip averaging). Emitted only when used.
    if module_uses_call(module, "__v_perm_b32") {
        shims.push_str("\nuint __v_perm_b32(uint a, uint b, uint sel) { uint r = 0u; for (uint j = 0u; j < 4u; j++) { uint s = (sel >> (j * 8u)) & 0xffu; uint byte = 0u; if (s < 8u) { uint src = (s < 4u) ? a : b; byte = (src >> ((s & 3u) * 8u)) & 0xffu; } else if (s == 12u) { byte = 0xffu; } r |= byte << (j * 8u); } return r; }\n");
    }
    // Framework `wave::inverse(m)` (lowered in `lower/expr.rs`): HLSL has no
    // matrix-inverse intrinsic, so adjugate/determinant expansions (exact
    // for any invertible 3x3/4x4; singular inputs are undefined on both
    // compilers). 3x3 and 4x4 overloads; HLSL resolves the call natively.
    // Emitted only when used.
    if module_uses_call(module, "__wave_inverse") {
        shims.push_str("\nfloat __wave_det2(float2 a, float2 b) { return a.x * b.y - a.y * b.x; }\n");
        shims.push_str("float __wave_det3(float3 a, float3 b, float3 c) { return dot(a, cross(b, c)); }\n");
        shims.push_str("float3x3 __wave_inverse(float3x3 m) { float3 r0 = m[0]; float3 r1 = m[1]; float3 r2 = m[2]; float3 a0 = float3(__wave_det2(r1.yz, r2.yz), -__wave_det2(r0.yz, r2.yz), __wave_det2(r0.yz, r1.yz)); float det = r0.x * a0.x + r1.x * a0.y + r2.x * a0.z; float3 a1 = float3(-__wave_det2(r1.xz, r2.xz), __wave_det2(r0.xz, r2.xz), -__wave_det2(r0.xz, r1.xz)); float3 a2 = float3(__wave_det2(r1.xy, r2.xy), -__wave_det2(r0.xy, r2.xy), __wave_det2(r0.xy, r1.xy)); return float3x3(a0 / det, a1 / det, a2 / det); }\n");
        shims.push_str("float4x4 __wave_inverse(float4x4 m) { float4 r0 = m[0]; float4 r1 = m[1]; float4 r2 = m[2]; float4 r3 = m[3]; float4 a0 = float4(__wave_det3(r1.yzw, r2.yzw, r3.yzw), -__wave_det3(r0.yzw, r2.yzw, r3.yzw), __wave_det3(r0.yzw, r1.yzw, r3.yzw), -__wave_det3(r0.yzw, r1.yzw, r2.yzw)); float det = dot(float4(r0.x, r1.x, r2.x, r3.x), a0); float4 a1 = float4(-__wave_det3(r1.xzw, r2.xzw, r3.xzw), __wave_det3(r0.xzw, r2.xzw, r3.xzw), -__wave_det3(r0.xzw, r1.xzw, r3.xzw), __wave_det3(r0.xzw, r1.xzw, r2.xzw)); float4 a2 = float4(__wave_det3(r1.xyw, r2.xyw, r3.xyw), -__wave_det3(r0.xyw, r2.xyw, r3.xyw), __wave_det3(r0.xyw, r1.xyw, r3.xyw), -__wave_det3(r0.xyw, r1.xyw, r2.xyw)); float4 a3 = float4(-__wave_det3(r1.xyz, r2.xyz, r3.xyz), __wave_det3(r0.xyz, r2.xyz, r3.xyz), -__wave_det3(r0.xyz, r1.xyz, r3.xyz), __wave_det3(r0.xyz, r1.xyz, r2.xyz)); return float4x4(a0 / det, a1 / det, a2 / det, a3 / det); }\n");
    }
    // GCN `S_BFM_B64(width, offset)` (lowered in `lower/expr.rs`):
    // contiguous bitfield mask, 32-bit saturating to match the
    // toolchain's 32-bit `ulong` model. Emitted only when used.
    if module_uses_call(module, "__s_bfm_b64") {
        shims.push_str("\nuint __s_bfm_b64(uint w, uint o) { return (w >= 32u ? 0xffffffffu : ((1u << w) - 1u)) << o; }\n");
    }
    let _ = write!(out, "{shims}\n");
}

fn module_uses_m_pi(module: &crate::ir::ShaderModule) -> bool {
    for f in &module.functions {
        for s in &f.body {
            if stmt_uses_m_pi(s) {
                return true;
            }
        }
    }
    for g in &module.globals {
        if let Some(v) = &g.const_value {
            if v.contains("M_PI") {
                return true;
            }
        }
    }
    false
}

/// Stable topological order of struct definitions for emission:
/// dependencies (field types, resource elements, method signatures)
/// precede dependents. PSSL headers forward-reference freely; HLSL
/// does not. Original module order breaks ties (no-op for ordered
/// modules) and cycles (left loud).
fn sort_structs_for_emit(module: &crate::ir::ShaderModule) -> Vec<&crate::ir::StructDecl> {
    use std::collections::{HashMap, HashSet};
    fn deps_of(
        s: &crate::ir::StructDecl,
        module: &crate::ir::ShaderModule,
        names: &HashSet<&str>,
    ) -> HashSet<String> {
        let mut deps = HashSet::new();
        let mut note_ty = |t: &crate::ir::IrType| {
            match t {
                crate::ir::IrType::Named(n) => {
                    if names.contains(n.as_str()) && n != &s.name {
                        deps.insert(n.clone());
                    }
                }
                crate::ir::IrType::Resource { element, .. } => {
                    if let crate::ir::IrType::Named(n) = element.as_ref() {
                        if names.contains(n.as_str()) && n != &s.name {
                            deps.insert(n.clone());
                        }
                    }
                }
                _ => {}
            }
        };
        for f in &s.fields {
            note_ty(&f.ty);
        }
        // Method-body local types (`StateProxy state_;` inside
        // `Route::walkAround`): HLSL needs the named struct defined
        // before the struct whose method declares the local, exactly
        // like fields and signatures. Without this edge the sort can
        // emit the user before its local's type (`unknown type name`).
        fn locals_in(stmts: &[crate::ir::IrStmt], out: &mut Vec<crate::ir::IrType>) {
            for stmt in stmts {
                match stmt {
                    crate::ir::IrStmt::VarDecl { ty, .. } => out.push(ty.clone()),
                    crate::ir::IrStmt::Block(ss) => locals_in(ss, out),
                    crate::ir::IrStmt::If { then_branch, else_branch, .. } => {
                        locals_in(then_branch, out);
                        if let Some(els) = else_branch {
                            locals_in(els, out);
                        }
                    }
                    crate::ir::IrStmt::For { init, body, .. } => {
                        locals_in(std::slice::from_ref(init), out);
                        locals_in(body, out);
                    }
                    crate::ir::IrStmt::While { body, .. } => locals_in(body, out),
                    crate::ir::IrStmt::Switch { cases, .. } => {
                        for c in cases {
                            locals_in(&c.body, out);
                        }
                    }
                    _ => {}
                }
            }
        }
        for m in &module.functions {
            if m.scope.as_deref() != Some(s.name.as_str()) {
                continue;
            }
            note_ty(&m.return_type);
            for p in &m.parameters {
                note_ty(&p.ty);
            }
            let mut locals = Vec::new();
            locals_in(&m.body, &mut locals);
            for ty in &locals {
                note_ty(ty);
            }
        }
        deps
    }
    let names: HashSet<&str> = module.structs.iter().map(|s| s.name.as_str()).collect();
    // Duplicate definitions (re-inlined headers, forward declarations
    // like `struct ExposureSRT;`) collapse for ordering: the first
    // non-empty definition wins, else the first occurrence. Emission
    // still dedups by name. (An empty fwd-decl winning would leave
    // member uses dangling.)
    let mut order: Vec<&crate::ir::StructDecl> = Vec::new();
    let mut seen: HashSet<&str> = HashSet::new();
    for s in &module.structs {
        if s.fields.is_empty() {
            continue;
        }
        if seen.insert(s.name.as_str()) {
            order.push(s);
        }
    }
    // Names seen only as empty (genuinely empty structs, SFINAE
    // helpers): keep the first occurrence, in order.
    for s in &module.structs {
        if s.fields.is_empty() && seen.insert(s.name.as_str()) {
            order.push(s);
        }
    }
    let deps: HashMap<&str, HashSet<String>> = order
        .iter()
        .map(|s| (s.name.as_str(), deps_of(s, module, &names)))
        .collect();
    let mut emitted: HashSet<&str> = HashSet::new();
    let mut sorted: Vec<&crate::ir::StructDecl> = Vec::with_capacity(order.len());
    while sorted.len() < order.len() {
        let mut progressed = false;
        for s in &order {
            if emitted.contains(s.name.as_str()) {
                continue;
            }
            let ready = deps
                .get(s.name.as_str())
                .is_none_or(|ds| ds.iter().all(|d| emitted.contains(d.as_str())));
            if ready {
                emitted.insert(s.name.as_str());
                sorted.push(s);
                progressed = true;
            }
        }
        if !progressed {
            // Cycle: keep the rest in original order (loud at DXC).
            for s in &order {
                if emitted.insert(s.name.as_str()) {
                    sorted.push(s);
                }
            }
        }
    }
    sorted
}

/// Whether any function body calls the named lowering shim, gating its
/// emission in `emit_intrinsic_shims` (`__wave_inverse` here;
/// `module_uses_maskbitcnt` below is the older maskbitcnt-specific twin).
fn module_uses_call(module: &crate::ir::ShaderModule, name: &str) -> bool {
    fn expr_uses(e: &crate::ir::IrExpr, name: &str) -> bool {
        use crate::ir::IrExpr;
        match e {
            IrExpr::Call { function, arguments } => {
                if function == name {
                    return true;
                }
                arguments.iter().any(|a| expr_uses(a, name))
            }
            IrExpr::Variable(_) | IrExpr::Literal(_) => false,
            IrExpr::Member { object, .. } => expr_uses(object, name),
            IrExpr::Index { object, index } | IrExpr::ResourceIndex { resource: object, index } => {
                expr_uses(object, name) || expr_uses(index, name)
            }
            IrExpr::Construct { arguments, .. } => arguments.iter().any(|a| expr_uses(a, name)),
            IrExpr::Assign { lhs, rhs } => expr_uses(lhs, name) || expr_uses(rhs, name),
            IrExpr::Unary { expr, .. } => expr_uses(expr, name),
            IrExpr::Binary { left, right, .. } => expr_uses(left, name) || expr_uses(right, name),
            IrExpr::Conditional { condition, then_branch, else_branch } => {
                expr_uses(condition, name)
                    || expr_uses(then_branch, name)
                    || expr_uses(else_branch, name)
            }
            IrExpr::InitializerList(elems) => elems.iter().any(|x| expr_uses(x, name)),
            _ => false,
        }
    }
    fn stmt_uses(s: &crate::ir::IrStmt, name: &str) -> bool {
        use crate::ir::IrStmt;
        match s {
            IrStmt::Return(Some(e)) | IrStmt::Expr(e) => expr_uses(e, name),
            IrStmt::Return(None) => false,
            IrStmt::VarDecl { init: Some(e), .. } => expr_uses(e, name),
            IrStmt::VarDecl { init: None, .. } => false,
            IrStmt::Block(ss) => ss.iter().any(|x| stmt_uses(x, name)),
            IrStmt::If { condition, then_branch, else_branch } => {
                expr_uses(condition, name)
                    || then_branch.iter().any(|x| stmt_uses(x, name))
                    || else_branch.iter().any(|ss| ss.iter().any(|x| stmt_uses(x, name)))
            }
            IrStmt::For { init, condition, update, body } => {
                stmt_uses(init, name)
                    || condition.as_ref().is_some_and(|c| expr_uses(c, name))
                    || update.as_ref().is_some_and(|u| expr_uses(u, name))
                    || body.iter().any(|x| stmt_uses(x, name))
            }
            IrStmt::While { condition, body } => {
                expr_uses(condition, name) || body.iter().any(|x| stmt_uses(x, name))
            }
            IrStmt::Switch { scrutinee, cases } => {
                expr_uses(scrutinee, name)
                    || cases.iter().any(|c| {
                        c.label.as_ref().is_some_and(|l| expr_uses(l, name))
                            || c.body.iter().any(|x| stmt_uses(x, name))
                    })
            }
            _ => false,
        }
    }
    module.functions.iter().any(|f| f.body.iter().any(|s| stmt_uses(s, name)))
}

/// Whether any function body calls a maskbitcnt lowering shim
/// (`__maskbitcnt_u32` / `__maskbitcnt_hi_u32`), gating its emission in
/// `emit_intrinsic_shims`.
fn module_uses_maskbitcnt(module: &crate::ir::ShaderModule, shim: &str) -> bool {
    fn expr_uses_maskbitcnt(e: &crate::ir::IrExpr, shim: &str) -> bool {
        use crate::ir::IrExpr;
        match e {
            IrExpr::Call { function, arguments } => {
                if function == shim {
                    return true;
                }
                arguments.iter().any(|a| expr_uses_maskbitcnt(a, shim))
            }
            IrExpr::Variable(_) | IrExpr::Literal(_) => false,
            IrExpr::Member { object, .. } => expr_uses_maskbitcnt(object, shim),
            IrExpr::Index { object, index } | IrExpr::ResourceIndex { resource: object, index } => {
                expr_uses_maskbitcnt(object, shim) || expr_uses_maskbitcnt(index, shim)
            }
            IrExpr::ResourceLoad {
                resource,
                coordinates,
                sample,
            } => {
                expr_uses_maskbitcnt(resource, shim)
                    || expr_uses_maskbitcnt(coordinates, shim)
                    || sample.as_ref().is_some_and(|s| expr_uses_maskbitcnt(s, shim))
            }
            IrExpr::Construct { arguments, .. } => {
                arguments.iter().any(|a| expr_uses_maskbitcnt(a, shim))
            }
            IrExpr::Unary { expr: inner, .. } => expr_uses_maskbitcnt(inner, shim),
            IrExpr::Binary { left, right, .. } => {
                expr_uses_maskbitcnt(left, shim) || expr_uses_maskbitcnt(right, shim)
            }
            IrExpr::Conditional {
                condition,
                then_branch,
                else_branch,
            } => {
                expr_uses_maskbitcnt(condition, shim)
                    || expr_uses_maskbitcnt(then_branch, shim)
                    || expr_uses_maskbitcnt(else_branch, shim)
            }
            IrExpr::Assign { lhs, rhs } => {
                expr_uses_maskbitcnt(lhs, shim) || expr_uses_maskbitcnt(rhs, shim)
            }
            IrExpr::InitializerList(elems) => {
                elems.iter().any(|e| expr_uses_maskbitcnt(e, shim))
            }
        }
    }
    fn stmt_uses_maskbitcnt(s: &crate::ir::function::IrStmt, shim: &str) -> bool {
        use crate::ir::function::IrStmt;
        match s {
            IrStmt::Return(Some(e)) | IrStmt::Expr(e) => expr_uses_maskbitcnt(e, shim),
            IrStmt::Return(None) => false,
            IrStmt::VarDecl { init: Some(e), .. } => expr_uses_maskbitcnt(e, shim),
            IrStmt::VarDecl { init: None, .. } => false,
            IrStmt::If {
                condition,
                then_branch,
                else_branch,
            } => {
                expr_uses_maskbitcnt(condition, shim)
                    || then_branch.iter().any(|s| stmt_uses_maskbitcnt(s, shim))
                    || else_branch
                        .as_ref()
                        .is_some_and(|b| b.iter().any(|s| stmt_uses_maskbitcnt(s, shim)))
            }
            IrStmt::For {
                init,
                condition,
                update,
                body,
            } => {
                stmt_uses_maskbitcnt(init, shim)
                    || condition
                        .as_ref()
                        .is_some_and(|c| expr_uses_maskbitcnt(c, shim))
                    || update
                        .as_ref()
                        .is_some_and(|u| expr_uses_maskbitcnt(u, shim))
                    || body.iter().any(|s| stmt_uses_maskbitcnt(s, shim))
            }
            IrStmt::While { condition, body } => {
                expr_uses_maskbitcnt(condition, shim)
                    || body.iter().any(|s| stmt_uses_maskbitcnt(s, shim))
            }
            IrStmt::Switch { scrutinee, cases } => {
                expr_uses_maskbitcnt(scrutinee, shim)
                    || cases.iter().any(|c| {
                        c.label
                            .as_ref()
                            .is_some_and(|l| expr_uses_maskbitcnt(l, shim))
                            || c.body.iter().any(|s| stmt_uses_maskbitcnt(s, shim))
                    })
            }
            IrStmt::Block(ss) => ss.iter().any(|s| stmt_uses_maskbitcnt(s, shim)),
        }
    }
    module
        .functions
        .iter()
        .any(|f| f.body.iter().any(|s| stmt_uses_maskbitcnt(s, shim)))
}

/// Names of module globals assigned anywhere in-shader (mutable state).
/// A plain (non-`static`) HLSL global lives in the implicit `$Globals`
/// cbuffer and is const-qualified, so any in-shader assignment to it
/// fails DXC (`cannot assign to variable ... with const-qualified
/// type`); such globals must emit `static`. Only true assignments
/// count (initializers do not mutate): the root of every `Assign` lhs
/// (`x = ..`, `a[i] = ..`, `s.m = ..` all root at a name).
fn assigned_globals(module: &crate::ir::ShaderModule) -> std::collections::HashSet<String> {
    use crate::ir::IrExpr;
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
    fn expr_assigned(e: &IrExpr, out: &mut std::collections::HashSet<String>) {
        match e {
            IrExpr::Assign { lhs, rhs } => {
                if let Some(root) = assign_root(lhs) {
                    out.insert(root.to_string());
                }
                expr_assigned(lhs, out);
                expr_assigned(rhs, out);
            }
            IrExpr::Call { arguments, .. } | IrExpr::Construct { arguments, .. } => {
                for a in arguments {
                    expr_assigned(a, out);
                }
            }
            IrExpr::Member { object, .. } => expr_assigned(object, out),
            IrExpr::Index { object, index } | IrExpr::ResourceIndex { resource: object, index } => {
                expr_assigned(object, out);
                expr_assigned(index, out);
            }
            IrExpr::ResourceLoad {
                resource,
                coordinates,
                sample,
            } => {
                expr_assigned(resource, out);
                expr_assigned(coordinates, out);
                if let Some(sample) = sample {
                    expr_assigned(sample, out);
                }
            }
            IrExpr::Unary { expr: inner, .. } => expr_assigned(inner, out),
            IrExpr::Binary { left, right, .. } => {
                expr_assigned(left, out);
                expr_assigned(right, out);
            }
            IrExpr::Conditional {
                condition,
                then_branch,
                else_branch,
            } => {
                expr_assigned(condition, out);
                expr_assigned(then_branch, out);
                expr_assigned(else_branch, out);
            }
            IrExpr::InitializerList(elems) => {
                for x in elems {
                    expr_assigned(x, out);
                }
            }
            IrExpr::Variable(_) | IrExpr::Literal(_) => {}
        }
    }
    fn stmt_assigned(s: &crate::ir::function::IrStmt, out: &mut std::collections::HashSet<String>) {
        use crate::ir::function::IrStmt;
        match s {
            IrStmt::Return(Some(e)) | IrStmt::Expr(e) => expr_assigned(e, out),
            IrStmt::Return(None) => {}
            IrStmt::VarDecl { init: Some(e), .. } => expr_assigned(e, out),
            IrStmt::VarDecl { init: None, .. } => {}
            IrStmt::If {
                condition,
                then_branch,
                else_branch,
            } => {
                expr_assigned(condition, out);
                for s in then_branch {
                    stmt_assigned(s, out);
                }
                if let Some(else_branch) = else_branch {
                    for s in else_branch {
                        stmt_assigned(s, out);
                    }
                }
            }
            IrStmt::For {
                init,
                condition,
                update,
                body,
            } => {
                stmt_assigned(init, out);
                if let Some(condition) = condition {
                    expr_assigned(condition, out);
                }
                if let Some(update) = update {
                    expr_assigned(update, out);
                }
                for s in body {
                    stmt_assigned(s, out);
                }
            }
            IrStmt::While { condition, body } => {
                expr_assigned(condition, out);
                for s in body {
                    stmt_assigned(s, out);
                }
            }
            IrStmt::Switch { scrutinee, cases } => {
                expr_assigned(scrutinee, out);
                for c in cases {
                    if let Some(label) = &c.label {
                        expr_assigned(label, out);
                    }
                    for s in &c.body {
                        stmt_assigned(s, out);
                    }
                }
            }
            IrStmt::Block(ss) => {
                for s in ss {
                    stmt_assigned(s, out);
                }
            }
        }
    }
    let mut out = std::collections::HashSet::new();
    for f in &module.functions {
        for s in &f.body {
            stmt_assigned(s, &mut out);
        }
    }
    out
}

fn stmt_uses_m_pi(s: &crate::ir::function::IrStmt) -> bool {
    use crate::ir::function::IrStmt;
    match s {
        IrStmt::Return(Some(e)) => expr_uses_m_pi(e),
        IrStmt::Return(None) => false,
        IrStmt::VarDecl { init: Some(e), .. } => expr_uses_m_pi(e),
        IrStmt::VarDecl { init: None, .. } => false,
        IrStmt::If {
            condition,
            then_branch,
            else_branch,
        } => {
            if expr_uses_m_pi(condition) {
                return true;
            }
            for ss in then_branch {
                if stmt_uses_m_pi(ss) {
                    return true;
                }
            }
            if let Some(eb) = else_branch {
                for ss in eb {
                    if stmt_uses_m_pi(ss) {
                        return true;
                    }
                }
            }
            false
        }
        IrStmt::For {
            init,
            condition,
            update,
            body,
        } => {
            if stmt_uses_m_pi(init) {
                return true;
            }
            if let Some(c) = condition {
                if expr_uses_m_pi(c) {
                    return true;
                }
            }
            if let Some(u) = update {
                if expr_uses_m_pi(u) {
                    return true;
                }
            }
            for ss in body {
                if stmt_uses_m_pi(ss) {
                    return true;
                }
            }
            false
        }
        IrStmt::While { condition, body } => {
            if expr_uses_m_pi(condition) {
                return true;
            }
            for ss in body {
                if stmt_uses_m_pi(ss) {
                    return true;
                }
            }
            false
        }
        IrStmt::Switch { scrutinee, cases } => {
            if expr_uses_m_pi(scrutinee) {
                return true;
            }
            for case in cases {
                if case.label.as_ref().is_some_and(expr_uses_m_pi) {
                    return true;
                }
                for ss in &case.body {
                    if stmt_uses_m_pi(ss) {
                        return true;
                    }
                }
            }
            false
        }
        IrStmt::Expr(e) => expr_uses_m_pi(e),
        IrStmt::Block(ss) => ss.iter().any(|x| stmt_uses_m_pi(x)),
    }
}

fn expr_uses_m_pi(e: &crate::ir::IrExpr) -> bool {
    match e {
        crate::ir::IrExpr::Variable(n) => n == "M_PI",
        crate::ir::IrExpr::Member { object, .. } => expr_uses_m_pi(object),
        crate::ir::IrExpr::Index { object, index } => expr_uses_m_pi(object) || expr_uses_m_pi(index),
        crate::ir::IrExpr::ResourceIndex { resource, index } => {
            expr_uses_m_pi(resource) || expr_uses_m_pi(index)
        }
        crate::ir::IrExpr::ResourceLoad {
            resource,
            coordinates,
            sample,
        } => {
            expr_uses_m_pi(resource)
                || expr_uses_m_pi(coordinates)
                || sample.as_ref().map_or(false, |s| expr_uses_m_pi(s))
        }
        crate::ir::IrExpr::Call { arguments, .. } => arguments.iter().any(|a| expr_uses_m_pi(a)),
        crate::ir::IrExpr::Construct { arguments, .. } => arguments.iter().any(|a| expr_uses_m_pi(a)),
        crate::ir::IrExpr::Unary { expr, .. } => expr_uses_m_pi(expr),
        crate::ir::IrExpr::Binary { left, right, .. } => expr_uses_m_pi(left) || expr_uses_m_pi(right),
        crate::ir::IrExpr::Conditional {
            condition,
            then_branch,
            else_branch,
        } => expr_uses_m_pi(condition) || expr_uses_m_pi(then_branch) || expr_uses_m_pi(else_branch),
        crate::ir::IrExpr::Assign { lhs, rhs } => expr_uses_m_pi(lhs) || expr_uses_m_pi(rhs),
        crate::ir::IrExpr::InitializerList(ee) => ee.iter().any(|a| expr_uses_m_pi(a)),
        crate::ir::IrExpr::Literal(_) => false,
    }
}

fn emit_resource(
    out: &mut String,
    resource: &Resource,
    assigned: &std::collections::HashSet<String>,
) {
    let register_suffix = resource
        .binding
        .map(|binding| {
            let prefix = match resource.resource_type {
                ResourceType::Texture => "t",
                ResourceType::Sampler => "s",
                ResourceType::ConstantBuffer => "b",
                ResourceType::StructuredBuffer => "t",
                ResourceType::RWBuffer | ResourceType::RWTexture => "u",
            };
            format!(" : register({prefix}{binding})")
        })
        .unwrap_or_default();

    // Resource member arrays keep their dimension on the flattened
    // binding (`RW_RegularBuffer<uint> bufs[2]` →
    // `RWStructuredBuffer<uint> bufs[2]`), so indexed uses stay valid.
    // Unbounded (`Some(0)`, hoisted texture arrays) emits `[]`.
    let array_suffix = resource
        .array_size
        .map(|n| {
            if n == 0 {
                "[]".to_string()
            } else {
                format!("[{n}]")
            }
        })
        .unwrap_or_default();

    match resource.resource_type {
        ResourceType::ConstantBuffer => {
            // Cbuffer members assigned in-shader (`m_clearColor = ...`
            // in clear shaders: the cbuffer acts as mutable scratch, not
            // host constants) cannot stay in a cbuffer (const-qualified):
            // they emit as `static` globals while untouched members keep
            // the cbuffer (host layout preserved for those). Shaders
            // without in-shader cbuffer stores are untouched.
            let (mut_fields, const_fields): (Vec<_>, Vec<_>) = resource
                .fields
                .iter()
                .partition(|f| assigned.contains(f.name.as_str()));
            for field in &mut_fields {
                let array = field
                    .array_size
                    .map(|n| if n == 0 { "[]".to_string() } else { format!("[{n}]") })
                    .unwrap_or_default();
                let _ = writeln!(
                    out,
                    "static {} {}{};",
                    hlsl_type(&field.ty),
                    hlsl_ident(&field.name),
                    array
                );
            }
            if const_fields.is_empty() {
                return;
            }
            let _ = writeln!(out, "cbuffer {}{} {{", resource.name, register_suffix);
            for field in const_fields {
                let array = field
                    .array_size
                    .map(|n| if n == 0 { "[]".to_string() } else { format!("[{n}]") })
                    .unwrap_or_default();
                let _ = writeln!(
                out,
                "    {} {}{};",
                hlsl_type(&field.ty),
                hlsl_ident(&field.name),
                array
            );
            }
            out.push_str("};\n");
        }
        ResourceType::Texture => {
            let element = resource
                .element_type
                .as_ref()
                .map(|ty| format!("<{}>", hlsl_type(ty)))
                .unwrap_or_default();
            let kind_name = resource_kind_name(&resource.kind);
            let _ = writeln!(
                out,
                "{kind_name}{element} {}{array_suffix}{register_suffix};",
                hlsl_ident(&resource.name)
            );
        }
        ResourceType::Sampler => {
            // Comparison samplers stay comparison: `SampleCmp*` requires
            // `SamplerComparisonState` (e.g. `static const
            // SamplerComparisonState shadow_sampler` for shadow filtering).
            // Plain samplers keep `SamplerState`; both are externally bound.
            let kind_name = if resource.kind.eq_ignore_ascii_case("samplercomparisonstate") {
                "SamplerComparisonState"
            } else {
                "SamplerState"
            };
            let _ = writeln!(
                out,
                "{kind_name} {}{array_suffix}{register_suffix};",
                hlsl_ident(&resource.name)
            );
        }
        ResourceType::StructuredBuffer => {
            // Buffer-of-textures (`RegularBuffer<Texture2D> textures`):
            // `StructuredBuffer<Texture2D>` is illegal HLSL (resources
            // cannot nest). The correct representation is an unbounded
            // resource array: `Texture2D textures[]`.
            if let Some(elem) = resource.element_type.as_ref() {
                if is_texture_element(elem) {
                    let _ = writeln!(
                        out,
                        "{} {}[]{register_suffix};",
                        hlsl_type(elem),
                        hlsl_ident(&resource.name)
                    );
                    return;
                }
            }
            let kind_name = resource_kind_name(&resource.kind);
            let element = if is_byte_address_kind(&resource.kind) {
                String::new()
            } else {
                resource
                    .element_type
                    .as_ref()
                    .map(|ty| format!("<{}>", hlsl_type(ty)))
                    .unwrap_or_default()
            };
            let _ = writeln!(
                out,
                "{kind_name}{element} {}{array_suffix}{register_suffix};",
                hlsl_ident(&resource.name)
            );
        }
        ResourceType::RWBuffer => {
            let kind_name = resource_kind_name(&resource.kind);
            let element = if is_byte_address_kind(&resource.kind) {
                String::new()
            } else {
                resource
                    .element_type
                    .as_ref()
                    .map(|ty| format!("<{}>", hlsl_type(ty)))
                    .unwrap_or_default()
            };
            let _ = writeln!(
                out,
                "{kind_name}{element} {}{array_suffix}{register_suffix};",
                hlsl_ident(&resource.name)
            );
        }
        ResourceType::RWTexture => {
            let element = resource
                .element_type
                .as_ref()
                .map(|ty| format!("<{}>", hlsl_type(ty)))
                .unwrap_or_default();
            let kind_name = resource_kind_name(&resource.kind);
            let _ = writeln!(
                out,
                "{kind_name}{element} {}{array_suffix}{register_suffix};",
                hlsl_ident(&resource.name)
            );
        }
    }
}

/// Methods to emit `static`: called as `S::m` (statically) while using
/// no instance state, so HLSL resolves the call natively (DXC rejects
/// `call to non-static member function without an object argument`).
/// Stateful methods (bare uses of struct fields) keep today's loud path.
/// Sibling this-calls propagate through a fixpoint: a static method may
/// only call static siblings.
fn static_methods(
    module: &crate::ir::ShaderModule,
) -> std::collections::HashSet<(String, String)> {
    use crate::ir::{IrExpr, IrStmt};
    fn vars_in_expr(e: &IrExpr, out: &mut std::collections::HashSet<String>) {
        match e {
            IrExpr::Variable(n) => {
                out.insert(n.clone());
            }
            IrExpr::Call { arguments, .. } => {
                for a in arguments {
                    vars_in_expr(a, out);
                }
            }
            IrExpr::Construct { arguments, .. } => {
                for a in arguments {
                    vars_in_expr(a, out);
                }
            }
            IrExpr::Member { object, .. } => vars_in_expr(object, out),
            IrExpr::Index { object, index } | IrExpr::ResourceIndex { resource: object, index } => {
                vars_in_expr(object, out);
                vars_in_expr(index, out);
            }
            IrExpr::ResourceLoad { resource, coordinates, sample } => {
                vars_in_expr(resource, out);
                vars_in_expr(coordinates, out);
                if let Some(s) = sample {
                    vars_in_expr(s, out);
                }
            }
            IrExpr::Assign { lhs, rhs } => {
                vars_in_expr(lhs, out);
                vars_in_expr(rhs, out);
            }
            IrExpr::Unary { expr, .. } => vars_in_expr(expr, out),
            IrExpr::Binary { left, right, .. } => {
                vars_in_expr(left, out);
                vars_in_expr(right, out);
            }
            IrExpr::Conditional { condition, then_branch, else_branch } => {
                vars_in_expr(condition, out);
                vars_in_expr(then_branch, out);
                vars_in_expr(else_branch, out);
            }
            IrExpr::InitializerList(elems) => {
                for x in elems {
                    vars_in_expr(x, out);
                }
            }
            IrExpr::Literal(_) => {}
            _ => {}
        }
    }
    fn vars_in_stmt(
        s: &IrStmt,
        out: &mut std::collections::HashSet<String>,
        locals: &mut std::collections::HashSet<String>,
    ) {
        match s {
            IrStmt::Return(Some(e)) | IrStmt::Expr(e) => vars_in_expr(e, out),
            IrStmt::Return(None) => {}
            IrStmt::VarDecl { name, init, .. } => {
                locals.insert(name.clone());
                if let Some(e) = init {
                    vars_in_expr(e, out);
                }
            }
            IrStmt::Block(ss) => {
                for x in ss {
                    vars_in_stmt(x, out, locals);
                }
            }
            IrStmt::If { condition, then_branch, else_branch } => {
                vars_in_expr(condition, out);
                for x in then_branch.iter().chain(else_branch.iter().flatten()) {
                    vars_in_stmt(x, out, locals);
                }
            }
            IrStmt::For { init, condition, update, body } => {
                vars_in_stmt(init, out, locals);
                if let Some(c) = condition {
                    vars_in_expr(c, out);
                }
                if let Some(u) = update {
                    vars_in_expr(u, out);
                }
                for x in body {
                    vars_in_stmt(x, out, locals);
                }
            }
            IrStmt::While { condition, body } => {
                vars_in_expr(condition, out);
                for x in body {
                    vars_in_stmt(x, out, locals);
                }
            }
            IrStmt::Switch { scrutinee, cases } => {
                vars_in_expr(scrutinee, out);
                for c in cases {
                    if let Some(l) = &c.label {
                        vars_in_expr(l, out);
                    }
                    for x in &c.body {
                        vars_in_stmt(x, out, locals);
                    }
                }
            }
            _ => {}
        }
    }
    fn calls_in_expr(e: &IrExpr, out: &mut std::collections::HashSet<String>) {
        match e {
            IrExpr::Call { function, arguments } => {
                out.insert(function.clone());
                for a in arguments {
                    calls_in_expr(a, out);
                }
            }
            IrExpr::Member { object, .. } => calls_in_expr(object, out),
            IrExpr::Index { object, index } | IrExpr::ResourceIndex { resource: object, index } => {
                calls_in_expr(object, out);
                calls_in_expr(index, out);
            }
            IrExpr::ResourceLoad { resource, coordinates, sample } => {
                calls_in_expr(resource, out);
                calls_in_expr(coordinates, out);
                if let Some(s) = sample {
                    calls_in_expr(s, out);
                }
            }
            IrExpr::Construct { arguments, .. } => {
                for a in arguments {
                    calls_in_expr(a, out);
                }
            }
            IrExpr::Assign { lhs, rhs } => {
                calls_in_expr(lhs, out);
                calls_in_expr(rhs, out);
            }
            IrExpr::Unary { expr, .. } => calls_in_expr(expr, out),
            IrExpr::Binary { left, right, .. } => {
                calls_in_expr(left, out);
                calls_in_expr(right, out);
            }
            IrExpr::Conditional { condition, then_branch, else_branch } => {
                calls_in_expr(condition, out);
                calls_in_expr(then_branch, out);
                calls_in_expr(else_branch, out);
            }
            IrExpr::InitializerList(elems) => {
                for x in elems {
                    calls_in_expr(x, out);
                }
            }
            IrExpr::Variable(_) | IrExpr::Literal(_) => {}
            _ => {}
        }
    }
    fn calls_in_stmt(s: &IrStmt, out: &mut std::collections::HashSet<String>) {
        match s {
            IrStmt::Return(Some(e)) | IrStmt::Expr(e) => calls_in_expr(e, out),
            IrStmt::Return(None) => {}
            IrStmt::VarDecl { init: Some(e), .. } => calls_in_expr(e, out),
            IrStmt::VarDecl { init: None, .. } => {}
            IrStmt::Block(ss) => {
                for x in ss {
                    calls_in_stmt(x, out);
                }
            }
            IrStmt::If { condition, then_branch, else_branch } => {
                calls_in_expr(condition, out);
                for x in then_branch.iter().chain(else_branch.iter().flatten()) {
                    calls_in_stmt(x, out);
                }
            }
            IrStmt::For { init, condition, update, body } => {
                calls_in_stmt(init, out);
                if let Some(c) = condition {
                    calls_in_expr(c, out);
                }
                if let Some(u) = update {
                    calls_in_expr(u, out);
                }
                for x in body {
                    calls_in_stmt(x, out);
                }
            }
            IrStmt::While { condition, body } => {
                calls_in_expr(condition, out);
                for x in body {
                    calls_in_stmt(x, out);
                }
            }
            IrStmt::Switch { scrutinee, cases } => {
                calls_in_expr(scrutinee, out);
                for c in cases {
                    if let Some(l) = &c.label {
                        calls_in_expr(l, out);
                    }
                    for x in &c.body {
                        calls_in_stmt(x, out);
                    }
                }
            }
            _ => {}
        }
    }
    // Struct -> methods and fields.
    let mut methods_of: std::collections::HashMap<&str, Vec<&Function>> =
        std::collections::HashMap::new();
    for f in &module.functions {
        if let Some(scope) = f.scope.as_deref() {
            methods_of.entry(scope).or_default().push(f);
        }
    }
    let mut fields_of: std::collections::HashMap<&str, std::collections::HashSet<&str>> =
        std::collections::HashMap::new();
    for s in &module.structs {
        fields_of.insert(
            s.name.as_str(),
            s.fields.iter().map(|f| f.name.as_str()).collect(),
        );
    }
    let globals: std::collections::HashSet<&str> =
        module.globals.iter().map(|g| g.name.as_str()).collect();
    let resources: std::collections::HashSet<&str> = module
        .resources
        .iter()
        .map(|r| r.name.as_str())
        .collect();
    let functions: std::collections::HashSet<&str> =
        module.functions.iter().map(|f| f.name.as_str()).collect();
    let structs: std::collections::HashSet<&str> =
        module.structs.iter().map(|s| s.name.as_str()).collect();
    // Bare `S::m` call targets anywhere (methods and free functions).
    let mut static_calls: std::collections::HashSet<(String, String)> =
        std::collections::HashSet::new();
    for f in &module.functions {
        let mut calls = std::collections::HashSet::new();
        for s in &f.body {
            calls_in_stmt(s, &mut calls);
        }
        for c in calls {
            if let Some((scope, method)) = c.split_once("::") {
                if methods_of.contains_key(scope)
                    && methods_of[scope].iter().any(|m| m.name == method)
                {
                    static_calls.insert((scope.to_string(), method.to_string()));
                }
            }
        }
    }
    // Instance-clean methods among the statically called.
    let mut candidates: std::collections::HashSet<(String, String)> =
        std::collections::HashSet::new();
    // (struct, method) -> sibling bare-call targets.
    let mut siblings: std::collections::HashMap<(String, String), Vec<String>> =
        std::collections::HashMap::new();
    for ((scope, method), _) in static_calls.iter().map(|t| (t, ())) {
        let Some(ms) = methods_of.get(scope.as_str()) else {
            continue;
        };
        let Some(m) = ms.iter().find(|m| &m.name == method) else {
            continue;
        };
        let mut vars = std::collections::HashSet::new();
        let mut locals = std::collections::HashSet::new();
        for s in &m.body {
            vars_in_stmt(s, &mut vars, &mut locals);
        }
        let params: std::collections::HashSet<&str> =
            m.parameters.iter().map(|p| p.name.as_str()).collect();
        let fields = fields_of.get(scope.as_str());
        let uses_instance = vars.iter().any(|v| {
            !params.contains(v.as_str())
                && !locals.contains(v)
                && !globals.contains(v.as_str())
                && !resources.contains(v.as_str())
                && !functions.contains(v.as_str())
                && !structs.contains(v.as_str())
                && fields.is_some_and(|fs| fs.contains(v.as_str()))
        });
        if uses_instance {
            continue;
        }
        let mut calls = std::collections::HashSet::new();
        for s in &m.body {
            calls_in_stmt(s, &mut calls);
        }
        let sibs: Vec<String> = calls
            .into_iter()
            .filter(|c| !c.contains("::") && !c.contains('.'))
            .filter(|c| {
                methods_of
                    .get(scope.as_str())
                    .is_some_and(|ms| ms.iter().any(|m| &m.name == c))
            })
            .collect();
        siblings.insert((scope.clone(), method.clone()), sibs);
        candidates.insert((scope.clone(), method.clone()));
    }
    // Fixpoint: drop candidates calling non-static siblings.
    loop {
        let drop: Vec<(String, String)> = candidates
            .iter()
            .filter(|(scope, method)| {
                siblings
                    .get(&(scope.clone(), method.clone()))
                    .is_some_and(|sibs| {
                        sibs.iter().any(|s| {
                            !candidates.contains(&(scope.clone(), s.clone()))
                        })
                    })
            })
            .cloned()
            .collect();
        if drop.is_empty() {
            break;
        }
        for d in drop {
            candidates.remove(&d);
        }
    }
    candidates
}

fn emit_struct(
    out: &mut String,
    structure: &StructDecl,
    declared: &std::collections::HashSet<&str>,
    methods: &[&Function],
    structs: &[StructDecl],
    statics: &std::collections::HashSet<(String, String)>,
) {
    let _ = writeln!(out, "struct {} {{", structure.name);
    // TEXCOORDn collision resolution: several PSSL spellings map into the
    // shared TEXCOORDn family (`COLOR` → `TEXCOORD0`, bare `TEXCOORD` →
    // `TEXCOORD0`), so one struct can map two fields to the same semantic
    // (DXC `input semantic ... used more than once`). Bump later fields to
    // the next free TEXCOORD index. Other families pass through untouched:
    // duplicate system semantics (two `SV_Position` inputs) are genuinely
    // invalid and stay loud.
    let mut used_semantics: std::collections::HashSet<String> =
        std::collections::HashSet::new();
    for field in &structure.fields {
        let mut semantic = field.semantic.as_deref().and_then(hlsl_semantic);
        if let Some(s) = semantic.as_ref() {
            if used_semantics.contains(s) {
                let upper = s.to_ascii_uppercase();
                if let Some(rest) = upper.strip_prefix("TEXCOORD") {
                    if rest.parse::<u32>().is_ok() {
                        let mut k = 0u32;
                        while used_semantics.contains(&format!("TEXCOORD{k}")) {
                            k += 1;
                        }
                        semantic = Some(format!("TEXCOORD{k}"));
                    }
                }
            }
        }
        let semantic = semantic.map(|s| {
            used_semantics.insert(s.clone());
            format!(" : {}", s)
        }).unwrap_or_default();
        let array = field
            .array_size
            .map(|n| if n == 0 { "[]".to_string() } else { format!("[{n}]") })
            .unwrap_or_default();
        // Enum-typed fields (a Named type that is not a declared struct and is a
// plain PSSL enum, e.g. `DrawMode`/`BvhType`) emit as `int` — PSSL enums
// are int-backed, and DXC rejects the raw enum name (`unknown type name
// 'DrawMode'`). `sce::`-qualified names are framework STRUCT types (not
// enums) and are excluded: they must be handled by framework mapping, not
// flattened to `int`. Sampler types (`SamplerState`, including framework
// names mapped to it) are excluded too: flattening them to `int` breaks
// sampler assignments through the struct (e.g. builtin-sampler tables).
// Bare texture spellings (`Texture2D tex`, no element) are GPU resources,
// not int-backed enums: they emit verbatim (HLSL's own float4 default,
// matching use sites like `tex.Sample(smp, uv)`). Same spelling list as
// the SRT walk and `is_texture_element` below.
        let fty = match &field.ty {
            IrType::Named(n)
                if !declared.contains(n.as_str())
                    && !n.starts_with("sce::")
                    && !crate::framework::is_sampler_type_name(n)
                    && !matches!(
                        n.to_ascii_lowercase().as_str(),
                        "texture1d" | "texture2d" | "texturecube" | "texture3d"
                    ) =>
            {
                "int".to_string()
            }
            _ => hlsl_type(&field.ty),
        };
        let _ = writeln!(
            out,
            "    {} {}{}{};",
            fty,
            hlsl_ident(&field.name),
            array,
            semantic
        );
    }
        // Same struct parsed from re-inlined headers can carry duplicate
    // method definitions: emit each signature once (DXC rejects
    // redefinition, but overloading by arity/types is legal and kept).
    let mut emitted_methods: std::collections::HashSet<String> =
        std::collections::HashSet::new();
    for method in methods {
        let sig = format!(
            "{}({})",
            method.name,
            method
                .parameters
                .iter()
                .map(|p| hlsl_type(&p.ty))
                .collect::<Vec<_>>()
                .join(",")
        );
        if !emitted_methods.insert(sig) {
            continue;
        }
        // Prototyped methods (declared `;` with no body — psr headers are
        // full of them, e.g. `BvhNode::getVertex`) cannot emit as `{}`:
        // a non-void `{}` is a hard DXC error (`control reaches end of
        // non-void function`), while dropping it is exactly right —
        // uncalled methods vanish silently, called ones fail loudly as
        // unresolved instead of miscompiling. Void `{}` stays (legal).
        if method.body.is_empty() && method.return_type != crate::ir::IrType::Void {
            continue;
        }
        let is_static = statics.contains(&(structure.name.clone(), method.name.clone()));
        emit_method(out, method, structs, is_static);
    }
    out.push_str("};\n");
}

/// Emits a vector-condition ternary in a `T x = (C ? A : B);` declaration
/// as per-component assigns (`T x; x.x = c0 ? A.x : B.x; …`). `C` must be
/// a bool-vector constructor with one scalar arg per component; branch
/// accesses take `.x/.y/.z/.w` except for scalar literals, which replicate
/// unchanged. Non-bool condition args normalize via `!= 0` (PSSL ints as
/// conditions mean nonzero); already-bool shapes (comparisons, logical
/// ops, `!`, bool literals) pass through. See the `VarDecl` arm.
fn emit_vector_ternary(
    out: &mut String,
    indent: usize,
    ty: &IrType,
    name: &str,
    width: u8,
    cargs: &[IrExpr],
    then_branch: &IrExpr,
    else_branch: &IrExpr,
) {
    const COMPONENTS: [&str; 4] = ["x", "y", "z", "w"];
    fn is_bool_expr(e: &IrExpr) -> bool {
        match e {
            IrExpr::Literal(IrLiteral::Bool(_)) => true,
            IrExpr::Binary { op, .. } => matches!(
                op,
                IrOp::Equal
                    | IrOp::NotEqual
                    | IrOp::Less
                    | IrOp::LessEqual
                    | IrOp::Greater
                    | IrOp::GreaterEqual
                    | IrOp::LogicalAnd
                    | IrOp::LogicalOr
            ),
            IrExpr::Unary { op, .. } => matches!(op, IrOp::Not),
            _ => false,
        }
    }
    fn component(base: &IrExpr, suffix: &str) -> IrExpr {
        // Scalar literals replicate per component; everything else takes
        // a component access (DXC verifies vector-ness).
        if matches!(base, IrExpr::Literal(_)) {
            base.clone()
        } else {
            IrExpr::Member {
                object: Box::new(base.clone()),
                member: suffix.to_string(),
            }
        }
    }
    let pad = "    ".repeat(indent);
    let var = hlsl_ident(name);
    let _ = writeln!(out, "{pad}{} {};", hlsl_type(ty), var);
    for (i, carg) in cargs.iter().enumerate().take(width as usize) {
        let suffix = COMPONENTS[i];
        let cond = if is_bool_expr(carg) {
            emit_expr(carg)
        } else {
            emit_expr(&IrExpr::Binary {
                left: Box::new(carg.clone()),
                op: IrOp::NotEqual,
                right: Box::new(IrExpr::Literal(IrLiteral::Number("0".to_string()))),
            })
        };
        let take = emit_expr(&component(then_branch, suffix));
        let skip = emit_expr(&component(else_branch, suffix));
        let _ = writeln!(out, "{pad}{var}.{suffix} = {cond} ? {take} : {skip};");
    }
}

/// Emits a struct brace-init as memberwise assignment (`S s = {a, b};` →
/// `S s; s.f = a; s.g = b;`), zero-filling missing trailing numeric
/// members. See the `VarDecl` arm of `emit_stmt`.
fn emit_struct_init(
    out: &mut String,
    indent: usize,
    ty: &IrType,
    name: &str,
    sdecl: &StructDecl,
    elems: &[IrExpr],
    structs: &[StructDecl],
) {
    let pad = "    ".repeat(indent);
    let var = hlsl_ident(name);
    // Full inits emit direct brace construction (`S x = {a, b};`), which
    // initializes `const` members correctly (unlike memberwise assignment
    // expansion, which cannot assign consts). Values identical to the
    // expanded form. Partial inits keep memberwise assigns + zero-fill
    // (brace-init zero-fill semantics vary, so stay explicit there).
    if elems.len() == sdecl.fields.len() {
        let inner = elems.iter().map(emit_expr).collect::<Vec<_>>().join(", ");
        let _ = writeln!(out, "{pad}{} {} = {{ {inner} }};", hlsl_type(ty), var);
        return;
    }
    let _ = writeln!(out, "{pad}{} {};", hlsl_type(ty), var);
    for (i, field) in sdecl.fields.iter().enumerate() {
        let target = format!("{}.{}", var, hlsl_ident(&field.name));
        if let Some(init) = elems.get(i) {
            let _ = writeln!(out, "{pad}{target} = {};", emit_expr(init));
        } else {
            emit_zero_assign(out, &pad, &target, &field.ty, structs);
        }
    }
}

/// Emits `target = 0;` for numeric types (scalar/vector/matrix — HLSL
/// splats the scalar), recursing into known structs. Resource and unknown
/// members are left unassigned: nothing valid can initialize them here.
fn emit_zero_assign(
    out: &mut String,
    pad: &str,
    target: &str,
    ty: &IrType,
    structs: &[StructDecl],
) {
    match ty {
        IrType::Int
        | IrType::UInt
        | IrType::Float
        | IrType::Half
        | IrType::Bool
        | IrType::Double
        | IrType::Vector { .. }
        | IrType::Matrix { .. } => {
            let _ = writeln!(out, "{pad}{target} = 0;");
        }
        IrType::Named(n) => {
            if let Some(sdecl) = structs.iter().find(|s| &s.name == n) {
                for field in &sdecl.fields {
                    let sub = format!("{target}.{}", hlsl_ident(&field.name));
                    emit_zero_assign(out, pad, &sub, &field.ty, structs);
                }
            }
        }
        _ => {}
    }
}

fn emit_function(out: &mut String, function: &Function, structs: &[StructDecl]) {
    emit_function_signature(out, function);

    out.push_str("\n{\n");

    for statement in &function.body {
        emit_stmt(out, statement, 1, structs);
    }

    out.push_str("}\n");
}

/// Emits a free-function signature (`[numthreads(..)] Ret name(params) [:]
/// return-semantic`) shared by definitions and prototypes.
fn emit_function_signature(out: &mut String, function: &Function) {
    if let Some((x, y, z)) = function.num_threads {
        let _ = writeln!(out, "[numthreads({x}, {y}, {z})]");
    }
    if let Some(n) = function.max_vertex_count {
        let _ = writeln!(out, "[maxvertexcount({n})]");
    }
    if let Some(t) = function.output_topology.as_deref() {
        let _ = writeln!(out, "[outputtopology(\"{t}\")]");
    }
    // Hull-shader tessellation state (from the PSSL `*_PATCH_TYPE` /
    // `OUTPUT_CONTROL_POINTS` / `PATCH_CONSTANT_FUNC` / `MAX_TESS_FACTOR`
    // attributes): emitted exactly as written; absent entries keep
    // today's loud path at DXC (hs_6_0 requires them).
    if let Some(d) = function.hull_domain.as_deref() {
        let _ = writeln!(out, "[domain(\"{d}\")]");
    }
    if let Some(p) = function.hull_partitioning.as_deref() {
        let _ = writeln!(out, "[partitioning(\"{p}\")]");
    }
    if let Some(n) = function.hull_control_points {
        let _ = writeln!(out, "[outputcontrolpoints({n})]");
    }
    if let Some(f) = function.hull_patch_func.as_deref() {
        let _ = writeln!(out, "[patchconstantfunc(\"{f}\")]");
    }
    if let Some(n) = function.hull_max_tess {
        let _ = writeln!(out, "[maxtessfactor({n}.0)]");
    }
    let _ = write!(
        out,
        "{} {}(",
        hlsl_type(&function.return_type),
        function.name
    );

    for (index, parameter) in function.parameters.iter().enumerate() {
        if index != 0 {
            out.push_str(", ");
        }
        // Mesh-shader output objects (`out vertices T[n]`, `out indices
        // uint3[n]`): object kind + element type + name + size, no
        // direction/primitive/semantic decorations.
        if let Some(kind) = parameter.mesh_object.as_deref() {
            let array = parameter
                .array_size
                .map(|n| if n == 0 { "[]".to_string() } else { format!("[{n}]") })
                .unwrap_or_default();
            let _ = write!(
                out,
                "out {} {} {}{}",
                kind,
                hlsl_type(&parameter.ty),
                hlsl_ident(&parameter.name),
                array
            );
            continue;
        }
        // GS direction (`inout`) and input primitive (`point`, …) precede
        // the type, HLSL order.
        let direction = parameter
            .direction
            .as_deref()
            .map(|d| format!("{d} "))
            .unwrap_or_default();
        let primitive = parameter
            .gs_primitive
            .as_deref()
            .map(|p| format!("{p} "))
            .unwrap_or_default();
        let semantic = parameter
            .semantic
            .as_deref()
            .and_then(hlsl_semantic)
            .map(|s| format!(" : {}", s))
            .unwrap_or_default();
        let array = parameter
            .array_size
            .map(|n| if n == 0 { "[]".to_string() } else { format!("[{n}]") })
            .unwrap_or_default();
        let _ = write!(
            out,
            "{direction}{primitive}{} {}{}{}",
            hlsl_type(&parameter.ty),
            hlsl_ident(&parameter.name),
            array,
            semantic
        );
    }

    out.push(')');

    if let Some(semantic) = &function.return_semantic {
        if let Some(mapped) = hlsl_semantic(semantic) {
            let _ = write!(out, " : {}", mapped);
        }
    }
}

/// Emits `;`-terminated prototypes for free functions so struct methods
/// (which see the whole class scope in C++) can call helpers defined
/// later in the module — HLSL needs declarations before use. Runs in two
/// phases around struct emission (prototypes may name struct types):
/// `builtin_sigs=true` emits only signatures without user-struct types
/// (before structs), `false` the rest (after). Generic-`T` functions are
/// skipped in both (concrete overloads below resolve those calls).
fn emit_prototypes(
    out: &mut String,
    module: &crate::ir::ShaderModule,
    declared: &std::collections::HashSet<&str>,
    builtin_sigs: bool,
) {
    fn mentions_struct(ty: &IrType, declared: &std::collections::HashSet<&str>) -> bool {
        match ty {
            IrType::Named(n) => declared.contains(n.as_str()),
            IrType::Resource { element, .. } => mentions_struct(element, declared),
            _ => false,
        }
    }
    let mut emitted: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut any = false;
    for function in &module.functions {
        // Methods live in structs, not here; undeclared-scope functions
        // keep free behavior below and need prototypes like the rest.
        if let Some(scope) = function.scope.as_deref() {
            if declared.contains(scope) {
                continue;
            }
        }
        if is_generic_t_function(function) {
            continue;
        }
        // Non-deduced `auto` returns cannot prototype (HLSL rejects `auto`
        // in declarations); the definition stays loud on its own.
        if matches!(&function.return_type, IrType::Named(n) if n == "auto") {
            continue;
        }
        let uses_struct = mentions_struct(&function.return_type, declared)
            || function.parameters.iter().any(|p| mentions_struct(&p.ty, declared));
        if uses_struct == builtin_sigs {
            continue;
        }
        // Deduplicate identical signatures (re-inlined headers); overloads
        // differ and are all kept.
        let mut sig = String::new();
        emit_function_signature(&mut sig, function);
        if !emitted.insert(sig.clone()) {
            continue;
        }
        out.push_str(&sig);
        out.push_str(";\n");
        any = true;
    }
    if any {
        out.push('\n');
    }
}

/// Emits a struct member function inside its struct body (`struct S {
/// ...; Ret m(params) { body } };`) so `obj.m()` calls resolve natively
/// in HLSL. Return semantics and `[numthreads]` never apply to methods
/// and are dropped; `static`/`const` qualifiers were already dropped at
/// parse (HLSL has no const methods). `structs` threads through for
/// body-statement emission like `emit_function`.
fn emit_method(
    out: &mut String,
    function: &Function,
    structs: &[StructDecl],
    is_static: bool,
) {
    // Statically-called instance-clean methods emit `static` so `S::m`
    // resolves natively (see `static_methods`).
    let _ = write!(
        out,
        "    {}{} {}(",
        if is_static { "static " } else { "" },
        hlsl_type(&function.return_type),
        function.name
    );

    for (index, parameter) in function.parameters.iter().enumerate() {
        if index != 0 {
            out.push_str(", ");
        }
        let array = parameter
            .array_size
            .map(|n| if n == 0 { "[]".to_string() } else { format!("[{n}]") })
            .unwrap_or_default();
        let _ = write!(
            out,
            "{} {}{}",
            hlsl_type(&parameter.ty),
            hlsl_ident(&parameter.name),
            array
        );
    }

    out.push_str(")\n    {\n");

    for statement in &function.body {
        emit_stmt(out, statement, 2, structs);
    }

    out.push_str("    }\n");
}

fn emit_stmt(out: &mut String, statement: &IrStmt, indent: usize, structs: &[StructDecl]) {
    let pad = "    ".repeat(indent);
    match statement {
        IrStmt::Return(None) => {
            let _ = writeln!(out, "{pad}return;");
        }
        IrStmt::Return(Some(expr)) => {
            let _ = writeln!(out, "{pad}return {};", emit_expr(expr));
        }
        IrStmt::VarDecl { ty, name, array_size, init } => {
            // Struct brace-init (`V2P res = {pos, frame};`): HLSL struct
            // initialization needs a complete member list and never takes
            // `[]`, so desugar to memberwise assignment. Missing trailing
            // members zero-fill (C semantics; HLSL does not zero-init
            // locals). Arrays, vectors, and nested-list inits keep the old
            // path; excess inits stay loud at DXC.
            if array_size.is_none() {
                if let (crate::ir::IrType::Named(sname), Some(IrExpr::InitializerList(elems))) =
                    (ty, init)
                {
                    if let Some(sdecl) = structs.iter().find(|s| &s.name == sname) {
                        if elems.len() <= sdecl.fields.len()
                            && elems.iter().all(|e| !matches!(e, IrExpr::InitializerList(_)))
                        {
                            emit_struct_init(out, indent, ty, name, sdecl, elems, structs);
                            return;
                        }
                    }
                }
            }
            // Vector-condition ternary (`bool3(c0,c1,c2) ? a : b` in
            // `T x = ...`): HLSL needs scalar conditions, so desugar to
            // per-component assigns. Only fires for a bool-vector
            // constructor condition with a matching-width vector declared
            // type; everything else (nested ternaries, `x = ...` assigns)
            // keeps the old loud path.
            if array_size.is_none() {
                if let (
                    crate::ir::IrType::Vector { width: n, .. },
                    Some(IrExpr::Conditional {
                        condition,
                        then_branch,
                        else_branch,
                    }),
                ) = (ty, init)
                {
                    if let IrExpr::Construct {
                        ty:
                            crate::ir::IrType::Vector {
                                base: crate::ir::IrScalar::Bool,
                                width: m,
                            },
                        arguments: cargs,
                    } = condition.as_ref()
                    {
                        if *m == *n && *n >= 2 && *n <= 4 && cargs.len() == *n as usize {
                            emit_vector_ternary(
                                out,
                                indent,
                                ty,
                                name,
                                *n,
                                cargs,
                                then_branch,
                                else_branch,
                            );
                            return;
                        }
                    }
                }
            }
            // `Foo x[] = { ... }` is an unsized array. Don't treat a vector
            // initializer like `half2 x = {0,0}` (which also uses
            // `InitializerList` but has a vector `ty`) as an array.
            let is_vector = matches!(ty, crate::ir::IrType::Vector { .. } | crate::ir::IrType::Matrix { .. });
            let is_unsized_array_init = !is_vector
                && array_size.is_none()
                && matches!(init, Some(IrExpr::InitializerList(_)));
            let array = if is_unsized_array_init {
                "[]".to_string()
            } else {
                array_size
                    .map(|n| if n == 0 { "[]".to_string() } else { format!("[{n}]") })
                    .unwrap_or_default()
            };
            if let Some(init) = init {
                let _ = writeln!(
                    out,
                    "{pad}{} {}{} = {};",
                    hlsl_type(ty),
                    hlsl_ident(name),
                    array,
                    emit_expr(init)
                );
            } else {
                let _ = writeln!(
                    out,
                    "{pad}{} {}{};",
                    hlsl_type(ty),
                    hlsl_ident(name),
                    array
                );
            }
        }
        IrStmt::If {
            condition,
            then_branch,
            else_branch,
        } => {
            let _ = writeln!(out, "{pad}if ({}) {{", emit_expr(condition));
            for s in then_branch {
                emit_stmt(out, s, indent + 1, structs);
            }
            if let Some(else_branch) = else_branch {
                let _ = writeln!(out, "{pad}}} else {{");
                for s in else_branch {
                    emit_stmt(out, s, indent + 1, structs);
                }
                let _ = writeln!(out, "{pad}}}");
            } else {
                let _ = writeln!(out, "{pad}}}");
            }
        }
        IrStmt::For {
            init,
            condition,
            update,
            body,
        } => {
            // For HLSL we need `for (init; cond; update)` with init without trailing `;` handling
            // `init` is a VarDecl or Expr, so we emit it inline.
            let init_str = match init.as_ref() {
                IrStmt::VarDecl { ty, name, array_size, init } => {
                    let array = array_size.map(|n| format!("[{n}]")).unwrap_or_default();
                    if let Some(init) = init {
                        format!(
                            "{} {}{} = {}",
                            hlsl_type(ty),
                            hlsl_ident(name),
                            array,
                            emit_expr(init)
                        )
                    } else {
                        format!("{} {}{}", hlsl_type(ty), hlsl_ident(name), array)
                    }
                }
                IrStmt::Expr(expr) => emit_expr(expr),
                IrStmt::Block(_) => String::new(),
                _ => String::new(),
            };
            let cond_str = condition.as_ref().map(|e| emit_expr(e)).unwrap_or_default();
            let update_str = update.as_ref().map(|e| emit_expr(e)).unwrap_or_default();
            let _ = writeln!(out, "{pad}for ({init_str}; {cond_str}; {update_str}) {{");
            for s in body {
                emit_stmt(out, s, indent + 1, structs);
            }
            let _ = writeln!(out, "{pad}}}");
        }
        IrStmt::While { condition, body } => {
            let _ = writeln!(out, "{pad}while ({}) {{", emit_expr(condition));
            for s in body {
                emit_stmt(out, s, indent + 1, structs);
            }
            let _ = writeln!(out, "{pad}}}");
        }
        IrStmt::Switch { scrutinee, cases } => {
            // HLSL `switch` has C fallthrough semantics, matching PSSL
            // exactly, so lowered case bodies emit verbatim.
            let _ = writeln!(out, "{pad}switch ({}) {{", emit_expr(scrutinee));
            let cpad = "    ".repeat(indent + 1);
            for case in cases {
                match &case.label {
                    Some(label) => {
                        let _ = writeln!(out, "{cpad}case {}:", emit_expr(label));
                    }
                    None => {
                        let _ = writeln!(out, "{cpad}default:");
                    }
                }
                for s in &case.body {
                    emit_stmt(out, s, indent + 2, structs);
                }
            }
            let _ = writeln!(out, "{pad}}}");
        }
        IrStmt::Expr(expr) => {
            let _ = writeln!(out, "{pad}{};", emit_expr(expr));
        }
        IrStmt::Block(statements) => {
            // For multi-declarator `float x, y;` we produce a Block of VarDecls.
            // Emitting that as a scoped block would hide `x`/`y` from later uses
            // (DXC: `use of undeclared identifier`). Emit VarDecl-only blocks flat.
            let is_vardecl_block = !statements.is_empty()
                && statements
                    .iter()
                    .all(|s| matches!(s, IrStmt::VarDecl { .. }));
            if is_vardecl_block {
                for statement in statements {
                    emit_stmt(out, statement, indent, structs);
                }
            } else {
                let _ = writeln!(out, "{pad}{{");
                for statement in statements {
                    emit_stmt(out, statement, indent + 1, structs);
                }
                let _ = writeln!(out, "{pad}}}");
            }
        }
    }
}

fn emit_expr(expr: &IrExpr) -> String {
    tracing::trace!(target: "pssl_shader::emit::expr", kind = %ir_expr_kind(expr), "emitting expression");
    match expr {
        IrExpr::Literal(IrLiteral::Number(value)) => value.clone(),
        IrExpr::Literal(IrLiteral::String(value)) => format!("\"{value}\""),
        IrExpr::Literal(IrLiteral::Bool(value)) => value.to_string(),
        IrExpr::Variable(name) => hlsl_ident(name),
        IrExpr::Member { object, member } => {
            format!("{}.{}", emit_expr(object), hlsl_ident(member))
        }
        IrExpr::Index { object, index } => {
            format!("{}[{}]", emit_expr(object), emit_expr(index))
        }
        IrExpr::ResourceIndex { resource, index } => {
            format!("{}[{}]", emit_expr(resource), emit_expr(index))
        }
        IrExpr::ResourceLoad {
            resource,
            coordinates,
            sample,
        } => {
            let coords = emit_expr(coordinates);
            let args = match sample {
                Some(s) => format!("{coords}, {}", emit_expr(s)),
                None => coords,
            };
            format!("{}.Load({args})", emit_expr(resource))
        }
        IrExpr::Call {
            function,
            arguments,
        } => {
            let args: Vec<String> = arguments.iter().map(emit_expr).collect();
            let args_str = args.join(", ");
            tracing::trace!(
                target: "pssl_shader::emit::call",
                function = %function,
                nargs = arguments.len(),
                "emitting call"
            );
            // PSSL `BitFieldExtract(v, offset, count)` has no direct HLSL builtin;
            // lower to `((v >> offset) & ((1u << count) - 1u))`.
            if function.eq_ignore_ascii_case("BitFieldExtract") && args.len() == 3 {
                return format!(
                    "(({} >> {}) & ((1u << {}) - 1u))",
                    args[0], args[1], args[2]
                );
            }
            // PSSL wave `ReadFirstLane(x)` -> HLSL `WaveReadLaneFirst(x)`
            // (read the value from the first active lane).
            if function.eq_ignore_ascii_case("ReadFirstLane") && args.len() == 1 {
                return format!("WaveReadLaneFirst({})", args[0]);
            }
            // PSSL wave `ballot(cond)` -> HLSL `WaveActiveBallot(cond)`
            // (return a bitmask of active lanes for which `cond` holds).
            if function.eq_ignore_ascii_case("ballot") && args.len() == 1 {
                return format!("WaveActiveBallot({})", args[0]);
            }
            // PSSL wave `ExclusivePrefixSum(x)` -> HLSL `WavePrefixSum(x)`
            // (exclusive lane-prefix sum; each active lane's 0-based index
            // among active lanes — wave-compaction idiom, evidenced with
            // `wave::IsFirstActiveLane` + compacted appends).
            if function.eq_ignore_ascii_case("exclusiveprefixsum") && args.len() == 1 {
                return format!("WavePrefixSum({})", args[0]);
            }
            // PSSL wave `InclusivePrefixSum(x)` -> `WavePrefixSum(x) + x`
            // (inclusive = exclusive + self; arithmetic identity, evidenced
            // as `InclusivePrefixSum(itemCount) - itemCount` for exclusive).
            // The argument evaluates twice, so only side-effect-free
            // arguments qualify (impure stays loud to avoid doubling side
            // effects).
            if function.eq_ignore_ascii_case("inclusiveprefixsum") && args.len() == 1 {
                fn ir_pure(e: &IrExpr) -> bool {
                    match e {
                        IrExpr::Literal(_) | IrExpr::Variable(_) => true,
                        IrExpr::Member { object, .. } => ir_pure(object),
                        IrExpr::Index { object, index } => {
                            ir_pure(object) && ir_pure(index)
                        }
                        IrExpr::ResourceIndex { resource, index } => {
                            ir_pure(resource) && ir_pure(index)
                        }
                        IrExpr::Construct { arguments, .. } => {
                            arguments.iter().all(ir_pure)
                        }
                        IrExpr::Unary { expr, .. } => ir_pure(expr),
                        IrExpr::Binary { left, right, .. } => {
                            ir_pure(left) && ir_pure(right)
                        }
                        IrExpr::Conditional { condition, then_branch, else_branch } => {
                            ir_pure(condition) && ir_pure(then_branch) && ir_pure(else_branch)
                        }
                        IrExpr::InitializerList(elems) => elems.iter().all(ir_pure),
                        // Calls (user or builtin), loads, and stores may
                        // have side effects or be non-idempotent: decline.
                        IrExpr::Call { .. }
                        | IrExpr::ResourceLoad { .. }
                        | IrExpr::Assign { .. } => false,
                    }
                }
                if ir_pure(&arguments[0]) {
                    return format!("(WavePrefixSum({0}) + ({0}))", args[0]);
                }
            }
            // PSSL `wave::copysign(x, y)` (scalar sign-copy: magnitude of
            // `x` with sign of `y`; per-lane args, compiler builtin with no
            // header definition) -> `((y) < 0 ? -(abs(x)) : (abs(x)))`.
            // HLSL has no `copysign`; the conditional is exact including
            // `y == 0` (yields `+|x|`, matching copysign) and type-generic
            // (unary minus/abs preserve the operand type). Requires the
            // `wave::` scope (bare `copysign` stays loud; could be a user
            // function).
            if args.len() == 2
                && (function.eq_ignore_ascii_case("wave::copysign")
                    || function.eq_ignore_ascii_case("wave.copysign"))
            {
                return format!(
                    "(({1}) < 0 ? -(abs({0})) : (abs({0})))",
                    args[0], args[1]
                );
            }
            // PSSL `rcp(x)` has no portable HLSL equivalent; lower to `1.0 / x`.
            if function.eq_ignore_ascii_case("rcp") && args.len() == 1 {
                return format!("(1.0 / {})", args[0]);
            }
            // PSSL `mad(a,b,c)` (multiply-add: a*b+c) — lower to inline
            // arithmetic to avoid relying on HLSL `mad` availability.
            if function.eq_ignore_ascii_case("mad") && args.len() == 3 {
                return format!("({}*{}+{})", args[0], args[1], args[2]);
            }
            // PSSL `CndMask(cond, a, b)` — conditional mask: `cond ? a : b`.
            if function.eq_ignore_ascii_case("cndmask") && args.len() == 3 {
                return format!("({} ? {} : {})", args[0], args[1], args[2]);
            }
            // PSSL `min3(a,b,c)` / `max3(a,b,c)` – no HLSL equivalent; lower to nested binary.
            if function.eq_ignore_ascii_case("min3") && args.len() == 3 {
                return format!("min(min({}, {}), {})", args[0], args[1], args[2]);
            }
            if function.eq_ignore_ascii_case("max3") && args.len() == 3 {
                return format!("max(max({}, {}), {})", args[0], args[1], args[2]);
            }
            // PSSL `GetViVjLinearCentroid()` and its perspective
            // siblings (compiler builtins: current-pixel positions,
            // float2; no in-tree definitions anywhere in the framework headers) → HLSL
            // `GetRenderTargetSamplePosition(0)` (sample-0 position).
            // Unthreaded leftovers only (the threading pass rewrites the
            // reachable ones to SV_Position values); note DXC rejects
            // this intrinsic for Vulkan targets, so survivors stay loud.
            if (function.eq_ignore_ascii_case("getvivjlinearcentroid")
                || function.eq_ignore_ascii_case("getvivjperspcenter")
                || function.eq_ignore_ascii_case("getvivjperspcentroid"))
                && args.is_empty()
            {
                return "GetRenderTargetSamplePosition(0)".to_string();
            }
            // PSSL AGC sampler factories – Vulkan samplers are externally bound.
            // `make_*_sampler()` → `g_*_sampler` (no call).
            if args.is_empty() {
                if function.eq_ignore_ascii_case("make_point_sampler") {
                    tracing::trace!(target: "pssl_shader::emit::sampler", "rewriting make_point_sampler() → g_point_sampler");
                    return "g_point_sampler".to_string();
                }
                if function.eq_ignore_ascii_case("make_bilinear_sampler") {
                    tracing::trace!(target: "pssl_shader::emit::sampler", "rewriting make_bilinear_sampler() → g_bilinear_sampler");
                    return "g_bilinear_sampler".to_string();
                }
                if function.eq_ignore_ascii_case("make_trilinear_sampler") {
                    tracing::trace!(target: "pssl_shader::emit::sampler", "rewriting make_trilinear_sampler() → g_trilinear_sampler");
                    return "g_trilinear_sampler".to_string();
                }
            }
            format!("{function}({args_str})")
        }
        IrExpr::Construct { ty, arguments } => {
            // Emit scalar/vector construction verbatim. PSSL vector-scalar casts
            // (e.g. `(float)vec.x` and `float(vec.x)`) are already lowered into
            // `IrExpr::Construct { Float, [Member { vec, "x" }] }` in `lower.rs`,
            // which emits as `float(vec.x)` without any guessing here.
            // HLSL `half2(0.5f)` with a single scalar is not accepted by DXC
            // (`too few elements, expected 2`); splat to `half2(0.5f, 0.5f)`.
            if let crate::ir::IrType::Vector { width, .. } = ty {
                if arguments.len() == 1 && *width > 1 {
                    let arg = emit_expr(&arguments[0]);
                    let args_str = (0..*width).map(|_| arg.clone()).collect::<Vec<_>>().join(", ");
                    tracing::trace!(
                        target: "pssl_shader::emit::cast",
                        ty = %hlsl_type(ty),
                        nargs = arguments.len(),
                        "emitting vector splat"
                    );
                    return format!("{}({args_str})", hlsl_type(ty));
                }
            }
            let args_str = arguments
                .iter()
                .map(emit_expr)
                .collect::<Vec<_>>()
                .join(", ");
            tracing::trace!(
                target: "pssl_shader::emit::cast",
                ty = %hlsl_type(ty),
                nargs = arguments.len(),
                "emitting constructor/cast"
            );
            format!("{}({args_str})", hlsl_type(ty))
        }
        IrExpr::Unary { op, expr } => {
            format!("({}{})", unary_symbol(*op), emit_expr(expr))
        }
        IrExpr::Binary { left, op, right } => {
            format!(
                "({} {} {})",
                emit_expr(left),
                binary_symbol(*op),
                emit_expr(right)
            )
        }
        IrExpr::Conditional {
            condition,
            then_branch,
            else_branch,
        } => format!(
            "({} ? {} : {})",
            emit_expr(condition),
            emit_expr(then_branch),
            emit_expr(else_branch)
        ),
        IrExpr::Assign { lhs, rhs } => {
            format!("{} = {}", emit_expr(lhs), emit_expr(rhs))
        }
        IrExpr::InitializerList(elems) => {
            // Empty lists (`uint a[1] = {};`, C++ value-init) are rejected
            // by DXC; a single zero fills numerics and arrays (missing
            // aggregate elements default to zero). Struct empties never
            // reach here — lowering expands them memberwise.
            if elems.is_empty() {
                return "{ 0 }".to_string();
            }
            let inner = elems.iter().map(emit_expr).collect::<Vec<_>>().join(", ");
            format!("{{ {inner} }}")
        }
    }
}

/// True for identifiers that are legal in PSSL/C but reserved as HLSL
/// modifiers or keywords (`uint snorm` misparses: DXC `modifiers must
/// appear before type`). Emitted HLSL must rename such variables,
/// parameters, and fields (framework convention: trailing underscore, as in
/// the framework's own `sampler_` parameters).
fn is_hlsl_reserved_ident(name: &str) -> bool {
    matches!(
        name,
        "snorm"
            | "unorm"
            | "linear"
            | "centroid"
            | "nointerpolation"
            | "noperspective"
            | "sample"
            | "precise"
            | "shared"
            | "groupshared"
            | "globallycoherent"
            | "row_major"
            | "column_major"
            | "uniform"
            | "volatile"
            | "extern"
            | "inline"
            // Geometry-shader primitive keywords double as reserved words
            // (`bool line` misparses: DXC `modifiers must appear before
            // type`), as do legacy HLSL type names usable as PSSL
            // identifiers (`sampler`, `texture`, ...).
            | "point"
            | "line"
            | "triangle"
            | "lineadj"
            | "triangleadj"
            | "sampler"
            | "texture"
            | "string"
            | "state"
            | "technique"
            | "pass"
            | "dword"
    )
}

/// True when an IR element type is itself a texture object type (bare
/// `Texture2D` / `TextureCube` / `Texture3D` names or a texture-kind
/// resource). A buffer of such elements must emit as a resource array,
/// never as `StructuredBuffer<T>`.
fn is_texture_element(ty: &crate::ir::IrType) -> bool {
    match ty {
        crate::ir::IrType::Resource { kind, .. } => {
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
        crate::ir::IrType::Named(n) => {
            matches!(
                n.to_ascii_lowercase().as_str(),
                "texture1d" | "texture2d" | "texturecube" | "texture3d"
            )
        }
        _ => false,
    }
}

/// Escape an HLSL-reserved identifier consistently at every emission site
/// (declarations and uses go through this helper, so they always agree).
/// Function and struct names are intentionally NOT escaped here: entry
/// points must match the name passed to DXC.
fn hlsl_ident(name: &str) -> String {
    if is_hlsl_reserved_ident(name) {
        format!("{name}_")
    } else {
        name.to_string()
    }
}

/// Maps a normalized PSSL resource kind to its bare HLSL type name (no `<...>`).
fn resource_kind_name<'a>(kind: &'a str) -> &'a str {
    match kind {
        "texture1d" => "Texture1D",
        "texture2d" => "Texture2D",
        "texture2darray" => "Texture2DArray",
        "texturecube" => "TextureCube",
        "texturecubearray" => "TextureCubeArray",
        "texture3d" => "Texture3D",
        "rwtexture2d" => "RWTexture2D",
        "rwtexture2darray" => "RWTexture2DArray",
        "rwtexture3d" => "RWTexture3D",
        "rwtexture1d" => "RWTexture1D",
        "rwtexturecube" => "RWTextureCube",
        "rwtexturecubearray" => "RWTextureCubeArray",
        "mstexture2d" => "Texture2DMS",
        // Read-write multisampled textures have no Vulkan mapping
        // (multisampled storage images are illegal); PSSL uses them
        // with explicit per-sample subscripts (`aoTex[uint3(coord,
        // sample)]`), which is value-identical as a texture array
        // (`RWTexture2DArray`, `arr[uint3]` — same syntax, same
        // values, no HW resolve involved).
        "rwmstexture2d" => "RWTexture2DArray",
        "regularbuffer" | "structuredbuffer" => "StructuredBuffer",
        "rwregularbuffer" | "rwstructuredbuffer" => "RWStructuredBuffer",
        "databuffer" => "Buffer",
        "rwdatabuffer" | "rwbuffer" => "RWBuffer",
        "appendregularbuffer" | "appendstructuredbuffer" => "AppendStructuredBuffer",
        "consumeregularbuffer" | "consumestructuredbuffer" => "ConsumeStructuredBuffer",
        "bytebuffer" | "byteaddressbuffer" => "ByteAddressBuffer",
        "rwbytebuffer" | "rwbyteaddressbuffer" => "RWByteAddressBuffer",
        "trianglestream" => "TriangleStream",
        // PSR top-level BVH descriptor (`sce::Psr::TopLevelBvhDescriptor`):
        // the DXR counterpart is `RaytracingAccelerationStructure`
        // (untyped — emitted without an `<element>` suffix because the
        // lowering leaves `element_type` empty, like byte-address kinds).
        "raytracingaccelerationstructure" => "RaytracingAccelerationStructure",
        other => other,
    }
}

/// True for kinds whose HLSL type is an untyped byte-address buffer
/// (`ByteAddressBuffer` / `RWByteAddressBuffer`) and therefore carries no
/// `<element>` suffix.
fn is_byte_address_kind(kind: &str) -> bool {
    matches!(
        kind,
        "bytebuffer"
            | "byteaddressbuffer"
            | "rwbytebuffer"
            | "rwbyteaddressbuffer"
    )
}

fn hlsl_type(ty: &IrType) -> String {
    match ty {
        IrType::Void => "void".to_string(),
        IrType::Bool => "bool".to_string(),
        IrType::Int => "int".to_string(),
        IrType::UInt => "uint".to_string(),
        IrType::Float => "float".to_string(),
        IrType::Half => "half".to_string(),
        IrType::Double => "double".to_string(),
        // Framework AGC sampler object (`sce::Agc::Core::Sampler`
        // in PSSL struct fields / function parameters): the correct HLSL
        // representation is `SamplerState`. Passing the `sce::`-qualified
        // name through verbatim is not valid HLSL (DXC: `use of undeclared
        // identifier 'sce'`).
        IrType::Named(name) if name == "sce::Agc::Core::Sampler" => {
            "SamplerState".to_string()
        }
        // Framework compatibility types (`crate::framework::framework_type`):
        // `sce::Agc::VrsRate` is a uint8-backed enum, emitted as `uint`
        // (values resolve to literals in lowering).
        IrType::Named(name) => match crate::framework::framework_type(name) {
            Some(mapped) => mapped.to_string(),
            None => name.clone(),
        },
        IrType::Resource {
            kind,
            access: _,
            element,
        } => {
            let kind_name = resource_kind_name(kind.as_str());
            // Flatten `RegularBuffer<RegularBuffer<T>>` which is not valid HLSL
            // (`StructuredBuffer<StructuredBuffer<matrix>>` is rejected as
            // `object 'StructuredBuffer<matrix>' is not allowed`). Collapse to
            // single indirection: `StructuredBuffer<T>`.
            if matches!(kind.as_str(), "regularbuffer" | "structuredbuffer") {
                if let IrType::Resource {
                    kind: inner_kind,
                    element: inner_elem,
                    ..
                } = &**element
                {
                    if matches!(
                        inner_kind.as_str(),
                        "regularbuffer" | "structuredbuffer" | "databuffer"
                    ) {
                        let inner_inner = hlsl_type(inner_elem);
                        return if inner_inner.ends_with('>') {
                            format!("{kind_name}<{} >", inner_inner)
                        } else {
                            format!("{kind_name}<{}>", inner_inner)
                        };
                    }
                }
            }
            // Byte-address buffers are untyped in HLSL (`ByteAddressBuffer`,
            // no `<element>` suffix): drop the element entirely.
            if is_byte_address_kind(kind.as_str()) {
                return kind_name.to_string();
            }
            let inner = hlsl_type(element);
            // DXC requires `> >` with a space for nested templates like
            // `StructuredBuffer<StructuredBuffer<matrix>>` (otherwise `>>` is
            // parsed as shift). Insert a space when the inner already ends with `>`.
            if inner.ends_with('>') {
                format!("{kind_name}<{} >", inner)
            } else {
                format!("{kind_name}<{}>", inner)
            }
        }
        IrType::Vector { base, width } => format!("{}{width}", base.name()),
        IrType::Matrix {
            base,
            rows,
            columns,
        } => {
            format!("{}{rows}x{columns}", base.name())
        }
    }
}

/// Maps a raw PSSL semantic name to a Vulkan HLSL semantic string. Returns
/// `None` for `S_SRT_DATA` (which has no HLSL equivalent and is omitted).
fn hlsl_semantic(name: &str) -> Option<String> {
    if name.eq_ignore_ascii_case("S_SRT_DATA") {
        return None;
    }
    let semantic = Semantic::Name(name.to_string());
    Some(map_semantic(&semantic).unwrap_or_else(|| name.to_string()))
}

fn binary_symbol(op: IrOp) -> &'static str {
    match op {
        IrOp::Add => "+",
        IrOp::Sub => "-",
        IrOp::Mul => "*",
        IrOp::Div => "/",
        IrOp::Mod => "%",
        IrOp::Equal => "==",
        IrOp::NotEqual => "!=",
        IrOp::Less => "<",
        IrOp::LessEqual => "<=",
        IrOp::Greater => ">",
        IrOp::GreaterEqual => ">=",
        IrOp::LogicalAnd => "&&",
        IrOp::LogicalOr => "||",
        IrOp::BitAnd => "&",
        IrOp::BitOr => "|",
        IrOp::BitXor => "^",
        IrOp::ShiftLeft => "<<",
        IrOp::ShiftRight => ">>",
        IrOp::Comma => ",",
        // Never valid as a binary operator; emitted defensively.
        IrOp::Not | IrOp::Negate | IrOp::BitNot => "",
    }
}

fn unary_symbol(op: IrOp) -> &'static str {
    match op {
        IrOp::Negate => "-",
        IrOp::Not => "!",
        IrOp::BitNot => "~",
        // Never valid as a unary operator; emitted defensively.
        _ => "",
    }
}

/// A short name for an [`IrExpr`] variant, used in emission tracing.
fn ir_expr_kind(expr: &IrExpr) -> &'static str {
    match expr {
        IrExpr::Literal(_) => "literal",
        IrExpr::Variable(_) => "variable",
        IrExpr::Member { .. } => "member",
        IrExpr::Index { .. } => "index",
        IrExpr::ResourceIndex { .. } => "resource_index",
        IrExpr::ResourceLoad { .. } => "resource_load",
        IrExpr::Call { .. } => "call",
        IrExpr::Construct { .. } => "cast",
        IrExpr::Unary { .. } => "unary",
        IrExpr::Binary { .. } => "binary",
        IrExpr::Conditional { .. } => "conditional",
        IrExpr::Assign { .. } => "assign",
        IrExpr::InitializerList(_) => "initializer_list",
    }
}
