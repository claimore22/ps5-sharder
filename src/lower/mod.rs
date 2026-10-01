//! Lowering: AST + semantic model -> Shader IR.
//!
//! The IR is HLSL-agnostic; the HLSL emitter reads only from this IR, never
//! directly from the AST (except through the semantic model). This keeps a
//! genuine IR layer between the source and any output backend.

mod expr;
mod deduce;
mod defargs;
mod function;
mod hoist;
mod methods;
mod ngg;
mod rename;
mod retinit;
mod srt;
mod stmt;
mod thread;

use crate::ast::*;
use crate::error::Result;
use crate::ir::{
    FieldDecl, IrValue, Resource, ResourceType, ShaderModule, Stage, StructDecl,
};
use crate::semantic::{SemanticModel, to_ir_type};

use std::collections::{BTreeSet, HashMap};

use function::{lower_function, resolve_const_literal};

/// Lowers a parsed `module` (with its `model`) into a Shader IR module.
pub fn lower_module(module: &Module, model: &SemanticModel) -> Result<ShaderModule> {
    // SRT-rooted global aliases (`static const auto x = srt->member;`)
    // dissolve first: uses rewrite to the member path (flattening
    // resolves it to the binding), so downstream passes never see the
    // unresolvable `auto` global.
    let module = deduce::resolve_srt_global_aliases(module);
    // C++14 `auto` returns resolve first (cloned module): unanimous
    // return-expr types become concrete; the rest stays loud at DXC.
    // `auto` locals deduce second (needs concrete call returns).
    // Structured bindings desugar third (needs deduced callee returns).
    let module = deduce::deduce_auto_returns(&module);
    // 0-arg position builtins thread SV_Position through new trailing
    // params (before auto deduction, so rewritten inits deduce float2).
    let module = thread::thread_position_builtins(
        &module,
        model.entry_point.as_ref().map(|ep| ep.function.as_str()),
    );
    // Directly-called capture-less lambdas inline, and immediate
    // `ExecuteInPixelOrder` critical sections splice (both before auto
    // deduction, so reduced inits deduce normally).
    let module = deduce::beta_reduce_lambdas(&module);
    let module = deduce::inline_execute_in_pixel_order(&module);
    // Integer-texture sampling desugars to GetDimensions + Load (before
    // auto deduction, so rewritten inits deduce normally).
    let module = deduce::desugar_uint_sample(&module);
    let module = deduce::deduce_auto_locals(&module);
    // Vector constructors with excess components truncate to first-N
    // (needs deduced widths from above).
    let module = deduce::truncate_vector_constructors(&module);
    // Vector-element typed buffers feeding atomics become structured
    // buffers (`RWBuffer<uint3>` rejects component atomics; structured
    // allows them). Runs before struct cloning so specializations agree.
    let module = deduce::atomic_structured_buffers(&module);
    // Scoped statics (`S::kMax`) resolve to literals everywhere (not
    // just attributes), so comparisons and sizes fold. Runs before
    // struct cloning so specializations agree.
    let module = deduce::resolve_scoped_consts(&module);
    // Constant-condition `if`s fold (taken branch splices, untaken
    // vanishes), eliminating `if constexpr(false)` bodies once their
    // tests become literal. Unconditionally semantics-preserving.
    let module = deduce::fold_const_if(&module);
    // Inert literal-zero on appends (`buf.Append(v, 0)` â†’ `buf.Append(v)`;
    // official ISA shows a single append with no second operand).
    let module = deduce::fold_append_zero(&module);
    // Template structs clone per concrete use spelling first (function
    // tables below see the clones); then template / `auto`-parameter
    // functions monomorphize per deduced binding (needs concrete call
    // argument types from above); `auto` returns re-resolve after
    // (clones may newly deduce).
    let module = deduce::monomorphize_structs(&module);
    // Value-parameter struct templates instantiate per concrete use
    // spelling after typename-only monomorphization (disjoint template
    // families), so `auto`-parameter functions below deduce against
    // concrete struct names.
    let module = deduce::monomorphize_struct_values(&module);
    let entry_name_mono = model.entry_point.as_ref().map(|ep| ep.function.as_str());
    // Vector params provably holding scalars (all callers pass scalars
    // that splat) narrow to scalars (fixes vector-typed array indices).
    // Runs before template deduction so narrowed types feed it.
    let module = deduce::narrow_vector_params(&module, entry_name_mono);
    // Struct construction in declarations (`S x = S(a,b)` â†’ `S x =
    // {a,b}`) for ctor-less aggregates (HLSL rejects `S(â€¦)` for structs;
    // brace-init is const-correct). Runs before template deduction.
    let module = deduce::desugar_struct_constructs(&module);
    // Whole-struct zero assignments/declarations (`s = 0`, `S s = 0`)
    // become memberwise zero assignments (HLSL rejects scalar-to-struct
    // conversion). Same soundness shape as the construct desugars.
    let module = deduce::desugar_struct_zero_assign(&module);
    // Expression-position struct constructions hoist to temporaries
    // (`f(S(a,b))` â†’ `S tmp = {a,b}; f(tmp)`), same soundness gates.
    let module = deduce::hoist_struct_constructs(&module);
    // Struct-field 2D array uses linearize (`f[i][j]` â†’ `f[i*D1+j]`,
    // declarations already flattened to products at parse). Runs before
    // template deduction so clones inherit linearized indices.
    let module = deduce::linearize_field_indices(&module);
    // Spelling-based `T::` monomorphization (`phong(V2P<T>)` + `T::kX`
    // with `V2P<PackVertex<...>>` callers): binds typenames via preserved
    // declaration spellings, evaluates statics, clones per call. Runs
    // before type-driven `monomorphize` (whose `T::` veto then sees only
    // unresolved cases).
    let module = deduce::monomorphize_qualified(&module, entry_name_mono);
    let module = deduce::monomorphize(&module, entry_name_mono);
    // Flattened row-slice array args materialize via temporaries
    // (`f(arr[x])` for array params â†’ copy-in/out around the call).
    // Runs AFTER monomorphization so template clones carry concrete
    // param types (generic `out Out q[2]` cannot match row-slices).
    let module = deduce::fix_rowslice_args(&module);
    // Re-fold constant `if`s produced by template/`T::` resolution above
    // (`if (false)` from `if constexpr(T::kHasTangent)` with
    // `HasTangent=false` must vanish; the earlier fold ran before these
    // literals existed). Idempotent and semantics-preserving.
    let module = deduce::fold_const_if(&module);
    // Integer texel offsets on sampling (`SampleLOD(..., int2)`) fold
    // into UV arithmetic (Vulkan forbids `Offset` on non-Gather reads).
    let module = deduce::fold_sample_offsets(&module);
    // Stateless functor calls inline (`isValid(i)` with empty-struct
    // predicate types), then empty-struct tag parameters drop (with
    // call-site args) once arities settled.
    let module = deduce::inline_stateless_functors(&module);
    let module = deduce::drop_empty_tag_params(&module);
    let module = deduce::deduce_auto_returns(&module);
    let module = deduce::desugar_bindings(&module);
    // Whole-struct SRT-mirror copies (`s_srt = srt;`) propagate to the
    // parameter before lowering (single binding set downstream).
    let entry_name = model.entry_point.as_ref().map(|ep| ep.function.as_str());
    let module = deduce::propagate_struct_alias(&module, entry_name);
    let module: &Module = &module;
    let mut ir = ShaderModule {
        stage: model.stage,
        entry_point: model.entry_point.as_ref().map(|ep| ep.function.clone()),
        ..ShaderModule::default()
    };

    // Collect resource-typed struct fields and the variable->struct mapping
    // needed to recognise resource member access (`srt_data.dest[...]`,
    // `srt_data.source.Load(...)`) during expression lowering.
    let mut resource_fields: HashMap<(String, String), Type> = HashMap::new();
    let mut var_struct: HashMap<String, String> = HashMap::new();

    for decl in &module.structs {
        for f in &decl.fields {
            if let Type::Resource { .. } = f.ty {
                resource_fields.insert((decl.name.clone(), f.name.clone()), f.ty.clone());
            }
        }
    }

    for decl in &module.globals {
        if let Type::Named(struct_name) = &decl.ty {
            var_struct.insert(decl.name.clone(), struct_name.clone());
        }
    }

    // Module-global types seed each function's lowering environment (params
    // and locals shadow them): grounds cast resolution and builtin-constant
    // fallback for globals. Top-level resources seed too (their kinds
    // drive call lowering such as single-dimension `GetDimensions`).
    let mut global_types: HashMap<String, Type> = module
        .globals
        .iter()
        .map(|decl| (decl.name.clone(), decl.ty.clone()))
        .collect();
    for res in &module.resources {
        global_types.entry(res.name.clone()).or_insert(Type::Resource {
            kind: res.kind.clone(),
            access: crate::ast::ResourceAccess::ReadOnly,
            element: Box::new(res.element_ty.clone().unwrap_or(Type::UInt)),
        });
    }

    // Struct field types for dotted member queries (`psIn.uv`): lets
    // lowering resolve struct-member vector shapes without new plumbing.
    let struct_fields: HashMap<String, HashMap<String, Type>> = module
        .structs
        .iter()
        .map(|decl| {
            (
                decl.name.clone(),
                decl.fields
                    .iter()
                    .map(|f| (f.name.clone(), f.ty.clone()))
                    .collect(),
            )
        })
        .collect();

    for decl in &module.structs {
        ir.structs.push(lower_struct(decl));
    }

    for resource in &module.resources {
        ir.resources.push(lower_resource(resource));
    }

    // S_SRT_DATA params become globals
    // (e.g. `Srt srt : S_SRT_DATA` -> `Srt srt;`)
    // Multi-entry libraries (`_lib` ray-tracing shaders) repeat the same
    // SRT param name in every entry for the same host binding: keep one
    // global per (name, type) so flattening emits one binding set rather
    // than redefinitions (DXC `redefinition of 'srt_m_â€¦'`). Same name
    // with a different type keeps today's loud path.
    for function in &module.functions {
        for param in &function.parameters {
            if let Some(sem) = &param.semantic {
                if sem.name().eq_ignore_ascii_case("S_SRT_DATA") {
                    if let Type::Named(struct_name) = &param.ty {
                        var_struct.insert(param.name.clone(), struct_name.clone());
                    }
                    let ty = to_ir_type(&param.ty);
                    let dup = ir.globals.iter().any(|g| g.name == param.name && g.ty == ty);
                    if dup {
                        continue;
                    }
                    ir.globals.push(crate::ir::GlobalVar {
                        ty,
                        name: param.name.clone(),
                        array_size: None,
                        const_value: None,
                        shared: false,
                    });
                }
            }
        }
    }

    for global in &module.globals {
        let const_value = module
            .constants
            .get(&global.name)
            .and_then(|expr| resolve_const_literal(&module.constants, expr));
        let mut ir_ty = to_ir_type(&global.ty);
        // `static const short2 k_wavefront_size = ushort2(4,16)` â€“ declared as
        // signed `short2` (Int) but initialized from unsigned `ushort2` (UInt)
        // and used as `uint2` (`out_pixel : uint2 = offset * k + thread_id`).
        // Keep the type consistent with the initializer to avoid a redundant
        // signed/unsigned conversion (`int2 = uint2(...)`).
        if let Some(init_expr) = module.constants.get(&global.name) {
            if let Expr::Constructor { ty: init_ty, .. } = init_expr {
                if let Type::Vector { base: init_base, width: init_w } = init_ty {
                    if let crate::ir::IrType::Vector { base: var_base, width: var_w } = &ir_ty {
                        if var_w == init_w {
                            let init_scalar = match init_base {
                                ScalarType::Int => crate::ir::IrScalar::Int,
                                ScalarType::UInt => crate::ir::IrScalar::UInt,
                                ScalarType::Float => crate::ir::IrScalar::Float,
                                ScalarType::Half => crate::ir::IrScalar::Half,
                                ScalarType::Bool => crate::ir::IrScalar::Bool,
                            };
                            if var_base != &init_scalar {
                                ir_ty = crate::ir::IrType::Vector {
                                    base: init_scalar,
                                    width: *var_w,
                                };
                                tracing::trace!(
                                    target: "pssl_shader::lower::global",
                                    global = %global.name,
                                    declared = ?global.ty,
                                    init_ty = ?init_ty,
                                    "aligning global vector signedness to initializer"
                                );
                            }
                        }
                    }
                }
            }
        }
        ir.globals.push(crate::ir::GlobalVar {
            ty: ir_ty,
            // Same reserved-keyword rename as locals/params/resources:
            // PSSL `sampler`/`texture` become `sampler_`/`texture_`
            // everywhere, so IR-level name matching (dead locals, alias
            // resolution, hoisting) sees one consistent spelling.
            name: if let Some(renamed) = rename::reserved_hlsl_rename(&global.name) {
                renamed.to_string()
            } else {
                global.name.clone()
            },
            array_size: global.array_size,
            const_value,
            shared: global.shared,
        });

        // Keep in constants for backward compatibility.
        ir.constants.push(IrValue::new(global.name.clone()));
    }

    for function in &module.functions {
        let is_entry = ir.entry_point.as_deref() == Some(function.name.as_str());
        ir.functions.push(lower_function(
            function,
            &resource_fields,
            &var_struct,
            &module.constants,
            &global_types,
            &struct_fields,
            is_entry,
        )?);
    }

    // Scalar zero-subscripts (`scene_data.light_count[0]` with scalar
    // `int light_count`, `light.light_type[0]`): PSSL permits indexing
    // a scalar (yields the scalar); HLSL rejects it (`subscripted value
    // is not an array, matrix, or vector`). Fold literal-`0` subscripts
    // on provably scalar, non-array bases (params, locals, globals,
    // dotted member paths through structs, buffer-element paths).
    // Vector/matrix bases keep component indexing (valid HLSL);
    // shadowed or multiply-declared names veto (scope-unsafe); anything
    // unresolvable keeps today's loud path. Runs before downstream
    // index-shape passes so they see the cleaned form.
    lower_scalar_zero_subscripts(&mut ir);

    // Collapse duplicate same-signature functions. HLSL has no overload
    // resolution, but PSSL headers legitimately arrive at identical
    // signatures (e.g. `ulong`â†’`uint` normalization makes a `uint` and a
    // `ulong` overload indistinguishable, or a re-inlined header repeats a
    // helper). Same (scope, name, return type, param types) keeps only one
    // definition.
    //
    // A bare forward declaration (`uint f(uint2);`, empty body) beside its
    // real definition must resolve to the definition: keeping the first
    // occurrence would pin the empty stub and emit a bodyless function
    // that DXC rejects (`control reaches end of non-void function`).
    // Preference is therefore: first non-empty body wins; all-empty keeps
    // the first occurrence. This preference applies to FREE functions
    // only. Scoped methods keep first-wins (a method forward declaration
    // beside its definition keeps the stub): preferring real method
    // bodies would newly surface buffer-element method calls
    // (`buf[i].m()`, which need load-call-storeback synthesis the IR
    // cannot express yet) and flip vacuously-passing shaders loud. Real
    // method bodies are a separate subsystem (see `StateProxy`
    // forwarding); methods stay at prior behavior until then.
    {
        use std::collections::HashMap;
        let mut best: HashMap<String, usize> = HashMap::new();
        for (idx, f) in ir.functions.iter().enumerate() {
            let params = f
                .parameters
                .iter()
                .map(|p| {
                    let arr = p
                        .array_size
                        .map(|n| if n == 0 { "[]".to_string() } else { format!("[{n}]") })
                        .unwrap_or_default();
                    format!("{}{arr}", p.ty.display())
                })
                .collect::<Vec<_>>()
                .join(",");
            let key = format!(
                "{}|{}|{}|{params}",
                f.scope.as_deref().unwrap_or(""),
                f.name,
                f.return_type.display()
            );
            match best.get(&key) {
                None => {
                    best.insert(key, idx);
                }
                Some(&prev) => {
                    // Free functions prefer a body over an empty stub;
                    // scoped methods keep the first occurrence.
                    if f.scope.is_none()
                        && ir.functions[prev].body.is_empty()
                        && !f.body.is_empty()
                    {
                        best.insert(key, idx);
                    }
                }
            }
        }
        let keep: std::collections::HashSet<usize> = best.values().copied().collect();
        let mut idx = 0usize;
        ir.functions.retain(|f| {
            let cur = idx;
            idx += 1;
            keep.contains(&cur)
        });
    }

    // Collapse overloads differing ONLY in return type (re-inlined
    // headers define e.g. `float4 FxaaPixelShader(same params)` and
    // `half4 FxaaPixelShader(same params)`; PSSL sees one variant per
    // configuration, but textual expansion keeps all, and HLSL rejects
    // return-only overloading). Keep the overload(s) matching a call
    // site's expected type (`T x = f(...)` / `return f(...)`); sites
    // disagreeing keep every involved overload (loud â€” genuinely
    // unrepresentable); unconstrained groups keep the first definition.
    // Scoped methods are exempt (same vacuity policy as above).
    {
        use std::collections::{HashMap, HashSet};
        fn params_key(f: &crate::ir::Function) -> String {
            f.parameters
                .iter()
                .map(|p| {
                    let arr = p
                        .array_size
                        .map(|n| if n == 0 { "[]".to_string() } else { format!("[{n}]") })
                        .unwrap_or_default();
                    format!("{}{arr}", p.ty.display())
                })
                .collect::<Vec<_>>()
                .join(",")
        }
        // Group free-function indices by (scope, name, params).
        let mut groups: HashMap<String, Vec<usize>> = HashMap::new();
        for (idx, f) in ir.functions.iter().enumerate() {
            if f.scope.is_some() {
                continue;
            }
            let key = format!(
                "{}|{}|{}",
                f.scope.as_deref().unwrap_or(""),
                f.name,
                params_key(f)
            );
            groups.entry(key).or_default().push(idx);
        }
        // Expected return types per function name from directly-typed
        // call positions (VarDecl init, bare `return f(...)`).
        fn expected_at_calls(
            stmts: &[crate::ir::IrStmt],
            callee: &str,
            ret_ty: &crate::ir::IrType,
            out: &mut HashSet<String>,
        ) {
            for s in stmts {
                match s {
                    crate::ir::IrStmt::VarDecl { ty, init: Some(e), .. } => {
                        if expr_is_call_to(e, callee) {
                            out.insert(ty.display());
                        }
                        nested_stmts(s, callee, ret_ty, out);
                    }
                    crate::ir::IrStmt::Return(Some(e)) => {
                        if expr_is_call_to(e, callee) {
                            out.insert(ret_ty.display());
                        }
                    }
                    _ => nested_stmts(s, callee, ret_ty, out),
                }
            }
        }
        fn expr_is_call_to(e: &crate::ir::IrExpr, callee: &str) -> bool {
            match e {
                crate::ir::IrExpr::Call { function, .. } => function == callee,
                _ => false,
            }
        }
        fn nested_stmts(
            s: &crate::ir::IrStmt,
            callee: &str,
            ret_ty: &crate::ir::IrType,
            out: &mut HashSet<String>,
        ) {
            match s {
                crate::ir::IrStmt::Block(ss) => expected_at_calls(ss, callee, ret_ty, out),
                crate::ir::IrStmt::If { then_branch, else_branch, .. } => {
                    expected_at_calls(then_branch, callee, ret_ty, out);
                    if let Some(els) = else_branch {
                        expected_at_calls(els, callee, ret_ty, out);
                    }
                }
                crate::ir::IrStmt::For { init, body, .. } => {
                    expected_at_calls(std::slice::from_ref(init.as_ref()), callee, ret_ty, out);
                    expected_at_calls(body, callee, ret_ty, out);
                }
                crate::ir::IrStmt::While { body, .. } => {
                    expected_at_calls(body, callee, ret_ty, out)
                }
                crate::ir::IrStmt::Switch { cases, .. } => {
                    for c in cases {
                        expected_at_calls(&c.body, callee, ret_ty, out);
                    }
                }
                _ => {}
            }
        }
        let mut drop_idx: HashSet<usize> = HashSet::new();
        for idxs in groups.values() {
            if idxs.len() < 2 {
                continue;
            }
            let returns: HashSet<String> = idxs
                .iter()
                .map(|&i| ir.functions[i].return_type.display())
                .collect();
            if returns.len() < 2 {
                continue;
            }
            let name = ir.functions[idxs[0]].name.clone();
            let mut expected: HashSet<String> = HashSet::new();
            for func in &ir.functions {
                expected_at_calls(&func.body, &name, &func.return_type, &mut expected);
            }
            // `auto`-typed positions constrain nothing (deduction owns
            // them, and they resolve from the kept overload).
            expected.remove("auto");
            let keep_returns: HashSet<String> = returns
                .intersection(&expected)
                .cloned()
                .collect();
            if keep_returns.is_empty() {
                // Unconstrained: keep the first definition (header order).
                let first = idxs[0];
                for &i in &idxs[1..] {
                    drop_idx.insert(i);
                }
                let _ = first;
            } else {
                for &i in idxs {
                    if !keep_returns.contains(&ir.functions[i].return_type.display()) {
                        drop_idx.insert(i);
                    }
                }
            }
        }
        if !drop_idx.is_empty() {
            let mut idx = 0usize;
            ir.functions.retain(|_| {
                let cur = idx;
                idx += 1;
                !drop_idx.contains(&cur)
            });
        }
    }

    // Drop empty-body free functions whose signatures name undeclared
    // host types (vectormath SIMD classes like `floatInVec`, whose
    // `SCE_VECTORMATH_ALWAYS_INLINE` definitions carry host-intrinsic
    // bodies the parser cannot keep): the stub can never be valid HLSL
    // (the type itself emits verbatim and DXC rejects it with `unknown
    // type name`, and the empty non-void body independently fails with
    // `control reaches end of non-void function`), yet name-based
    // reachability keeps it alive whenever any call site uses the same
    // bare name with plain numeric arguments (e.g. `clamp(F0, 0, 0.99)`),
    // which HLSL resolves natively once the stub is gone. Only free
    // functions are considered; scoped methods are exempt. Only
    // empty-body, non-void functions are candidates (a `void` stub is
    // legal HLSL, and a bodied function is real code). `auto`/`T`
    // spellings are exempt (deduction/generic machinery owns them), as
    // are signatures over declared structs, framework-mapped types, and
    // plain sampler spellings. Dropping cannot regress a validating
    // shader: no validating shader emits an empty non-void function
    // today (DXC rejects every one).
    {
        let declared: std::collections::HashSet<&str> =
            ir.structs.iter().map(|s| s.name.as_str()).collect();
        // A struct is HLSL-emissible only if every field type transitively
        // is: host SIMD wrappers (`floatInVec { vec_float4 mData; }`)
        // declare as ordinary structs, so a shallow "is declared" check
        // keeps them. `vec_float4` itself is a host vector typedef with no
        // declaration, which fails the recursion below.
        fn struct_emissible(
            n: &str,
            structs: &std::collections::HashMap<&str, &[crate::ir::FieldDecl]>,
            visiting: &mut Vec<String>,
            depth: usize,
        ) -> bool {
            if depth > 8 || visiting.iter().any(|v| v == n) {
                return true;
            }
            let Some(fields) = structs.get(n) else {
                return false;
            };
            visiting.push(n.to_string());
            let ok = fields
                .iter()
                .all(|f| ty_known_inner(&f.ty, structs, visiting, depth + 1));
            visiting.pop();
            ok
        }
        fn named_known(
            n: &str,
            structs: &std::collections::HashMap<&str, &[crate::ir::FieldDecl]>,
            visiting: &mut Vec<String>,
            depth: usize,
        ) -> bool {
            if crate::framework::framework_type(n).is_some()
                || crate::framework::is_sampler_type_name(n)
            {
                return true;
            }
            if matches!(
                n,
                "auto"
                    | "T"
                    | "void"
                    | "bool"
                    | "int"
                    | "uint"
                    | "float"
                    | "half"
                    | "double"
                    | "string"
                    | "SamplerState"
                    | "SamplerComparisonState"
                    | "ByteAddressBuffer"
                    | "RWByteAddressBuffer"
            ) {
                return true;
            }
            struct_emissible(n, structs, visiting, depth)
        }
        fn ty_known_inner(
            ty: &crate::ir::IrType,
            structs: &std::collections::HashMap<&str, &[crate::ir::FieldDecl]>,
            visiting: &mut Vec<String>,
            depth: usize,
        ) -> bool {
            match ty {
                crate::ir::IrType::Named(n) => named_known(n, structs, visiting, depth),
                crate::ir::IrType::Resource { element, .. } => {
                    ty_known_inner(element, structs, visiting, depth)
                }
                _ => true,
            }
        }
        let struct_map: std::collections::HashMap<&str, &[crate::ir::FieldDecl]> = ir
            .structs
            .iter()
            .map(|s| (s.name.as_str(), s.fields.as_slice()))
            .collect();
        let _ = declared;
        ir.functions.retain(|f| {
            if f.scope.is_some() || !f.body.is_empty() {
                return true;
            }
            if matches!(f.return_type, crate::ir::IrType::Void) {
                return true;
            }
            if !ty_known_inner(&f.return_type, &struct_map, &mut Vec::new(), 0) {
                return false;
            }
            if f.parameters.iter().any(|p| {
                !ty_known_inner(&p.ty, &struct_map, &mut Vec::new(), 0)
            }) {
                return false;
            }
            true
        });
    }

    // Drop free functions unreachable from the entry point (uncalled
    // header baggage with latent body issues, e.g. vectormath
    // double-precision helpers over `vec_ullongN`), then struct methods
    // unreachable from entry + free functions before SRT flattening and
    // hoisting see them: dead bodies must not pin resource fields or
    // create bindings. Fixpoint: each pass shrinks the other's roots
    // (freed callees orphan methods and vice versa) until neither
    // removes anything.
    loop {
        let before = ir.functions.len();
        methods::strip_unreachable_methods(&mut ir);
        methods::strip_unreachable_free_functions(&mut ir);
        if ir.functions.len() == before {
            break;
        }
    }

    // Struct-scoped enumerators (`MaterialSRT::Transparent` for a nested
    // `enum Flags { â€¦ }`): the parser registers enumerators as int
    // globals (top-level behavior), so the qualifier folds to the bare
    // name here. First: every downstream pass must see bare names.
    fold_scoped_enum_uses(&mut ir);

    // User SGPR reads (`__read_user_sgpr(N)`, lowered to `__user_sgprN`
    // variables) need backing globals: host-provided root constants that
    // the HLSL emitter declares as plain externally-settable globals.
    for index in collect_user_sgpr_indices(&ir) {
        let name = crate::framework::user_sgpr_name(index);
        if !ir.globals.iter().any(|g| g.name == name) {
            ir.globals.push(crate::ir::GlobalVar {
                ty: crate::ir::IrType::UInt,
                name,
                array_size: None,
                const_value: None,
                shared: false,
            });
        }
    }

    // C++ default arguments: fill short call sites from callee defaults.
    // Runs BEFORE SRT elision so defaulted trailing params complete the
    // call first (`f(srt, x)` â†’ `f(srt, x, 0)`), letting elision's strict
    // arity check fire (`â†’ f(x, 0)`). Elision removes the same index from
    // calls and signatures, so filled defaults stay aligned.
    defargs::lower_default_args(&mut ir);

    // Flatten SRT root globals that mix GPU resources with scalar uniforms into
    // separate resource bindings + a scalar cbuffer, so the emitted HLSL is
    // accepted by DXC.
    srt::flatten_srt(&mut ir);

    // Materialization residue (`ViewGlobalsSRT globals; â€¦`,
    // `__srtarg_N; â€¦`) for elided params nobody reads is dead weight â€”
    // and twin-typed stores into full-typed members actively mismatch
    // after splitting. Drop provably-dead locals now so hoisting sees
    // zero references and strips the members from struct definitions.
    srt::drop_dead_locals(&mut ir);

    // Hoist resource members nested inside structs (e.g. buffers of
    // textures, buffers of structs-with-buffers) into top-level bindings.
    hoist::hoist_nested_resources(&mut ir);

    // Collapse duplicate identical globals (re-inlined headers declare
    // the same `static const uint k = v;` twice): the emitter keeps the
    // first per name, but `method_binding_params` below treats any
    // same-name pair as ambiguous and drops the binding entirely, leaving
    // struct methods that read the constant with an undeclared identifier.
    // Same (name, type, array size, const value, shared flag) keeps the
    // first; genuinely conflicting declarations are kept (ambiguity stays
    // loud as before).
    {
        use std::collections::HashSet;
        let mut seen: HashSet<String> = HashSet::new();
        ir.globals.retain(|g| {
            let key = format!(
                "{}|{}|{:?}|{:?}|{}",
                g.name,
                g.ty.display(),
                g.array_size,
                g.const_value,
                g.shared
            );
            seen.insert(key)
        });
    }

    // Struct methods referencing module bindings (emitted after structs)
    // gain implicit binding parameters, passed at call sites where the
    // bindings are visible.
    methods::method_binding_params(&mut ir);

    // SRT elision removes helper params (`RenderMeshSRT* pSrt`), leaving
    // `pSrt.getX()` calls dangling (object undeclared, extras unappended).
    // Single-return getters inline to flattened bindings; passing shaders
    // have no dangling roots and are untouched. Runs BEFORE the split
    // below: splitting materializes whole-`S` buffer reads (including
    // inside method bodies), which would turn single-return getters
    // multi-statement and block their inlining. Inlining first leaves
    // plain `S x = buf[i]` shapes the split already knows how to
    // materialize at call sites.
    methods::inline_dangling_root_methods(&mut ir);

    // Methods killed by the inlining above (no remaining callers) are
    // dead: strip them so their signatures (e.g. `StructuredBuffer<S>`
    // extras) cannot leak undeclared/nested types into struct emission.
    // Reachability-based, so passing shaders are untouched.
    methods::strip_unreachable_methods(&mut ir);

    // Inlining and stripping orphan more materialization residue (locals
    // whose only reader was an inlined call or a stripped function):
    // drop them before splitting so twin-typed stores never dangle.
    srt::drop_dead_locals(&mut ir);

    // Single-assignment resource aliases (`static Texture2D PbrDFG; ...
    // PbrDFG = srt_tex;`): SPIR-V forbids the store, so resolve reads
    // to the module-scope value and erase the global. Before splitting
    // so the twin machinery never sees alias locals.
    srt::erase_resource_aliases(&mut ir);
    // Split buffers of structs whose texture members hoisting could not
    // absorb (e.g. sampled only through materialized locals): the buffer
    // keeps its name with a scalars-only twin element, texture members
    // become per-buffer arrays, whole-struct copies materialize.
    // Runs after hoisting so it only sees the residue (already-hoisted
    // members are stripped from the structs), and after inlining so
    // inlined getters present plain buffer reads.
    hoist::split_textured_struct_buffers(&mut ir);

    // UAV mipmap views (`tex.MipMaps[m][c]`): per-texture unbounded view
    // arrays, so mip-indexed reads/writes resolve (single HLSL views
    // cannot address mips). Runs after splits (bindings compose).
    hoist::hoist_mipmap_views(&mut ir);

    // NGG cull-data builtin (`Ngg::CullData<T>`): `Set` drops (no Vulkan
    // effect), `GetTriangle` becomes a zero `T[3]` (culling conservatively
    // disabled). Shaders without the builtin are untouched.
    ngg::lower_ngg_cull_data(&mut ir);

    // Struct-constructor returns (`return S(a, b)`) are not valid HLSL;
    // desugar to a temporary plus memberwise assignments (mirrors the
    // emitter's declaration-side struct init).
    retinit::lower_struct_returns(&mut ir);

    // Explicit-`uniform` entry parameters (`uniform float4x4 wvp`) are
    // host constants: Vulkan entries take only varyings/resources, so
    // they move into a cbuffer (uses keep the bare name, which resolves
    // to the member). Helpers keep by-value params; only the entry moves.
    hoist_entry_uniform_params(&mut ir);

    // Entry-IO varyings without semantics gain TEXCOORDn (the source links by
    // member name; Vulkan requires a semantic per field). Only the
    // emitter reads field semantics, so no earlier pass is affected.
    fill_entry_varying_semantics(&mut ir);

    // NGG-output mesh entries become DX mesh output objects (`out
    // vertices` / `out indices` + `SetMeshOutputCounts`). Last: it
    // appends semantic-less entry params that the TEXCOORD fill above
    // must not touch.
    ngg::lower_mesh_outputs(&mut ir);

    // Hull entries gain the implicit control-point ID input (DXC hs_6_0
    // requires `SV_OutputControlPointID`; PSSL leaves it implicit).
    // Declaring an extra system input is behavior-preserving; bodies
    // that never index by point ignore it.
    lower_hull_entry(&mut ir);

    tracing::debug!(
        target: "pssl_shader::ir",
        functions = ir.functions.len(),
        structs = ir.structs.len(),
        resources = ir.resources.len(),
        globals = ir.globals.len(),
        "produced IR module"
    );

    Ok(ir)
}

/// Moves explicit-`uniform` entry parameters into a `uniform_params`
/// cbuffer (`uniform float4x4 wvp` in edge-animation vertex entries).
/// Vulkan entry functions accept only varyings and resources; a bare
/// host constant would fail DXC (`semantic string missing`). Only
/// scalar/vector/matrix params without direction move (resources,
/// structs, and `out` params keep today's loud path); uses keep the
/// bare name and resolve to the cbuffer member. Shaders without such
/// parameters are untouched.
fn hoist_entry_uniform_params(ir: &mut ShaderModule) {
    let Some(entry_name) = ir.entry_point.clone() else {
        return;
    };
    let Some(entry) = ir.functions.iter_mut().find(|f| f.name == entry_name) else {
        return;
    };
    let mut moved: Vec<(String, crate::ir::IrType, Option<u32>)> = Vec::new();
    entry.parameters.retain(|p| {
        let plain_value = p.direction.is_none()
            && matches!(
                &p.ty,
                crate::ir::IrType::Bool
                    | crate::ir::IrType::Int
                    | crate::ir::IrType::UInt
                    | crate::ir::IrType::Float
                    | crate::ir::IrType::Half
                    | crate::ir::IrType::Double
                    | crate::ir::IrType::Vector { .. }
                    | crate::ir::IrType::Matrix { .. }
            );
        if p.is_uniform && plain_value {
            moved.push((p.name.clone(), p.ty.clone(), p.array_size));
            return false;
        }
        true
    });
    if moved.is_empty() {
        return;
    }
    let mut name = "uniform_params".to_string();
    let mut n = 0u32;
    while ir.resources.iter().any(|r| r.name == name) {
        n += 1;
        name = format!("uniform_params{n}");
    }
    ir.resources.push(Resource {
        name,
        resource_type: ResourceType::ConstantBuffer,
        kind: "constantbuffer".to_string(),
        element_type: None,
        descriptor_set: Some(0),
        binding: None,
        register: None,
        space: None,
        array_size: None,
        fields: moved
            .into_iter()
            .map(|(fname, fty, arr)| FieldDecl {
                ty: fty,
                name: fname,
                semantic: None,
                array_size: arr,
            })
            .collect(),
    });
}

/// Assign `TEXCOORDn` to entry-IO struct members lacking semantics.
///
/// The source platform links VS→PS varyings by member name, so sample headers leave
/// them bare (`sunspace_position`, `uv`, â€¦); Vulkan HLSL requires a
/// semantic on every entry-input/output field (DXC `semantic string
/// missing`). Members already carrying one (`SV_Position`,
/// `TEXCOORDm`, `NORMAL`, â€¦) keep it; bare members take the lowest
/// free `TEXCOORD` indices in field order, so the separately compiled
/// producer/consumer sides (same struct definition, same rule) agree.
///
/// Only entry *input* structs plus vertex-like *output* structs:
/// fragment outputs are render targets (`SV_Targetn` â€” guessed only for
/// the unambiguous single bare numeric return, see below),
/// ray-pipeline stages have their own IO rules (attributes/payloads
/// must not gain semantics), and the `S_SRT_DATA` root is uniforms,
/// not varyings. Semantics on non-signature uses of the same struct
/// (locals) are ignored by DXC, so sharing the definition is safe.
/// Appends the hull control-point ID input (`uint <fresh> :
/// SV_OutputControlPointID`) when the hull entry lacks one (DXC hs_6_0
/// requires it; PSSL leaves it implicit). Declaring an extra system
/// input is behavior-preserving; bodies that never index by point
/// ignore it. Entries that already declare it keep today's path.
fn lower_hull_entry(ir: &mut ShaderModule) {
    use crate::ir::{IrStmt, IrType};
    if !matches!(ir.stage, Some(Stage::Hull)) {
        return;
    }
    let Some(entry_name) = ir.entry_point.clone() else {
        return;
    };
    let Some(entry) = ir.functions.iter_mut().find(|f| f.name == entry_name) else {
        return;
    };
    let has_id = entry.parameters.iter().any(|p| {
        p.semantic
            .as_deref()
            .is_some_and(|s| s.eq_ignore_ascii_case("SV_OutputControlPointID"))
    });
    if has_id {
        return;
    }
    fn collect_names(stmts: &[IrStmt], out: &mut std::collections::HashSet<String>) {
        for s in stmts {
            match s {
                IrStmt::VarDecl { name, .. } => {
                    out.insert(name.clone());
                }
                IrStmt::Block(ss) => collect_names(ss, out),
                IrStmt::If { then_branch, else_branch, .. } => {
                    collect_names(then_branch, out);
                    if let Some(eb) = else_branch {
                        collect_names(eb, out);
                    }
                }
                IrStmt::For { init, body, .. } => {
                    collect_names(std::slice::from_ref(init), out);
                    collect_names(body, out);
                }
                IrStmt::While { body, .. } => collect_names(body, out),
                IrStmt::Switch { cases, .. } => {
                    for c in cases {
                        collect_names(&c.body, out);
                    }
                }
                _ => {}
            }
        }
    }
    let mut taken: std::collections::HashSet<String> =
        entry.parameters.iter().map(|p| p.name.clone()).collect();
    collect_names(&entry.body, &mut taken);
    let mut name = "__controlPointID".to_string();
    let mut n = 0u32;
    while taken.contains(&name) {
        n += 1;
        name = format!("__controlPointID{n}");
    }
    entry.parameters.push(crate::ir::function::Parameter {
        ty: IrType::UInt,
        name,
        semantic: Some("SV_OutputControlPointID".to_string()),
        array_size: None,
        direction: None,
        gs_primitive: None,
        default: None,
        is_uniform: false,
        mesh_object: None,
    });
}

fn fill_entry_varying_semantics(ir: &mut ShaderModule) {
    // Ray-pipeline stages: attributes/payloads must not gain semantics.
    if matches!(
        &ir.stage,
        Some(Stage::AnyHit) | Some(Stage::ClosestHit) | Some(Stage::Miss) | Some(Stage::Intersection)
    ) {
        return;
    }
    let Some(entry_name) = ir.entry_point.clone() else {
        return;
    };
    // Fragment entries returning a bare scalar/vector (`float3 main(...)`
    // with no return semantic, as in the IBL filter pixel shaders) can
    // only mean render target 0: fragment outputs have no other legal
    // form. Struct/void returns keep today's loud path.
    if matches!(&ir.stage, Some(Stage::Fragment)) {
        if let Some(entry) = ir.functions.iter_mut().find(|f| f.name == entry_name) {
            let is_bare_numeric = entry.return_semantic.is_none()
                && matches!(
                    &entry.return_type,
                    crate::ir::IrType::Bool
                        | crate::ir::IrType::Int
                        | crate::ir::IrType::UInt
                        | crate::ir::IrType::Float
                        | crate::ir::IrType::Half
                        | crate::ir::IrType::Double
                        | crate::ir::IrType::Vector { .. }
                );
            if is_bare_numeric {
                entry.return_semantic = Some("SV_Target0".to_string());
            }
            // Fragment `S_POSITION` inputs must be float4 (Vulkan
            // VUID-FragCoord: a float3 input produces a 3-component
            // FragCoord variable, invalid SPIR-V). PSSL spells them
            // float3 (`float3 position : S_POSITION`); widen the
            // parameter â€” xyz values are identical and bodies use
            // components. Other widths/shadings keep today's loud path.
            for p in &mut entry.parameters {
                let is_pos = p.semantic.as_deref().is_some_and(|s| {
                    s.eq_ignore_ascii_case("S_POSITION") || s.eq_ignore_ascii_case("SV_Position")
                });
                if is_pos
                    && matches!(
                        &p.ty,
                        crate::ir::IrType::Vector {
                            base: crate::ir::IrScalar::Float,
                            width: 3
                        }
                    )
                {
                    p.ty = crate::ir::IrType::Vector {
                        base: crate::ir::IrScalar::Float,
                        width: 4,
                    };
                }
            }
        }
    }
    let Some(entry) = ir.functions.iter().find(|f| f.name == entry_name) else {
        return;
    };
    let mut io_structs: Vec<String> = Vec::new();
    for p in &entry.parameters {
        if p.semantic
            .as_deref()
            .is_some_and(|s| s.eq_ignore_ascii_case("S_SRT_DATA"))
        {
            continue;
        }
        if let crate::ir::IrType::Named(s) = &p.ty {
            if !io_structs.contains(s) {
                io_structs.push(s.clone());
            }
        }
        // Geometry-shader stream parameters (`inout TriangleStream<GsOut>`
        // â€” the output struct hides inside the resource element, so the
        // bare pass above misses it and bare members fail DXC with
        // `semantic string missing`). Stream kinds are entry-IO-only by
        // construction (never buffer elements), so collecting the element
        // struct is safe; other resource elements keep today's loud path.
        if let crate::ir::IrType::Resource { kind, element, .. } = &p.ty {
            let k = kind.to_ascii_lowercase();
            if k == "trianglestream" || k == "pointstream" || k == "linestream" {
                if let crate::ir::IrType::Named(s) = element.as_ref() {
                    if !io_structs.contains(s) {
                        io_structs.push(s.clone());
                    }
                }
            }
        }
    }
    // Vertex-like outputs are varyings too (fragment outputs are targets,
    // compute returns void).
    if matches!(
        &ir.stage,
        Some(Stage::Vertex) | Some(Stage::Geometry) | Some(Stage::Hull) | Some(Stage::Domain)
    ) {
        if let crate::ir::IrType::Named(s) = &entry.return_type {
            if !io_structs.contains(s) {
                io_structs.push(s.clone());
            }
        }
    }
    // No early return on empty `io_structs`: the struct loop below
    // no-ops then (per-struct `continue`), and the bare-entry-param
    // COLOR pass at the end must still run (entries with no IO structs
    // at all, e.g. `main(uv : TEXCOORD0, color : COLOR0)`).
    let is_vertex_entry = matches!(&ir.stage, Some(Stage::Vertex));
    // Declared-struct name set (owned): classifies struct-typed members
    // without borrowing `ir` across the mutation loop below.
    let struct_names: std::collections::HashSet<String> =
        ir.structs.iter().map(|s| s.name.clone()).collect();
    for idx in 0..ir.structs.len() {
        if !io_structs.contains(&ir.structs[idx].name) {
            continue;
        }
        // Pool seeds from the whole subtree (direct + nested explicits,
        // freshly read each iteration so fills assigned to earlier
        // structs are visible): MULTIEXPORT's `numVerts : TEXCOORD1`
        // must survive `vertex`'s inner leaves. Flat structs contribute
        // only their own fields: no-op for them.
        let mut used =
            io_subtree_texcoords(&ir.structs, &struct_names, &ir.structs[idx].name.clone());
        let st = &mut ir.structs[idx];
        // COLOR-family members (`COLOR` = 0, `COLORn` = n, mirroring
        // `map_semantic`) claim their natural index so bare members skip
        // it; on clash with an explicit TEXCOORD they relocate to the
        // lowest free index instead of colliding (DXC `used more than
        // once`). Same deterministic rule producer/consumer sides share.
        // Structs without COLOR fields â€” or without clash â€” are untouched.
        //
        // NGG index semantics have no direct Vulkan spelling and are
        // handled alongside: `S_NGG_VERTEX/INSTANCE_INDEX` are exactly
        // the system IDs (sample headers say so) in VERTEX entries, but
        // other stages (e.g. geometry) reject system inputs, so there
        // they join the TEXCOORD pool; `S_NGG_GROUP_VERTEX_INDEXn`
        // (subgroup-relative indices) always does. Other `S_NGG_*`
        // (outputs, wave index) keep today's loud path.
        let mut ngg_system: Vec<usize> = Vec::new();
        for (idx, f) in st.fields.iter().enumerate() {
            let Some(sem) = f.semantic.as_deref() else {
                continue;
            };
            let upper = sem.to_ascii_uppercase();
            if upper == "S_NGG_VERTEX_INDEX" || upper == "S_NGG_INSTANCE_INDEX" {
                ngg_system.push(idx);
            }
        }
        for idx in ngg_system {
            let sem = st.fields[idx].semantic.as_deref().unwrap_or("");
            if is_vertex_entry {
                let sys = if sem.eq_ignore_ascii_case("S_NGG_VERTEX_INDEX") {
                    "SV_VertexID"
                } else {
                    "SV_InstanceID"
                };
                st.fields[idx].semantic = Some(sys.to_string());
            } else {
                // Join the TEXCOORD pool below via a placeholder the
                // claim loop treats as bare-adjacent: rewrite to an
                // explicit free index directly.
                let mut k = 0u32;
                while used.contains(&k) {
                    k += 1;
                }
                st.fields[idx].semantic = Some(format!("TEXCOORD{k}"));
                used.push(k);
            }
        }
        let claimed_index = |sem: &str| -> Option<u32> {
            let upper = sem.to_ascii_uppercase();
            if upper == "COLOR" {
                return Some(0);
            }
            if let Some(n) = upper
                .strip_prefix("COLOR")
                .and_then(|r| r.parse::<u32>().ok())
            {
                return Some(n);
            }
            upper
                .strip_prefix("S_NGG_GROUP_VERTEX_INDEX")
                .and_then(|r| r.parse::<u32>().ok())
        };
        for f in &mut st.fields {
            let Some(sem) = f.semantic.as_deref() else {
                continue;
            };
            let Some(natural) = claimed_index(sem) else {
                continue;
            };
            if used.contains(&natural) {
                let mut k = 0u32;
                while used.contains(&k) {
                    k += 1;
                }
                f.semantic = Some(format!("TEXCOORD{k}"));
                used.push(k);
            } else {
                used.push(natural);
            }
        }
        let mut next = 0u32;
        for f in &mut st.fields {
            if f.semantic.is_some() {
                continue;
            }
            // Struct-typed members flatten in DX linkage: a semantic on
            // the member itself is invalid and the leaves link
            // individually, so the tree pass below assigns them. Claiming
            // a pool index here would both emit an invalid semantic and
            // waste the slot (and collide once the leaves fill it).
            if let crate::ir::IrType::Named(n) = &f.ty {
                if f.array_size.is_none() && struct_names.contains(n) {
                    continue;
                }
            }
            while used.contains(&next) {
                next += 1;
            }
            f.semantic = Some(format!("TEXCOORD{next}"));
            used.push(next);
        }
    }
    // Nested-varying tree fill (see `plan_nested_fill`): bare leaves
    // under struct-typed IO members gain subtree-free TEXCOORDs and
    // enclosing member semantics drop. Roots whose tree shares one
    // definition across instances stay loud (no silent aliasing).
    {
        let mut tree_filled: std::collections::HashSet<String> =
            std::collections::HashSet::new();
        for root in io_structs.clone() {
            let Some((assigns, drops)) =
                plan_nested_fill(&ir.structs, &struct_names, &root, &tree_filled)
            else {
                continue;
            };
            for (sname, ..) in &assigns {
                tree_filled.insert(sname.clone());
            }
            for (sname, fname, sem) in assigns {
                if let Some(decl) = ir.structs.iter_mut().find(|s| s.name == sname) {
                    if let Some(f) = decl.fields.iter_mut().find(|f| f.name == fname) {
                        f.semantic = Some(sem);
                    }
                }
            }
            for (sname, fname) in drops {
                if let Some(decl) = ir.structs.iter_mut().find(|s| s.name == sname) {
                    if let Some(f) = decl.fields.iter_mut().find(|f| f.name == fname) {
                        f.semantic = None;
                    }
                }
            }
        }
    }
    // Bare entry parameters (`float4 main(float2 uv : TEXCOORD0, float4
    // color : COLOR0)`): COLOR-family params claim their natural index
    // exactly like COLOR struct members, so a COLOR param sharing an
    // index with an explicit TEXCOORD param relocates to the lowest free
    // index (same deterministic rule both sides share). Params without
    // COLOR semantics â€” including bare params with no semantic â€” are
    // untouched, as are shaders without a clash. The pool seeds from the
    // (already relocated) IO-struct members plus the params' own explicit
    // TEXCOORDs so cross-shape clashes relocate too.
    {
        let mut used: Vec<u32> = Vec::new();
        // Seed from the IO subtrees (fresh post-tree-fill state), so
        // params never claim a nested leaf's index.
        for name in &io_structs {
            for n in io_subtree_texcoords(&ir.structs, &struct_names, name) {
                if !used.contains(&n) {
                    used.push(n);
                }
            }
        }
        if let Some(entry) = ir.functions.iter().find(|f| f.name == entry_name) {
            for p in &entry.parameters {
                if p.semantic
                    .as_deref()
                    .is_some_and(|s| s.eq_ignore_ascii_case("S_SRT_DATA"))
                {
                    continue;
                }
                if let Some(sem) = p.semantic.as_deref() {
                    if let Some(rest) = sem.to_ascii_uppercase().strip_prefix("TEXCOORD") {
                        if let Ok(n) = rest.parse::<u32>() {
                            if !used.contains(&n) {
                                used.push(n);
                            }
                        }
                    }
                }
            }
        }
        let claimed_index = |sem: &str| -> Option<u32> {
            let upper = sem.to_ascii_uppercase();
            if upper == "COLOR" {
                return Some(0);
            }
            upper
                .strip_prefix("COLOR")
                .and_then(|r| r.parse::<u32>().ok())
        };
        if let Some(entry) = ir.functions.iter_mut().find(|f| f.name == entry_name) {
            for p in &mut entry.parameters {
                if p.semantic
                    .as_deref()
                    .is_some_and(|s| s.eq_ignore_ascii_case("S_SRT_DATA"))
                {
                    continue;
                }
                let Some(sem) = p.semantic.clone() else {
                    continue;
                };
                let Some(natural) = claimed_index(&sem) else {
                    continue;
                };
                if used.contains(&natural) {
                    let mut k = 0u32;
                    while used.contains(&k) {
                        k += 1;
                    }
                    p.semantic = Some(format!("TEXCOORD{k}"));
                    used.push(k);
                } else {
                    used.push(natural);
                }
            }
        }
        // Bare numeric entry params (`float4 main(float2 uv)` with no
        // semantic — the source links them by member name/order): DXC requires
        // a semantic string on every entry input (`semantic string
        // missing`). Assign the lowest free TEXCOORD index from the same
        // pool, so producer/consumer sides sharing field order agree.
        // Resources keep no semantic (bindings, not varyings), structs
        // keep none (members carry their own), and compute (or other
        // non-raster) entries keep today's loud path â€” a bare compute
        // param is a uniform that belongs in a cbuffer, not a varying.
        if matches!(&ir.stage, Some(Stage::Vertex) | Some(Stage::Fragment)) {
            if let Some(entry) = ir.functions.iter_mut().find(|f| f.name == entry_name) {
                for p in &mut entry.parameters {
                    if p.semantic.is_some() {
                        continue;
                    }
                    let is_numeric = matches!(
                        &p.ty,
                        crate::ir::IrType::Bool
                            | crate::ir::IrType::Int
                            | crate::ir::IrType::UInt
                            | crate::ir::IrType::Float
                            | crate::ir::IrType::Half
                            | crate::ir::IrType::Double
                            | crate::ir::IrType::Vector { .. }
                            | crate::ir::IrType::Matrix { .. }
                    );
                    if !is_numeric {
                        continue;
                    }
                    let mut k = 0u32;
                    while used.contains(&k) {
                        k += 1;
                    }
                    p.semantic = Some(format!("TEXCOORD{k}"));
                    used.push(k);
                }
            }
        }
    }
}

/// Explicit `TEXCOORDn` indices anywhere in a struct subtree (the root's
/// own fields plus, recursively, non-array members of declared struct
/// type). Cycle-guarded; flat structs contribute only their own fields.
///
/// PSSL nests varyings (`MULTIEXPORT_GS_OUTPUT.vertex : TEXCOORD0` whose
/// `VS_OUTPUT` leaves carry the real linkage): the entry-varying pool
/// must see nested explicits (e.g. sibling `numVerts : TEXCOORD1`) or
/// sibling bare fills collide with them (`used more than once`).
fn io_subtree_texcoords(
    structs: &[StructDecl],
    struct_names: &std::collections::HashSet<String>,
    root: &str,
) -> Vec<u32> {
    fn walk(
        structs: &[StructDecl],
        struct_names: &std::collections::HashSet<String>,
        name: &str,
        seen: &mut Vec<String>,
        out: &mut Vec<u32>,
    ) {
        if seen.iter().any(|s| s == name) {
            return;
        }
        seen.push(name.to_string());
        let Some(decl) = structs.iter().find(|s| s.name == name) else {
            return;
        };
        for f in &decl.fields {
            if let Some(sem) = f.semantic.as_deref() {
                if let Some(rest) = sem.to_ascii_uppercase().strip_prefix("TEXCOORD") {
                    if let Ok(n) = rest.parse::<u32>() {
                        if !out.contains(&n) {
                            out.push(n);
                        }
                    }
                }
            }
            if let crate::ir::IrType::Named(n) = &f.ty {
                if f.array_size.is_none() && struct_names.contains(n) {
                    walk(structs, struct_names, n, seen, out);
                }
            }
        }
    }
    let mut out = Vec::new();
    let mut seen = Vec::new();
    walk(structs, struct_names, root, &mut seen, &mut out);
    out
}

/// Plan nested-varying fills for one IO root: bare leaves under
/// non-array struct-typed members gain the lowest free `TEXCOORD` index
/// (pool = the root subtree's current explicits), and enclosing
/// struct-typed members shed their semantic (a semantic on a
/// struct-typed member is invalid DX linkage: `invalid usage of
/// semantic 'TEXCOORD0' in shader profile gs`, and DXC otherwise
/// flattens the member onto the sibling indices).
///
/// Returns `None` when the tree shares one struct definition through
/// two paths with fills outstanding (per-instance locations cannot
/// share one definition) or touches `shared_filled` (assigned by an
/// earlier root): the caller then leaves the root loud rather than
/// alias two instances onto one index. `Some` with empty vectors is a
/// no-op (flat or fully-linked roots are untouched).
#[allow(clippy::type_complexity)]
fn plan_nested_fill(
    structs: &[StructDecl],
    struct_names: &std::collections::HashSet<String>,
    root: &str,
    shared_filled: &std::collections::HashSet<String>,
) -> Option<(Vec<(String, String, String)>, Vec<(String, String)>)> {
    fn visit(
        structs: &[StructDecl],
        struct_names: &std::collections::HashSet<String>,
        name: &str,
        pool: &mut Vec<u32>,
        assigns: &mut Vec<(String, String, String)>,
        drops: &mut Vec<(String, String)>,
        filled: &mut std::collections::HashSet<String>,
        stack: &mut Vec<String>,
        is_root: bool,
    ) -> bool {
        if stack.iter().any(|s| s == name) {
            return true;
        }
        if !is_root && filled.contains(name) {
            return false;
        }
        let Some(decl) = structs.iter().find(|s| s.name == name) else {
            return true;
        };
        stack.push(name.to_string());
        let mut assigned_here = false;
        // Field snapshot: `decl` borrows `structs` while `pool`/`assigns`
        // grow independently (owned locals), so no aliasing.
        let fields: Vec<(String, crate::ir::IrType, Option<u32>, Option<String>)> = decl
            .fields
            .iter()
            .map(|f| (f.name.clone(), f.ty.clone(), f.array_size, f.semantic.clone()))
            .collect();
        for (fname, fty, arr, sem) in &fields {
            let sub = match fty {
                crate::ir::IrType::Named(n)
                    if arr.is_none() && struct_names.contains(n) =>
                {
                    Some(n.clone())
                }
                _ => None,
            };
            if let Some(sub) = sub {
                if sem.is_some() {
                    drops.push((name.to_string(), fname.clone()));
                }
                if !visit(
                    structs,
                    struct_names,
                    &sub,
                    pool,
                    assigns,
                    drops,
                    filled,
                    stack,
                    false,
                ) {
                    return false;
                }
            } else if sem.is_none() {
                let mut k = 0u32;
                while pool.contains(&k) {
                    k += 1;
                }
                pool.push(k);
                assigns.push((name.to_string(), fname.clone(), format!("TEXCOORD{k}")));
                assigned_here = true;
            }
        }
        stack.pop();
        if assigned_here {
            filled.insert(name.to_string());
        }
        true
    }
    let mut pool = io_subtree_texcoords(structs, struct_names, root);
    let mut assigns = Vec::new();
    let mut drops = Vec::new();
    let mut filled = shared_filled.clone();
    let mut stack = Vec::new();
    if !visit(
        structs,
        struct_names,
        root,
        &mut pool,
        &mut assigns,
        &mut drops,
        &mut filled,
        &mut stack,
        true,
    ) {
        return None;
    }
    Some((assigns, drops))
}

/// Folds literal-`0` subscripts on provably scalar, non-array bases.
///
/// PSSL permits indexing a scalar (`int light_count; ...
/// scene_data.light_count[0]` yields the scalar); HLSL rejects it.
/// Only fires when the base resolves (params, locals, globals, dotted
/// member paths, buffer-element paths) to a scalar type with no array
/// size. Vector/matrix bases keep indexing (valid components);
/// shadowed or multiply-declared names veto; everything else stays
/// loud.
fn lower_scalar_zero_subscripts(ir: &mut ShaderModule) {
    use crate::ir::{IrExpr, IrLiteral, IrStmt};
    use std::collections::{HashMap, HashSet};

    fn is_scalar_ty(ty: &crate::ir::IrType) -> bool {
        matches!(
            ty,
            crate::ir::IrType::Bool
                | crate::ir::IrType::Int
                | crate::ir::IrType::UInt
                | crate::ir::IrType::Float
                | crate::ir::IrType::Half
                | crate::ir::IrType::Double
        )
    }
    fn is_zero_index(e: &IrExpr) -> bool {
        if let IrExpr::Literal(IrLiteral::Number(s)) = e {
            let t = s.trim_end_matches(['u', 'U', 'l', 'L']);
            return t.parse::<u32>().is_ok_and(|v| v == 0);
        }
        false
    }
    // Resolve a path expression to (type, array_size): Variable roots
    // from `env`, Member steps through `fields`, Index steps consume one
    // array level (or a buffer element / vector component / matrix row).
    // `None` when unresolvable.
    #[allow(clippy::too_many_arguments)]
    fn resolve_path(
        e: &IrExpr,
        env: &HashMap<String, (crate::ir::IrType, Option<u32>)>,
        fields: &HashMap<String, HashMap<String, (crate::ir::IrType, Option<u32>)>>,
    ) -> Option<(crate::ir::IrType, Option<u32>)> {
        match e {
            IrExpr::Variable(v) => env.get(v).cloned(),
            IrExpr::Member { object, member } => {
                let (ty, _) = resolve_path(object, env, fields)?;
                if let crate::ir::IrType::Named(s) = &ty {
                    let f = fields.get(s.as_str())?.get(member.as_str())?;
                    Some((f.0.clone(), f.1))
                } else {
                    None
                }
            }
            IrExpr::Index { object, .. } | IrExpr::ResourceIndex { resource: object, .. } => {
                // Buffer-element / array-element step. An ARRAY OF
                // BUFFERS indexed selects one buffer (`bufs[k]` stays a
                // buffer); only a directly-indexed buffer yields its
                // element, and only a plain array yields its element
                // type. (Order matters: the plain-`Resource` arm must
                // not fire for arrays of buffers.)
                let (ty, arr) = resolve_path(object, env, fields)?;
                match (ty, arr) {
                    (t @ crate::ir::IrType::Resource { .. }, Some(_)) => Some((t, None)),
                    (crate::ir::IrType::Resource { element, .. }, None) => {
                        Some((*element, None))
                    }
                    (t, Some(_)) => Some((t, None)),
                    (crate::ir::IrType::Vector { base, .. }, None) => {
                        let scalar = match base {
                            crate::ir::IrScalar::Float => crate::ir::IrType::Float,
                            crate::ir::IrScalar::Half => crate::ir::IrType::Half,
                            crate::ir::IrScalar::Int => crate::ir::IrType::Int,
                            crate::ir::IrScalar::UInt => crate::ir::IrType::UInt,
                            crate::ir::IrScalar::Bool => crate::ir::IrType::Bool,
                        };
                        Some((scalar, None))
                    }
                    (crate::ir::IrType::Matrix { base, columns, .. }, None) => {
                        Some((
                            crate::ir::IrType::Vector {
                                base,
                                width: columns,
                            },
                            None,
                        ))
                    }
                    _ => None,
                }
            }
            _ => None,
        }
    }
    // struct name -> field name -> (ty, array_size).
    let fields: HashMap<String, HashMap<String, (crate::ir::IrType, Option<u32>)>> = ir
        .structs
        .iter()
        .map(|s| {
            (
                s.name.clone(),
                s.fields
                    .iter()
                    .map(|f| (f.name.clone(), (f.ty.clone(), f.array_size)))
                    .collect(),
            )
        })
        .collect();
    let globals: HashMap<String, (crate::ir::IrType, Option<u32>)> = ir
        .globals
        .iter()
        .map(|g| (g.name.clone(), (g.ty.clone(), g.array_size)))
        .collect();
    for func in &mut ir.functions {
        // Name -> (ty, array_size) for params + locals. Multiply-declared
        // names veto (shadowing makes a flat map unsound); globals fill
        // only names the function never declares.
        let mut env: HashMap<String, (crate::ir::IrType, Option<u32>)> = HashMap::new();
        let mut dup: HashSet<String> = HashSet::new();
        let mut declare = |name: String, ty: crate::ir::IrType, arr: Option<u32>| {
            if env.contains_key(&name) || dup.contains(&name) {
                env.remove(&name);
                dup.insert(name);
            } else {
                env.insert(name, (ty, arr));
            }
        };
        for p in &func.parameters {
            declare(p.name.clone(), p.ty.clone(), p.array_size);
        }
        fn collect_locals(
            stmts: &[IrStmt],
            declare: &mut impl FnMut(String, crate::ir::IrType, Option<u32>),
        ) {
            for s in stmts {
                match s {
                    IrStmt::VarDecl { ty, name, array_size, .. } => {
                        declare(name.clone(), ty.clone(), *array_size);
                    }
                    IrStmt::Block(ss) => collect_locals(ss, declare),
                    IrStmt::If { then_branch, else_branch, .. } => {
                        collect_locals(then_branch, declare);
                        if let Some(els) = else_branch {
                            collect_locals(els, declare);
                        }
                    }
                    IrStmt::For { init, body, .. } => {
                        collect_locals(std::slice::from_ref(init.as_ref()), declare);
                        collect_locals(body, declare);
                    }
                    IrStmt::While { body, .. } => collect_locals(body, declare),
                    IrStmt::Switch { cases, .. } => {
                        for c in cases {
                            collect_locals(&c.body, declare);
                        }
                    }
                    _ => {}
                }
            }
        }
        collect_locals(&func.body, &mut declare);
        for (name, entry) in &globals {
            if !env.contains_key(name) && !dup.contains(name) {
                env.insert(name.clone(), entry.clone());
            }
        }
        fn fold_expr(
            e: &mut IrExpr,
            env: &HashMap<String, (crate::ir::IrType, Option<u32>)>,
            fields: &HashMap<String, HashMap<String, (crate::ir::IrType, Option<u32>)>>,
        ) {
            // Bottom-up: children first so nested folds complete.
            match e {
                IrExpr::Call { arguments, .. } | IrExpr::Construct { arguments, .. } => {
                    for a in arguments {
                        fold_expr(a, env, fields);
                    }
                }
                IrExpr::Member { object, .. } => fold_expr(object, env, fields),
                IrExpr::Index { object, index } | IrExpr::ResourceIndex { resource: object, index } => {
                    fold_expr(object, env, fields);
                    fold_expr(index, env, fields);
                }
                IrExpr::ResourceLoad { resource, coordinates, sample } => {
                    fold_expr(resource, env, fields);
                    fold_expr(coordinates, env, fields);
                    if let Some(s) = sample {
                        fold_expr(s, env, fields);
                    }
                }
                IrExpr::Unary { expr: inner, .. } => fold_expr(inner, env, fields),
                IrExpr::Binary { left, right, .. } => {
                    fold_expr(left, env, fields);
                    fold_expr(right, env, fields);
                }
                IrExpr::Conditional { condition, then_branch, else_branch } => {
                    fold_expr(condition, env, fields);
                    fold_expr(then_branch, env, fields);
                    fold_expr(else_branch, env, fields);
                }
                IrExpr::Assign { lhs, rhs } => {
                    fold_expr(lhs, env, fields);
                    fold_expr(rhs, env, fields);
                }
                IrExpr::InitializerList(elems) => {
                    for x in elems {
                        fold_expr(x, env, fields);
                    }
                }
                _ => {}
            }
            if let IrExpr::Index { object, index } = e {
                if is_zero_index(index) {
                    if let Some((ty, arr)) = resolve_path(object, env, fields) {
                        if arr.is_none() && is_scalar_ty(&ty) {
                            *e = (**object).clone();
                        }
                    }
                }
            }
        }
        fn fold_stmt(
            s: &mut IrStmt,
            env: &HashMap<String, (crate::ir::IrType, Option<u32>)>,
            fields: &HashMap<String, HashMap<String, (crate::ir::IrType, Option<u32>)>>,
        ) {
            match s {
                IrStmt::Return(Some(e)) | IrStmt::Expr(e) => fold_expr(e, env, fields),
                IrStmt::Return(None) => {}
                IrStmt::VarDecl { init: Some(e), .. } => fold_expr(e, env, fields),
                IrStmt::VarDecl { init: None, .. } => {}
                IrStmt::Block(ss) => {
                    for x in ss {
                        fold_stmt(x, env, fields);
                    }
                }
                IrStmt::If { condition, then_branch, else_branch } => {
                    fold_expr(condition, env, fields);
                    for x in then_branch.iter_mut().chain(else_branch.iter_mut().flatten()) {
                        fold_stmt(x, env, fields);
                    }
                }
                IrStmt::For { init, condition, update, body } => {
                    fold_stmt(init.as_mut(), env, fields);
                    if let Some(c) = condition {
                        fold_expr(c, env, fields);
                    }
                    if let Some(u) = update {
                        fold_expr(u, env, fields);
                    }
                    for x in body {
                        fold_stmt(x, env, fields);
                    }
                }
                IrStmt::While { condition, body } => {
                    fold_expr(condition, env, fields);
                    for x in body {
                        fold_stmt(x, env, fields);
                    }
                }
                IrStmt::Switch { scrutinee, cases } => {
                    fold_expr(scrutinee, env, fields);
                    for c in cases {
                        if let Some(l) = &mut c.label {
                            fold_expr(l, env, fields);
                        }
                        for x in &mut c.body {
                            fold_stmt(x, env, fields);
                        }
                    }
                }
            }
        }
        for s in &mut func.body {
            fold_stmt(s, &env, &fields);
        }
        // Entry default-arg expressions (lowered with the parameter env).
        // Defaults cannot declare locals; params alone seed lookup.
        let penv: HashMap<String, (crate::ir::IrType, Option<u32>)> = func
            .parameters
            .iter()
            .map(|q| (q.name.clone(), (q.ty.clone(), q.array_size)))
            .collect();
        for p in &mut func.parameters {
            if let Some(d) = &mut p.default {
                fold_expr(d, &penv, &fields);
            }
        }
    }
}

/// Fold struct-scoped enumerator uses (`S::E`) to bare names (`E`).
///
/// Nested struct enums (`enum Flags { DoubleSided = 1, Transparent = 2 }`
/// inside `MaterialSRT`), top-level enums used as values
/// (`OutputEncoding::PQ`), and namespace-qualified constants all spell
/// scope explicitly but have no HLSL spelling; the parser registers
/// enumerators as `int` globals (values are shader-irrelevant for
/// validation â€” only names matter), so qualification is redundant.
/// Folds only when `E` names an int-typed module global, `E` is not
/// shadowed by a parameter/local of the using function, the scope head
/// is not itself a global (instance access is never scope), the head is
/// not `sce` (dedicated framework paths own those), and `E` is not
/// `value`/`type` (SFINAE members, never bare enumerators). Anything
/// else keeps failing loudly at DXC.
fn fold_scoped_enum_uses(ir: &mut ShaderModule) {
    use crate::ir::{IrExpr, IrStmt};
    use std::collections::HashSet;

    let globals: HashSet<String> = ir.globals.iter().map(|g| g.name.clone()).collect();
    let int_globals: HashSet<String> = ir
        .globals
        .iter()
        .filter(|g| {
            matches!(
                &g.ty,
                crate::ir::IrType::Int | crate::ir::IrType::UInt
            )
        })
        .map(|g| g.name.clone())
        .collect();
    if globals.is_empty() {
        return;
    }
    fn fold_expr(e: &mut IrExpr, int_globals: &HashSet<String>, globals: &HashSet<String>, shadowed: &HashSet<String>) {
        match e {
            IrExpr::Variable(v) => {
                if v.contains("::") && !shadowed.contains(v.as_str()) {
                    let parts: Vec<&str> = v.split("::").collect();
                    if parts.len() >= 2 && parts[0] != "sce" && !parts[0].is_empty() {
                        let last = parts[parts.len() - 1];
                        // Scope head must not be a value (instance
                        // access is never scope); the tail must be an
                        // int-typed global (enumerators), unshadowed,
                        // and never an SFINAE member.
                        if !globals.contains(parts[0])
                            && int_globals.contains(last)
                            && !shadowed.contains(last)
                            && last != "value"
                            && last != "type"
                        {
                            *v = last.to_string();
                        }
                    }
                }
            }
            IrExpr::Member { object, .. } => fold_expr(object, int_globals, globals, shadowed),
            IrExpr::Index { object, index } | IrExpr::ResourceIndex { resource: object, index } => {
                fold_expr(object, int_globals, globals, shadowed);
                fold_expr(index, int_globals, globals, shadowed);
            }
            IrExpr::ResourceLoad {
                resource,
                coordinates,
                sample,
            } => {
                fold_expr(resource, int_globals, globals, shadowed);
                fold_expr(coordinates, int_globals, globals, shadowed);
                if let Some(sample) = sample {
                    fold_expr(sample, int_globals, globals, shadowed);
                }
            }
            IrExpr::Call { arguments, .. } => {
                // Scoped callees (`S::method`) are method calls, not
                // enumerators â€” leave the callee, fold the arguments.
                for a in arguments {
                    fold_expr(a, int_globals, globals, shadowed);
                }
            }
            IrExpr::Construct { arguments, .. } => {
                for a in arguments {
                    fold_expr(a, int_globals, globals, shadowed);
                }
            }
            IrExpr::Unary { expr: inner, .. } => fold_expr(inner, int_globals, globals, shadowed),
            IrExpr::Binary { left, right, .. } => {
                fold_expr(left, int_globals, globals, shadowed);
                fold_expr(right, int_globals, globals, shadowed);
            }
            IrExpr::Conditional {
                condition,
                then_branch,
                else_branch,
            } => {
                fold_expr(condition, int_globals, globals, shadowed);
                fold_expr(then_branch, int_globals, globals, shadowed);
                fold_expr(else_branch, int_globals, globals, shadowed);
            }
            IrExpr::Assign { lhs, rhs } => {
                fold_expr(lhs, int_globals, globals, shadowed);
                fold_expr(rhs, int_globals, globals, shadowed);
            }
            IrExpr::InitializerList(elems) => {
                for x in elems {
                    fold_expr(x, int_globals, globals, shadowed);
                }
            }
            IrExpr::Literal(_) => {}
        }
    }
    fn fold_stmt(s: &mut IrStmt, int_globals: &HashSet<String>, globals: &HashSet<String>, shadowed: &HashSet<String>) {
        match s {
            IrStmt::Return(e) => {
                if let Some(e) = e {
                    fold_expr(e, int_globals, globals, shadowed);
                }
            }
            IrStmt::VarDecl { init, .. } => {
                if let Some(init) = init {
                    fold_expr(init, int_globals, globals, shadowed);
                }
            }
            IrStmt::Block(inner) => {
                for s in inner {
                    fold_stmt(s, int_globals, globals, shadowed);
                }
            }
            IrStmt::If {
                condition,
                then_branch,
                else_branch,
            } => {
                fold_expr(condition, int_globals, globals, shadowed);
                for s in then_branch {
                    fold_stmt(s, int_globals, globals, shadowed);
                }
                if let Some(b) = else_branch {
                    for s in b {
                        fold_stmt(s, int_globals, globals, shadowed);
                    }
                }
            }
            IrStmt::For {
                init,
                condition,
                update,
                body,
            } => {
                fold_stmt(init, int_globals, globals, shadowed);
                if let Some(condition) = condition {
                    fold_expr(condition, int_globals, globals, shadowed);
                }
                if let Some(update) = update {
                    fold_expr(update, int_globals, globals, shadowed);
                }
                for s in body {
                    fold_stmt(s, int_globals, globals, shadowed);
                }
            }
            IrStmt::While { condition, body } => {
                fold_expr(condition, int_globals, globals, shadowed);
                for s in body {
                    fold_stmt(s, int_globals, globals, shadowed);
                }
            }
            IrStmt::Switch { scrutinee, cases } => {
                fold_expr(scrutinee, int_globals, globals, shadowed);
                for c in cases {
                    if let Some(label) = &mut c.label {
                        fold_expr(label, int_globals, globals, shadowed);
                    }
                    for s in &mut c.body {
                        fold_stmt(s, int_globals, globals, shadowed);
                    }
                }
            }
            IrStmt::Expr(e) => fold_expr(e, int_globals, globals, shadowed),
        }
    }
    fn collect_locals(s: &IrStmt, out: &mut HashSet<String>) {
        match s {
            IrStmt::VarDecl { name, .. } => {
                out.insert(name.clone());
            }
            IrStmt::Block(inner) => {
                for s in inner {
                    collect_locals(s, out);
                }
            }
            IrStmt::If {
                then_branch,
                else_branch,
                ..
            } => {
                for s in then_branch {
                    collect_locals(s, out);
                }
                if let Some(b) = else_branch {
                    for s in b {
                        collect_locals(s, out);
                    }
                }
            }
            IrStmt::For { init, body, .. } => {
                collect_locals(init, out);
                for s in body {
                    collect_locals(s, out);
                }
            }
            IrStmt::While { body, .. } => {
                for s in body {
                    collect_locals(s, out);
                }
            }
            IrStmt::Switch { cases, .. } => {
                for c in cases {
                    for s in &c.body {
                        collect_locals(s, out);
                    }
                }
            }
            _ => {}
        }
    }
    for func in &mut ir.functions {
        let mut shadowed: HashSet<String> = HashSet::new();
        for p in &func.parameters {
            shadowed.insert(p.name.clone());
        }
        for s in &func.body {
            collect_locals(s, &mut shadowed);
        }
        for s in &mut func.body {
            fold_stmt(s, &int_globals, &globals, &shadowed);
        }
    }
}

fn lower_struct(decl: &crate::ast::StructDecl) -> StructDecl {    let fields = decl
        .fields
        .iter()
        .map(|f| FieldDecl {
            ty: to_ir_type(&f.ty),
            name: f.name.clone(),
            semantic: f.semantic.as_ref().map(|s| s.name().to_string()),
            array_size: f.array_size,
        })
        .collect();

    StructDecl {
        name: decl.name.clone(),
        fields,
    }
}

fn lower_resource(decl: &crate::ast::ResourceDecl) -> Resource {
    let normalized = decl.kind.to_ascii_lowercase().replace('_', "");

    let resource_type = match normalized.as_str() {
        "texture1d" | "texture2d" | "texture2darray" | "texturecube" | "texturecubearray"
        | "texture3d" => ResourceType::Texture,

        "samplerstate" | "samplercomparisonstate" => ResourceType::Sampler,

        "rwtexture2d" | "rwtexture2darray" | "rwtexture3d" | "rwtexture1d" => {
            ResourceType::RWTexture
        }

        "databuffer"
            | "regularbuffer"
            | "structuredbuffer"
            | "buffer"
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

        "constantbuffer" | "cbuffer" => ResourceType::ConstantBuffer,

        _ => ResourceType::StructuredBuffer,
    };

    let element_type = decl.element_ty.as_ref().map(to_ir_type);
    let binding = decl.register.map(|(_, n)| n);
    let register = decl.register.map(|(_, n)| n);

    let fields = decl
        .fields
        .iter()
        .map(|f| FieldDecl {
            ty: to_ir_type(&f.ty),
            name: f.name.clone(),
            semantic: f.semantic.as_ref().map(|s| s.name().to_string()),
            array_size: f.array_size,
        })
        .collect();

    // HLSL `sampler`/`texture` are reserved keywords.
    //
    // Rename the PSSL resource here and rewrite all references in function
    // bodies via `rename_sampler_in_stmt`.
    let name = if let Some(renamed) = rename::reserved_hlsl_rename(&decl.name) {
        renamed.to_string()
    } else {
        decl.name.clone()
    };

    Resource {
        name,
        resource_type,
        kind: normalized,
        element_type,
        descriptor_set: Some(0),
        binding,
        register,
        space: decl.register_space,
        array_size: None,
        fields,
    }
}

/// Collects `__user_sgpr{N}` indices referenced anywhere in lowered
/// function bodies, so `lower_module` can declare backing globals.
fn collect_user_sgpr_indices(ir: &ShaderModule) -> BTreeSet<u32> {
    use crate::ir::{IrExpr, IrStmt};
    fn expr(e: &IrExpr, out: &mut BTreeSet<u32>) {
        match e {
            IrExpr::Variable(name) => {
                if let Some(index) = crate::framework::user_sgpr_index(name) {
                    out.insert(index);
                }
            }
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
            IrExpr::Call { arguments, .. } | IrExpr::Construct { arguments, .. } => {
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
    fn stmt(s: &IrStmt, out: &mut BTreeSet<u32>) {
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
    let mut out = BTreeSet::new();
    for f in &ir.functions {
        for s in &f.body {
            stmt(s, &mut out);
        }
    }
    out
}
