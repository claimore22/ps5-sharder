//! NGG primitive-shader cull-data builtin (`Ngg::CullData<T>`).
//!
//! Provided by the PSSL compiler for `[CULL_FUNC]`-attributed NGG primitive
//! shaders; no header defines it, so it reaches lowering as an opaque
//! scoped call (the parser preserves `<T>` for `CullData` only). Vulkan
//! has no per-primitive cull hook inside a vertex shader, so the compat
//! layer conservatively disables culling:
//! - `Ngg::CullData<T>::Set(v);` statements are no-ops → dropped.
//! - `T name[3] = Ngg::CullData<T>::GetTriangle();` (or `auto name = ...`)
//!   becomes a zero-initialized `T[3]`. Zeroed cull data never culls (all
//!   clip flags clear, zero area), so rendering stays correct.
//!
//! `Ngg::` is only ever `Ngg::CullData<...>` corpus-wide (four source lines:
//! `Set`/`GetTriangle` × `MyCullVertex`/`float4`), and the rewrite is
//! generic over `T`.

use crate::ir::{IrExpr, IrLiteral, IrOp, IrStmt, IrType, ShaderModule, StructDecl};

/// Splits an `Ngg::CullData<T>::Set` / `::GetTriangle` callee into
/// `("Set" | "GetTriangle", T)`. `None` for anything else.
fn culldata_parts(callee: &str) -> Option<(&str, &str)> {
    let start = callee.find("CullData<")? + "CullData<".len();
    let rest = &callee[start..];
    if let Some(t) = rest.strip_suffix(">::Set") {
        Some(("Set", t))
    } else if let Some(t) = rest.strip_suffix(">::GetTriangle") {
        Some(("GetTriangle", t))
    } else {
        None
    }
}

/// Whether `t` is a plausible single-type template arg (no nesting).
fn is_plain_type_name(t: &str) -> bool {
    !t.is_empty() && t.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// Maps a `CullData<T>` element spelling to its IR type: builtins
/// (`float4`, ...) via the parser's own table, anything else by name.
fn element_type(t: &str) -> IrType {
    match crate::parser::types::builtin_type(t) {
        Some(ty) => crate::semantic::types::to_ir_type(&ty),
        None => IrType::Named(t.to_string()),
    }
}

/// Zero value of `ty`: scalars → `0`, vectors/matrices → single-scalar
/// construct (the emitter splats to `float4(0, 0, 0, 0)`). Structs are NOT
/// constructed positionally (`MyCullVertex(...)` is rejected by DXC:
/// "constructors only defined for numeric base types") — they expand to
/// memberwise assignments via [`push_zero_assigns`].
fn zero_value(ty: &IrType) -> Option<IrExpr> {
    let zero = || IrExpr::Literal(IrLiteral::Number("0".to_string()));
    match ty {
        IrType::Bool
        | IrType::Int
        | IrType::UInt
        | IrType::Float
        | IrType::Half
        | IrType::Double => Some(zero()),
        IrType::Vector { .. } | IrType::Matrix { .. } => Some(IrExpr::Construct {
            ty: ty.clone(),
            arguments: vec![zero()],
        }),
        // Structs, void, resources: no single-expression zero (see above).
        _ => None,
    }
}

/// Appends `path = zero` assignments for `ty` (memberwise through structs
/// and arrays), e.g. `tri[0].m_projectedXY = float2(0, 0); ...`.
fn push_zero_assigns(
    out: &mut Vec<IrStmt>,
    path: IrExpr,
    ty: &IrType,
    structs: &[StructDecl],
    array_size: Option<u32>,
    depth: usize,
) {
    if let Some(n) = array_size {
        for i in 0..n {
            push_zero_assigns(
                out,
                IrExpr::Index {
                    object: Box::new(path.clone()),
                    index: Box::new(IrExpr::Literal(IrLiteral::Number(i.to_string()))),
                },
                ty,
                structs,
                None,
                depth,
            );
        }
        return;
    }
    match ty {
        IrType::Named(name) => match structs.iter().find(|s| &s.name == name) {
            Some(decl) if depth < 8 => {
                for field in &decl.fields {
                    push_zero_assigns(
                        out,
                        IrExpr::Member {
                            object: Box::new(path.clone()),
                            member: field.name.clone(),
                        },
                        &field.ty,
                        structs,
                        field.array_size,
                        depth + 1,
                    );
                }
            }
            // Unknown `T`: stay loud at DXC rather than silently
            // miscompiling.
            _ => out.push(IrStmt::Expr(IrExpr::Call {
                function: format!("Ngg::CullData::GetTriangle /* unknown element type {name} */"),
                arguments: Vec::new(),
            })),
        },
        _ => match zero_value(ty) {
            Some(zero) => out.push(IrStmt::Expr(IrExpr::Assign {
                lhs: Box::new(path),
                rhs: Box::new(zero),
            })),
            // Unrepresentable element (void/resource): stay loud at DXC.
            None => out.push(IrStmt::Expr(IrExpr::Call {
                function: "Ngg::CullData::GetTriangle /* unrepresentable element type */"
                    .to_string(),
                arguments: Vec::new(),
            })),
        },
    }
}

/// A cull-data triangle always has three vertices.
const TRIANGLE_SIZE: usize = 3;

/// Lowers `Ngg::CullData<T>` calls in every function body (recursing into
/// nested blocks): `Set` statements drop, `GetTriangle` initializers become
/// zero `T[3]` arrays. Shaders without the builtin are untouched.
pub(crate) fn lower_ngg_cull_data(ir: &mut ShaderModule) {
    for function in &mut ir.functions {
        lower_ngg_in_stmts(&mut function.body, &ir.structs);
    }
}

fn lower_ngg_in_stmts(stmts: &mut Vec<IrStmt>, structs: &[StructDecl]) {
    // `GetTriangle` declarations expand to several statements (bare
    // declaration plus assignments), so they splice into the parent list:
    // wrapping them in a block would scope the variable away from its
    // later uses.
    let mut i = 0;
    while i < stmts.len() {
        match expand_ngg_get_triangle(&stmts[i], structs) {
            Some(expanded) => {
                let n = expanded.len();
                stmts.splice(i..=i, expanded);
                i += n;
            }
            None => {
                lower_ngg_in_stmt(&mut stmts[i], structs);
                i += 1;
            }
        }
    }
}

/// Expands `T name[3] = Ngg::CullData<T>::GetTriangle();` (or the `auto`
/// form) into a bare `T name[3];` plus memberwise zero assignments.
/// `None` for any other statement.
fn expand_ngg_get_triangle(stmt: &IrStmt, structs: &[StructDecl]) -> Option<Vec<IrStmt>> {
    let IrStmt::VarDecl {
        ty,
        name,
        init: Some(IrExpr::Call {
            function,
            arguments,
        }),
        ..
    } = stmt
    else {
        return None;
    };
    let elem_name = match culldata_parts(function) {
        Some(("GetTriangle", t)) if is_plain_type_name(t) && arguments.is_empty() => t.to_string(),
        _ => return None,
    };
    // An explicit `T name[3]` keeps its declared type, `auto` takes the
    // builtin's `T`.
    let elem = if matches!(ty, IrType::Named(n) if n == "auto") {
        element_type(&elem_name)
    } else {
        ty.clone()
    };
    // Bare declaration plus memberwise zero assignments (struct
    // positional construction is not valid HLSL).
    let mut expanded = vec![IrStmt::VarDecl {
        ty: elem.clone(),
        name: name.clone(),
        array_size: Some(TRIANGLE_SIZE as u32),
        init: None,
    }];
    push_zero_assigns(
        &mut expanded,
        IrExpr::Variable(name.clone()),
        &elem,
        structs,
        Some(TRIANGLE_SIZE as u32),
        0,
    );
    Some(expanded)
}

/// Lowers an NGG-output mesh entry to DX mesh output objects.
///
/// A PSSL mesh entry returns an `NGG_OUTPUT`-shaped struct per lane
/// (`{output, nggPrimExport, nggVertCount, nggPrimCount}` with
/// `S_NGG_OUTPUT_*` member semantics); DXC `ms_6_5` instead requires
/// `out vertices` / `out indices` objects plus `SetMeshOutputCounts`.
/// The rewrite appends the two objects (sized by `[num_verts]` /
/// `[num_prims]`, falling back to `numthreads.x`) and replaces every
/// `return e;` with counts, predicated per-lane exports, and a bare
/// return:
///
/// - `verts[tid] = e.output` when `tid < e.nggVertCount`;
/// - `tris[tid] = unpack(e.nggPrimExport)` when `tid < e.nggPrimCount`,
///   where unpack reverses the `makeHwPrimForExport` packing
///   (`v0 | v1<<10 | v2<<20`, 10-bit index fields).
///
/// Soundness gates (anything else keeps today's loud path):
/// - stage is mesh and the entry returns a struct with EXACTLY the four
///   NGG fields (three `S_NGG_OUTPUT_*` uints plus one declared struct
///   member); extra fields would be silently dropped otherwise;
/// - a `S_GROUP_THREAD_ID` lane-index parameter exists (predication and
///   indexing need it);
/// - every `return` yields the NGG struct (bare `return;` or foreign
///   values decline);
/// - sizes resolve to nonzero.
///
/// `SetMeshOutputCounts` uniformity follows from NGG subgroup semantics
/// (counts are meshlet-level, identical across lanes by construction).
pub(crate) fn lower_mesh_outputs(ir: &mut ShaderModule) {
    use crate::ir::{IrOp, IrType as T, Stage};

    if !matches!(ir.stage, Some(Stage::Mesh)) {
        return;
    }
    let Some(entry_name) = ir.entry_point.clone() else {
        return;
    };
    let Some(entry_idx) = ir.functions.iter().position(|f| f.name == entry_name) else {
        return;
    };
    // NGG linkage markers (`S_NGG_OUTPUT_*`) are entry-linkage only: strip
    // them from non-entry params (helpers like mesh_lib's bare `main`
    // carry the same out-params; the markers are not valid HLSL semantics
    // and would fail DXC after the entry itself validates).
    for func in ir.functions.iter_mut() {
        if func.name == entry_name {
            continue;
        }
        for p in func.parameters.iter_mut() {
            if p.semantic
                .as_deref()
                .is_some_and(|s| s.to_ascii_uppercase().starts_with("S_NGG_OUTPUT"))
            {
                p.semantic = None;
            }
        }
    }
    // NGG payload shape: three S_NGG_OUTPUT-semantic members plus the
    // single struct-typed vertex payload.
    struct NggShape {
        struct_name: String,
        output_member: String,
        output_struct: String,
        prim_member: String,
        vert_count_member: String,
        prim_count_member: String,
    }
    fn ngg_shape(sname: &str, structs: &[StructDecl]) -> Option<NggShape> {
        let decl = structs.iter().find(|s| s.name == sname)?;
        if decl.fields.len() != 4 {
            return None;
        }
        let mut prim = None;
        let mut vert_count = None;
        let mut prim_count = None;
        let mut output = None;
        for f in &decl.fields {
            let sem = f.semantic.as_deref().unwrap_or("");
            if sem.eq_ignore_ascii_case("S_NGG_OUTPUT_PRIMITIVE") {
                prim = Some(f.name.clone());
            } else if sem.eq_ignore_ascii_case("S_NGG_OUTPUT_VERTEX_COUNT") {
                vert_count = Some(f.name.clone());
            } else if sem.eq_ignore_ascii_case("S_NGG_OUTPUT_PRIMITIVE_COUNT") {
                prim_count = Some(f.name.clone());
            } else if let T::Named(s) = &f.ty {
                if structs.iter().any(|d| &d.name == s) {
                    if output.is_some() {
                        return None;
                    }
                    output = Some((f.name.clone(), s.clone()));
                } else {
                    return None;
                }
            } else {
                return None;
            }
        }
        Some(NggShape {
            struct_name: sname.to_string(),
            output_member: output.clone()?.0,
            output_struct: output?.1,
            prim_member: prim?,
            vert_count_member: vert_count?,
            prim_count_member: prim_count?,
        })
    }
    // Shape B (out-parameter entries like mesh_lib's `main_common`):
    // void return with `out` NGG-marked params. Shape A (NGG struct
    // return) follows for non-void returns.
    if matches!(ir.functions[entry_idx].return_type, T::Void) {
        lower_mesh_outparams(ir, entry_idx);
        return;
    }
    let ret_struct = match &ir.functions[entry_idx].return_type {
        T::Named(s) => s.clone(),
        _ => return,
    };
    let shape = match ngg_shape(&ret_struct, &ir.structs) {
        Some(s) => s,
        None => return,
    };
    // Lane index: the S_GROUP_THREAD_ID parameter.
    let tid = match ir.functions[entry_idx].parameters.iter().find(|p| {
        p.semantic
            .as_deref()
            .is_some_and(|s| s.eq_ignore_ascii_case("S_GROUP_THREAD_ID") || s.eq_ignore_ascii_case("SV_GroupThreadID"))
    }) {
        Some(p) => p.name.clone(),
        None => return,
    };
    // Export bounds.
    let nv = ir.functions[entry_idx]
        .mesh_vert_count
        .or_else(|| ir.functions[entry_idx].num_threads.map(|(x, _, _)| x));
    let np = ir.functions[entry_idx]
        .mesh_prim_count
        .or_else(|| ir.functions[entry_idx].num_threads.map(|(x, _, _)| x));
    let (nv, np) = match (nv, np) {
        (Some(nv), Some(np)) if nv > 0 && np > 0 => (nv, np),
        _ => return,
    };
    // Every return must yield the NGG struct. Inference over entry
    // params, locals (first declaration wins), and globals; calls
    // resolve through unique same-name function returns.
    let entry = &ir.functions[entry_idx];
    let mut env: std::collections::HashMap<String, T> = std::collections::HashMap::new();
    for p in &entry.parameters {
        env.insert(p.name.clone(), p.ty.clone());
    }
    fn collect_tys(stmts: &[IrStmt], env: &mut std::collections::HashMap<String, T>) {
        for s in stmts {
            match s {
                IrStmt::VarDecl { name, ty, .. } => {
                    env.entry(name.clone()).or_insert(ty.clone());
                }
                IrStmt::Block(ss) => collect_tys(ss, env),
                IrStmt::If { then_branch, else_branch, .. } => {
                    collect_tys(then_branch, env);
                    if let Some(eb) = else_branch {
                        collect_tys(eb, env);
                    }
                }
                IrStmt::For { init, body, .. } => {
                    collect_tys(std::slice::from_ref(init.as_ref()), env);
                    collect_tys(body, env);
                }
                IrStmt::While { body, .. } => collect_tys(body, env),
                IrStmt::Switch { cases, .. } => {
                    for c in cases {
                        collect_tys(&c.body, env);
                    }
                }
                _ => {}
            }
        }
    }
    collect_tys(&entry.body, &mut env);
    for g in &ir.globals {
        env.entry(g.name.clone()).or_insert(g.ty.clone());
    }
    // Function return types by name (None = ambiguous/unknown).
    let mut returns: std::collections::HashMap<String, Option<T>> = std::collections::HashMap::new();
    for f in &ir.functions {
        match returns.get(&f.name) {
            None => {
                returns.insert(f.name.clone(), Some(f.return_type.clone()));
            }
            Some(Some(prev)) if *prev != f.return_type => {
                returns.insert(f.name.clone(), None);
            }
            _ => {}
        }
    }
    fn infer_ngg(
        e: &IrExpr,
        env: &std::collections::HashMap<String, T>,
        structs: &[StructDecl],
        returns: &std::collections::HashMap<String, Option<T>>,
        want: &str,
    ) -> bool {
        let ty: Option<T> = match e {
            IrExpr::Variable(n) => env.get(n).cloned(),
            IrExpr::Member { object, member } => {
                let Some(T::Named(s)) = infer_ty(object, env, structs, returns) else {
                    return false;
                };
                structs
                    .iter()
                    .find(|d| &d.name == &s)
                    .and_then(|d| d.fields.iter().find(|f| &f.name == member))
                    .map(|f| f.ty.clone())
            }
            IrExpr::Construct { ty, .. } => Some(ty.clone()),
            IrExpr::Call { function, arguments } => match returns.get(function) {
                Some(Some(t)) => Some(t.clone()),
                _ => None,
            },
            _ => None,
        };
        matches!(ty, Some(T::Named(s)) if s == want)
    }
    fn infer_ty(
        e: &IrExpr,
        env: &std::collections::HashMap<String, T>,
        structs: &[StructDecl],
        returns: &std::collections::HashMap<String, Option<T>>,
    ) -> Option<T> {
        match e {
            IrExpr::Variable(n) => env.get(n).cloned(),
            IrExpr::Member { object, member } => {
                let T::Named(s) = infer_ty(object, env, structs, returns)? else {
                    return None;
                };
                structs
                    .iter()
                    .find(|d| &d.name == &s)
                    .and_then(|d| d.fields.iter().find(|f| &f.name == member))
                    .map(|f| f.ty.clone())
            }
            IrExpr::Construct { ty, .. } => Some(ty.clone()),
            IrExpr::Call { function, .. } => returns.get(function)?.clone(),
            _ => None,
        }
    }
    fn all_returns_ngg(
        stmts: &[IrStmt],
        env: &std::collections::HashMap<String, T>,
        structs: &[StructDecl],
        returns: &std::collections::HashMap<String, Option<T>>,
        want: &str,
    ) -> bool {
        stmts.iter().all(|s| match s {
            IrStmt::Return(Some(e)) => infer_ngg(e, env, structs, returns, want),
            IrStmt::Return(None) => false,
            IrStmt::Block(ss) => all_returns_ngg(ss, env, structs, returns, want),
            IrStmt::If { then_branch, else_branch, .. } => {
                all_returns_ngg(then_branch, env, structs, returns, want)
                    && else_branch
                        .as_ref()
                        .is_some_and(|eb| all_returns_ngg(eb, env, structs, returns, want))
            }
            IrStmt::For { body, .. } | IrStmt::While { body, .. } => {
                all_returns_ngg(body, env, structs, returns, want)
            }
            IrStmt::Switch { cases, .. } => cases
                .iter()
                .all(|c| all_returns_ngg(&c.body, env, structs, returns, want)),
            _ => true,
        })
    }
    if !all_returns_ngg(
        &ir.functions[entry_idx].body,
        &env,
        &ir.structs,
        &returns,
        &shape.struct_name,
    ) {
        return;
    }
    // Rewrite: append output objects, void return, per-return expansion.
    let entry = &mut ir.functions[entry_idx];
    // Collision-free object/temp names.
    let mut taken: std::collections::HashSet<String> = std::collections::HashSet::new();
    for p in &entry.parameters {
        taken.insert(p.name.clone());
    }
    fn collect_names(stmts: &[IrStmt], taken: &mut std::collections::HashSet<String>) {
        for s in stmts {
            match s {
                IrStmt::VarDecl { name, .. } => {
                    taken.insert(name.clone());
                }
                IrStmt::Block(ss) => collect_names(ss, taken),
                IrStmt::If { then_branch, else_branch, .. } => {
                    collect_names(then_branch, taken);
                    if let Some(eb) = else_branch {
                        collect_names(eb, taken);
                    }
                }
                IrStmt::For { init, body, .. } => {
                    collect_names(std::slice::from_ref(init.as_ref()), taken);
                    collect_names(body, taken);
                }
                IrStmt::While { body, .. } => collect_names(body, taken),
                IrStmt::Switch { cases, .. } => {
                    for c in cases {
                        collect_names(&c.body, taken);
                    }
                }
                _ => {}
            }
        }
    }
    collect_names(&entry.body, &mut taken);
    let mut fresh = |base: &str, taken: &mut std::collections::HashSet<String>| {
        let mut name = base.to_string();
        let mut n = 0u32;
        while taken.contains(&name) {
            n += 1;
            name = format!("{base}{n}");
        }
        taken.insert(name.clone());
        name
    };
    let verts_name = fresh("__meshVerts", &mut taken);
    let tris_name = fresh("__meshTris", &mut taken);
    let ret_name = fresh("__ngg_ret", &mut taken);
    entry.parameters.push(crate::ir::function::Parameter {
        ty: T::Named(shape.output_struct.clone()),
        name: verts_name.clone(),
        semantic: None,
        array_size: Some(nv),
        direction: Some("out".to_string()),
        gs_primitive: None,
        default: None,
        is_uniform: false,
        mesh_object: Some("vertices".to_string()),
    });
    entry.parameters.push(crate::ir::function::Parameter {
        ty: T::Vector { base: crate::ir::IrScalar::UInt, width: 3 },
        name: tris_name.clone(),
        semantic: None,
        array_size: Some(np),
        direction: Some("out".to_string()),
        gs_primitive: None,
        default: None,
        is_uniform: false,
        mesh_object: Some("indices".to_string()),
    });
    entry.return_type = T::Void;
    // Expand each `return e;` (e already proven NGG-typed).
    fn member(base: IrExpr, field: &str) -> IrExpr {
        IrExpr::Member { object: Box::new(base), member: field.to_string() }
    }
    fn expand_returns(
        stmts: &mut Vec<IrStmt>,
        shape: &NggShape,
        tid: &str,
        verts: &str,
        tris: &str,
        ret: &str,
    ) {
        let mut i = 0;
        while i < stmts.len() {
            let replacement: Option<Vec<IrStmt>> = match &stmts[i] {
                IrStmt::Return(Some(e)) => {
                    let base = IrExpr::Variable(ret.to_string());
                    let vcount = member(base.clone(), &shape.vert_count_member);
                    let pcount = member(base.clone(), &shape.prim_count_member);
                    let pexport = member(base.clone(), &shape.prim_member);
                    let output = member(base, &shape.output_member);
                    let tid_e = IrExpr::Variable(tid.to_string());
                    let unpack = IrExpr::Construct {
                        ty: T::Vector { base: crate::ir::IrScalar::UInt, width: 3 },
                        arguments: vec![
                            IrExpr::Binary {
                                left: Box::new(pexport.clone()),
                                op: IrOp::BitAnd,
                                right: Box::new(IrExpr::Literal(IrLiteral::Number(
                                    "1023u".to_string(),
                                ))),
                            },
                            IrExpr::Binary {
                                left: Box::new(IrExpr::Binary {
                                    left: Box::new(pexport.clone()),
                                    op: IrOp::ShiftRight,
                                    right: Box::new(IrExpr::Literal(IrLiteral::Number(
                                        "10".to_string(),
                                    ))),
                                }),
                                op: IrOp::BitAnd,
                                right: Box::new(IrExpr::Literal(IrLiteral::Number(
                                    "1023u".to_string(),
                                ))),
                            },
                            IrExpr::Binary {
                                left: Box::new(IrExpr::Binary {
                                    left: Box::new(pexport),
                                    op: IrOp::ShiftRight,
                                    right: Box::new(IrExpr::Literal(IrLiteral::Number(
                                        "20".to_string(),
                                    ))),
                                }),
                                op: IrOp::BitAnd,
                                right: Box::new(IrExpr::Literal(IrLiteral::Number(
                                    "1023u".to_string(),
                                ))),
                            },
                        ],
                    };
                    Some(vec![
                        IrStmt::VarDecl {
                            ty: T::Named(shape.struct_name.clone()),
                            name: ret.to_string(),
                            array_size: None,
                            init: Some(e.clone()),
                        },
                        IrStmt::Expr(IrExpr::Call {
                            function: "SetMeshOutputCounts".to_string(),
                            arguments: vec![vcount.clone(), pcount.clone()],
                        }),
                        IrStmt::If {
                            condition: IrExpr::Binary {
                                left: Box::new(tid_e.clone()),
                                op: IrOp::Less,
                                right: Box::new(vcount),
                            },
                            then_branch: vec![IrStmt::Expr(IrExpr::Assign {
                                lhs: Box::new(IrExpr::Index {
                                    object: Box::new(IrExpr::Variable(verts.to_string())),
                                    index: Box::new(tid_e.clone()),
                                }),
                                rhs: Box::new(output),
                            })],
                            else_branch: None,
                        },
                        IrStmt::If {
                            condition: IrExpr::Binary {
                                left: Box::new(tid_e.clone()),
                                op: IrOp::Less,
                                right: Box::new(pcount),
                            },
                            then_branch: vec![IrStmt::Expr(IrExpr::Assign {
                                lhs: Box::new(IrExpr::Index {
                                    object: Box::new(IrExpr::Variable(tris.to_string())),
                                    index: Box::new(tid_e),
                                }),
                                rhs: Box::new(unpack),
                            })],
                            else_branch: None,
                        },
                        IrStmt::Return(None),
                    ])
                }
                _ => None,
            };
            if let Some(seq) = replacement {
                let n = seq.len();
                stmts.splice(i..=i, seq);
                // Recurse into the spliced block bodies (nested returns
                // inside the new ifs are bare and stay).
                i += n;
            } else {
                match &mut stmts[i] {
                    IrStmt::Block(ss) => expand_returns(ss, shape, tid, verts, tris, ret),
                    IrStmt::If { then_branch, else_branch, .. } => {
                        expand_returns(then_branch, shape, tid, verts, tris, ret);
                        if let Some(eb) = else_branch {
                            expand_returns(eb, shape, tid, verts, tris, ret);
                        }
                    }
                    IrStmt::For { body, .. } | IrStmt::While { body, .. } => {
                        expand_returns(body, shape, tid, verts, tris, ret);
                    }
                    IrStmt::Switch { cases, .. } => {
                        for c in cases.iter_mut() {
                            expand_returns(&mut c.body, shape, tid, verts, tris, ret);
                        }
                    }
                    _ => {}
                }
                i += 1;
            }
        }
    }
    expand_returns(
        &mut entry.body,
        &shape,
        &tid,
        &verts_name,
        &tris_name,
        &ret_name,
    );
}

/// Shape-B mesh entries: void return with `out` NGG-marked params
/// (mesh_lib's `main_common` forwarding to a worker). Appends
/// `out vertices` / `out indices` objects plus a closing epilogue that
/// exports the per-lane out-params through them:
///
/// - `SetMeshOutputCounts(vc, pc)`;
/// - `verts[tid] = vertex` when `tid < vc`;
/// - `tris[tid] = unpack(prim)` when `tid < pc`.
///
/// Gates: exactly one `out` struct param (the vertex payload), `out`
/// uint params carrying the three `S_NGG_OUTPUT_*` semantics, a lane
/// index, nonzero sizes, and no `return <value>` anywhere (bare returns
/// expand in place; valued returns decline the whole entry).
fn lower_mesh_outparams(ir: &mut ShaderModule, entry_idx: usize) {
    use crate::ir::{IrOp, IrType as T};

    let entry = &ir.functions[entry_idx];
    let mut prim = None;
    let mut vert_count = None;
    let mut prim_count = None;
    let mut payload = None;
    for p in &entry.parameters {
        let is_out = p.direction.as_deref().is_some_and(|d| d.eq_ignore_ascii_case("out"));
        match &p.ty {
            T::UInt if is_out => {
                let sem = p.semantic.as_deref().unwrap_or("");
                if sem.eq_ignore_ascii_case("S_NGG_OUTPUT_PRIMITIVE") {
                    if prim.is_some() {
                        return;
                    }
                    prim = Some(p.name.clone());
                } else if sem.eq_ignore_ascii_case("S_NGG_OUTPUT_VERTEX_COUNT") {
                    if vert_count.is_some() {
                        return;
                    }
                    vert_count = Some(p.name.clone());
                } else if sem.eq_ignore_ascii_case("S_NGG_OUTPUT_PRIMITIVE_COUNT") {
                    if prim_count.is_some() {
                        return;
                    }
                    prim_count = Some(p.name.clone());
                }
            }
            T::Named(s) if is_out => {
                if ir.structs.iter().any(|d| &d.name == s) {
                    if payload.is_some() {
                        return;
                    }
                    payload = Some((p.name.clone(), s.clone()));
                }
            }
            _ => {}
        }
    }
    let (prim, vert_count, prim_count) = match (prim, vert_count, prim_count) {
        (Some(a), Some(b), Some(c)) => (a, b, c),
        _ => return,
    };
    let (payload_name, payload_ty) = match payload {
        Some(p) => p,
        None => return,
    };
    let tid = match ir.functions[entry_idx].parameters.iter().find(|p| {
        p.semantic
            .as_deref()
            .is_some_and(|s| s.eq_ignore_ascii_case("S_GROUP_THREAD_ID") || s.eq_ignore_ascii_case("SV_GroupThreadID"))
    }) {
        Some(p) => p.name.clone(),
        None => return,
    };
    let nv = ir.functions[entry_idx]
        .mesh_vert_count
        .or_else(|| ir.functions[entry_idx].num_threads.map(|(x, _, _)| x));
    let np = ir.functions[entry_idx]
        .mesh_prim_count
        .or_else(|| ir.functions[entry_idx].num_threads.map(|(x, _, _)| x));
    let (nv, np) = match (nv, np) {
        (Some(nv), Some(np)) if nv > 0 && np > 0 => (nv, np),
        _ => return,
    };
    // Bare returns expand in place; valued returns decline everything.
    fn has_valued_return(stmts: &[IrStmt]) -> bool {
        stmts.iter().any(|s| match s {
            IrStmt::Return(Some(_)) => true,
            IrStmt::Block(ss) => has_valued_return(ss),
            IrStmt::If { then_branch, else_branch, .. } => {
                has_valued_return(then_branch)
                    || else_branch.as_ref().is_some_and(|eb| has_valued_return(eb))
            }
            IrStmt::For { body, .. } | IrStmt::While { body, .. } => has_valued_return(body),
            IrStmt::Switch { cases, .. } => {
                cases.iter().any(|c| has_valued_return(&c.body))
            }
            _ => false,
        })
    }
    if has_valued_return(&ir.functions[entry_idx].body) {
        return;
    }
    let entry = &mut ir.functions[entry_idx];
    let mut taken: std::collections::HashSet<String> = std::collections::HashSet::new();
    for p in &entry.parameters {
        taken.insert(p.name.clone());
    }
    let mut fresh = |base: &str, taken: &mut std::collections::HashSet<String>| {
        let mut name = base.to_string();
        let mut n = 0u32;
        while taken.contains(&name) {
            n += 1;
            name = format!("{base}{n}");
        }
        taken.insert(name.clone());
        name
    };
    let verts_name = fresh("__meshVerts", &mut taken);
    let tris_name = fresh("__meshTris", &mut taken);
    // The NGG out-params are internal conduits (worker call arguments),
    // not DX linkage: remove them from the entry signature and redeclare
    // as function locals of the same types, so the existing body (which
    // forwards them to the worker) keeps working verbatim.
    let mut local_decls: Vec<IrStmt> = Vec::new();
    for (name, ty) in [
        (prim.clone(), T::UInt),
        (vert_count.clone(), T::UInt),
        (prim_count.clone(), T::UInt),
        (payload_name.clone(), T::Named(payload_ty.clone())),
    ] {
        local_decls.push(IrStmt::VarDecl {
            ty,
            name: name.clone(),
            array_size: None,
            init: None,
        });
    }
    entry
        .parameters
        .retain(|p| p.name != prim && p.name != vert_count && p.name != prim_count && p.name != payload_name);
    entry.parameters.push(crate::ir::function::Parameter {
        ty: T::Named(payload_ty),
        name: verts_name.clone(),
        semantic: None,
        array_size: Some(nv),
        direction: Some("out".to_string()),
        gs_primitive: None,
        default: None,
        is_uniform: false,
        mesh_object: Some("vertices".to_string()),
    });
    entry.parameters.push(crate::ir::function::Parameter {
        ty: T::Vector { base: crate::ir::IrScalar::UInt, width: 3 },
        name: tris_name.clone(),
        semantic: None,
        array_size: Some(np),
        direction: Some("out".to_string()),
        gs_primitive: None,
        default: None,
        is_uniform: false,
        mesh_object: Some("indices".to_string()),
    });
    fn epilogue(
        body: &mut Vec<IrStmt>,
        tid: &str,
        verts: &str,
        tris: &str,
        vc: &str,
        pc: &str,
        prim: &str,
        payload: &str,
    ) {
        let tid_e = || IrExpr::Variable(tid.to_string());
        let unpack = |pexport: IrExpr| IrExpr::Construct {
            ty: T::Vector { base: crate::ir::IrScalar::UInt, width: 3 },
            arguments: vec![
                IrExpr::Binary {
                    left: Box::new(pexport.clone()),
                    op: IrOp::BitAnd,
                    right: Box::new(IrExpr::Literal(IrLiteral::Number("1023u".to_string()))),
                },
                IrExpr::Binary {
                    left: Box::new(IrExpr::Binary {
                        left: Box::new(pexport.clone()),
                        op: IrOp::ShiftRight,
                        right: Box::new(IrExpr::Literal(IrLiteral::Number("10".to_string()))),
                    }),
                    op: IrOp::BitAnd,
                    right: Box::new(IrExpr::Literal(IrLiteral::Number("1023u".to_string()))),
                },
                IrExpr::Binary {
                    left: Box::new(IrExpr::Binary {
                        left: Box::new(pexport),
                        op: IrOp::ShiftRight,
                        right: Box::new(IrExpr::Literal(IrLiteral::Number("20".to_string()))),
                    }),
                    op: IrOp::BitAnd,
                    right: Box::new(IrExpr::Literal(IrLiteral::Number("1023u".to_string()))),
                },
            ],
        };
        let export_one = |arr: &str, count: &str, value: IrExpr| IrStmt::If {
            condition: IrExpr::Binary {
                left: Box::new(tid_e()),
                op: IrOp::Less,
                right: Box::new(IrExpr::Variable(count.to_string())),
            },
            then_branch: vec![IrStmt::Expr(IrExpr::Assign {
                lhs: Box::new(IrExpr::Index {
                    object: Box::new(IrExpr::Variable(arr.to_string())),
                    index: Box::new(tid_e()),
                }),
                rhs: Box::new(value),
            })],
            else_branch: None,
        };
        // Expand bare returns in place, then append the fall-off-end
        // epilogue (unreachable when every path returns; harmless then).
        fn expand_bare(
            stmts: &mut Vec<IrStmt>,
            tid: &str,
            verts: &str,
            tris: &str,
            vc: &str,
            pc: &str,
            prim: &str,
            payload: &str,
            epilogue: &dyn Fn(
                &mut Vec<IrStmt>,
                &str,
                &str,
                &str,
                &str,
                &str,
                &str,
                &str,
            ),
        ) {
            let mut i = 0;
            while i < stmts.len() {
                if matches!(stmts[i], IrStmt::Return(None)) {
                    let mut seq = Vec::new();
                    epilogue(&mut seq, tid, verts, tris, vc, pc, prim, payload);
                    seq.push(IrStmt::Return(None));
                    // Skip past the spliced sequence: its trailing
                    // `Return(None)` must not expand again (that looped
                    // forever, growing memory without bound).
                    let n = seq.len();
                    stmts.splice(i..=i, seq);
                    i += n;
                    continue;
                }
                match &mut stmts[i] {
                    IrStmt::Block(ss) => expand_bare(ss, tid, verts, tris, vc, pc, prim, payload, epilogue),
                    IrStmt::If { then_branch, else_branch, .. } => {
                        expand_bare(then_branch, tid, verts, tris, vc, pc, prim, payload, epilogue);
                        if let Some(eb) = else_branch {
                            expand_bare(eb, tid, verts, tris, vc, pc, prim, payload, epilogue);
                        }
                    }
                    IrStmt::For { body, .. } | IrStmt::While { body, .. } => {
                        expand_bare(body, tid, verts, tris, vc, pc, prim, payload, epilogue);
                    }
                    IrStmt::Switch { cases, .. } => {
                        for c in cases.iter_mut() {
                            expand_bare(&mut c.body, tid, verts, tris, vc, pc, prim, payload, epilogue);
                        }
                    }
                    _ => {}
                }
                i += 1;
            }
        }
        let emit_epilogue = |out: &mut Vec<IrStmt>,
                               tid: &str,
                               verts: &str,
                               tris: &str,
                               vc: &str,
                               pc: &str,
                               prim: &str,
                               payload: &str| {
            out.push(IrStmt::Expr(IrExpr::Call {
                function: "SetMeshOutputCounts".to_string(),
                arguments: vec![
                    IrExpr::Variable(vc.to_string()),
                    IrExpr::Variable(pc.to_string()),
                ],
            }));
            out.push(export_one(
                verts,
                vc,
                IrExpr::Variable(payload.to_string()),
            ));
            out.push(export_one(
                tris,
                pc,
                unpack(IrExpr::Variable(prim.to_string())),
            ));
        };
        expand_bare(
            body, tid, verts, tris, vc, pc, prim, payload, &emit_epilogue,
        );
        emit_epilogue(body, tid, verts, tris, vc, pc, prim, payload);
    }
    epilogue(
        &mut entry.body,
        &tid,
        &verts_name,
        &tris_name,
        &vert_count,
        &prim_count,
        &prim,
        &payload_name,
    );
    // Locals first: the worker call below reads them.
    local_decls.append(&mut entry.body);
    entry.body = local_decls;
}

fn lower_ngg_in_stmt(stmt: &mut IrStmt, structs: &[StructDecl]) {
    match stmt {
        IrStmt::Expr(IrExpr::Call { function, .. }) => {
            if matches!(culldata_parts(function), Some(("Set", t)) if is_plain_type_name(t)) {
                // Void builtin with no Vulkan effect: elide (same empty-
                // block shape as the other statement-level elisions).
                *stmt = IrStmt::Block(Vec::new());
            }
        }
        IrStmt::VarDecl { .. } => {
            // `GetTriangle` declarations expand at the parent statement
            // list (see above); a lone one here (e.g. a `for` init) stays
            // loud at DXC.
        }
        IrStmt::Block(inner) => lower_ngg_in_stmts(inner, structs),
        IrStmt::If {
            then_branch,
            else_branch,
            ..
        } => {
            lower_ngg_in_stmts(then_branch, structs);
            if let Some(else_branch) = else_branch {
                lower_ngg_in_stmts(else_branch, structs);
            }
        }
        IrStmt::For { init, body, .. } => {
            // Condition/update are value-position expressions: a cull
            // builtin there has no declaration context to rewrite into
            // (absent corpus-wide), so only the init statement is visited.
            lower_ngg_in_stmt(init, structs);
            lower_ngg_in_stmts(body, structs);
        }
        IrStmt::While { body, .. } => lower_ngg_in_stmts(body, structs),
        IrStmt::Switch { cases, .. } => {
            for case in cases {
                lower_ngg_in_stmts(&mut case.body, structs);
            }
        }
        _ => {}
    }
}
