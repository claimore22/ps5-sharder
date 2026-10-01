//! Expression lowering: AST expr -> IR expr.

use std::collections::HashMap;

use crate::ast::Type;
use crate::ir::{IrExpr, IrLiteral, IrOp, IrScalar, IrType};
use crate::semantic::to_ir_type;

pub(crate) fn lower_expr(
    expr: &crate::ast::Expr,
    env: &HashMap<String, crate::ast::Type>,
    resource_fields: &HashMap<(String, String), Type>,
    var_struct: &HashMap<String, String>,
) -> IrExpr {
    match expr {
        crate::ast::Expr::Number(value) => IrExpr::Literal(IrLiteral::Number(value.clone())),

        crate::ast::Expr::String(value) => IrExpr::Literal(IrLiteral::String(value.clone())),

        crate::ast::Expr::Bool(value) => IrExpr::Literal(IrLiteral::Bool(*value)),

        crate::ast::Expr::Variable(name) => {
            // Framework enumerators (`sce::Agc::VrsRate::k1x1`, …) resolve
            // to integer literals here so every downstream walker sees
            // plain values; unknown `sce::` names pass through and keep
            // failing loudly at DXC.
            if let Some(value) = crate::framework::framework_const(name) {
                IrExpr::Literal(IrLiteral::Number(value.to_string()))
            } else if env.get(name).is_none() {
                // Bare PSSL builtin constants (`PI`) resolve only when no
                // variable of that name is in scope (params, locals, and
                // globals all shadow the builtin); anything else passes
                // through and keeps failing loudly at DXC.
                match crate::framework::framework_float_const(name) {
                    Some(value) => IrExpr::Literal(IrLiteral::Number(value.to_string())),
                    None => IrExpr::Variable(name.clone()),
                }
            } else {
                IrExpr::Variable(name.clone())
            }
        }

        crate::ast::Expr::Member { object, member } => {
            // PSSL compatibility:
            //
            // A scalar cast of a vector can be immediately followed by a
            // component access:
            //
            //     (float)vec.x
            //
            // The parser represents this approximately as:
            //
            // Member {
            //     object: Constructor {
            //         ty: Float,
            //         arguments: [vec]
            //     },
            //     member: "x"
            // }
            //
            // PSSL semantics treat this as:
            //
            //     float(vec.x)
            //
            if let crate::ast::Expr::Constructor { ty, arguments } = object.as_ref() {
                let is_scalar_target = matches!(
                    ty,
                    crate::ast::Type::Float
                        | crate::ast::Type::Half
                        | crate::ast::Type::Int
                        | crate::ast::Type::UInt
                        | crate::ast::Type::Bool
                        | crate::ast::Type::Double
                );

                if is_scalar_target && arguments.len() == 1 && member.len() == 1 {
                    let inner = &arguments[0];
                    let lowered_inner =
                        lower_expr(inner, env, resource_fields, var_struct);

                    let member_expr = IrExpr::Member {
                        object: Box::new(lowered_inner),
                        member: member.clone(),
                    };

                    return IrExpr::Construct {
                        ty: to_ir_type(ty),
                        arguments: vec![member_expr],
                    };
                }
            }

            IrExpr::Member {
                object: Box::new(lower_expr(object, env, resource_fields, var_struct)),
                member: member.clone(),
            }
        }

        crate::ast::Expr::Index { object, index } => {
            // `resource[coord]` where `resource` is a resource-typed struct
            // member (e.g. `srt_data.dest[coord]`) — represent explicitly so
            // the backend knows this is a resource index, not array indexing.
            if is_resource_member_expr(env, resource_fields, var_struct, object) {
                IrExpr::ResourceIndex {
                    resource: Box::new(lower_expr(object, env, resource_fields, var_struct)),
                    index: Box::new(lower_expr(index, env, resource_fields, var_struct)),
                }
            } else {
                IrExpr::Index {
                    object: Box::new(lower_expr(object, env, resource_fields, var_struct)),
                    index: Box::new(lower_expr(index, env, resource_fields, var_struct)),
                }
            }
        }

        crate::ast::Expr::Call {
            function,
            arguments,
        } => {
            // `resource.Load(coords[, sample])` where the callee object is a
            // resource-typed struct member (e.g. `srt_data.source.Load(...)`).
            if let crate::ast::Expr::Member { object, member } = function.as_ref()
                && member.eq_ignore_ascii_case("Load")
                && is_resource_member_expr(env, resource_fields, var_struct, object)
            {
                let mut coords = arguments
                    .first()
                    .map(|a| lower_expr(a, env, resource_fields, var_struct))
                    .unwrap_or(IrExpr::Literal(IrLiteral::Number("0".to_string())));
                let mut sample = arguments
                    .get(1)
                    .map(|a| lower_expr(a, env, resource_fields, var_struct));
                // PSSL `Load(coords, offset)` with a VECTOR second argument is a
                // texel OFFSET added to the coordinate, not HLSL's sample index
                // (always a scalar). HLSL requires offset immediates, so a
                // variable vector offset folds into the coordinate
                // (`Load(c, o)` -> `Load(c + pad(o))`); the spatial offset
                // occupies the leading components and the trailing mip/sample is
                // untouched. A scalar second argument stays the sample index.
                let folded = match (arguments.first(), arguments.get(1), &sample) {
                    (Some(coord_ast), Some(off_ast), Some(off_ir)) => {
                        match (vector_info(coord_ast, env), vector_info(off_ast, env)) {
                            (Some((base, cw)), Some((_, ow))) if ow >= 2 && cw >= ow => {
                                Some(load_offset_fold(
                                    coords.clone(),
                                    off_ir.clone(),
                                    base,
                                    cw,
                                    ow,
                                ))
                            }
                            _ => None,
                        }
                    }
                    _ => None,
                };
                if let Some(new_coords) = folded {
                    sample = None;
                    coords = new_coords;
                }
                return IrExpr::ResourceLoad {
                    resource: Box::new(lower_expr(object, env, resource_fields, var_struct)),
                    coordinates: Box::new(coords),
                    sample: sample.map(Box::new),
                };
            }

            // PSSL `GatherLOD*` (explicit-LOD texture gathers) have no HLSL/
            // SPIR-V equivalent: Vulkan samplers derive the LOD themselves.
            // Lower to a plain `Gather*` and drop the trailing explicit-LOD
            // argument (`GatherLODRed(s, uv, lod)` -> `GatherRed(s, uv)`).
            if let crate::ast::Expr::Member { object, member } = function.as_ref() {
                let l = member.to_ascii_lowercase().replace('_', "");
                if let Some(hlsl_method) = gather_lod_hlsl_name(&l) {
                    if let Some(obj_path) = flatten_member_path(object) {
                        let mut lowered: Vec<IrExpr> = arguments
                            .iter()
                            .map(|a| lower_expr(a, env, resource_fields, var_struct))
                            .collect();
                        lowered.pop();
                        return IrExpr::Call {
                            function: format!(
                                "{}.{hlsl_method}",
                                super::rename::rename_member_root(&obj_path)
                            ),
                            arguments: lowered,
                        };
                    }
                }
            }

            let name = match function.as_ref() {
                crate::ast::Expr::Variable(name)
                    if name.eq_ignore_ascii_case("__read_user_sgpr")
                        && arguments.len() == 1 =>
                {
                    // User SGPR read (`uint frame = __read_user_sgpr(0)`):
                    // the index is a host-provided root constant; lower to
                    // the backing global (`__user_sgpr0`), declared by
                    // `lower_module`. Non-literal indices stay a call and
                    // keep failing loudly at DXC.
                    if let crate::ast::Expr::Number(value) = &arguments[0] {
                        if let Some(index) = parse_u32_literal(value) {
                            return IrExpr::Variable(
                                crate::framework::user_sgpr_name(index),
                            );
                        }
                    }
                    name.clone()
                }
                crate::ast::Expr::Variable(name)
                    if name.eq_ignore_ascii_case("maskbitcnt")
                        && arguments.len() == 1 =>
                {
                    // Wave prefix-bit-count (`MaskBitCnt(m)`): popcount of
                    // the mask bits below the current lane (AMD mbcnt; PSSL
                    // `ulong` is 32-bit so one dword suffices). Lowers to
                    // the `__maskbitcnt_u32` shim (emitted gated-on-use).
                    // The all-ones literal (`~0ul`) is the lane index
                    // itself: `WaveGetLaneIndex()`.
                    if let crate::ast::Expr::Unary { op, expr } = &arguments[0] {
                        if matches!(op, crate::ast::UnaryOp::BitNot) {
                            if let crate::ast::Expr::Number(value) = expr.as_ref() {
                                if parse_u32_literal(value) == Some(0) {
                                    return IrExpr::Call {
                                        function: "WaveGetLaneIndex".to_string(),
                                        arguments: Vec::new(),
                                    };
                                }
                            }
                        }
                    }
                    // Canonicalize numeric literals (`0ul` is invalid HLSL);
                    // anything else lowers normally into the shim.
                    let lowered = match &arguments[0] {
                        crate::ast::Expr::Number(value) => match parse_u32_literal(value) {
                            Some(n) => IrExpr::Literal(IrLiteral::Number(n.to_string())),
                            None => lower_expr(&arguments[0], env, resource_fields, var_struct),
                        },
                        _ => lower_expr(&arguments[0], env, resource_fields, var_struct),
                    };
                    return IrExpr::Call {
                        function: "__maskbitcnt_u32".to_string(),
                        arguments: vec![lowered],
                    };
                }
                crate::ast::Expr::Variable(name)
                    if (name.eq_ignore_ascii_case("__v_mbcnt_lo_u32_b32")
                        || name.eq_ignore_ascii_case("__v_mbcnt_hi_u32_b32"))
                        && arguments.len() == 2 =>
                {
                    // Raw GCN prefix-bit-count (`V_MBCNT_LO/HI_U32_B32(mask,
                    // prev)`): set bits below the current lane within the
                    // low/high 32-lane half, plus the carry-in. The `_lo`
                    // half reuses the `__maskbitcnt_u32` shim; the `_hi`
                    // half gets its own (`__maskbitcnt_hi_u32`, relative
                    // lane). Together over all-ones masks they yield the
                    // exact wave64 lane index (the observed `tid` idiom).
                    let shim = if name.eq_ignore_ascii_case("__v_mbcnt_lo_u32_b32") {
                        "__maskbitcnt_u32"
                    } else {
                        "__maskbitcnt_hi_u32"
                    };
                    let mut lowered: Vec<IrExpr> = arguments
                        .iter()
                        .map(|a| lower_expr(a, env, resource_fields, var_struct))
                        .collect();
                    let mask = lowered.remove(0);
                    let prev = lowered.remove(0);
                    return IrExpr::Binary {
                        left: Box::new(prev),
                        op: IrOp::Add,
                        right: Box::new(IrExpr::Call {
                            function: shim.to_string(),
                            arguments: vec![mask],
                        }),
                    };
                }
                crate::ast::Expr::Variable(name)
                    if (name.eq_ignore_ascii_case("wave64::__v_cndmask_b32")
                        || name.eq_ignore_ascii_case("wave32::__v_cndmask_b32"))
                        && arguments.len() == 3 =>
                {
                    // GCN `V_CNDMASK_B32(S0, S1, VCC)`: per-lane
                    // `VCC ? S1 : S0`. Observed as `(0, 1, p)` (predicate
                    // idiom in culling headers): plain ternary, exact
                    // under either wave-size prefix (per-lane select).
                    let mut lowered = arguments
                        .iter()
                        .map(|a| lower_expr(a, env, resource_fields, var_struct));
                    let (Some(a), Some(b), Some(c)) =
                        (lowered.next(), lowered.next(), lowered.next())
                    else {
                        unreachable!("guarded by arguments.len() == 3")
                    };
                    return IrExpr::Conditional {
                        condition: Box::new(c),
                        then_branch: Box::new(b),
                        else_branch: Box::new(a),
                    };
                }
                crate::ast::Expr::Variable(name)
                    if name.eq_ignore_ascii_case("wave64::__v_cmp_gt_u32")
                        && arguments.len() == 2 =>
                {
                    // GCN `V_CMP_GT_U32(S0, S1)`: per-lane `S0 > S1`.
                    let mut lowered = arguments
                        .iter()
                        .map(|a| lower_expr(a, env, resource_fields, var_struct));
                    let (Some(a), Some(b)) = (lowered.next(), lowered.next()) else {
                        unreachable!("guarded by arguments.len() == 2")
                    };
                    return IrExpr::Binary {
                        left: Box::new(a),
                        op: IrOp::Greater,
                        right: Box::new(b),
                    };
                }
                crate::ast::Expr::Variable(name)
                    if name.eq_ignore_ascii_case("popcnt")
                        && arguments.len() == 1
                        && matches!(&arguments[0], crate::ast::Expr::Call { function, arguments: inner } if matches!(function.as_ref(), crate::ast::Expr::Variable(b) if b.eq_ignore_ascii_case("ballot")) && inner.len() == 1) =>
                {
                    // `popcnt(ballot(v))` counts lanes where `v` holds:
                    // HLSL `WaveActiveCountBits(v)` exactly (scalar uint).
                    // (Bare `popcnt(x)` maps to `countbits(x)` in the arm
                    // below; a ballot mask is uint4, which `countbits`
                    // cannot reduce.)
                    let crate::ast::Expr::Call { arguments: inner, .. } = &arguments[0] else {
                        unreachable!("guarded by matches! above")
                    };
                    return IrExpr::Call {
                        function: "WaveActiveCountBits".to_string(),
                        arguments: vec![lower_expr(
                            &inner[0],
                            env,
                            resource_fields,
                            var_struct,
                        )],
                    };
                }
                crate::ast::Expr::Variable(name)
                    if (name.eq_ignore_ascii_case("popcnt")
                        || name.eq_ignore_ascii_case("countsetbits"))
                        && arguments.len() == 1 =>
                {
                    // `popcnt(x)` / `CountSetBits(x)` are both HLSL
                    // `countbits(x)` exactly (popcount of the operand).
                    return IrExpr::Call {
                        function: "countbits".to_string(),
                        arguments: arguments
                            .iter()
                            .map(|a| lower_expr(a, env, resource_fields, var_struct))
                            .collect(),
                    };
                }
                crate::ast::Expr::Variable(name)
                    if (name.eq_ignore_ascii_case("__s_bfm_b64")
                        || name.eq_ignore_ascii_case("wave64::__s_bfm_b64")
                        || name.eq_ignore_ascii_case("wave32::__s_bfm_b64"))
                        && arguments.len() == 2 =>
                {
                    // GCN `S_BFM_B64(width, offset)` (contiguous bitfield
                    // mask): lowers to the `__s_bfm_b64` shim (emitted
                    // gated-on-use). 32-bit saturating, matching the
                    // toolchain's 32-bit `ulong` model (see the mbcnt
                    // shims); exact for widths < 32 and for zero/nonzero
                    // tests at any width.
                    return IrExpr::Call {
                        function: "__s_bfm_b64".to_string(),
                        arguments: arguments
                            .iter()
                            .map(|a| lower_expr(a, env, resource_fields, var_struct))
                            .collect(),
                    };
                }
                crate::ast::Expr::Variable(name)
                    if name.eq_ignore_ascii_case("__invariant") && arguments.len() == 1 =>
                {
                    // `__invariant(x)` (invariance annotation on the value:
                    // the compiler must not reassociate it). Value-identity
                    // by definition, so desugar to the operand; HLSL has
                    // no such annotation. Other arities keep today's loud
                    // path.
                    return lower_expr(&arguments[0], env, resource_fields, var_struct);
                }
                crate::ast::Expr::Variable(name)
                    if name.eq_ignore_ascii_case("wave::inverse") && arguments.len() == 1 =>
                {
                    // Framework `wave::inverse(m)` (no in-tree definition; system
                    // namespace like `wave64::`): true matrix inverse.
                    // HLSL has no inverse intrinsic, so this lowers to the
                    // `__wave_inverse` adjugate shim (3x3/4x4 overloads,
                    // emitted gated-on-use); HLSL resolves the overload
                    // natively, so no type info is needed here.
                    return IrExpr::Call {
                        function: "__wave_inverse".to_string(),
                        arguments: arguments
                            .iter()
                            .map(|a| lower_expr(a, env, resource_fields, var_struct))
                            .collect(),
                    };
                }
                crate::ast::Expr::Variable(name)
                    if name.eq_ignore_ascii_case("wave::isfirstactivelane")
                        && arguments.is_empty() =>
                {
                    // Framework `wave::IsFirstActiveLane()` (first active lane in
                    // the wave) is HLSL `WaveIsFirstLane()` exactly.
                    return IrExpr::Call {
                        function: "WaveIsFirstLane".to_string(),
                        arguments: Vec::new(),
                    };
                }
                crate::ast::Expr::Variable(name)
                    if name.eq_ignore_ascii_case("wave::getactivelanecount")
                        && arguments.is_empty() =>
                {
                    // Framework `wave::GetActiveLaneCount()` (active lane count)
                    // is HLSL `WaveGetLaneCount()` exactly.
                    return IrExpr::Call {
                        function: "WaveGetLaneCount".to_string(),
                        arguments: Vec::new(),
                    };
                }
                // NOTE: Target barycentric queries (`GetParameterP0/1/2(x)`,
                // evaluating a pixel input at primitive vertices) have no
                // usable HLSL spelling in this toolchain: DXC rejects
                // `EvaluateAttributeAtVertex` (all profiles, including
                // `-HV 2021`) and Vulkan has no per-vertex evaluation, so
                // these stay loud (vel_cj deferred shaders).
                crate::ast::Expr::Variable(name)
                    if name.eq_ignore_ascii_case("__v_lerp_u8") && arguments.len() == 3 =>
                {
                    // GCN `V_LERP_U8(a, b, 1)` (unsigned byte-wise mean, used
                    // as `avg2` for mip downsampling): per-byte
                    // `(a + b + 1) >> 1`, exact via carry-free packed
                    // arithmetic (`((a^b)>>1)&0x7f.. + (a&b)` is the
                    // floor mean; `(a^b)&0x01..` rounds halves up). Only
                    // the observed `c == 1` shape; anything else stays
                    // loud (today's path).
                    let is_one = matches!(&arguments[2], crate::ast::Expr::Number(value) if parse_u32_literal(value) == Some(1));
                    if is_one {
                        let mut lowered = arguments
                            .iter()
                            .take(2)
                            .map(|a| lower_expr(a, env, resource_fields, var_struct));
                        let (Some(a), Some(b)) = (lowered.next(), lowered.next()) else {
                            unreachable!("guarded by arguments.len() == 3")
                        };
                        let xor_ab = IrExpr::Binary {
                            left: Box::new(a.clone()),
                            op: IrOp::BitXor,
                            right: Box::new(b.clone()),
                        };
                        let floor_mean = IrExpr::Binary {
                            left: Box::new(IrExpr::Binary {
                                left: Box::new(IrExpr::Binary {
                                    left: Box::new(xor_ab.clone()),
                                    op: IrOp::ShiftRight,
                                    right: Box::new(IrExpr::Literal(IrLiteral::Number(
                                        "1".to_string(),
                                    ))),
                                }),
                                op: IrOp::BitAnd,
                                right: Box::new(IrExpr::Literal(IrLiteral::Number(
                                    "0x7f7f7f7f".to_string(),
                                ))),
                            }),
                            op: IrOp::Add,
                            right: Box::new(IrExpr::Binary {
                                left: Box::new(a),
                                op: IrOp::BitAnd,
                                right: Box::new(b),
                            }),
                        };
                        return IrExpr::Binary {
                            left: Box::new(floor_mean),
                            op: IrOp::Add,
                            right: Box::new(IrExpr::Binary {
                                left: Box::new(xor_ab),
                                op: IrOp::BitAnd,
                                right: Box::new(IrExpr::Literal(IrLiteral::Number(
                                    "0x01010101".to_string(),
                                ))),
                            }),
                        };
                    }
                    name.clone()
                }
                crate::ast::Expr::Variable(name)
                    if (name.eq_ignore_ascii_case("__v_pk_min_f16")
                        || name.eq_ignore_ascii_case("__v_pk_max_f16")
                        || name.eq_ignore_ascii_case("__v_pk_add_f16")
                        || name.eq_ignore_ascii_case("__v_pk_mul_f16"))
                        && arguments.len() == 2 =>
                {
                    // GCN packed-fp16 ops (`__v_pk_min_f16(a, b)` etc. on
                    // half2 values, bc7Encode): HLSL `min`/`max`/`+`/`*`
                    // on vectors are exact (same lanes, same rounding).
                    let mut lowered = arguments
                        .iter()
                        .map(|a| lower_expr(a, env, resource_fields, var_struct));
                    let (Some(a), Some(b)) = (lowered.next(), lowered.next()) else {
                        unreachable!("guarded by arguments.len() == 2")
                    };
                    let lower = name.to_ascii_lowercase();
                    if lower == "__v_pk_min_f16" {
                        return IrExpr::Call {
                            function: "min".to_string(),
                            arguments: vec![a, b],
                        };
                    }
                    if lower == "__v_pk_max_f16" {
                        return IrExpr::Call {
                            function: "max".to_string(),
                            arguments: vec![a, b],
                        };
                    }
                    let op = if lower == "__v_pk_add_f16" { IrOp::Add } else { IrOp::Mul };
                    return IrExpr::Binary {
                        left: Box::new(a),
                        op,
                        right: Box::new(b),
                    };
                }
                crate::ast::Expr::Variable(name)
                    if name.eq_ignore_ascii_case("sad") && arguments.len() == 3 =>
                {
                    // Sum of absolute differences with accumulator
                    // (`sad(a, b, c)` = per-component `|a - b| + c`,
                    // upscale mip averaging): `max(a,b) - min(a,b)` is
                    // exact for unsigned components (no wraparound, unlike
                    // `abs(a - b)`), and HLSL resolves the accumulator
                    // addition natively. Anything else keeps today's path.
                    let mut lowered = arguments
                        .iter()
                        .map(|a| lower_expr(a, env, resource_fields, var_struct));
                    let (Some(a), Some(b), Some(c)) =
                        (lowered.next(), lowered.next(), lowered.next())
                    else {
                        unreachable!("guarded by arguments.len() == 3")
                    };
                    return IrExpr::Binary {
                        left: Box::new(IrExpr::Binary {
                            left: Box::new(IrExpr::Call {
                                function: "max".to_string(),
                                arguments: vec![a.clone(), b.clone()],
                            }),
                            op: IrOp::Sub,
                            right: Box::new(IrExpr::Call {
                                function: "min".to_string(),
                                arguments: vec![a, b],
                            }),
                        }),
                        op: IrOp::Add,
                        right: Box::new(c),
                    };
                }
                crate::ast::Expr::Variable(name)
                    if name.eq_ignore_ascii_case("__get_wavemode") && arguments.is_empty() =>
                {
                    // AMD `__get_wavemode()` returns the wave width marker;
                    // HLSL's wave width is `WaveGetLaneCount()` (the framework's
                    // own SM6 mapping: `sceGotNumWaveLanes()` →
                    // `WaveGetLaneCount()`).
                    return IrExpr::Call {
                        function: "WaveGetLaneCount".to_string(),
                        arguments: Vec::new(),
                    };
                }
                crate::ast::Expr::Variable(name)
                    if (name.eq_ignore_ascii_case("threadgroupmemorybarriersync")
                        || name.eq_ignore_ascii_case("sharedmemorybarriersync")
                        || name.eq_ignore_ascii_case("memorybarriersync")
                        || name.eq_ignore_ascii_case("pssl_sync_threads"))
                        && arguments.is_empty() =>
                {
                    // Workgroup barrier: the framework spells the Vulkan form
                    // itself (`#define ThreadGroupMemoryBarrierSync
                    // GroupMemoryBarrierWithGroupSync` in a shared
                    // CPU/GPU types header), and raw DXC confirms
                    // `ThreadGroupMemoryBarrierSync` is undeclared under
                    // `-spirv` while `GroupMemoryBarrierWithGroupSync`
                    // compiles.
                    return IrExpr::Call {
                        function: "GroupMemoryBarrierWithGroupSync".to_string(),
                        arguments: Vec::new(),
                    };
                }
                crate::ast::Expr::Variable(name)
                    if name.eq_ignore_ascii_case("threadgroupmemorybarrier")
                        && arguments.is_empty() =>
                {
                    // The non-sync barrier (`ThreadGroupMemoryBarrier`, no
                    // `Sync` suffix): HLSL `GroupMemoryBarrier()` exactly
                    // (memory + visibility, no execution sync). The shared
                    // CPU/GPU types header maps it to `GroupMemoryBarrier`.
                    return IrExpr::Call {
                        function: "GroupMemoryBarrier".to_string(),
                        arguments: Vec::new(),
                    };
                }
                crate::ast::Expr::Variable(name)
                    if name.eq_ignore_ascii_case("atomicadd")
                        && (arguments.len() == 2 || arguments.len() == 3) =>
                {
                    // `AtomicAdd(dst, v[, orig])` is HLSL
                    // `InterlockedAdd(dst, v[, orig])` exactly (same framework
                    // `#define`). Observed with 2 and 3 args; anything
                    // else keeps today's loud path.
                    return IrExpr::Call {
                        function: "InterlockedAdd".to_string(),
                        arguments: arguments
                            .iter()
                            .map(|a| lower_expr(a, env, resource_fields, var_struct))
                            .collect(),
                    };
                }
                crate::ast::Expr::Variable(name)
                    if (name.eq_ignore_ascii_case("atomicor")
                        || name.eq_ignore_ascii_case("atomicmax")
                        || name.eq_ignore_ascii_case("atomicmin")
                        || name.eq_ignore_ascii_case("atomicand")
                        || name.eq_ignore_ascii_case("atomicxor"))
                        && (arguments.len() == 2 || arguments.len() == 3) =>
                {
                    // `AtomicOr/Max/Min/And/Xor(dst, v[, orig])` are HLSL
                    // `InterlockedOr/Max/Min/And/Xor(dst, v[, orig])`
                    // exactly (same framework `#define` family as AtomicAdd
                    // above). Observed with 2 args; 3-arg form kept for
                    // symmetry. Anything else keeps today's loud path.
                    let hlsl = match name.to_ascii_lowercase().as_str() {
                        "atomicor" => "InterlockedOr",
                        "atomicmax" => "InterlockedMax",
                        "atomicmin" => "InterlockedMin",
                        "atomicand" => "InterlockedAnd",
                        _ => "InterlockedXor",
                    };
                    return IrExpr::Call {
                        function: hlsl.to_string(),
                        arguments: arguments
                            .iter()
                            .map(|a| lower_expr(a, env, resource_fields, var_struct))
                            .collect(),
                    };
                }
                crate::ast::Expr::Variable(name)
                    if name.eq_ignore_ascii_case("reversebits") && arguments.len() == 1 =>
                {
                    // `ReverseBits(x)` is HLSL `reversebits(x)` exactly
                    // (PSSL capitalizes; HLSL does not).
                    return IrExpr::Call {
                        function: "reversebits".to_string(),
                        arguments: arguments
                            .iter()
                            .map(|a| lower_expr(a, env, resource_fields, var_struct))
                            .collect(),
                    };
                }
                crate::ast::Expr::Variable(name)
                    if arguments.len() == 1
                        && unpack_byte_index(&name.to_ascii_lowercase()).is_some() =>
                {
                    // `UnpackByteN(x)` extracts byte N (LSB-first: N=0 is
                    // bits 0-7): `(x >> 8N) & 0xff`. Verified against four
                    // independent GBuffer packings in the framework headers
                    // (albedo.r=Byte3 … occlusion=Byte0 of the same word).
                    let n = unpack_byte_index(&name.to_ascii_lowercase())
                        .expect("guarded by unpack_byte_index");
                    let x = lower_expr(&arguments[0], env, resource_fields, var_struct);
                    return IrExpr::Binary {
                        left: Box::new(IrExpr::Binary {
                            left: Box::new(x),
                            op: IrOp::ShiftRight,
                            right: Box::new(IrExpr::Literal(IrLiteral::Number(
                                (8 * n).to_string(),
                            ))),
                        }),
                        op: IrOp::BitAnd,
                        right: Box::new(IrExpr::Literal(IrLiteral::Number(
                            "0xffu".to_string(),
                        ))),
                    };
                }
                crate::ast::Expr::Variable(name)
                    if name.eq_ignore_ascii_case("packuint2touint") && arguments.len() == 2 =>
                {
                    // `PackUInt2ToUInt(x, y)` packs two 16-bit halves (pixel
                    // coordinates): `(x & 0xFFFF) | ((y & 0xFFFF) << 16)`.
                    // Layout verified against the inverse `unpack_pixel_pos`
                    // in the same lighting-resolve header
                    // (`x = packed & 0xFFFF; y = packed >> 16`).
                    let x = lower_expr(&arguments[0], env, resource_fields, var_struct);
                    let y = lower_expr(&arguments[1], env, resource_fields, var_struct);
                    let mask = || {
                        Box::new(IrExpr::Literal(IrLiteral::Number("0xFFFFu".to_string())))
                    };
                    return IrExpr::Binary {
                        left: Box::new(IrExpr::Binary {
                            left: Box::new(x),
                            op: IrOp::BitAnd,
                            right: mask(),
                        }),
                        op: IrOp::BitOr,
                        right: Box::new(IrExpr::Binary {
                            left: Box::new(IrExpr::Binary {
                                left: Box::new(y),
                                op: IrOp::BitAnd,
                                right: mask(),
                            }),
                            op: IrOp::ShiftLeft,
                            right: Box::new(IrExpr::Literal(IrLiteral::Number(
                                "16".to_string(),
                            ))),
                        }),
                    };
                }
                crate::ast::Expr::Variable(name)
                    if name.eq_ignore_ascii_case("readlane") && arguments.len() == 2 =>
                {
                    // `ReadLane(v, lane)` is HLSL `WaveReadLaneAt(v, lane)`
                    // exactly (same header family as `LaneSwizzle`).
                    return IrExpr::Call {
                        function: "WaveReadLaneAt".to_string(),
                        arguments: arguments
                            .iter()
                            .map(|a| lower_expr(a, env, resource_fields, var_struct))
                            .collect(),
                    };
                }
                crate::ast::Expr::Variable(name)
                    if name.eq_ignore_ascii_case("exclusiveprefixsum")
                        && (arguments.len() == 1 || arguments.len() == 2) =>
                {
                    // `ExclusivePrefixSum(v[, maxWaveCount])` (per-lane
                    // exclusive scan) is HLSL `WavePrefixSum(v)` under the
                    // toolchain's wave==subgroup model (same assumption as
                    // every other wave intrinsic here). The optional wave
                    // count is a static bound with no DX equivalent.
                    return IrExpr::Call {
                        function: "WavePrefixSum".to_string(),
                        arguments: vec![lower_expr(
                            &arguments[0],
                            env,
                            resource_fields,
                            var_struct,
                        )],
                    };
                }
                crate::ast::Expr::Variable(name)
                    if arguments.len() == 1
                        && crosslane_wave_name(&name.to_ascii_lowercase()).is_some() =>                {
                    // Cross-lane wave reductions map exactly to HLSL
                    // `WaveActive*` intrinsics (same header family as
                    // `LaneSwizzle`/`MaskBitCnt`).
                    let wave = crosslane_wave_name(&name.to_ascii_lowercase())
                        .expect("guarded by crosslane_wave_name");
                    return IrExpr::Call {
                        function: wave.to_string(),
                        arguments: arguments
                            .iter()
                            .map(|a| lower_expr(a, env, resource_fields, var_struct))
                            .collect(),
                    };
                }
                crate::ast::Expr::Variable(name)
                    if name.eq_ignore_ascii_case("laneswizzle") && arguments.len() == 4 =>
                {
                    // AMD ds_swizzle cross-lane read, by observed
                    // (mode, mask, offset) shape across the corpus (235
                    // sites, one header family — anything else stays a
                    // call and fails loudly at DXC):
                    // - butterfly `(v, 0x1f, 0, K)`: lane XOR K
                    // - prefix steps `(v, mode, mask, 0)` for the five
                    //   fixed pairs below: lane minus 2^k (the value is
                    //   consumed only where the step applies, matching
                    //   the predicated PSSL use).
                    // Both lower to `WaveReadLaneAt`, which is exact.
                    if let Some(lane) =
                        lane_swizzle_lane(&arguments[1], &arguments[2], &arguments[3])
                    {
                        let lane_index = IrExpr::Call {
                            function: "WaveGetLaneIndex".to_string(),
                            arguments: Vec::new(),
                        };
                        let (op, shift) = match lane {
                            LaneOffset::Xor(k) => (IrOp::BitXor, k),
                            LaneOffset::Sub(k) => (IrOp::Sub, k),
                        };
                        return IrExpr::Call {
                            function: "WaveReadLaneAt".to_string(),
                            arguments: vec![
                                lower_expr(&arguments[0], env, resource_fields, var_struct),
                                IrExpr::Binary {
                                    left: Box::new(lane_index),
                                    op,
                                    right: Box::new(IrExpr::Literal(IrLiteral::Number(
                                        shift.to_string(),
                                    ))),
                                },
                            ],
                        };
                    }
                    name.clone()
                }
                crate::ast::Expr::Variable(name) => name.clone(),

                crate::ast::Expr::Member { object, member }
                    if {
                        let l = member.to_ascii_lowercase().replace('_', "");
                        matches!(
                            l.as_str(),
                            "sample"
                                | "samplelevel"
                                | "samplelod"
                                | "samplebias"
                                | "samplegrad"
                                | "samplecmp"
                                | "samplecmplod0"
                                | "load"
                                | "load4"
                                | "store"
                                | "store4"
                                | "gather"
                                | "gatherred"
                                | "gatherred0"
                                | "gathergreen"
                                | "gatherblue"
                                | "getdimensionsfast"
                                | "getdimensions"
                                | "getlod"
                                | "getlodunclamped"
                        )
                    } =>
                {
                    // -------------------------------------------------
                    // PSSL -> HLSL intrinsic mapping
                    // -------------------------------------------------
                    //
                    // PSSL:
                    //
                    //     tex.GetDimensionsFast(x, y)
                    //     tex.SampleLOD(sampler, uv, lod)
                    //     tex.SampleCmpLOD0(sampler, uv, z)
                    //     tex.GetLOD(sampler, uv)
                    //     tex.GetLODUnclamped(sampler, uv)
                    //
                    // HLSL:
                    //
                    //     tex.GetDimensions(x, y)
                    //     tex.SampleLevel(sampler, uv, lod)
                    //     tex.SampleCmpLevelZero(sampler, uv, z)
                    //     tex.CalculateLevelOfDetail(sampler, uv)
                    //     tex.CalculateLevelOfDetailUnclamped(sampler, uv)
                    //
                    // `GetDimensionsFast` / `SampleLOD` / `SampleCmpLOD0` are
                    // Target-specific spellings not understood by DXC.
                    // (`SampleCmp` already matches HLSL and passes through.)
                    //
                    // The callee object may be a member chain (e.g.
                    // `srt.tx->src`), not just a bare variable, so flatten it
                    // into a dotted path (`srt.tx.src`) before appending the
                    // HLSL method name.
                    if let Some(obj_path) = flatten_member_path(object) {
                        let method = if member.eq_ignore_ascii_case("GetDimensionsFast") {
                            "GetDimensions".to_string()
                        } else if member.eq_ignore_ascii_case("SampleLOD") {
                            "SampleLevel".to_string()
                        } else if member.eq_ignore_ascii_case("SampleCmpLOD0") {
                            "SampleCmpLevelZero".to_string()
                        } else if member.eq_ignore_ascii_case("GetLOD") {
                            "CalculateLevelOfDetail".to_string()
                        } else if member.eq_ignore_ascii_case("GetLODUnclamped") {
                            "CalculateLevelOfDetailUnclamped".to_string()
                        } else {
                            member.clone()
                        };
                        format!(
                            "{}.{method}",
                            super::rename::rename_member_root(&obj_path)
                        )
                    } else {
                        format!("<complex callee>.{member}")
                    }
                }
                crate::ast::Expr::Member { object, member }
                    if member.eq_ignore_ascii_case("mipmaps") && arguments.len() == 2 =>
                {
                    // `tex.MipMaps(0, coord)` reads mip 0: HLSL
                    // `tex.Load(coord)` exactly. Other mips normalize to
                    // index form (`tex.MipMaps(m, c)` →
                    // `tex.MipMaps[m][c]`), which the mip-view hoist pass
                    // rebinds to per-mip UAV views (single views cannot
                    // address mips). (Before the user-method catch-all
                    // below, which would otherwise pass the PSSL spelling
                    // through.)
                    let is_zero_mip = matches!(&arguments[0], crate::ast::Expr::Number(value) if parse_u32_literal(value) == Some(0));
                    if is_zero_mip {
                        if let Some(obj_path) = flatten_member_path(object) {
                            return IrExpr::Call {
                                function: format!(
                                    "{}.Load",
                                    super::rename::rename_member_root(&obj_path)
                                ),
                                arguments: vec![lower_expr(
                                    &arguments[1],
                                    env,
                                    resource_fields,
                                    var_struct,
                                )],
                            };
                        }
                    }
                    let mut lowered = arguments
                        .iter()
                        .map(|a| lower_expr(a, env, resource_fields, var_struct));
                    let (Some(mip), Some(coord)) = (lowered.next(), lowered.next()) else {
                        unreachable!("guarded by arguments.len() == 2")
                    };
                    return IrExpr::Index {
                        object: Box::new(IrExpr::Index {
                            object: Box::new(IrExpr::Member {
                                object: Box::new(lower_expr(
                                    object,
                                    env,
                                    resource_fields,
                                    var_struct,
                                )),
                                member: member.clone(),
                            }),
                            index: Box::new(mip),
                        }),
                        index: Box::new(coord),
                    };
                }
                crate::ast::Expr::Member { object, member }
                    if member.eq_ignore_ascii_case("atomicadd") =>
                {
                    // `buf.AtomicAdd(addr, v[, orig])` is HLSL
                    // `buf.InterlockedAdd(addr, v[, orig])` exactly (same
                    // framework `#define AtomicAdd InterlockedAdd` as the free
                    // function below). Observed with 2 and 3 args.
                    if let Some(obj_path) = flatten_member_path(object) {
                        return IrExpr::Call {
                            function: format!(
                                "{}.InterlockedAdd",
                                super::rename::rename_member_root(&obj_path)
                            ),
                            arguments: arguments
                                .iter()
                                .map(|a| lower_expr(a, env, resource_fields, var_struct))
                                .collect(),
                        };
                    }
                    format!("<complex callee>.{member}")
                }
                crate::ast::Expr::Member { object, member } => {
                    // User-defined struct member calls (`srt.method(args)`):
                    // methods are parsed (scope recorded) and emitted as
                    // HLSL struct methods, so a flattenable object path
                    // passes through verbatim and resolves natively
                    // (including overloads). Unflattenable objects keep
                    // complex-callee; genuinely missing methods fail at
                    // DXC — both loud. (Resource/intrinsic methods match
                    // the arm above first and are unaffected.)
                    if let Some(obj_path) = flatten_member_path(object) {
                        return IrExpr::Call {
                            function: format!(
                                "{obj_path}.{member}",
                                obj_path = super::rename::rename_member_root(&obj_path)
                            ),
                            arguments: arguments
                                .iter()
                                .map(|a| lower_expr(a, env, resource_fields, var_struct))
                                .collect(),
                        };
                    }
                    format!("<complex callee>.{member}")
                }

                _ => "<complex callee>".to_string(),
            };

            IrExpr::Call {
                function: name,
                arguments: arguments
                    .iter()
                    .map(|a| lower_expr(a, env, resource_fields, var_struct))
                    .collect(),
            }
        }

        crate::ast::Expr::Constructor { ty, arguments } => {
            // `SamplerState(samplerVar)` conversions (e.g. the
            // `SamplerState(gnmSampler[i])` elements of builtin-sampler
            // builders) are the identity: Vulkan samplers are externally
            // bound, so wrapping an already-sampler value is a no-op.
            // Builder chains (calls, not variables) keep today's loud
            // path — an unbound configurator has no HLSL meaning.
            if arguments.len() == 1
                && matches!(ty, crate::ast::Type::Named(n) if crate::framework::is_sampler_type_name(n))
                && let Some(root) = expr_root_variable(&arguments[0])
                && let Some(arg_ty) = env.get(root)
                && matches!(arg_ty, crate::ast::Type::Named(n) if crate::framework::is_sampler_type_name(n))
            {
                return lower_expr(&arguments[0], env, resource_fields, var_struct);
            }
            let lowered_ty = to_ir_type(ty);
            tracing::trace!(
                target: "pssl_shader::lower::expr",
                target_ty = ?lowered_ty,
                nargs = arguments.len(),
                "lowering constructor/cast"
            );

            // Matrix narrowing (`float3x3(viewProj4x4)`): PSSL takes the
            // upper-left submatrix; HLSL has no matrix cast, so select
            // rows explicitly (`float3x3(m[0].xyz, m[1].xyz, m[2].xyz)`).
            // Only for reference args (variables, member/index chains —
            // matrix-typed in the corpus) that are NOT literals and NOT
            // env-known scalars/vectors (e.g. `float3x3(float3Var)` keeps
            // today's path). Unknown roots stay loud at DXC if wrong.
            if let crate::ast::Type::Matrix { rows, columns, .. } = ty
                && arguments.len() == 1
                && !matches!(&arguments[0], crate::ast::Expr::Number(_) | crate::ast::Expr::Bool(_))
                && expr_root_variable(&arguments[0]).is_some_and(|root| {
                    !matches!(
                        env.get(root),
                        Some(crate::ast::Type::Float
                            | crate::ast::Type::Half
                            | crate::ast::Type::Int
                            | crate::ast::Type::UInt
                            | crate::ast::Type::Bool
                            | crate::ast::Type::Double
                            | crate::ast::Type::Vector { .. })
                    )
                })
            {
                if let Some(rows) = matrix_narrow_rows(&arguments[0], *rows, *columns) {
                    if let crate::ir::IrType::Matrix { .. } = lowered_ty {
                        return IrExpr::Construct {
                            ty: lowered_ty,
                            arguments: rows
                                .iter()
                                .map(|r| lower_expr(r, env, resource_fields, var_struct))
                                .collect(),
                        };
                    }
                }
            }

            // PSSL compatibility: `(float)vec` where `vec` is a vector is a
            // scalar extraction cast, equivalent to `float(vec.x)`. This is
            // distinguished from `float(scalar)` (a no-op identity cast) using
            // the declared type of the single argument.
            let is_scalar_target = matches!(
                ty,
                crate::ast::Type::Float
                    | crate::ast::Type::Half
                    | crate::ast::Type::Int
                    | crate::ast::Type::UInt
                    | crate::ast::Type::Bool
                    | crate::ast::Type::Double
            );
            if is_scalar_target
                && arguments.len() == 1
                && let crate::ast::Expr::Variable(arg_name) = &arguments[0]
                && let Some(crate::ast::Type::Vector { .. }) = env.get(arg_name)
            {
                tracing::trace!(
                    target: "pssl_shader::lower::expr",
                    var = %arg_name,
                    "vector cast lowered to first-component extraction (vec.x)"
                );
                IrExpr::Construct {
                    ty: lowered_ty,
                    arguments: vec![IrExpr::Member {
                        object: Box::new(IrExpr::Variable(arg_name.clone())),
                        member: "x".to_string(),
                    }],
                }
            } else if matches!(ty, crate::ast::Type::Vector { .. })
                && arguments.len() == 1
                && !matches!(&arguments[0], crate::ast::Expr::Number(_))
            {
                // Single-vector-arg construction (`int2(v)`, `half2(x)`):
                // fold to the argument, except a pure same-width vector
                // of a different base (`int2(float2v)`), which expands to
                // explicit per-component construction. Rationale: the
                // emitter splats single args, so a surviving single-arg
                // vector construct must either be a true no-op or be
                // expanded here; duplicating impure arguments would
                // duplicate side effects, so those fold (status quo).
                // Keep `half2(scalar)` splats (single Number) as a real
                // construct.
                const COMPONENTS: [&str; 4] = ["x", "y", "z", "w"];
                let convert = match ty {
                    crate::ast::Type::Vector { base: tb, width: tw } => {
                        match vector_info(&arguments[0], env) {
                            Some((sb, sw)) if sb == *tb && sw == *tw => None,
                            Some((_, sw)) if sw == *tw && expr_is_pure(&arguments[0]) => {
                                let lowered = lower_expr(
                                    &arguments[0],
                                    env,
                                    resource_fields,
                                    var_struct,
                                );
                                Some(IrExpr::Construct {
                                    ty: lowered_ty.clone(),
                                    arguments: (0..*tw)
                                        .map(|i| IrExpr::Member {
                                            object: Box::new(lowered.clone()),
                                            member: COMPONENTS[i as usize].to_string(),
                                        })
                                        .collect(),
                                })
                            }
                            _ => None,
                        }
                    }
                    _ => None,
                };
                match convert {
                    Some(expr) => expr,
                    None => lower_expr(&arguments[0], env, resource_fields, var_struct),
                }
            } else {
                IrExpr::Construct {
                    ty: lowered_ty,
                    arguments: arguments
                        .iter()
                        .map(|a| lower_expr(a, env, resource_fields, var_struct))
                        .collect(),
                }
            }
        }

        crate::ast::Expr::Unary { op, expr } => IrExpr::Unary {
            op: lower_unary(op),
            expr: Box::new(lower_expr(expr, env, resource_fields, var_struct)),
        },

        crate::ast::Expr::Binary { left, op, right } => {
            // Bitwise ops on float/half (`float2 v & 1`): PSSL truncates
            // to int component-wise; HLSL requires int/uint operands.
            // Pure float sides expand to explicit per-component int
            // construction (`int2(v.x, v.y)`); other sides pass through.
            // Purity-gated: expansion duplicates evaluation, so operands
            // with calls/assignments keep today's loud path.
            if matches!(
                op,
                crate::ast::BinaryOp::BitAnd
                    | crate::ast::BinaryOp::BitOr
                    | crate::ast::BinaryOp::BitXor
                    | crate::ast::BinaryOp::ShiftLeft
                    | crate::ast::BinaryOp::ShiftRight
            ) && expr_is_pure(left)
                && expr_is_pure(right)
            {
                let floatish = |e: &crate::ast::Expr| match vector_info(e, env) {
                    Some((crate::ast::ScalarType::Float, w))
                    | Some((crate::ast::ScalarType::Half, w)) => Some(w),
                    _ => match e {
                        crate::ast::Expr::Variable(n) => match env.get(n) {
                            Some(crate::ast::Type::Float)
                            | Some(crate::ast::Type::Half)
                            | Some(crate::ast::Type::Double) => Some(1),
                            _ => None,
                        },
                        _ => None,
                    },
                };
                let lw = floatish(left);
                let rw = floatish(right);
                if lw.is_some() || rw.is_some() {
                    // Convert only float sides; int/uint/bool sides pass
                    // through untouched. Vector sides expand to explicit
                    // per-component construction (single-vector-arg
                    // construction would mis-emit under splatting).
                    const COMPONENTS: [&str; 4] = ["x", "y", "z", "w"];
                    let convert = |e: &crate::ast::Expr, w: Option<u8>| {
                        let lowered = lower_expr(e, env, resource_fields, var_struct);
                        match w {
                            Some(w) if w > 1 => IrExpr::Construct {
                                ty: IrType::Vector {
                                    base: IrScalar::Int,
                                    width: w,
                                },
                                arguments: (0..w)
                                    .map(|i| IrExpr::Member {
                                        object: Box::new(lowered.clone()),
                                        member: COMPONENTS[i as usize].to_string(),
                                    })
                                    .collect(),
                            },
                            Some(_) => IrExpr::Construct {
                                ty: IrType::Int,
                                arguments: vec![lowered],
                            },
                            None => lowered,
                        }
                    };
                    return IrExpr::Binary {
                        left: Box::new(convert(left, lw)),
                        op: lower_binary(op),
                        right: Box::new(convert(right, rw)),
                    };
                }
            }
            // Vector short-circuit (`bool3 a && bool3 b`): HLSL rejects
            // `&&`/`||` on vectors ("use 'and'"). Lower pure-bool-vector
            // operands to the component-wise `and()`/`or()` intrinsics
            // (which broadcast scalars, as probed). Purity-gated: `and`
            // evaluates both sides, so operands with calls/assignments
            // keep today's loud path rather than dropping short-circuit.
            // Scalar `&&`/`||` (and anything unresolvable) are untouched.
            if matches!(
                op,
                crate::ast::BinaryOp::LogicalAnd | crate::ast::BinaryOp::LogicalOr
            ) && expr_is_pure(left)
                && expr_is_pure(right)
            {
                let left_info = vector_info(left, env);
                let right_info = vector_info(right, env);
                let vec_side = [left_info, right_info].iter().any(|info| {
                    matches!(info, Some((crate::ast::ScalarType::Bool, _)))
                });
                if vec_side {
                    return IrExpr::Call {
                        function: if matches!(op, crate::ast::BinaryOp::LogicalAnd) {
                            "and".to_string()
                        } else {
                            "or".to_string()
                        },
                        arguments: vec![
                            lower_expr(left, env, resource_fields, var_struct),
                            lower_expr(right, env, resource_fields, var_struct),
                        ],
                    };
                }
            }
            // Wave ballot against zero (`(ballot(v) != 0)`,
            // `(ballot(v) == 0)`): PSSL `ballot` returns a scalar lane
            // mask, but HLSL `WaveActiveBallot` returns uint4, so the
            // comparison is a bool4 vector — rejected as a ternary
            // condition (and wrong elsewhere). `!= 0` asks "any lane
            // active" → `any(...)`; `== 0` asks "no lane active" →
            // `all(... == 0)`. Only literal-zero comparisons qualify;
            // anything else keeps today's path.
            if matches!(
                op,
                crate::ast::BinaryOp::Equal | crate::ast::BinaryOp::NotEqual
            ) {
                // `(ballot, zero)` in either order (`0 != ballot(v)` ≡
                // `ballot(v) != 0`).
                fn as_ballot(e: &crate::ast::Expr) -> bool {
                    matches!(e, crate::ast::Expr::Call { function, arguments }
                        if arguments.len() == 1
                            && matches!(function.as_ref(), crate::ast::Expr::Variable(b) if b.eq_ignore_ascii_case("ballot")))
                }
                fn as_zero(e: &crate::ast::Expr) -> bool {
                    match e {
                        crate::ast::Expr::Number(s) => {
                            let t = s.trim_end_matches(|c| {
                                c == 'u' || c == 'U' || c == 'l' || c == 'L'
                            });
                            t.parse::<u32>().is_ok_and(|v| v == 0)
                        }
                        _ => false,
                    }
                }
                if (as_ballot(left) && as_zero(right))
                    || (as_ballot(right) && as_zero(left))
                {
                    let cmp = IrExpr::Binary {
                        left: Box::new(lower_expr(left, env, resource_fields, var_struct)),
                        op: lower_binary(op),
                        right: Box::new(lower_expr(right, env, resource_fields, var_struct)),
                    };
                    let agg = if matches!(op, crate::ast::BinaryOp::NotEqual) {
                        "any"
                    } else {
                        "all"
                    };
                    return IrExpr::Call {
                        function: agg.to_string(),
                        arguments: vec![cmp],
                    };
                }
            }
            IrExpr::Binary {
                left: Box::new(lower_expr(left, env, resource_fields, var_struct)),
                op: lower_binary(op),
                right: Box::new(lower_expr(right, env, resource_fields, var_struct)),
            }
        }

        crate::ast::Expr::Conditional {
            condition,
            then_branch,
            else_branch,
        } => {
            // Vector-condition ternary (`(x < k) ? a : b` with vector
            // operands): HLSL `?:` needs a scalar condition — lower to
            // `select(cond, true, false)`, which takes a bool vector and
            // works in every position (returns, args, nesting), unlike
            // the declaration-only memberwise path in the emitter.
            //
            // Only fires when the emitter's memberwise gate cannot: that
            // path owns bool-vector-constructor conditions (kept verbatim
            // here so its tests stay byte-identical). Branches must be
            // pure (`select` evaluates both sides) with matching known
            // widths; scalar literal/variable branches splat to the
            // condition width. Anything else keeps today's loud path.
            if let Some(select) = lower_vector_select(
                condition,
                then_branch,
                else_branch,
                env,
                resource_fields,
                var_struct,
            ) {
                return select;
            }
            IrExpr::Conditional {
                condition: Box::new(lower_expr(condition, env, resource_fields, var_struct)),
                then_branch: Box::new(lower_expr(then_branch, env, resource_fields, var_struct)),
                else_branch: Box::new(lower_expr(else_branch, env, resource_fields, var_struct)),
            }
        }

        crate::ast::Expr::Assign { lhs, rhs } => IrExpr::Assign {
            lhs: Box::new(lower_expr(lhs, env, resource_fields, var_struct)),
            rhs: Box::new(lower_expr(rhs, env, resource_fields, var_struct)),
        },

        crate::ast::Expr::InitializerList(elems) => IrExpr::InitializerList(
            elems
                .iter()
                .map(|e| lower_expr(e, env, resource_fields, var_struct))
                .collect(),
        ),

        // Unreduced lambdas (captures, passed-as-value, multi-statement):
        // same loud `__lambda()` shape as the legacy opaque placeholder.
        crate::ast::Expr::Lambda { .. } => IrExpr::Construct {
            ty: IrType::Named("__lambda".to_string()),
            arguments: Vec::new(),
        },
    }
}

/// Lowers a vector-condition ternary to HLSL `select(cond, true, false)`
/// when the emitter's declaration-only memberwise path cannot fire.
/// Returns `None` (caller keeps the loud `?:`) unless: the condition is
/// a non-constructor vector of width 2..=4 (bool-vector constructors
/// stay verbatim for the memberwise gate), both branches are pure and
/// have known widths matching the condition width (scalar
/// literal/variable branches splat), and non-bool conditions wrap with
/// `!= 0`.
fn lower_vector_select(
    condition: &crate::ast::Expr,
    then_branch: &crate::ast::Expr,
    else_branch: &crate::ast::Expr,
    env: &std::collections::HashMap<String, crate::ast::Type>,
    resource_fields: &HashMap<(String, String), Type>,
    var_struct: &HashMap<String, String>,
) -> Option<IrExpr> {
    use crate::ast::{ScalarType, Type};
    // Memberwise domain: bool-vector-constructor conditions.
    if let crate::ast::Expr::Constructor { ty, .. } = condition {
        if matches!(
            ty,
            Type::Vector {
                base: ScalarType::Bool,
                ..
            }
        ) {
            return None;
        }
    }
    let (cond_base, cond_width) = vector_info(condition, env)?;
    if cond_width < 2 || cond_width > 4 {
        return None;
    }
    if !expr_is_pure(then_branch) || !expr_is_pure(else_branch) {
        return None;
    }
    // Known branch widths: vectors from `vector_info`, scalars
    // (literals and scalar-typed variables) as width 1 for splatting.
    fn branch_width(
        e: &crate::ast::Expr,
        env: &std::collections::HashMap<String, crate::ast::Type>,
    ) -> Option<u8> {
        if let Some((_, w)) = vector_info(e, env) {
            return Some(w);
        }
        match e {
            crate::ast::Expr::Number(_) | crate::ast::Expr::Bool(_) => Some(1),
            crate::ast::Expr::Variable(name) => match env.get(name) {
                Some(
                    Type::Float
                    | Type::Half
                    | Type::Int
                    | Type::UInt
                    | Type::Bool
                    | Type::Double,
                ) => Some(1),
                _ => None,
            },
            _ => None,
        }
    }
    let (wt, we) = (branch_width(then_branch, env)?, branch_width(else_branch, env)?);
    if (wt != cond_width && wt != 1) || (we != cond_width && we != 1) {
        return None;
    }
    let mut cond_ir = lower_expr(condition, env, resource_fields, var_struct);
    if cond_base != ScalarType::Bool {
        cond_ir = IrExpr::Binary {
            left: Box::new(cond_ir),
            op: IrOp::NotEqual,
            right: Box::new(IrExpr::Literal(IrLiteral::Number("0".to_string()))),
        };
    }
    // Splat scalar branches to the condition width.
    fn splat(
        branch: &crate::ast::Expr,
        width: u8,
        cond_width: u8,
        env: &std::collections::HashMap<String, crate::ast::Type>,
        resource_fields: &HashMap<(String, String), Type>,
        var_struct: &HashMap<String, String>,
    ) -> Option<IrExpr> {
        let lowered = lower_expr(branch, env, resource_fields, var_struct);
        if width == cond_width {
            return Some(lowered);
        }
        let base = match branch {
            crate::ast::Expr::Bool(_) => IrScalar::Bool,
            crate::ast::Expr::Number(s) => {
                let t = s.trim_end_matches(|c| c == 'f' || c == 'F' || c == 'h' || c == 'H');
                if t.starts_with("0x") || t.starts_with("0X") {
                    IrScalar::UInt
                } else if t.ends_with('u') || t.ends_with('U') {
                    IrScalar::UInt
                } else if t.contains('.') || t.contains('e') || t.contains('E') {
                    IrScalar::Float
                } else {
                    IrScalar::Int
                }
            }
            crate::ast::Expr::Variable(name) => match env.get(name) {
                Some(Type::Float) => IrScalar::Float,
                Some(Type::Half) => IrScalar::Half,
                Some(Type::Int) => IrScalar::Int,
                Some(Type::UInt) => IrScalar::UInt,
                Some(Type::Bool) => IrScalar::Bool,
                _ => return None,
            },
            _ => return None,
        };
        Some(IrExpr::Construct {
            ty: IrType::Vector { base, width: cond_width },
            arguments: vec![lowered],
        })
    }
    let then_ir = splat(then_branch, wt, cond_width, env, resource_fields, var_struct)?;
    let else_ir = splat(else_branch, we, cond_width, env, resource_fields, var_struct)?;
    Some(IrExpr::Call {
        function: "select".to_string(),
        arguments: vec![cond_ir, then_ir, else_ir],
    })
}

/// Root variable of a value expression (`v`, `v[i]`, `v.m` → `v`);
/// `None` for calls and anything else (e.g. builder chains).
fn expr_root_variable(expr: &crate::ast::Expr) -> Option<&String> {    match expr {
        crate::ast::Expr::Variable(name) => Some(name),
        crate::ast::Expr::Index { object, .. } | crate::ast::Expr::Member { object, .. } => {
            expr_root_variable(object)
        }
        _ => None,
    }
}

/// Side-effect-free expressions (no calls, no assignments): safe to
/// duplicate or reorder, notably for vector `&&`/`||` → `and()`/`or()`
/// which evaluates both sides.
fn expr_is_pure(expr: &crate::ast::Expr) -> bool {
    match expr {
        crate::ast::Expr::Number(_)
        | crate::ast::Expr::String(_)
        | crate::ast::Expr::Bool(_)
        | crate::ast::Expr::Variable(_) => true,
        crate::ast::Expr::Call { function, arguments } => {
            // Pure arithmetic intrinsics: duplicating them into both
            // sides of `select`/`and`/`or` cannot change values (no
            // memory, derivative, wave, or control effects). Method
            // calls (texture/buffer sampling, struct methods), wave
            // ops, atomics, barriers, and user functions stay loud:
            // `select` evaluates both sides, so any effectful callee
            // would execute unconditionally.
            if let crate::ast::Expr::Variable(name) = function.as_ref() {
                if is_pure_intrinsic(name) {
                    return arguments.iter().all(expr_is_pure);
                }
            }
            false
        }
        crate::ast::Expr::Member { object, .. } => expr_is_pure(object),
        crate::ast::Expr::Index { object, index } => {
            expr_is_pure(object) && expr_is_pure(index)
        }
        crate::ast::Expr::Constructor { arguments, .. } => {
            arguments.iter().all(expr_is_pure)
        }
        crate::ast::Expr::Unary { expr: inner, .. } => expr_is_pure(inner),
        crate::ast::Expr::Binary { left, right, .. } => {
            expr_is_pure(left) && expr_is_pure(right)
        }
        crate::ast::Expr::Conditional {
            condition,
            then_branch,
            else_branch,
        } => expr_is_pure(condition) && expr_is_pure(then_branch) && expr_is_pure(else_branch),
        crate::ast::Expr::InitializerList(elems) => elems.iter().all(expr_is_pure),
        // Calls (intrinsics included: they may write globals/buffers)
        // and assignments stay loud.
        _ => false,
    }
}

/// Pure arithmetic intrinsic names for [`expr_is_pure`]: per-component
/// math with no memory, derivative, wave, or control effects, so
/// evaluating both sides of a duplicated select/and/or is
/// value-identical. Deliberately excludes texture/buffer sampling
/// (implicit derivatives), wave ballot/broadcast (divergence
/// dependent), atomics/barriers (side effects), and user functions
/// (unknown bodies).
fn is_pure_intrinsic(name: &str) -> bool {
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
    )
}

/// Dotted path of a variable/member chain (`srt.view.wvp`), else `None`.
fn member_path(expr: &crate::ast::Expr) -> Option<String> {
    match expr {
        crate::ast::Expr::Variable(name) => Some(name.clone()),
        crate::ast::Expr::Member { object, member } => {
            member_path(object).map(|p| format!("{p}.{member}"))
        }
        _ => None,
    }
}

/// Vector-ness of an expression: `Some((base, width))` for vectors
/// (including bool-typed comparison/logical results), else `None`.
/// Unknown shapes (calls, indexing, unresolvable members) yield `None`
/// so vector-only rewrites never misfire.
fn vector_info(
    expr: &crate::ast::Expr,
    env: &std::collections::HashMap<String, crate::ast::Type>,
) -> Option<(crate::ast::ScalarType, u8)> {
    use crate::ast::BinaryOp::*;
    match expr {
        crate::ast::Expr::Variable(name) => match env.get(name) {
            Some(crate::ast::Type::Vector { base, width }) => Some((*base, *width)),
            _ => None,
        },
        crate::ast::Expr::Member { object, member } => {
            // Dotted environment entries first (`psIn.uv` registered for
            // struct-typed variables): resolves struct members exactly.
            if let Some(path) = member_path(expr) {
                if let Some(crate::ast::Type::Vector { base, width }) = env.get(&path) {
                    return Some((*base, *width));
                }
            }
            let (base, _) = vector_info(object, env)?;
            if member.len() > 1
                && member.len() <= 4
                && member.chars().all(|c| "xyzwrgbaXYZWRGBA".contains(c))
            {
                Some((base, member.len() as u8))
            } else {
                None
            }
        }
        crate::ast::Expr::Unary { op, expr: inner } => {
            use crate::ast::UnaryOp::*;
            match op {
                // `!boolN` stays a bool vector; arithmetic preserves shape.
                Not => {
                    if let Some((crate::ast::ScalarType::Bool, w)) = vector_info(inner, env) {
                        Some((crate::ast::ScalarType::Bool, w))
                    } else {
                        None
                    }
                }
                Plus | Minus | BitNot => vector_info(inner, env),
            }
        }
        crate::ast::Expr::Binary { left, op, right } => match op {
            Add | Sub | Mul | Div | Mod => vector_info(left, env).or_else(|| vector_info(right, env)),
            Equal | NotEqual | Less | LessEqual | Greater | GreaterEqual => {
                let w = vector_info(left, env)
                    .map(|(_, w)| w)
                    .or_else(|| vector_info(right, env).map(|(_, w)| w));
                w.map(|w| (crate::ast::ScalarType::Bool, w))
            }
            LogicalAnd | LogicalOr => [left, right].iter().find_map(|e| match vector_info(e, env) {
                Some((crate::ast::ScalarType::Bool, w)) => Some((crate::ast::ScalarType::Bool, w)),
                _ => None,
            }),
            _ => None,
        },
        crate::ast::Expr::Conditional {
            then_branch,
            else_branch,
            ..
        } => vector_info(then_branch, env).or_else(|| vector_info(else_branch, env)),
        crate::ast::Expr::Constructor { ty, .. } => match ty {
            crate::ast::Type::Vector { base, width } => Some((*base, *width)),
            _ => None,
        },
        crate::ast::Expr::Call { function, arguments } => {
            let name = match function.as_ref() {
                crate::ast::Expr::Variable(n) => n.to_ascii_lowercase(),
                _ => return None,
            };
            // Bool predicates (`isnan`/`isinf`/`isfinite`): the result
            // is bool with the argument's width, so `!isnan(float3)` is
            // a bool3 and an enclosing `&&`/`||` must lower to
            // element-wise `and()`/`or()` (HLSL rejects short-circuit
            // operators on vectors). Scalar (width-1 / unresolvable)
            // arguments yield scalar bools, so stay `None` here and keep
            // scalar short-circuit semantics.
            if matches!(name.as_str(), "isnan" | "isinf" | "isfinite") {
                for a in arguments {
                    if let Some((_, w)) = vector_info(a, env) {
                        if w > 1 {
                            return Some((crate::ast::ScalarType::Bool, w));
                        }
                        return None;
                    }
                }
                return None;
            }
            // Same-shape pure builtins (`pow(float3, float)` → float3):
            // result width follows the first vector argument, so
            // vector ternaries over builtin calls lower to `select`
            // instead of staying loud. Only the duplication-safe
            // intrinsics qualify (reductions like `dot`/`length` return
            // scalars and must never splat to vector width).
            if !matches!(
                name.as_str(),
                "pow" | "mul"
                    | "normalize"
                    | "saturate"
                    | "clamp"
                    | "lerp"
                    | "mix"
                    | "min"
                    | "max"
                    | "abs"
                    | "sign"
                    | "floor"
                    | "ceil"
                    | "round"
                    | "trunc"
                    | "frac"
                    | "fmod"
                    | "smoothstep"
                    | "reflect"
                    | "refract"
                    | "faceforward"
                    | "cross"
                    | "mad"
                    | "cndmask"
                    | "select"
            ) {
                return None;
            }
            for a in arguments {
                if let Some((base, w)) = vector_info(a, env) {
                    return Some((base, w));
                }
            }
            None
        }
        crate::ast::Expr::Assign { rhs, .. } => vector_info(rhs, env),
        _ => None,
    }
}

/// Row selections for matrix narrowing (`float3x3(m)` → `m[0].xyz`,
/// `m[1].xyz`, `m[2].xyz`): the upper-left `rows`×`columns` submatrix.
/// `None` outside 1..=4 (HLSL has no larger matrices).
fn matrix_narrow_rows(
    arg: &crate::ast::Expr,
    rows: u8,
    columns: u8,
) -> Option<Vec<crate::ast::Expr>> {
    if !(1..=4).contains(&rows) || !(1..=4).contains(&columns) {
        return None;
    }
    let swizzle = match columns {
        1 => Some("x"),
        2 => Some("xy"),
        3 => Some("xyz"),
        _ => None,
    };
    Some(
        (0..rows)
            .map(|i| {
                let row = crate::ast::Expr::Index {
                    object: Box::new(arg.clone()),
                    index: Box::new(crate::ast::Expr::Number(i.to_string())),
                };
                match swizzle {
                    Some(s) => crate::ast::Expr::Member {
                        object: Box::new(row),
                        member: s.to_string(),
                    },
                    None => row,
                }
            })
            .collect(),
    )
}

/// Maps `unpackbyte0..3` (lowercased) to the byte index (LSB-first).
/// Only these four exist; anything else returns `None` and stays loud.
fn unpack_byte_index(lower: &str) -> Option<u32> {
    match lower {
        "unpackbyte0" => Some(0),
        "unpackbyte1" => Some(1),
        "unpackbyte2" => Some(2),
        "unpackbyte3" => Some(3),
        _ => None,
    }
}

/// Maps a cross-lane reduction intrinsic (lowercased) to its exact HLSL
/// `WaveActive*` equivalent. Only observed single-arg corpus shapes are
/// listed; anything else returns `None` and stays loud at DXC.
fn crosslane_wave_name(lower: &str) -> Option<&'static str> {    match lower {
        "crosslanemin" => Some("WaveActiveMin"),
        "crosslanemax" => Some("WaveActiveMax"),
        "crosslaneor" => Some("WaveActiveBitOr"),
        "crosslaneadd" => Some("WaveActiveSum"),
        _ => None,
    }
}

/// Lane computation for a `LaneSwizzle(v, mode, mask, off)` call that maps
/// to `WaveReadLaneAt`: either XOR (butterfly) or subtract (prefix steps)
/// by a literal shift.
enum LaneOffset {
    Xor(u32),
    Sub(u32),
}

/// Resolves the `(mode, mask, off)` operands of `LaneSwizzle` to a lane
/// computation, or `None` for unobserved shapes (left loud at DXC).
/// Observed corpus shapes, all from one header family:
/// butterfly `(0x1f, 0, K)` and the five prefix-step pairs.
fn lane_swizzle_lane(
    mode: &crate::ast::Expr,
    mask: &crate::ast::Expr,
    off: &crate::ast::Expr,
) -> Option<LaneOffset> {
    use crate::ast::Expr;
    let num = |e: &Expr| match e {
        Expr::Number(value) => parse_u32_literal(value),
        _ => None,
    };
    let (mode, mask, off) = (num(mode)?, num(mask)?, num(off)?);
    if mode == 0x1f && mask == 0 {
        return Some(LaneOffset::Xor(off));
    }
    match (mode, mask) {
        (0x1e, 0x00) => Some(LaneOffset::Sub(1)),
        (0x1c, 0x01) => Some(LaneOffset::Sub(2)),
        (0x18, 0x03) => Some(LaneOffset::Sub(4)),
        (0x10, 0x07) => Some(LaneOffset::Sub(8)),
        (0x00, 0x0f) => Some(LaneOffset::Sub(16)),
        _ => None,
    }
}

/// Parses an integer literal lexeme (`0`, `1`, `0x0`, `1u`, `0b10`,
/// `1'000`) to `u32`. Returns `None` for non-integers (floats,
/// identifiers). Binary literals matter: cross-lane helpers spell
/// butterfly masks as `0b1`/`0b10`/`0b100` (LaneSwizzle), which
/// previously fell through to loud verbatim calls.
fn parse_u32_literal(lexeme: &str) -> Option<u32> {
    let t = lexeme.trim();
    // Integer suffixes (`1u`, `0ul`, `0x1eu`): strip from the end.
    let t = t.strip_suffix('l').or_else(|| t.strip_suffix('L')).unwrap_or(t);
    let t = t.strip_suffix('u').or_else(|| t.strip_suffix('U')).unwrap_or(t);
    // C++ digit separators (`1'000`, `0x1'0000`): the lexer keeps them
    // in the lexeme.
    let t: String = t.chars().filter(|c| *c != '\'').collect();
    if let Some(hex) = t.strip_prefix("0x").or_else(|| t.strip_prefix("0X")) {
        u32::from_str_radix(hex, 16).ok()
    } else if let Some(bin) = t.strip_prefix("0b").or_else(|| t.strip_prefix("0B")) {
        u32::from_str_radix(bin, 2).ok()
    } else {
        t.parse().ok()
    }
}

/// Maps a normalized PSSL `GatherLOD*` method name to its HLSL `Gather*`
/// equivalent (explicit LOD is unsupported in HLSL/SPIR-V). Returns `None`
/// for anything that is not an explicit-LOD gather.
fn gather_lod_hlsl_name(normalized: &str) -> Option<&'static str> {
    match normalized {
        "gatherlod" => Some("Gather"),
        "gatherlodred" => Some("GatherRed"),
        "gatherlodgreen" => Some("GatherGreen"),
        "gatherlodblue" => Some("GatherBlue"),
        "gatherlodalpha" => Some("GatherAlpha"),
        "gatherlodcmp" => Some("GatherCmp"),
        "gatherlodcmpred" => Some("GatherCmpRed"),
        _ => None,
    }
}

/// Flattens a `Variable` / `Member` expression chain into a dotted path string
/// (e.g. `srt.tx->src` → `"srt.tx.src"`). Returns `None` for non-pathable
/// expressions.
fn flatten_member_path(expr: &crate::ast::Expr) -> Option<String> {
    match expr {
        crate::ast::Expr::Variable(v) => Some(v.clone()),
        crate::ast::Expr::Member { object, member } => {
            flatten_member_path(object).map(|p| format!("{p}.{member}"))
        }
        // Bindless texture arrays: `textures[0].Sample(...)` /
        // `textures[i].Sample(...)` are valid HLSL (indexable resource
        // arrays), so keep the index instead of degrading to
        // `<complex callee>`. Only simple indices (literals, variables)
        // flatten; anything else stays unrepresentable.
        crate::ast::Expr::Index { object, index } => {
            let base = flatten_member_path(object)?;
            let idx = match index.as_ref() {
                crate::ast::Expr::Number(n) => n.clone(),
                // Dotted name paths (`idx.x`, `input.entityIndex`) spell
                // identically in HLSL, so flatten like variables; only
                // real expressions (arithmetic, calls) stay unrepresentable.
                _ => flatten_member_path(index)?,
            };
            Some(format!("{base}[{idx}]"))
        }
        _ => None,
    }
}

/// Returns the name of the struct that `object` (a `Variable`) refers to, if
/// it is a struct-typed global or local. Used to locate resource members.
fn object_struct_name(
    env: &HashMap<String, crate::ast::Type>,
    var_struct: &HashMap<String, String>,
    object: &crate::ast::Expr,
) -> Option<String> {
    let crate::ast::Expr::Variable(var) = object else {
        return None;
    };
    if let Some(s) = var_struct.get(var) {
        return Some(s.clone());
    }
    if let Some(crate::ast::Type::Named(s)) = env.get(var) {
        return Some(s.clone());
    }
    None
}

/// Whether `expr` is `object.member` where `member` is a resource-typed field
/// of the struct that `object` refers to (e.g. `srt_data.dest` when `dest` is
/// an `RW_Texture2D<float>` field of `DepthResolveSrt`).
fn is_resource_member_expr(
    env: &HashMap<String, crate::ast::Type>,
    resource_fields: &HashMap<(String, String), Type>,
    var_struct: &HashMap<String, String>,
    expr: &crate::ast::Expr,
) -> bool {
    let crate::ast::Expr::Member { object, member } = expr else {
        return false;
    };
    let Some(struct_name) = object_struct_name(env, var_struct, object) else {
        return false;
    };
    resource_fields.contains_key(&(struct_name, member.clone()))
}

/// Folds a variable vector texel offset into a `Load` coordinate:
/// `Load(c, o)` -> `Load(c + intN(o.x, o.y, ..., 0, ...))`. The offset's
/// `ow` leading components fill the head of a `cw`-wide vector (same scalar
/// base), with the remaining components zeroed (the trailing mip/sample).
/// HLSL requires immediate offsets, so a runtime offset must be additively
/// baked into the coordinate.
fn load_offset_fold(
    coords: IrExpr,
    offset: IrExpr,
    base: crate::ast::ScalarType,
    cw: u8,
    ow: u8,
) -> IrExpr {
    const COMPONENTS: [&str; 4] = ["x", "y", "z", "w"];
    let base = match base {
        crate::ast::ScalarType::Float => IrScalar::Float,
        crate::ast::ScalarType::Half => IrScalar::Half,
        crate::ast::ScalarType::Int => IrScalar::Int,
        crate::ast::ScalarType::UInt => IrScalar::UInt,
        crate::ast::ScalarType::Bool => IrScalar::Bool,
    };
    let mut args: Vec<IrExpr> = (0..ow)
        .map(|i| IrExpr::Member {
            object: Box::new(offset.clone()),
            member: COMPONENTS[i as usize].to_string(),
        })
        .collect();
    for _ in ow..cw {
        args.push(IrExpr::Literal(IrLiteral::Number("0".to_string())));
    }
    IrExpr::Binary {
        left: Box::new(coords),
        op: IrOp::Add,
        right: Box::new(IrExpr::Construct {
            ty: IrType::Vector { base, width: cw },
            arguments: args,
        }),
    }
}

pub(crate) fn lower_binary(op: &crate::ast::BinaryOp) -> IrOp {
    use crate::ast::BinaryOp::*;

    match op {
        Add => IrOp::Add,
        Sub => IrOp::Sub,
        Mul => IrOp::Mul,
        Div => IrOp::Div,
        Mod => IrOp::Mod,

        Equal => IrOp::Equal,
        NotEqual => IrOp::NotEqual,
        Less => IrOp::Less,
        LessEqual => IrOp::LessEqual,
        Greater => IrOp::Greater,
        GreaterEqual => IrOp::GreaterEqual,

        LogicalAnd => IrOp::LogicalAnd,
        LogicalOr => IrOp::LogicalOr,

        BitAnd => IrOp::BitAnd,
        BitOr => IrOp::BitOr,
        BitXor => IrOp::BitXor,

        ShiftLeft => IrOp::ShiftLeft,
        ShiftRight => IrOp::ShiftRight,
        Comma => IrOp::Comma,
    }
}

pub(crate) fn lower_unary(op: &crate::ast::UnaryOp) -> IrOp {
    use crate::ast::UnaryOp::*;

    match op {
        Plus => IrOp::Add,
        Minus => IrOp::Negate,
        Not => IrOp::Not,
        BitNot => IrOp::BitNot,
    }
}
