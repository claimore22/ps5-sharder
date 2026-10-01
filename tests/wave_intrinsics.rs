use ps5_shader::parse_shader;

// Regression tests for wave bit intrinsics (clustered-geometry family):
// `MaskBitCnt(m)` = popcount of mask bits below the current lane (AMD
// mbcnt; PSSL `ulong` is 32-bit) lowers to the `__maskbitcnt_u32` shim;
// the all-ones literal (`~0ul`) is the lane index itself.
// `popcnt(x)` is HLSL `countbits(x)` exactly.

fn emit(source: &str) -> String {
    let module = parse_shader(source).expect("parser accepted shader");
    let model = ps5_shader::semantic::analyze(&module).expect("semantic analysis succeeded");
    let ir = ps5_shader::lower::lower_module(&module, &model).expect("lowering succeeded");
    ps5_shader::emit::emit_vulkan_hlsl(&ir)
}

const PRELUDE: &str = "[NUM_THREADS(64,1,1)]\nvoid main(uint position : S_DISPATCH_THREAD_ID) {\n";

#[test]
fn maskbitcnt_lowes_to_shim_call() {
    let hlsl = emit(&format!("{PRELUDE}ulong mask = 0u;\nuint index = MaskBitCnt(mask);\n}}"));
    assert!(
        hlsl.contains("__maskbitcnt_u32(mask)"),
        "must call the shim, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("uint __maskbitcnt_u32(uint m)"),
        "shim must be emitted gated-on-use, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("MaskBitCnt"),
        "intrinsic must not leak into HLSL, got:\n{hlsl}"
    );
}

#[test]
fn maskbitcnt_all_ones_is_lane_index() {
    let hlsl = emit(&format!("{PRELUDE}uint laneId = MaskBitCnt(~0ul);\n}}"));
    assert!(
        hlsl.contains("WaveGetLaneIndex()"),
        "~0ul must become the lane index, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("MaskBitCnt"),
        "intrinsic must not leak into HLSL, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("0ul"),
        "ul-suffixed literal must not leak into HLSL, got:\n{hlsl}"
    );
}

#[test]
fn laneswizzle_binary_butterfly_maps() {
    // Butterfly masks spelled binary (`0b1`/`0b10`/`0b100`) must parse:
    // `LaneSwizzle(v, 0x1f, 0, K)` is lane-XOR-K (downsample mip chain).
    let hlsl = emit(&format!(
        "{PRELUDE}float x = LaneSwizzle(0.5f, 0x1f, 0, 0b10);\nfloat y = LaneSwizzle(x, 0x1f, 0, 0b1);\n}}"
    ));
    assert!(
        hlsl.contains("WaveReadLaneAt(0.5f, (WaveGetLaneIndex() ^ 2))")
            || hlsl.contains("WaveReadLaneAt(0.5f, (WaveGetLaneIndex()^2))"),
        "binary butterfly must map, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("LaneSwizzle"),
        "intrinsic must not leak into HLSL, got:\n{hlsl}"
    );
}

#[test]
fn popcnt_lowes_to_countbits() {
    let hlsl = emit(&format!("{PRELUDE}uint c = popcnt(position);\n}}"));
    assert!(
        hlsl.contains("countbits(position)"),
        "popcnt must become countbits, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("popcnt"),
        "intrinsic must not leak into HLSL, got:\n{hlsl}"
    );
}

#[test]
fn shim_absent_when_unused() {
    let hlsl = emit(&format!("{PRELUDE}uint c = position + 1u;\n}}"));
    assert!(
        !hlsl.contains("__maskbitcnt_u32"),
        "shim must not be emitted when unused, got:\n{hlsl}"
    );
}

#[test]
fn laneswizzle_butterfly_is_xor_readlane() {
    let hlsl = emit(&format!("{PRELUDE}uint s = LaneSwizzle(position.x, 0x1fu, 0u, 0x4u);\n}}"));
    assert!(
        hlsl.contains("WaveReadLaneAt(position.x, (WaveGetLaneIndex() ^ 4))"),
        "butterfly must xor the lane, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("LaneSwizzle"),
        "intrinsic must not leak into HLSL, got:\n{hlsl}"
    );
}

#[test]
fn laneswizzle_prefix_step_is_shifted_readlane() {
    let hlsl = emit(&format!("{PRELUDE}uint s = LaneSwizzle(position.x, 0x18u, 0x03u, 0u);\n}}"));
    assert!(
        hlsl.contains("WaveReadLaneAt(position.x, (WaveGetLaneIndex() - 4))"),
        "prefix step must shift the lane, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("LaneSwizzle"),
        "intrinsic must not leak into HLSL, got:\n{hlsl}"
    );
}

#[test]
fn laneswizzle_unknown_shape_stays_loud() {
    let hlsl = emit(&format!("{PRELUDE}uint s = LaneSwizzle(position.x, 0x2au, 0u, 0u);\n}}"));
    assert!(
        hlsl.contains("LaneSwizzle"),
        "unobserved shape must stay loud, got:\n{hlsl}"
    );
}

#[test]
fn readlane_is_wave_readlaneat() {
    let hlsl = emit(&format!("{PRELUDE}uint s = ReadLane(position.x, 31u);\n}}"));
    assert!(
        hlsl.contains("WaveReadLaneAt(position.x, 31u)"),
        "ReadLane must map exactly, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("ReadLane("),
        "intrinsic must not leak into HLSL, got:\n{hlsl}"
    );
}

#[test]
fn crosslane_reductions_map_to_wave_active() {    let hlsl = emit(&format!(
        "{PRELUDE}uint a = CrossLaneAdd(position.x);\n\
         uint o = CrossLaneOr(position.x);\n\
         uint m = CrossLaneMin(position.x);\n\
         uint x = CrossLaneMax(position.x);\n}}"
    ));
    for want in [
        "WaveActiveSum(position.x)",
        "WaveActiveBitOr(position.x)",
        "WaveActiveMin(position.x)",
        "WaveActiveMax(position.x)",
    ] {
        assert!(hlsl.contains(want), "missing `{want}`, got:\n{hlsl}");
    }
    assert!(
        !hlsl.contains("CrossLane"),
        "intrinsics must not leak into HLSL, got:\n{hlsl}"
    );
}

#[test]
fn unpackbyte_extracts_lsb_first_bytes() {
    // GBuffer packing: Byte3/2/1 = albedo RGB, Byte0 = occlusion of the
    // same word (LSB-first numbering verified in Framework headers).
    // (Consumed through a sink call: unread locals drop in lowering,
    // and dead pure inits cascade.)
    let hlsl = emit(&format!(
        "void sink(uint x) {{}}\n{PRELUDE}uint r = UnpackByte3(position.x);\n\
         uint g = UnpackByte2(position.x);\n\
         uint b = UnpackByte1(position.x);\n\
         uint o = UnpackByte0(position.x);\n\
         sink(r + g + b + o);\n}}"
    ));
    for want in [
        "((position.x >> 24) & 0xffu)",
        "((position.x >> 16) & 0xffu)",
        "((position.x >> 8) & 0xffu)",
        "((position.x >> 0) & 0xffu)",
    ] {
        assert!(hlsl.contains(want), "missing `{want}`, got:\n{hlsl}");
    }
    assert!(
        !hlsl.contains("UnpackByte"),
        "intrinsics must not leak into HLSL, got:\n{hlsl}"
    );
}

#[test]
fn packuint2touint_packs_x_low_y_high() {
    // Inverse of the header's unpack_pixel_pos (x = packed & 0xFFFF,
    // y = packed >> 16) in resolve_lighting_c.h.
    let hlsl = emit(&format!(
        "void sink(uint x) {{}}\n{PRELUDE}uint p = PackUInt2ToUInt(position, 7u);\nsink(p);\n}}"
    ));
    assert!(
        hlsl.contains("((position & 0xFFFFu) | ((7u & 0xFFFFu) << 16))"),
        "pack layout must match unpack, got:\n{hlsl}"
    );
    assert!(!hlsl.contains("PackUInt2ToUInt"), "got:\n{hlsl}");
}

#[test]
fn countsetbits_maps_to_countbits() {
    let hlsl = emit(&format!(
        "void sink(uint x) {{}}\n{PRELUDE}uint r = CountSetBits(position.x);\nsink(r);\n}}"
    ));
    assert!(
        hlsl.contains("countbits(position.x)"),
        "CountSetBits must lower to countbits, got:\n{hlsl}"
    );
    assert!(!hlsl.contains("CountSetBits"), "got:\n{hlsl}");
}

#[test]
fn v_mbcnt_lo_hi_lower_to_shims() {
    // Raw GCN `__v_mbcnt_lo/hi_u32_b32(mask, prev)` (the wave64 lane-index
    // idiom): `_lo` reuses `__maskbitcnt_u32`, `_hi` gets its own shim.
    let hlsl = emit(&format!(
        "void sink(uint x) {{}}\n{PRELUDE}uint t = __v_mbcnt_hi_u32_b32(0xffffffffu, __v_mbcnt_lo_u32_b32(0xffffffffu, 0u));\nsink(t);\n}}"
    ));
    assert!(
        hlsl.contains("__maskbitcnt_hi_u32(0xffffffffu)"),
        "hi half must use its shim, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("__maskbitcnt_u32(0xffffffffu)"),
        "lo half must reuse the lo shim, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("uint __maskbitcnt_hi_u32(uint m)"),
        "hi shim must be emitted gated-on-use, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("__v_mbcnt_"),
        "raw intrinsics must not leak, got:\n{hlsl}"
    );
}

#[test]
fn wave64_cndmask_lowers_to_ternary() {
    // GCN `wave64::__v_cndmask_b32(S0, S1, VCC)` is per-lane
    // `VCC ? S1 : S0` (predicate idiom in culling headers).
    let hlsl = emit(&format!(
        "void sink(uint x) {{}}\n{PRELUDE}uint p = wave64::__v_cndmask_b32(0u, 1u, position);\nsink(p);\n}}"
    ));
    assert!(
        hlsl.contains("((position) ? (1u) : (0u))") || hlsl.contains("(position ? 1u : 0u)"),
        "cndmask must become a ternary, got:\n{hlsl}"
    );
    assert!(!hlsl.contains("__v_cndmask_b32"), "got:\n{hlsl}");
}

#[test]
fn wave32_cndmask_lowers_to_ternary() {
    // The `wave32::` prefix spells the same per-lane instruction
    // (predicate idiom in multi-wave culling headers).
    let hlsl = emit(&format!(
        "void sink(uint x) {{}}\n{PRELUDE}uint p = wave32::__v_cndmask_b32(0u, 1u, position);\nsink(p);\n}}"
    ));
    assert!(
        hlsl.contains("((position) ? (1u) : (0u))") || hlsl.contains("(position ? 1u : 0u)"),
        "cndmask must become a ternary, got:\n{hlsl}"
    );
    assert!(!hlsl.contains("__v_cndmask_b32"), "got:\n{hlsl}");
}

#[test]
fn invariant_annotation_desugars_to_operand() {
    // `__invariant(x)` is a pure invariance annotation (the compiler must
    // not reassociate the value): value-identity, so it desugars to the
    // operand. HLSL has no such annotation (culling determinant).
    let hlsl = emit(&format!(
        "void sink(float x) {{}}\n{PRELUDE}float det = __invariant(position * 2u);\nsink(det);\n}}"
    ));
    assert!(
        !hlsl.contains("__invariant"),
        "annotation must not leak into HLSL, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("sink("),
        "operand value must survive, got:\n{hlsl}"
    );
}

#[test]
fn s_bfm_b64_lowers_to_gated_shim() {
    // GCN `S_BFM_B64(width, offset)` (contiguous bitfield mask, wave
    // predicate idiom) lowers to the gated `__s_bfm_b64` shim, 32-bit
    // saturating like the toolchain's `ulong` model.
    let hlsl = emit(&format!(
        "void sink(uint x) {{}}\n{PRELUDE}uint m = __s_bfm_b64(16u, 0u);\nsink(m);\n}}"
    ));
    assert!(
        hlsl.contains("__s_bfm_b64(16u, 0u)"),
        "must call the shim, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("uint __s_bfm_b64(uint w, uint o)"),
        "shim must be emitted gated-on-use, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("S_BFM") && !hlsl.contains("s_bfm_b64(t"),
        "raw shape must not leak, got:\n{hlsl}"
    );
}

#[test]
fn vivj_linear_centroid_threads_sv_position() {
    // `GetViVjLinearCentroid()` is a PS5 compiler builtin (current
    // pixel's linear centroid, float2; defined nowhere in the Framework) with
    // no direct HLSL spelling: SV_Position threads through (a fresh
    // entry param here — this entry has no position input), and `auto`
    // inits from it deduce float2 (basic_draw_p, vis_draw_p).
    let hlsl = emit(
        "struct Pout { float4 c : S_TARGET_OUTPUT0; };\n\
         float wireframeScale(float2 pos) { auto ij = pos; auto bc = ij.x + ij.y; return bc; }\n\
         Pout main(float2 uv : TEXCOORD0) {\n\
         \tauto ij = GetViVjLinearCentroid();\n\
         \tPout o; o.c = float4(wireframeScale(ij), 0, 0, 1); return o;\n\
         }",
    );
    assert!(
        hlsl.contains("SV_Position"),
        "entry must gain an SV_Position source, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("float2 ij = "),
        "centroid auto must deduce float2, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("GetViVjLinearCentroid") && !hlsl.contains("auto "),
        "builtin and auto must not leak, got:\n{hlsl}"
    );
}

#[test]
fn vivj_persp_siblings_thread_like_centroid() {
    // `GetViVjPerspCenter()` / `GetViVjPerspCentroid()` are the same
    // builtin family (staticmesh deferred pixel shaders): they thread
    // identically and deduce float2.
    let hlsl = emit(
        "struct Pin { float4 pos : S_POSITION; };\n\
         struct Pout { float4 c : S_TARGET_OUTPUT0; };\n\
         Pout main(Pin input) {\n\
         \tfloat4 ijk = float4(GetViVjPerspCentroid(), 0.f, 1.f);\n\
         \tfloat c = GetViVjPerspCenter().x;\n\
         \tPout o; o.c = ijk + c; return o;\n\
         }",
    );
    assert!(
        hlsl.contains("input.pos.xy"),
        "persp calls must thread to the position, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("GetViVjPerspCentroid(") && !hlsl.contains("GetViVjPerspCenter("),
        "persp builtins must not leak, got:\n{hlsl}"
    );
}

#[test]
fn vivj_linear_centroid_uses_struct_position() {
    // When the entry input struct already carries SV_Position
    // (`float4 pos : S_POSITION`), no fresh param is added: the call
    // sites use `input.pos.xy` (basic_draw_p, vis_draw_p shape).
    let hlsl = emit(
        "struct Pin { float4 pos : S_POSITION; };\n\
         struct Pout { float4 c : S_TARGET_OUTPUT0; };\n\
         float wireframeScale() { auto ij = GetViVjLinearCentroid(); return ij.x + ij.y; }\n\
         Pout main(Pin input) {\n\
         \tPout o; o.c = float4(wireframeScale(), 0, 0, 1); return o;\n\
         }",
    );
    assert!(
        hlsl.contains("wireframeScale(input.pos.xy)"),
        "call must take the struct position, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("float wireframeScale(float2 __pos_"),
        "helper must gain a float2 param, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("GetViVjLinearCentroid") && !hlsl.contains("auto "),
        "builtin and auto must not leak, got:\n{hlsl}"
    );
}

#[test]
fn wave_inverse_lowers_to_adjugate_shim() {
    // PS5 `wave::inverse(m)` (system namespace, no in-tree definition)
    // is a true matrix inverse; HLSL has no intrinsic, so it lowers to
    // the gated `__wave_inverse` adjugate shim (3x3/4x4 overloads), and
    // `auto` inits from it deduce the matrix type (basic_draw_p).
    let hlsl = emit(
        "struct SRT { float4x4 v; };\n\
         SRT *srt : S_SRT_DATA;\n\
         float4 main(float3 dir : TEXCOORD0) : S_TARGET_OUTPUT {\n\
         \tauto inv = wave::inverse(srt->v);\n\
         \treturn float4(mul(inv, float4(dir, 0.0f)), 1.0f);\n\
         }",
    );
    assert!(
        hlsl.contains("float4x4 inv = __wave_inverse("),
        "inverse auto must deduce float4x4, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("float4x4 __wave_inverse(float4x4 m)"),
        "4x4 adjugate shim must be emitted, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("wave::inverse"),
        "system namespace must not leak, got:\n{hlsl}"
    );
}

#[test]
fn v_lerp_u8_ones_maps_to_byte_average() {
    // GCN `V_LERP_U8(a, b, 1)` is the unsigned byte-wise mean
    // (per-byte `(a + b + 1) >> 1`, exact via carry-free packed
    // arithmetic), used as `avg2` for mip downsampling (upscale).
    let hlsl = emit(
        "RWStructuredBuffer<uint> outBuf;\n\
         [NUM_THREADS(64,1,1)]\n\
         void main(uint position : S_DISPATCH_THREAD_ID) {\n\
         \tuint m = __v_lerp_u8(100u, 200u, 1);\n\
         \toutBuf[0] = m;\n\
         }",
    );
    assert!(
        !hlsl.contains("__v_lerp_u8"),
        "intrinsic must not leak into HLSL, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("0x7f7f7f7f") && hlsl.contains("0x01010101"),
        "packed byte average must emit, got:\n{hlsl}"
    );
}

#[test]
fn v_perm_b32_emits_gated_shim() {
    // GCN `V_PERM_B32(a, b, sel)` (byte permute) has no HLSL spelling:
    // calls pass through to a gated `__v_perm_b32` shim (upscale).
    let hlsl = emit(&format!(
        "{PRELUDE}uint r = __v_perm_b32(1u, 2u, 0x05010400u);\nposition = r;\n}}"
    ));
    assert!(
        hlsl.contains("uint __v_perm_b32(uint a, uint b, uint sel)"),
        "perm shim must be emitted, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("__v_perm_b32(1u, 2u, 0x05010400u)"),
        "call must pass through, got:\n{hlsl}"
    );
}

#[test]
fn v_pk_f16_maps_to_vector_ops() {
    // GCN packed-fp16 ops on half2 values are HLSL `min`/`max`/`+`/`*`
    // exactly (bc7Encode).
    let hlsl = emit(
        "RWStructuredBuffer<float> outBuf;\n\
         [NUM_THREADS(1,1,1)]\n\
         void main(uint i : S_DISPATCH_THREAD_ID) {\n\
         \thalf2 a = half2(1, 2);\n\
         \thalf2 b = half2(3, 4);\n\
         \toutBuf[0] = __v_pk_min_f16(a, b).x + __v_pk_max_f16(a, b).x + __v_pk_add_f16(a, b).x + __v_pk_mul_f16(a, b).x;\n\
         }",
    );
    assert!(
        !hlsl.contains("__v_pk_"),
        "packed intrinsics must not leak, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("min(a, b)") && hlsl.contains("max(a, b)"),
        "min/max must map, got:\n{hlsl}"
    );
}

#[test]
fn wave_mode_intrinsics_map() {
    // `__get_wavemode()` → `WaveGetLaneCount()`; `__kWaveModeWave64/32`
    // marker constants resolve to 64/32 so the comparison is valid HLSL.
    let hlsl = emit(&format!(
        "void sink(uint x) {{}}\n{PRELUDE}uint w = (__get_wavemode() == __kWaveModeWave64) ? 64 : 32;\nsink(w);\n}}"
    ));
    assert!(
        hlsl.contains("WaveGetLaneCount() == 64"),
        "wave mode must map to WaveGetLaneCount()==64, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("__get_wavemode") && !hlsl.contains("__kWaveModeWave64"),
        "AMD intrinsics must not leak, got:\n{hlsl}"
    );
}
