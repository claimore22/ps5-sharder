use ps5_shader::parse_shader;

// Regression tests for resource method spellings (DXC `expected
// expression` via `<complex callee>`): PSSL-only method names lower to
// their exact HLSL equivalents. Unlisted methods keep failing loudly.

fn emit(source: &str) -> String {
    let module = parse_shader(source).expect("parser accepted shader");
    let model = ps5_shader::semantic::analyze(&module).expect("semantic analysis succeeded");
    let ir = ps5_shader::lower::lower_module(&module, &model).expect("lowering succeeded");
    ps5_shader::emit::emit_vulkan_hlsl(&ir)
}

const COMPUTE_PRELUDE: &str = "[NUM_THREADS(1,1,1)]\nvoid main(uint position : S_DISPATCH_THREAD_ID) {\n";

#[test]
fn getlod_maps_to_calculate_level_of_detail() {
    // `tex.GetLOD(sampler, uv)` / `GetLODUnclamped` are HLSL
    // `CalculateLevelOfDetail[Unclamped]` exactly (texture streaming).
    let hlsl = emit(
        "Texture2D<float4> tex;\n\
         SamplerState smp;\n\
         float4 main(float2 uv : TEXCOORD0) : S_TARGET_OUTPUT {\n\
         \tfloat lod = tex.GetLOD(smp, uv);\n\
         \tfloat lodu = tex.GetLODUnclamped(smp, uv);\n\
         \treturn tex.SampleLevel(smp, uv, lod + lodu);\n\
         }",
    );
    assert!(
        hlsl.contains(".CalculateLevelOfDetail(smp, uv)")
            && hlsl.contains(".CalculateLevelOfDetailUnclamped(smp, uv)"),
        "LOD queries must map, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("GetLOD"),
        "PSSL spelling must not leak, got:\n{hlsl}"
    );
}

#[test]
fn assigned_cbuffer_member_demotes_to_static() {
    // Cbuffer members assigned in-shader (clear shaders copying
    // SRT -> cbuffer scratch) cannot stay const-qualified: they emit
    // as `static` globals while untouched members keep the cbuffer.
    let hlsl = emit(
        "cbuffer ClearColor : register(b0) { float4 m_clearColor; float4 m_other; };\n\
         float4 main(float4 x : TEXCOORD0) : S_TARGET_OUTPUT {\n\
         \tm_clearColor = x;\n\
         \treturn m_clearColor + m_other;\n\
         }",
    );
    assert!(
        hlsl.contains("static float4 m_clearColor;"),
        "assigned member must demote, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("m_other;") && !hlsl.contains("static float4 m_other;"),
        "untouched member must keep the cbuffer, got:\n{hlsl}"
    );
}

#[test]
fn uint_sample_desugars_to_load() {
    // Integer texels cannot filter: `uint v = tex.Sample(smp, uv)` on a
    // `Texture2D<uint>` becomes a GetDimensions fetch plus a mip-0 Load
    // (agc_metadata uint_copy).
    let hlsl = emit(
        "Texture2D<uint> tex : register(t0);\n\
         SamplerState smp;\n\
         [NUM_THREADS(8,8,1)]\n\
         void main(uint2 tid : S_DISPATCH_THREAD_ID) {\n\
         \tuint v = tex.Sample(smp, float2(tid) / 64.0f);\n\
         }",
    );
    assert!(
        hlsl.contains(".Load(int3("),
        "uint sample must become Load, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("GetDimensions("),
        "dims fetch must precede, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains(".Sample("),
        "Sample on uint must not leak, got:\n{hlsl}"
    );
}

#[test]
fn samplecmp_passes_through() {
    let hlsl = emit(&format!(
        "{COMPUTE_PRELUDE}Texture2D<float> tex;\n\
         SamplerComparisonState smp;\n\
         float z = tex.SampleCmp(smp, float2(0.5, 0.5), 0.25);\n}}"
    ));
    assert!(
        hlsl.contains(".SampleCmp("),
        "SampleCmp must survive, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("complex callee"),
        "no complex-callee fallback, got:\n{hlsl}"
    );
}

#[test]
fn samplecmplod0_becomes_level_zero() {
    let hlsl = emit(&format!(
        "{COMPUTE_PRELUDE}Texture2D<float> tex;\n\
         SamplerComparisonState smp;\n\
         float z = tex.SampleCmpLOD0(smp, float2(0.5, 0.5), 0.25);\n}}"
    ));
    assert!(
        hlsl.contains(".SampleCmpLevelZero("),
        "SampleCmpLOD0 must map, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("complex callee"),
        "no complex-callee fallback, got:\n{hlsl}"
    );
}

#[test]
fn bytebuffer_store4_and_load4_pass_through() {
    let hlsl = emit(&format!(
        "{COMPUTE_PRELUDE}RW_ByteBuffer buf;\n\
         buf.Store4(0, uint4(1, 2, 3, 4));\n\
         uint x = buf.Load4(0).x;\n}}"
    ));
    assert!(
        hlsl.contains(".Store4("),
        "Store4 must survive, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains(".Load4("),
        "Load4 must survive, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("complex callee"),
        "no complex-callee fallback, got:\n{hlsl}"
    );
}

#[test]
fn mipmaps_mip0_read_becomes_load() {
    let hlsl = emit(&format!(
        "{COMPUTE_PRELUDE}Texture2D<float> tex;\n\
         float d = tex.MipMaps(0, int2(3, 4));\n}}"
    ));
    assert!(
        hlsl.contains(".Load(int2(3, 4))"),
        "mip-0 read must become Load, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("complex callee"),
        "no complex-callee fallback, got:\n{hlsl}"
    );
}

#[test]
fn bytebuffer_param_emits_single_definition() {
    // Bare `ByteBuffer` is untyped: its element must not read as generic
    // `T`, or the emitter stamps 8 identical overloads (DXC `redefinition
    // of 'findBatch'` across the blendshape-decoder family).
    let hlsl = emit(
        "bool hasData(ByteBuffer buf, uint i) { return buf.Load(i) != 0; }\n\
         [NUM_THREADS(1,1,1)]\n\
         void main(uint tid : S_DISPATCH_THREAD_ID) {\n\
         ByteBuffer buf;\n\
         bool b = hasData(buf, tid);\n}",
    );
    assert_eq!(
        hlsl.matches("bool hasData(").count(),
        1,
        "exactly one definition, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("ByteAddressBuffer buf"),
        "untyped buffer emission, got:\n{hlsl}"
    );
}

#[test]
fn rwmstexture2d_emits_rwtexture2darray() {
    // Read-write multisampled textures have no Vulkan mapping, so
    // `rwmstexture2d<float>` becomes `RWTexture2DArray<float>` (explicit
    // per-sample subscripts are value-identical either way).
    let hlsl = emit(&format!(
        "rwmstexture2d<float> ao : register(u0);\n\
         {COMPUTE_PRELUDE}ao[uint3(0u, 0u, 1u)] = 0.5;\n}}"
    ));
    assert!(
        hlsl.contains("RWTexture2DArray<float>"),
        "RW MS kind must map to array, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("ao[uint3(0u, 0u, 1u)]"),
        "per-sample subscript carries over, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("rwmstexture2d"),
        "lowercase spelling must not leak, got:\n{hlsl}"
    );
}

#[test]
fn texture1d_member_emits_texture1d() {
    // `Texture1D<float>` is a resource kind like the other dimensions:
    // members must not collapse to int.
    let hlsl = emit(&format!(
        "void sink(float x) {{}}\n\
         struct T {{ Texture1D<float> lut; }};\n\
         T t;\n\
         {COMPUTE_PRELUDE}float x = t.lut[0];\nsink(x);\n}}"
    ));
    assert!(
        hlsl.contains("Texture1D<float> __hoist_T_lut;"),
        "1D texture member must hoist with its type, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("__hoist_T_lut[0]"),
        "uses must follow the hoisted binding, got:\n{hlsl}"
    );
    assert!(!hlsl.contains("int lut"), "got:\n{hlsl}");
}

#[test]
fn srt_mirror_alias_propagates_to_param() {
    // `s_srt = srt;` as the entry's first statement (same struct type)
    // aliases the mirror to the parameter: uses rewrite, the store drops,
    // and the mirror global is removed (no duplicate binding set).
    let hlsl = emit(
        "struct SRT { uint x; Texture2D<float> tex; };\n\
         SRT s_srt;\n\
         SamplerState smp;\n\
         [NUM_THREADS(1,1,1)]\n\
         void main(SRT srt, uint i : S_DISPATCH_THREAD_ID) {\n\
         s_srt = srt;\n\
         float v = s_srt.x + srt.x;\n\
         float4 c = s_srt.tex.Sample(smp, float2(v, v));\n}",
    );
    assert!(
        !hlsl.contains("s_srt"),
        "mirror must be fully eliminated, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("srt.x"),
        "uses must redirect to the parameter, got:\n{hlsl}"
    );
}

#[test]
fn unknown_resource_methods_pass_through() {
    // Unknown member calls pass through verbatim (`obj.method()`) so
    // user-defined struct methods resolve natively; genuinely missing
    // methods fail at DXC instead — still loud, never silent.
    let hlsl = emit(&format!(
        "{COMPUTE_PRELUDE}Texture2D<float> tex;\n\
         float d = tex.SomeUnknownMethod(0);\n}}"
    ));
    assert!(
        hlsl.contains("tex.SomeUnknownMethod(0)"),
        "unknown methods must pass through, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("complex callee"),
        "no complex-callee fallback, got:\n{hlsl}"
    );
}

#[test]
fn dotted_index_in_method_object_flattens() {
    // `bufs[a.b].Sample(...)` (asset-streaming indirect textures):
    // dotted member paths spell identically in HLSL, so they flatten
    // into the callee like literals and variables do (only expressions
    // needing real lowering stay loud).
    let hlsl = emit(&format!(
        "{COMPUTE_PRELUDE}StructuredBuffer<Texture2D<float4>> bufs;\n\
         SamplerState smp;\n\
         uint2 idx;\n\
         float4 c = bufs[idx.x].Sample(smp, float2(0.5, 0.5));\n\
         c += bufs[0].Sample(smp, float2(0.5, 0.5));\n}}"
    ));
    assert!(
        hlsl.contains("bufs[idx.x].Sample(smp,"),
        "dotted index must flatten, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("bufs[0].Sample(smp,"),
        "literal index must keep working, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("complex callee"),
        "no complex-callee fallback, got:\n{hlsl}"
    );
}
