use ps5_shader::parse_shader;

// Regression test for the DXC `use of undeclared identifier 'sce'` cluster:
// the Sony AGC framework sampler object (`sce::Agc::Core::Sampler` in PSSL
// struct fields / function parameters) must be emitted as HLSL
// `SamplerState`, never passed through verbatim with its `sce::` scope
// (DXC has no `sce` namespace).

fn emit(source: &str) -> String {
    let module = parse_shader(source).expect("parser accepted shader");
    let model = ps5_shader::semantic::analyze(&module).expect("semantic analysis succeeded");
    let ir = ps5_shader::lower::lower_module(&module, &model).expect("lowering succeeded");
    ps5_shader::emit::emit_vulkan_hlsl(&ir)
}

#[test]
fn sce_agc_core_sampler_param_emits_sampler_state() {
    let hlsl = emit(
        "struct UserData { sce::Agc::Core::Sampler m_sampler; };\n\
         half f(UserData u) { return 0; }",
    );
    assert!(
        hlsl.contains("SamplerState m_sampler"),
        "sampler field must emit as SamplerState, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("sce::"),
        "no sce::-qualified names may leak into HLSL, got:\n{hlsl}"
    );
}

// PSSL identifiers that are HLSL reserved modifiers (`uint snorm` misparses:
// DXC `modifiers must appear before type`) are escaped with a trailing
// underscore, consistently at declarations and uses.
#[test]
fn hlsl_reserved_param_name_is_escaped() {
    let hlsl = emit(
        "float decodeSnorm(uint snorm, int numBits) {\n\
         uint mask = ((1 << (numBits - 1)) - 1);\n\
         return float(snorm & mask); }",
    );
    assert!(
        hlsl.contains("uint snorm_"),
        "reserved param must be escaped, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("snorm_ & mask"),
        "escaped param uses must agree, got:\n{hlsl}"
    );
}

#[test]
fn depth_compare_chain_selects_comparison_sampler() {
    // A standalone `sampler.init()...setDepthCompareFunction(...)` chain
    // configures a COMPARISON sampler: the local must declare
    // `SamplerComparisonState` (`SampleCmp*` requires it), exactly like
    // builder chains inside initializers already do.
    let hlsl = emit(
        "float f(Texture2D<float> tex, float2 uv) {\n\
         sce::Agc::Core::Sampler sampler;\n\
         sampler.init().setDepthCompareFunction(sce::Agc::Core::Sampler::DepthCompare::kLessEqual);\n\
         return tex.SampleCmpLevelZero(sampler, uv, 0.5);\n}",
    );
    assert!(
        hlsl.contains("SamplerComparisonState sampler_"),
        "chain must select comparison type, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("SampleCmpLevelZero(sampler_, "),
        "uses must agree, got:\n{hlsl}"
    );
}

#[test]
fn hlsl_reserved_texture_name_is_escaped() {
    // `Texture2D<float4> texture : register(t0)` — `texture` is an HLSL
    // keyword (legacy type), so the declaration and every use become
    // `texture_`, mirroring the `sampler` -> `sampler_` rename.
    let hlsl = emit(
        "Texture2D<float4> texture : register(t0);\n\
         SamplerState smp;\n\
         [NUM_THREADS(1,1,1)]\n\
         void main(uint i : S_DISPATCH_THREAD_ID) {\n\
         float3 c = texture.Sample(smp, float2(0.5, 0.5)).rgb;\n}",
    );
    assert!(
        hlsl.contains("texture_.Sample("),
        "texture uses must agree on the escape, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains(" texture.Sample(") && !hlsl.contains("\ttexture.Sample("),
        "bare keyword use must not leak, got:\n{hlsl}"
    );
}

// A buffer of textures (`RegularBuffer<Texture2D>`) must emit as a
// resource array (`Texture2D textures[]`): `StructuredBuffer<Texture2D>`
// nests resources illegally (DXC fatal invalid-opaque-type follow-on to
// the `expected expression` cluster).
#[test]
fn buffer_of_textures_emits_resource_array() {
    let hlsl = emit(
        "RegularBuffer<Texture2D> textures : register(t0);\n\
         SamplerState smp;\n\
         float4 f(float2 uv) { return textures[0].Sample(smp, uv); }",
    );
    assert!(
        hlsl.contains("Texture2D textures[]"),
        "texture buffer must emit as resource array, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("StructuredBuffer<Texture2D>"),
        "nested resources are illegal HLSL, got:\n{hlsl}"
    );
}

// Method calls on indexed bindless resources (`textures[0].Sample(...)`)
// must keep the index: degrading to `<complex callee>` is never valid HLSL
// (DXC `expected expression` cluster).
#[test]
fn indexed_resource_sample_keeps_index() {
    let hlsl = emit(
        "struct V2P { float4 pos : S_POSITION; };\n\
         RegularBuffer<Texture2D> textures : register(t0);\n\
         SamplerState smp;\n\
         float4 f(V2P input) { return textures[0].Sample(smp, input.pos.xy); }",
    );
    assert!(
        hlsl.contains("textures[0].Sample"),
        "indexed sample must survive lowering, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("<complex callee>"),
        "no unlowered callee placeholders may leak, got:\n{hlsl}"
    );
}

// Vulkan sampler configuration mutators (`sampler.init()`,
// `sampler.setMipFilterMode(...)`) have no runtime counterpart and must be
// elided, not left as `<complex callee>()` statements.
#[test]
fn sampler_config_calls_are_elided() {
    let hlsl = emit(
        "struct V2P { float4 pos : S_POSITION; };\n\
         [CxxSymbol(\"Shader::ps\")]\n\
         float4 main(V2P In) : S_TARGET_OUTPUT0 {\n\
         sce::Agc::Core::Sampler sampler;\n\
         sampler.init();\n\
         sampler.setMipFilterMode(sce::Agc::Core::Sampler::MipFilterMode::kLinear);\n\
         sampler.setCubeWrap(sce::Agc::Core::Sampler::CubeWrap::kEnable);\n\
         return 0.0; }",
    );
    assert!(
        !hlsl.contains("<complex callee>"),
        "sampler config calls must be elided, got:\n{hlsl}"
    );
}

// PSSL `GatherLODRed(s, uv, lod)` (explicit-LOD gather) has no HLSL/SPIR-V
// equivalent: lower to `GatherRed(s, uv)` and drop the LOD argument.
#[test]
fn gather_lod_lowers_to_gather_without_lod() {
    let hlsl = emit(
        "struct V2P { float4 pos : S_POSITION; };\n\
         Texture2D<float> depth;\n\
         SamplerState s;\n\
         float4 main(V2P In) : S_TARGET_OUTPUT0 {\n\
         return depth.GatherLODRed(s, In.pos.xy, 0.0).xxxx; }",
    );
    assert!(
        hlsl.contains("depth.GatherRed(s, In.pos.xy)"),
        "GatherLODRed must emit as GatherRed without lod, got:\n{hlsl}"
    );
}

// Resource-kind vocabulary must use HLSL casing (`Texture2DArray`, not the
// lowercased IR kind `texture2darray`): DXC has no `texture2darray` template.
// Texture-array struct members hoist to same-typed globals, preserving the
// casing end to end.
#[test]
fn texture_array_kinds_emit_hlsl_casing() {
    let hlsl = emit(
        "struct T { Texture2DArray<float> a; };\n\
         T g;\n\
         SamplerState smp;\n\
         float f(float2 uv) { return g.a.Sample(smp, float3(uv, 0)).x; }",
    );
    assert!(
        hlsl.contains("Texture2DArray<float> __hoist_T_a;"),
        "hoisted texture array must keep HLSL casing, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("texture2darray"),
        "lowercased kind leaked into HLSL, got:\n{hlsl}"
    );
}

// PSSL wave/bit intrinsics lower to their HLSL equivalents instead of
// leaking as undeclared identifiers.
#[test]
fn pssl_intrinsics_map_to_hlsl() {
    let hlsl = emit(
        "uint f(uint idx, uint check) {\n\
         uint b = BitFieldExtract(idx, 0, 1);\n\
         uint f = ReadFirstLane(check);\n\
         uint4 m = ballot(idx == check);\n\
         return b + f + m.x;\n\
         }",
    );
    assert!(
        hlsl.contains("((idx >> 0) & ((1u << 1) - 1u))"),
        "BitFieldExtract must lower to shift-and-mask, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("WaveReadLaneFirst(check)"),
        "ReadFirstLane must map to WaveReadLaneFirst, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("WaveActiveBallot("),
        "ballot must map to WaveActiveBallot, got:\n{hlsl}"
    );
}

// A struct field whose type is an enum (Named type that isn't a declared
// struct) must emit as `int`, not the raw enum name (DXC: unknown type
// name 'DrawMode').
#[test]
fn enum_typed_struct_field_emits_as_int() {
    let hlsl = emit(
        "struct DrawSrt { float zoom; DrawMode drawMode; };\n\
         DrawSrt srt;\n\
         float f() { return srt.zoom; }",
    );
    assert!(
        hlsl.contains("int drawMode;"),
        "enum field must emit as int, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("DrawMode drawMode"),
        "raw enum name must not leak, got:\n{hlsl}"
    );
}

// A generic `template<T>` function with a `Texture2D<T>` param and `T` in
// its body/locals must emit float-family overloads with `T` substituted
// throughout (DXC: `unknown type name 'T'`).
#[test]
fn generic_t_function_emits_float_overloads_with_body_substitution() {
    let hlsl = emit(
        "template <typename T>\n\
         T downscale(T v, uint s) {\n\
         T sum = v;\n\
         return T(1.f) + sum;\n\
         }\n\
         float4 f() { return downscale(float4(1,2,3,4), 1u); }",
    );
    assert!(
        hlsl.contains("float4 downscale(float4 v, uint s)"),
        "must emit a float4 overload, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("T downscale"),
        "raw generic T must not leak into a declaration, got:\n{hlsl}"
    );
}

// Array-typed function parameters must keep their size (`float3 L[4]`,
// not `float3 L`): DXC rejects `vector element index '3' is out of bounds`
// when an array param is emitted as a scalar vector.
#[test]
fn array_typed_parameter_keeps_its_size() {
    let hlsl = emit(
        "float f(float3 points[4], uint i) {\n\
         return points[3].x + points[i].y;\n\
         }",
    );
    assert!(
        hlsl.contains("float f(float3 points[4], uint i)"),
        "array param must keep its size, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("float f(float3 points, uint i)"),
        "array param must not be flattened to scalar, got:\n{hlsl}"
    );
}

// Array-typed function parameters (`float3 L[4]`) must keep their size in
// the emitted signature, not collapse to a bare scalar (DXC: `vector
// element index '3' is out of bounds`).
#[test]
fn array_parameters_keep_their_size() {
    let hlsl = emit(
        "float f(float3 L[4], float3 points[4], bool twoSided) {\n\
         return L[3].x + points[0].y; }",
    );
    assert!(
        hlsl.contains("float f(float3 L[4], float3 points[4], bool twoSided)"),
        "array params must keep [N], got:\n{hlsl}"
    );
}

// Array-typed function parameters keep their size (`float3 L[4]`), so
// indexing them as arrays works (DXC: `vector element index '3' is out of
// bounds`).
#[test]
fn array_function_parameters_keep_size() {
    let hlsl = emit(
        "float f(float3 L[4], float3 points[4]) {\n\
         return dot(L[3], points[3]);\n\
         }",
    );
    assert!(
        hlsl.contains("float f(float3 L[4], float3 points[4])"),
        "array param must keep its size, got:\n{hlsl}"
    );
}

// Framework sampler construction has no HLSL runtime meaning (Vulkan
// samplers are immutable and externally bound): a builder chain, `= {}`
// or sampler-to-sampler copy folds to a bare `SamplerState` declaration
// instead of degrading to `<complex callee>`.
#[test]
fn sampler_builder_chain_folds_to_bare_declaration() {
    // (Locals are consumed through a sink: unread locals are dead code
    // and drop in lowering.)
    let hlsl = emit(
        "void sink(SamplerState s) {}\n\
         [NUM_THREADS(1,1,1)]\n\
         void main(uint2 position : S_DISPATCH_THREAD_ID) {\n\
         const sce::Agc::Core::Sampler samplerSpec = sce::Agc::Core::Sampler()\n\
             .init()\n\
             .setXyFilterMode(sce::Agc::Core::Sampler::FilterMode::kBilinear)\n\
             .setWrapMode(sce::Agc::Core::Sampler::WrapMode::kWrap);\n\
         const SamplerState cubemapSampler = SamplerState(samplerSpec);\n\
         sink(samplerSpec);\n\
         sink(cubemapSampler);\n\
         }",
    );
    assert!(
        hlsl.contains("SamplerState samplerSpec;"),
        "builder chain must fold to bare decl, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("SamplerState cubemapSampler;"),
        "sampler copy must fold to bare decl, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("complex callee"),
        "no complex-callee fallback, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("sce::"),
        "no sce::-qualified names may leak into HLSL, got:\n{hlsl}"
    );
}

#[test]
fn sampler_brace_init_folds_to_bare_declaration() {
    let hlsl = emit("void sink(SamplerState s) {}\n[NUM_THREADS(1,1,1)]\nvoid main(uint2 position : S_DISPATCH_THREAD_ID) { sce::Agc::Core::Sampler s = {}; sink(s); }");
    assert!(
        hlsl.contains("SamplerState s;"),
        "= {{}} must fold to bare decl, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("sce::"),
        "no sce::-qualified names may leak into HLSL, got:\n{hlsl}"
    );
}

#[test]
fn ordinary_construction_inits_are_kept() {
    // The builder-init fold is gated on sampler types: construction of an
    // ordinary struct must keep its initializer untouched. (`b` is
    // consumed: unread locals drop in lowering.)
    let hlsl = emit("struct B { float x; }; float sinkB(B b) { return b.x; } [NUM_THREADS(1,1,1)]\nvoid main(uint2 position : S_DISPATCH_THREAD_ID) { B b = B(1.0); float y = sinkB(b); }");
    assert!(
        hlsl.contains("B b = B(1.0);"),
        "ordinary init must be kept, got:\n{hlsl}"
    );
}
