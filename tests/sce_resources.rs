use ps5_shader::parse_shader;

// Regression tests for resource-like `sce::` framework objects:
// `sce::Agc::Core::Texture` (a 2D texture, e.g. `FsrTexLuts` members),
// `sce::Psr::TopLevelBvhDescriptor` (a ray-tracing acceleration
// structure, e.g. GBUFFER `m_bvh`), bare `Sampler` builder chains,
// mutated globals, and tutorial sampler factories. All are GPU-side
// objects or host-fed state with a defined HLSL lowering — never
// cbuffer scalars or `sce::`-leaking names.

fn emit(source: &str) -> String {
    let module = parse_shader(source).expect("parser accepted shader");
    let model = ps5_shader::semantic::analyze(&module).expect("semantic analysis succeeded");
    let ir = ps5_shader::lower::lower_module(&module, &model).expect("lowering succeeded");
    ps5_shader::emit::emit_vulkan_hlsl(&ir)
}

#[test]
fn framework_texture_member_emits_texture2d() {
    // Used members hoist to top-level bindings (unused ones strip,
    // exactly like ordinary resource members).
    let hlsl = emit(
        "struct Luts { sce::Agc::Core::Texture horizontal; sce::Agc::Core::Texture vertical; };\n\
         Luts l;\n\
         SamplerState smp;\n\
         float4 f(float2 uv) { return l.horizontal.Sample(smp, uv); }\n\
         [NUM_THREADS(1,1,1)]\n\
         void main(uint i : S_DISPATCH_THREAD_ID) {\n\
         float4 c = f(float2(0.5, 0.5));\n\
         }",
    );
    assert!(
        hlsl.contains("Texture2D<float4> __hoist_Luts_horizontal;"),
        "framework texture must hoist to Texture2D<float4>, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("sce::"),
        "no framework residue may leak, got:\n{hlsl}"
    );
}

#[test]
fn framework_bvh_member_emits_acceleration_structure() {
    let hlsl = emit(
        "struct Lighting { sce::Psr::TopLevelBvhDescriptor m_bvh; uint m_flags; };\n\
         Lighting l;\n\
         void take(RaytracingAccelerationStructure a) { }\n\
         [NUM_THREADS(1,1,1)]\n\
         void main(uint i : S_DISPATCH_THREAD_ID) {\n\
         take(l.m_bvh);\n\
         }",
    );
    assert!(
        hlsl.contains("RaytracingAccelerationStructure l_m_bvh;"),
        "BVH descriptor must flatten to a RaytracingAccelerationStructure binding, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("sce::"),
        "no framework residue may leak, got:\n{hlsl}"
    );
}

#[test]
fn framework_texture_root_flattens_to_binding() {
    // A mixed root (framework texture + scalar) must flatten like any
    // other resource: top-level texture binding plus scalar cbuffer —
    // never a texture-typed cbuffer field.
    let hlsl = emit(
        "struct Srt { sce::Agc::Core::Texture tex; float k; };\n\
         Srt srt : S_SRT_DATA;\n\
         SamplerState smp;\n\
         float4 f(float2 uv) { return srt.tex.Sample(smp, uv) * srt.k; }",
    );
    assert!(
        hlsl.contains("Texture2D<float4> srt_tex;"),
        "framework texture must hoist to a top-level binding, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("cbuffer srt_constants"),
        "scalars must collect in a cbuffer, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("sce::"),
        "no framework residue may leak, got:\n{hlsl}"
    );
}

#[test]
fn bare_sampler_builder_chain_elides() {
    // `using namespace sce::Agc::Core` idiom: bare `Sampler()` fluent
    // chains configure an externally-bound sampler and elide to a bare
    // declaration (same as the qualified chains). (Consumed through a
    // sink: unread locals drop in lowering.)
    let hlsl = emit(
        "void sink(SamplerState s) {}\n\
         [NUM_THREADS(1,1,1)]\n\
         void main(uint i : S_DISPATCH_THREAD_ID) {\n\
         const Sampler s = Sampler().init().setWrapMode(Sampler::WrapMode::kWrap);\n\
         sink(s);\n\
         }",
    );
    assert!(
        hlsl.contains("SamplerState s;"),
        "builder chain must elide to a bare SamplerState, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("complex callee"),
        "no complex-callee fallback, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("sce::"),
        "no framework residue may leak, got:\n{hlsl}"
    );
}

#[test]
fn compare_chain_declares_comparison_sampler() {
    // A chain selecting a depth-compare function configures a
    // COMPARISON sampler: HLSL has no SamplerState→SamplerComparisonState
    // conversion, so the declaration itself must be comparison-typed.
    let hlsl = emit(
        "Texture2DArray<float> tex;\n\
         float f(float2 uv, float d, int c) {\n\
         const Sampler s = Sampler().init().setDepthCompareFunction(Sampler::DepthCompare::kLess);\n\
         SamplerComparisonState scs = SamplerComparisonState(s);\n\
         return tex.SampleCmpLevelZero(scs, float3(saturate(uv), c), d).x;\n\
         }",
    );
    assert!(
        hlsl.contains("SamplerComparisonState s;"),
        "compare chain must declare a comparison sampler, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("complex callee"),
        "no complex-callee fallback, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("sce::"),
        "no framework residue may leak, got:\n{hlsl}"
    );
}

#[test]
fn mutated_global_emits_static() {
    // In-shader mutation (RNG state) cannot live in the implicit
    // `$Globals` cbuffer (DXC: const-qualified); it emits `static`.
    // Untouched globals keep today's plain shape (host-bindable).
    let hlsl = emit(
        "uint m_state;\n\
         uint u_readonly;\n\
         uint next(uint seed) { m_state = seed; return m_state + u_readonly; }\n\
         [NUM_THREADS(1,1,1)]\n\
         void main(uint i : S_DISPATCH_THREAD_ID) {\n\
         uint x = next(i);\n\
         }",
    );
    assert!(
        hlsl.contains("static uint m_state;"),
        "mutated global must emit static, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("static uint u_readonly;"),
        "read-only global must stay plain, got:\n{hlsl}"
    );
}

#[test]
fn tutorial_sampler_factory_stubs_by_shape() {
    // `static constexpr SamplerState MakeX()` factories with
    // `Sampler().init().set*()` bodies stub to a trivial return by
    // shape (names vary per tutorial; the listed factories keep their
    // dedicated paths).
    let hlsl = emit(
        "static constexpr SamplerState MakeWrapLinearSampler() {\n\
         return SamplerState(sce::Agc::Core::Sampler().init().setWrapMode(sce::Agc::Core::Sampler::WrapMode::kWrap));\n\
         }\n\
         [NUM_THREADS(1,1,1)]\n\
         void main(uint i : S_DISPATCH_THREAD_ID) {\n\
         SamplerState s = MakeWrapLinearSampler();\n\
         }",
    );
    assert!(
        hlsl.contains("return s;"),
        "factory must stub to a trivial return, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("complex callee"),
        "no complex-callee fallback, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("sce::"),
        "no framework residue may leak, got:\n{hlsl}"
    );
}

#[test]
fn identical_getters_inline_on_dangling_root() {    // Two structs sharing one getter body (`RenderMeshSRT::getMaterial`
    // / `ForwardRenderSRT::getMaterial` in the wild) inline at a
    // dangling call: either body yields the same code, so the shared
    // name is unambiguous in effect.
    let hlsl = emit(
        "struct A { uint id; Texture2D<float4> tex; uint getId() { return id; } };\n\
         struct B { uint id; Texture2D<float4> tex; uint getId() { return id; } };\n\
         A srt : S_SRT_DATA;\n\
         uint fetch(A* p) { return p->getId(); }\n\
         [NUM_THREADS(1,1,1)]\n\
         void main(uint i : S_DISPATCH_THREAD_ID) {\n\
         uint x = fetch(srt);\n\
         }",
    );
    assert!(
        !hlsl.contains("getId()"),
        "dangling identical getter must inline, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("p->"),
        "no dangling root may remain, got:\n{hlsl}"
    );
}

#[test]
fn framework_buffer_member_hoists_byteaddress() {
    // `sce::Agc::Core::Buffer` is the untyped GPU buffer object: used
    // members hoist to `ByteAddressBuffer` bindings (unused ones strip,
    // like ordinary resource members).
    let hlsl = emit(
        "struct Geo { sce::Agc::Core::Buffer buf; uint stride; };\n\
         Geo g;\n\
         uint f(uint addr) { return g.buf.Load(addr); }\n\
         [NUM_THREADS(1,1,1)]\n\
         void main(uint i : S_DISPATCH_THREAD_ID) {\n\
         uint x = f(i);\n\
         }",
    );
    assert!(
        hlsl.contains("ByteAddressBuffer g_buf;"),
        "framework buffer must flatten to a ByteAddressBuffer binding, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("sce::"),
        "no framework residue may leak, got:\n{hlsl}"
    );
}

#[test]
fn bottom_level_bvh_emits_acceleration_structure() {
    // The bottom-level descriptor shares HLSL's single
    // acceleration-structure type with the top-level one.
    let hlsl = emit(
        "struct Build { sce::Psr::BottomLevelBvhDescriptor blas; uint n; };\n\
         Build b;\n\
         void take(RaytracingAccelerationStructure a) { }\n\
         [NUM_THREADS(1,1,1)]\n\
         void main(uint i : S_DISPATCH_THREAD_ID) {\n\
         take(b.blas);\n\
         }",
    );
    assert!(
        hlsl.contains("RaytracingAccelerationStructure b_blas;"),
        "BLAS descriptor must flatten to a binding, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("sce::"),
        "no framework residue may leak, got:\n{hlsl}"
    );
}
