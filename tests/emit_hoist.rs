use ps5_shader::parse_shader;

// Regression tests for descriptor hoisting (DXC nesting family:
// `StructuredBuffer<Texture2D<float4>>`,
// `StructuredBuffer<GeometrySRT>`-with-buffers, ...): resource members
// nested inside structs become top-level bindings, access paths rewritten.

fn emit(source: &str) -> String {
    let module = parse_shader(source).expect("parser accepted shader");
    let model = ps5_shader::semantic::analyze(&module).expect("semantic analysis succeeded");
    let ir = ps5_shader::lower::lower_module(&module, &model).expect("lowering succeeded");
    ps5_shader::emit::emit_vulkan_hlsl(&ir)
}

// Rule A2: buffer-of-buffer-members splits like textures when member
// hoisting cannot (uses through a materialized local ban the static
// path): each buffer member becomes a per-buffer unbounded array of
// same-typed buffers, and whole-struct values materialize from the
// twin + arrays (matl_skinned GeometrySRT).
#[test]
fn buffer_member_splits_to_buffer_array() {
    let hlsl = emit(
        "struct Geo { RegularBuffer<float3> pos; uint n; };\n\
         RegularBuffer<Geo> geos : register(t0);\n\
         float3 f(uint i, uint j) { Geo g = geos[i]; return g.pos[j] + g.n; }",
    );
    assert!(
        hlsl.contains("StructuredBuffer<float3> geos_pos_arr[]"),
        "buffer member must split to a buffer array, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("g.pos = geos_pos_arr[i]"),
        "materialized member must bind the array element, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("StructuredBuffer<Geo>"),
        "resourceful element must be gone, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("RegularBuffer<Geo>"),
        "no buffer-of-resourceful-struct may remain, got:\n{hlsl}"
    );
}

// Rule M: UAV mipmap views (`tex.MipMaps[m][c]`) hoist to an unbounded
// array of same-typed views (single HLSL views cannot address mips),
// for reads and writes alike (agc_build_mips downsample).
#[test]
fn mipmap_views_hoist_to_view_array() {
    let hlsl = emit(
        "RWTexture2DArray<float4> tex : register(u0);\n\
         [NUM_THREADS(8,8,1)]\n\
         void main(uint3 tid : S_DISPATCH_THREAD_ID) {\n\
         int3 itc = int3(tid);\n\
         tex.MipMaps[1][itc] = float4(1, 0, 0, 1);\n\
         tex.MipMaps[2][itc] = tex.MipMaps[1][itc] * 2.0f;\n\
         }",
    );
    assert!(
        hlsl.contains("RWTexture2DArray<float4> __mips_tex[]"),
        "mip views must hoist, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("__mips_tex[1][itc]"),
        "mip indexing must transfer, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("MipMaps"),
        "no MipMaps member may remain, got:\n{hlsl}"
    );
}

// Rule A: buffer-of-textures member behind a static path hoists to a
// same-typed global; the struct navigation drops, the member index stays.
#[test]
fn nested_texture_buffer_hoists_to_global() {
    let hlsl = emit(
        "struct Mat { RegularBuffer<Texture2D<float4>> tex; };\n\
         struct SRT { Mat m; };\n\
         SRT srt;\n\
         SamplerState smp;\n\
         float4 f(float2 uv) { return srt.m.tex[0].Sample(smp, uv); }",
    );
    assert!(
        hlsl.contains("__hoist_Mat_tex[0].Sample"),
        "nested texture buffer must hoist, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("StructuredBuffer<Texture2D<float4>>"),
        "nested resource template must be gone, got:\n{hlsl}"
    );
}

// Rule B-indexed: buffer-of-structs-with-buffers hoists to an array
// global; the outer index transfers onto it.
#[test]
fn nested_buffer_member_hoists_with_index_transfer() {
    let hlsl = emit(
        "struct Geo { RegularBuffer<float3> pos; };\n\
         RegularBuffer<Geo> geos : register(t0);\n\
         float3 f(uint i, uint j) { return geos[i].pos[j]; }",
    );
    assert!(
        hlsl.contains("__hoist_Geo_pos_arr[i][j]"),
        "outer index must transfer to the array global, got:\n{hlsl}"
    );
}

// Materialized struct temporaries have no static backing store: leave
// them (and their definitions) untouched.
#[test]
fn materialized_struct_copy_is_untouched() {
    let hlsl = emit(
        "struct Geo { RegularBuffer<float3> pos; };\n\
         RegularBuffer<Geo> geos : register(t0);\n\
         float3 f(uint i, uint j) { Geo g = geos[i]; return g.pos[j]; }",
    );
    assert!(
        !hlsl.contains("__hoist_"),
        "copies must not hoist, got:\n{hlsl}"
    );
}

// PSSL append buffer (`AppendRegularBuffer<T>`) emits as HLSL
// `AppendStructuredBuffer<T>` — both as a hoisted global and as a
// struct/local (materialized-copy) member.
#[test]
fn append_regular_buffer_maps_to_append_structured() {
    let hlsl = emit(
        "struct ParticleInstance { float4 pos; };\n\
         struct UpdateInfo { AppendRegularBuffer<ParticleInstance> outParticles; };\n\
         UpdateInfo info;\n\
         void f() {\n\
             AppendRegularBuffer<ParticleInstance> loc = info.outParticles;\n\
             loc.Append(ParticleInstance(0));\n\
         }",
    );
    assert!(
        hlsl.contains("AppendStructuredBuffer<ParticleInstance>"),
        "append buffer must map to AppendStructuredBuffer, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("AppendRegularBuffer"),
        "raw PSSL kind must not leak, got:\n{hlsl}"
    );
}

// PSSL byte buffer (`ByteBuffer` / `RW_ByteBuffer`) maps to the untyped
// HLSL `ByteAddressBuffer` / `RWByteAddressBuffer` (no element suffix).
#[test]
fn byte_buffer_maps_to_byte_address_without_element() {
    let hlsl = emit(
        "struct S { ByteBuffer buf; };\n\
         S s;\n\
         uint f() { return s.buf.Load(0); }",
    );
    assert!(
        hlsl.contains("ByteAddressBuffer"),
        "byte buffer must map to ByteAddressBuffer, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("ByteBuffer<"),
        "byte buffer must not carry an element template, got:\n{hlsl}"
    );
}

// Const-arithmetic array sizes (`ao_store[k_lds_size]` where
// `k_lds_size = k_wavefront_size * k_lds_line_size`) must fold to a concrete
// size instead of emitting `[]` (DXC: array type needs explicit size).
#[test]
fn const_arithmetic_array_size_is_folded() {
    let hlsl = emit(
        "static const int k_radius = 2;\n\
         static const int k_wavefront = 8;\n\
         static const int k_line = (k_wavefront + k_radius * 2);\n\
         static const int k_size = k_wavefront * k_line;\n\
         groupshared float ao_store[k_size];\n\
         float f(uint i) { return ao_store[i]; }",
    );
    assert!(
        hlsl.contains("ao_store[96]"),
        "const-arithmetic array size must fold to 96, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("ao_store[]"),
        "array must not be emitted unsized, got:\n{hlsl}"
    );
}

// Two different SRT structs carrying the same scalar member name must not
// collide in the flattened cbuffer (DXC: `redefinition of 'mipScale'`).
#[test]
fn scalar_cbuffer_dedups_colliding_member_names() {
let hlsl = emit(
        "struct EnvInfo { float envLod; float mipScale; };\n\
         struct Options { float envLod; float mipScale; uint tonemap; };\n\
         struct Srt { EnvInfo env; Options options; Texture2D<float4> tex; };\n\
         Srt srt : S_SRT_DATA;\n\
         SamplerState smp;\n\
         float4 f(float2 uv) { return float4(srt.env.mipScale + srt.options.mipScale, 0, 0, 1) + srt.tex.Sample(smp, uv); }",
    );
    assert!(
        hlsl.contains("options_mipScale"),
        "colliding scalar member must be disambiguated, got:\n{hlsl}"
    );
    // Inside the `srt_constants` cbuffer, `mipScale` must appear exactly once
    // and the colliding second member must be renamed to `options_mipScale`.
    let cb: String = hlsl
        .lines()
        .skip_while(|l| !l.contains("cbuffer srt_constants"))
        .skip(1)
        .take_while(|l| !l.contains("};"))
        .collect::<Vec<_>>()
        .join("\n");
    let decls = cb.matches("float mipScale;").count();
    assert_eq!(
        decls, 1,
        "cbuffer must declare mipScale exactly once (deduped), cbuffer:\n{cb}"
    );
    assert!(
        cb.contains("float options_mipScale;"),
        "colliding member must be renamed, cbuffer:\n{cb}"
    );
}

// Buffer-of-structs-with-textures: the buffer keeps its name with a
// scalars-only twin element, texture members become per-buffer arrays,
// and whole-struct copies materialize from both pieces.
#[test]
fn textured_struct_buffer_splits() {
    let hlsl = emit(
        "struct Mat { float k; Texture2D<float4> tex; };\n\
         StructuredBuffer<Mat> mats;\n\
         SamplerState smp;\n\
         float4 f(float2 uv, uint i) {\n\
         Mat m = mats[i];\n\
         return m.tex.Sample(smp, uv) * m.k;\n\
         }\n\
         [NUM_THREADS(1,1,1)]\n\
         void main(uint i : S_DISPATCH_THREAD_ID) {\n\
         float4 c = f(float2(0.5, 0.5), i);\n\
         }",
    );
    assert!(
        hlsl.contains("StructuredBuffer<MatScalars> mats;"),
        "buffer must take the scalars twin, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("Texture2D<float4> mats_tex_arr[];"),
        "texture member must become a per-buffer array, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("m.tex = mats_tex_arr[i];") || hlsl.contains("m.tex=mats_tex_arr[i];"),
        "copy must materialize the texture from the array, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("StructuredBuffer<Mat>"),
        "no textured buffer declaration may leak, got:\n{hlsl}"
    );
}

#[test]
fn helper_buffer_param_splits_to_caller_buffer() {
    // A helper taking `StructuredBuffer<S>` by value re-points at the
    // caller's split buffer and drops the parameter (a by-value struct
    // parameter cannot be split at call sites): the existing
    // whole-struct materialization then handles the element.
    let hlsl = emit(
        "struct Mat { float k; Texture2D<float4> tex; };\n\
         StructuredBuffer<Mat> mats;\n\
         SamplerState smp;\n\
         float4 f(StructuredBuffer<Mat> buf, uint i, float2 uv) {\n\
         Mat m = buf[i];\n\
         return m.tex.Sample(smp, uv) * m.k;\n\
         }\n\
         [NUM_THREADS(1,1,1)]\n\
         void main(uint i : S_DISPATCH_THREAD_ID) {\n\
         float4 c = f(mats, i, float2(0.5, 0.5));\n}",
    );
    assert!(
        !hlsl.contains("StructuredBuffer<Mat>"),
        "no textured buffer declaration may leak, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("f(i, float2(0.5, 0.5))"),
        "call must drop the buffer argument, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("m.tex = mats_tex_arr[i];") || hlsl.contains("m.tex=mats_tex_arr[i];"),
        "copy must materialize from the caller arrays, got:\n{hlsl}"
    );
}

#[test]
fn textured_struct_member_use_rewrites() {
    let hlsl = emit(
        "struct Mat { float k; Texture2D<float4> tex; };\n\
         StructuredBuffer<Mat> mats;\n\
         SamplerState smp;\n\
         float4 f(float2 uv, uint i) {\n\
         return mats[i].tex.Sample(smp, uv);\n\
         }\n\
         [NUM_THREADS(1,1,1)]\n\
         void main(uint i : S_DISPATCH_THREAD_ID) {\n\
         float4 c = f(float2(0.5, 0.5), i);\n\
         }",
    );
    assert!(
        hlsl.contains("__hoist_Mat_tex_arr[i].Sample("),
        "direct global-path member use stays with hoisting, got:\n{hlsl}"
    );
}

#[test]
fn textured_struct_with_scalar_array_materializes() {
    let hlsl = emit(
        "struct Mat { uint m; uint pad[2]; Texture2D<float4> tex; };\n\
         StructuredBuffer<Mat> mats;\n\
         SamplerState smp;\n\
         float4 f(float2 uv, uint i) {\n\
         Mat m = mats[i];\n\
         return m.tex.Sample(smp, uv) + m.m + m.pad[1];\n\
         }\n\
         [NUM_THREADS(1,1,1)]\n\
         void main(uint i : S_DISPATCH_THREAD_ID) {\n\
         float4 c = f(float2(0.5, 0.5), i);\n\
         }",
    );
    assert!(
        hlsl.contains("m.pad[0] = ") || hlsl.contains("m.pad[0]= "),
        "scalar arrays must unroll on materialize, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("StructuredBuffer<Mat>"),
        "no textured buffer declaration may leak, got:\n{hlsl}"
    );
}

#[test]
fn textured_struct_copy_in_main_materializes() {
    let hlsl = emit(
        "struct MI { uint m_materialIndex; };\n\
         struct Mat { float k; Texture2D<float4> tex; };\n\
         StructuredBuffer<Mat> mats;\n\
         StructuredBuffer<MI> mis;\n\
         SamplerState smp;\n\
         struct PSIn { float2 m_uv; };\n\
         float4 main(PSIn input) : SV_Target0 {\n\
         uint materialIndex = mis[0].m_materialIndex;\n\
         Mat material = mats[materialIndex];\n\
         float3 diffuse = material.tex.Sample(smp, input.m_uv).rgb * material.k;\n\
         return float4(diffuse, 1.f);\n\
         }",
    );
    assert!(
        hlsl.contains("Mat material;"),
        "copy must become a bare local, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("material.tex = "),
        "texture must assign from the array, got:\n{hlsl}"
    );
}

#[test]
fn textured_struct_full_material_copy_materializes() {
    let hlsl = emit(
        "struct MI { uint m_materialIndex; };\n\
         struct Mat { uint m_model; uint pad[2]; float3 m_diffuseWeights; Texture2D<float4> tex; Texture2D<float3> spec; };\n\
         StructuredBuffer<Mat> mats;\n\
         StructuredBuffer<MI> mis;\n\
         SamplerState smp;\n\
         struct PSIn { float2 m_uv; };\n\
         float4 main(PSIn input) : SV_Target0 {\n\
         uint materialIndex = mis[0].m_materialIndex;\n\
         Mat material = mats[materialIndex];\n\
         float3 diffuse = material.tex.Sample(smp, input.m_uv).rgb * material.m_diffuseWeights;\n\
         return float4(diffuse, 1.f);\n\
         }",
    );
    assert!(
        hlsl.contains("Mat material;"),
        "copy must become a bare local, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("material.tex = mats_tex_arr[materialIndex];")
            || hlsl.contains("material.tex=mats_tex_arr[materialIndex];"),
        "texture must assign from the array, got:\n{hlsl}"
    );
}
