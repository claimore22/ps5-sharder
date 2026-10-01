use ps5_shader::parse_shader;

// Regression tests for entry-IO varying semantics: PS5 links VS→PS
// varyings by member name, so sample headers leave them bare; Vulkan
// HLSL requires a semantic per field. Bare members of entry input
// structs (and vertex-like output structs) take the lowest free
// TEXCOORD indices in field order, so separately compiled
// producer/consumer sides agree. Carried semantics stay untouched,
// render-target outputs are never guessed, and the S_SRT_DATA root is
// uniforms, not varyings.

fn emit(source: &str) -> String {
    let module = parse_shader(source).expect("parser accepted shader");
    let model = ps5_shader::semantic::analyze(&module).expect("semantic analysis succeeded");
    let ir = ps5_shader::lower::lower_module(&module, &model).expect("lowering succeeded");
    ps5_shader::emit::emit_vulkan_hlsl(&ir)
}

#[test]
fn color_member_relocates_on_texcoord_clash() {
    // `m_color : COLOR` (= TEXCOORD0) plus explicit `m_uv0 : TEXCOORD0`
    // must not collide: the COLOR member takes the lowest free index.
    // Deterministic per struct so producer/consumer sides agree.
    let hlsl = emit(
        "struct Pin { float4 m_color : COLOR; float2 m_uv0 : TEXCOORD0; };\n\
         struct Pout { float4 c : S_TARGET_OUTPUT0; };\n\
         Pout main(Pin i) { Pout o; o.c = float4(i.m_color.rgb, 1); return o; }",
    );
    assert!(
        hlsl.contains("float4 m_color : TEXCOORD1;"),
        "clashing COLOR must relocate, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("float2 m_uv0 : TEXCOORD0;"),
        "explicit TEXCOORD must stand, got:\n{hlsl}"
    );
}

#[test]
fn ngg_group_index_relocates_to_free_texcoord() {
    // `S_NGG_GROUP_VERTEX_INDEXn` has no Vulkan spelling: it takes the
    // lowest free TEXCOORD like a COLOR would. `S_NGG_VERTEX_INDEX` joins
    // the pool too outside vertex entries (geometry rejects system IDs);
    // in vertex entries it maps exactly to `SV_VertexID`.
    let hlsl = emit(
        "struct Pin { uint vertex_index : S_NGG_VERTEX_INDEX; uint vtx0 : S_NGG_GROUP_VERTEX_INDEX0; float2 uv : TEXCOORD0; };\n\
         struct Pout { float4 c : S_TARGET_OUTPUT0; };\n\
         Pout main(Pin i) { Pout o; o.c = float4(i.vtx0, i.uv, 1); return o; }",
    );
    assert!(
        hlsl.contains("uint vertex_index : TEXCOORD1;"),
        "non-vertex NGG index must take lowest free, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("uint vtx0 : TEXCOORD2;"),
        "group index must take next free, got:\n{hlsl}"
    );
}

#[test]
fn ngg_vertex_index_maps_exactly_in_vertex_entries() {
    let hlsl = emit(
        "struct Pin { uint vertex_index : S_NGG_VERTEX_INDEX; float2 uv : TEXCOORD0; };\n\
         struct Vout { float4 pos : S_POSITION; };\n\
         Vout main(Pin i) { Vout o; o.pos = float4(i.uv, i.vertex_index, 1); return o; }",
    );
    assert!(
        hlsl.contains("uint vertex_index : SV_VertexID;"),
        "vertex index must map exactly, got:\n{hlsl}"
    );
}

#[test]
fn uniform_entry_param_hoists_to_cbuffer() {
    // `uniform float4x4 wvp` on entries is a host constant: Vulkan
    // entries take only varyings/resources, so it moves into a cbuffer
    // while uses keep the bare name.
    let hlsl = emit(
        "[CxxSymbol(\"Shader::vv\")]\n\
         void main(float3 aPosition : POSITION, uniform float4x4 wvp, out float4 vPosition : S_POSITION) {\n\
         vPosition = mul(wvp, float4(aPosition, 1.0));\n}",
    );
    assert!(
        hlsl.contains("cbuffer uniform_params"),
        "uniform param must hoist, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("float4x4 wvp;"),
        "member must keep the name, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("mul(wvp, float4(aPosition, 1.0))"),
        "uses must resolve to the member, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("float4x4 wvp,") && !hlsl.contains(", uniform"),
        "param must leave the signature, got:\n{hlsl}"
    );
}

#[test]
fn geometry_stream_output_struct_gains_texcoords() {
    // `inout TriangleStream<GsOut>` hides the output struct in the
    // resource element: its bare members must still gain TEXCOORDs.
    let hlsl = emit(
        "struct GsOut { float4 position : S_POSITION; float alpha; uint id; };\n\
         void main(point float4 input[1], inout TriangleStream<GsOut> triStream) {\n\
         GsOut vert; vert.alpha = 1; vert.id = 2u;\n\
         vert.position = input[0]; triStream.Append(vert);\n}",
    );
    assert!(
        hlsl.contains("float alpha : TEXCOORD0;"),
        "bare stream member must gain TEXCOORD0, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("uint id : TEXCOORD1;"),
        "second bare member must gain TEXCOORD1, got:\n{hlsl}"
    );
}

#[test]
fn fragment_bare_numeric_return_targets_zero() {
    // Pixel entries returning a bare scalar/vector with no return
    // semantic (`float3 main(...)`, IBL filter shaders) can only mean
    // render target 0 — fragment outputs have no other legal form.
    let hlsl = emit(
        "float3 main(float3 position : S_POSITION) : S_TARGET_OUTPUT0 { return position; }",
    );
    assert!(
        hlsl.contains(": SV_Target0"),
        "carried target must stay, got:\n{hlsl}"
    );
}

#[test]
fn fragment_bare_return_without_semantic_targets_zero() {
    let hlsl = emit(
        "[CxxSymbol(\"Shader::ps\")]\nfloat3 main(float3 position : S_POSITION) { return position; }",
    );
    assert!(
        hlsl.contains(": SV_Target0"),
        "bare numeric fragment return must target 0, got:\n{hlsl}"
    );
}

#[test]
fn fragment_float3_position_input_widens_to_float4() {
    // Vulkan VUID-FragCoord: a float3 SV_Position input produces an
    // invalid 3-component FragCoord; widen to float4 (xyz identical).
    let hlsl = emit(
        "[CxxSymbol(\"Shader::ps\")]\nfloat3 main(float3 position : S_POSITION) { return position + 0.5f; }",
    );
    assert!(
        hlsl.contains("float4 position : SV_Position"),
        "position input must widen, got:\n{hlsl}"
    );
}

#[test]
fn fragment_input_bare_members_gain_texcoords() {
    let hlsl = emit(
        "struct Pin { float4 pos : S_POSITION; float3 wp; float2 uv; };\n\
         struct Pout { float4 c : S_TARGET_OUTPUT0; };\n\
         Pout main(Pin i) { Pout o; o.c = float4(i.wp, 1); return o; }",
    );
    assert!(
        hlsl.contains("float4 pos : SV_Position;"),
        "carried position semantic must stay, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("float3 wp : TEXCOORD0;"),
        "first bare member must take TEXCOORD0, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("float2 uv : TEXCOORD1;"),
        "second bare member must take TEXCOORD1, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("float4 c : SV_Target0;"),
        "render-target output must stay guessed-free, got:\n{hlsl}"
    );
}

#[test]
fn existing_texcoord_indices_are_skipped() {
    // A carried TEXCOORD2 must not collide with assigned indices.
    let hlsl = emit(
        "struct Pin { float2 a : TEXCOORD2; float2 b; float2 c; };\n\
         struct Pout { float4 o : S_TARGET_OUTPUT0; };\n\
         Pout main(Pin i) { Pout o; o.o = float4(i.b + i.c, 0, 1); return o; }",
    );
    assert!(
        hlsl.contains("float2 b : TEXCOORD0;"),
        "fill must start at lowest free index, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("float2 c : TEXCOORD1;"),
        "fill must continue in order, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("float2 a : TEXCOORD2;"),
        "carried index must stay, got:\n{hlsl}"
    );
}

#[test]
fn vertex_output_bare_members_gain_texcoords() {
    let hlsl = emit(
        "struct Vout { float4 pos : S_POSITION; float2 uv; };\n\
         Vout main(float3 p : POSITION) { Vout o; o.pos = float4(p, 1); o.uv = p.xy; return o; }",
    );
    assert!(
        hlsl.contains("float2 uv : TEXCOORD0;"),
        "vertex output varying must gain a semantic, got:\n{hlsl}"
    );
}

#[test]
fn srt_root_struct_is_not_varyings() {
    // The S_SRT_DATA root carries uniforms: its members must not gain
    // varying semantics even when bare (cbuffer fields stay clean).
    let hlsl = emit(
        "struct Srt { Texture2D<float4> tex; float k; float j; };\n\
         Srt srt : S_SRT_DATA;\n\
         SamplerState smp;\n\
         struct Pout { float4 o : S_TARGET_OUTPUT0; };\n\
         Pout main(float2 uv : TEXCOORD0) { Pout o; o.o = srt.tex.Sample(smp, uv) * srt.k + srt.j; return o; }",
    );
    assert!(
        hlsl.contains("cbuffer srt_constants"),
        "root must still flatten to a cbuffer, got:\n{hlsl}"
    );
    let cb: String = hlsl
        .lines()
        .skip_while(|l| !l.contains("cbuffer srt_constants"))
        .skip(1)
        .take_while(|l| !l.contains("};"))
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        !cb.contains("TEXCOORD"),
        "cbuffer fields must stay semantic-free, got:\n{cb}"
    );
}

#[test]
fn color_param_relocates_on_texcoord_clash() {
    // Bare entry params (`float4 main(float2 uv : TEXCOORD0, float4
    // color : COLOR0)`, jobified_render font_p): the COLOR param claims
    // TEXCOORD0 like a COLOR member, so it relocates to the lowest free
    // index instead of colliding with the explicit TEXCOORD param.
    let hlsl = emit(
        "float4 main(float2 uv : TEXCOORD0, float4 color : COLOR0) : S_TARGET_OUTPUT {\n\
         \treturn color + float4(uv, 0, 1);\n\
         }",
    );
    assert!(
        hlsl.contains("float2 uv : TEXCOORD0"),
        "explicit TEXCOORD param must stand, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("float4 color : TEXCOORD1"),
        "clashing COLOR param must relocate, got:\n{hlsl}"
    );
}

#[test]
fn nested_varying_leaves_fill_around_sibling_explicit() {
    // PSSL nests varyings (`MULTIEXPORT_GS_OUTPUT.vertex : TEXCOORD0`
    // wrapping a bare `VS_OUTPUT` beside `numVerts : TEXCOORD1`): the
    // nested bare leaves take subtree-free indices in field order
    // (skipping the sibling's explicit index, so producer/consumer
    // sides sharing field order agree), and the enclosing struct-typed
    // member keeps no semantic (one on it is invalid DX linkage).
    let hlsl = emit(
        "struct Inner { float4 a; float2 b; };\n\
         struct Wrapper { Inner inner; uint n : TEXCOORD1; };\n\
         struct Pout { float4 c : S_TARGET_OUTPUT0; };\n\
         Pout main(Wrapper i) { Pout o; o.c = i.inner.a + float4(i.inner.b, i.n, 1); return o; }",
    );
    assert!(
        hlsl.contains("float4 a : TEXCOORD0;"),
        "first nested leaf must take the lowest free index, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("float2 b : TEXCOORD2;"),
        "second leaf must skip the sibling explicit index, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("Inner inner;"),
        "enclosing struct member must stay semantic-free, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("uint n : TEXCOORD1;"),
        "sibling explicit must stand, got:\n{hlsl}"
    );
}

#[test]
fn doubly_nested_struct_stays_loud() {
    // One struct definition nested through two members cannot share a
    // single location per instance: the planner must decline (leaves
    // stay bare and loud) rather than alias both instances onto one
    // index.
    let hlsl = emit(
        "struct Inner { float4 x; };\n\
         struct Wrapper { Inner a; Inner b; };\n\
         struct Pout { float4 c : S_TARGET_OUTPUT0; };\n\
         Pout main(Wrapper i) { Pout o; o.c = i.a.x + i.b.x; return o; }",
    );
    assert!(
        hlsl.contains("float4 x;"),
        "shared nested leaves must stay bare (loud), got:\n{hlsl}"
    );
}
