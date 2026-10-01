use ps5_shader::parse_shader;

// Regression tests for SRT-subobject handling: locals aliasing whole
// `srt.*` subtrees materialize from the flattened pieces, and
// whole-subobject paths in value positions become same-scope
// temporaries. Shadowing (`srt` naming anything but the flattened root)
// keeps today's loud path.

fn emit(source: &str) -> String {
    let module = parse_shader(source).expect("parser accepted shader");
    let model = ps5_shader::semantic::analyze(&module).expect("semantic analysis succeeded");
    let ir = ps5_shader::lower::lower_module(&module, &model).expect("lowering succeeded");
    ps5_shader::emit::emit_vulkan_hlsl(&ir)
}

const PRELUDE: &str = "struct Inner { float v[2]; float k; };\n\
     struct Root { Inner inner; Texture2D<float4> tex; float w; };\n\
     Root srt;\n\
     SamplerState smp;\n\
     void sink4(float4 x) {}\n\
     void sink(float x) {}\n";

#[test]
fn subobject_decl_materializes_memberwise() {
    let hlsl = emit(
        &format!("{PRELUDE}\n\
         [NUM_THREADS(1,1,1)]\n\
         void main(uint i : S_DISPATCH_THREAD_ID) {{\n\
         Inner x = srt.inner;\n\
         float4 c = srt.tex.SampleLevel(smp, float2(0.5, 0.5), 0);\n\
         float a = x.v[1] + x.k + srt.w;\n\
         sink4(c);\n\
         sink(a);\n\
         }}"),
    );
    assert!(
        hlsl.contains("Inner x;"),
        "alias decl must become a bare local, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("x.v[0] = v[0]"),
        "array members must assign per element, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("x.k = k"),
        "scalar members must assign from the cbuffer, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("srt.inner"),
        "no subobject path may leak, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("float v[2];"),
        "array members must keep array shape in the cbuffer, got:\n{hlsl}"
    );
}

#[test]
fn subobject_call_arg_splices_temporary() {
    // A helper taking a flattened subobject by value with a single
    // direct call site elides the parameter (rebinding to the flattened
    // leaves) instead of splicing a temporary: identical values, no
    // temporary. (Multi-site or indirect uses still splice.)
    let hlsl = emit(
        &format!("{PRELUDE}\n\
         float f(Inner s, float w) {{ return s.k + w; }}\n\
         [NUM_THREADS(1,1,1)]\n\
         void main(uint i : S_DISPATCH_THREAD_ID) {{\n\
         float y = f(srt.inner, srt.w);\n\
         }}"),
    );
    assert!(
        !hlsl.contains("Inner s"),
        "subobject param must elide, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("f(w)"),
        "call must drop the subobject argument, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("srt.inner"),
        "no subobject path may leak, got:\n{hlsl}"
    );
}

#[test]
fn shadowing_root_name_skips_rewrite() {
    let hlsl = emit(
        &format!("struct Other {{ float z; }};\n\
         Other o;\n\
         {PRELUDE}\n\
         float h(Other srt) {{ return srt.z; }}\n\
         [NUM_THREADS(1,1,1)]\n\
         void main(uint i : S_DISPATCH_THREAD_ID) {{\n\
         float q = h(o);\n\
         Inner x = srt.inner;\n\
         float a = x.k;\n\
         sink(q);\n\
         sink(a);\n\
         }}"),
    );
    assert!(
        hlsl.contains("srt.z"),
        "shadowed uses must stay untouched, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("Inner x;"),
        "global-root uses must still materialize, got:\n{hlsl}"
    );
}

#[test]
fn indexed_subobject_uses_buffer_binding() {
    let hlsl = emit(
        &format!("struct Item {{ float4x4 m; }};\n\
         struct Root2 {{ Item items; Texture2D<float4> tex; }};\n\
         Root2 srt2;\n\
         void sink(float x) {{}}\n\
         {PRELUDE}\n\
         [NUM_THREADS(1,1,1)]\n\
         void main(uint i : S_DISPATCH_THREAD_ID) {{\n\
         float4x4 m = srt2.items[i].m;\n\
         float4 c = srt2.tex.SampleLevel(smp, float2(0.5, 0.5), 0);\n\
         sink(m[0][0] + c.x);\n\
         }}"),
    );
    assert!(
        hlsl.contains("StructuredBuffer<Item> srt2_items;"),
        "indexed struct member must hoist as a buffer, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("srt2_items[i].m"),
        "indexed use must rewrite to the binding, got:\n{hlsl}"
    );
}

#[test]
fn method_call_on_subobject_materializes_receiver() {
    // `srt.inner.method(...)` after root flattening: the dotted call
    // string must materialize the struct prefix (`Inner __srtarg_N`
    // from the flattened leaves) and retarget the receiver, or `srt`
    // dangles (PSR `srt.m_camera.generate` pattern).
    let hlsl = emit(
        &format!("struct Lens {{ float f; float g() {{ return f * 2.0; }} }};\n\
         struct Root3 {{ Lens lens; Texture2D<float4> tex; float w; }};\n\
         Root3 srt3;\n\
         {PRELUDE}\n\
         [NUM_THREADS(1,1,1)]\n\
         void main(uint i : S_DISPATCH_THREAD_ID) {{\n\
         float y = srt3.lens.g() + srt3.w;\n\
         sink(y);\n\
         }}"),
    );
    assert!(
        !hlsl.contains("srt3.lens.g"),
        "subobject receiver must not leak, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains(".g()"),
        "method must survive on the materialized temp, got:\n{hlsl}"
    );
}

#[test]
fn block_outputs_escape_with_hoisted_temps() {
    // A desugared-destructure block holding a subobject call must not
    // trap its outputs: temps hoist before the block, the block stays
    // VarDecl-only (emitted flat), and later uses resolve.
    let hlsl = emit(
        &format!("struct Lens {{ float f; float g() {{ return f * 2.0; }} }};\n\
         struct Pair {{ float a; float b; }};\n\
         Pair mk(float x) {{ Pair p; p.a = x; p.b = x; return p; }}\n\
         struct Root4 {{ Lens lens; Texture2D<float4> tex; }};\n\
         Root4 srt4;\n\
         RWBuffer<float> outBuf;\n\
         [NUM_THREADS(1,1,1)]\n\
         void main(uint i : S_DISPATCH_THREAD_ID) {{\n\
         float t = srt4.lens.g();\n\
         auto const [u, v] = mk(t);\n\
         outBuf[0] = u + v;\n\
         }}"),
    );
    assert!(
        !hlsl.contains("srt4.lens.g"),
        "subobject receiver must not leak, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("outBuf[0]"),
        "block outputs must stay visible, got:\n{hlsl}"
    );
}

#[test]
fn struct_array_index_transfers_to_binding() {
    // `srt bufs[k].leaf` through a sized intermediate struct array
    // (psfx classify SRT): the leaf binding carries the intermediate
    // dimension and the use transfers the index inward
    // (`srt.bufs[k].tile[j]` -> `BINDING[k][j]`).
    let hlsl = emit(
        &format!("struct Slot {{ RWStructuredBuffer<uint> tile; uint tag; }};\n\
         struct Root5 {{ Slot bufs[2]; Texture2D<float4> tex; float w; }};\n\
         Root5 srt5;\n\
         RWBuffer<uint> outBuf;\n\
         [NUM_THREADS(1,1,1)]\n\
         void main(uint i : S_DISPATCH_THREAD_ID) {{\n\
         srt5.bufs[i].tile[0] = srt5.bufs[1u].tag + srt5.w;\n\
         outBuf[0] = srt5.bufs[0u].tile[i];\n\
         }}"),
    );
    assert!(
        hlsl.contains("srt5_bufs_tile[2]"),
        "leaf binding must carry the intermediate dimension, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("srt5_bufs_tile[i][0]") || hlsl.contains("srt5_bufs_tile[i]"),
        "index must transfer inward, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("srt5.bufs"),
        "no struct-array path may leak, got:\n{hlsl}"
    );
}

#[test]
fn struct_array_element_alias_resolves() {
    // `Slot s = srt.bufs[k]; s.tile[j]` (whole-element locals over
    // struct arrays): the alias dissolves into the indexed path and
    // transfers like a direct use.
    let hlsl = emit(
        &format!("struct Slot2 {{ RWStructuredBuffer<uint> tile; uint tag; }};\n\
         struct Root6 {{ Slot2 bufs[2]; Texture2D<float4> tex; float w; }};\n\
         Root6 srt6;\n\
         RWBuffer<uint> outBuf;\n\
         [NUM_THREADS(1,1,1)]\n\
         void main(uint i : S_DISPATCH_THREAD_ID) {{\n\
         Slot2 s = srt6.bufs[i];\n\
         s.tile[0] = s.tag + srt6.w;\n\
         outBuf[0] = s.tile[i];\n\
         }}"),
    );
    assert!(
        !hlsl.contains("Slot2 s =") && !hlsl.contains("srt6.bufs"),
        "alias decl and paths must dissolve, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("srt6_bufs_tile[i][0]"),
        "aliased store must transfer inward, got:\n{hlsl}"
    );
}

#[test]
fn subobject_param_elides_to_flattened_leaves() {
    // Helpers taking a flattened subobject by value
    // (`f(srt.inner, ...)`) drop the parameter and rebind `param.*`
    // to the flattened leaves; forwarding chains
    // (`f` passing its param to an elided `g`) converge too.
    let hlsl = emit(
        &format!("struct Scene {{ float k; Texture2D<float4> tex; }};\n\
         struct Root7 {{ Scene scene; Texture2D<float4> tx; float w; }};\n\
         Root7 srt7;\n\
         SamplerState smp;\n\
         float g(Scene sc, float w) {{ return sc.k * w; }}\n\
         float f(Scene sc, float w) {{ return g(sc, w) + sc.k; }}\n\
         RWBuffer<float> outBuf;\n\
         [NUM_THREADS(1,1,1)]\n\
         void main(uint i : S_DISPATCH_THREAD_ID) {{\n\
         outBuf[0] = f(srt7.scene, srt7.w);\n\
         }}"),
    );
    assert!(
        hlsl.contains("float g(float w)") && hlsl.contains("float f(float w)"),
        "subobject params must elide, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("g(w)"),
        "forwarded call must drop the subobject argument, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("srt7.scene"),
        "no subobject path may leak, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("outBuf[0]"),
        "entry must survive, got:\n{hlsl}"
    );
}

#[test]
fn indexed_scalar_element_copy_materializes() {
    // `Light l = sc.lights[i];` over an all-scalar element struct
    // through a helper param (forward-lighting shape): memberwise copy
    // from the dimensioned cbuffer arrays (scalar arrays unrolled with
    // a linearized index). Root-anchored all-scalar arrays hoist
    // earlier; this covers the param-rooted remainder.
    let hlsl = emit(
        &format!("struct Light {{ float4 pos; int pad[2]; }};\n\
         struct Scene2 {{ Light lights[3]; float w; }};\n\
         struct Root8 {{ Scene2 scene; float z; Texture2D<float4> tx; }};\n\
         Root8 srt8;\n\
         float f(Scene2 sc, float w, uint i) {{ Light l = sc.lights[i]; return l.pos.x + l.pad[1] + w; }}\n\
         RWBuffer<float> outBuf;\n\
         [NUM_THREADS(1,1,1)]\n\
         void main(uint i : S_DISPATCH_THREAD_ID) {{\n\
         outBuf[0] = f(srt8.scene, srt8.scene.w, i);\n\
         }}"),
    );
    assert!(
        hlsl.contains("Light l;"),
        "copy must become a bare local, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("l.pos = "),
        "scalar members must copy from the arrays, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("l.pad[0] = ") && hlsl.contains("l.pad[1] = "),
        "array members must unroll, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("srt8.scene.lights") && !hlsl.contains("sc.lights"),
        "no indexed path may leak, got:\n{hlsl}"
    );
}

#[test]
fn static_const_auto_alias_dissolves() {
    // `static const auto frameBuffer = srt->frameBuffer;` (jobified
    // compute shaders): global initializers never survive parsing, so
    // the alias erases and uses rewrite to the member path (flattening
    // resolves it to the binding). Resources substitute unconditionally
    // (reference semantics); data needs a write-free backing.
    let hlsl = emit(
        "struct Srt9 { RWTexture2D<float4> frameBuffer; float time; };\n\
         Srt9 srt9 : S_SRT_DATA;\n\
         static const auto frameBuffer = srt9.frameBuffer;\n\
         static const auto time = srt9.time;\n\
         [NUM_THREADS(1,1,1)]\n\
         void main(uint i : S_DISPATCH_THREAD_ID) {\n\
         frameBuffer[uint2(i, 0)] = float4(time, 0, 0, 1);\n\
         }",
    );
    assert!(
        !hlsl.contains("auto frameBuffer") && !hlsl.contains("auto time"),
        "alias globals must dissolve, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("srt9_frameBuffer[uint2(i, 0)]"),
        "resource uses must rebind to the flattened binding, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains(" frameBuffer["),
        "no bare alias store may leak, got:\n{hlsl}"
    );
}

#[test]
fn variable_load_offset_folds_into_coordinate() {
    // `depth.Load(int3(...), off)` with a runtime `int2` offset is a texel
    // offset (PSSL adds it to the coordinate); HLSL requires immediate
    // offsets, so it folds into the coordinate instead of remaining a
    // (invalid) variable second argument.
    let hlsl = emit(
        "struct Srt9 { Texture2D<float> depth; float sourceLevel; };\n\
         Srt9 srt9 : S_SRT_DATA;\n\
         RWBuffer<float> dst;\n\
         [NUM_THREADS(1,1,1)]\n\
         void main(uint2 i : S_DISPATCH_THREAD_ID) {\n\
         int2 off = int2(1, 0);\n\
         float d = srt9.depth.Load(int3((int2)i, (int)srt9.sourceLevel), off);\n\
         dst[0] = d;\n\
         }",
    );
    assert!(
        hlsl.contains("int3(off.x, off.y, 0)"),
        "variable offset must fold into the coordinate, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains(", off)"),
        "offset must not remain a Load argument, got:\n{hlsl}"
    );
}
