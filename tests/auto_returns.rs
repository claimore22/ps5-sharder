use ps5_shader::parse_shader;

// Regression tests for C++14/17 features in PSR headers:
// function-local structs (lifted to module scope), `auto` return
// deduction (unanimous returns), and structured bindings (splat once
// the callee return is known). Unknowable shapes stay loud at DXC.

fn emit(source: &str) -> String {
    let module = parse_shader(source).expect("parser accepted shader");
    let model = ps5_shader::semantic::analyze(&module).expect("semantic analysis succeeded");
    let ir = ps5_shader::lower::lower_module(&module, &model).expect("lowering succeeded");
    ps5_shader::emit::emit_vulkan_hlsl(&ir)
}

#[test]
fn local_struct_lifts_to_module_scope() {
    let module = parse_shader(
        "[NUM_THREADS(1,1,1)]\n\
         void main(uint position : S_DISPATCH_THREAD_ID) {\n\
         struct output { float3 m_org; float3 m_dir; };\n\
         output r;\n\
         }",
    )
    .unwrap();
    assert!(
        module.structs.iter().any(|s| s.name.starts_with("output__local")),
        "local struct must lift mangled, got: {:?}",
        module.structs.iter().map(|s| &s.name).collect::<Vec<_>>()
    );
}

#[test]
fn auto_return_deduces_from_variable() {
    let hlsl = emit(
        "uint helper(uint x) { return x + 1u; }\n\
         [NUM_THREADS(1,1,1)]\n\
         void main(uint position : S_DISPATCH_THREAD_ID) {\n\
         uint y = helper(position);\n\
         }",
    );
    assert!(
        hlsl.contains("uint helper(uint x)"),
        "trivial sanity, got:\n{hlsl}"
    );
    let hlsl = emit(
        "struct Pair { float a; float b; };\n\
         auto make(float x) { Pair p; return p; }\n\
         [NUM_THREADS(1,1,1)]\n\
         void main(uint position : S_DISPATCH_THREAD_ID) {\n\
         Pair q = make(position);\n\
         }",
    );
    assert!(
        hlsl.contains("Pair make(float x)"),
        "auto must deduce to Pair, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("auto make"),
        "no auto return may leak, got:\n{hlsl}"
    );
}

#[test]
fn structured_binding_splats_known_struct() {
    // (The bindings escape through a call: unread bindings are dead
    // code and drop in lowering.)
    let hlsl = emit(
        "struct Pair { float a; float b; };\n\
         Pair make(float x) { Pair p; return p; }\n\
         float sink(float a, float b) { return a + b; }\n\
         [NUM_THREADS(1,1,1)]\n\
         void main(uint position : S_DISPATCH_THREAD_ID) {\n\
         auto [u, v] = make(position);\n\
         float w = sink(u, v);\n\
         }",
    );
    assert!(
        hlsl.contains("u = __destructure"),
        "binding must splat members, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("auto [u"),
        "binding syntax must not leak, got:\n{hlsl}"
    );
}

#[test]
fn mixed_returns_stay_auto_and_loud() {
    // Non-unanimous returns cannot deduce: `auto` stays and fails loudly
    // at DXC rather than guessing.
    let hlsl = emit(
        "auto f(uint x) { if (x == 1u) { return x; } return float4(x, 1, 1, 1); }\n\
         [NUM_THREADS(1,1,1)]\n\
         void main(uint position : S_DISPATCH_THREAD_ID) {\n\
         uint y = f(position);\n\
         }",
    );
    assert!(
        hlsl.contains("auto f(uint x)"),
        "mixed returns must stay auto (loud), got:\n{hlsl}"
    );
}

#[test]
fn auto_local_from_buffer_index_deduces() {
    // `const auto pos = srt->position[index];` (vis_draw_vv): the indexed
    // buffer element type becomes the local's type (HLSL has no `auto`).
    let hlsl = emit(
        "struct SRT { DataBuffer<float3> position; };\n\
         SRT *srt : S_SRT_DATA;\n\
         float4 main(uint index : S_VERTEX_ID) : S_TARGET_OUTPUT {\n\
         \tconst auto pos = srt->position[index];\n\
         \treturn float4(pos, 1.0f);\n\
         }",
    );
    assert!(
        !hlsl.contains("auto pos"),
        "auto must deduce, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("float3 pos"),
        "buffer element type must land, got:\n{hlsl}"
    );
}

#[test]
fn auto_local_swizzle_and_arith_deduce() {
    // Swizzles narrow vectors; int+float arithmetic comes out float;
    // comparisons come out bool. Later autos see earlier ones.
    let hlsl = emit(
        "[NUM_THREADS(1,1,1)]\n\
         void main(uint position : S_DISPATCH_THREAD_ID) {\n\
         float4 tan = float4(1, 0, 0, 1);\n\
         auto v = tan.xyz;\n\
         auto w = tan.w;\n\
         auto m = 0;\n\
         auto x = m + 1.0f;\n\
         auto b = x > 0.5f;\n\
         tan.x = v.x + w + x + (b ? 1.0f : 0.0f);\n\
         }",
    );
    for want in ["float3 v =", "float w =", "int m =", "float x =", "bool b ="] {
        assert!(
            hlsl.contains(want),
            "auto must deduce to `{want}`, got:\n{hlsl}"
        );
    }
    assert!(!hlsl.contains("auto "), "no auto must leak, got:\n{hlsl}");
}

#[test]
fn auto_local_from_call_and_member_deduce() {
    // Calls take the callee's concrete return (post return-deduction);
    // member paths take the field type, chained.
    let hlsl = emit(
        "struct M { float4 mvp; };\n\
         struct SRT { M mat; float k; };\n\
         float3 heat(float s, bool b = false) { return float3(s, s, s); }\n\
         SRT *srt : S_SRT_DATA;\n\
         float4 main(float2 uv : TEXCOORD0) : S_TARGET_OUTPUT {\n\
         \tauto h = heat(uv.x);\n\
         \tauto m = srt->mat.mvp;\n\
         \tauto k = srt->k;\n\
         \treturn m * float4(h + k, 1.0f);\n\
         }",
    );
    for want in ["float3 h =", "float4 m =", "float k ="] {
        assert!(
            hlsl.contains(want),
            "auto must deduce to `{want}`, got:\n{hlsl}"
        );
    }
}

#[test]
fn auto_local_array_index_and_splat_deduce() {
    // Array indexing yields the element type (an array of vectors is
    // NOT a vector indexing to scalar), and scalar-vector arithmetic
    // broadcasts (`HALTON[i] - 0.5f` stays float2; basic_draw_p).
    let hlsl = emit(
        "static float2 HALTON[8] = { float2(0.5f, 0.33f), float2(0.25f, 0.67f), float2(0.75f, 0.11f), float2(0.13f, 0.44f), float2(0.63f, 0.78f), float2(0.38f, 0.22f), float2(0.88f, 0.56f), float2(0.06f, 0.89f) };\n\
         [NUM_THREADS(1,1,1)]\n\
         void main(uint position : S_DISPATCH_THREAD_ID) {\n\
         \tauto point = HALTON[position % 8] - 0.5f;\n\
         \tauto px = point.x;\n\
         \tposition = uint(px * 100.0f);\n\
         }",
    );
    assert!(
        hlsl.contains("float2 point_ ="),
        "array element + splat must stay float2, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("float px ="),
        "component must be scalar, got:\n{hlsl}"
    );
}

#[test]
fn captureless_lambda_beta_reduces() {
    // `auto convertUInt16 = [](ulong v) { return ...; };` with direct
    // calls inlines per call site (clustered-geometry vertex_shading.h):
    // the declaration drops and `auto` inits from reduced calls deduce.
    let hlsl = emit(
        "struct W { uint w; };\n\
         float4 uncompressBoneWeights(uint input) {\n\
         \tfloat4 ret;\n\
         \tauto convertUInt16 = [](ulong v) { return (v & 0xffff) / 65535.0f; };\n\
         \tret.x = convertUInt16(input);\n\
         \tret.y = convertUInt16(input >> 16);\n\
         \treturn ret;\n\
         }\n\
         [NUM_THREADS(1,1,1)]\n\
         void main(uint position : S_DISPATCH_THREAD_ID) {\n\
         \tfloat4 w = uncompressBoneWeights(position);\n\
         }",
    );
    assert!(
        !hlsl.contains("convertUInt16"),
        "lambda must inline fully, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("auto ") && !hlsl.contains("__lambda"),
        "no auto or placeholder must leak, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("65535.0f"),
        "lambda body must survive at call sites, got:\n{hlsl}"
    );
}

#[test]
fn byvalue_capture_beta_reduces_when_frozen() {
    // `auto getCompactIndex64 = [vert_ballot](uint index) { ... };`
    // (culling compaction idiom): the by-value capture is never mutated,
    // so the definition-site value equals the value at every call and
    // the lambda inlines with the name in place. The two-statement body
    // (pure temp + return) elides through the temp.
    let hlsl = emit(
        "struct Pout { float4 c : S_TARGET_OUTPUT0; };\n\
         Pout main(uint idx : TEXCOORD0) {\n\
         \tulong const vert_ballot = ballot(true);\n\
         \tuint i0 = idx;\n\
         \tauto getCompactIndex64 = [vert_ballot](uint index) -> uint {\n\
         \t\tulong const m = (ulong(1) << index) - ulong(1);\n\
         \t\treturn popcnt(m & vert_ballot);\n\
         \t};\n\
         \ti0 = getCompactIndex64(i0);\n\
         \tPout o; o.c = float4(i0, 0, 0, 1); return o; }",
    );
    assert!(
        !hlsl.contains("getCompactIndex64"),
        "lambda must inline fully, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("auto ") && !hlsl.contains("__lambda"),
        "no auto or placeholder must leak, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("countbits"),
        "compaction body must survive at the call site, got:\n{hlsl}"
    );
}

#[test]
fn byvalue_capture_mutated_stays_loud() {
    // A by-value capture mutated between definition and call freezes
    // the old value: inlining the name would read the new one, so the
    // lambda must stay loud instead of miscompiling.
    let hlsl = emit(
        "struct Pout { float4 c : S_TARGET_OUTPUT0; };\n\
         Pout main(uint idx : TEXCOORD0) {\n\
         \tuint k = idx;\n\
         \tauto f = [k](uint v) -> uint { return v + k; };\n\
         \tk = 7u;\n\
         \tPout o; o.c = float4(f(idx), 0, 0, 1); return o; }",
    );
    assert!(
        !hlsl.contains("v + k") && !hlsl.contains("+ (k)"),
        "mutated capture must not inline (would read the new value), got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("__lambda") || hlsl.contains("f(idx)"),
        "unreduced call must stay loud (placeholder or dangling callee), got:\n{hlsl}"
    );
}

#[test]
fn stateless_functor_inlines() {
    // `isValid(i)` with an empty-struct predicate type inlines the
    // single-return `operator()` body (bc1 `AllValid`); the tag
    // parameter then drops via the empty-tag pass.
    let hlsl = emit(
        "struct AllValid { bool operator()(uint i) { return true; } };\n\
         float pick(AllValid pred, uint i, float x) { return pred(i) ? x : 0.0f; }\n\
         float4 main(float x : TEXCOORD0) : S_TARGET_OUTPUT {\n\
         \treturn float4(pick(AllValid(), 1u, x), 0, 0, 1);\n\
         }",
    );
    assert!(
        !hlsl.contains("AllValid()") && !hlsl.contains("pred("),
        "functor must inline away, got:\n{hlsl}"
    );
}

#[test]
fn empty_tag_params_drop() {
    // Empty tag structs (`AllValid()`) cannot initialize in HLSL: the
    // parameter (when unreferenced) and its pure arguments drop
    // together at every call site.
    let hlsl = emit(
        "struct AllValid {};\n\
         float pick(AllValid tag, float x) { return x; }\n\
         float4 main(float x : TEXCOORD0) : S_TARGET_OUTPUT {\n\
         \treturn float4(pick(AllValid(), x), 0, 0, 1);\n\
         }",
    );
    assert!(
        hlsl.contains("float pick(float x)"),
        "tag param must drop, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("pick(x)"),
        "tag arg must drop, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("AllValid()"),
        "empty init must not leak, got:\n{hlsl}"
    );
}

#[test]
fn template_struct_monomorphizes() {
    // `BcnEncodeSrt<uint2>`-style uses clone the struct with `T`
    // substituted in members; the template def drops once retargeted.
    let hlsl = emit(
        "template<typename T> struct Buf { RW_Texture2D<T> dst; float k; };\n\
         Buf<uint2> *srt : S_SRT_DATA;\n\
         float4 main(float2 uv : TEXCOORD0) : S_TARGET_OUTPUT {\n\
         \tsrt->dst[uint2(1, 2)] = float2(uv).xyxy;\n\
         \treturn float4(srt->k, 0, 0, 1);\n\
         }",
    );
    assert!(
        hlsl.contains("RWTexture2D<uint2>"),
        "member must substitute, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("RWTexture2D<T>"),
        "template residue must not leak, got:\n{hlsl}"
    );
}

#[test]
fn template_typename_param_monomorphizes() {
    // `template<typename T> float3 ident(const T& x)` clones per call
    // argument type; the template def drops once retargeted.
    let hlsl = emit(
        "template<typename T> float3 ident(const T& x) { return x; }\n\
         struct Pin { float3 p : TEXCOORD0; };\n\
         struct Pout { float4 c : S_TARGET_OUTPUT0; };\n\
         Pout main(Pin input) {\n\
         \tPout o; o.c = float4(ident(input.p), 1.0f); return o;\n\
         }",
    );
    assert!(
        hlsl.contains("ident__mono"),
        "call must retarget to a clone, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("const T&") && !hlsl.contains("(T "),
        "template residue must not leak, got:\n{hlsl}"
    );
}

#[test]
fn auto_param_monomorphizes() {
    // `void sink(auto x)`-style params deduce from call arguments.
    let hlsl = emit(
        "float take(auto x, float y) { return x + y; }\n\
         [NUM_THREADS(1,1,1)]\n\
         void main(uint position : S_DISPATCH_THREAD_ID) {\n\
         float y = take(0.5f, 1.0f);\n\
         }",
    );
    assert!(
        hlsl.contains("take__mono"),
        "call must retarget to a clone, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("auto x"),
        "auto param must not leak, got:\n{hlsl}"
    );
}

#[test]
fn explicit_bool_template_arg_substitutes() {
    // `template<bool B> f()` + `f<true>()` substitutes the literal
    // (synthetic value-param local takes the init).
    let hlsl = emit(
        "template<bool B> float pick(float x) { if (B) { return x; } return 0.0f; }\n\
         float4 main(float x : TEXCOORD0) : S_TARGET_OUTPUT {\n\
         \treturn float4(pick<true>(x), 0, 0, 1);\n\
         }",
    );
    assert!(
        hlsl.contains("pick__mono"),
        "call must retarget to a clone, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("pick<true>"),
        "explicit args must not leak, got:\n{hlsl}"
    );
}

#[test]
fn execute_in_pixel_order_inlines() {
    // `ExecuteInPixelOrder([&]() { ... });` splices the body in place
    // (definition and call are atomic, so captures bind correctly);
    // the ordering guarantee itself has no HLSL spelling (documented).
    let hlsl = emit(
        "struct V2P { float4 pos : S_POSITION; float4 color : TEXCOORD0; };\n\
         struct Srt { RW_Texture2D<float4> tex; };\n\
         Srt g_srt : S_SRT_DATA;\n\
         [CxxSymbol(\"Shader::ps\")]\n\
         void main(V2P input) {\n\
         \tfloat4 color = input.color;\n\
         \tExecuteInPixelOrder([&]() { float4 d = color; d = d.zyxw + color; });\n\
         }",
    );
    assert!(
        !hlsl.contains("ExecuteInPixelOrder") && !hlsl.contains("__lambda"),
        "critical section must splice, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("d.zyxw"),
        "lambda body must survive, got:\n{hlsl}"
    );
}

#[test]
fn ref_capture_lambda_beta_reduces() {
    // `auto s4 = [&ss](int y, int x) { return ...; };` with direct
    // calls inlines per call site (captures bind in the shared function
    // scope); by-value captures never do (frozen values).
    let hlsl = emit(
        "RWStructuredBuffer<uint> outBuf;\n\
         [NUM_THREADS(8,8,1)]\n\
         void main(uint2 tid : S_DISPATCH_THREAD_ID) {\n\
         \tuint ss = tid.x + tid.y;\n\
         \tauto s4 = [&ss](int y, int x) { return ss + uint(x + y); };\n\
         \toutBuf[0] = s4(1, 2);\n\
         }",
    );
    assert!(
        !hlsl.contains("s4(") && !hlsl.contains("__lambda"),
        "ref-capture lambda must inline, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("ss + uint"),
        "lambda body must survive, got:\n{hlsl}"
    );
}

#[test]
fn lambda_passed_as_value_stays_loud() {
    // A lambda passed as a value (ray-tracing any-hit callback) cannot
    // beta-reduce: it keeps the loud placeholder (compute_sun_shadows).
    let hlsl = emit(
        "void sink(auto f) {}\n\
         [NUM_THREADS(1,1,1)]\n\
         void main(uint position : S_DISPATCH_THREAD_ID) {\n\
         \tauto alphaTestLambda = [](uint v) { return v > 0u; };\n\
         \tsink(alphaTestLambda);\n\
         }",
    );
    assert!(
        hlsl.contains("__lambda"),
        "passed lambda must stay loud, got:\n{hlsl}"
    );
}

#[test]
fn auto_local_array_copy_stays_auto() {
    // Whole-array copies (`auto t = srt.transforms;` with an array
    // member) keep `auto`: the element type alone would mistype the
    // declaration, while the subobject materialization completes the
    // real array shape from the path (staticmesh_vv).
    let hlsl = emit(
        "struct T { float4 mvp; };\n\
         struct SRT { T transforms[2]; };\n\
         SRT *srt : S_SRT_DATA;\n\
         struct Vout { float4 pos : S_POSITION; };\n\
         Vout main(uint i : S_VERTEX_ID) {\n\
         \tauto transforms = srt->transforms;\n\
         \tVout o; o.pos = mul(transforms[0].mvp, float4(0, 0, 0, 1)); return o;\n\
         }",
    );
    assert!(
        !hlsl.contains("DeferredInputTransforms transforms = ")
            && !hlsl.contains("T transforms = srt"),
        "array copy must not become a scalar decl, got:\n{hlsl}"
    );
}

#[test]
fn auto_local_unknown_stays_loud() {
    // Unknowable inits (opaque framework calls) keep today's loud
    // `auto` rather than guessing.
    let hlsl = emit(
        "[NUM_THREADS(1,1,1)]\n\
         void main(uint position : S_DISPATCH_THREAD_ID) {\n\
         auto u = sce::Psr::NoIntersection();\n\
         position = u.x;\n\
         }",
    );
    assert!(
        hlsl.contains("auto u"),
        "unknowable autos must stay loud, got:\n{hlsl}"
    );
}
