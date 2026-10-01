use ps5_shader::parse_shader;

// Regression tests for `sce::Gnm::Sampler` (GNM sampler descriptors lower
// to externally-bound `SamplerState`) and struct-constructor returns.

fn emit(source: &str) -> String {
    let module = parse_shader(source).expect("parser accepted shader");
    let model = ps5_shader::semantic::analyze(&module).expect("semantic analysis succeeded");
    let ir = ps5_shader::lower::lower_module(&module, &model).expect("lowering succeeded");
    ps5_shader::emit::emit_vulkan_hlsl(&ir)
}

#[test]
fn gnm_sampler_array_declares_sampler_state() {
    let hlsl = emit(
        "struct BuiltinSamplers { SamplerState a; SamplerState b; };\n\
         BuiltinSamplers getBuiltinSamplers() {\n\
         sce::Gnm::Sampler gnmSampler[9];\n\
         gnmSampler[1] = gnmSampler[0];\n\
         return BuiltinSamplers(SamplerState(gnmSampler[0]), SamplerState(gnmSampler[1]));\n\
         }\n\
         [NUM_THREADS(1,1,1)]\n\
         void main(uint i : S_DISPATCH_THREAD_ID) {\n\
         BuiltinSamplers s = getBuiltinSamplers();\n\
         }",
    );
    assert!(
        hlsl.contains("SamplerState gnmSampler[9]"),
        "Gnm sampler array must declare SamplerState, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("sce::"),
        "no framework residue may leak, got:\n{hlsl}"
    );
}

#[test]
fn struct_constructor_return_desugars() {
    let hlsl = emit(
        "struct Pair { float x; float y; };\n\
         Pair make() { return Pair(1.0, 2.0); }\n\
         [NUM_THREADS(1,1,1)]\n\
         void main(uint i : S_DISPATCH_THREAD_ID) {\n\
         Pair p = make();\n\
         }",
    );
    assert!(
        hlsl.contains("__retinit_0;"),
        "return must desugar to a temporary, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("__retinit_0.x = 1.0"),
        "members must assign through the temporary, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("return __retinit_0;"),
        "temporary must be returned, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("return Pair("),
        "struct construction must be gone, got:\n{hlsl}"
    );
}

#[test]
fn htile_constant_resolves() {
    // (Consumed through a sink: unread locals drop in lowering.)
    let hlsl = emit(
        "void sink(uint x) {}\n\
         [NUM_THREADS(1,1,1)]\n\
         void main(uint i : S_DISPATCH_THREAD_ID) {\n\
         uint w = sce::Agc::Constants::kHtileWidthAndHeight;\n\
         sink(w);\n\
         }",
    );
    assert!(
        hlsl.contains("= 8"),
        "HTILE constant must resolve to 8, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("sce::"),
        "no framework residue may leak, got:\n{hlsl}"
    );
}

#[test]
fn matrix_narrow_selects_upper_left() {
    let hlsl = emit(
        "struct M4 { float4x4 m; };\n\
         M4 g_m4;\n\
         float m() {\n\
         float3 n = mul(float3x3(g_m4.m), float3(0.0, 1.0, 0.0));\n\
         return n.x;\n\
         }\n\
         [NUM_THREADS(1,1,1)]\n\
         void main(uint i : S_DISPATCH_THREAD_ID) {\n\
         float x = m();\n\
         }",
    );
    assert!(
        hlsl.contains("float3x3(g_m4.m[0].xyz, g_m4.m[1].xyz, g_m4.m[2].xyz)"),
        "matrix narrowing must select rows explicitly, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("float3x3(g_m4.m)"),
        "bare matrix cast must be gone, got:\n{hlsl}"
    );
}

#[test]
fn nested_array_init_vectorizes_rows() {
    let hlsl = emit(
        "float rowSum(float x, float w) {\n\
         const float M[4][4] = {{-1.0, 3.0, -3.0, 1.0},\n\
         {3.0, -6.0, 3.0, 0.0},\n\
         {-3.0, 0.0, 3.0, 0.0},\n\
         {1.0, 4.0, 1.0, 0.0}};\n\
         return M[0][0] + M[3][3];\n\
         }\n\
         [NUM_THREADS(1,1,1)]\n\
         void main(uint i : S_DISPATCH_THREAD_ID) {\n\
         float y = rowSum(0.5, 1.0);\n\
         }",
    );
    assert!(
        hlsl.contains("float4 M[4]"),
        "nested init must become a vector array, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("M[0][0]"),
        "double indexing must survive on the vector array, got:\n{hlsl}"
    );
}

#[test]
fn struct_zero_cast_expands_memberwise() {
    let hlsl = emit(
        "struct PBRInfo { float x; float y; };\n\
         PBRInfo f() {\n\
         PBRInfo ret = (PBRInfo)0;\n\
         return ret;\n\
         }\n\
         [NUM_THREADS(1,1,1)]\n\
         void main(uint i : S_DISPATCH_THREAD_ID) { PBRInfo r = f(); }",
    );
    assert!(
        hlsl.contains("PBRInfo ret;"),
        "zero-cast must become a bare declaration, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("ret.x = 0"),
        "members must zero-assign, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("= PBRInfo"),
        "struct construction must be gone, got:\n{hlsl}"
    );
}

#[test]
fn struct_bare_zero_init_expands_memberwise() {
    // `VS_OUT output = 0;` (sprite/particle vertex shaders): DXC rejects
    // `literal int` struct init, so the bare zero desugars exactly like
    // the `(S)0` cast form. Scalar `uint x = 0;` must not match.
    let hlsl = emit(
        "struct VS_OUT { float4 pos; float2 uv; };\n\
         VS_OUT f(uint z) {\n\
         VS_OUT output = 0;\n\
         uint w = 0;\n\
         output.pos = float4(w + z, 2, 3, 4);\n\
         return output;\n\
         }\n\
         [NUM_THREADS(1,1,1)]\n\
         void main(uint i : S_DISPATCH_THREAD_ID) { VS_OUT o = f(i); }",
    );
    assert!(
        hlsl.contains("VS_OUT output;"),
        "bare zero must become a bare declaration, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("output.pos = 0") && hlsl.contains("output.uv = 0"),
        "members must zero-assign, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("uint w = 0"),
        "scalar zero init must pass through, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("output = 0"),
        "no struct literal-zero init may remain, got:\n{hlsl}"
    );
}

#[test]
fn struct_empty_list_init_expands_memberwise() {
    // `S name = {};` (C++ value-init) desugars like the zero forms;
    // numeric arrays fall back to `{ 0 }` (missing elements default).
    let hlsl = emit(
        "struct VS_OUT { float4 pos; float2 uv; };\n\
         VS_OUT f(uint z) {\n\
         VS_OUT output = {};\n\
         uint w[1] = {};\n\
         output.pos = float4(w[0] + z, 2, 3, 4);\n\
         return output;\n\
         }\n\
         [NUM_THREADS(1,1,1)]\n\
         void main(uint i : S_DISPATCH_THREAD_ID) { VS_OUT o = f(i); }",
    );
    assert!(
        hlsl.contains("VS_OUT output;"),
        "empty list must become a bare declaration, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("output.pos = 0") && hlsl.contains("output.uv = 0"),
        "members must zero-assign, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("uint w[1] = { 0 }"),
        "numeric empty list must zero-fill, got:\n{hlsl}"
    );
}

#[test]
fn numeric_empty_array_init_fills_zeros() {
    // `uint a[2] = {};` needs all elements (DXC rejects partial/empty
    // aggregate inits); missing elements would default to zero anyway.
    let hlsl = emit(
        "void sink(uint x) {}\n\
         [NUM_THREADS(1,1,1)]\n\
         void main(uint i : S_DISPATCH_THREAD_ID) {\n\
         uint a[2] = {};\n\
         a[i] = 1u;\n\
         sink(a[0] + a[1]);\n}",
    );
    assert!(
        hlsl.contains("uint a[2] = { 0, 0 }"),
        "array empty list must fill, got:\n{hlsl}"
    );
    assert!(!hlsl.contains("{  }"), "no empty list may remain, got:\n{hlsl}");
}

#[test]
fn default_args_fill_at_call_site() {
    let hlsl = emit(
        "struct Cfg { float k; };\n\
         Cfg g_cfg;\n\
         float shape(float x, Cfg c = g_cfg) { return x + c.k; }\n\
         [NUM_THREADS(1,1,1)]\n\
         void main(uint i : S_DISPATCH_THREAD_ID) {\n\
         float y = shape(0.5);\n\
         }",
    );
    assert!(
        hlsl.contains("shape(0.5, g_cfg)"),
        "missing trailing default must be filled, got:\n{hlsl}"
    );
}

#[test]
fn bare_pi_resolves_to_literal() {
    // (Consumed through a sink: unread locals drop in lowering.)
    let hlsl = emit(
        "void sink(float x) {}\n\
         [NUM_THREADS(1,1,1)]\n\
         void main(uint i : S_DISPATCH_THREAD_ID) {\n\
         float a = 2.0 * PI;\n\
         sink(a);\n\
         }",
    );
    assert!(
        hlsl.contains("3.1415926536f"),
        "bare PI must resolve to pi, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains(" PI"),
        "no PI identifier may leak, got:\n{hlsl}"
    );
}

#[test]
fn local_pi_shadows_builtin() {
    let hlsl = emit(
        "float f(float3 a, float3 b) {\n\
         const float PI = 3.141593f;\n\
         return PI - 1.0;\n\
         }\n\
         [NUM_THREADS(1,1,1)]\n\
         void main(uint i : S_DISPATCH_THREAD_ID) {\n\
         float y = f(float3(0,0,0), float3(1,0,0));\n\
         }",
    );
    assert!(
        hlsl.contains("PI - 1.0") || hlsl.contains("PI-1.0"),
        "local PI must survive as a variable, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("3.1415926536f"),
        "builtin must not leak into shadowing scope, got:\n{hlsl}"
    );
}
