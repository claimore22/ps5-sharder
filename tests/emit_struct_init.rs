use ps5_shader::parse_shader;

// Regression tests for struct brace-init lowering (DXC `too few elements
// in vector initialization` / `cannot initialize return object of type
// 'V2P' with an lvalue of type 'V2P [1]'`): HLSL struct initialization
// needs a complete member list and never takes `[]`, so `S s = {...}`
// desugars to memberwise assignment with zero-filled trailing members
// (C semantics; HLSL does not zero-init locals).
//
// (Locals are consumed below: unread locals are dead code and drop in
// lowering.)

fn emit(source: &str) -> String {
    let module = parse_shader(source).expect("parser accepted shader");
    let model = ps5_shader::semantic::analyze(&module).expect("semantic analysis succeeded");
    let ir = ps5_shader::lower::lower_module(&module, &model).expect("lowering succeeded");
    ps5_shader::emit::emit_vulkan_hlsl(&ir)
}

const PRELUDE: &str = "struct V2P { float4 pos : S_POSITION; uint frame : TEXCOORD0; float2 uv : TEXCOORD1; };\n";

#[test]
fn partial_struct_init_desugars_to_memberwise_with_zerofill() {
    let hlsl = emit(&format!(
        "{PRELUDE}void sink(float x) {{}}\n[NUM_THREADS(1,1,1)]\n\
         void main(uint2 position : S_DISPATCH_THREAD_ID) {{\n\
         float4 pos = float4(0, 0, 0, 1);\n\
         V2P res = {{pos, position.x}};\n\
         sink(res.pos.x + res.uv.x);\n\
         }}",
    ));
    assert!(
        hlsl.contains("V2P res;"),
        "must declare without [] or init, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("res.pos = pos;"),
        "member assign must be emitted, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("res.uv = 0;"),
        "missing member must zero-fill, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("res[]"),
        "no array dimension on struct init, got:\n{hlsl}"
    );
}

#[test]
fn complete_struct_init_desugars_without_brackets() {
    let hlsl = emit(&format!(
        "{PRELUDE}void sink(float x) {{}}\n[NUM_THREADS(1,1,1)]\n\
         void main(uint2 position : S_DISPATCH_THREAD_ID) {{\n\
         V2P res = {{float4(0, 0, 0, 1), position.x, position}};\n\
         sink(res.frame + res.uv.x);\n\
         }}",
    ));
    assert!(
        hlsl.contains("V2P res;"),
        "must declare without init, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("res.frame = position.x;"),
        "member assign must be emitted, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("res[]"),
        "no array dimension on struct init, got:\n{hlsl}"
    );
}

#[test]
fn array_brace_init_keeps_brackets() {
    // Plain array brace-init is unaffected by the struct rule.
    let hlsl = emit(
        "void sink(float x) {}\n[NUM_THREADS(1,1,1)]\n\
         void main(uint2 position : S_DISPATCH_THREAD_ID) {\n\
         float k[4] = {0.0, 0.5, 1.0, 1.5};\n\
         sink(k[0] + k[3]);\n\
         }",
    );
    assert!(
        hlsl.contains("float k[4]"),
        "array init must keep its dimension, got:\n{hlsl}"
    );
}

#[test]
fn array_constructor_sugar_becomes_init_list() {
    // `float3 colors[] = float3[](...)` is an unsized array init (the
    // size comes from the list; HLSL accepts `T x[] = {...}`). The old
    // `false` placeholder broke the init type (ibl metallic).
    let hlsl = emit(
        "RWStructuredBuffer<float> outBuf;\n\
         [NUM_THREADS(1,1,1)]\n\
         void main(uint i : S_DISPATCH_THREAD_ID) {\n\
         \tfloat3 colors[] = float3[](float3(0.3, 0, 0), float3(0, 0, 0.3));\n\
         \toutBuf[0] = colors[i % 2].x;\n\
         }",
    );
    assert!(
        hlsl.contains("float3 colors[] = "),
        "array init must survive, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("= false"),
        "placeholder must not leak, got:\n{hlsl}"
    );
}

#[test]
fn excess_constructor_components_truncate() {
    // `uint2(a, b_vec2)` takes the first 2 scalar components (scalar
    // conversion takes `.x`; arrays flatten element-wise); HLSL rejects
    // the excess outright (ibl metallic).
    let hlsl = emit(
        "RWStructuredBuffer<uint> outBuf;\n\
         uint4 main(uint2 i : S_DISPATCH_THREAD_ID) : S_TARGET_OUTPUT {\n\
         \tuint2 v = uint2(i.x, i);\n\
         \toutBuf[0] = v.x;\n\
         \treturn v;\n\
         }",
    );
    assert!(
        hlsl.contains("uint2(i.x, i.x)") || hlsl.contains("uint2((i.x), (i.x))"),
        "excess must truncate to first-N, got:\n{hlsl}"
    );
}

#[test]
fn forward_decl_loses_to_real_definition() {
    // `struct ExposureSRT;` (empty forward declaration) plus a real
    // definition under the same name must emit the definition
    // (exposure shaders); otherwise member uses dangle. All-empty
    // duplicates still keep the first.
    let hlsl = emit(
        "struct ExposureSRT;\n\
         struct ExposureSRT { float m_exposureLevel; };\n\
         [NUM_THREADS(1,1,1)]\n\
         void main(uint position : S_DISPATCH_THREAD_ID) {\n\
         ExposureSRT e; e.m_exposureLevel = 1.0f;\n\
         }",
    );
    assert!(
        hlsl.contains("float m_exposureLevel;"),
        "real definition must win, got:\n{hlsl}"
    );
}

#[test]
fn forward_referenced_structs_emit_in_dependency_order() {
    // PSSL headers forward-reference structs (`TransformedNodeData`
    // holds `MeshData*` before `MeshData` is defined); HLSL needs
    // dependencies first, so definitions topologically sort at emit
    // (alpha_tested_geometry pixel_shaders).
    let hlsl = emit(
        "struct TransformedNodeData { float4x4 model; MeshData mesh; };\n\
         struct MeshData { uint attribute_stride; };\n\
         [NUM_THREADS(1,1,1)]\n\
         void main(uint position : S_DISPATCH_THREAD_ID) {\n\
         TransformedNodeData t; t.mesh.attribute_stride = position;\n\
         }",
    );
    let mesh_at = hlsl.find("struct MeshData {").expect("MeshData emitted");
    let use_at = hlsl.find("MeshData mesh;").expect("use emitted");
    assert!(
        mesh_at < use_at,
        "dependency must precede use, got:\n{hlsl}"
    );
}
