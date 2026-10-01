use ps5_shader::parse_shader;

// Regression tests for the NGG cull-data builtin (`Ngg::CullData<T>`):
// `Set` drops (no Vulkan effect), `GetTriangle` becomes a zero `T[3]`
// (culling conservatively disabled — zeroed data never culls).

fn emit(source: &str) -> String {
    let module = parse_shader(source).expect("parser accepted shader");
    let model = ps5_shader::semantic::analyze(&module).expect("semantic analysis succeeded");
    let ir = ps5_shader::lower::lower_module(&module, &model).expect("lowering succeeded");
    ps5_shader::emit::emit_vulkan_hlsl(&ir)
}

#[test]
fn culldata_set_statement_drops() {
    let hlsl = emit(
        "struct V { float2 p; };\n\
         V g_v;\n\
         [NUM_THREADS(1,1,1)]\n\
         void main(uint i : S_DISPATCH_THREAD_ID) {\n\
         V v = g_v;\n\
         Ngg::CullData<V>::Set(v);\n\
         }",
    );
    assert!(
        !hlsl.contains("CullData"),
        "Set must be elided, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("Ngg"),
        "no Ngg residue may leak, got:\n{hlsl}"
    );
}

#[test]
fn culldata_get_triangle_auto_becomes_zero_struct_array() {
    let hlsl = emit(
        "struct MyCullVertex { float2 m_projectedXY; uint m_clipFlags; };\n\
         bool culled(MyCullVertex tri[3]) { return tri[0].m_clipFlags != 0; }\n\
         bool MyCullFunc() {\n\
         auto tri = Ngg::CullData<MyCullVertex>::GetTriangle();\n\
         return culled(tri);\n\
         }\n\
         [NUM_THREADS(1,1,1)]\n\
         void main(uint i : S_DISPATCH_THREAD_ID) {\n\
         bool c = MyCullFunc();\n\
         }",
    );
    assert!(
        hlsl.contains("MyCullVertex tri[3]"),
        "auto must resolve to an explicit T[3], got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("tri[0].m_projectedXY = float2(0, 0)"),
        "struct elements must zero memberwise (positional struct \
         construction is invalid HLSL), got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("tri[3] = {"),
        "expansion must splice into the parent scope, not a braced \
         init or nested block, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("GetTriangle"),
        "builtin must be gone, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("Ngg"),
        "no Ngg residue may leak, got:\n{hlsl}"
    );
}

#[test]
fn culldata_get_triangle_float4_keeps_declared_type() {
    let hlsl = emit(
        "bool visible(float4 a, float4 b, float4 c) { return (a.x + b.y + c.z) > 0.0f; }\n\
         bool cullFunc() {\n\
         float4 triangle_arr[3] = Ngg::CullData<float4>::GetTriangle();\n\
         return visible(triangle_arr[0], triangle_arr[1], triangle_arr[2]);\n\
         }\n\
         [NUM_THREADS(1,1,1)]\n\
         void main(uint i : S_DISPATCH_THREAD_ID) {\n\
         bool c = cullFunc();\n\
         }",
    );
    assert!(
        hlsl.contains("float4 triangle_arr[3];"),
        "declared array must become a bare declaration, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("triangle_arr[0] = float4(0, 0, 0, 0)"),
        "elements must zero-assign, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("Ngg"),
        "no Ngg residue may leak, got:\n{hlsl}"
    );
}
