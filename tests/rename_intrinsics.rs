use ps5_shader::parse_shader;

// Regression tests for reserved-identifier rename consistency in
// texture-intrinsic lowering: `texture.GetDimensionsFast(...)` must
// rename the root (`texture_`) exactly as the resource declaration
// does. Previously the intrinsic path built the callee spelling from the
// raw PSSL name, emitting `texture.GetDimensions` against a declared
// `texture_` (DXC "dot operator on a type" / undeclared).

fn emit(source: &str) -> String {
    let module = parse_shader(source).expect("parser accepted shader");
    let model = ps5_shader::semantic::analyze(&module).expect("semantic analysis succeeded");
    let ir = ps5_shader::lower::lower_module(&module, &model).expect("lowering succeeded");
    ps5_shader::emit::emit_vulkan_hlsl(&ir)
}

const TEXTURE_NAME: &str = "Texture2D<float> texture;\n\
     [NUM_THREADS(1,1,1)]\n\
     void main(float2 uv : TEXCOORD0, float4 color : COLOR0) {\n\
     float2 tex_size;\n\
     texture.GetDimensionsFast(tex_size.x, tex_size.y);\n\
     color.a = texture[int2(uv * tex_size)];\n\
     }\n";

#[test]
fn texture_intrinsic_roots_rename_consistently() {
    let hlsl = emit(TEXTURE_NAME);
    assert!(
        hlsl.contains("Texture2D<float> texture_;"),
        "declaration must rename, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("texture_.GetDimensions("),
        "intrinsic root must rename to match, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("texture.GetDimensions"),
        "raw spelling must not survive intrinsic paths, got:\n{hlsl}"
    );
}