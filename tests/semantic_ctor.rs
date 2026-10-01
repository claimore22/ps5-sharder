use ps5_shader::parse_shader;
use ps5_shader::semantic;

// Regression test for the largest semantic error class at the start of the
// semantics campaign (`type mismatch: expected float3, found float3`):
// builtin constructors must resolve to real vector types so binary-op
// checking sees identical types on both sides.
#[test]
fn semantic_accepts_builtin_ctor_in_binary_op() {
    let source = r#"
        float3 ReinhardTonemap(float3 orig) {
            orig = clamp(orig, 0.0, (1.0/0.0));
            return orig / (float3(1.0) + orig);
        }
    "#;
    let module = parse_shader(source).expect("parser accepted shader");
    semantic::analyze(&module).expect("semantic analysis succeeded");
}

fn emit(source: &str) -> String {
    let module = parse_shader(source).expect("parser accepted shader");
    let model = semantic::analyze(&module).expect("semantic analysis succeeded");
    let ir = ps5_shader::lower::lower_module(&module, &model).expect("lowering succeeded");
    ps5_shader::emit::emit_vulkan_hlsl(&ir)
}

// C++ paren initialization (`float2 texcoord(a, b)` in loop bodies,
// bloom shaders) keeps exact constructor values: the legacy skip
// synthesized `0` (a silent wrong value), and dropped dependents.
#[test]
fn paren_init_keeps_constructor_values() {
    let hlsl = emit(
        "struct Pout { float4 c : S_TARGET_OUTPUT0; };\n\
         Pout main(float4 position : S_POSITION) {\n\
         \tfloat3 output = 0;\n\
         \tfor (float offset = -7.0f; offset <= 7.0f; offset += 2.f) {\n\
         \t\tfloat blendFactor = 0.5f;\n\
         \t\tfloat2 texcoord(offset + position.x + blendFactor, position.y);\n\
         \t\toutput += float3(texcoord.x, texcoord.y, 0);\n\
         \t}\n\
         \tPout o; o.c = float4(output, 1); return o; }",
    );
    assert!(
        hlsl.contains("float2 texcoord = float2("),
        "paren-init must keep the constructor, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("blendFactor"),
        "constructor dependents must survive, got:\n{hlsl}"
    );
}
