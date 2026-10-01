use ps5_shader::emit::emit_vulkan_hlsl;
use ps5_shader::lower::lower_module;
use ps5_shader::parse_shader;
use ps5_shader::semantic;

fn unit_pipeline(source: &str) -> String {
    let module = parse_shader(source).expect("parser accepted shader");
    let model = semantic::analyze(&module).expect("semantic analysis succeeded");
    let ir = lower_module(&module, &model).expect("lowering succeeded");
    emit_vulkan_hlsl(&ir)
}

#[test]
fn parses_and_emits_basic_pssl() {
    let source = r#"
        struct V2P
        {
            float4 pos : S_POSITION;
            float3 color : COLOR;
        };

        [CxxSymbol("Shader::ps")]
        float4 main(V2P input) : S_TARGET_OUTPUT0
        {
            return float4(input.color, 1.0);
        }
    "#;

    let hlsl = unit_pipeline(source);

    assert!(hlsl.contains("SV_Position"));
    assert!(hlsl.contains("SV_Target0"));
    assert!(hlsl.contains("return float4(input.color, 1.0);"));
}
