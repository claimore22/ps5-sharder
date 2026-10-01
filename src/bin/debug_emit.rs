fn main() {
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
    let module = ps5_shader::parse_shader(source).expect("parse");
    let model = ps5_shader::semantic::analyze(&module).expect("semantic");
    let ir = ps5_shader::lower::lower_module(&module, &model).expect("lower");
    let hlsl = ps5_shader::emit::emit_vulkan_hlsl(&ir);
    println!("---HLSL---\n{}", hlsl);
}
