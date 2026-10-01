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
    println!("AST: {:#?}", module);
}
