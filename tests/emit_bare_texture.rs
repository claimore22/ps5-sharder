use ps5_shader::parse_shader;

// Regression tests for bare texture struct members: PSSL allows
// `Texture2D tex` with no `<element>` (e.g. `FontSrt::fontTexture` in
// jobified_render font_p). They are GPU resources, not int-backed
// enums, so the struct must keep a `Texture2D` member (HLSL's own
// float4 default) instead of flattening to `int`.

fn emit(source: &str) -> String {
    let module = parse_shader(source).expect("parser accepted shader");
    let model = ps5_shader::semantic::analyze(&module).expect("semantic analysis succeeded");
    let ir = ps5_shader::lower::lower_module(&module, &model).expect("lowering succeeded");
    ps5_shader::emit::emit_vulkan_hlsl(&ir)
}

#[test]
fn bare_texture_member_emits_texture_not_int() {
    let hlsl = emit(
        "struct FontSrt { Texture2D fontTexture; SamplerState bilinearSampler; };\n\
         FontSrt *srt : S_SRT_DATA;\n\
         float4 main(float2 uv : TEXCOORD0) : S_TARGET_OUTPUT {\n\
         \tfloat4 font = srt->fontTexture.Sample(srt->bilinearSampler, uv);\n\
         \treturn font;\n\
         }",
    );
    assert!(
        hlsl.contains("Texture2D fontTexture;"),
        "bare texture member must stay a texture, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("int fontTexture;"),
        "bare texture member must not flatten to int, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains(".Sample("),
        "Sample use through the member must survive, got:\n{hlsl}"
    );
}
