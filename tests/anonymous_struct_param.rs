use ps5_shader::parse_shader;

// Regression tests for anonymous inline struct parameters
// (`f(struct { uint mip; uint slice; } srt)`): the body must register as
// a real named struct so the param type and member accesses (`srt.mip`)
// resolve, instead of the opaque `anonymous_struct` placeholder (DXC
// "unknown type name 'anonymous_struct'").

fn emit(source: &str) -> String {
    let module = parse_shader(source).expect("parser accepted shader");
    let model = ps5_shader::semantic::analyze(&module).expect("semantic analysis succeeded");
    let ir = ps5_shader::lower::lower_module(&module, &model).expect("lowering succeeded");
    ps5_shader::emit::emit_vulkan_hlsl(&ir)
}

const ANON: &str = "RWBuffer<uint4> dst;\n\
     [NUM_THREADS(1,1,1)]\n\
     void main(struct { uint mip; uint slice; } srt : S_SRT_DATA) {\n\
     uint a = srt.mip;\n\
     dst[0] = uint4(a, srt.slice, 0, 1);\n\
     }\n";

#[test]
fn anonymous_struct_param_registers_fields() {
    let hlsl = emit(ANON);
    assert!(
        !hlsl.contains("anonymous_struct"),
        "placeholder must not survive, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("mip"),
        "member must resolve, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("srt.mip"),
        "member access must emit, got:\n{hlsl}"
    );
}