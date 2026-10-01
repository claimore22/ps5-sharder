use ps5_shader::parse_shader;

// Regression tests for scoped static constants in attributes
// (geometry shaders: `[MAX_VERTEX_COUNT((S::kMax - 1) * 4)]` must fold
// to `[maxvertexcount(N)]`; DXC rejects entries without the bound).
// `S::m` qualification is only legal on statics, so recorded literals
// are the statics' values; unresolvable shapes stay loud.

fn emit(source: &str) -> String {
    let module = parse_shader(source).expect("parser accepted shader");
    let model = ps5_shader::semantic::analyze(&module).expect("semantic analysis succeeded");
    let ir = ps5_shader::lower::lower_module(&module, &model).expect("lowering succeeded");
    ps5_shader::emit::emit_vulkan_hlsl(&ir)
}

const SCOPED_MAX: &str = "struct Particle {\n\
     static const uint kMaxGens = 10;\n\
     float4 pos;\n\
     };\n\
     struct GS_OUT {\n\
     float4 p : S_POSITION;\n\
     };\n\
     [MAX_VERTEX_COUNT((Particle::kMaxGens - 1) * 4)]\n\
     void main(inout TriangleStream<GS_OUT> stream, uint id : S_INSTANCE_ID) {\n\
     GS_OUT o; o.p = float4(id, 0, 0, 1);\n\
     stream.Append(o);\n\
     }\n";

#[test]
fn scoped_static_max_vertex_count_folds() {
    let hlsl = emit(SCOPED_MAX);
    assert!(
        hlsl.contains("[maxvertexcount(36)]"),
        "scoped const arithmetic must fold to 36, got:\n{hlsl}"
    );
}

const BARE_MAX: &str = "static const uint kMaxGens = 10;\n\
     struct GS_OUT {\n\
     float4 p : S_POSITION;\n\
     };\n\
     [MAX_VERTEX_COUNT((kMaxGens - 1) * 4)]\n\
     void main(inout TriangleStream<GS_OUT> stream, uint id : S_INSTANCE_ID) {\n\
     GS_OUT o; o.p = float4(id, 0, 0, 1);\n\
     stream.Append(o);\n\
     }\n";

#[test]
fn bare_const_max_vertex_count_still_folds() {
    // Unqualified constants keep working (no regression from scoping).
    let hlsl = emit(BARE_MAX);
    assert!(
        hlsl.contains("[maxvertexcount(36)]"),
        "bare const arithmetic must still fold, got:\n{hlsl}"
    );
}
