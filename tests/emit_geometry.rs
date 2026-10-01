use ps5_shader::parse_shader;

// Regression tests for traditional geometry shaders (DXC `compile
// unsupported for stage geometry`): `TriangleBuffer<T>` stream params,
// `Point T x` input topology, `inout` directions, and
// `[MAX_VERTEX_COUNT(N)]` all lower to gs_6_0 HLSL shapes.

fn emit(source: &str) -> String {
    let module = parse_shader(source).expect("parser accepted shader");
    let model = ps5_shader::semantic::analyze(&module).expect("semantic analysis succeeded");
    let ir = ps5_shader::lower::lower_module(&module, &model).expect("lowering succeeded");
    ps5_shader::emit::emit_vulkan_hlsl(&ir)
}

#[test]
fn gs_stream_signature_emits() {
    let hlsl = emit(
        "struct GS_OUTPUT { float4 position : S_POSITION; };\n\
         struct ES_OUTPUT { float4 position : S_POSITION; };\n\
         struct SRT { float2 spriteSize; };\n\
         [CxxSymbol(\"Shader::gs\")]\n\
         [MAX_VERTEX_COUNT(4)]\n\
         void main(inout TriangleBuffer<GS_OUTPUT> TriStream, Point ES_OUTPUT input[1]) {\n\
         GS_OUTPUT output;\n\
         TriStream.Append(output);\n\
         }\n",
    );
    assert!(
        hlsl.contains("[maxvertexcount(4)]"),
        "maxvertexcount must emit, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("inout TriangleStream<GS_OUTPUT> TriStream"),
        "stream param must map, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("point ES_OUTPUT input[1]"),
        "point primitive must emit lowercase, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("TriStream.Append(output);"),
        "Append must survive, got:\n{hlsl}"
    );
}

#[test]
fn max_vertex_count_drives_geometry_stage() {
    let module = parse_shader(
        "[MAX_VERTEX_COUNT(4)]\n\
         void main(Point float4 input[1]) {\n\
         }",
    )
    .unwrap();
    let model = ps5_shader::semantic::analyze(&module).unwrap();
    assert_eq!(
        model.stage,
        Some(ps5_shader::ir::Stage::Geometry),
        "MAX_VERTEX_COUNT must imply geometry stage"
    );
}

#[test]
fn gs_primitive_captured_lowercase() {
    let module = parse_shader(
        "void main(Point float4 input[1]) {\n\
         }",
    )
    .unwrap();
    let param = &module.functions[0].parameters[0];
    assert_eq!(param.gs_primitive.as_deref(), Some("point"));
}

#[test]
fn unresolvable_max_vertex_count_stays_loud() {
    // A non-constant bound cannot become `[maxvertexcount]`; the attribute
    // must not fabricate a value.
    let hlsl = emit(
        "[MAX_VERTEX_COUNT(kUnknownConst)]\n\
         void main(Point float4 input[1]) {\n\
         }",
    );
    assert!(
        !hlsl.contains("maxvertexcount"),
        "unresolvable bound must stay loud, got:\n{hlsl}"
    );
}

#[test]
fn max_vertex_count_folds_const_arithmetic() {
    // `static const uint NUM_VERTS = 16` + `[MAX_VERTEX_COUNT(NUM_VERTS
    // * 3)]` folds like array sizes do.
    let hlsl = emit(
        "static const uint NUM_VERTS = 16;\n\
         [MAX_VERTEX_COUNT(NUM_VERTS * 3)]\n\
         void main(Point float4 input[1]) {\n\
         }",
    );
    assert!(
        hlsl.contains("[maxvertexcount(48)]"),
        "const arithmetic must fold, got:\n{hlsl}"
    );
}
