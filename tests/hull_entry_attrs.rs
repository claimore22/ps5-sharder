use ps5_shader::parse_shader;

// Regression tests for hull-shader entry prerequisites (tessellation
// family): PSSL tessellation attributes map to HLSL hull attributes,
// patch types keep their arguments, the patch-constant function pins
// against dead-code stripping, tessellation-factor semantics map to
// system values, and the implicit control-point ID materializes (DXC
// hs_6_0 requires it; PSSL leaves it implicit).

fn emit(source: &str) -> String {
    let module = parse_shader(source).expect("parser accepted shader");
    let model = ps5_shader::semantic::analyze(&module).expect("semantic analysis succeeded");
    let ir = ps5_shader::lower::lower_module(&module, &model).expect("lowering succeeded");
    ps5_shader::emit::emit_vulkan_hlsl(&ir)
}

const HULL: &str = "struct HS_IN { float2 pos : POSITION; };\n\
     struct HS_OUT { float2 pos : POSITION; };\n\
     struct HS_CONST { float edge[4] : S_EDGE_TESS_FACTOR; float inside[2] : S_INSIDE_TESS_FACTOR; };\n\
     HS_CONST ConstantsHS(OutputPatch<HS_IN, 1> patch) {\n\
     HS_CONST o; o.edge[0] = 1.0; o.inside[0] = 1.0; return o; }\n\
     [DOMAIN_PATCH_TYPE(\"quad\")]\n\
     [PARTITIONING_TYPE(\"integer\")]\n\
     [OUTPUT_TOPOLOGY_TYPE(\"triangle_cw\")]\n\
     [OUTPUT_CONTROL_POINTS(1)]\n\
     [PATCH_CONSTANT_FUNC(\"ConstantsHS\")]\n\
     [MAX_TESS_FACTOR(64)]\n\
     HS_OUT main(InputPatch<HS_IN, 1> patch) {\n\
     HS_OUT o; o.pos = patch[0].pos; return o; }\n";

#[test]
fn hull_patch_types_keep_arguments() {
    let hlsl = emit(HULL);
    assert!(
        hlsl.contains("InputPatch<HS_IN, 1>"),
        "input patch must keep element and count, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("OutputPatch<HS_IN, 1>"),
        "patch-constant input must keep arguments, got:\n{hlsl}"
    );
}

#[test]
fn hull_tess_attributes_map() {
    let hlsl = emit(HULL);
    for attr in [
        "[domain(\"quad\")]",
        "[partitioning(\"integer\")]",
        "[outputtopology(\"triangle_cw\")]",
        "[outputcontrolpoints(1)]",
        "[patchconstantfunc(\"ConstantsHS\")]",
        "[maxtessfactor(64.0)]",
    ] {
        assert!(hlsl.contains(attr), "missing {attr}, got:\n{hlsl}");
    }
}

#[test]
fn hull_patch_constant_func_survives() {
    let hlsl = emit(HULL);
    assert!(
        hlsl.contains("HS_CONST ConstantsHS("),
        "patch-constant function must not strip, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("SV_TessFactor") && hlsl.contains("SV_InsideTessFactor"),
        "tessfactor semantics must map to system values, got:\n{hlsl}"
    );
}

#[test]
fn hull_control_point_id_materializes() {
    let hlsl = emit(HULL);
    assert!(
        hlsl.contains("SV_OutputControlPointID"),
        "control-point ID input must materialize, got:\n{hlsl}"
    );
}
