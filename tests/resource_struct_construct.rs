use ps5_shader::parse_shader;

// Regression tests for resource-holding struct construction
// (`MatData(texA, texB)` → `MatData tmp = {texA, texB}; f(tmp)`).
// HLSL rejects function-style construction for structs, but brace-init
// of resource members plus by-value passing is legal (DXC-verified with
// valid SPIR-V). Only fires for ctor-less aggregates with full arity
// (a present user constructor could carry logic); anything else stays
// loud. Memberwise-assignment expansion would break `const`/resource
// members, so full inits emit direct braces.

fn emit(source: &str) -> String {
    let module = parse_shader(source).expect("parser accepted shader");
    let model = ps5_shader::semantic::analyze(&module).expect("semantic analysis succeeded");
    let ir = ps5_shader::lower::lower_module(&module, &model).expect("lowering succeeded");
    ps5_shader::emit::emit_vulkan_hlsl(&ir)
}

const RESSTRUCT: &str = "Texture2D<float4> g_a;\n\
     Texture2D<float4> g_b;\n\
     SamplerState g_smp;\n\
     struct MatData { Texture2D<float4> t0; Texture2D<float4> t1; };\n\
     float4 useit(MatData m, float2 uv) { return m.t0.SampleLevel(g_smp, uv, 0) + m.t1.SampleLevel(g_smp, uv, 0); }\n\
     [NUM_THREADS(1,1,1)]\n\
     void main(uint i : S_DISPATCH_THREAD_ID) {\n\
     float4 c = useit(MatData(g_a, g_b), float2(0.5, 0.5));\n\
     }\n";

#[test]
fn resource_struct_construction_braces() {
    let hlsl = emit(RESSTRUCT);
    assert!(
        !hlsl.contains("MatData(g_a"),
        "function-style struct construction must desugar, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("MatData __struct_tmp"),
        "temporary must materialize, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("useit(__struct_tmp"),
        "call must retarget to temp, got:\n{hlsl}"
    );
}
