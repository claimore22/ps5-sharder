use ps5_shader::parse_shader;

// Regression tests for vector-param narrowing: a vector-typed parameter
// provably holding scalar values (every call site passes a scalar that
// splats) narrows to scalar, fixing vector-typed array indices
// (`buf[base + offset]` with `int2 offset` is illegal HLSL; scalar index
// is legal). Splat invariance makes this exact (all lanes equal the
// scalar). Non-scalar args, vector-syntax uses (`P.m`, `P[i]`, direct
// call args), or missing call sites decline (stay loud).

fn emit(source: &str) -> String {
    let module = parse_shader(source).expect("parser accepted shader");
    let model = ps5_shader::semantic::analyze(&module).expect("semantic analysis succeeded");
    let ir = ps5_shader::lower::lower_module(&module, &model).expect("lowering succeeded");
    ps5_shader::emit::emit_vulkan_hlsl(&ir)
}

const NARROW: &str = "thread_group_memory float store_[16];\n\
     int idx2(int a, int b) { return a + b; }\n\
     float fetch(int2 off, int base) {\n\
     return store_[base + off];\n\
     }\n\
     [NUM_THREADS(1,1,1)]\n\
     void main(uint i : S_DISPATCH_THREAD_ID) {\n\
     float v = fetch(idx2(i, i), i);\n\
     }\n";

#[test]
fn splatted_vector_param_narrows() {
    // `off` receives only scalar `int` (from `idx2`), used only as an
    // index addend: narrows `int2` → `int`, fixing the subscript.
    let hlsl = emit(NARROW);
    assert!(
        hlsl.contains("float fetch(int off, int base)"),
        "param must narrow to scalar, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("int2 off"),
        "vector spelling must be gone, got:\n{hlsl}"
    );
}

const VECTOR_ARG: &str = "thread_group_memory float store_[16];\n\
     float fetch(int2 off, int base) {\n\
     return store_[base + off];\n\
     }\n\
     [NUM_THREADS(1,1,1)]\n\
     void main(uint i : S_DISPATCH_THREAD_ID) {\n\
     int2 o = int2(i, i + 1u);\n\
     float v = fetch(o, i);\n\
     }\n";

#[test]
fn true_vector_arg_declines() {
    // Genuinely vector argument (`int2(i,i+1)`, lanes differ): must NOT
    // narrow (would change semantics); stays loud at DXC.
    let hlsl = emit(VECTOR_ARG);
    assert!(
        hlsl.contains("int2 off"),
        "vector param must survive (loud), got:\n{hlsl}"
    );
}
