use ps5_shader::parse_shader;

// Regression tests for `GetDimensions` lowering: HLSL requires integer
// (`uint`) output parameters, so `int` destinations go through a `uint`
// temporary with an explicit `int()` conversion back. Buffers and 1D
// textures take a single dimension; 2D textures keep the established
// two-temporary `GetDimensionsFast` shape.

fn emit(source: &str) -> String {
    let module = parse_shader(source).expect("parser accepted shader");
    let model = ps5_shader::semantic::analyze(&module).expect("semantic analysis succeeded");
    let ir = ps5_shader::lower::lower_module(&module, &model).expect("lowering succeeded");
    ps5_shader::emit::emit_vulkan_hlsl(&ir)
}

#[test]
fn buffer_int_destination_uses_uint_temporary() {
    let hlsl = emit(
        "void sink(int x) {}\n\
         Buffer<uint3> hist;\n\
         [NUM_THREADS(1,1,1)]\n\
         void main(uint i : S_DISPATCH_THREAD_ID) {\n\
         int n = 0;\n\
         hist.GetDimensions(n);\n\
         sink(n);\n\
         }",
    );
    assert!(
        hlsl.contains("uint n_dim;"),
        "must declare a uint temporary, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("hist.GetDimensions(n_dim);"),
        "must call with the temporary, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("n = int(n_dim);"),
        "must convert back explicitly, got:\n{hlsl}"
    );
}

#[test]
fn buffer_uint_destination_passes_through() {
    let hlsl = emit(
        "void sink(uint x) {}\n\
         Buffer<uint3> hist;\n\
         [NUM_THREADS(1,1,1)]\n\
         void main(uint i : S_DISPATCH_THREAD_ID) {\n\
         uint n = 0;\n\
         hist.GetDimensions(n);\n\
         sink(n);\n\
         }",
    );
    assert!(
        hlsl.contains("hist.GetDimensions(n);"),
        "uint destination must pass through, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("n_dim;"),
        "no temporary needed, got:\n{hlsl}"
    );
}
