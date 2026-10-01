use ps5_shader::parse_shader;

// Regression tests for SRT helper overloads + removed-root method inlining
// + void-cast elision (pSrt wall: `vertex_shading` overloads lost
// `VertexInput` while keeping `pSrt`, then `pSrt.getX()` dangled).

fn emit(source: &str) -> String {
    let module = parse_shader(source).expect("parser accepted shader");
    let model = ps5_shader::semantic::analyze(&module).expect("semantic analysis succeeded");
    let ir = ps5_shader::lower::lower_module(&module, &model).expect("lowering succeeded");
    ps5_shader::emit::emit_vulkan_hlsl(&ir)
}

#[test]
fn overload_elision_keeps_non_srt_params() {
    let hlsl = emit(
        "struct Srt { Texture2D<float4> tex; float x; };\n\
         Srt srt;\n\
         float helper(Srt s, uint a) { return s.x + a; }\n\
         float helper(Srt s, uint a, uint b) { return s.x + a + b; }\n\
         [NUM_THREADS(1,1,1)]\n\
         void main(uint i : S_DISPATCH_THREAD_ID) {\n\
         float a = helper(srt, i);\n\
         float b = helper(srt, i, i);\n\
         }",
    );
    assert!(
        hlsl.contains("helper(uint a)"),
        "1-arg overload must keep its param, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("helper(uint a, uint b)"),
        "2-arg overload must keep both params, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("Srt s"),
        "SRT params must be elided from all overloads, got:\n{hlsl}"
    );
}

#[test]
fn dangling_single_return_getter_inlines() {
    let hlsl = emit(
        "struct S { RegularBuffer<float> b; uint i; float get() { return b[i]; } };\n\
         S srt;\n\
         float helper(S s, uint v) { return s.get(); }\n\
         [NUM_THREADS(1,1,1)]\n\
         void main(uint k : S_DISPATCH_THREAD_ID) {\n\
         float x = helper(srt, k);\n\
         }",
    );
    assert!(
        hlsl.contains("__hoist_S_b"),
        "getter must inline to hoisted binding, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("s.get()"),
        "dangling call must be gone, got:\n{hlsl}"
    );
}

#[test]
fn void_cast_statements_elide() {
    let hlsl = emit(
        "[NUM_THREADS(1,1,1)]\n\
         void main(uint i : S_DISPATCH_THREAD_ID) {\n\
         uint x = i; (void)x;\n\
         }",
    );
    assert!(
        !hlsl.contains("void("),
        "void casts must not leak to HLSL, got:\n{hlsl}"
    );
}
