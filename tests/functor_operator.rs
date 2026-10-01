use ps5_shader::parse_shader;

// Regression tests for C++ `operator()` functor inlining (DXC `type
// 'ValidBits' does not provide a call operator` on `isValid(i)`):
// a struct method named `operator()` with a single `return` inlines at
// `obj(args)` call sites, rebinding bare field references to the callee
// (`m_bits` -> `isValid.m_bits`), matching C++ member lookup. Stateless
// tags (`AllValid`) and stateful functors (`ValidBits`, `InSubset`)
// share the path; sibling-method references and non-return bodies stay
// loud.

fn emit(source: &str) -> String {
    let module = parse_shader(source).expect("parser accepted shader");
    let model = ps5_shader::semantic::analyze(&module).expect("semantic analysis succeeded");
    let ir = ps5_shader::lower::lower_module(&module, &model).expect("lowering succeeded");
    ps5_shader::emit::emit_vulkan_hlsl(&ir)
}

const STATEFUL: &str = "struct ValidBits {\n\
     uint m_bits;\n\
     void set(uint i, uint v) { m_bits = (m_bits | (v << i)); }\n\
     bool operator()(uint i) const { return BitFieldExtract(m_bits, i, 1); }\n\
     };\n\
     [NUM_THREADS(1,1,1)]\n\
     void main(uint tid : S_DISPATCH_THREAD_ID) {\n\
     ValidBits isValid;\n\
     isValid.set(0u, 1u);\n\
     bool b = isValid(0u);\n\
     }\n";

#[test]
fn stateful_functor_call_inlines_with_field_rebind() {
    let hlsl = emit(STATEFUL);
    assert!(
        !hlsl.contains("isValid("),
        "functor call must inline, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("isValid.m_bits"),
        "field must rebind to callee, got:\n{hlsl}"
    );
}

#[test]
fn stateful_functor_operator_method_dropped() {
    // The inlined `operator()` is unreachable afterwards and strips;
    // no invalid HLSL `operator` method may survive.
    let hlsl = emit(STATEFUL);
    assert!(
        !hlsl.contains("operator"),
        "no operator method may emit, got:\n{hlsl}"
    );
}

const STATELESS: &str = "RWBuffer<uint> outBuf;\n\
     struct AllValid {\n\
     bool operator()(uint i) { return true; }\n\
     };\n\
     [NUM_THREADS(1,1,1)]\n\
     void main(uint tid : S_DISPATCH_THREAD_ID) {\n\
     AllValid ok;\n\
     bool b = ok(0u);\n\
     outBuf[0] = b ? 1u : 0u;\n\
     }\n";

#[test]
fn stateless_functor_call_inlines_to_constant() {
    let hlsl = emit(STATELESS);
    assert!(
        !hlsl.contains("ok("),
        "stateless functor call must inline, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("bool b = true"),
        "body must substitute to constant, got:\n{hlsl}"
    );
}
