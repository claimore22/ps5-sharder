use ps5_shader::parse_shader;

// Regression tests for constant-`if` folding (C++ `if constexpr`
// elimination once tests become literal): `if (true)` splices the taken
// block, `if (false)` vanishes (or splices `else`). Unconditionally
// semantics-preserving (untaken branches never execute), and required to
// drop compile-time-invalid references in discarded branches (e.g.
// `input.m_uv0` when `kUvCount == 0`). Non-literal conditions keep
// today's path.

fn emit(source: &str) -> String {
    let module = parse_shader(source).expect("parser accepted shader");
    let model = ps5_shader::semantic::analyze(&module).expect("semantic analysis succeeded");
    let ir = ps5_shader::lower::lower_module(&module, &model).expect("lowering succeeded");
    ps5_shader::emit::emit_vulkan_hlsl(&ir)
}

const IF_FALSE: &str = "RWBuffer<uint> outBuf;\n\
     [NUM_THREADS(1,1,1)]\n\
     void main(uint i : S_DISPATCH_THREAD_ID) {\n\
     uint x = 1u;\n\
     if (false) { x = 2u; }\n\
     outBuf[0] = x;\n\
     }\n";

#[test]
fn if_false_vanishes() {
    let hlsl = emit(IF_FALSE);
    assert!(
        !hlsl.contains("x = 2u") && !hlsl.contains("x = (2u)"),
        "untaken branch must vanish, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("outBuf[0]"),
        "live code must survive, got:\n{hlsl}"
    );
}

const IF_TRUE_ELSE: &str = "RWBuffer<uint> outBuf;\n\
     [NUM_THREADS(1,1,1)]\n\
     void main(uint i : S_DISPATCH_THREAD_ID) {\n\
     uint x = 1u;\n\
     if (true) { x = 2u; } else { x = 3u; }\n\
     outBuf[0] = x;\n\
     }\n";

#[test]
fn if_true_splices_taken_drops_else() {
    let hlsl = emit(IF_TRUE_ELSE);
    assert!(
        !hlsl.contains("3u"),
        "else branch must vanish, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("2u"),
        "taken branch must splice, got:\n{hlsl}"
    );
}

const IF_CONSTEXPR_CMP: &str = "struct Cfg {\n\
     static const uint kN = 0;\n\
     };\n\
     RWBuffer<uint> outBuf;\n\
     [NUM_THREADS(1,1,1)]\n\
     void main(uint i : S_DISPATCH_THREAD_ID) {\n\
     uint x = 1u;\n\
     if (Cfg::kN > 0) { x = 2u; }\n\
     outBuf[0] = x;\n\
     }\n";

#[test]
fn if_constexpr_comparison_folds() {
    // Qualified static resolves to `0`, comparison folds, dead branch
    // vanishes. This is the `if constexpr(T::kUvCount > 0)` shape once
    // `T::kUvCount` resolves.
    let hlsl = emit(IF_CONSTEXPR_CMP);
    assert!(
        !hlsl.contains("x = 2u") && !hlsl.contains("x = (2u)"),
        "dead constexpr branch must vanish, got:\n{hlsl}"
    );
}

const IF_RUNTIME: &str = "RWBuffer<uint> outBuf;\n\
     [NUM_THREADS(1,1,1)]\n\
     void main(uint i : S_DISPATCH_THREAD_ID) {\n\
     uint x = 1u;\n\
     if (i > 0u) { x = 2u; }\n\
     outBuf[0] = x;\n\
     }\n";

#[test]
fn runtime_if_untouched() {
    let hlsl = emit(IF_RUNTIME);
    assert!(
        hlsl.contains("if ("),
        "runtime if must survive, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("2u"),
        "live branch must survive, got:\n{hlsl}"
    );
}
