use ps5_shader::parse_shader;

// Regression tests for struct-scoped enumerators: nested `enum Flags`
// inside a struct registers its enumerators as int globals (top-level
// precedent — values are shader-irrelevant for validation), and
// qualified uses (`S::E`) fold to the bare name in lowering. Deeper
// `sce::…` paths keep their dedicated framework paths, and shadowing
// locals block the fold.

fn emit(source: &str) -> String {
    let module = parse_shader(source).expect("parser accepted shader");
    let model = ps5_shader::semantic::analyze(&module).expect("semantic analysis succeeded");
    let ir = ps5_shader::lower::lower_module(&module, &model).expect("lowering succeeded");
    ps5_shader::emit::emit_vulkan_hlsl(&ir)
}

#[test]
fn nested_enum_use_folds_to_bare_name() {
    let hlsl = emit(
        "struct M { uint m_flags; enum Flags { DoubleSided = 1, Transparent = 2 }; };\n\
         bool f(M m) { return (m.m_flags & M::Transparent) != 0; }\n\
         [NUM_THREADS(1,1,1)]\n\
         void main(uint i : S_DISPATCH_THREAD_ID) {\n\
         M m; m.m_flags = 3; bool b = f(m);\n\
         }",
    );
    assert!(
        !hlsl.contains("M::Transparent"),
        "qualified enumerator must fold, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("m.m_flags & Transparent"),
        "use must read the bare enumerator, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("static const int Transparent = 2;"),
        "enumerator must register with its value, got:\n{hlsl}"
    );
}

#[test]
fn shadowing_local_blocks_fold() {
    // A local named like the enumerator captures the bare name, so the
    // qualified form must stay (loud rather than mis-resolved).
    let hlsl = emit(
        "struct M { uint m_flags; enum Flags { Transparent = 2 }; };\n\
         bool f(M m) { uint Transparent = 7; return (m.m_flags & M::Transparent) != 0; }\n\
         [NUM_THREADS(1,1,1)]\n\
         void main(uint i : S_DISPATCH_THREAD_ID) {\n\
         M m; m.m_flags = 3; bool b = f(m);\n\
         }",
    );
    assert!(
        hlsl.contains("M::Transparent"),
        "shadowed enumerator must stay qualified, got:\n{hlsl}"
    );
}

#[test]
fn sce_paths_do_not_fold() {    // Framework paths are not struct scopes: untouched here (their own
    // elision paths own them).
    let hlsl = emit(
        "struct M { uint m_flags; enum Flags { Transparent = 2 }; };\n\
         SamplerState f(Sampler s) { return SamplerState(s); }\n\
         [NUM_THREADS(1,1,1)]\n\
         void main(uint i : S_DISPATCH_THREAD_ID) {\n\
         Sampler s; SamplerState t = f(s);\n\
         }",
    );
    assert!(
        !hlsl.contains("sce::"),
        "no framework residue may leak, got:\n{hlsl}"
    );
}

#[test]
fn top_level_enum_types_and_values() {
    // Top-level enums are int-backed: the name works in type position
    // (params, casts) and evaluated values become `static const`
    // constants usable in `case` labels.
    let hlsl = emit(
        "enum OutputEncoding { PQ, P3D65, LinearNits };\n\
         enum Flags { A = 1, B, C = 8, D };\n\
         float f(OutputEncoding e) { switch (e) { case PQ: return 1.0; case P3D65: return 2.0; default: return 0.0; } }\n\
         [NUM_THREADS(1,1,1)]\n\
         void main(uint i : S_DISPATCH_THREAD_ID) {\n\
         float x = f(PQ) + B + D;\n\
         }",
    );
    assert!(
        hlsl.contains("float f(int e)"),
        "enum type must lower to int, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("static const int PQ = 0;"),
        "auto value must start at 0, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("static const int B = 2;"),
        "auto value must increment, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("static const int D = 9;"),
        "auto value must continue after explicit, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("case PQ:"),
        "case labels must use the enumerator, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("OutputEncoding::"),
        "qualified uses must fold, got:\n{hlsl}"
    );
}

#[test]
fn unevaluatable_enum_stays_value_less() {
    // An enumerator whose value cannot be evaluated keeps today's
    // loud path (bare int global, no constant) rather than a guessed
    // value that would miscompile silently.
    let hlsl = emit(
        "enum E { A = SOME_UNKNOWN_MACRO(3), B };\n\
         [NUM_THREADS(1,1,1)]\n\
         void main(uint i : S_DISPATCH_THREAD_ID) {\n\
         uint x = A;\n\
         }",
    );
    assert!(
        hlsl.contains("int A;"),
        "unevaluatable enumerator must stay a bare global, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("static const int A"),
        "no guessed constant may emit, got:\n{hlsl}"
    );
}
