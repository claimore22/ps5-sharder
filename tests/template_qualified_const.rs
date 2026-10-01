use ps5_shader::parse_shader;

// Regression tests for spelling-based `T::` monomorphization
// (`phong(V2P<T>)` + `T::kUvCount` called with `V2P<PackVertex<…>>`):
// declaration spellings bind typenames to concrete template-ids,
// struct statics evaluate under bound value-params (+ recorded
// defaults), `T::kX` substitutes to literals, clones retarget, and
// `if constexpr` dead branches fold. Unresolvable shapes decline
// (stay loud), never guess.

fn emit(source: &str) -> String {
    let module = parse_shader(source).expect("parser accepted shader");
    let model = ps5_shader::semantic::analyze(&module).expect("semantic analysis succeeded");
    let ir = ps5_shader::lower::lower_module(&module, &model).expect("lowering succeeded");
    ps5_shader::emit::emit_vulkan_hlsl(&ir)
}

const QUALIFIED: &str = "template<int N>\n\
     struct S {\n\
     static constexpr int kN = N;\n\
     static constexpr bool kB = N > 0;\n\
     float4 v;\n\
     };\n\
     template<typename T>\n\
     struct W {\n\
     float4 wpos : S_POSITION;\n\
     };\n\
     template<typename T>\n\
     float4 f(W<T> w, float4 d) {\n\
     if (T::kB) { return d * float4(T::kN, 0, 0, 1); }\n\
     return d;\n\
     }\n\
     [NUM_THREADS(1,1,1)]\n\
     void main(uint i : S_DISPATCH_THREAD_ID) {\n\
     W<S<1>> x;\n\
     x.wpos = float4(i, 0, 0, 1);\n\
     float4 d = float4(1, 1, 1, 1);\n\
     d = f(x, d);\n\
     }\n";

#[test]
fn qualified_const_resolves_and_folds() {
    // `T := S<1>`: `kB = (1>0) = true` keeps the branch, `kN = 1`
    // scales. No `T::` may survive; no dead `if (false)` may survive.
    let hlsl = emit(QUALIFIED);
    assert!(
        !hlsl.contains("T::"),
        "all T:: must resolve, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("if (false)") && !hlsl.contains("if ((false))"),
        "dead branch must fold, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("float4(1, 0, 0, 1)") || hlsl.contains("float4(1,0,0,1)"),
        "kN=1 must substitute, got:\n{hlsl}"
    );
}

const QUALIFIED_ZERO: &str = "template<int N>\n\
     struct S {\n\
     static constexpr int kN = N;\n\
     static constexpr bool kB = N > 0;\n\
     float4 v;\n\
     };\n\
     template<typename T>\n\
     struct W {\n\
     float4 wpos : S_POSITION;\n\
     };\n\
     template<typename T>\n\
     float4 f(W<T> w, float4 d) {\n\
     if (T::kB) { return d * float4(9, 9, 9, 9); }\n\
     return d;\n\
     }\n\
     [NUM_THREADS(1,1,1)]\n\
     void main(uint i : S_DISPATCH_THREAD_ID) {\n\
     W<S<0>> x;\n\
     x.wpos = float4(i, 0, 0, 1);\n\
     float4 d = float4(1, 1, 1, 1);\n\
     d = f(x, d);\n\
     }\n";

#[test]
fn qualified_false_drops_branch() {
    // `T := S<0>`: `kB = (0>0) = false` drops the branch entirely
    // (the `float4(9,9,9,9)` must vanish with it).
    let hlsl = emit(QUALIFIED_ZERO);
    assert!(
        !hlsl.contains("T::"),
        "all T:: must resolve, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("9, 9, 9, 9") && !hlsl.contains("9,9,9,9"),
        "dead branch body must vanish, got:\n{hlsl}"
    );
}
