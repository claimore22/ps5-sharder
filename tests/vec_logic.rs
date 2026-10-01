use ps5_shader::parse_shader;

// Regression tests for vector short-circuit lowering: HLSL rejects
// `&&`/`||` on bool vectors, so pure-bool-vector operands lower to the
// component-wise `and()`/`or()` intrinsics (which broadcast scalars).
// Scalar logic and impure operands keep today's loud path.

fn emit(source: &str) -> String {
    let module = parse_shader(source).expect("parser accepted shader");
    let model = ps5_shader::semantic::analyze(&module).expect("semantic analysis succeeded");
    let ir = ps5_shader::lower::lower_module(&module, &model).expect("lowering succeeded");
    ps5_shader::emit::emit_vulkan_hlsl(&ir)
}

#[test]
fn vector_and_becomes_intrinsic() {
    let hlsl = emit(
        "float3 f(float3 nits, float mn, float mx) {\n\
         bool3 cri = ((nits >= mn) && (nits < mx));\n\
         return nits;\n\
         }\n\
         [NUM_THREADS(1,1,1)]\n\
         void main(uint i : S_DISPATCH_THREAD_ID) {\n\
         float3 y = f(float3(1,2,3), 0.0, 10.0);\n\
         }",
    );
    assert!(
        hlsl.contains("and("),
        "vector && must become and(), got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("&&"),
        "no vector && may leak, got:\n{hlsl}"
    );
}

#[test]
fn scalar_and_keeps_short_circuit() {
    let hlsl = emit(
        "float f(float a, float b) {\n\
         bool ok = ((a > 0.0) && (b > 0.0));\n\
         return ok ? 1.0 : 0.0;\n\
         }\n\
         [NUM_THREADS(1,1,1)]\n\
         void main(uint i : S_DISPATCH_THREAD_ID) {\n\
         float y = f(1.0, 2.0);\n\
         }",
    );
    assert!(
        hlsl.contains("&&"),
        "scalar && must keep short-circuit, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("and("),
        "scalar logic must not become and(), got:\n{hlsl}"
    );
}

#[test]
fn impure_vector_and_stays_loud() {
    let hlsl = emit(
        "bool3 g(float x) { return bool3(x > 0.0); }\n\
         float3 f(float3 nits) {\n\
         bool3 cri = (g(nits.x) && (nits.x > 0.0));\n\
         return nits;\n\
         }\n\
         [NUM_THREADS(1,1,1)]\n\
         void main(uint i : S_DISPATCH_THREAD_ID) {\n\
         float3 y = f(float3(1,2,3));\n\
         }",
    );
    assert!(
        hlsl.contains("&&"),
        "impure operands must keep && (loud at DXC), got:\n{hlsl}"
    );
}

#[test]
fn float_bitwise_converts_to_int() {
    let hlsl = emit(
        "bool2 isOdd(float2 val) {\n\
         return (val & 1);\n\
         }\n\
         [NUM_THREADS(1,1,1)]\n\
         void main(uint i : S_DISPATCH_THREAD_ID) {\n\
         bool2 b = isOdd(float2(1.5, 2.5));\n\
         }",
    );
    assert!(
        hlsl.contains("int2("),
        "float bitwise must convert to int, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("val & 1"),
        "bare float & must be gone, got:\n{hlsl}"
    );
}

#[test]
fn int_bitwise_untouched() {
    let hlsl = emit(
        "uint f(uint a, uint b) {\n\
         return (a & b);\n\
         }\n\
         [NUM_THREADS(1,1,1)]\n\
         void main(uint i : S_DISPATCH_THREAD_ID) {\n\
         uint y = f(6u, 3u);\n\
         }",
    );
    assert!(
        hlsl.contains("(a & b)"),
        "int bitwise must pass through, got:\n{hlsl}"
    );
}

#[test]
fn isnan_isinf_vector_and_becomes_intrinsic() {
    // Bool predicates (`isnan`/`isinf`/`isfinite`) return bool vectors
    // matching the argument width, so `!isnan(f3) && !isinf(f3)` is a
    // bool3 `&&` that must lower to element-wise `and()` (histogram
    // shaders: DXC rejects short-circuit operators on vectors).
    let hlsl = emit(
        "RWBuffer<float4> dst;\n\
         [NUM_THREADS(1,1,1)]\n\
         void main(uint i : S_DISPATCH_THREAD_ID) {\n\
         float3 v = float3(1.0, 2.0, 3.0);\n\
         bool ok = all((!isnan(v) && !isinf(v)));\n\
         dst[0] = float4(ok, ok, ok, 1.0);\n\
         }",
    );
    assert!(
        hlsl.contains("and("),
        "predicate-vector && must become and(), got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("&&"),
        "no vector && may leak, got:\n{hlsl}"
    );
}

#[test]
fn scalar_isnan_keeps_short_circuit() {
    // Scalar predicates stay scalar: `&&` keeps short-circuit semantics.
    let hlsl = emit(
        "RWBuffer<float4> dst;\n\
         [NUM_THREADS(1,1,1)]\n\
         void main(uint i : S_DISPATCH_THREAD_ID) {\n\
         float v = 1.0;\n\
         bool ok = ((!isnan(v)) && (v > 0.0));\n\
         dst[0] = float4(ok, ok, ok, 1.0);\n\
         }",
    );
    assert!(
        hlsl.contains("&&"),
        "scalar && must keep short-circuit, got:\n{hlsl}"
    );
}
