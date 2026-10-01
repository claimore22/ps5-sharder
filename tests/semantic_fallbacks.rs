use ps5_shader::parse_shader;
use ps5_shader::semantic;

// Regression tests for the semantic 960 -> 1000 campaign batch:
// ternary scalar splat, known-struct unknown-member opaque fallback
// (frozen parser drops `union { ... }` bodies), and unknown-variable
// opaque fallback (macro-generated names, dropped template params).

fn analyzes(source: &str) {
    let module = parse_shader(source).expect("parser accepted shader");
    semantic::analyze(&module).expect("semantic analysis succeeded");
}

// HLSL broadcasts a scalar across a ternary vector arm
// (`cond ? float3 : float` and `cond ? float : float3`).
#[test]
fn ternary_scalar_splat_both_directions() {
    analyzes(
        "float3 f(bool c, float3 v, float s) { return c ? v : s; }",
    );
    analyzes(
        "float3 g(bool c, float s, float3 v) { return c ? s : v; }",
    );
}

// Union members dropped by the frozen parser (`Hit { ... union {
// triangleID; ... }; }`) resolve opaquely instead of erroring.
#[test]
fn unknown_member_on_known_struct_is_opaque() {
    analyzes(
        "struct Hit { uint instanceIndex; float hitDistance; };\n\
         float f(Hit h) { return h.triangleID; }",
    );
}

// Macro-generated / template-param names with no visible declaration
// (e.g. `MODE_VISUALIZE_PROBES`, `flags`) model opaquely; comparisons
// yield bool as usual.
#[test]
fn unknown_variable_is_opaque() {
    analyzes(
        "float f(float m) { return m == MODE_VISUALIZE_PROBES ? 1.0 : 0.0; }",
    );
}
