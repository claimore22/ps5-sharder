use ps5_shader::parse_shader;

// Regression tests for vector-condition ternary desugar (DXC `condition
// for short-circuiting ternary operator must be scalar`): `T x =
// boolN(c0..) ? a : b` in a vector-typed declaration emits per-component
// assigns. Non-bool condition args normalize via `!= 0`; already-bool
// shapes pass through; scalar literal branches replicate per component.
//
// (Results escape through sinks: unread locals are dead code and drop
// in lowering.)

fn emit(source: &str) -> String {
    let module = parse_shader(source).expect("parser accepted shader");
    let model = ps5_shader::semantic::analyze(&module).expect("semantic analysis succeeded");
    let ir = ps5_shader::lower::lower_module(&module, &model).expect("lowering succeeded");
    ps5_shader::emit::emit_vulkan_hlsl(&ir)
}

const PRELUDE: &str = "void sink3(float3 x) {}\nvoid sink2(float2 x) {}\nvoid sink(float x) {}\n[NUM_THREADS(1,1,1)]\nvoid main(uint position : S_DISPATCH_THREAD_ID) {\n";

#[test]
fn bool3_ternary_desugars_to_memberwise() {
    let hlsl = emit(&format!(
         "{PRELUDE}float3 v = float3(position, position + 1u, position + 2u);\n\
          float3 w = float3(0, 0, 0);\n\
          float3 r = bool3(position & 1u, position & 2u, position & 4u) ? v : w;\n\
          sink3(r);\n}}"
    ));
    assert!(
        hlsl.contains("float3 r;"),
        "must declare without init, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("r.x = ((position & 1u) != 0) ? v.x : w.x;"),
        "int conditions must normalize, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("r.z = ((position & 4u) != 0) ? v.z : w.z;"),
        "all components must be covered, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("bool3("),
        "vector ternary must not survive, got:\n{hlsl}"
    );
}

#[test]
fn bool_conditions_pass_through_unwrapped() {
    let hlsl = emit(&format!(
         "{PRELUDE}float3 v = float3(position, position + 1u, position + 2u);\n\
          float2 r = bool2(position == 1u, position == 2u) ? v.xy : v.yz;\n\
          sink2(r);\n}}"
    ));
    assert!(
        hlsl.contains("r.x = (position == 1u) ? v.xy.x : v.yz.x;"),
        "comparison conditions must not gain != 0, got:\n{hlsl}"
    );
}

#[test]
fn scalar_literal_branches_replicate() {
    let hlsl = emit(&format!(
         "{PRELUDE}float2 r = bool2(position == 1u, position == 2u) ? 1.0 : 0.0;\n\
          sink2(r);\n}}"
    ));
    assert!(
        hlsl.contains("r.x = (position == 1u) ? 1.0 : 0.0;"),
        "scalar branches must replicate, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("r.y = (position == 2u) ? 1.0 : 0.0;"),
        "scalar branches must replicate, got:\n{hlsl}"
    );
}

#[test]
fn scalar_ternary_is_untouched() {
    let hlsl = emit(&format!(
        "{PRELUDE}float r = position == 1u ? 2.0 : 3.0;\n\
         sink(r);\n}}"
    ));
    assert!(
        hlsl.contains("float r = ((position == 1u) ? 2.0 : 3.0);"),
        "scalar ternary must keep its shape, got:\n{hlsl}"
    );
}

#[test]
fn return_position_vector_ternary_becomes_select() {
    // Non-declaration positions (returns, args, nesting) cannot use
    // the memberwise path: vector conditions lower to `select`.
    let hlsl = emit(
        "float3 curve(float3 x, float k) { return (x < k) ? (x * k) : (x + k); }\n\
         [NUM_THREADS(1,1,1)]\n\
         void main(uint position : S_DISPATCH_THREAD_ID) {\n\
         float3 c = curve(float3(position, 0, 0), 1.0);\n\
         }",
    );
    assert!(
        hlsl.contains("select("),
        "vector ternary must become select, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("? (x * k)"),
        "vector ?: must not survive, got:\n{hlsl}"
    );
}

#[test]
fn impure_branches_stay_loud() {    // `select` evaluates both sides: branches with calls keep the loud
    // `?:` (no semantic change, unlike the pure fast path).
    let hlsl = emit(
        "float3 g(float3 x) { return x; }\n\
         float3 f(float3 x, bool3 c) { return c ? g(x) : x; }\n\
         [NUM_THREADS(1,1,1)]\n\
         void main(uint position : S_DISPATCH_THREAD_ID) {\n\
         float3 r = f(float3(position, 0, 0), bool3(true, false, true));\n\
         }",
    );
    assert!(
        !hlsl.contains("select("),
        "impure branches must not become select, got:\n{hlsl}"
    );
}

#[test]
fn int_vector_condition_wraps_not_equal() {
    // Non-bool vector conditions normalize through `!= 0`, like the
    // memberwise path does per component.
    let hlsl = emit(
        "float3 f(float3 x, int3 m) { return (m ? x : -x); }\n\
         [NUM_THREADS(1,1,1)]\n\
         void main(uint position : S_DISPATCH_THREAD_ID) {\n\
         float3 r = f(float3(position, 0, 0), int3(1, 0, 1));\n\
         }",
    );
    assert!(
        hlsl.contains("select("),
        "int-vector ternary must become select, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("!= 0"),
        "int condition must normalize, got:\n{hlsl}"
    );
}
