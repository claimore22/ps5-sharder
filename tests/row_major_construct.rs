use ps5_shader::parse_shader;

// Regression tests for `row_major|column_major matrix(…)` construction
// in initializers: the qualifier must not split into a stray value plus
// a detached statement. `row_major` fill order transposes to HLSL's
// column-major default with identical entries (verified value-for-value);
// `column_major` keeps order. Square float matrices only (2×2/3×3/4×4;
// bare `matrix` infers from 4/9/16 args); anything else stays loud.

fn emit(source: &str) -> String {
    let module = parse_shader(source).expect("parser accepted shader");
    let model = ps5_shader::semantic::analyze(&module).expect("semantic analysis succeeded");
    let ir = ps5_shader::lower::lower_module(&module, &model).expect("lowering succeeded");
    ps5_shader::emit::emit_vulkan_hlsl(&ir)
}

const ROW_MAJOR: &str = "RWBuffer<float4> dst;\n\
     [NUM_THREADS(1,1,1)]\n\
     void main(uint i : S_DISPATCH_THREAD_ID) {\n\
     float2 xv = float2(1.0, 2.0);\n\
     float2 yv = float2(3.0, 4.0);\n\
     row_major matrix m = row_major matrix(xv.x, xv.y, 0.0, 0.0, yv.x, yv.y, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);\n\
     dst[0] = mul(m, float4(1, 0, 0, 1));\n\
     }\n";

#[test]
fn row_major_construction_transposes() {
    let hlsl = emit(ROW_MAJOR);
    assert!(
        !hlsl.contains("row_major_"),
        "qualifier must not leak as a value, got:\n{hlsl}"
    );
    // Row-major rows (x,y,0,0),(z,w,0,0) → column-major cols
    // (x,z,0,0),(y,w,0,0): entries preserved, layout defaulted.
    assert!(
        hlsl.contains("float4x4 m = float4x4("),
        "must be a single fused construction, got:\n{hlsl}"
    );
    // Transposed order: xv.x, yv.x, ... (column 0 = old row heads).
    let at = hlsl.find("float4x4 m = float4x4(").expect("construction");
    let snippet = &hlsl[at..(at + 120).min(hlsl.len())];
    assert!(
        snippet.contains("xv.x") && snippet.find("xv.x").unwrap() < snippet.find("xv.y").unwrap_or(usize::MAX),
        "column-major order must lead each column, got:\n{hlsl}"
    );
}

const COLUMN_MAJOR: &str = "RWBuffer<float4> dst;\n\
     [NUM_THREADS(1,1,1)]\n\
     void main(uint i : S_DISPATCH_THREAD_ID) {\n\
     column_major float4x4 m = column_major float4x4(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);\n\
     dst[0] = mul(m, float4(1, 0, 0, 1));\n\
     }\n";

#[test]
fn column_major_construction_keeps_order() {
    let hlsl = emit(COLUMN_MAJOR);
    assert!(
        !hlsl.contains("column_major_") && !hlsl.contains("row_major_"),
        "qualifiers must not leak, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("float4x4 m = float4x4(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1)")
            || hlsl.contains("float4x4 m = float4x4(1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1)"),
        "column-major order must pass through, got:\n{hlsl}"
    );
}
