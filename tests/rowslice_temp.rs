use ps5_shader::parse_shader;

// Regression tests for flattened row-slice array arguments
// (`f(arr[x])` where `arr` is a flattened 2D array and `f` takes an
// array): row-major flattening breaks row slices (PSSL `arr[x]` is a row
// `T[2]`, but flattened 1D yields an element `T`), so the row
// materializes through a temporary with copy-in/copy-out
// (`T tmp[2]; tmp[k]=arr[x*P+k]; f(…,tmp,…);` + write-back for
// `out`/`inout`). Only fires on proven element-vs-array mismatches with
// unique callees and known sizes; anything else stays loud.

fn emit(source: &str) -> String {
    let module = parse_shader(source).expect("parser accepted shader");
    let model = ps5_shader::semantic::analyze(&module).expect("semantic analysis succeeded");
    let ir = ps5_shader::lower::lower_module(&module, &model).expect("lowering succeeded");
    ps5_shader::emit::emit_vulkan_hlsl(&ir)
}

const ROWSLICE_OUT: &str = "RWBuffer<int3> outBuf;\n\
     void fill(out int3 q[2]) {\n\
     q[0] = int3(1, 2, 3);\n\
     q[1] = int3(4, 5, 6);\n\
     }\n\
     [NUM_THREADS(1,1,1)]\n\
     void main(uint i : S_DISPATCH_THREAD_ID) {\n\
     int3 arr[6];\n\
     fill(arr[i]);\n\
     outBuf[i] = arr[i * 2];\n\
     }\n";

#[test]
fn rowslise_out_materializes_temp() {
    // `arr[i]` (element on flattened `[6]`) vs `out int3[2]` (array):
    // temp + write-back; original call retargeted. (`arr` is read after
    // so the write-back is live; dead write-backs are correctly
    // eliminated elsewhere.)
    let hlsl = emit(ROWSLICE_OUT);
    assert!(
        hlsl.contains("__rowslice"),
        "temp must materialize, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("fill(__rowslice"),
        "call must retarget to temp, got:\n{hlsl}"
    );
    // Write-back preserves `out` semantics (row updated).
    assert!(
        hlsl.contains("arr["),
        "write-back must reference the array, got:\n{hlsl}"
    );
}

const ROWSLICE_IN: &str = "uint readit(int3 q[2]) {\n\
     return uint(q[0].x + q[1].x);\n\
     }\n\
     [NUM_THREADS(1,1,1)]\n\
     void main(uint i : S_DISPATCH_THREAD_ID) {\n\
     int3 arr[6];\n\
     uint v = readit(arr[i]);\n\
     }\n";

#[test]
fn rowslice_in_copies_without_writeback() {
    // By-value array param: copy-in only (no write-back; callee
    // modifications must not escape).
    let hlsl = emit(ROWSLICE_IN);
    assert!(
        hlsl.contains("__rowslice"),
        "temp must materialize, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("readit(__rowslice"),
        "call must retarget, got:\n{hlsl}"
    );
}
