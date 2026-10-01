use ps5_shader::parse_shader;

// Regression tests for wave prefix-sum intrinsics:
// PSSL `ExclusivePrefixSum(x)` → HLSL `WavePrefixSum(x)` (exclusive
// lane-prefix sum; wave-compaction idiom with `IsFirstActiveLane`).
// PSSL `InclusivePrefixSum(x)` → `WavePrefixSum(x) + x` (inclusive =
// exclusive + self; arithmetic identity). Inclusive duplicates its
// argument, so only side-effect-free arguments qualify (impure stays
// loud to avoid doubling side effects).

fn emit(source: &str) -> String {
    let module = parse_shader(source).expect("parser accepted shader");
    let model = ps5_shader::semantic::analyze(&module).expect("semantic analysis succeeded");
    let ir = ps5_shader::lower::lower_module(&module, &model).expect("lowering succeeded");
    ps5_shader::emit::emit_vulkan_hlsl(&ir)
}

const EXCLUSIVE: &str = "RWBuffer<uint> outBuf;\n\
     [NUM_THREADS(64,1,1)]\n\
     void main(uint i : S_DISPATCH_THREAD_ID) {\n\
     uint lane = ExclusivePrefixSum(1u);\n\
     outBuf[0] = lane;\n\
     }\n";

#[test]
fn exclusive_becomes_wave_prefix_sum() {
    let hlsl = emit(EXCLUSIVE);
    assert!(
        hlsl.contains("WavePrefixSum(1u)") || hlsl.contains("WavePrefixSum(1)"),
        "must become WavePrefixSum, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("ExclusivePrefixSum"),
        "PSSL spelling must be gone, got:\n{hlsl}"
    );
}

const INCLUSIVE: &str = "RWBuffer<uint> outBuf;\n\
     [NUM_THREADS(64,1,1)]\n\
     void main(uint i : S_DISPATCH_THREAD_ID) {\n\
     uint lane = InclusivePrefixSum(i);\n\
     outBuf[0] = lane;\n\
     }\n";

#[test]
fn inclusive_becomes_wave_prefix_plus_self() {
    let hlsl = emit(INCLUSIVE);
    assert!(
        hlsl.contains("WavePrefixSum(i)") && hlsl.contains("+ (i)"),
        "must become WavePrefixSum(i)+(i), got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("InclusivePrefixSum"),
        "PSSL spelling must be gone, got:\n{hlsl}"
    );
}

#[test]
fn exclusive_two_arg_drops_wave_bound() {
    // `ExclusivePrefixSum(v, maxWaveCount)` (multi-wave culling
    // headers): the wave count is a static bound with no DX equivalent
    // and drops; the scan itself is still `WavePrefixSum(v)`.
    let hlsl = emit(
        "RWBuffer<uint> outBuf;\n\
         [NUM_THREADS(64,1,1)]\n\
         void main(uint i : S_DISPATCH_THREAD_ID) {\n\
         uint lane = ExclusivePrefixSum(i, 2u);\n\
         outBuf[0] = lane;\n\
         }\n",
    );
    assert!(
        hlsl.contains("WavePrefixSum(i)"),
        "must become WavePrefixSum(i), got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("ExclusivePrefixSum"),
        "PSSL spelling must be gone, got:\n{hlsl}"
    );
}
