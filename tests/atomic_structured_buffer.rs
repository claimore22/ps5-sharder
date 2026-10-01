use ps5_shader::parse_shader;

// Regression tests for atomics on vector-element typed buffers
// (histogram shaders: DXC rejects `InterlockedAdd` on a `RWBuffer<uint3>`
// component — "texture buffer whose result type is non-scalar" — while
// `RWStructuredBuffer<uint3>` allows component-select atomics).
// `RW_DataBuffer<Vector>` feeding an atomic retargets to structured;
// scalar buffers are untouched; `GetDimensions` on structured buffers
// takes `(count, stride)`.

fn emit(source: &str) -> String {
    let module = parse_shader(source).expect("parser accepted shader");
    let model = ps5_shader::semantic::analyze(&module).expect("semantic analysis succeeded");
    let ir = ps5_shader::lower::lower_module(&module, &model).expect("lowering succeeded");
    ps5_shader::emit::emit_vulkan_hlsl(&ir)
}

const ATOMIC_VEC3: &str = "struct Srt {\n\
     RW_DataBuffer<uint3> m_histogram;\n\
     };\n\
     [NUM_THREADS(1,1,1)]\n\
     void main(uint i : S_DISPATCH_THREAD_ID, Srt srt : S_SRT_DATA) {\n\
     int n = 0;\n\
     srt.m_histogram.GetDimensions(n);\n\
     AtomicAdd(srt.m_histogram[i].x, 1);\n\
     }\n";

#[test]
fn vector_databuffer_atomic_target_becomes_structured() {
    let hlsl = emit(ATOMIC_VEC3);
    assert!(
        hlsl.contains("RWStructuredBuffer<uint3>"),
        "atomic-targeted vector buffer must be structured, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("RWBuffer<uint3>"),
        "typed-buffer spelling must be gone, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("InterlockedAdd("),
        "atomic must survive, got:\n{hlsl}"
    );
}

#[test]
fn structured_getdimensions_takes_stride() {
    // `StructuredBuffer.GetDimensions` requires `(count, stride)`.
    let hlsl = emit(ATOMIC_VEC3);
    assert!(
        hlsl.contains("GetDimensions("),
        "GetDimensions must survive, got:\n{hlsl}"
    );
    // The count temp feeds a 2-arg call (stride dummy follows).
    assert!(
        hlsl.contains("_stride"),
        "structured GetDimensions needs a stride arg, got:\n{hlsl}"
    );
}

const ATOMIC_SCALAR: &str = "struct Srt {\n\
     RW_DataBuffer<uint> m_count;\n\
     };\n\
     [NUM_THREADS(1,1,1)]\n\
     void main(uint i : S_DISPATCH_THREAD_ID, Srt srt : S_SRT_DATA) {\n\
     AtomicAdd(srt.m_count[i], 1u);\n\
     }\n";

#[test]
fn scalar_databuffer_atomic_stays_typed() {
    // Scalar-element buffers support `RWBuffer` atomics; no retarget.
    let hlsl = emit(ATOMIC_SCALAR);
    assert!(
        hlsl.contains("RWBuffer<uint>"),
        "scalar buffer must stay typed, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("RWStructuredBuffer<uint>"),
        "no structured retarget for scalars, got:\n{hlsl}"
    );
}

const ATOMIC_PARAM_FWD: &str = "struct Srt {\n\
     RW_DataBuffer<uint3> m_hist;\n\
     };\n\
     void bump(RW_DataBuffer<uint3> dst, uint i) {\n\
     AtomicAdd(dst[i].x, 1u);\n\
     }\n\
     [NUM_THREADS(1,1,1)]\n\
     void main(uint i : S_DISPATCH_THREAD_ID, Srt srt : S_SRT_DATA) {\n\
     bump(srt.m_hist, i);\n\
     }\n";

#[test]
fn param_forwarded_atomic_retargets_together() {
    // Atomic through a forwarded param: the param AND the linked
    // call-site resource retarget together (signatures stay
    // consistent); the atomic survives legally.
    let hlsl = emit(ATOMIC_PARAM_FWD);
    assert!(
        hlsl.contains("RWStructuredBuffer<uint3>"),
        "param and resource must both be structured, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("RWBuffer<uint3>"),
        "no typed-buffer spelling may remain, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("InterlockedAdd("),
        "atomic must survive, got:\n{hlsl}"
    );
}
