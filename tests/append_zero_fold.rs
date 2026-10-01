use ps5_shader::parse_shader;

// Regression tests for inert literal-zero folding on append-buffer
// `Append` (`buf.Append(v, 0)` → `buf.Append(v)`).
// PSSL `AppendRegularBuffer<T>` spells the append with a trailing
// literal `0`; HLSL `AppendStructuredBuffer<T>.Append` takes the value
// alone, and official ISA shows a single `ds_append` with no second
// operand, proving the `0` carries no content. Only literal-zero folds;
// nonzero/dynamic second arguments stay loud (meaning unestablished).

fn emit(source: &str) -> String {
    let module = parse_shader(source).expect("parser accepted shader");
    let model = ps5_shader::semantic::analyze(&module).expect("semantic analysis succeeded");
    let ir = ps5_shader::lower::lower_module(&module, &model).expect("lowering succeeded");
    ps5_shader::emit::emit_vulkan_hlsl(&ir)
}

const APPEND_ZERO: &str = "struct Srt {\n\
     AppendRegularBuffer<uint4> m_out;\n\
     };\n\
     [NUM_THREADS(1,1,1)]\n\
     void main(uint i : S_DISPATCH_THREAD_ID, Srt srt : S_SRT_DATA) {\n\
     uint4 v = uint4(i, 0, 0, 1);\n\
     srt.m_out.Append(v, 0);\n\
     }\n";

#[test]
fn append_literal_zero_folds() {
    let hlsl = emit(APPEND_ZERO);
    assert!(
        hlsl.contains(".Append("),
        "append must survive, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains(", 0)"),
        "inert zero must fold out, got:\n{hlsl}"
    );
}

const APPEND_NONZERO: &str = "struct Srt {\n\
     AppendRegularBuffer<uint4> m_out;\n\
     };\n\
     [NUM_THREADS(1,1,1)]\n\
     void main(uint i : S_DISPATCH_THREAD_ID, Srt srt : S_SRT_DATA) {\n\
     uint4 v = uint4(i, 0, 0, 1);\n\
     srt.m_out.Append(v, 1);\n\
     }\n";

#[test]
fn append_nonzero_stays_loud() {
    // Nonzero second argument: meaning unestablished, must not fold.
    let hlsl = emit(APPEND_NONZERO);
    assert!(
        hlsl.contains(", 1)"),
        "nonzero second arg must survive (loud at DXC), got:\n{hlsl}"
    );
}
