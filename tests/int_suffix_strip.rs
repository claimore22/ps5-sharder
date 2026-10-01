use ps5_shader::parse_shader;

// Regression tests for short-integer suffix stripping (`128us` → `128`).
// HLSL rejects `s`/`S`/`us`/`US` suffixes ("invalid suffix 'us'"); the
// values always fit `int`, so plain form is exact. `u`/`U`/`l`/`L`/`f`/
// `F`/`h`/`H` (HLSL-legal) and hex digits pass through untouched.

fn emit(source: &str) -> String {
    let module = parse_shader(source).expect("parser accepted shader");
    let model = ps5_shader::semantic::analyze(&module).expect("semantic analysis succeeded");
    let ir = ps5_shader::lower::lower_module(&module, &model).expect("lowering succeeded");
    ps5_shader::emit::emit_vulkan_hlsl(&ir)
}

const US_SUFFIX: &str = "RWBuffer<uint> outBuf;\n\
     [NUM_THREADS(1,1,1)]\n\
     void main(uint i : S_DISPATCH_THREAD_ID) {\n\
     uint v = (128us >> i);\n\
     outBuf[0] = v;\n\
     }\n";

#[test]
fn us_suffix_strips() {
    let hlsl = emit(US_SUFFIX);
    assert!(
        !hlsl.contains("128us") && !hlsl.contains("128US"),
        "us suffix must strip, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("128"),
        "value must survive, got:\n{hlsl}"
    );
}

const HEX_US_SUFFIX: &str = "RWBuffer<uint> outBuf;\n\
     [NUM_THREADS(1,1,1)]\n\
     void main(uint i : S_DISPATCH_THREAD_ID) {\n\
     uint v = 0xffffus;\n\
     outBuf[0] = v;\n\
     }\n";

#[test]
fn hex_us_suffix_strips_preserving_digits() {
    // `0xffffus`: trailing `us` strips, hex `ff` digits stay (`0xffff`,
    // not `0xff` or `0x`).
    let hlsl = emit(HEX_US_SUFFIX);
    assert!(
        hlsl.contains("0xffff"),
        "hex digits must survive, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("0xffffus"),
        "us suffix must strip, got:\n{hlsl}"
    );
}
