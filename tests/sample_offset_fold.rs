use ps5_shader::parse_shader;

// Regression tests for integer-texel-offset folding on texture sampling
// (SPIR-V `VUID-RuntimeSpirv-Offset-10213` on `SampleLOD(..., int2)`):
// Vulkan forbids `Offset` on non-Gather image reads, so an offsetted
// sample folds into UV arithmetic
// (`tex.SampleLevel(s, uv, lod, off)` ->
// `tex.SampleLevel(s, uv + float2(off)/float2(__dims), lod)`) with a
// function-entry `GetDimensions` providing the texel scale. Method
// spellings are preserved for downstream mapping.

fn emit(source: &str) -> String {
    let module = parse_shader(source).expect("parser accepted shader");
    let model = ps5_shader::semantic::analyze(&module).expect("semantic analysis succeeded");
    let ir = ps5_shader::lower::lower_module(&module, &model).expect("lowering succeeded");
    ps5_shader::emit::emit_vulkan_hlsl(&ir)
}

const OFFSET_SAMPLE: &str = "Texture2D<float4> src;\n\
     SamplerState srcSmp;\n\
     RWBuffer<float4> dst;\n\
     [NUM_THREADS(1,1,1)]\n\
     void main(uint2 tid : S_DISPATCH_THREAD_ID) {\n\
     float2 loc = float2(tid) / 16.0f;\n\
     dst[0] = src.SampleLevel(srcSmp, loc, 0, int2(1, 0));\n\
     }\n";

#[test]
fn offset_sample_folds_into_uv_with_dims() {
    let hlsl = emit(OFFSET_SAMPLE);
    assert!(
        !hlsl.contains("SampleLevel(srcSmp, loc, 0, int2"),
        "offset must fold out of the sample, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("GetDimensions("),
        "dims must come from GetDimensions, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("SampleLevel("),
        "sample must survive (offset folded to UV), got:\n{hlsl}"
    );
}

const PLAIN_SAMPLE: &str = "Texture2D<float4> src;\n\
     SamplerState srcSmp;\n\
     RWBuffer<float4> dst;\n\
     [NUM_THREADS(1,1,1)]\n\
     void main(uint2 tid : S_DISPATCH_THREAD_ID) {\n\
     float2 loc = float2(tid) / 16.0f;\n\
     dst[0] = src.SampleLevel(srcSmp, loc, 0);\n\
     }\n";

#[test]
fn plain_sample_untouched() {
    // No offset: no folding, no dims temp, no GetDimensions.
    let hlsl = emit(PLAIN_SAMPLE);
    assert!(
        !hlsl.contains("__sample_dims"),
        "no dims temp without an offset, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("SampleLevel(srcSmp, loc, 0)"),
        "plain sample must pass through, got:\n{hlsl}"
    );
}
