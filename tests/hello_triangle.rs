//! Acceptance test: hello_triangle PSSL shader.
//!
//! The shader corpus lives at `tests/shaders/hello_triangle/shader.pssl`.
//!
//! Two categories:
//!   * unit tests               -> always run (lexer..emitter, no DXC)
//!   * DXC integration tests    -> run when DXC is available, otherwise they
//!                                  clearly report "DXC unavailable".

use std::path::PathBuf;

use ps5_shader::emit::emit_vulkan_hlsl;
use ps5_shader::error::Result;
use ps5_shader::lower::lower_module;
use ps5_shader::parse_shader;
use ps5_shader::semantic;
use ps5_shader::spirv;

fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn shader_source() -> String {
    let path = manifest_dir().join("tests/shaders/hello_triangle/shader.pssl");
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read shader.pssl at {}: {e}", path.display()))
}

fn dxc_dir() -> Option<PathBuf> {
    let dir = manifest_dir().join("dxc");
    if dir.join("dxcompiler.dll").exists() {
        Some(dir)
    } else {
        None
    }
}

/// Runs the non-DXC portion of the pipeline and returns the emitted HLSL.
fn unit_pipeline(source: &str) -> Result<String> {
    let module = parse_shader(source)?;
    let model = semantic::analyze(&module)?;
    let ir = lower_module(&module, &model)?;
    Ok(emit_vulkan_hlsl(&ir))
}

/// Runs the full pipeline including DXC + SPIRV-Tools validation.
fn dxc_pipeline(source: &str) -> Result<Vec<u8>> {
    let module = parse_shader(source)?;
    let model = semantic::analyze(&module)?;
    let stage = model.stage.ok_or_else(|| {
        ps5_shader::error::Ps5ShaderError::new(
            "no stage determined",
            ps5_shader::error::Phase::DxcCompile,
        )
    })?;
    let entry = model
        .entry_point
        .as_ref()
        .map(|ep| ep.function.clone())
        .ok_or_else(|| {
            ps5_shader::error::Ps5ShaderError::new(
                "no entry point",
                ps5_shader::error::Phase::DxcCompile,
            )
        })?;

    let ir = lower_module(&module, &model)?;
    let hlsl = emit_vulkan_hlsl(&ir);
    let bytes = ps5_shader::dxc::compile_hlsl(&hlsl, &entry, stage)?;
    let words = spirv::bytes_to_words(&bytes)?;
    spirv::validate_spirv(&words)?;
    Ok(bytes)
}

// --------------------------------------------------------------------------
// Unit tests (always run)
// --------------------------------------------------------------------------

#[test]
fn parses_semantics_and_emits_valid_hlsl() {
    let hlsl = unit_pipeline(&shader_source()).expect("unit pipeline should succeed");

    assert!(hlsl.contains("struct V2P"));
    assert!(hlsl.contains("float4 pos : SV_Position"));
    assert!(hlsl.contains("float3 color : TEXCOORD0"));
    assert!(hlsl.contains("float4 main(V2P input) : SV_Target0"));
    assert!(hlsl.contains("return float4(input.color, 1.0);"));
}

#[test]
fn emitted_hlsl_is_deterministic() {
    let source = shader_source();
    let first = unit_pipeline(&source).expect("first emission");
    let second = unit_pipeline(&source).expect("second emission");
    assert_eq!(first, second, "emitter must be deterministic");
}

// --------------------------------------------------------------------------
// DXC integration tests
// --------------------------------------------------------------------------

fn with_dxc_path<T>(f: impl FnOnce() -> T) -> T {
    if let Some(dir) = dxc_dir() {
        // Prepend the DXC directory so dxcompiler.dll / dxil.dll are locatable.
        let existing = std::env::var("PATH").unwrap_or_default();
        let joined = format!("{};{}", dir.display(), existing);
        // SAFETY: single-threaded test runner; setting PATH is required so that
        // the DXC loader can locate dxcompiler.dll / dxil.dll.
        unsafe { std::env::set_var("PATH", joined) };
    }
    f()
}

#[test]
fn full_pipeline_through_dxc_and_validation() {
    with_dxc_path(|| {
        match dxc_dir() {
            None => {
                eprintln!(
                    "\n!! DXC UNAVAILABLE: dxcompiler.dll not found in {} — \
                     DXC integration test NOT run.\n",
                    manifest_dir().join("dxc").display()
                );
                return;
            }
            Some(_) => {}
        }

        match dxc_pipeline(&shader_source()) {
            Ok(spirv_bytes) => {
                // Ensure we actually produced a SPIR-V binary, not an empty result.
                assert!(
                    !spirv_bytes.is_empty(),
                    "DXC produced an empty SPIR-V binary"
                );
                println!("hello_triangle -> DXC -> SPIR-V -> SPIRV-Tools validation: PASS");
            }
            Err(error) => {
                // Distinguish "DXC unavailable" (environmental) from a genuine
                // compilation failure, which must fail the test.
                if let ps5_shader::error::Phase::DxcCompile = error.phase {
                    if matches_error_to_dxc_unavailable(&error) {
                        eprintln!(
                            "\n!! DXC UNAVAILABLE: could not load DirectXShaderCompiler: {error}\n"
                        );
                        return;
                    }
                }
                panic!("DXC pipeline failed: {error}");
            }
        }
    });
}

/// Heuristics to detect that DXC itself could not be loaded (as opposed to a
/// real HLSL compilation error).
fn matches_error_to_dxc_unavailable(error: &ps5_shader::error::Ps5ShaderError) -> bool {
    let message = &error.message;
    message.contains("LoadLibrary")
        || message.contains("libloading")
        || message.contains("dxcompiler.dll")
        || error
            .source_message()
            .is_some_and(|source| source.to_string().contains("libloading"))
}
