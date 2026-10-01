//! Acceptance test: vertex_position PSSL shader.
//!
//! The shader corpus lives at `tests/shaders/vertex_position/shader.pssl`.
//!
//! This test demonstrates the end-to-end translation:
//!   S_POSITION (PSSL) -> SV_Position (HLSL) -> BuiltIn Position (SPIR-V)
//! for a vertex shader stage.

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
    let path = manifest_dir().join("tests/shaders/vertex_position/shader.pssl");
    std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "failed to read vertex_position shader at {}: {e}",
            path.display()
        )
    })
}

fn dxc_dir() -> Option<PathBuf> {
    let dir = manifest_dir().join("dxc");
    if dir.join("dxcompiler.dll").exists() {
        Some(dir)
    } else {
        None
    }
}

fn unit_pipeline(source: &str) -> Result<String> {
    let module = parse_shader(source)?;
    let model = semantic::analyze(&module)?;
    let ir = lower_module(&module, &model)?;
    Ok(emit_vulkan_hlsl(&ir))
}

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

    assert!(
        hlsl.contains("SV_Position"),
        "emitted HLSL should contain SV_Position (from S_POSITION), got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("TEXCOORD0"),
        "emitted HLSL should contain TEXCOORD0 (from TEXCOORD0 input), got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("return inPos"),
        "emitted HLSL should return inPos, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("float4 main"),
        "emitted HLSL should contain entry point, got:\n{hlsl}"
    );
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
        let existing = std::env::var("PATH").unwrap_or_default();
        let joined = format!("{};{}", dir.display(), existing);
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

        let spirv_bytes = match dxc_pipeline(&shader_source()) {
            Ok(bytes) => bytes,
            Err(error) => {
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
        };

        assert!(
            !spirv_bytes.is_empty(),
            "DXC produced an empty SPIR-V binary"
        );

        // Inspect: verify S_POSITION survived as BuiltIn Position.
        let inspection = spirv::inspect_bytes(&spirv_bytes).expect("rspirv inspection failed");

        assert!(
            inspection
                .builtins
                .iter()
                .any(|b| b.name.as_deref() == Some("Position")),
            "expected a BuiltIn Position decoration (S_POSITION -> SV_Position -> Position), \
             but found builtins: {:?}",
            inspection.builtins
        );

        assert!(
            inspection.inputs.iter().any(|v| v.location == Some(0)),
            "expected an input at location 0 (TEXCOORD0), inputs: {:?}",
            inspection.inputs
        );

        assert!(
            inspection
                .outputs
                .iter()
                .any(|v| v.builtin.as_deref() == Some("Position")),
            "expected an output with BuiltIn Position, outputs: {:?}",
            inspection.outputs
        );

        // Stage must be Vertex.
        assert!(
            inspection
                .entry_points
                .iter()
                .any(|ep| ep.execution_model.as_deref() == Some("Vertex")),
            "expected Vertex execution model, entry points: {:?}",
            inspection.entry_points
        );

        println!(
            "vertex_position -> DXC -> SPIR-V -> SPIRV-Tools validation: PASS\n{}",
            spirv::dump_inspection(&inspection)
        );
    });
}

fn matches_error_to_dxc_unavailable(error: &ps5_shader::error::Ps5ShaderError) -> bool {
    let message = &error.message;
    message.contains("LoadLibrary")
        || message.contains("libloading")
        || message.contains("dxcompiler.dll")
        || error
            .source_message()
            .is_some_and(|source| source.to_string().contains("libloading"))
}
