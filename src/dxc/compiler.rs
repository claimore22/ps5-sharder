//! DXC integration: compile HLSL to SPIR-V via hassle-rs (DirectXShaderCompiler).
//!
//! The compiler captures the SPIR-V binary, plus any diagnostics produced by
//! DXC. Diagnostics are never discarded: on a compile failure the full DXC
//! message is surfaced as a [`Ps5ShaderError`] with phase
//! [`Phase::DxcCompile`].

use crate::error::{Phase, Ps5ShaderError, Result};
use crate::ir::Stage;

/// Compiles `hlsl` (with entry point `entry`) to a SPIR-V binary for `stage`.
///
/// Returns the SPIR-V bytes on success. Uses Vulkan-oriented DXC options
/// (`-spirv`) and an appropriate shader model profile for the stage.
pub fn compile_hlsl(hlsl: &str, entry: &str, stage: Stage) -> Result<Vec<u8>> {
    let profile = match stage {
        Stage::Vertex => "vs_6_0",
        Stage::Fragment => "ps_6_0",
        Stage::Compute => "cs_6_0",
        Stage::Domain => "ds_6_0",
        Stage::Domain => "ds_6_0",
        Stage::Geometry => "gs_6_0",
        // Hull compiles under `hs_6_0` (verified against bundled DXC
        // 1.9 with `-spirv`). Incomplete hull HLSL (missing patch
        // attributes, patch-constant function, or tessfactor semantics)
        // fails loudly at DXC with real diagnostics — fail-closed, never
        // miscompiled under a raster profile.
        Stage::Hull => "hs_6_0",
        // Mesh compiles under `ms_6_5` (verified against bundled DXC
        // 1.9 with `-spirv`). Incomplete mesh HLSL (missing
        // vertices/indices/counts/topology) fails loudly at DXC with real
        // diagnostics — fail-closed, never miscompiled under a raster
        // profile. Full NGG-output lowering (per-lane scattering,
        // uniformity) remains future work; until then mesh shaders stay
        // red with truthful HLSL errors instead of this opaque gate.
        Stage::Mesh => "ms_6_5",
        // Remaining non-raster stages have no DXC profile in this pipeline
        // yet (they need lib profiles plus matching emitter
        // support), so report a clean error instead of miscompiling under
        // the wrong profile. Semantic validation still succeeds for these
        // stages.
        Stage::AnyHit
        | Stage::ClosestHit
        | Stage::Miss
        | Stage::Intersection
        | Stage::Domain => {
            return Err(Ps5ShaderError::at(
                format!(
                    "DXC compile unsupported for stage `{}` (entry `{entry}`)",
                    stage.as_str()
                ),
                Phase::DxcCompile,
                crate::error::Span::default(),
            ));
        }
    };

    let args: &[&str] = &["-spirv", "-fspv-target-env=vulkan1.3"];

    match hassle_rs::compile_hlsl("generated.hlsl", hlsl, entry, profile, args, &[]) {
        Ok(bytes) => Ok(bytes),
        Err(error) => match error {
            hassle_rs::HassleError::CompileError(diagnostics) => Err(Ps5ShaderError::at(
                format!("DXC failed to compile HLSL:\n{diagnostics}"),
                Phase::DxcCompile,
                crate::error::Span::default(),
            )),
            other => Err(Ps5ShaderError::with_source(
                format!("DXC invocation failed for entry `{entry}` / profile `{profile}`"),
                Phase::DxcCompile,
                other,
            )),
        },
    }
}
