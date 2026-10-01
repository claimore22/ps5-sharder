//! SPIR-V validation via SPIRV-Tools. This is the *authoritative* validator
//! for the pipeline (distinct from the rspirv inspection layer, which is
//! debugging/analysis only and never determines validity).

use spirv_tools::val;
use spirv_tools::val::Validator;

use crate::error::{Phase, Ps5ShaderError, Result};

/// Validates a SPIR-V binary (given as 32-bit little-endian words) against
/// the Vulkan 1.3 SPIR-V environment.
pub fn validate_spirv(words: &[u32]) -> Result<()> {
    let validator = val::create(Some(spirv_tools::TargetEnv::Vulkan_1_3));
    validator.validate(words, None).map_err(|error| {
        Ps5ShaderError::with_source(error.to_string(), Phase::SpirvValidate, error)
    })
}

/// Converts a raw SPIR-V byte buffer (as produced by DXC, little-endian
/// 32-bit words) into a `Vec<u32>` suitable for the validator.
///
/// Returns a validation-phase error if the byte length is not a multiple of
/// four (malformed SPIR-V).
pub fn bytes_to_words(bytes: &[u8]) -> Result<Vec<u32>> {
    if bytes.len() % 4 != 0 {
        return Err(Ps5ShaderError::at(
            format!(
                "SPIR-V binary length {} is not a multiple of 4 bytes",
                bytes.len()
            ),
            Phase::SpirvValidate,
            crate::error::Span::default(),
        ));
    }

    let words = bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    Ok(words)
}
