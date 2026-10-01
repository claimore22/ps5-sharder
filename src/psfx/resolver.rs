//! Header resolver – content-based PSSL shader-header detection.
//!
//! We do **not** use `path.ends_with("psfx_*.h")`. A header is a PSSL shader
//! header if its *content* (and include context) contains PSSL/AGC constructs.

use std::path::{Path, PathBuf};

/// Returns `true` if `content` looks like a PSSL shader header.
pub fn is_pssl_shader_header(content: &str) -> bool {
    // Fast pre-check: any of these substrings indicates PSSL/AGC usage.
    // Mirrors the heuristic from the approved plan:
    // `#ifdef __PSSL__`, `Texture2D`, `RW_Texture2D`, `SamplerState`,
    // `sce::Agc::`, `CndMask`, `__get_wavemode`.
    const MARKERS: &[&str] = &[
        "__PSSL__",
        "Texture2D",
        "RW_Texture2D",
        "SamplerState",
        "sce::Agc::",
        "CndMask",
        "__get_wavemode",
        "__kWaveMode",
    ];
    let mut hits = 0;
    for m in MARKERS {
        if content.contains(m) {
            hits += 1;
            if hits >= 2 {
                return true;
            }
        }
    }
    // Single strong marker plus include-context hint is also enough.
    // `SamplerState` alone is distinctive.
    content.contains("SamplerState") || hits >= 1 && content.contains("Texture2D")
}

/// Resolves `include` relative to `including_file` using the same search order
/// as `preprocess::include_resolver`. Returns the on-disk path of the framework header.
/// This does **not** translate; translation is `translator::translate_header`.
pub fn resolve_header(
    include: &str,
    including_file: &Path,
    search_roots: &[PathBuf],
) -> Option<PathBuf> {
    // 1) Relative to including file
    if let Some(parent) = including_file.parent() {
        let cand = parent.join(include);
        if cand.exists() {
            return Some(cand);
        }
    }
    // 2) Header search roots (preprocess uses `tests/shaders` etc.)
    for root in search_roots {
        let cand = root.join(include);
        if cand.exists() {
            return Some(cand);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn detects_pssl_header() {
        let src = r#"
            #ifdef __PSSL__
            Texture2D<half4> g_tex;
            SamplerState g_sam;
            sce::Agc::Core::Sampler s;
            half x = CndMask(a,b,c);
            uint n = __get_wavemode();
        "#;
        assert!(is_pssl_shader_header(src));
    }
    #[test]
    fn rejects_plain_c_header() {
        assert!(!is_pssl_shader_header("#pragma once\nint x;"));
    }
}
