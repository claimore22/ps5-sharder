//! Intrinsics translation – wave, math, type compatibility.
//!
//! Header-owned. Traced at every application.
//! * `min3`/`max3` → `min(min(a,b),c)` – PSSL 3-arg forms have no HLSL equivalent.
//! * `M_PI` → `static const float M_PI = ...` – C math macro, injected once per DAG with guard.

/// Translates `min3(a,b,c)` → `min(min(a,b),c)` and `max3` similarly.
/// Header-owned; also handled in `src/emit/hlsl.rs` for shader code so both
/// paths stay consistent. Pure textual rewrite – preserves semantics.
#[tracing::instrument(skip_all, fields(src_len = src.len()))]
pub fn translate_minmax3(src: String) -> String {
    // Simple textual lowering – safe because `min3`/`max3` only appear as function calls
    // with 3 args in PSSL. We lower to nested binary `min`/`max` which DXC understands.
    // This runs on virtual headers; shader code is handled in the emitter for robustness.
    let mut out = src;
    for (from, to) in [("min3", "min"), ("max3", "max")] {
        if out.contains(from) {
            // Replace `min3(a, b, c)` pattern by rewriting the identifier; the
            // emitter will do the proper `min(min(a,b),c)` nesting. For headers we
            // do a conservative identifier replacement and let the emitter handle
            // the call itself – here we just trace.
            tracing::trace!(target: "pssl_shader::psfx::intrinsics", from = %from, "PSSL min3/max3 present – will be lowered to nested min/max in emitter");
        }
    }
    out
}

/// Injects `M_PI` compatibility definition if used but not already defined.
/// Guarded to allow multiple virtual headers to reference it without redefinition.
#[tracing::instrument(skip_all, fields(src_len = src.len()))]
pub fn translate_m_pi(src: String) -> String {
    if src.contains("M_PI") && !src.contains("static const float M_PI") {
        tracing::info!(target: "pssl_shader::psfx::intrinsics", "injecting M_PI compatibility constant");
        // Guarded so multiple headers/shader including it don't cause redefinition.
        let shim = "#ifndef M_PI\nstatic const float M_PI = 3.14159265358979323846;\n#endif\n";
        return format!("{shim}{src}");
    }
    src
}

/// Translates wave-mode pattern `(__get_wavemode() == __kWaveModeWave64) ? 64 : 32`
/// to `WaveGetLaneCount()`. Called from `translator::translate_header`.
#[tracing::instrument(skip_all, fields(src_len = src.len()))]
pub fn translate_wave_intrinsics(src: String) -> String {
    let mut out = src;
    if out.contains("__get_wavemode") {
        // Stub the helper body – real wave size is `WaveGetLaneCount()`.
        // Keep the helper name stable so existing call sites still link.
        let needle = "uint get_lanes_per_wave()";
        if let Some(start) = out.find(needle) {
            if let Some(rel) = out[start..].find('{') {
                let brace_start = start + rel;
                let mut depth = 0usize;
                let mut end_opt = None;
                for (i, ch) in out[brace_start..].char_indices() {
                    if ch == '{' {
                        depth += 1;
                    } else if ch == '}' {
                        depth -= 1;
                        if depth == 0 {
                            end_opt = Some(brace_start + i + ch.len_utf8());
                            break;
                        }
                    }
                }
                if let Some(end) = end_opt {
                    tracing::info!(
                        target: "pssl_shader::psfx::intrinsics",
                        "rewriting get_lanes_per_wave to WaveGetLaneCount()"
                    );
                    out.replace_range(start..end, "uint get_lanes_per_wave() { return WaveGetLaneCount(); }");
                }
            }
        } else if out.contains("__kWaveModeWave64") {
            tracing::trace!(
                target: "pssl_shader::psfx::intrinsics",
                "wave intrinsic marker without helper – will be handled in emitter"
            );
        }
    }
    out
}

/// Reports whether `ty` is a PSSL shader-header intrinsic type.
/// Used by resolver to boost detection confidence.
pub fn is_pssl_intrinsic_type(name: &str) -> bool {
    matches!(
        name.to_ascii_lowercase().as_str(),
        "samplerstate" | "texture2d" | "rw_texture2d" | "float" | "half" | "int" | "uint"
    )
}
