//! Resource/sampler translation – header-owned.
//!
//! AGC `sce::Agc::Core::Sampler` construction is NOT shader algorithm code.
//! In Vulkan HLSL these are externally bound `SamplerState` objects.

/// Translates `sce::Agc::Core::Sampler` factory helpers
/// (`make_point_sampler` / `make_bilinear_sampler` / `make_trilinear_sampler`)
/// to externally bound samplers. The original framework body that calls
/// `sampler_desc.init()` / `setWrapMode` etc. is elided; the consumer shader
/// will sample via `g_bilinear_sampler` etc. supplied by the binding layer.
///
/// For `make_bilinear_sampler` we elide entirely – `src/emit/hlsl.rs` already
/// emits `make_bilinear_sampler_shim` unconditionally, so keeping the factory
/// would cause DXC `redefinition of 'make_bilinear_sampler'`.
#[tracing::instrument(skip_all, fields(src_len = src.len()))]
pub fn translate_sampler_factories(mut src: String) -> String {
    // AGC factories are shader-algorithm-irrelevant; they construct
    // `sce::Agc::Core::Sampler` which has no Vulkan/DXC equivalent.
    // Elide all three entirely – Vulkan samplers are externally bound as
    // `SamplerState g_point_sampler` / `g_bilinear_sampler` /
    // `g_trilinear_sampler` (emitted by `emit_intrinsic_shims`). Call sites
    // `make_bilinear_sampler()` are rewritten to `g_bilinear_sampler` in the
    // HLSL emitter (see `src/emit/hlsl.rs`).
    for name in [
        "make_point_sampler",
        "make_bilinear_sampler",
        "make_trilinear_sampler",
    ] {
        let needle = format!("constexpr SamplerState {name}()");
        if let Some(start) = src.find(&needle) {
            if let Some(rel) = src[start..].find('{') {
                let brace_start = start + rel;
                let mut depth = 0usize;
                let mut end_opt = None;
                for (i, ch) in src[brace_start..].char_indices() {
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
                        target: "pssl_shader::psfx::resources",
                        factory = %name,
                        "eliding AGC sampler factory – Vulkan sampler will be externally bound"
                    );
                    src.replace_range(
                        start..end,
                        &format!("/* {name} elided – use g_{} */", name.trim_start_matches("make_")),
                    );
                }
            }
        }
    }
    src
}
