//! PSSL → HLSL textual rules – header-owned.
//!
//! These are pure content transforms applied to framework header source before it is
//! handed to the parser. Every rule is traced.

/// Applies all header rules to `src` in a deterministic order.
#[tracing::instrument(skip_all, fields(src_len = src.len()))]
pub fn apply_rules(mut src: String) -> String {
    let orig_len = src.len();
    src = rule_cnd_mask(src);
    src = rule_rcp(src);
    src = rule_wave_intrinsics(src);
    src = rule_constexpr_cleanup(src);
    tracing::trace!(
        target: "pssl_shader::psfx::rules",
        orig_len,
        new_len = src.len(),
        "rules applied"
    );
    src
}

#[tracing::instrument(skip_all)]
fn rule_cnd_mask(src: String) -> String {
    // `CndMask` is lowered to ternary `(cond ? a : b)` in `src/emit/hlsl.rs`
    // and recognised as builtin `cndmask` in `src/semantic/mod.rs`. Headers
    // that inline it need no text rewrite – just trace.
    if src.contains("CndMask") {
        tracing::trace!(target: "pssl_shader::psfx::rules", "CndMask present – will be lowered in emitter");
    }
    src
}

#[tracing::instrument(skip_all)]
fn rule_rcp(src: String) -> String {
    if src.contains("rcp(") {
        tracing::trace!(target: "pssl_shader::psfx::rules", "rcp rule: will be lowered to (1.0/x) in emitter");
    }
    src
}

#[tracing::instrument(skip_all)]
fn rule_wave_intrinsics(src: String) -> String {
    if src.contains("__get_wavemode") {
        tracing::trace!(
            target: "pssl_shader::psfx::rules",
            "wave intrinsic marker found – translator will stub/replace"
        );
    }
    src
}

#[tracing::instrument(skip_all)]
fn rule_constexpr_cleanup(src: String) -> String {
    // Headers use `constexpr` liberally; parser already handles it as storage
    // qualifier (`src/parser/stmt.rs:209`). No rewrite needed today, but trace.
    if src.contains("constexpr") {
        tracing::trace!(target: "pssl_shader::psfx::rules", "constexpr present");
    }
    src
}
