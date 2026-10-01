//! One-header translator – `header_src → virtual hlsl_src`.
//!
//! Pure function. All header-owned transforms live here; shader-owned
//! concerns (SRT flattening, entry-point binding) are *not* handled here.

use std::path::Path;

use crate::psfx::{cache, graph::HeaderDag, intrinsics, resources, rules};

/// Result of translating a single header.
#[derive(Debug, Clone)]
pub struct TranslatedHeader {
    pub path: std::path::PathBuf,
    pub original_len: usize,
    pub translated: String,
}

/// Translates all headers in `dag` to virtual HLSL, writing best-effort debug
/// copies into `cache/*/` (git-ignored). Returns the map `path → translated`.
#[tracing::instrument(skip_all, fields(dag_nodes = dag.nodes.len()))]
pub fn translate_dag(dag: &HeaderDag) -> std::collections::HashMap<std::path::PathBuf, String> {
    let key = cache::cache_key(dag);
    tracing::info!(
        target: "pssl_shader::psfx::translator",
        dag_len = dag.nodes.len(),
        dag_hash = %key.dag_hash,
        "translating DAG"
    );
    let mut out = std::collections::HashMap::new();
    for p in &dag.topo_order {
        if let Some(node) = dag.nodes.get(p) {
            let tr = translate_header(p, &node.content);
            cache::write_debug_cache(p, &tr.translated, &key);
            out.insert(p.clone(), tr.translated);
        }
    }
    out
}

/// Translates a single header's `src` to virtual HLSL.
#[tracing::instrument(skip_all, fields(path = %path.display(), src_len = src.len()))]
pub fn translate_header(path: &Path, src: &str) -> TranslatedHeader {
    let orig = src.len();
    // Strip a leading byte-order mark before the rule pipeline runs; the
    // preamble it prepends would otherwise bury the BOM mid-stream where
    // downstream strip checks (index 0 only) cannot reach it.
    let src = src.strip_prefix('\u{FEFF}').unwrap_or(src);
    let mut cur = src.to_string();
    tracing::trace!(target: "pssl_shader::psfx::translator", "starting translation");
    cur = rules::apply_rules(cur);
    cur = resources::translate_sampler_factories(cur);
    cur = intrinsics::translate_wave_intrinsics(cur);
    cur = intrinsics::translate_minmax3(cur);
    cur = intrinsics::translate_m_pi(cur);
    tracing::info!(
        target: "pssl_shader::psfx::translator",
        path = %path.display(),
        orig_len = orig,
        new_len = cur.len(),
        "header translated"
    );
    TranslatedHeader {
        path: path.to_path_buf(),
        original_len: orig,
        translated: cur,
    }
}
