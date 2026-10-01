//! Cache – deterministic, content-addressed, not committed.
//!
//! `cache/<identity>/<dag-hash>/*.hlsl` is `.gitignored` debug-only.
//! Source of truth remains `header source + translator rules version`.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

use crate::psfx::{HeaderDag, TOOLCHAIN_IDENTITY, TRANSLATOR_VERSION};

/// Cache key components.
#[derive(Debug, Clone)]
pub struct CacheKey {
    pub toolchain: String,
    pub rules_version: String,
    pub dag_hash: String,
}

pub fn cache_key(dag: &HeaderDag) -> CacheKey {
    let mut hasher = DefaultHasher::new();
    // Hash every transitive header's content in topo order (already DAG-hash).
    for p in &dag.topo_order {
        if let Some(node) = dag.nodes.get(p) {
            node.path.hash(&mut hasher);
            node.content.hash(&mut hasher);
        }
    }
    // Include toolchain identity + rules version so a rules bump invalidates.
    TOOLCHAIN_IDENTITY.hash(&mut hasher);
    TRANSLATOR_VERSION.hash(&mut hasher);
    let dag_hash = format!("{:016x}", hasher.finish());
    tracing::trace!(
        target: "pssl_shader::psfx::cache",
        dag_len = dag.nodes.len(),
        dag_hash = %dag_hash,
        "cache key computed"
    );
    CacheKey {
        toolchain: TOOLCHAIN_IDENTITY.to_string(),
        rules_version: TRANSLATOR_VERSION.to_string(),
        dag_hash,
    }
}

/// Path for the debug cache: `cache/<toolchain-identity>/<dag-hash>/` (git-ignored).
pub fn cache_dir(key: &CacheKey) -> PathBuf {
    PathBuf::from("cache").join(format!("toolchain-{}", key.toolchain)).join(&key.dag_hash)
}

/// Writes a translated virtual header into the debug cache for inspection.
/// Failures are non-fatal (best-effort).
#[tracing::instrument(skip_all, fields(path = %path.display()))]
pub fn write_debug_cache(path: &Path, translated: &str, key: &CacheKey) {
    let dir = cache_dir(key);
    if let Err(e) = std::fs::create_dir_all(&dir) {
        tracing::warn!(target: "pssl_shader::psfx::cache", error = %e, "create cache dir failed");
        return;
    }
    let file = dir.join(
        path.file_name()
            .map(|f| format!("{}.hlsl", f.to_string_lossy()))
            .unwrap_or_else(|| "header.hlsl".to_string()),
    );
    if let Err(e) = std::fs::write(&file, translated) {
        tracing::warn!(target: "pssl_shader::psfx::cache", error = %e, path = %file.display(), "write cache failed");
    } else {
        tracing::trace!(target: "pssl_shader::psfx::cache", path = %file.display(), "cache written");
    }
}
