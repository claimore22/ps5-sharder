//! Header dependency DAG – builds and topologically sorts the transitive
//! `psfx_*.h` include graph. Every node is a *PSSL shader header* detected by
//! content (see `resolver::is_pssl_shader_header`), not by filename suffix.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

/// A node in the header DAG – the on-disk header path plus its raw content.
#[derive(Debug, Clone)]
pub struct HeaderNode {
    pub path: PathBuf,
    pub content: String,
    pub includes: Vec<String>,
}

/// The full transitive header DAG rooted at a shader's entry point.
#[derive(Debug, Default)]
pub struct HeaderDag {
    pub nodes: HashMap<PathBuf, HeaderNode>,
    pub topo_order: Vec<PathBuf>,
}

impl HeaderDag {
    pub fn len(&self) -> usize {
        self.nodes.len()
    }
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

/// Builds the DAG by walking `#include` directives starting from `entry`.
/// Only headers that satisfy `is_pssl_shader_header` are retained as nodes;
/// plain C headers (`<cstdint>`, `<stdarg.h>`) are ignored.
#[tracing::instrument(skip_all, fields(entry = %entry.display()))]
pub fn build_dag(
    entry: &Path,
    entry_content: &str,
    search_roots: &[PathBuf],
) -> HeaderDag {
    tracing::trace!(target: "pssl_shader::psfx::graph", "building DAG");
    let mut dag = HeaderDag::default();
    let mut visited = HashSet::new();
    let mut stack = Vec::new();
    visit_entry(entry, entry_content, search_roots, &mut dag, &mut visited, &mut stack);
    dag.topo_order = topological_sort(&dag);
    tracing::info!(
        target: "pssl_shader::psfx::graph",
        nodes = dag.nodes.len(),
        topo_len = dag.topo_order.len(),
        "DAG built"
    );
    for p in &dag.topo_order {
        tracing::trace!(target: "pssl_shader::psfx::graph", path = %p.display(), "topo node");
    }
    dag
}

fn visit_entry(
    path: &Path,
    content: &str,
    search_roots: &[PathBuf],
    dag: &mut HeaderDag,
    visited: &mut HashSet<PathBuf>,
    stack: &mut Vec<PathBuf>,
) {
    if visited.contains(path) {
        return;
    }
    visited.insert(path.to_path_buf());
    let includes = extract_includes(content);
    for inc in &includes {
        if let Some(resolved) = crate::psfx::resolver::resolve_header(inc, path, search_roots) {
            if let Ok(src) = std::fs::read_to_string(&resolved) {
                if crate::psfx::resolver::is_pssl_shader_header(&src) {
                    tracing::trace!(
                        target: "pssl_shader::psfx::graph",
                        include = %inc,
                        resolved = %resolved.display(),
                        "PSSL header include"
                    );
                    stack.push(resolved.clone());
                    visit_entry(&resolved, &src, search_roots, dag, visited, stack);
                    stack.pop();
                    let node = HeaderNode {
                        path: resolved.clone(),
                        content: src.clone(),
                        includes: extract_includes(&src),
                    };
                    dag.nodes.insert(resolved, node);
                } else {
                    tracing::trace!(
                        target: "pssl_shader::psfx::graph",
                        include = %inc,
                        resolved = %resolved.display(),
                        "non-PSSL header, skipping"
                    );
                }
            }
        }
    }
}

fn extract_includes(content: &str) -> Vec<String> {
    content
        .lines()
        .filter_map(|l| {
            let t = l.trim();
            if t.starts_with("#include") {
                t.split('"')
                    .nth(1)
                    .or_else(|| t.split('<').nth(1).and_then(|s| s.split('>').next()))
                    .map(|s| s.to_string())
            } else {
                None
            }
        })
        .collect()
}

fn topological_sort(dag: &HeaderDag) -> Vec<PathBuf> {
    // Cheap DFS post-order; size is tiny (≤ ~10 headers).
    let mut order = Vec::new();
    let mut seen = HashSet::new();
    for k in dag.nodes.keys() {
        dfs(k, dag, &mut seen, &mut order);
    }
    order
}

fn dfs(path: &Path, dag: &HeaderDag, seen: &mut HashSet<PathBuf>, order: &mut Vec<PathBuf>) {
    if seen.contains(path) {
        return;
    }
    seen.insert(path.to_path_buf());
    if let Some(node) = dag.nodes.get(path) {
        for inc in &node.includes {
            // Map include string back to resolved path if present in DAG.
            if let Some(resolved) = dag
                .nodes
                .keys()
                .find(|p| p.file_name().map(|f| f == inc.as_str()).unwrap_or(false))
            {
                dfs(resolved, dag, seen, order);
            }
        }
    }
    order.push(path.to_path_buf());
}
