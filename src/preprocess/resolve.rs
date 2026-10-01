//! Controlled `#include` resolver.
//!
//! Resolves includes by class + configured roots; never a whole-filesystem search.

use std::path::{Path, PathBuf};

use super::classify::classify_include;
use super::report::IncludeResolution;

/// Resolves an include to a file on disk, or classifies an unresolvable one.
pub(crate) fn resolve_include(
    name: &str,
    current_dir: Option<&Path>,
    config: &super::PreprocessorConfig,
) -> IncludeResolution {
    let class = classify_include(name);
    let is_angle = name.starts_with('<');

    // C/C++ leaf infrastructure headers are NEVER inlined, *even if a real file of
    // that name exists* in the framework include tree. Inlining e.g. `#include <cstdint>`
    // would drag in the entire C++ standard library. We classify these by category
    // (C/C++ stdlib). HostApi anchors (e.g. `agc.h`) are NOT early-ignored here —
    // they are resolved and recorded but their subtree is skipped by the caller.
    if class.is_leaf_host() {
        return IncludeResolution::Ignored(class);
    }

    // 1. Directory of the current file.
    if let Some(dir) = current_dir {
        let candidate = dir.join(name);
        if candidate.exists() {
            return IncludeResolution::Found(candidate);
        }
        // 1b. Ancestor directories for subpath includes (`#include
        // "shaders/lighting.pssli"` from `<proj>/ibl/x.pssl` must bind the
        // project-local `<proj>/shaders/lighting.pssli`, not an unrelated
        // same-subpath header under an include root — C quote-include
        // semantics). Bare basenames keep the existing roots behaviour
        // below. Bounded (4 levels) and never searching past an enclosing
        // project root (`Cargo.toml`), so test-tree copies cannot escape
        // into unrelated repository files.
        if name.contains('/') || name.contains('\\') {
            if let Some(found) = find_in_ancestors(dir, name) {
                return IncludeResolution::Found(found);
            }
        }
    }

    // 2. PSSL/engine/project include roots.
    for root in &config.include_dirs {
        if let Some(p) = try_in_root(root, name) {
            return IncludeResolution::Found(p);
        }
    }

    // 3. System/sysroot include roots.
    for root in &config.system_include_dirs {
        if let Some(p) = try_in_root(root, name) {
            return IncludeResolution::Found(p);
        }
    }

    // 4. Bare-basename lookup scoped to the configured include roots.
    //
    // `tests/shaders/<copied>/shader.pssl` frequently `#include "V2P.h"` where
    // `V2P.h` lives elsewhere in the framework tree. We resolve it by searching *only
    // the explicit `include_dirs` roots* (deterministic, never arbitrary files
    // elsewhere on disk), preferring an exact path-suffix match.
    if !name.contains('/') && !name.contains('\\') {
        if let Some(p) = find_in_roots(&config.include_dirs, name) {
            return IncludeResolution::Found(p);
        }
    }

    // Nothing found on disk. Host-infra headers (`allow_missing_system_includes`)
    // are handled by the caller; everything else is a real error. The old
    // whole-filesystem basename search is intentionally gone: it could resolve
    // unrelated headers (e.g. `foo/bar/types.h` → `other/types.h`) which is
    // unsafe for a deterministic compiler.
    let reason = if is_angle {
        format!("angle include {name:?} not found in system roots")
    } else {
        format!("include {name:?} not found relative to source or configured roots")
    };
    IncludeResolution::NotFound(class, reason)
}

/// Searches ancestor directories of `dir` (parent, grandparent, …) for a
/// relative `name`, bounding the walk and stopping before any directory
/// that owns the tree (`Cargo.toml` marker) so resolution can neither
/// escape the project nor bind repository-top files by accident.
fn find_in_ancestors(dir: &Path, name: &str) -> Option<PathBuf> {
    let mut current = dir.parent();
    for _ in 0..4 {
        let Some(ancestor) = current else {
            break;
        };
        if ancestor.join("Cargo.toml").exists() {
            break;
        }
        let candidate = ancestor.join(name);
        if candidate.is_file() {
            return Some(candidate);
        }
        current = ancestor.parent();
    }
    None
}

/// Attempts a deterministic basename lookup across only the given roots.
/// When multiple files share `basename`, prefers a PSSL shader header (content
/// contains `__PSSL__`/`SamplerState`/`sce::Agc` etc.) over a plain C++ header.
/// This disambiguates `psfx_common.h` which exists as both
/// `common/include/psfx/psfx_common.h` (C++ namespace) and
/// `common/source/psfx/shaders/psfx_common.h` (PSSL helpers with
/// `make_*_sampler`/`decode_linear_z`).
fn find_in_roots(roots: &[PathBuf], basename: &str) -> Option<PathBuf> {
    let mut candidates = Vec::new();
    for root in roots {
        collect_by_basename(root, basename, &mut candidates);
    }
    if candidates.is_empty() {
        return None;
    }
    if candidates.len() == 1 {
        return Some(candidates.into_iter().next().unwrap());
    }
    // Prefer PSSL header when ambiguous
    let mut pssl_candidates = Vec::new();
    let mut other = Vec::new();
    for p in candidates {
        if let Ok(src) = std::fs::read_to_string(&p) {
            if crate::psfx::resolver::is_pssl_shader_header(&src) {
                pssl_candidates.push(p);
            } else {
                other.push(p);
            }
        } else {
            other.push(p);
        }
    }
    // If any PSSL header exists, prefer it; tie-break by shortest path (most specific)
    // which for `psfx_common.h` prefers `.../shaders/psfx_common.h` (96 lines, PSSL)
    // over `.../include/psfx/psfx_common.h` (C++).
    if !pssl_candidates.is_empty() {
        pssl_candidates.sort_by_key(|p| p.to_string_lossy().len());
        return Some(pssl_candidates.into_iter().next().unwrap());
    }
    other.sort_by_key(|p| p.to_string_lossy().len());
    Some(other.into_iter().next().unwrap())
}

fn collect_by_basename(dir: &Path, basename: &str, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if let Some(n) = path.file_name().and_then(|n| n.to_str()) {
                if n == ".git" {
                    continue;
                }
            }
            collect_by_basename(&path, basename, out);
        } else if path.file_name().map(|n| n == basename).unwrap_or(false) {
            out.push(path);
        }
    }
}

fn find_by_basename(dir: &Path, basename: &str) -> Option<PathBuf> {
    let mut candidates = Vec::new();
    collect_by_basename(dir, basename, &mut candidates);
    candidates.into_iter().next()
}

/// Tries to resolve `name` under `root`, allowing `name` to carry a subpath.
fn try_in_root(root: &Path, name: &str) -> Option<PathBuf> {
    let cand = root.join(name);
    if cand.is_file() {
        return Some(cand);
    }
    // Allow the caller to pass a subpath (`graphics/common/foo.h`) that lives
    // under the root; match by normalized suffix so we never bind to an arbitrary
    // same-basename file elsewhere.
    if name.contains('/') || name.contains('\\') {
        let normalized = name.replace('\\', "/");
        if let Some(found) = find_by_suffix(root, &normalized) {
            return Some(found);
        }
    }
    None
}

/// Deterministically locates a header by exact normalized path suffix under `root`.
fn find_by_suffix(dir: &Path, suffix: &str) -> Option<PathBuf> {
    let entries = std::fs::read_dir(dir).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if let Some(n) = path.file_name().and_then(|n| n.to_str()) {
                if n == ".git" {
                    continue;
                }
            }
            if let Some(p) = find_by_suffix(&path, suffix) {
                return Some(p);
            }
        } else {
            let rel = path.to_string_lossy().replace('\\', "/");
            if rel.ends_with(suffix) {
                return Some(path);
            }
        }
    }
    None
}
