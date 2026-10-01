//! Post-preprocessing include report and dependency diagnostics.

use std::collections::HashSet;
use std::path::PathBuf;

use super::classify::IncludeClass;

/// One `#include` observed during preprocessing, with its resolution outcome.
#[derive(Debug, Clone)]
pub struct IncludeEvent {
    /// The include name as written (e.g. `"psfx_common.h"` or `<cstdint>`).
    pub name: String,
    /// Language-family classification.
    pub class: IncludeClass,
    /// Where the include came from; last element is the deepest origin.
    pub resolution: IncludeResolution,
    /// Chain of including file names (empty for top-level source).
    pub provenance: Vec<String>,
}

/// Outcome of resolving a single include.
#[derive(Debug, Clone)]
pub enum IncludeResolution {
    /// Resolved to a real file on disk and preprocessed.
    Found(PathBuf),
    /// Classified as C/C++ host infra; intentionally not inlined.
    Ignored(IncludeClass),
    /// Could not be resolved on disk.
    NotFound(IncludeClass, String),
}

/// Post-preprocessing include-classification report.
#[derive(Debug, Default, Clone)]
pub struct PreprocessReport {
    /// Every include encountered, in encounter order (deepest-first within nesting).
    pub includes: Vec<IncludeEvent>,
    /// Count of includes that resolved to a real shader file and were expanded.
    pub resolved: usize,
    /// Alias for `resolved` (shader headers actually inlined).
    pub resolved_shader: usize,
    /// Count of host API chains skipped at an anchor (e.g. `agc.h`).
    pub host_api_chains_skipped: usize,
    /// Names of host API anchors whose subtrees were skipped.
    pub host_api_chain_names: Vec<String>,
    /// Count of C/C++ infrastructure headers ignored as leaves.
    pub ignored_host: usize,
    /// Count of shader-relevant includes that could not be resolved.
    pub unresolved_shader: usize,
    /// Unique names of C/C++ infrastructure headers that were ignored (sorted).
    pub ignored_names: Vec<String>,
    /// Warnings about shader symbols whose providing header was ignored.
    pub dependency_warnings: Vec<String>,
}

impl PreprocessReport {
    pub(crate) fn finalize(&mut self) {
        self.resolved = 0;
        self.resolved_shader = 0;
        self.host_api_chains_skipped = 0;
        self.host_api_chain_names.clear();
        self.ignored_host = 0;
        self.unresolved_shader = 0;
        let mut infra_set: HashSet<String> = HashSet::new();
        for ev in &self.includes {
            match &ev.resolution {
                IncludeResolution::Found(_) => {
                    // HostApi anchors are recorded as Found but counted as skipped
                    if matches!(ev.class, IncludeClass::HostApi(_)) {
                        self.host_api_chains_skipped += 1;
                        self.host_api_chain_names.push(ev.name.clone());
                    } else {
                        self.resolved += 1;
                        self.resolved_shader += 1;
                    }
                }
                IncludeResolution::Ignored(class) => {
                    if matches!(class, IncludeClass::HostApi(_)) {
                        self.host_api_chains_skipped += 1;
                        self.host_api_chain_names.push(ev.name.clone());
                    } else if class.is_leaf_host() {
                        infra_set.insert(ev.name.clone());
                    } else {
                        // Shader re-include guard suppress (cycle) — not counted
                    }
                }
                IncludeResolution::NotFound(class, _) if class.is_host() => {
                    if matches!(class, IncludeClass::HostApi(_)) {
                        self.host_api_chains_skipped += 1;
                        self.host_api_chain_names.push(ev.name.clone());
                    } else {
                        infra_set.insert(ev.name.clone());
                    }
                }
                IncludeResolution::NotFound(..) => self.unresolved_shader += 1,
            }
        }
        // Deduplicate host chain names as well
        {
            let mut seen = HashSet::new();
            self.host_api_chain_names.retain(|n| seen.insert(n.clone()));
            self.host_api_chain_names.sort();
        }
        let mut infra: Vec<String> = infra_set.into_iter().collect();
        infra.sort();
        self.ignored_names = infra.clone();
        self.ignored_host = infra.len();
    }
}

/// Result of a diagnostic-rich preprocessing run.
#[derive(Debug, Clone)]
pub struct PreprocessOutput {
    /// The expanded PSSL source.
    pub source: String,
    /// Include classifier report.
    pub report: PreprocessReport,
}

/// Symbols that, if present *directly* in the shader-authored source, indicate
/// the shader references a type/macro whose providing host header we ignored.
///
/// We scan the **root shader source** (not the expanded output) so that a symbol
/// used inside an inlined host header (e.g. `uint32_t` inside an anchor's
/// transitive headers) does not
/// produce a spurious warning. Layered, not exhaustive.
const CRITICAL_C_SYMBOLS: &[&str] = &[
    "uint8_t", "uint16_t", "uint32_t", "uint64_t", "int8_t", "int16_t", "int32_t", "int64_t",
    "uintptr_t", "intptr_t", "size_t", "ptrdiff_t", "va_list", "va_start", "va_arg", "va_end",
    "INT8_MAX", "INT16_MAX", "INT32_MAX", "INT64_MAX", "UINT8_MAX", "UINT16_MAX", "UINT32_MAX",
    "UINT64_MAX", "SIZE_MAX", "UINT_MAX", "INT_MAX", "NULL",
];

fn strip_comments_for_scan(src: &str) -> String {
    // Minimal comment stripping: remove `//` line comments and `/* */` blocks.
    // Good enough to avoid `// uint32_t` false positives.
    let mut out = String::with_capacity(src.len());
    let bytes = src.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if i + 1 < bytes.len() && bytes[i] == b'/' && bytes[i + 1] == b'/' {
            // line comment
            i += 2;
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
        } else if i + 1 < bytes.len() && bytes[i] == b'/' && bytes[i + 1] == b'*' {
            i += 2;
            while i + 1 < bytes.len() && !(bytes[i] == b'*' && bytes[i + 1] == b'/') {
                i += 1;
            }
            if i + 1 < bytes.len() {
                i += 2;
            }
        } else {
            out.push(bytes[i] as char);
            i += 1;
        }
    }
    out
}

/// Scans the shader-authored source for symbols provided by ignored host headers
/// and populates `report.dependency_warnings` when found.
pub(crate) fn check_dependency_warnings(shader_authored_src: &str, report: &mut PreprocessReport) {
    if report.ignored_names.is_empty() {
        return;
    }
    let scanned = strip_comments_for_scan(shader_authored_src);
    let mut found: Vec<&str> = Vec::new();
    for sym in CRITICAL_C_SYMBOLS {
        if has_word(&scanned, sym) {
            found.push(sym);
        }
    }
    if found.is_empty() {
        return;
    }
    // De-duplicated, sorted ignored list (at most a handful unique names).
    let ignored_list = report.ignored_names.join(", ");
    report.dependency_warnings = found
        .iter()
        .map(|s| {
            format!(
                "shader references `{s}` but its providing C/C++ header was classified as host-only: {ignored_list}"
            )
        })
        .collect();
}

fn has_word(haystack: &str, needle: &str) -> bool {
    let n = needle.as_bytes();
    let bytes = haystack.as_bytes();
    let mut i = 0;
    while i + n.len() <= haystack.len() {
        if &bytes[i..i + n.len()] == n
            && is_word_boundary_before(bytes, i)
            && is_word_boundary_after(bytes, i + n.len())
        {
            return true;
        }
        i += 1;
    }
    false
}

fn is_word_boundary_before(bytes: &[u8], pos: usize) -> bool {
    if pos == 0 {
        return true;
    }
    let prev = bytes[pos - 1] as char;
    !(prev.is_ascii_alphanumeric() || prev == '_')
}

fn is_word_boundary_after(bytes: &[u8], pos: usize) -> bool {
    if pos >= bytes.len() {
        return true;
    }
    let next = bytes[pos] as char;
    !(next.is_ascii_alphanumeric() || next == '_')
}
