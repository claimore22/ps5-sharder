//! PSSL preprocessor: include resolution, classification, and directive handling.
//!
//! Twinned with the [`crate::parser`] module, this is *not* a general C
//! preprocessor. It handles the subset of directives the PSSL corpus uses:
//! `#include`, object-like and function-like `#define` (fixed-arity,
//! `##`-paste), `#undef`, and basic `#ifdef`/`#ifndef`/`#if`
//! conditional blocks.
//!
//! The pipeline is:
//!   [`preprocess_with_config`] / [`preprocess`] -> expanded PSSL text, with
//!   [`preprocess_reported`] additionally returning a full include report.

pub mod classify;
mod cond;
mod directive;
mod report;
mod resolve;
#[cfg(test)]
mod tests;

pub use classify::{
    IncludeClass, HostApiFamily, IncludeKind, classify_include, include_kind,
    is_ignorable_system_include,
};
pub use report::{IncludeEvent, IncludeResolution, PreprocessOutput, PreprocessReport};

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::error::{Phase, Ps5ShaderError, Result};

use cond::eval_cond;
use directive::{
    expand_defines, expand_func_defines, is_define_directive, is_include_directive, parse_define,
    parse_func_define, parse_include_name, parse_undef,
};
use report::check_dependency_warnings;
use resolve::resolve_include;

const MAX_INCLUDE_DEPTH: usize = 16;

/// Configuration for include resolution.
///
/// Resolution order (first match wins):
///   1. directory of the current file
///   2. `include_dirs` (PSSL engine/project include roots)
///   3. `system_include_dirs` (toolchain/sysroot include roots)
///   4. known compatibility headers (skipped per `allow_missing_system_includes`)
#[derive(Debug, Clone, Default)]
pub struct PreprocessorConfig {
    /// PSSL / engine / project include roots (`#include "..."` search path).
    pub include_dirs: Vec<PathBuf>,
    /// C/C++ toolchain/sysroot include roots (`#include <...>` search path).
    pub system_include_dirs: Vec<PathBuf>,
    /// When `true`, a missing `System` compatibility header is dropped silently.
    pub allow_missing_system_includes: bool,
}

impl PreprocessorConfig {
    /// A config with the platform include roots commonly needed by PSSL shaders.
    ///
    /// These are explicit, deterministic roots — not a global basename search.
    pub fn platform_defaults() -> Self {
        let mut cfg = PreprocessorConfig {
            include_dirs: Vec::new(),
            system_include_dirs: Vec::new(),
            allow_missing_system_includes: true,
        };
        for p in platform_roots() {
            if p.exists() {
                cfg.include_dirs.push(p);
            }
        }
        cfg
    }
}

/// The include trees searched (in order) for PSSL/include-common headers.
///
/// Platform include roots are supplied by the caller via
/// [`PreprocessorConfig::include_dirs`]; this hook exists so downstream
/// embedders can inject environment-specific roots. It returns an empty
/// list by default — no proprietary paths are hardcoded here.
pub fn platform_roots() -> Vec<PathBuf> {
    Vec::new()
}

/// Preprocesses PSSL source text with the default configuration.
///
/// Convenience wrapper around [`preprocess_with_config`].
pub fn preprocess(source: &str, file_path: Option<&Path>) -> Result<String> {
    let cfg = PreprocessorConfig::platform_defaults();
    preprocess_with_config(source, file_path, &cfg)
}

/// Preprocesses PSSL source text and returns both the expanded source and a
/// full include-classification report.
///
/// This is the diagnostic-rich entry point: it records every include (with its
/// provenance chain and resolution), classifies each into a language family, and
/// warns if the shader references symbols that an ignored host header would have
/// provided. The expanded source keeps line structure so diagnostics stay useful.
pub fn preprocess_reported(
    source: &str,
    file_path: Option<&Path>,
    config: &PreprocessorConfig,
) -> Result<PreprocessOutput> {
    let source = strip_bom(source);
    let mut defines: HashMap<String, String> = HashMap::new();
    defines.insert("__PSSL__".to_string(), "1".to_string());    // Target platform: the corpus is console code, so platform branches
    // guarded by `__PROSPERO__` (current-gen devkit) are live. (`__ORBIS__` is prior-gen
    // and stays undefined.)
    defines.insert("__PROSPERO__".to_string(), "1".to_string());
    // The PSSL compiler predefines `__WAVE__` for shader compilation:
    // GPU wave headers gate wave-intrinsic helpers behind it
    // (only `#if defined(__WAVE__)` includes and `BasicGsOut`-style
    // structs behind `#ifdef __WAVE__`). Without it, wave-capable shaders
    // lose structs their entry points reference. Only 3 framework headers test
    // it, all in one geometry-optimization tutorial family, so the blast
    // radius is tiny and exactly the shaders that need it.
    defines.insert("__WAVE__".to_string(), "1".to_string());
    // Function-like macros (`#define GBUFFER_OP(slot, ...) ...`, `##`
    // pastes) live in a separate map; `#undef` drops from both.
    let mut func_defines: HashMap<String, (Vec<String>, String)> = HashMap::new();
    // The "mark parameter used" suppressor (`PSSL_UNUSED(odd);` in
    // psfx headers): the PSSL toolchain provides it, the headers never
    // define it. Empty expansion (a bare `;` parses as an empty
    // statement; `((void)(x))` would break if the argument itself is
    // undefined). Only this exact idiom-name is predefined.
    func_defines.insert("PSSL_UNUSED".to_string(), (vec!["x".to_string()], "".to_string()));
    let mut include_stack: Vec<PathBuf> = Vec::new();
    let mut report = PreprocessReport::default();
    let mut provenance: Vec<String> = Vec::new();
    let current_dir = effective_root_dir(file_path);
    let expanded = preprocess_inner(
        &source,
        current_dir.as_deref(),
        &mut defines,
        &mut func_defines,
        &mut include_stack,
        &mut report,
        &mut provenance,
        0,
        config,
    )?;
    report.finalize();
    // Dependency diagnostics: only warn if a symbol appears *directly* in the
    // shader-authored source (the root file), not inside an inlined host header.
    // `source` here is the original (pre-expansion) root input, so scanning it
    // avoids false positives from `agc.h`-style host API that was never inlined.
    let shader_authored = strip_bom(source);
    check_dependency_warnings(shader_authored, &mut report);
    Ok(PreprocessOutput {
        source: expanded,
        report,
    })
}

/// Preprocesses PSSL source text.
///
/// Handles:
/// * UTF-8 BOM stripping
/// * `#include "file"` / `#include <file>` — configurable resolver, depth-limited, cycle-detected
/// * `#define NAME value` — object-like, expanded on subsequent lines
/// * `#define NAME(p, ...) body` — function-like (fixed arity, `##`
///   paste), expanded with balanced-arg parsing; `#undef` drops both
/// * `#ifdef` / `#ifndef` / `#if` / `#else` / `#elif` / `#endif` — basic conditional blocks
///
/// This is NOT a C preprocessor.
pub fn preprocess_with_config(
    source: &str,
    file_path: Option<&Path>,
    config: &PreprocessorConfig,
) -> Result<String> {
    preprocess_reported(source, file_path, config).map(|o| o.source)
}

fn strip_bom(source: &str) -> &str {
    source.strip_prefix('\u{FEFF}').unwrap_or(source)
}

/// The directory relative includes in the *root* file resolve against.
///
/// Normally this is the containing directory of the file being preprocessed.
/// Corpus copies under `tests/shaders/<name>/shader.pssl` are flat — their
/// `../include/...` includes would otherwise point at `tests/shaders/include`.
/// Those copies are never simplified; a sibling `ORIGIN.txt` records the original
/// path they were copied from. When present, we anchor the root file at that
/// recorded path's directory, so relative includes resolve in the actual framework
/// tree exactly as they did before the copy. Nested includes are unaffected:
/// they already switch to the on-disk header's own directory.
fn effective_root_dir(file_path: Option<&Path>) -> Option<PathBuf> {
    let copy_dir = file_path.and_then(|p| p.parent());
    if let Some(dir) = copy_dir {
        if let Ok(text) = std::fs::read_to_string(dir.join("ORIGIN.txt")) {
            if let Some(line) = text.lines().map(str::trim).find(|l| !l.is_empty()) {
                let origin = Path::new(line);
                if let Some(parent) = origin.parent() {
                    if parent.is_dir() {
                        return Some(parent.to_path_buf());
                    }
                }
            }
        }
    }
    copy_dir.map(Path::to_path_buf)
}

fn canonical(path: &Path) -> PathBuf {
    path.canonicalize().unwrap_or_else(|_| path.to_path_buf())
}

#[allow(clippy::too_many_arguments)]
fn preprocess_inner(
    source: &str,
    current_dir: Option<&Path>,
    defines: &mut HashMap<String, String>,
    func_defines: &mut HashMap<String, (Vec<String>, String)>,
    include_stack: &mut Vec<PathBuf>,
    report: &mut PreprocessReport,
    provenance: &mut Vec<String>,
    depth: usize,
    config: &PreprocessorConfig,
) -> Result<String> {
    use classify::classify_include;

    if depth > MAX_INCLUDE_DEPTH {
        return Err(Ps5ShaderError::new(
            format!("include depth exceeded {MAX_INCLUDE_DEPTH}"),
            Phase::Preprocess,
        ));
    }

    let source = strip_bom(source);
    let mut out = String::new();
    let mut cond_stack: Vec<bool> = Vec::new();
    let mut active = true;
    // Inside a `\`-continued directive (`#define`, `#if`, ...): continuation
    // lines are directive text, never source. For `#define`s we accumulate
    // the continuation into `pending_define` so multi-line function-like
    // macros register their full body; other continued directives (e.g.
    // multi-line `#if`) still just consume lines (each emits `\n` so line
    // numbers are preserved).
    let mut in_continuation = false;
    let mut pending_define: Option<String> = None;

    for line in source.lines() {
        let trimmed = line.trim_start();
        if in_continuation {
            out.push('\n');
            let ended = !line.trim_end().ends_with('\\');
            if let Some(buf) = pending_define.as_mut() {
                let piece = line.trim_end();
                buf.push(' ');
                buf.push_str(piece.trim_end_matches('\\').trim_end());
            }
            if ended {
                if let Some(buf) = pending_define.take() {
                    if let Some((name, params, body)) = parse_func_define(&buf) {
                        // Marker-gated registration (see
                        // `is_expandable_macro_body`): expression macros and
                        // free-function factories expand; stringize/C++
                        // operators/host intrinsics stay on the parser's
                        // macro-skip path as before multiline capture
                        // existed (Parse 1000/1000).
                        if is_expandable_macro_body(&body) {
                            func_defines.insert(name, (params, body));
                        }
                    } else if let Some((name, value)) = parse_define(&buf) {
                        // Same gate for object-like multiline bodies
                        // (`_SCE_MACRO_END` carries `__pragma`, which the
                        // PSSL parser never accepts).
                        if is_expandable_macro_body(&value) {
                            defines.insert(name, value);
                        }
                    }
                }
                in_continuation = false;
            }
            continue;
        }
        if trimmed.starts_with('#') && line.trim_end().ends_with('\\') {
            in_continuation = true;
            if is_define_directive(trimmed) {
                pending_define = Some(trimmed.trim_end().trim_end_matches('\\').trim_end().to_string());
            }
        }

        // Handle conditional compilation (`#ifdef` / `#ifndef` / `#if` / `#else` / `#elif` / `#endif`)
        if trimmed.starts_with("#ifdef") {
            let name = trimmed["#ifdef".len()..]
                .trim()
                .split_whitespace()
                .next()
                .unwrap_or("");
            let cond = defines.contains_key(name) || func_defines.contains_key(name);
            cond_stack.push(active);
            active = active && cond;
            out.push('\n');
            continue;
        }
        if trimmed.starts_with("#ifndef") {
            let name = trimmed["#ifndef".len()..]
                .trim()
                .split_whitespace()
                .next()
                .unwrap_or("");
            let cond = !defines.contains_key(name) && !func_defines.contains_key(name);
            cond_stack.push(active);
            active = active && cond;
            out.push('\n');
            continue;
        }
        if trimmed.starts_with("#if ") || trimmed == "#if" {
            let expr = trimmed[3..].trim();
            // NOTE: expression evaluation is intentionally limited to `defined(X)`
            // / `!defined(X)` guards and a leading `0`. A full C expression
            // evaluator is out of scope for this pass.
            let cond = eval_cond(expr, defines);
            cond_stack.push(active);
            active = active && cond;
            out.push('\n');
            continue;
        }
        if trimmed.starts_with("#else") {
            if let Some(parent) = cond_stack.last().copied() {
                // Flip the last condition, but respect parent active
                active = parent && !active;
            }
            out.push('\n');
            continue;
        }
        if trimmed.starts_with("#elif") {
            if let Some(parent) = cond_stack.last().copied() {
                let expr = trimmed["#elif".len()..].trim();
                let cond = eval_cond(expr, defines);
                active = parent && cond && !active;
            }
            out.push('\n');
            continue;
        }
        if trimmed.starts_with("#endif") {
            if let Some(parent) = cond_stack.pop() {
                active = parent;
            }
            out.push('\n');
            continue;
        }
        if !active {
            out.push('\n');
            continue;
        }

        if is_include_directive(trimmed) {
            let include_name = parse_include_name(trimmed).ok_or_else(|| {
                Ps5ShaderError::new(
                    format!("malformed #include directive: {line}"),
                    Phase::Preprocess,
                )
            })?;

            tracing::trace!(
                target: "pssl_shader::preprocess",
                include = %include_name,
                current = %current_dir.map(|d| d.display().to_string()).unwrap_or_default(),
                "resolving include"
            );

            let class = classify_include(&include_name);
            tracing::debug!(
                target: "pssl_shader::preprocess",
                include = %include_name,
                class = class.label(),
                provenance = ?provenance,
                "include classified"
            );

            // HostApi anchor: resolved and recorded, but subtree never expanded.
            // This collapses `psfx_common.h → agc.h → thousands of host headers`.
            if matches!(class, IncludeClass::HostApi(_)) {
                match resolve_include(&include_name, current_dir, config) {
                    IncludeResolution::Found(resolved) => {
                        report.includes.push(IncludeEvent {
                            name: include_name.clone(),
                            class,
                            resolution: IncludeResolution::Found(resolved.clone()),
                            provenance: provenance.clone(),
                        });
                        tracing::debug!(
                            target: "pssl_shader::preprocess",
                            include = %include_name,
                            class = class.label(),
                            resolved = %resolved.display(),
                            "host API anchor resolved; subtree skipped"
                        );
                        out.push('\n');
                        continue;
                    }
                    IncludeResolution::Ignored(_) => {
                        report.includes.push(IncludeEvent {
                            name: include_name.clone(),
                            class,
                            resolution: IncludeResolution::Ignored(class),
                            provenance: provenance.clone(),
                        });
                        tracing::debug!(
                            target: "pssl_shader::preprocess",
                            include = %include_name,
                            class = class.label(),
                            provenance = ?provenance,
                            "host API anchor not found; treated as empty"
                        );
                        out.push('\n');
                        continue;
                    }
                    IncludeResolution::NotFound(_, _) => {
                        report.includes.push(IncludeEvent {
                            name: include_name.clone(),
                            class,
                            resolution: IncludeResolution::Ignored(class),
                            provenance: provenance.clone(),
                        });
                        if config.allow_missing_system_includes {
                            tracing::warn!(
                                target: "pssl_shader::preprocess",
                                include = %include_name,
                                class = class.label(),
                                "host API anchor not found; treating as empty"
                            );
                            out.push('\n');
                            continue;
                        }
                        return Err(Ps5ShaderError::new(
                            format!("cannot resolve include {include_name:?}"),
                            Phase::Preprocess,
                        ));
                    }
                }
            }

            match resolve_include(&include_name, current_dir, config) {
                IncludeResolution::Found(resolved) => {
                    report.includes.push(IncludeEvent {
                        name: include_name.clone(),
                        class,
                        resolution: IncludeResolution::Found(resolved.clone()),
                        provenance: provenance.clone(),
                    });
                    tracing::debug!(
                        target: "pssl_shader::preprocess",
                        include = %include_name,
                        resolved = %resolved.display(),
                        "include resolved"
                    );
                    let canonical = canonical(&resolved);
                    if include_stack.contains(&canonical) {
                        // The file is already being expanded higher on the stack.
                        // Real C preprocessors emit nothing here because well-formed
                        // headers carry an include guard (`#ifndef H / #define H ...`)
                        // that is already active mid-expansion. Treat a re-include as
                        // empty rather than erroring: this matches guarded-header
                        // semantics and prevents infinite recursion. (Self-referential
                        // *unguarded* headers are pathological; this avoids blow-up.)
                        report.includes.push(IncludeEvent {
                            name: include_name.clone(),
                            class: IncludeClass::Shader,
                            resolution: IncludeResolution::Ignored(IncludeClass::Shader),
                            provenance: provenance.clone(),
                        });
                        tracing::debug!(
                            target: "pssl_shader::preprocess",
                            include = %include_name,
                            resolved = %resolved.display(),
                            "re-include of file already on expansion stack; guard suppresses (empty)"
                        );
                        out.push('\n');
                        continue;
                    }
                    let mut included_source = std::fs::read_to_string(&resolved).map_err(|e| {
                        Ps5ShaderError::with_source(
                            format!("failed to read include {}", resolved.display()),
                            Phase::Preprocess,
                            e,
                        )
                    })?;
                    // Strip the byte-order mark *before* translation. The translator's
                    // preamble shifts a leading BOM to a later offset, where the
                    // recursion-entry `strip_bom` (index 0 only) cannot see it.
                    if included_source.starts_with('\u{FEFF}') {
                        included_source = included_source[3..].to_string();
                    }
                    // Virtual-header translation: PSSL shader headers (psfx_*.h etc.)
                    // are detected by content, not filename. The source file stays
                    // read-only; we feed a virtual HLSL translation to the parser.
                    let included_source = if crate::psfx::resolver::is_pssl_shader_header(&included_source) {
                        tracing::info!(
                            target: "pssl_shader::psfx::resolver",
                            include = %include_name,
                            resolved = %resolved.display(),
                            "PSSL shader header detected – translating to virtual HLSL"
                        );
                        let tr = crate::psfx::translator::translate_header(&resolved, &included_source);
                        tracing::trace!(
                            target: "pssl_shader::psfx::translator",
                            path = %resolved.display(),
                            orig_len = tr.original_len,
                            new_len = tr.translated.len(),
                            "virtual header emitted"
                        );
                        tr.translated
                    } else {
                        included_source
                    };
                    let included_dir = resolved.parent().map(Path::to_path_buf);
                    let resolved_name = resolved
                        .file_name()
                        .map(|n| n.to_string_lossy().to_string())
                        .unwrap_or_else(|| include_name.clone());
                    include_stack.push(canonical);
                    provenance.push(resolved_name);
                    let expanded = preprocess_inner(
                        &included_source,
                        included_dir.as_deref(),
                        defines,
                        func_defines,
                        include_stack,
                        report,
                        provenance,
                        depth + 1,
                        config,
                    )?;
                    provenance.pop();
                    include_stack.pop();
                    out.push_str(&expanded);
                    // Ensure trailing newline if included file didn't end with one
                    if !expanded.ends_with('\n') {
                        out.push('\n');
                    }
                    continue;
                }
                IncludeResolution::Ignored(class) => {
                    report.includes.push(IncludeEvent {
                        name: include_name.clone(),
                        class,
                        resolution: IncludeResolution::Ignored(class),
                        provenance: provenance.clone(),
                    });
                    tracing::debug!(
                        target: "pssl_shader::preprocess",
                        include = %include_name,
                        class = class.label(),
                        provenance = ?provenance,
                        "host/C++ infrastructure header ignored; treating as empty"
                    );
                    out.push('\n');
                    continue;
                }
                IncludeResolution::NotFound(class, _reason) => {
                    let host_and_allowed =
                        class.is_host() && config.allow_missing_system_includes;
                    report.includes.push(IncludeEvent {
                        name: include_name.clone(),
                        class,
                        resolution: IncludeResolution::NotFound(class, String::new()),
                        provenance: provenance.clone(),
                    });
                    if host_and_allowed {
                        tracing::warn!(
                            target: "pssl_shader::preprocess",
                            include = %include_name,
                            class = class.label(),
                            "optional host header unavailable; treating as empty"
                        );
                        out.push('\n');
                        continue;
                    }
                    return Err(Ps5ShaderError::new(
                        format!("cannot resolve include {include_name:?}"),
                        Phase::Preprocess,
                    ));
                }
            }
        }

        if is_define_directive(trimmed) {
            // Only single-line defines register directly; multi-line
            // (`\`-continued) defines accumulate in `pending_define` above
            // and register once their continuation ends. This line (`\`
            // present) is skipped here and handled by that path.
            if !line.trim_end().ends_with('\\') {
                if let Some((name, params, body)) = parse_func_define(trimmed) {
                    func_defines.insert(name, (params, body));
                } else if let Some((name, value)) = parse_define(trimmed) {
                    defines.insert(name, value);
                }
            }
            // Preserve line number
            out.push('\n');
            continue;
        }

        if let Some(name) = parse_undef(trimmed) {
            defines.remove(&name);
            func_defines.remove(&name);
            // Preserve line number
            out.push('\n');
            continue;
        }

        if trimmed.starts_with('#') {
            // Unknown directive — ignore but preserve line number
            out.push('\n');
            continue;
        }

        // Normal line — expand defines (object-like, then function-like,
        // rescan to a fixpoint so nested factories like
        // `CONCAT(S_TARGET_OUTPUT, slot)` fully unfold).
        let mut expanded = line.to_string();
        for _ in 0..8 {
            let next = expand_func_defines(&expand_defines(&expanded, defines), func_defines);
            if next == expanded {
                break;
            }
            expanded = next;
        }
        out.push_str(&expanded);
        out.push('\n');
    }

    Ok(out)
}

/// Whether a multiline macro body is safe to expand.
///
/// The expander supports `##` paste and bodies the PSSL parser accepts
/// (expressions like the `\`-continued `W64(x)` wave guard; free-function
/// factories like `DECLARE_SH_OPS`, whose `ShAdd`/`ApplyCosinusLobe`
/// definitions validate end-to-end). It does NOT support:
/// - `#` stringize (would emit `#24`-style fragments, e.g. the `CALLER`
///   entry-point factory);
/// - C++ member operators (`operator[]` in vectormath `_OP...`);
/// - MSVC/host intrinsics (`__pragma` in `_SCE_MACRO_END`,
///   `__declspec`/`__debugbreak`, `__m128`/`__m256` vector types).
/// Such macros stay unregistered so invocations keep the parser's
/// macro-skip path (the pre-multiline-capture behavior that held Parse
/// 1000/1000). Gating is marker-based, not shape-based: banning `{}`/`;`
/// outright also blocked helpful factories (6 sky-shader regressions).
fn is_expandable_macro_body(body: &str) -> bool {
    let bytes = body.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'#' {
            // `##` paste is supported; a lone `#` is stringize (unsupported).
            if bytes.get(i + 1) == Some(&b'#') {
                i += 2;
            } else {
                return false;
            }
            continue;
        }
        if bytes[i].is_ascii_alphabetic() || bytes[i] == b'_' {
            let mut j = i + 1;
            while j < bytes.len() && (bytes[j].is_ascii_alphanumeric() || bytes[j] == b'_') {
                j += 1;
            }
            let word = &body[i..j];
            if word == "operator"
                || word == "__pragma"
                || word == "__declspec"
                || word == "__debugbreak"
                || word.starts_with("__m128")
                || word.starts_with("__m256")
                || word.starts_with("__m64")
            {
                return false;
            }
            i = j;
            continue;
        }
        i += 1;
    }
    true
}
