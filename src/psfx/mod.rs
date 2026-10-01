//! `psfx` virtual-header translation pipeline.
//!
//! External framework headers are read-only. This crate translates them
//! to Vulkan HLSL on demand as *virtual* headers. No `assets/hlsl/psfx_common.hlsl`
//! is ever committed.
//!
//! Pipeline:
//! ```text
//! Framework header (psfx_common.h, psfx_types.h, ...) ─┐
//!                                                ├──► resolver ──► graph (DAG) ──► translator ──┐
//! PSSL shader (.pssl) ───────────────────────────┘                                             ├──► HLSL ──► DXC ──► SPIR-V
//!                                                                                              │
//!                                                                                           cache (transitive DAG hash, /cache/*/)
//! ```
//!
//! Headers are detected by content (`#ifdef __PSSL__`, `Texture2D`, `sce::Agc::`, …),
//! not by `path.endsWith("psfx_*.h")`, so the layout remains portable across
//! framework revisions. Responsibility split:
//! * `src/psfx/*` owns header-intrinsic translation (CndMask, rcp, samplers, wave, constexpr, types).
//! * `src/lower/srt.rs` + `src/emit/hlsl.rs` own shader-lowering concerns (SRT flattening,
//!   `srt` param elision, `half2[]`, binding decisions).

pub mod cache;
pub mod graph;
pub mod intrinsics;
pub mod resources;
pub mod resolver;
pub mod rules;
pub mod translator;

pub use cache::{cache_key, CacheKey};
pub use graph::{HeaderDag, HeaderNode};
pub use resolver::{is_pssl_shader_header, resolve_header};
pub use translator::{translate_header, TranslatedHeader};

/// Toolchain identity string (e.g. `"10.00.00.40"`). Used in cache key.
pub const TOOLCHAIN_IDENTITY: &str = "toolchain-10.00";

/// Translator rules version – bump when `rules.rs`/`intrinsics.rs`/`resources.rs` change.
pub const TRANSLATOR_VERSION: &str = "0.1.0";
