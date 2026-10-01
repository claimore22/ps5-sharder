//! Include classification: mapping an `#include` target to a language family.

/// Broad language-family of an `#include` target, which drives everything that
/// follows: whether we resolve + preprocess it, ignore it, or hard-error.
///
/// The preprocessor is *not* a C/C++ preprocessor, and framework headers routinely pull
/// in C/C++ standard-library and compiler/platform implementation headers (e.g.
/// `stdint.h`, `stdarg.h`, `machine/acle-compat.h`) purely for host toolchain
/// compatibility. Those are not PSSL dependencies: inlining them would drag in an
/// entire libc that the shader neither needs nor can compile.
///
/// Classification is explicit and **category-based**, not a per-header allow-list:
///
/// * [`Shader`](IncludeClass::Shader)  → resolve and preprocess.
/// * [`CStdLib`](IncludeClass::CStdLib) → C standard-library leaf (`stdint.h`,
///   `stdarg.h`, …); ignored leaf (never inlined).
/// * [`CxxStdLib`](IncludeClass::CxxStdLib) → C++ standard-library leaf
///   (`cstdint`, `atomic`, …); ignored leaf.
/// * [`HostApi`](IncludeClass::HostApi) → a host C++ API *subtree* (e.g. the
///   AGC runtime anchored at `agc.h`). The anchor is resolved and recorded, but
///   its entire subtree is never expanded.
/// * [`Unknown`](IncludeClass::Unknown) → shader-relevant but not classifiable as
///   host infra; a missing one is a hard error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IncludeClass {
    /// A real PSSL / engine / project / framework shader header. Must resolve.
    Shader,
    /// C standard-library leaf header.
    CStdLib,
    /// C++ standard-library leaf header.
    CxxStdLib,
    /// A host C++ API subtree. Recorded; subtree not expanded.
    HostApi(HostApiFamily),
    /// Not classifiable as host infra; a missing one is a hard error.
    Unknown,
}

/// Host C++ API family that owns a subtree (reached via an anchor header).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HostApiFamily {
    /// Framework AGC runtime (`agc.h`); thousands of transitive host headers.
    Agc,
}

impl HostApiFamily {
    pub fn label(self) -> &'static str {
        match self {
            HostApiFamily::Agc => "HOST_API_AGC",
        }
    }
}

impl IncludeClass {
    pub(crate) fn is_host(self) -> bool {
        matches!(
            self,
            IncludeClass::CStdLib | IncludeClass::CxxStdLib | IncludeClass::HostApi(_)
        )
    }

    /// Whether this class is a leaf-level C/C++ header (ignored individually).
    pub(crate) fn is_leaf_host(self) -> bool {
        matches!(self, IncludeClass::CStdLib | IncludeClass::CxxStdLib)
    }

    pub fn label(self) -> &'static str {
        match self {
            IncludeClass::Shader => "shader",
            IncludeClass::CStdLib => "C_STDLIB",
            IncludeClass::CxxStdLib => "CXX_STDLIB",
            IncludeClass::HostApi(f) => f.label(),
            IncludeClass::Unknown => "unknown",
        }
    }
}

/// Classifies an include path (with its `<...>` / `"..."` wrapper intact) into a
/// language family.
///
/// This is **category-based**, not a whack-a-mole name list: compiler/platform
/// implementation headers are recognized by path prefix (`machine/`, `sys/`,
/// `bits/`, arch dirs), and C/C++ standard-library headers by name pattern/category.
/// Anything else that we cannot resolve is treated as a shader-relevant `Unknown`
/// (hard error), so we never silently drop something PSSL might depend on.
pub fn classify_include(path: &str) -> IncludeClass {
    // Strip the angle/quote wrapper for matching.
    let stripped = path
        .trim_start_matches('<')
        .trim_start_matches('"')
        .trim_end_matches('>')
        .trim_end_matches('"');

    // Host C++ API *anchor*: `agc.h` is the AGC runtime entry point. Once we hit
    // it, we stop expanding the entire host subtree. This is intentionally a tiny
    // anchor set — we do NOT classify `vectormath.h` as host yet, so
    // `psfx_ssr.h → vectormath.h → floatInVec.h` continues to be processed until
    // proven host-only.
    if HOST_ANCHORS.binary_search(&stripped).is_ok() {
        return IncludeClass::HostApi(HostApiFamily::Agc);
    }

    let base = stripped.split(['/', '\\']).last().unwrap_or(stripped);

    // C++ standard-library headers (c-prefix / std C++ headers).
    if CXX_STDLIB_NAMES.binary_search(&base).is_ok() {
        return IncludeClass::CxxStdLib;
    }
    // C standard-library headers.
    if C_STDLIB_NAMES.binary_search(&base).is_ok() {
        return IncludeClass::CStdLib;
    }

    // Compiler/platform implementation dirs are also leaf host infra (e.g.
    // `machine/acle-compat.h`), but we keep them distinct from C/C++ stdlib.
    // For reporting they are shown as C/C++ infra; classify as CStdLib-like leaf
    // for now (ignored individually) — the agc.h anchor already prevents the
    // deep `machine/` cascade from being reached via the host subtree.
    let first_seg = stripped.split(['/', '\\']).next().unwrap_or("");
    if matches!(
        first_seg,
        "machine" | "sys" | "bits" | "x86" | "arm" | "arm64" | "aarch64" | "gnu" | "dinkumware"
    ) {
        // Treat as CStdLib leaf for reporting (still ignored individually).
        return IncludeClass::CStdLib;
    }

    // Not classifiable as host infra — shader-relevant by default.
    IncludeClass::Unknown
}

/// Legacy 3-way classifier kept for compatibility/reporting.
///
/// Maps the richer [`IncludeClass`] onto the old coarse tri-state used by earlier
/// callers. Prefer [`classify_include`] for new code.
pub fn include_kind(path: &str) -> IncludeKind {
    match classify_include(path) {
        IncludeClass::Shader => IncludeKind::Pssl,
        IncludeClass::CStdLib | IncludeClass::CxxStdLib | IncludeClass::HostApi(_) => {
            IncludeKind::System
        }
        IncludeClass::Unknown => IncludeKind::Pssl,
    }
}

/// Whether this include is C/C++ host infrastructure that PSSL is not responsible
/// for resolving. Kept for compatibility — prefer [`classify_include`].
pub fn is_ignorable_system_include(path: &str) -> bool {
    classify_include(path).is_host()
}

/// Coarse 3-way classification, retained for API compatibility with older callers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IncludeKind {
    Pssl,
    Project,
    System,
}

/// Known host C/C++ API anchor headers. Sorted ascending for binary_search.
/// Keep this tiny and evidence-driven: `agc.h` is proven to fan out into
/// thousands of host headers; `vectormath.h` is intentionally NOT here yet.
const HOST_ANCHORS: &[&str] = &["agc.h"];

/// C++ standard-library header basenames. Sorted ascending for binary_search.
const CXX_STDLIB_NAMES: &[&str] = &[
    "algorithm", "array", "atomic", "bitset", "cassert", "ccomplex", "cctype", "cerrno", "cfenv",
    "cfloat", "chrono", "cinttypes", "ciso646", "climits", "clocale", "cmath", "csetjmp",
    "csignal", "cstdarg", "cstdbool", "cstddef", "cstdint", "cstdio", "cstdlib", "cstring",
    "ctgmath", "ctime", "cwchar", "cwctype", "deque", "exception", "functional", "initializer_list",
    "iterator", "limits", "list", "map", "memory", "mutex", "new", "numeric", "queue", "random",
    "ratio", "regex", "set", "shared_mutex", "stack", "stdexcept", "string", "thread", "tuple",
    "type_traits", "typeinfo", "unordered_map", "unordered_set", "utility", "valarray", "vector",
];

/// C standard-library header basenames. Sorted ascending for binary_search.
const C_STDLIB_NAMES: &[&str] = &[
    "assert.h", "complex.h", "ctype.h", "errno.h", "fenv.h", "float.h", "inttypes.h", "iso646.h",
    "limits.h", "locale.h", "math.h", "setjmp.h", "signal.h", "stdalign.h", "stdarg.h", "stdatomic.h",
    "stdbool.h", "stddef.h", "stdint.h", "stdio.h", "stdlib.h", "stdnoreturn.h", "string.h",
    "tgmath.h", "threads.h", "time.h", "uchar.h", "wchar.h", "wctype.h",
];
