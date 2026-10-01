use super::*;

#[test]
fn strips_bom() {
    let src = "\u{FEFF}float4 main() : S_TARGET_OUTPUT0 { return float4(0,0,0,1); }";
    let out = preprocess(src, None).unwrap();
    assert!(!out.starts_with('\u{FEFF}'));
    assert!(out.contains("float4 main"));
}

#[test]
fn defined_with_space_before_paren_evaluates() {
    // Framework style `#if defined ( __PROSPERO__ )` (space before paren) must
    // behave like `defined(__PROSPERO__)`. `__PROSPERO__` is in the default config
    // (live branch kept), `__ORBIS__` is not (else branch kept).
    let src = "#if defined ( __PROSPERO__ )\nkept_prospero\n#else\ndropped_prospero\n#endif\n#if defined ( __ORBIS__ )\ndropped_orbis\n#else\nkept_orbis\n#endif\n";
    let out = preprocess(src, None).unwrap();
    assert!(out.contains("kept_prospero"), "out: {out}");
    assert!(!out.contains("dropped_prospero"), "out: {out}");
    assert!(out.contains("kept_orbis"), "out: {out}");
    assert!(!out.contains("dropped_orbis"), "out: {out}");
}

#[test]
fn negated_defined_with_space_evaluates() {
    // `#if !defined ( __ORBIS__ )` is true (kept); with `__PROSPERO__`
    // defined, `!defined ( __PROSPERO__ )` is false (dropped).
    let src = "#if !defined ( __ORBIS__ )\nkept\n#endif\n#if !defined ( __PROSPERO__ )\ndropped\n#endif\n";
    let out = preprocess(src, None).unwrap();
    assert!(out.contains("kept"), "out: {out}");
    assert!(!out.contains("dropped"), "out: {out}");
}

#[test]
fn define_expands() {
    let src = "#define M_PI 3.14159\nfloat x = M_PI;\n";
    let out = preprocess(src, None).unwrap();
    assert!(out.contains("3.14159"));
    assert!(!out.contains("#define"));
}

#[test]
fn multiline_define_body_does_not_leak() {
    // A `\`-continued (function-like) `#define` must be consumed wholly;
    // its body lines must not leak into the source as bogus code.
    let src = "#define DECLARE_SH_OPS(NAME, ORDER) \\\n\tstatic void ShReset(out NAME r) \\\n\t{ ORDER; }\nfloat4 main() : S_TARGET_OUTPUT0 { return float4(0,0,0,1); }";
    let out = preprocess(src, None).unwrap();
    assert!(!out.contains("ShReset"));
    assert!(!out.contains("ORDER"));
    assert!(out.contains("float4 main"));
}

#[test]
fn nested_defines_expand_to_fixpoint() {
    // `LDS_X` → `(THREADS_X + 2)` → `(16 + 2)` regardless of map order.
    let src = "#define THREADS_X 16\n#define LDS_X (THREADS_X + 2)\nint a[LDS_X];\n";
    let out = preprocess(src, None).unwrap();
    assert!(!out.contains("THREADS_X"), "out: {out}");
    assert!(!out.contains("LDS_X"), "out: {out}");
    assert!(out.contains("(16 + 2)"), "out: {out}");
}

#[test]
fn numeric_comparison_conditions() {
    // `#if (PS > 12)` with PS=12 drops the block; `== 29` keeps it.
    let src = "#define PS 12\n#if (PS > 12)\nbad1\n#endif\n#if (PS == 12)\ngood\n#endif\n#if (PS != 12)\nbad2\n#endif\n";
    let out = preprocess(src, None).unwrap();
    assert!(!out.contains("bad1"), "out: {out}");
    assert!(!out.contains("bad2"), "out: {out}");
    assert!(out.contains("good"), "out: {out}");
}

#[test]
fn unknown_comparison_abstains_to_keep() {
    // Unresolvable sides keep the block (pre-existing keep-unknown).
    let src = "#if (SOME_FEATURE > 12)\nkept\n#endif\n";
    let out = preprocess(src, None).unwrap();
    assert!(out.contains("kept"), "out: {out}");
}

#[test]
fn define_word_boundary() {
    let src = "#define A 1\nfloat AA = A;\n";
    let out = preprocess(src, None).unwrap();
    // AA should remain AA, not "1A" or "11"
    assert!(out.contains("AA"));
    assert!(out.contains("= 1;"));
}

#[test]
fn system_compat_header_classified() {
    assert_eq!(include_kind("machine/acle-compat.h"), IncludeKind::System);
    assert_eq!(include_kind("sys/acle-compat.h"), IncludeKind::System);
    assert_eq!(include_kind("cstdint"), IncludeKind::System);
    assert_eq!(include_kind("some_pssl.h"), IncludeKind::Pssl);
    assert!(is_ignorable_system_include("machine/acle-compat.h"));
}

#[test]
fn classify_three_families() {
    // C/C++ standard-library headers by name/category — distinct C vs C++.
    assert_eq!(classify_include("cstdint"), IncludeClass::CxxStdLib);
    assert_eq!(classify_include("<cstdint>"), IncludeClass::CxxStdLib);
    assert_eq!(classify_include("\"stdint.h\""), IncludeClass::CStdLib);
    assert_eq!(classify_include("<stdarg.h>"), IncludeClass::CStdLib);
    // Host API anchor — tiny curated set; vectormath intentionally NOT host.
    assert_eq!(
        classify_include("agc.h"),
        IncludeClass::HostApi(HostApiFamily::Agc)
    );
    assert_eq!(
        classify_include("<agc.h>"),
        IncludeClass::HostApi(HostApiFamily::Agc)
    );
    assert_eq!(classify_include("vectormath.h"), IncludeClass::Unknown);
    // Unknown shader-relevant headers are NOT silently host-ignored.
    assert_eq!(classify_include("psfx_common.h"), IncludeClass::Unknown);
    assert_eq!(classify_include("V2P.h"), IncludeClass::Unknown);
}

#[test]
fn pssl_unused_predefined_empty() {
    // The unused-parameter suppressor is predefined (empty): the
    // statement vanishes instead of leaking an undeclared identifier.
    let src = "void f(bool odd) { PSSL_UNUSED(odd); }\n";
    let out = preprocess_reported(src, None, &PreprocessorConfig::platform_defaults())
        .unwrap()
        .source;
    assert!(
        !out.contains("PSSL_UNUSED"),
        "suppressor must expand away, got:\n{out}"
    );
}

#[test]
fn unresolved_shader_header_errors_unknown_class() {
    // An unknown, unresolvable shader-relevant header is a hard error.
    let src = "#include \"definitely_missing_header.h\"\n";
    let cfg = PreprocessorConfig::platform_defaults();
    let err = preprocess_reported(src, None, &cfg).unwrap_err();
    assert!(err.to_string().contains("cannot resolve include"));
}

#[test]
fn report_counts_and_provenance() {
    use std::fs;
    let base = std::env::temp_dir().join(format!("pssl_rep_{}", std::process::id()));
    let root = base.join("root");
    fs::create_dir_all(&root).unwrap();
    // A resolvable PSSL header that itself pulls in host infra + another header.
    fs::write(
        root.join("stage.h"),
        "#include <stdint.h>\n#include \"sub.h\"\nSTAGE_DEF\n",
    )
    .unwrap();
    fs::write(root.join("sub.h"), "SUB_DEF\n").unwrap();
    let cfg = PreprocessorConfig {
        include_dirs: vec![root.clone()],
        allow_missing_system_includes: true,
        ..Default::default()
    };
    let src = "#include \"stage.h\"\nfloat x = STAGE_DEF + SUB_DEF;\n";
    let out = preprocess_reported(src, None, &cfg).unwrap();

    assert!(out.source.contains("STAGE_DEF"));
    assert!(out.source.contains("SUB_DEF"));
    // stage.h and sub.h resolve; stdint.h is ignored host infra.
    assert_eq!(out.report.resolved, 2);
    assert_eq!(out.report.ignored_host, 1);
    assert_eq!(out.report.unresolved_shader, 0);
    // Provenance chain: sub.h's including file was stage.h.
    let sub = out
        .report
        .includes
        .iter()
        .find(|e| e.name == "sub.h")
        .expect("sub.h event recorded");
    assert_eq!(sub.provenance.last().map(String::as_str), Some("stage.h"));
    // stdint.h classified as CStdLib with provenance stage.h.
    let si = out
        .report
        .includes
        .iter()
        .find(|e| e.name == "stdint.h")
        .expect("stdint.h event recorded");
    assert_eq!(si.class, IncludeClass::CStdLib);
    let _ = fs::remove_dir_all(&base);
}

#[test]
fn ancestor_subpath_include_prefers_project_local_header() {
    // `#include "shaders/lighting.pssli"` from `<proj>/ibl/x.pssl` binds
    // the project-local `<proj>/shaders/lighting.pssli`, not an unrelated
    // same-subpath header under an include root (C quote-include locality).
    use std::fs;
    let base = std::env::temp_dir().join(format!("pssl_anc_{}", std::process::id()));
    let proj_ibl = base.join("proj").join("ibl");
    fs::create_dir_all(&proj_ibl).unwrap();
    fs::create_dir_all(base.join("proj").join("shaders")).unwrap();
    fs::create_dir_all(base.join("extroot").join("shaders")).unwrap();
    fs::write(
        base.join("proj").join("shaders").join("lighting.pssli"),
        "LOCAL_LIGHTING\n",
    )
    .unwrap();
    fs::write(
        base.join("extroot").join("shaders").join("lighting.pssli"),
        "EXT_LIGHTING\n",
    )
    .unwrap();
    let main = proj_ibl.join("shader.pssl");
    fs::write(&main, "#include \"shaders/lighting.pssli\"\n").unwrap();
    let cfg = PreprocessorConfig {
        include_dirs: vec![base.join("extroot")],
        allow_missing_system_includes: true,
        ..Default::default()
    };
    let src = std::fs::read_to_string(&main).unwrap();
    let out = preprocess_reported(&src, Some(&main), &cfg).unwrap();
    assert!(out.source.contains("LOCAL_LIGHTING"), "out: {}", out.source);
    assert!(!out.source.contains("EXT_LIGHTING"), "out: {}", out.source);
    let _ = fs::remove_dir_all(&base);
}

#[test]
fn ancestor_walk_stops_at_project_root() {
    // The ancestor walk never searches past an enclosing project root
    // (`Cargo.toml`): with no project-local header, resolution falls
    // through to include roots even when the project root itself carries a
    // same-subpath file.
    use std::fs;
    let base = std::env::temp_dir().join(format!("pssl_ancstop_{}", std::process::id()));
    let proj_ibl = base.join("proj").join("ibl");
    fs::create_dir_all(&proj_ibl).unwrap();
    fs::create_dir_all(base.join("extroot").join("shaders")).unwrap();
    fs::write(base.join("Cargo.toml"), "[package]\n").unwrap();
    fs::write(base.join("shaders_marker"), "x\n").unwrap();
    fs::create_dir_all(base.join("shaders")).unwrap();
    fs::write(base.join("shaders").join("lighting.pssli"), "ROOT_LIGHTING\n").unwrap();
    fs::write(
        base.join("extroot").join("shaders").join("lighting.pssli"),
        "EXT_LIGHTING\n",
    )
    .unwrap();
    let main = proj_ibl.join("shader.pssl");
    fs::write(&main, "#include \"shaders/lighting.pssli\"\n").unwrap();
    let cfg = PreprocessorConfig {
        include_dirs: vec![base.join("extroot")],
        allow_missing_system_includes: true,
        ..Default::default()
    };
    let src = std::fs::read_to_string(&main).unwrap();
    let out = preprocess_reported(&src, Some(&main), &cfg).unwrap();
    assert!(out.source.contains("EXT_LIGHTING"), "out: {}", out.source);
    assert!(!out.source.contains("ROOT_LIGHTING"), "out: {}", out.source);
    let _ = fs::remove_dir_all(&base);
}

#[test]
fn host_anchor_stops_subtree() {
    use std::fs;
    let base = std::env::temp_dir().join(format!("pssl_anchor_{}", std::process::id()));
    let root = base.join("root");
    fs::create_dir_all(&root).unwrap();
    // agc.h anchor includes many host leaves, but they must NOT be traversed.
    fs::write(
        root.join("agc.h"),
        "#include <stdint.h>\n#include \"agc_internals.h\"\nAGC_BODY\n",
    )
    .unwrap();
    fs::write(root.join("agc_internals.h"), "SHOULD_NOT_APPEAR\n").unwrap();
    fs::write(root.join("psfx_common.h"), "#include \"agc.h\"\nPSFX_BODY\n").unwrap();
    let cfg = PreprocessorConfig {
        include_dirs: vec![root.clone()],
        allow_missing_system_includes: true,
        ..Default::default()
    };
    let src = "#include \"psfx_common.h\"\nfloat x = PSFX_BODY;\n";
    let out = preprocess_reported(src, None, &cfg).unwrap();
    assert!(out.source.contains("PSFX_BODY"));
    // agc.h body must NOT be inlined (subtree stopped at anchor).
    assert!(!out.source.contains("AGC_BODY"));
    assert!(!out.source.contains("SHOULD_NOT_APPEAR"));
    assert_eq!(out.report.host_api_chains_skipped, 1);
    assert!(out.report.host_api_chain_names.contains(&"agc.h".to_string()));
    // No CStdLib leaves reached via the anchor subtree.
    assert!(out.report.ignored_names.is_empty() || !out.report.ignored_names.contains(&"stdint.h".to_string()));
    let _ = fs::remove_dir_all(&base);
}

#[test]
fn dependency_warning_when_ignored_symbol_used() {
    // A shader that uses a symbol an ignored host header would provide must
    // warn, so we never silently drop something the shader depends on.
    let sentence = "uint32_t counter;";
    let src = format!("#include <cstdint>\n{sentence}\n");
    let cfg = PreprocessorConfig::platform_defaults();
    let out = preprocess_reported(&src, None, &cfg)
        .unwrap_or_else(|e| panic!("should preprocess, got {e}"));
    assert!(
        !out.report.dependency_warnings.is_empty(),
        "expected a dependency warning referencing uint32_t, got {:?}",
        out.report.dependency_warnings
    );
    assert!(
        out.report
            .dependency_warnings
            .iter()
            .any(|w| w.contains("uint32_t")),
        "warning should mention uint32_t: {:?}",
        out.report.dependency_warnings
    );
}

#[test]
fn no_dependency_warning_without_usage() {
    let src = "#include <stdint.h>\nfloat x = 1.0;\n";
    let cfg = PreprocessorConfig::platform_defaults();
    let out = preprocess_reported(src, None, &cfg).unwrap();
    assert!(
        out.report.dependency_warnings.is_empty(),
        "no symbol used, expected no warnings, got {:?}",
        out.report.dependency_warnings
    );
}

#[test]
fn missing_system_include_is_ignored() {
    let src = "#include <machine/acle-compat.h>\nfloat x = 1.0;\n";
    let cfg = PreprocessorConfig::platform_defaults();
    let out = preprocess_with_config(src, None, &cfg).unwrap();
    // The host compat header is dropped; shader body still preprocesses.
    assert!(out.contains("float x = 1.0;"));
    assert!(!out.contains("machine/acle-compat.h"));
}

#[test]
fn missing_pssl_include_is_error() {
    let src = "#include \"definitely_missing_pssl_header.h\"\n";
    let cfg = PreprocessorConfig::platform_defaults();
    let err = preprocess_with_config(src, None, &cfg).unwrap_err();
    assert!(err.to_string().contains("cannot resolve include"));
}

#[test]
fn negated_parenthesized_conjunction_evaluates() {
    // `#if !(defined(A) && defined(B))` when neither A nor B is defined must
    // be true (kept). A naive split on `&&` would mis-evaluate it as
    // `!defined(A) && defined(B)` → false, silently dropping the guarded
    // declarations (e.g. a skinned-mesh vertex input).
    let src = "#if !(defined(SKINNED) && defined(DEPTHONLY))\nkept\n#else\ndropped\n#endif\n";
    let out = preprocess(src, None).unwrap();
    assert!(out.contains("kept"));
    assert!(!out.contains("dropped"));
}

#[test]
fn negated_parenthesized_disjunction_evaluates() {
    // `#if !(defined(A) || defined(B))` with A defined: the group is true,
    // negated to false, so the `#if` branch is dropped and `#else` kept.
    let src = "#define A 1\n#if !(defined(A) || defined(B))\ndropped\n#else\nkept\n#endif\n";
    let out = preprocess(src, None).unwrap();
    assert!(out.contains("kept"));
    assert!(!out.contains("dropped"));
}

#[test]
fn defined_guard_false_evaluates() {
    // `#if defined(SOME_UNDEFINED)` without `!` should still be true in the
    // limited model if the macro is undefined; the `!defined` case is what we
    // want excluded.
    let src = "#define __PSSL__ 1\n#if !defined(__PSSL__)\nbad\n#else\ngood\n#endif\n";
    let cfg = PreprocessorConfig::platform_defaults();
    let out = preprocess_with_config(src, None, &cfg).unwrap();
    assert!(out.contains("good"));
    assert!(!out.contains("bad"));
}

#[test]
fn origin_anchored_relative_include_resolves_in_include_tree() {
    // A flat corpus copy (`tests/shaders/<name>/shader.pssl`) with a sibling
    // ORIGIN.txt recording the original path it came from must resolve `../include/`
    // includes against the *recorded* path's directory, not the flat copy dir.
    use std::fs;

    let base = std::env::temp_dir().join(format!("pssl_origin_{}", std::process::id()));
    // Real framework layout: .../shaders/frontend/sky.pssl + .../shaders/include/x.h
    let src_shaders = base.join("shaders");
    let src_frontend = src_shaders.join("frontend");
    let src_include = src_shaders.join("include");
    fs::create_dir_all(&src_frontend).unwrap();
    fs::create_dir_all(&src_include).unwrap();
    let src_file = src_frontend.join("sky.pssl");
    fs::write(
        &src_file,
        "#include \"../include/shader_srts.h\"\nfloat4 sky = SRT_BODY;\n",
    )
    .unwrap();
    fs::write(src_include.join("shader_srts.h"), "#ifndef H\n#define H\nSRT_BODY\n#endif\n").unwrap();

    // Flat corpus copy of that shader, with ORIGIN.txt naming the original file.
    let copy = base.join("copy");
    fs::create_dir_all(&copy).unwrap();
    let copy_file = copy.join("shader.pssl");
    fs::write(&copy_file, fs::read_to_string(&src_file).unwrap()).unwrap();
    fs::write(copy.join("ORIGIN.txt"), src_file.to_string_lossy().as_bytes()).unwrap();

    let cfg = PreprocessorConfig {
        include_dirs: vec![base.clone()],
        allow_missing_system_includes: true,
        ..Default::default()
    };
    let out = preprocess_with_config(
        &fs::read_to_string(&copy_file).unwrap(),
        Some(&copy_file),
        &cfg,
    )
    .unwrap();
    assert!(
        out.contains("SRT_BODY"),
        "ORIGIN-anchored include must inline shader_srts.h:\n{out}"
    );

    let _ = fs::remove_dir_all(&base);
}

#[test]
fn bom_in_included_translated_header_is_stripped() {
    // Framework headers carry a UTF-8 BOM. When an included header is translated to
    // virtual HLSL, the translator prepends a preamble, sliding a leading BOM
    // to a later offset where the recursion-entry strip (index 0 only) misses
    // it. The BOM must be stripped *before* translation, so no U+FEFF ever
    // reaches the lexer.
    use std::fs;

    let base = std::env::temp_dir().join(format!("pssl_bom_{}", std::process::id()));
    let root = base.join("root");
    fs::create_dir_all(&root).unwrap();
    // SamplerState marker makes this a PSSL shader header (gets translated).
    fs::write(
        root.join("bom_sampler.h"),
        format!("\u{FEFF}/* vendor header */\nSamplerState g_sampler;\nBOM_BODY\n"),
    )
    .unwrap();
    let src = "#include \"bom_sampler.h\"\nfloat x = BOM_BODY;\n";
    let file = root.join("main.pssl");
    fs::write(&file, src).unwrap();

    let cfg = PreprocessorConfig {
        include_dirs: vec![root.clone()],
        ..Default::default()
    };
    let out = preprocess_with_config(&src, Some(&file), &cfg)
        .unwrap_or_else(|e| panic!("should preprocess, got {e}"));
    assert!(
        !out.contains('\u{FEFF}'),
        "no U+FEFF may reach the lexer:\n{out:?}"
    );
    assert!(out.contains("BOM_BODY"), "header must be inlined:\n{out:?}");

    let _ = fs::remove_dir_all(&base);
}

#[test]
fn translate_header_strips_leading_bom() {
    // Direct guard against the translator's preamble burying a BOM mid-stream:
    // output must never contain U+FEFF regardless of what the rule pipeline
    // prepends.
    let src = "\u{FEFF}SamplerState g_sampler;\n#define M_PI 3.14159\n";
    let path = std::path::Path::new("bom_sampler.h");
    let tr = crate::psfx::translator::translate_header(path, src);
    assert!(
        !tr.translated.contains('\u{FEFF}'),
        "translated header leaked a BOM:\n{:?}",
        tr.translated
    );
    assert!(tr.translated.contains("g_sampler"));
}

#[test]
fn quoted_include_resolves_relative_to_including_file() {
    // Two shaders in separate directories both `#include "stage_interface.h"`,
    // and a same-basename `stage_interface.h` also exists in a configured
    // include root. Each shader must resolve its *own* sibling header (the
    // including file's directory), NOT the root one or whichever is found first.
    use std::fs;

    let base = std::env::temp_dir().join(format!("pssl_incl_{}", std::process::id()));
    let dir_a = base.join("a");
    let dir_b = base.join("b");
    let root = base.join("root");
    for d in [&dir_a, &dir_b, &root] {
        fs::create_dir_all(d).unwrap();
    }

    fs::write(root.join("stage_interface.h"), "ROOT_STAGE").unwrap();
    fs::write(dir_a.join("stage_interface.h"), "STAGE_A").unwrap();
    fs::write(dir_b.join("stage_interface.h"), "STAGE_B").unwrap();
    fs::write(
        dir_a.join("shader.pssl"),
        "#include \"stage_interface.h\"\nfloat xa = STAGE_A;\n",
    )
    .unwrap();
    fs::write(
        dir_b.join("shader.pssl"),
        "#include \"stage_interface.h\"\nfloat xb = STAGE_B;\n",
    )
    .unwrap();

    // Only the loopback `root` is configured; no extra include roots, so nothing else
    // can satisfy the include. The local sibling must win.
    let cfg = PreprocessorConfig {
        include_dirs: vec![root.clone()],
        ..Default::default()
    };

    let file_a = dir_a.join("shader.pssl");
    let out_a = preprocess_with_config(
        &fs::read_to_string(&file_a).unwrap(),
        Some(&file_a),
        &cfg,
    )
    .unwrap();
    assert!(
        out_a.contains("float xa = STAGE_A;"),
        "shader A must resolve its own stage_interface.h, got:\n{out_a}"
    );
    assert!(!out_a.contains("ROOT_STAGE"), "A must not use root header:\n{out_a}");

    let file_b = dir_b.join("shader.pssl");
    let out_b = preprocess_with_config(
        &fs::read_to_string(&file_b).unwrap(),
        Some(&file_b),
        &cfg,
    )
    .unwrap();
    assert!(
        out_b.contains("float xb = STAGE_B;"),
        "shader B must resolve its own stage_interface.h, got:\n{out_b}"
    );
    assert!(!out_b.contains("ROOT_STAGE"), "B must not use root header:\n{out_b}");

    let _ = fs::remove_dir_all(&base);
}

#[test]
fn c_std_headers_ignored_even_when_file_exists() {
    // `cstdint` / `stdint.h` are known C/C++ host-compat headers. They must
    // be *ignored* (treated as empty) whether included with quotes or angle
    // brackets — even when a real file of that name is present in an include
    // root. Inlining them would drag in the entire C++ standard library.
    use std::fs;

    let base = std::env::temp_dir().join(format!("pssl_cstd_{}", std::process::id()));
    let root = base.join("root");
    fs::create_dir_all(&root).unwrap();
    // A real file would normally defeat the System classification.
    fs::write(root.join("cstdint"), "INLINED_CSTDINT").unwrap();
    fs::write(root.join("stdint.h"), "INLINED_STDINT").unwrap();

    let mut cfg = PreprocessorConfig {
        include_dirs: vec![root.clone()],
        ..Default::default()
    };
    cfg.allow_missing_system_includes = true;

    for inc in ["\"cstdint\"", "<cstdint>", "\"stdint.h\"", "<stdint.h>"] {
        let src = format!("#include {inc}\nfloat x = 1.0;\n");
        let out = preprocess_with_config(&src, None, &cfg)
            .unwrap_or_else(|e| panic!("{inc} should preprocess, got {e}"));
        assert!(
            !out.contains("INLINED"),
            "{inc} should NOT inline the header, got:\n{out}"
        );
        assert!(
            out.contains("float x = 1.0;"),
            "{inc} should keep the body, got:\n{out}"
        );
    }

    let _ = fs::remove_dir_all(&base);
}

#[test]
fn multiline_func_define_registers_full_body() {
    // A `\`-continued function-like `#define` accumulates its full body
    // and expands correctly (the earlier one-line-only capture bound the
    // name to `\` and corrupted calls).
    let src = "#define W64(x) \\\n\t((x == 64 && __get_wavemode() == __kWaveModeWave64))\nint f(int l) { if (W64(l)) { return 1; } return 0; }\n";
    let out = preprocess(src, None).unwrap();
    assert!(
        out.contains("if (((l == 64 && __get_wavemode() == __kWaveModeWave64)))"),
        "invocation must expand to the joined body, out: {out}"
    );
    assert!(!out.contains("if (\\)"), "out: {out}");
}

#[test]
fn multiline_stringize_macro_stays_unexpanded() {
    // `#` stringize is unsupported (would emit `#24`-style fragments that
    // fail parsing, e.g. the `CALLER` entry-point factory in
    // closest_hit_caller_lib). Such multiline macros must not register,
    // so invocations keep the parser's macro-skip path (Parse 1000/1000).
    let src = "#define CALLER(N) \\\n[cxxsymbol(\"X_\" #N \"_y\")] void caller_##N() { f<N>(); }\nCALLER(24)\n";
    let out = preprocess(src, None).unwrap();
    assert!(out.contains("CALLER(24)"), "invocation must stay verbatim, out: {out}");
    assert!(!out.contains("#24"), "stringize must not emit, out: {out}");
}

#[test]
fn multiline_statement_macro_stays_unexpanded() {
    // Host fragments (`_SCE_MACRO_END` with `__pragma(warning(disable:
    // 4127))`) are C++ the PSSL parser never accepts. They must not
    // register either (same loud macro-skip path as before multiline
    // capture existed).
    let src = "#define MY_END \\\n__pragma(warning(disable: 4127)) } while (0)\nvoid f() { do { } MY_END }\n";
    let out = preprocess(src, None).unwrap();
    assert!(out.contains("MY_END"), "invocation must stay verbatim, out: {out}");
    assert!(!out.contains("4127"), "statement body must not expand, out: {out}");
}

#[test]
fn multiline_function_factory_registers() {
    // Free-function factories (`DECLARE_SH_OPS`-style: `static void`
    // definitions, no stringize/operators/host intrinsics) expand: the
    // sky shaders validate on their `ShAdd`/`ApplyCosinusLobe` bodies.
    let src = "#define MK_OPS(T) \\\nstatic void myAdd(inout T a, T b) { a += b; }\nMK_OPS(float)\n";
    let out = preprocess(src, None).unwrap();
    assert!(
        out.contains("static void myAdd(inout float a, float b)"),
        "factory must expand, out: {out}"
    );
    assert!(!out.contains("MK_OPS(float)"), "out: {out}");
}

#[test]
fn multiline_operator_macro_stays_unexpanded() {
    // C++ member operators (vectormath `_OP...` with `operator[]`) never
    // lower: the macro must not register.
    let src = "#define MK_OP(T) \\\nT operator [](int i) { return v[i]; }\nMK_OP(float)\n";
    let out = preprocess(src, None).unwrap();
    assert!(out.contains("MK_OP(float)"), "invocation must stay verbatim, out: {out}");
    assert!(!out.contains("operator []"), "operator body must not expand, out: {out}");
}

#[test]
fn wave_define_is_predefined() {
    // The PSSL compiler predefines `__WAVE__` (gates GPU wave headers
    // and wave-gated output structs).
    let src = "#ifdef __WAVE__\nkept_wave\n#else\ndropped_wave\n#endif\n";
    let out = preprocess(src, None).unwrap();
    assert!(out.contains("kept_wave"), "out: {out}");
    assert!(!out.contains("dropped_wave"), "out: {out}");
}

#[test]
fn func_macro_expands_with_args() {
    // GBUFFER-style X-macros: fixed-arity invocation with balanced,
    // comma-split arguments (nested calls stay in one argument).
    let src = "#define MADD(a, b, c) ((a) * (b) + (c))\nfloat x = MADD(u + 1, f(v, w), 2.0);\n";
    let out = preprocess(src, None).unwrap();
    assert!(
        out.contains("((u + 1) * (f(v, w)) + (2.0))"),
        "args must substitute whole, out: {out}"
    );
    assert!(!out.contains("MADD"), "out: {out}");
}

#[test]
fn empty_param_func_macro_expands() {
    // `NAME()`-style function-like macros (zero params) must expand; the
    // previous arg parser returned one empty string arg, so `()` never
    // matched a zero-arity macro.
    let src = "#define FOO() 42\nuint f(uint i) { return i + FOO(); }\n";
    let out = preprocess(src, None).unwrap();
    assert!(out.contains("i + 42;"), "out: {out}");
    assert!(!out.contains("FOO()"), "out: {out}");
}

#[test]
fn token_paste_unfolds_nested_factories() {
    // The IBL `DEFERRED_PS_OUTPUT` idiom: `CONCAT`/`CONCAT_` plus an
    // X-macro emitting `type name : SEMANTICn;` members.
    let src = "#define CONCAT_(x, y) x ## y\n\
               #define CONCAT(x, y) CONCAT_(x, y)\n\
               #define GBUFFER_OP(slot, name, type, format, swizzle) type name : CONCAT(S_TARGET_OUTPUT, slot);\n\
               struct O {\n\
               GBUFFER_OP(1, g_a, float4, F, S)\n\
               GBUFFER_OP(2, g_b, float4, F, S)\n\
               };\n\
               #undef GBUFFER_OP\n";
    let out = preprocess(src, None).unwrap();
    assert!(
        out.contains("float4 g_a : S_TARGET_OUTPUT1;"),
        "slot must paste, out: {out}"
    );
    assert!(
        out.contains("float4 g_b : S_TARGET_OUTPUT2;"),
        "slot must paste, out: {out}"
    );
    assert!(!out.contains("GBUFFER_OP"), "out: {out}");
    assert!(!out.contains("##"), "paste must resolve, out: {out}");
}

#[test]
fn undef_drops_both_macro_kinds() {
    let src = "#define PLAIN 7\n#define TWICE(x) ((x) + (x))\nfloat a = PLAIN;\nfloat b = TWICE(3);\n#undef PLAIN\n#undef TWICE\nfloat c = PLAIN;\nfloat d = TWICE(4);\n";
    let out = preprocess(src, None).unwrap();
    assert!(out.contains("float a = 7;"), "out: {out}");
    assert!(out.contains("float b = ((3) + (3));"), "out: {out}");
    // After `#undef`, invocations stay verbatim (loud downstream).
    assert!(out.contains("float c = PLAIN;"), "out: {out}");
    assert!(out.contains("float d = TWICE(4);"), "out: {out}");
}

#[test]
fn func_arity_mismatch_stays_verbatim() {
    // Wrong arity is not an invocation: leave it for a loud failure
    // rather than mis-expanding.
    let src = "#define ADD(a, b) ((a) + (b))\nfloat x = ADD(1);\nfloat y = ADD(1, 2, 3);\nfloat z = ADD(1, 2);\n";
    let out = preprocess(src, None).unwrap();
    assert!(out.contains("float x = ADD(1);"), "out: {out}");
    assert!(out.contains("float y = ADD(1, 2, 3);"), "out: {out}");
    assert!(out.contains("float z = ((1) + (2));"), "out: {out}");
}

#[test]
fn ifdef_sees_func_macros() {
    // Header guards and feature tests work for function-like macros too.
    let src = "#define GUARD(x) (x)\n#ifdef GUARD\nkept\n#endif\n#ifndef GUARD\ndropped\n#endif\n";
    let out = preprocess(src, None).unwrap();
    assert!(out.contains("kept"), "out: {out}");
    assert!(!out.contains("dropped"), "out: {out}");
}
