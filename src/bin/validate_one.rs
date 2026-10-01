use std::env;
use std::path::{Path, PathBuf};

fn main() {
    ps5_shader::tracing::init_tracing();

    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));

    // Make DXC visible (same as bulk runner)
    let dxc_dir = manifest_dir.join("dxc");
    if dxc_dir.join("dxcompiler.dll").exists() {
        let existing = std::env::var("PATH").unwrap_or_default();
        unsafe {
            std::env::set_var("PATH", format!("{};{}", dxc_dir.display(), existing));
        }
    }

    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage:");
        eprintln!("  cargo run --bin validate_one -- <shader-name>");
        eprintln!("  cargo run --bin validate_one -- <path/to/shader.pssl>");
        eprintln!();
        eprintln!("Examples:");
        eprintln!("  cargo run --bin validate_one -- agc_simple_compute_cs");
        eprintln!(
            "  cargo run --bin validate_one -- tests/shaders/agc_simple_compute_cs/shader.pssl"
        );
        std::process::exit(1);
    }

    let input = &args[1];
    let path = resolve_shader_path(&manifest_dir, input);

    println!("Validating: {}", path.display());
    println!();

    match validate_one(&path) {
        Ok(()) => {
            println!("✅ Fully validated (PSSL → HLSL → DXC → SPIR-V)");
        }
        Err(e) => {
            eprintln!("❌ {}", e);
            std::process::exit(1);
        }
    }
}

fn resolve_shader_path(manifest_dir: &Path, input: &str) -> PathBuf {
    let p = PathBuf::from(input);

    // Full path given
    if p.exists() && p.extension().map(|e| e == "pssl").unwrap_or(false) {
        return p;
    }

    // Treat as directory name under tests/shaders/
    let candidate = manifest_dir
        .join("tests/shaders")
        .join(input)
        .join("shader.pssl");

    if candidate.exists() {
        return candidate;
    }

    eprintln!("Could not find shader for '{}'", input);
    eprintln!("Looked for: {}", candidate.display());
    std::process::exit(1);
}

fn emit_preprocess_summary(report: &ps5_shader::preprocess::PreprocessReport) {
    use ps5_shader::preprocess::{IncludeClass, IncludeResolution};

    // Per-include classification trace (retained for forensic visibility).
    for ev in &report.includes {
        let class = ev.class.label().to_string();
        let res = match &ev.resolution {
            IncludeResolution::Found(p) => format!("resolved -> {}", p.display()),
            IncludeResolution::Ignored(_) => {
                if matches!(ev.class, IncludeClass::HostApi(_)) {
                    "host API chain skipped".to_string()
                } else {
                    "ignored (C/C++ infra)".to_string()
                }
            }
            IncludeResolution::NotFound(..) => "unresolved".to_string(),
        };
        let prov = if ev.provenance.is_empty() {
            String::new()
        } else {
            format!("  <- {}", ev.provenance.join(" -> "))
        };
        tracing::debug!(
            target: "pssl_shader::preprocess",
            name = %ev.name,
            class = %class,
            resolution = %res,
            provenance = %prov,
            "include"
        );
    }

    if report.includes.is_empty() {
        return;
    }

    println!();
    println!("  Include summary");
    println!("  ────────────────────────────────");
    println!("  Resolved shader headers       {:>3}", report.resolved_shader);
    println!("  Host API chains skipped       {:>3}", report.host_api_chains_skipped);
    if !report.host_api_chain_names.is_empty() {
        println!(
            "    chains: {}",
            report
                .host_api_chain_names
                .iter()
                .map(|s| format!("`{s}`"))
                .collect::<Vec<_>>()
                .join(" ")
        );
    }
    println!("  C/C++ infrastructure ignored  {:>3}", report.ignored_host);
    if !report.ignored_names.is_empty() {
        println!("  Ignored infrastructure:");
        for n in &report.ignored_names {
            println!("    {n}");
        }
    }
    println!("  Include events                {:>3}", report.includes.len());
    println!("  Shader-relevant missing       {:>3}", report.unresolved_shader);
    for w in &report.dependency_warnings {
        println!("  ⚠ {w}");
    }
}

fn validate_one(path: &Path) -> Result<(), String> {
    let name = path
        .parent()
        .and_then(|p| p.file_name())
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| path.display().to_string());

    let raw = std::fs::read_to_string(path).map_err(|e| format!("READ_ERROR: {e}"))?;

    // 1. Preprocess
    tracing::debug!(target: "pssl_shader::preprocess", path = %path.display(), "preprocessing");
    let cfg = ps5_shader::preprocess::PreprocessorConfig::platform_defaults();
    let out = ps5_shader::preprocess::preprocess_reported(&raw, Some(path), &cfg).map_err(|e| {
        let msg = e.to_string();
        let first = msg.lines().next().unwrap_or(&msg);
        format!("preprocess: {first}")
    })?;
    let pre = out.source;
    let report = out.report;

    tracing::info!(target: "pssl_shader::preprocess", input_bytes = raw.len(), output_bytes = pre.len(), "OK");
    emit_preprocess_summary(&report);
    println!("✓ Preprocess");
    let _ = std::fs::write(
        std::env::temp_dir().join("ps5-shader-pre_dump.pssl"),
        &pre,
    );

    tracing::trace!(target: "pssl_shader::psfx::resolver", pre_len = pre.len(), "preprocessed source ready for parsing");
    // 2. Parse
    tracing::debug!(target: "pssl_shader::parse", "parsing");
    let m = ps5_shader::parse_shader(&pre).map_err(|e| {
        let msg = e.to_string();
        let first = msg.lines().next().unwrap_or(&msg);
        let mut suffix = String::new();
        if let Some(span) = e.span {
            let line = pre[..span.start.min(pre.len())].matches('\n').count() + 1;
            suffix = format!(" (offset {} line {line})", span.start);
        }
        format!("parse: {first}{suffix}")
    })?;

    tracing::info!(target: "pssl_shader::parse", "OK");
    println!("✓ Parse");

    // 3. Semantic
    tracing::debug!(target: "pssl_shader::semantic", "analyzing");
    let model = ps5_shader::semantic::analyze(&m).map_err(|e| {
        let msg = e.to_string();
        let first = msg.lines().next().unwrap_or(&msg);
        format!("semantic: {first}")
    })?;

    tracing::info!(target: "pssl_shader::semantic", "OK");
    println!("✓ Semantic");

    // 4. Lower → IR
    tracing::debug!(target: "pssl_shader::lower", "lowering to IR");
    let ir = ps5_shader::lower::lower_module(&m, &model).map_err(|e| {
        let msg = e.to_string();
        let first = msg.lines().next().unwrap_or(&msg);
        format!("lower: {first}")
    })?;

    tracing::info!(target: "pssl_shader::lower", functions = ir.functions.len(), resources = ir.resources.len(), "OK");
    println!("✓ IR");

    // 5. Emit HLSL
    tracing::debug!(target: "pssl_shader::emit", "emitting HLSL");
    let hlsl = ps5_shader::emit::emit_vulkan_hlsl(&ir);
    tracing::info!(target: "pssl_shader::emit", bytes = hlsl.len(), "OK");
    println!("✓ HLSL emitted ({} bytes)", hlsl.len());

    // Optional: write the generated HLSL next to the shader for inspection
    let hlsl_path = path.with_extension("generated.hlsl");
    let _ = std::fs::write(&hlsl_path, &hlsl);
    println!("  → wrote {}", hlsl_path.display());

    // Failure snapshots: keep the last failing HLSL next to the shader as
    // `shader.fail.hlsl` plus the full diagnostics as `shader.fail.log`
    // (both gitignored, like `shader.generated.hlsl`) so a red sample's
    // exact reproducer survives beside it. Removed on success.
    let fail_path = path.with_extension("fail.hlsl");
    let fail_log_path = path.with_extension("fail.log");
    let snapshot_failure = |why: &str, detail: &str| {
        let _ = std::fs::write(&fail_path, &hlsl);
        let _ = std::fs::write(&fail_log_path, detail);
        println!("  → wrote {} + {} ({why})", fail_path.display(), fail_log_path.display());
    };

    let stage = model.stage.ok_or_else(|| {
        snapshot_failure("no stage", "no stage: semantic analysis produced no shader stage");
        "no stage".to_string()
    })?;

    let entry = model
        .entry_point
        .as_ref()
        .map(|ep| ep.function.clone())
        .ok_or_else(|| {
            snapshot_failure("no entry", "no entry: semantic analysis produced no entry point");
            "no entry".to_string()
        })?;

    // 6. DXC
    tracing::debug!(target: "pssl_shader::dxc", entry = %entry, stage = ?stage, "compiling via DXC");
    let dxc_bytes = ps5_shader::dxc::compile_hlsl(&hlsl, &entry, stage).map_err(|e| {
        let msg = e.to_string();
        let first = msg
            .lines()
            .find(|l| l.contains("error:"))
            .unwrap_or(msg.lines().next().unwrap_or(&msg));
        snapshot_failure("dxc", &msg);
        format!("DXC: {first}")
    })?;

    tracing::info!(target: "pssl_shader::dxc", bytes = dxc_bytes.len(), "OK");
    println!("✓ DXC ({} bytes of DXIL/SPIR-V)", dxc_bytes.len());

    // 7. SPIR-V validate
    let words = ps5_shader::spirv::bytes_to_words(&dxc_bytes).map_err(|e| {
        let detail = format!("spirv decode: {e}");
        snapshot_failure("spirv-decode", &detail);
        format!("spirv decode: {e}")
    })?;

    tracing::debug!(target: "pssl_shader::validate", words = words.len(), "validating SPIR-V");
    ps5_shader::spirv::validate_spirv(&words).map_err(|e| {
        let msg = e.to_string();
        let first = msg.lines().next().unwrap_or(&msg);
        snapshot_failure("spirv-validate", &msg);
        format!("validate: {first}")
    })?;
    tracing::info!(target: "pssl_shader::validate", "OK");

    // Success: stale failure snapshots must not linger beside a green shader.
    let _ = std::fs::remove_file(&fail_path);
    let _ = std::fs::remove_file(&fail_log_path);

    println!("✓ SPIR-V validated");
    println!();
    println!("Shader:  {name}");
    println!("Stage:   {:?}", stage);
    println!("Entry:   {entry}");

    Ok(())
}
