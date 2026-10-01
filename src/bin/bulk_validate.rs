use std::path::{Path, PathBuf};

use rayon::prelude::*;
use serde_json::json;

/// One shader's result row:
/// `(name, parse, semantic, ir, hlsl, dxc, spirv, notes)`.
type Row = (String, String, String, String, String, String, String, String);

/// Persists the per-shader DXC outcome for the differential corpus:
/// `analysis/dxc-result.json` beside `semantic.json` / `ir.json`
/// (PSSL -> semantic -> IR -> HLSL -> DXC vs official `.ags` reference).
fn record_dxc_result(dir: &Path, detail: &serde_json::Value) {
    let out_dir = dir.join("analysis");
    if std::fs::create_dir_all(&out_dir).is_ok() {
        if let Ok(text) = serde_json::to_string(detail) {
            let _ = std::fs::write(out_dir.join("dxc-result.json"), text);
        }
    }
}

/// Prints the CLI help/usage text.
fn print_help() {
    println!(
        "bulk_validate — validate PSSL shaders and update shader-DXC-validated.md\n\
         \n\
         Ran without options, this help is shown. Options:\n\
         \n\
         \x20 --full\n\
         \x20     Regenerate the entire report: re-validate every shader in the\n\
         \x20     corpus and rewrite all of shader-DXC-validated.md from scratch.\n\
         \n\
         \x20 --lines <N> <N> ...\n\
         \x20     Validate only the shaders listed on the given line numbers of\n\
         \x20     shader-DXC-validated.md (table rows), then update just those rows.\n\
         \x20     Example: --lines 26 36\n\
         \n\
         \x20 --match <keyword>\n\
         \x20     Validate every shader whose current report Notes contain\n\
         \x20     <keyword> but that do NOT yet validate (no check at SPIR-V), then\n\
         \x20     update just those rows. Example: --match cstdint\n\
         \n\
         \x20 --limit <N>\n\
         \x20     Cap the number of shaders validated to at most N. Combine with\n\
         \x20     --lines or --match. Example: --match cstdint --limit 20\n\
         \n\
         \x20 NAME [NAME ...]\n\
         \x20     Validate the shader(s) whose tests/shaders folder name matches.\n\
         \x20     Any bare (non-flag) argument is treated as a shader name. Each row\n\
         \x20     is updated in place.\n\
         \x20     Example: sample__audio_video__api_vdecsw__common__shader_vv\n\
         \n\
         \x20 --sequential\n\
         \x20     Validate one shader at a time, single-threaded, instead of in\n\
         \x20     parallel. Each shader runs under a watchdog that abandons it and\n\
         \x20     records a `timeout` row after `--timeout` seconds, so a single\n\
         \x20     hanging compile cannot stall the whole run.\n\
         \n\
         \x20 --timeout <secs>\n\
         \x20     Per-shader watchdog seconds (default 5). Implies --sequential.\n\
         \n\
         \x20 --help | -h\n\
         \x20     Show this help.\n"
    );
}

/// Returns the shader directory for a shader name if it exists.
fn shader_dir(shaders_root: &Path, name: &str) -> Option<PathBuf> {
    let dir = shaders_root.join(name);
    if dir.join("shader.pssl").exists() {
        Some(dir)
    } else {
        None
    }
}

/// Reads `shader-DXC-validated.md` and extracts the shader names on the given
/// 1-based line numbers. Line numbering follows the file as-is (including the
/// header). Returns names, skipping blank/non-table lines and out-of-range lines.
fn names_from_md_lines(manifest_dir: &Path, lines: &[u32]) -> Vec<String> {
    let md_path = manifest_dir.join("shader-DXC-validated.md");
    let content = match std::fs::read_to_string(&md_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("cannot read {}: {e}", md_path.display());
            std::process::exit(2);
        }
    };
    let all_lines: Vec<&str> = content.lines().collect();
    let mut names = Vec::new();
    let mut skipped = 0usize;
    for &n in lines {
        let idx = n as usize;
        if idx == 0 || idx > all_lines.len() {
            eprintln!("warning: line {n} out of range (file has {} lines)", all_lines.len());
            skipped += 1;
            continue;
        }
        let text = all_lines[idx - 1]; // 1-based
        let trimmed = text.trim();
        // A table row looks like `| name | ✅ | ... | notes |`.
        if !trimmed.starts_with('|') {
            eprintln!("warning: line {n} is not a table row, skipping: {text:?}");
            skipped += 1;
            continue;
        }
        let cols: Vec<&str> = text.split('|').map(|s| s.trim()).collect();
        // cols[0] is empty (before first `|`), cols[1] is the shader name.
        let name = cols.get(1).map(|s| s.trim()).unwrap_or("");
        if name.is_empty() {
            eprintln!("warning: line {n} has no shader name, skipping: {text:?}");
            skipped += 1;
            continue;
        }
        names.push(name.to_string());
    }
    if skipped > 0 {
        println!(
            "note: {}/{} requested line(s) were skipped (not shader rows, out of range, or empty)",
            skipped,
            lines.len()
        );
    }
    names
}

/// Reads `shader-DXC-validated.md` and returns the names of shaders whose table
/// row Notes contain `keyword` (case-insensitive) but whose SPIR-V cell is *not*
/// a check mark — i.e. shaders that failed for a reason containing `keyword` and
/// have not yet validated.
fn names_matching_md_keyword(manifest_dir: &Path, keyword: &str) -> Vec<String> {
    let md_path = manifest_dir.join("shader-DXC-validated.md");
    let content = match std::fs::read_to_string(&md_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("cannot read {}: {e}", md_path.display());
            std::process::exit(2);
        }
    };
    let kw = keyword.to_lowercase();
    let mut names = Vec::new();
    for line in content.lines() {
        let trimmed = line.trim_matches(' ');
        if !trimmed.starts_with('|') {
            continue;
        }
        let cells: Vec<&str> = trimmed.split('|').map(|s| s.trim()).collect();
        // cells: ["", name, parse, sem, ir, hlsl, dxc, spirv, notes, ""]
        let name = cells.get(1).map(|s| s.trim()).unwrap_or("");
        let spirv = cells.get(7).map(|s| s.trim()).unwrap_or("");
        let notes = cells.get(8).map(|s| s.trim()).unwrap_or("");
        if name.is_empty() {
            continue;
        }
        if spirv == "✅" {
            continue; // already validated — skip
        }
        if notes.to_lowercase().contains(&kw) {
            names.push(name.to_string());
        }
    }
    names
}

/// Runs the PSSL -> HLSL -> DXC -> SPIR-V pipeline for a single shader and
/// returns its table row: `(parse, semantic, ir, hlsl, dxc, spirv, notes)`.
fn validate_one(dir: &Path) -> (String, String, String, String, String, String, String) {
    let path = dir.join("shader.pssl");
    let name = dir
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_default();
    let raw = std::fs::read_to_string(&path).unwrap_or_else(|e| format!("READ_ERROR: {e}"));

    let pre = match ps5_shader::preprocess::preprocess(&raw, Some(&path)) {
        Ok(s) => s,
        Err(e) => {
            let msg = e.to_string();
            let first = msg.lines().next().unwrap_or(&msg).replace('|', "/");
            record_dxc_result(
                dir,
                &json!({"shader": name, "parse": "❌", "error": format!("preprocess: {first}")}),
            );
            return ("❌".into(), "—".into(), "—".into(), "—".into(), "—".into(), "—".into(), format!("preprocess: {first}"));
        }
    };
    let m = match ps5_shader::parse_shader(&pre) {
        Ok(m) => m,
        Err(e) => {
            let msg = e.to_string();
            let first = msg.lines().next().unwrap_or(&msg).replace('|', "/");
            let short = first.chars().take(80).collect::<String>();
            record_dxc_result(
                dir,
                &json!({"shader": name, "parse": "❌", "error": format!("parse: {short}")}),
            );
            return ("❌".into(), "—".into(), "—".into(), "—".into(), "—".into(), "—".into(), format!("parse: {short}"));
        }
    };
    let model = match ps5_shader::semantic::analyze(&m) {
        Ok(model) => model,
        Err(e) => {
            let msg = e.to_string();
            let first = msg.lines().next().unwrap_or(&msg).replace('|', "/");
            let short = first.chars().take(80).collect::<String>();
            record_dxc_result(
                dir,
                &json!({"shader": name, "parse": "✅", "semantic": "❌", "error": format!("semantic: {short}")}),
            );
            return ("✅".into(), "❌".into(), "—".into(), "—".into(), "—".into(), "—".into(), format!("semantic: {short}"));
        }
    };
    let ir = match ps5_shader::lower::lower_module(&m, &model) {
        Ok(ir) => ir,
        Err(e) => {
            let msg = e.to_string();
            let first = msg.lines().next().unwrap_or(&msg).replace('|', "/");
            let short = first.chars().take(80).collect::<String>();
            record_dxc_result(
                dir,
                &json!({"shader": name, "parse": "✅", "semantic": "✅", "ir": "❌", "error": format!("lower: {short}")}),
            );
            return ("✅".into(), "✅".into(), "❌".into(), "—".into(), "—".into(), "—".into(), format!("lower: {short}"));
        }
    };
    let hlsl = ps5_shader::emit::emit_vulkan_hlsl(&ir);
    let hlsl_len = hlsl.len();
    let (stage, entry) = match (model.stage, model.entry_point.as_ref().map(|ep| ep.function.clone())) {
        (Some(stage), Some(entry)) => (stage, entry),
        (None, _) => {
            record_dxc_result(
                dir,
                &json!({"shader": name, "parse": "✅", "semantic": "✅", "ir": "✅", "hlsl": "✅", "dxc": "❌", "hlsl_bytes": hlsl_len, "error": "no stage"}),
            );
            return ("✅".into(), "✅".into(), "✅".into(), "✅".into(), "❌".into(), "—".into(), "no stage".into());
        }
        (_, None) => {
            record_dxc_result(
                dir,
                &json!({"shader": name, "parse": "✅", "semantic": "✅", "ir": "✅", "hlsl": "✅", "dxc": "❌", "hlsl_bytes": hlsl_len, "error": "no entry"}),
            );
            return ("✅".into(), "✅".into(), "✅".into(), "✅".into(), "❌".into(), "—".into(), "no entry".into());
        }
    };
    let stage_name = format!("{stage:?}");
    let dxc_bytes = match ps5_shader::dxc::compile_hlsl(&hlsl, &entry, stage) {
        Ok(b) => b,
        Err(e) => {
            let msg = e.to_string();
            let first = msg
                .lines()
                .find(|l| l.contains("error:"))
                .unwrap_or(msg.lines().next().unwrap_or(&msg))
                .replace('|', "/");
            let short = first.chars().take(120).collect::<String>();
            record_dxc_result(
                dir,
                &json!({"shader": name, "parse": "✅", "semantic": "✅", "ir": "✅", "hlsl": "✅", "dxc": "❌", "stage": stage_name, "entry": entry, "hlsl_bytes": hlsl_len, "dxc_error": short}),
            );
            return ("✅".into(), "✅".into(), "✅".into(), "✅".into(), "❌".into(), "—".into(), format!("DXC: {short}"));
        }
    };
    let words = match ps5_shader::spirv::bytes_to_words(&dxc_bytes) {
        Ok(w) => w,
        Err(e) => {
            let short = e.to_string().chars().take(80).collect::<String>();
            record_dxc_result(
                dir,
                &json!({"shader": name, "parse": "✅", "semantic": "✅", "ir": "✅", "hlsl": "✅", "dxc": "✅", "spirv": "❌", "stage": stage_name, "entry": entry, "hlsl_bytes": hlsl_len, "spirv_error": short}),
            );
            return ("✅".into(), "✅".into(), "✅".into(), "✅".into(), "✅".into(), "❌".into(), format!("spirv words: {short}"));
        }
    };
    match ps5_shader::spirv::validate_spirv(&words) {
        Ok(()) => {
            record_dxc_result(
                dir,
                &json!({"shader": name, "parse": "✅", "semantic": "✅", "ir": "✅", "hlsl": "✅", "dxc": "✅", "spirv": "✅", "stage": stage_name, "entry": entry, "hlsl_bytes": hlsl_len}),
            );
            ("✅".into(), "✅".into(), "✅".into(), "✅".into(), "✅".into(), "✅".into(), "".into())
        }
        Err(e) => {
            let msg = e.to_string();
            let first = msg.lines().next().unwrap_or(&msg).replace('|', "/");
            let short = first.chars().take(80).collect::<String>();
            record_dxc_result(
                dir,
                &json!({"shader": name, "parse": "✅", "semantic": "✅", "ir": "✅", "hlsl": "✅", "dxc": "✅", "spirv": "❌", "stage": stage_name, "entry": entry, "hlsl_bytes": hlsl_len, "spirv_error": format!("validate: {short}")}),
            );
            ("✅".into(), "✅".into(), "✅".into(), "✅".into(), "✅".into(), "❌".into(), format!("validate: {short}"))
        }
    }
}

fn main() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let dxc_dir = manifest_dir.join("dxc");
    if dxc_dir.join("dxcompiler.dll").exists() {
        let existing = std::env::var("PATH").unwrap_or_default();
        unsafe { std::env::set_var("PATH", format!("{};{}", dxc_dir.display(), existing)) };
    }
    // Parse CLI args (see `print_help` for descriptions):
    //   --lines <N> <N> ...  : validate only the shaders on those MD table lines (patch)
    //   --match <keyword>     : validate shaders whose MD row Notes contain <keyword>
    //                           but whose SPIR-V cell is not a check mark (patch)
    //   --limit <N>           : cap the selected shaders to at most N
    //   --full                : regenerate the entire corpus + MD
    //   NAME [NAME ...]       : validate the shader(s) whose tests/shaders folder
    //                           name matches (bare, non-flag arguments)
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        print_help();
        std::process::exit(0);
    }
    let mut line_targets: Vec<u32> = Vec::new();
    let mut name_targets: Vec<String> = Vec::new();
    let mut match_keyword: Option<String> = None;
    let mut limit: Option<usize> = None;
    let mut full = false;
    let mut sequential = false;
    let mut timeout_secs: Option<u64> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--lines" => {
                i += 1;
                while i < args.len() && args[i].parse::<u32>().is_ok() {
                    line_targets.push(args[i].parse::<u32>().unwrap());
                    i += 1;
                }
                if line_targets.is_empty() {
                    eprintln!("error: --lines requires at least one line number");
                    std::process::exit(2);
                }
            }
            "--match" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("error: --match requires a keyword");
                    std::process::exit(2);
                }
                match_keyword = Some(args[i].clone());
            }
            "--limit" => {
                i += 1;
                match args.get(i).and_then(|s| s.parse::<usize>().ok()) {
                    Some(n) => limit = Some(n),
                    None => {
                        eprintln!("error: --limit requires a positive integer");
                        std::process::exit(2);
                    }
                }
            }
            "--full" => {
                full = true;
            }
            "--sequential" => {
                sequential = true;
            }
            "--timeout" => {
                i += 1;
                match args.get(i).and_then(|s| s.parse::<u64>().ok()) {
                    Some(n) if n > 0 => {
                        timeout_secs = Some(n);
                        sequential = true; // timeout only makes sense single-threaded
                    }
                    _ => {
                        eprintln!("error: --timeout requires a positive integer");
                        std::process::exit(2);
                    }
                }
            }
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => {
                if other.starts_with('-') {
                    eprintln!("error: unknown argument {other:?}");
                    print_help();
                    std::process::exit(2);
                }
                // Bare argument: treat as a shader folder name.
                name_targets.push(other.to_string());
            }
        }
        i += 1;
    }


    let shaders_root = manifest_dir.join("tests/shaders");
    let all_dirs: Vec<PathBuf> = {
        let mut v: Vec<PathBuf> = std::fs::read_dir(&shaders_root)
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.join("shader.pssl").exists())
            .collect();
        v.sort();
        v
    };

    // Determine which shaders to validate. `--full` regenerates the whole corpus.
    // Otherwise selection is the (capped) union of names from bare shader-name
    // arguments, `--lines` names and any `--match` keyword, patching only those
    // rows. A bare `--limit` without any target set falls back to a full run.
    let has_selection = !line_targets.is_empty() || !name_targets.is_empty() || match_keyword.is_some();
    let (dirs, patch, match_desc) = if full || !has_selection {
        (all_dirs, false, String::new())
    } else {
        let mut names: Vec<String> = names_from_md_lines(&manifest_dir, &line_targets);
        // Bare shader-name arguments map 1:1 onto tests/shaders folder names.
        names.extend(name_targets.iter().cloned());
        if let Some(kw) = &match_keyword {
            names.extend(names_matching_md_keyword(&manifest_dir, kw));
        }
        // De-duplicate, preserving order.
        let mut seen = std::collections::HashSet::new();
        names.retain(|n| seen.insert(n.clone()));

        // Apply limit.
        if let Some(n) = limit {
            names.truncate(n);
        }

        let mut selected: Vec<PathBuf> = Vec::new();
        for name in &names {
            match shader_dir(&shaders_root, name) {
                Some(d) => selected.push(d),
                None => eprintln!("warning: shader {name:?} not found under {}", shaders_root.display()),
            }
        }
        selected.sort();
        let desc = match (&match_keyword, limit) {
            (Some(k), Some(n)) => format!(" (match {k:?}, limit {n})"),
            (Some(k), None) => format!(" (match {k:?})"),
            (None, Some(n)) => format!(" (limit {n})"),
            (None, None) => String::new(),
        };
        (selected, true, desc)
    };

    if dirs.is_empty() {
        eprintln!("no shaders to validate");
        std::process::exit(1);
    }

    println!(
        "Validating {} shader(s){}...",
        dirs.len(),
        if patch { &match_desc } else { "" }
    );
    let pb = indicatif::ProgressBar::new(dirs.len() as u64);
    pb.set_style(
        indicatif::ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} {msg}",
        )
        .unwrap()
        .progress_chars("#>-"),
    );
    pb.set_message("compiling");

    let mut results: Vec<Row> = if sequential {
        let secs = timeout_secs.unwrap_or(5);
        println!("sequential, per-shader watchdog {secs}s...");
        let mut out = Vec::with_capacity(dirs.len());
        for dir in &dirs {
            let name = dir.file_name().unwrap().to_string_lossy().to_string();
            let dir = dir.clone();
            let worker_name = name.clone();
            let (tx, rx) = std::sync::mpsc::channel();
            let worker = std::thread::spawn(move || {
                let (parse, semantic, ir, hlsl, dxc, spirv, notes) = validate_one(&dir);
                let _ = tx.send((worker_name, parse, semantic, ir, hlsl, dxc, spirv, notes));
            });
            let row = match rx.recv_timeout(std::time::Duration::from_secs(secs)) {
                Ok(row) => row,
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                    eprintln!("  timeout: {name} (>{secs}s)");
                    (
                        name,
                        "—".into(),
                        "—".into(),
                        "—".into(),
                        "—".into(),
                        "—".into(),
                        "—".into(),
                        format!("timeout: >{secs}s"),
                    )
                }
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                    (
                        name,
                        "—".into(),
                        "—".into(),
                        "—".into(),
                        "—".into(),
                        "—".into(),
                        "—".into(),
                        "worker panicked".into(),
                    )
                }
            };
            // The watchdog lets a hung worker run in the background; it dies
            // with the process. We record the timeout row so MD totals hold.
            let _ = worker;
            pb.inc(1);
            out.push(row);
        }
        out
    } else {
        dirs.par_iter()
            .map(|dir| {
                let name = dir.file_name().unwrap().to_string_lossy().to_string();
                let (parse, semantic, ir, hlsl, dxc, spirv, notes) = validate_one(dir);
                let r = (name, parse, semantic, ir, hlsl, dxc, spirv, notes);
                pb.inc(1);
                r
            })
            .collect()
    };
    pb.finish_with_message("done");
    results.sort_by(|a, b| a.0.cmp(&b.0));

    let md_path = manifest_dir.join("shader-DXC-validated.md");

    if patch {
        // Patch mode: update only the selected shader rows in place, preserving
        // every other entry exactly as it is in the existing file.
        patch_md(&md_path, &results);
    } else {
        write_full_md(&md_path, &results);
    }

    let validated = results
        .iter()
        .filter(|(_, _, _, _, _, _, spirv, _)| spirv == "✅")
        .count();
    println!(
        "Wrote {} with {} entry/entries updated, validated {}/{}",
        md_path.display(),
        results.len(),
        validated,
        results.len()
    );
}

/// Recomputes the summary counts from the markdown table's `✅` markers.
///
/// Table row: `| name | parse | semantic | ir | hlsl | dxc | spirv | notes |`.
/// After splitting on `|`: index 1 = name, 2 = parse, 6 = dxc, 7 = spirv.
fn count_summary(table_rows: &[String]) -> (usize, usize, usize) {
    let mut parse = 0;
    let mut dxc = 0;
    let mut validated = 0;
    for r in table_rows {
        let cells: Vec<&str> = r.split('|').map(|s| s.trim()).collect();
        if cells.get(2).copied() == Some("✅") {
            parse += 1;
        }
        if cells.get(6).copied() == Some("✅") {
            dxc += 1;
        }
        if cells.get(7).copied() == Some("✅") {
            validated += 1;
        }
    }
    (parse, dxc, validated)
}

/// Rewrites the whole markdown file from a full set of results.
fn write_full_md(md_path: &Path, results: &[Row]) {
    let mut md = String::new();
    md.push_str("# Shader DXC Validation — PSSL → HLSL → DXC → SPIR-V\n\n");
    md.push_str(&format!("Total shaders: {}\n\n", results.len()));
    let validated = results.iter().filter(|r| r.6 == "✅").count();
    let dxc_ok = results.iter().filter(|r| r.5 == "✅").count();
    let parse_ok = results.iter().filter(|r| r.1 == "✅").count();
    md.push_str(&format!("- Parse: {}/{}\n", parse_ok, results.len()));
    md.push_str(&format!("- DXC: {}/{}\n", dxc_ok, results.len()));
    md.push_str(&format!("- Validated (SPIR-V): {}/{}\n\n", validated, results.len()));
    md.push_str("| Shader | Parse | Semantic | IR | HLSL | DXC | SPIR-V | Notes |\n");
    md.push_str("|---|---|---|---|---|---|---|---|\n");
    for (name, parse, semantic, ir, hlsl, dxc, spirv, notes) in results {
        md.push_str(&format!("| {name} | {parse} | {semantic} | {ir} | {hlsl} | {dxc} | {spirv} | {notes} |\n"));
    }
    md.push_str("\n## Pipeline Coverage\n\n```\n");
    let total = results.len() as f32;
    let stages = [
        ("Parse", parse_ok),
        ("Semantic", results.iter().filter(|r| r.2 == "✅").count()),
        ("IR", results.iter().filter(|r| r.3 == "✅").count()),
        ("HLSL", results.iter().filter(|r| r.4 == "✅").count()),
        ("DXC", dxc_ok),
        ("Validated", validated),
    ];
    for (label, count) in stages {
        let pct = (count as f32 / total * 100.0) as u32;
        let bar = "█".repeat((pct / 5) as usize);
        md.push_str(&format!("{label:<10} {count:>4}/{} ({pct:>2}%) {bar}\n", results.len()));
    }
    md.push_str("```\n");
    std::fs::write(md_path, md).unwrap();
}

/// Patch mode: updates only the rows for the given results in an existing markdown
/// table, preserving all other rows and re-deriving the summary counts + table
/// (keeping any trailing sections like the histogram intact where possible).
fn patch_md(md_path: &Path, results: &[Row]) {
    let content = std::fs::read_to_string(md_path).unwrap_or_else(|e| {
        eprintln!("cannot read {} for patch: {e}", md_path.display());
        std::process::exit(2);
    });
    let mut lines: Vec<String> = content.lines().map(|s| s.to_string()).collect();
    let updated: std::collections::HashMap<&str, &Row> = results
        .iter()
        .map(|r| (r.0.as_str(), r))
        .collect();

    // Replace matching table rows in place.
    let mut changed = 0usize;
    let mut table_rows: Vec<String> = Vec::new();
    for line in &mut lines {
        let trimmed = line.trim_matches(' ');
        // Detect a table row structurally (9 data pipes) so rows with all
        // `—` cells (timeouts) still count, instead of requiring a ✅/❌.
        let cells: Vec<&str> = trimmed.split('|').collect();
        let name = cells.get(1).map(|c| c.trim()).unwrap_or("");
        let is_row = cells.len() == 10
            && !name.is_empty()
            && name != "Shader"
            && !name.starts_with("---")
            && !name.chars().all(|c| c == '-');
        if is_row {
            let name = cells[1].trim();
            if let Some(row) = updated.get(name) {
                *line = format!("| {} | {} | {} | {} | {} | {} | {} | {} |", row.0, row.1, row.2, row.3, row.4, row.5, row.6, row.7);
                changed += 1;
            }
            table_rows.push(line.clone());
        }
    }

    // Recompute summary counts from the (patched) table rows and rewrite the
    // header lines: `- Parse`, `- DXC`, `- Validated`.
    let (parse_ok, dxc_ok, validated) = count_summary(&table_rows);
    let total = table_rows.len();
    for line in &mut lines {
        if line.starts_with("- Parse:") {
            *line = format!("- Parse: {parse_ok}/{total}");
        } else if line.starts_with("- DXC:") {
            *line = format!("- DXC: {dxc_ok}/{total}");
        } else if line.starts_with("- Validated") {
            *line = format!("- Validated (SPIR-V): {validated}/{total}");
        }
    }

    std::fs::write(md_path, lines.join("\n") + "\n").unwrap();
    println!("Patched {changed} row(s) in {}", md_path.display());
}
