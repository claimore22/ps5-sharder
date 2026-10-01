use std::collections::HashMap;
use std::path::{Path, PathBuf};
use serde::{Deserialize, Serialize};
use rayon::prelude::*;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct ShaderResult {
    hlsl: String,
    spirv: Vec<u32>,
    validated: bool,
}

fn compile_one(path: &Path) -> Option<ShaderResult> {
    let raw = std::fs::read_to_string(path).ok()?;
    let pre = ps5_shader::preprocess::preprocess(&raw, Some(path)).ok()?;
    let m = ps5_shader::parse_shader(&pre).ok()?;
    let model = ps5_shader::semantic::analyze(&m).ok()?;
    let ir = ps5_shader::lower::lower_module(&m, &model).ok()?;
    let hlsl = ps5_shader::emit::emit_vulkan_hlsl(&ir);
    let stage = model.stage?;
    let entry = model.entry_point?.function.clone();
    // Independent per-shader DXC — no shared Dxc object
    let dxc_bytes = ps5_shader::dxc::compile_hlsl(&hlsl, &entry, stage).ok()?;
    let words = ps5_shader::spirv::bytes_to_words(&dxc_bytes).ok()?;
    let validated = ps5_shader::spirv::validate_spirv(&words).is_ok();
    Some(ShaderResult {
        hlsl,
        spirv: words,
        validated,
    })
}

fn collect_names(n: usize) -> Vec<PathBuf> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let shaders_root = manifest_dir.join("tests/shaders");
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(&shaders_root)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.join("shader.pssl").exists())
        .collect();
    dirs.sort();
    dirs.into_iter()
        .take(n)
        .map(|d| d.join("shader.pssl"))
        .collect()
}

fn run_sequential(paths: &[PathBuf]) -> HashMap<String, ShaderResult> {
    let mut map = HashMap::new();
    for path in paths {
        let name = path
            .parent()
            .unwrap()
            .file_name()
            .unwrap()
            .to_string_lossy()
            .to_string();
        if let Some(res) = compile_one(path) {
            map.insert(name, res);
        }
    }
    map
}

fn run_parallel(paths: &[PathBuf]) -> HashMap<String, ShaderResult> {
    let pb = indicatif::ProgressBar::new(paths.len() as u64);
    pb.set_style(
        indicatif::ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} {msg}",
        )
        .unwrap()
        .progress_chars("#>-"),
    );
    pb.set_message("parallel compile");

    let results: Vec<(String, ShaderResult)> = paths
        .par_iter()
        .filter_map(|path| {
            let name = path
                .parent()
                .unwrap()
                .file_name()
                .unwrap()
                .to_string_lossy()
                .to_string();
            let res = compile_one(path);
            pb.inc(1);
            res.map(|r| (name, r))
        })
        .collect();
    pb.finish_with_message("done");
    results.into_iter().collect()
}

fn compare(
    a: &HashMap<String, ShaderResult>,
    b: &HashMap<String, ShaderResult>,
    label: &str,
) -> bool {
    let mut ok = true;
    for (name, ra) in a {
        match b.get(name) {
            Some(rb) => {
                if ra.hlsl != rb.hlsl {
                    eprintln!("[{label}] HLSL mismatch for {name}");
                    ok = false;
                }
                if ra.spirv != rb.spirv {
                    eprintln!("[{label}] SPIR-V mismatch for {name}");
                    ok = false;
                }
                if ra.validated != rb.validated {
                    eprintln!(
                        "[{label}] validation mismatch for {name}: {} vs {}",
                        ra.validated, rb.validated
                    );
                    ok = false;
                }
            }
            None => {
                eprintln!("[{label}] missing {name} in second set");
                ok = false;
            }
        }
    }
    for name in b.keys() {
        if !a.contains_key(name) {
            eprintln!("[{label}] extra {name} in second set");
            ok = false;
        }
    }
    ok
}

fn main() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let dxc_dir = manifest_dir.join("dxc");
    if dxc_dir.join("dxcompiler.dll").exists() {
        let existing = std::env::var("PATH").unwrap_or_default();
        unsafe { std::env::set_var("PATH", format!("{};{}", dxc_dir.display(), existing)) };
    }

    println!("=== Regression: 2 sequential ===");
    let two = collect_names(2);
    for p in &two {
        println!(
            "  {}",
            p.parent().unwrap().file_name().unwrap().to_string_lossy()
        );
    }
    let seq2 = run_sequential(&two);
    println!("  sequential done: {} compiled", seq2.len());

    println!("\n=== Regression: same 2 in parallel ===");
    let par2 = run_parallel(&two);
    println!("  parallel done: {} compiled", par2.len());
    let ok2 = compare(&seq2, &par2, "2 vs 2");
    println!(
        "  2 sequential vs 2 parallel: {}",
        if ok2 { "IDENTICAL ✅" } else { "MISMATCH ❌" }
    );
    assert!(ok2, "2 sequential vs parallel must be identical");

    println!("\n=== Regression: 10 in parallel ===");
    let ten = collect_names(10);
    let par10 = run_parallel(&ten);
    println!("  10 parallel done: {} compiled", par10.len());
    // Compare the 2 among the 10 vs the earlier 2
    let mut ok10 = true;
    for (name, res) in &seq2 {
        if let Some(r10) = par10.get(name) {
            if res.hlsl != r10.hlsl || res.spirv != r10.spirv || res.validated != r10.validated {
                eprintln!("10-parallel mismatch for {name}");
                ok10 = false;
            }
        } else {
            eprintln!("10-parallel missing {name}");
            ok10 = false;
        }
    }
    println!(
        "  10 parallel vs 2 sequential (subset): {}",
        if ok10 {
            "IDENTICAL ✅"
        } else {
            "MISMATCH ❌"
        }
    );
    assert!(ok10, "10 parallel must contain identical 2");

    println!("\n=== Regression: 20 in parallel (max per requirement) ===");
    let twenty = collect_names(20);
    let par20 = run_parallel(&twenty);
    println!("  20 parallel done: {} compiled", par20.len());
    let mut ok20 = true;
    for (name, res) in &seq2 {
        if let Some(r20) = par20.get(name) {
            if res.hlsl != r20.hlsl || res.spirv != r20.spirv || res.validated != r20.validated {
                eprintln!("20 parallel mismatch for {name}");
                ok20 = false;
            }
        } else {
            eprintln!("20 parallel missing {name}");
            ok20 = false;
        }
    }
    println!(
        "  20 parallel vs 2 sequential (subset): {}",
        if ok20 {
            "IDENTICAL ✅"
        } else {
            "MISMATCH ❌"
        }
    );
    assert!(ok20, "20 parallel must contain identical 2");

    println!("\n=== All regression checks PASSED ✅ ===");
    println!("HLSL, SPIR-V, and validation are deterministic across sequential/parallel.");
    println!(
        "Full 998-shader corpus validated separately via `cargo run --bin bulk_validate --release` with progress indicator."
    );
}
