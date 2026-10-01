//! One-shot inventory of the `sce` DXC wall: for every shader whose
//! validated.md Notes mention `undeclared identifier 'sce'`, run
//! preprocess -> parse -> analyze -> lower -> emit and list the `sce::`
//! symbols surviving into the emitted HLSL with their use-site lines.
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let md = std::fs::read_to_string(manifest_dir.join("shader-DXC-validated.md")).unwrap();
    let mut names = Vec::new();
    // Join wrapped table rows: continuation lines don't start with '|'.
    let joined = md.replace("\r\n", "\n");
    let mut rows: Vec<String> = Vec::new();
    for line in joined.lines() {
        if line.starts_with('|') {
            rows.push(line.to_string());
        } else if let Some(last) = rows.last_mut() {
            last.push_str(line);
        }
    }
    for row in &rows {
        let cells: Vec<&str> = row.split('|').map(|s| s.trim()).collect();
        if cells.len() < 9 {
            continue;
        }
        let name = cells[1];
        let spirv = cells[7];
        let notes = cells[8];
        if name.is_empty() || name == "Shader" || spirv == "✅" {
            continue;
        }
        if notes.contains("undeclared identifier 'sce'") {
            names.push(name.to_string());
        }
    }
    eprintln!("sce-blocked shaders: {}", names.len());

    // symbol -> (shaders, example lines), plus shader -> symbols for
    // minimal-flip-set analysis.
    let mut symbols: BTreeMap<String, (BTreeSet<String>, Vec<String>)> = BTreeMap::new();
    let mut per_shader: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut pipeline_fail = 0usize;
    let mut no_sce_in_hlsl = Vec::new();
    for name in &names {
        let path = manifest_dir.join("tests/shaders").join(name).join("shader.pssl");
        let raw = match std::fs::read_to_string(&path) {
            Ok(s) => s,
            Err(_) => {
                pipeline_fail += 1;
                continue;
            }
        };
        let pre = match ps5_shader::preprocess::preprocess(&raw, Some(&path)) {
            Ok(s) => s,
            Err(_) => {
                pipeline_fail += 1;
                continue;
            }
        };
        let m = match ps5_shader::parse_shader(&pre) {
            Ok(m) => m,
            Err(_) => {
                pipeline_fail += 1;
                continue;
            }
        };
        let model = match ps5_shader::semantic::analyze(&m) {
            Ok(model) => model,
            Err(_) => {
                pipeline_fail += 1;
                continue;
            }
        };
        let ir = match ps5_shader::lower::lower_module(&m, &model) {
            Ok(ir) => ir,
            Err(_) => {
                pipeline_fail += 1;
                continue;
            }
        };
        let hlsl = ps5_shader::emit::emit_vulkan_hlsl(&ir);
        let mut found = false;
        for line in hlsl.lines() {
            if !line.contains("sce::") {
                continue;
            }
            found = true;
            for sym in extract_symbols(line) {
                let e = symbols.entry(sym.clone()).or_insert_with(|| (BTreeSet::new(), Vec::new()));
                e.0.insert(name.clone());
                per_shader.entry(name.clone()).or_default().insert(sym);
                if e.1.len() < 3 {
                    e.1.push(format!("{name}: {}", line.trim()));
                }
            }
        }
        if !found {
            no_sce_in_hlsl.push(name.clone());
        }
    }

    let mut ranked: Vec<_> = symbols.into_iter().collect();
    ranked.sort_by(|a, b| b.1 .0.len().cmp(&a.1 .0.len()));
    for (sym, (shaders, examples)) in &ranked {
        println!("{:>5}  {}", shaders.len(), sym);
        for ex in examples {
            println!("          e.g. {ex}");
        }
    }
    println!("=== shaders with NO sce:: in HLSL: {} ===", no_sce_in_hlsl.len());
    for n in &no_sce_in_hlsl {
        println!("  {n}");
    }
    eprintln!("pipeline failures (pre-emit): {pipeline_fail}");
    // Per-shader symbol sets for minimal-flip-set analysis. Family key:
    // collapse to top-level `sce::A(::B)?` prefixes.
    let mut csv = String::from("shader,ns_psr,ns_gnm_sampler,ns_agc_texbuf,ns_agc_sampler,ns_other\n");
    for (name, syms) in &per_shader {
        let mut psr = false;
        let mut gnm_smp = false;
        let mut agc_tb = false;
        let mut agc_smp = false;
        let mut other: BTreeSet<String> = BTreeSet::new();
        for s in syms {
            if s.starts_with("sce::Psr::") {
                psr = true;
            } else if s.starts_with("sce::Gnm::Sampler") || s.starts_with("sce::Gnm::k") {
                gnm_smp = true;
            } else if s.starts_with("sce::Agc::Core::Texture")
                || s.starts_with("sce::Agc::Core::Buffer")
                || s.starts_with("sce::FrToolkit::")
            {
                agc_tb = true;
            } else if s.starts_with("sce::Agc::Core::Sampler") {
                agc_smp = true;
            } else {
                other.insert(s.clone());
            }
        }
        let bit = |b: bool| if b { "1" } else { "0" };
        csv.push_str(&format!(
            "{},{},{},{},{},\"{}\"\n",
            name,
            bit(psr),
            bit(gnm_smp),
            bit(agc_tb),
            bit(agc_smp),
            other.into_iter().collect::<Vec<_>>().join(";")
        ));
    }
    std::fs::write(manifest_dir.join("other_references/sce_per_shader.csv"), csv).unwrap();
}

/// Extracts `sce::...` symbol tokens from an HLSL line: from `sce::`
/// through alphanumerics, `_`, `:`, `<`, `>`.
fn extract_symbols(line: &str) -> Vec<String> {
    let mut out = Vec::new();
    let bytes = line.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if line[i..].starts_with("sce::") {
            let mut j = i + 5;
            while j < bytes.len() {
                let c = bytes[j] as char;
                if c.is_ascii_alphanumeric() || c == '_' || c == ':' || c == '<' || c == '>' {
                    j += 1;
                } else {
                    break;
                }
            }
            // Trim trailing ':' from `sce::X::` fragments.
            let mut sym = line[i..j].trim_end_matches(':').to_string();
            if sym.len() > 5 {
                // Normalize `sce::Agc::Core::Sampler::member` call chains to
                // the type prefix when a fourth segment is a lowercase
                // member/call rather than a type.
                out.push(sym.clone());
                let _ = &mut sym;
            }
            i = j.max(i + 1);
        } else {
            i += 1;
        }
    }
    out.sort();
    out.dedup();
    out
}
