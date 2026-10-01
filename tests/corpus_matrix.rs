//! Corpus compatibility matrix — diagnostic, not a gate.
//! Iterates every `tests/shaders/*/shader.pssl` in a local validation
//! corpus (not shipped with the repo) and runs the full pipeline
//! PSSL→parse→semantic→IR→HLSL→DXC→SPIR-V→validate, recording the first
//! failing phase. Failures are expected and valuable. Skips quietly when
//! the corpus directory is absent.

use std::path::PathBuf;

fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

#[test]
fn corpus_compatibility_matrix() {
    // Ensure DXC is found.
    let dxc_dir = manifest_dir().join("dxc");
    if dxc_dir.join("dxcompiler.dll").exists() {
        let existing = std::env::var("PATH").unwrap_or_default();
        unsafe { std::env::set_var("PATH", format!("{};{}", dxc_dir.display(), existing)) };
    }

    let shaders_root = manifest_dir().join("tests/shaders");
    let mut entries: Vec<PathBuf> = Vec::new();
    let Ok(rd) = std::fs::read_dir(&shaders_root) else {
        println!("corpus matrix skipped: {} not present", shaders_root.display());
        return;
    };
    for entry in rd {
        let entry = entry.unwrap();
        let dir = entry.path();
        if dir.is_dir() && dir.join("shader.pssl").exists() {
            entries.push(dir);
        }
    }
    entries.sort();

    println!("\n| Shader | Parse | Semantic | IR | HLSL | DXC | SPIR-V | Notes |");
    println!("|---|---|---|---|---|---|---|---|");

    for dir in entries {
        let name = dir.file_name().unwrap().to_string_lossy().to_string();
        let path = dir.join("shader.pssl");
        let raw = std::fs::read_to_string(&path).unwrap_or_else(|e| format!("READ_ERROR: {e}"));
        // Preprocess (BOM, #include, #define) — controlled resolver, depth-limited
        let source = match ps5_shader::preprocess::preprocess(&raw, Some(&path)) {
            Ok(s) => s,
            Err(e) => {
                // Treat preprocess failure as parse-blocking
                let msg = e.to_string();
                let first = msg.lines().next().unwrap_or(&msg);
                println!(
                    "| {name} | ❌ | — | — | — | — | — | #include/define | preprocess: {} |",
                    first[..80.min(first.len())].replace('|', "/")
                );
                continue;
            }
        };

        let mut notes = String::new();
        // Quick feature hints
        if source.contains("#include") {
            notes.push_str("#include ");
        }
        if source.contains("Texture2D") {
            notes.push_str("Texture2D ");
        }
        if source.contains("SamplerState") {
            notes.push_str("SamplerState ");
        }
        if source.contains("ConstantBuffer") {
            notes.push_str("ConstantBuffer ");
        }
        if source.contains("RW_") {
            notes.push_str("RW_ ");
        }
        if source.contains("cbuffer") {
            notes.push_str("cbuffer ");
        }
        if source.contains("NUM_THREADS") {
            notes.push_str("NUM_THREADS ");
        }
        if source.contains("S_SRT_DATA") {
            notes.push_str("S_SRT_DATA ");
        }
        if source.contains("half") {
            notes.push_str("half ");
        }
        if source.contains("column_major") {
            notes.push_str("column_major ");
        }
        if source.contains("->") {
            notes.push_str("-> ");
        }

        let mut parse = "—";
        let mut semantic = "—";
        let mut ir = "—";
        let mut hlsl = "—";
        let mut dxc = "—";
        let mut spirv = "—";
        let mut fail_phase = "";

        // --- parse ---
        let module = match ps5_shader::parse_shader(&source) {
            Ok(m) => {
                parse = "✅";
                Some(m)
            }
            Err(e) => {
                parse = "❌";
                fail_phase = "parse";
                // capture first 80 chars of error
                let msg = e.to_string();
                notes.push_str(&format!(
                    " | parse: {}",
                    msg.lines().next().unwrap_or(&msg)[..80.min(msg.len())].replace('|', "/")
                ));
                None
            }
        };

        let (model, ir_module) = if let Some(module) = module {
            // --- semantic ---
            match ps5_shader::semantic::analyze(&module) {
                Ok(m) => {
                    semantic = "✅";
                    // --- lower ---
                    match ps5_shader::lower::lower_module(&module, &m) {
                        Ok(ir_mod) => {
                            ir = "✅";
                            (Some(m), Some(ir_mod))
                        }
                        Err(e) => {
                            ir = "❌";
                            if fail_phase.is_empty() {
                                fail_phase = "IR";
                                notes.push_str(&format!(
                                    " | IR: {}",
                                    e.to_string().lines().next().unwrap_or("")
                                        [..80.min(e.to_string().len())]
                                        .replace('|', "/")
                                ));
                            }
                            (Some(m), None)
                        }
                    }
                }
                Err(e) => {
                    semantic = "❌";
                    if fail_phase.is_empty() {
                        fail_phase = "semantic";
                        notes.push_str(&format!(
                            " | semantic: {}",
                            e.to_string().lines().next().unwrap_or("")
                                [..80.min(e.to_string().len())]
                                .replace('|', "/")
                        ));
                    }
                    (None, None)
                }
            }
        } else {
            (None, None)
        };

        if let (Some(model), Some(ir_module)) = (model, ir_module) {
            let hlsl_str = ps5_shader::emit::emit_vulkan_hlsl(&ir_module);
            hlsl = "✅";
            let stage = model.stage;
            let entry = model.entry_point.as_ref().map(|ep| ep.function.clone());
            match (stage, entry) {
                (Some(stage), Some(entry)) => {
                    match ps5_shader::dxc::compile_hlsl(&hlsl_str, &entry, stage) {
                        Ok(bytes) => {
                            dxc = "✅";
                            let words = ps5_shader::spirv::bytes_to_words(&bytes).unwrap();
                            match ps5_shader::spirv::validate_spirv(&words) {
                                Ok(()) => spirv = "✅",
                                Err(e) => {
                                    spirv = "❌";
                                    if fail_phase.is_empty() {
                                        fail_phase = "SPIR-V";
                                        notes.push_str(&format!(
                                            " | validate: {}",
                                            e.to_string().lines().next().unwrap_or("")
                                                [..80.min(e.to_string().len())]
                                                .replace('|', "/")
                                        ));
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            dxc = "❌";
                            if fail_phase.is_empty() {
                                fail_phase = "DXC";
                                let msg = e.to_string();
                                let first = msg.lines().next().unwrap_or(&msg);
                                notes.push_str(&format!(
                                    " | DXC: {}",
                                    first[..80.min(first.len())].replace('|', "/")
                                ));
                            }
                        }
                    }
                }
                _ => {
                    dxc = "❌";
                    if fail_phase.is_empty() {
                        fail_phase = "DXC";
                        notes.push_str(" | no stage/entry");
                    }
                }
            }
        }

        let _ = fail_phase;
        println!("| {name} | {parse} | {semantic} | {ir} | {hlsl} | {dxc} | {spirv} | {notes} |");
    }
    println!();
}
