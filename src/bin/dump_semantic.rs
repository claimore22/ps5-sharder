//! Batch-dump our semantic snapshot per test shader.
//!
//! For each `tests/shaders/*/shader.pssl`, runs the same front-end as
//! bulk_validate (preprocess -> parse -> analyze -> lower) and writes
//! `tests/shaders/*/analysis/semantic.json`: stage, entry, threads,
//! inputs, output, resources, globals, user structs, or an error.
//!
//! This is the permanent baseline of what our semantic layer believes
//! about every shader. Diff it after parser/semantic changes to detect
//! regressions; cross-check it against the `.ags` reference dumps
//! (`check_ags_refs.py`) to find bugs the binary stage counters hide.
use std::path::{Path, PathBuf};

use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use serde_json::{json, Value};

use ps5_shader::ir::{IrType, ResourceType};

fn json_type(ty: &IrType) -> String {
    match ty {
        IrType::Void => "void".to_string(),
        IrType::Bool => "bool".to_string(),
        IrType::Int => "int".to_string(),
        IrType::UInt => "uint".to_string(),
        IrType::Float => "float".to_string(),
        IrType::Half => "half".to_string(),
        IrType::Double => "double".to_string(),

        IrType::Vector { base, width } => {
            format!("{}{width}", base.name())
        }

        IrType::Matrix {
            base,
            rows,
            columns,
        } => {
            format!("{}{rows}x{columns}", base.name())
        }

        IrType::Named(n) => n.clone(),

        IrType::Resource { kind, element, .. } => {
            format!("{}<{}>", kind, json_type(element))
        }
    }
}

fn resource_class(t: ResourceType) -> &'static str {
    match t {
        ResourceType::Texture => "texture",
        ResourceType::Sampler => "sampler",
        ResourceType::ConstantBuffer => "cbuffer",
        ResourceType::StructuredBuffer => "buffer",
        ResourceType::RWBuffer => "rwbuffer",
        ResourceType::RWTexture => "rwtexture",
    }
}

fn main() {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let dir = manifest.join("tests").join("shaders");

    let mut names = Vec::new();

    if let Ok(rd) = std::fs::read_dir(&dir) {
        for entry in rd.flatten() {
            if entry.path().is_dir() {
                if let Some(name) = entry.file_name().to_str() {
                    names.push(name.to_string());
                }
            }
        }
    }

    names.sort();

    let pb = ProgressBar::new(names.len() as u64);

    pb.set_style(
        ProgressStyle::with_template(
            "[{elapsed_precise}] {bar:40.cyan/blue} \
             {pos}/{len} ({percent}%) {msg}",
        )
        .unwrap()
        .progress_chars("##-"),
    );

    let results: Vec<bool> = names
        .par_iter()
        .map(|name| {
            let path = dir.join(name).join("shader.pssl");

            if !path.exists() {
                pb.inc(1);
                return true;
            }

            let (value, ir) = analyze_one(name, &path);

            let success = value.get("error").is_none();

            let out_dir = dir.join(name).join("analysis");

            if std::fs::create_dir_all(&out_dir).is_ok() {
                if let Ok(text) = serde_json::to_string(&value) {
                    let _ = std::fs::write(
                        out_dir.join("semantic.json"),
                        text,
                    );
                }
                // Differential corpus: our IR snapshot beside the semantic
                // one (PSSL -> semantic.json -> ir.json -> HLSL -> DXC vs
                // official .ags reference).
                if let Some(ir) = ir {
                    if let Ok(text) = serde_json::to_string(&ir) {
                        let _ = std::fs::write(out_dir.join("ir.json"), text);
                    }
                }
            }

            pb.inc(1);

            success
        })
        .collect();

    pb.finish_with_message("done");

    let ok = results.iter().filter(|&&v| v).count();
    let fail = results.len() - ok;

    println!(
        "wrote analysis/semantic.json: ok={ok} error={fail}"
    );
}

fn analyze_one(name: &str, path: &std::path::Path) -> (Value, Option<Value>) {
    let raw = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => return (json!({"name": name, "error": format!("read: {e}")}), None),
    };
    // Same front-end as bulk_validate: preprocess (resolves headers,
    // strips BOM) then parse.
    let pre = match ps5_shader::preprocess::preprocess(&raw, Some(path)) {
        Ok(s) => s,
        Err(e) => {
            let msg = e.to_string();
            let first = msg.lines().next().unwrap_or(&msg);
            return (json!({"name": name, "error": format!("preprocess: {first}")}), None);
        }
    };
    let module = match ps5_shader::parse_shader(&pre) {
        Ok(m) => m,
        Err(e) => {
            let msg = e.to_string();
            let first = msg.lines().next().unwrap_or(&msg);
            return (json!({"name": name, "error": format!("parse: {first}")}), None);
        }
    };
    let model = match ps5_shader::semantic::analyze(&module) {
        Ok(m) => m,
        Err(e) => {
            let msg = e.to_string();
            let first = msg.lines().next().unwrap_or(&msg);
            return (json!({"name": name, "error": format!("semantic: {first}")}), None);
        }
    };
    let stage: Option<&str> = model.stage.map(|s| s.as_str());
    let entry: Option<&str> = model
        .entry_point
        .as_ref()
        .map(|e| e.function.as_str());

    // Entry details need lowering; failure keeps stage/entry.
    let mut threads: Option<(u32, u32, u32)> = None;
    let mut inputs = Vec::new();
    let mut output: Value = Value::Null;
    let mut resources = Vec::new();
    let mut globals = Vec::new();
    let mut structs = Vec::new();
    let mut ir_value: Option<Value> = None;
    if let Ok(ir) = ps5_shader::lower::lower_module(&module, &model) {
        if let Some(ep) = model.entry_point.as_ref() {
            if let Some(f) = ir.functions.iter().find(|f| f.name == ep.function) {
                threads = f.num_threads;
                for p in &f.parameters {
                    inputs.push(json!({
                        "name": p.name,
                        "semantic": p.semantic,
                        "type": json_type(&p.ty),
                    }));
                }
                output = match f.return_semantic.as_ref() {
                    Some(rs) => json!({"type": json_type(&f.return_type), "semantic": rs}),
                    None => json!({"type": json_type(&f.return_type)}),
                };
            }
        }
        for r in &ir.resources {
            resources.push(json!({
                "name": r.name,
                "class": resource_class(r.resource_type),
                "element": r.element_type.as_ref().map(json_type),
                "binding": r.binding,
                "register": r.register,
                "space": r.space,
            }));
        }
        for g in &ir.globals {
            globals.push(json!({"name": g.name, "type": json_type(&g.ty)}));
        }
        for s in &ir.structs {
            let fields: Vec<Value> = s
                .fields
                .iter()
                .map(|f| {
                    json!({
                        "name": f.name,
                        "type": json_type(&f.ty),
                        "semantic": f.semantic,
                    })
                })
                .collect();
            structs.push(json!({"name": s.name, "fields": fields}));
        }
        ir_value = serde_json::to_value(&ir).ok();
    }
    (
        json!({
            "name": name,
            "stage": stage,
            "entry": entry,
            "threads": threads.map(|(x, y, z)| vec![x, y, z]),
            "inputs": inputs,
            "output": output,
            "resources": resources,
            "globals": globals,
            "structs": structs,
        }),
        ir_value,
    )
}
