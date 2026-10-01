//! Compiles a raw HLSL file with DXC and reports Ok/Err.
//! Debug tool for bisecting DXC native crashes:
//! `dxc_probe <file.hlsl> <entry> <vs|ps|cs>`.
use std::path::PathBuf;

fn main() {
    ps5_shader::tracing::init_tracing();
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let dxc_dir = manifest.join("dxc");
    if dxc_dir.join("dxcompiler.dll").exists() {
        let existing = std::env::var("PATH").unwrap_or_default();
        unsafe {
            std::env::set_var("PATH", format!("{};{}", dxc_dir.display(), existing));
        }
    }
    let args: Vec<String> = std::env::args().collect();
    let hlsl = std::fs::read_to_string(&args[1]).unwrap();
    let entry = args[2].clone();
    let stage = match args[3].as_str() {
        "vs" => ps5_shader::ir::Stage::Vertex,
        "ps" => ps5_shader::ir::Stage::Fragment,
        _ => ps5_shader::ir::Stage::Compute,
    };
    match ps5_shader::dxc::compile_hlsl(&hlsl, &entry, stage) {
        Ok(b) => println!("DXC-OK {} bytes", b.len()),
        Err(e) => {
            let msg = e.to_string();
            let first = msg.lines().next().unwrap_or(&msg);
            println!("DXC-ERR {first}");
        }
    }
}
