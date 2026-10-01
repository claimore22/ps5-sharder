use std::path::Path;

fn main() {
    let name = std::env::args().nth(1).unwrap();
    let shaders_root = Path::new("tests/shaders");
    let dir = shaders_root.join(&name);
    let path = dir.join("shader.pssl");
    let raw = std::fs::read_to_string(&path).unwrap();
    let pre = ps5_shader::preprocess::preprocess(&raw, Some(&path)).unwrap();
    let m = ps5_shader::parse_shader(&pre).unwrap();
    println!("structs: {}", m.structs.len());
    for s in &m.structs {
        println!("  struct {}", s.name);
    }
    println!("functions: {}", m.functions.len());
    for f in &m.functions {
        println!("  fn {} (scope {:?}) ret {:?}", f.name, f.scope, f.return_type);
    }
    println!("globals: {}", m.globals.len());
    for g in &m.globals {
        println!("  global {} : {:?}", g.name, g.ty);
    }
    println!("resources: {}", m.resources.len());
    for r in &m.resources {
        println!("  resource {}", r.name);
    }
}