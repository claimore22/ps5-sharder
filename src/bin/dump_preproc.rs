use std::path::Path;

fn main() {
    let name = std::env::args().nth(1).unwrap();
    let shaders_root = Path::new("tests/shaders");
    let dir = shaders_root.join(&name);
    let path = dir.join("shader.pssl");
    let raw = std::fs::read_to_string(&path).unwrap();
    let pre = ps5_shader::preprocess::preprocess(&raw, Some(&path)).unwrap();
    let needle = std::env::args().nth(2).unwrap();
    for (i, line) in pre.lines().enumerate() {
        if line.contains(&needle) {
            println!("{}: {}", i + 1, line.trim());
        }
    }
}