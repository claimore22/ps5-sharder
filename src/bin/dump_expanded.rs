//! Prints the full preprocessed PSSL source for a test shader.
//!
//! Unlike `dump_preproc` (needle grep), this emits everything, so lexer
//! spans from parse errors can be located byte-exactly:
//! `dump_expanded <shader-dir-name> > expanded.pssl`.
use std::path::Path;

fn main() {
    let name = std::env::args().nth(1).unwrap();
    let path = Path::new("tests/shaders").join(&name).join("shader.pssl");
    let raw = std::fs::read_to_string(&path).unwrap();
    let pre = ps5_shader::preprocess::preprocess(&raw, Some(&path)).unwrap();
    print!("{pre}");
}
