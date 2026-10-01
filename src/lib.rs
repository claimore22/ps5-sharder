#![forbid(unsafe_code)]

pub mod ast;
pub mod diagnostics;
pub mod dxc;
pub mod emit;
pub mod error;
pub mod framework;
pub mod ir;
pub mod lexer;
pub mod lower;
pub mod parser;
pub mod preprocess;
pub mod psfx;
pub mod semantic;
pub mod spirv;
pub mod token;
pub mod tracing;

pub use ast::Module;
pub use error::{Phase, Ps5ShaderError, Result, Span};
pub use parser::parse_shader;
