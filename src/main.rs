use std::env;
use std::path::PathBuf;
use std::process::ExitCode;

use ps5_shader::error::{Phase, Ps5ShaderError, Result};
use ps5_shader::tracing::init_tracing;
use ps5_shader::{
    Module, diagnostics::SourceMap, emit::emit_vulkan_hlsl, lower::lower_module, parse_shader,
    semantic, spirv,
};

const USAGE: &str = "\
ps5-shader - PSSL to Vulkan shader compiler

Usage: ps5-shader compile <shader.pssl> [options]

Options:
  --emit-hlsl          Print the generated Vulkan HLSL.
  --emit-spirv <file>  Write the SPIR-V binary to <file>.
  --validate           Validate the SPIR-V with SPIRV-Tools (Vulkan 1.3).
  --dump-ast           Print the parsed AST (Debug form).
  --dump-ir            Print the Shader IR (Debug form).
  --dump-spirv         Inspect the SPIR-V structurally via rspirv.
  -h, --help           Print this help.";

struct Options {
    input: PathBuf,
    emit_hlsl: bool,
    emit_spirv: Option<PathBuf>,
    validate: bool,
    dump_ast: bool,
    dump_ir: bool,
    dump_spirv: bool,
}

fn main() -> ExitCode {
    init_tracing();

    let args: Vec<String> = env::args().skip(1).collect();

    if args.is_empty() || args.iter().any(|a| a == "-h" || a == "--help") {
        println!("{USAGE}");
        return ExitCode::SUCCESS;
    }

    let options = match parse_args(&args) {
        Ok(options) => options,
        Err(error) => {
            eprintln!("error: {error}");
            eprintln!("{USAGE}");
            return ExitCode::from(2);
        }
    };

    match run(options) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}

fn parse_args(args: &[String]) -> std::result::Result<Options, String> {
    // Support both `compile <file>` and a bare `<file>` (defaulting to compile).
    let mut index = 0usize;
    if args.first().map(String::as_str) == Some("compile") {
        index = 1;
    }

    let mut input: Option<PathBuf> = None;
    let mut emit_hlsl = false;
    let mut emit_spirv = None;
    let mut validate = false;
    let mut dump_ast = false;
    let mut dump_ir = false;
    let mut dump_spirv = false;

    while index < args.len() {
        let arg = &args[index];
        match arg.as_str() {
            "--emit-hlsl" => emit_hlsl = true,
            "--emit-spirv" => {
                index += 1;
                let value = args.get(index).ok_or("--emit-spirv requires a file path")?;
                emit_spirv = Some(PathBuf::from(value));
            }
            "--validate" => validate = true,
            "--dump-ast" => dump_ast = true,
            "--dump-ir" => dump_ir = true,
            "--dump-spirv" => dump_spirv = true,
            _ if arg.starts_with('-') => return Err(format!("unknown option `{arg}`")),
            _ => {
                if input.is_some() {
                    return Err("multiple input files given".to_string());
                }
                input = Some(PathBuf::from(arg));
            }
        }
        index += 1;
    }

    let input = input.ok_or("no input file given")?;

    Ok(Options {
        input,
        emit_hlsl,
        emit_spirv,
        validate,
        dump_ast,
        dump_ir,
        dump_spirv,
    })
}

fn run(options: Options) -> Result<()> {
    let raw_source = std::fs::read_to_string(&options.input).map_err(|error| {
        Ps5ShaderError::with_source(
            format!("failed to read {}", options.input.display()),
            Phase::Io,
            error,
        )
    })?;
    let report = |phase: &str, status: &str| {
        tracing::info!(target: "pssl_shader::validate", phase, status, "pipeline stage");
        println!("{phase:.<24} {status}");
    };

    // 0. Preprocess (BOM, #include, #define)
    let source = match ps5_shader::preprocess::preprocess(&raw_source, Some(&options.input)) {
        Ok(source) => {
            report("Preprocessing", "OK");
            source
        }
        Err(mut error) => {
            error.file = Some(options.input.clone());
            return Err(error);
        }
    };
    let source_map = SourceMap::new(&source);

    // 1. Parse
    let module: Module = match parse_shader(&source) {
        Ok(module) => {
            report("Parsing", "OK");
            module
        }
        Err(mut error) => {
            error.file = Some(options.input.clone());
            attach_position(&mut error, &source_map);
            return Err(error);
        }
    };

    if options.dump_ast {
        println!("\n== AST ==");
        println!("{module:#?}");
    }

    // 2. Semantic analysis
    let model = match semantic::analyze(&module) {
        Ok(model) => {
            report("Semantic analysis", "OK");
            model
        }
        Err(mut error) => {
            error.file = Some(options.input.clone());
            attach_position(&mut error, &source_map);
            return Err(error);
        }
    };

    // 3. Lower to Shader IR
    let ir = match lower_module(&module, &model) {
        Ok(ir) => {
            report("Shader IR", "OK");
            ir
        }
        Err(mut error) => {
            error.file = Some(options.input.clone());
            attach_position(&mut error, &source_map);
            return Err(error);
        }
    };

    if options.dump_ir {
        println!("\n== IR ==");
        println!("{ir:#?}");
    }

    // 4. Emit Vulkan HLSL
    let hlsl = emit_vulkan_hlsl(&ir);
    report("HLSL emission", "OK");

    if options.emit_hlsl {
        println!("\n== Generated HLSL ==");
        print!("{hlsl}");
    }

    // Stage and entry point are required for DXC.
    let stage = model.stage.ok_or_else(|| {
        Ps5ShaderError::new(
            "cannot compile: no shader stage could be determined",
            Phase::DxcCompile,
        )
    })?;
    let entry = model
        .entry_point
        .as_ref()
        .map(|ep| ep.function.clone())
        .ok_or_else(|| {
            Ps5ShaderError::new(
                "cannot compile: no entry point identified",
                Phase::DxcCompile,
            )
        })?;

    // 5. DXC compile to SPIR-V
    let spirv_bytes = match ps5_shader::dxc::compile_hlsl(&hlsl, &entry, stage) {
        Ok(bytes) => {
            report("DXC compilation", "OK");
            bytes
        }
        Err(mut error) => {
            error.file = Some(options.input.clone());
            attach_position(&mut error, &source_map);
            return Err(error);
        }
    };
    let spirv_words = spirv::bytes_to_words(&spirv_bytes)?;

    if let Some(path) = &options.emit_spirv {
        std::fs::write(path, &spirv_bytes).map_err(|error| {
            Ps5ShaderError::with_source(
                format!("failed to write {}", path.display()),
                Phase::Io,
                error,
            )
        })?;
        println!("SPIR-V written to {}", path.display());
    }

    // 6. SPIRV-Tools validation (the authoritative validator)
    if options.validate {
        match spirv::validate_spirv(&spirv_words) {
            Ok(()) => {
                report("SPIR-V validation", "OK");
            }
            Err(error) => return Err(error),
        }
    }

    // 7. rspirv inspection (debugging/analysis only)
    if options.dump_spirv {
        match spirv::inspect_bytes(&spirv_bytes) {
            Ok(inspection) => {
                println!("\n== SPIR-V Inspection (rspirv) ==");
                print!("{}", spirv::dump_inspection(&inspection));
            }
            Err(error) => {
                eprintln!("warning: SPIR-V inspection failed: {error}");
            }
        }
    }

    Ok(())
}

fn attach_position(error: &mut Ps5ShaderError, source_map: &SourceMap) {
    if let Some(span) = error.span {
        let (line, column) = source_map.position(span.start);
        error.line = Some(line);
        error.column = Some(column);
    }
}
