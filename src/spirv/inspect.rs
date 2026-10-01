//! SPIR-V inspection via rspirv.
//!
//! This is a debugging/analysis layer only. It is NOT part of the core
//! validation path and must never be used to determine whether a SPIR-V
//! binary is valid — SPIRV-Tools (see [`super::validate`]) is the validator.

use rspirv::dr;
use rspirv::spirv::{Op, Word};

use crate::error::{Phase, Ps5ShaderError, Result};

/// A summary of the structural facts extracted from a SPIR-V module.
#[derive(Debug, Clone, Default)]
pub struct SpirvInspection {
    pub version: Option<(u8, u8)>,
    pub generator: Option<u32>,
    pub bound: Option<u32>,
    pub capabilities: Vec<String>,
    pub entry_points: Vec<EntryPointInfo>,
    pub descriptor_sets: Vec<Binding>,
    pub bindings: Vec<Binding>,
    pub locations: Vec<Location>,
    pub builtins: Vec<BuiltIn>,
    pub inputs: Vec<IoVar>,
    pub outputs: Vec<IoVar>,
    pub decorations: Vec<DecorationInfo>,
    pub types: Vec<String>,
    pub instruction_count: usize,
}

#[derive(Debug, Clone, Default)]
pub struct EntryPointInfo {
    pub id: Word,
    pub name: String,
    pub execution_model: Option<String>,
    pub interface: Vec<Word>,
}

#[derive(Debug, Clone, Default)]
pub struct Binding {
    pub name: Option<String>,
    pub set: Option<u32>,
    pub binding: Option<u32>,
    pub id: Word,
}

#[derive(Debug, Clone, Default)]
pub struct Location {
    pub name: Option<String>,
    pub location: Option<u32>,
    pub id: Word,
}

/// A `BuiltIn` decoration: `target` variable has the given built-in (`name`).
#[derive(Debug, Clone, Default)]
pub struct BuiltIn {
    pub name: Option<String>,
    pub id: Word,
}

/// A single entry-point interface variable (input or output), annotated with
/// whichever interface info is known about it (location and/or built-in).
#[derive(Debug, Clone, Default)]
pub struct IoVar {
    pub name: Option<String>,
    pub location: Option<u32>,
    pub builtin: Option<String>,
    pub id: Word,
}

#[derive(Debug, Clone, Default)]
pub struct DecorationInfo {
    pub target: Word,
    pub kind: Option<String>,
    pub args: Vec<u32>,
    /// The built-in name when `kind` is `BuiltIn` (e.g. "Position").
    pub builtin_name: Option<String>,
}

/// Loads and inspects a SPIR-V binary (as raw bytes from DXC).
pub fn inspect_bytes(bytes: &[u8]) -> Result<SpirvInspection> {
    let module = dr::load_bytes(bytes).map_err(|state| {
        Ps5ShaderError::at(
            format!("failed to load SPIR-V: {state}"),
            Phase::Inspect,
            crate::error::Span::default(),
        )
    })?;
    Ok(inspect_module(&module))
}

/// Inspects an already-loaded [`dr::Module`] and produces a structural summary.
pub fn inspect_module(module: &dr::Module) -> SpirvInspection {
    let mut inspection = SpirvInspection::default();

    if let Some(header) = &module.header {
        inspection.version = Some(decode_version(header.version));
        inspection.generator = Some(header.generator);
        inspection.bound = Some(header.bound);
    }

    for capability in &module.capabilities {
        // OpCapability: operand[0] = Capability
        let name = capability
            .operands
            .first()
            .map(|operand| format!("{operand:?}"))
            .unwrap_or_default();
        inspection.capabilities.push(name);
    }

    for instruction in &module.entry_points {
        inspection.entry_points.push(parse_entry_point(instruction));
    }

    // Annotations: descriptor sets, bindings, locations, and generic decorations.
    for instruction in &module.annotations {
        let Some(opcode) = opcode(instruction) else {
            continue;
        };
        match opcode {
            Op::Decorate => {
                inspection.decorations.push(parse_decorate(instruction));
            }
            Op::MemberDecorate => {
                let name = operand_str(instruction, 2).unwrap_or_default();
                let target = id_ref(instruction, 0);
                inspection.decorations.push(DecorationInfo {
                    target,
                    kind: Some(name),
                    args: literal_ints(instruction, 3),
                    builtin_name: None,
                });
            }
            _ => {}
        }
    }

    // Classify decorations into descriptors/bindings/locations/built-ins.
    // Snapshot the decoration data first so we can own (not borrow) inspection.
    let decoration_refs: Vec<(Word, Option<String>, Vec<u32>, Option<String>)> = inspection
        .decorations
        .iter()
        .map(|d| {
            (
                d.target,
                d.kind.clone(),
                d.args.clone(),
                d.builtin_name.clone(),
            )
        })
        .collect();
    for (target, kind, args, builtin_name) in decoration_refs {
        match kind.as_deref() {
            Some("DescriptorSet") => {
                let set = args.first().copied();
                attach_set(&mut inspection, target, set);
            }
            Some("Binding") => {
                let binding = args.first().copied();
                attach_binding(&mut inspection, target, binding);
            }
            Some("Location") => {
                let location = args.first().copied();
                attach_location(&mut inspection, target, location);
            }
            Some("BuiltIn") => {
                attach_builtin(&mut inspection, target, builtin_name);
            }
            _ => {}
        }
    }

    // Separate entry-point interface variables into inputs vs outputs by their
    // storage class, annotating them with whatever interface info we know.
    collect_inputs_outputs(&mut inspection, &module);

    // Types from the global type/constant/value section.
    for instruction in &module.types_global_values {
        let Some(opcode) = opcode(instruction) else {
            continue;
        };
        if is_type_opcode(opcode) {
            inspection.types.push(format_type(instruction, opcode));
        }
    }

    inspection.instruction_count = module.all_inst_iter().count();

    inspection
}

// --------------------------------------------------------------------------
// Per-instruction parsers
// --------------------------------------------------------------------------

fn opcode(instruction: &dr::Instruction) -> Option<Op> {
    Some(instruction.class.opcode)
}

/// Decodes a SPIR-V header version word into `(major, minor)`.
fn decode_version(word: Word) -> (u8, u8) {
    ((word >> 16) as u8, ((word >> 8) & 0xff) as u8)
}

fn id_ref(instruction: &dr::Instruction, index: usize) -> Word {
    match instruction.operands.get(index) {
        Some(dr::Operand::IdRef(word)) => *word,
        _ => 0,
    }
}

fn literal_int(instruction: &dr::Instruction, index: usize) -> Option<u32> {
    match instruction.operands.get(index) {
        Some(dr::Operand::LiteralBit32(value)) => Some(*value),
        _ => None,
    }
}

fn literal_ints(instruction: &dr::Instruction, start: usize) -> Vec<u32> {
    instruction
        .operands
        .iter()
        .skip(start)
        .filter_map(|operand| match operand {
            dr::Operand::LiteralBit32(value) => Some(*value),
            _ => None,
        })
        .collect()
}

fn literal_string(instruction: &dr::Instruction, index: usize) -> Option<String> {
    match instruction.operands.get(index) {
        Some(dr::Operand::LiteralString(value)) => Some(value.clone()),
        _ => None,
    }
}

fn operand_str(instruction: &dr::Instruction, index: usize) -> Option<String> {
    instruction
        .operands
        .get(index)
        .map(|operand| format!("{operand:?}"))
}

fn parse_entry_point(instruction: &dr::Instruction) -> EntryPointInfo {
    let execution_model = match instruction.operands.first() {
        Some(dr::Operand::ExecutionModel(model)) => format!("{model:?}"),
        _ => String::new(),
    };
    let id = id_ref(instruction, 1);
    let name = literal_string(instruction, 2).unwrap_or_default();
    let interface = instruction
        .operands
        .iter()
        .skip(3)
        .filter_map(|operand| match operand {
            dr::Operand::IdRef(word) => Some(*word),
            _ => None,
        })
        .collect();
    EntryPointInfo {
        id,
        name,
        execution_model: Some(execution_model),
        interface,
    }
}

fn parse_decorate(instruction: &dr::Instruction) -> DecorationInfo {
    let target = id_ref(instruction, 0);
    let kind = match instruction.operands.get(1) {
        Some(dr::Operand::Decoration(decoration)) => format!("{decoration:?}"),
        _ => String::new(),
    };
    let args = literal_ints(instruction, 2);
    // Built-in decorations carry a `BuiltIn` enum operand rather than a
    // literal integer; capture its bare name (e.g. "Position" / "FragCoord")
    // instead of "BuiltIn(Position)".
    let builtin_name = if kind == "BuiltIn" {
        match instruction.operands.get(2) {
            Some(dr::Operand::BuiltIn(value)) => Some(format!("{value:?}")),
            other => other.map(|o| format!("{o:?}")),
        }
    } else {
        None
    };
    DecorationInfo {
        target,
        kind: Some(kind),
        args,
        builtin_name,
    }
}

fn is_type_opcode(opcode: Op) -> bool {
    matches!(
        opcode,
        Op::TypeVoid
            | Op::TypeBool
            | Op::TypeInt
            | Op::TypeFloat
            | Op::TypeVector
            | Op::TypeMatrix
            | Op::TypeArray
            | Op::TypeRuntimeArray
            | Op::TypeStruct
            | Op::TypePointer
            | Op::TypeFunction
            | Op::TypeImage
            | Op::TypeSampler
            | Op::TypeSampledImage
    )
}

fn format_type(instruction: &dr::Instruction, opcode: Op) -> String {
    match opcode {
        Op::TypeInt => {
            let width = literal_int(instruction, 0).unwrap_or(0);
            let signed = literal_int(instruction, 1).unwrap_or(0);
            format!("TypeInt(width={width}, signed={signed})")
        }
        Op::TypeFloat => {
            let width = literal_int(instruction, 0).unwrap_or(0);
            format!("TypeFloat(width={width})")
        }
        Op::TypeVector => {
            let component = id_ref(instruction, 0);
            let count = literal_int(instruction, 1).unwrap_or(0);
            format!("TypeVector(component=%{component}, count={count})")
        }
        Op::TypeMatrix => {
            let column_type = id_ref(instruction, 0);
            let count = literal_int(instruction, 1).unwrap_or(0);
            format!("TypeMatrix(column_type=%{column_type}, count={count})")
        }
        Op::TypePointer => {
            let storage = instruction
                .operands
                .first()
                .map(|o| format!("{o:?}"))
                .unwrap_or_default();
            let pointee = id_ref(instruction, 1);
            format!("TypePointer(storage={storage}, pointee=%{pointee})")
        }
        Op::TypeStruct => {
            let members = instruction
                .operands
                .iter()
                .filter_map(|operand| match operand {
                    dr::Operand::IdRef(word) => Some(format!("%{word}")),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join(", ");
            format!("TypeStruct({members})")
        }
        Op::TypeArray => {
            let element = id_ref(instruction, 0);
            let length = id_ref(instruction, 1);
            format!("TypeArray(element=%{element}, length=%{length})")
        }
        _ => format!("{opcode:?}"),
    }
}

// --------------------------------------------------------------------------
// Decoration classification helpers
// --------------------------------------------------------------------------

fn attach_set(inspection: &mut SpirvInspection, id: Word, set: Option<u32>) {
    if inspection.descriptor_sets.iter().any(|b| b.id == id) {
        return;
    }
    inspection.descriptor_sets.push(Binding {
        name: None,
        set,
        binding: None,
        id,
    });
}

fn attach_binding(inspection: &mut SpirvInspection, id: Word, binding: Option<u32>) {
    if inspection.bindings.iter().any(|b| b.id == id) {
        return;
    }
    inspection.bindings.push(Binding {
        name: None,
        set: None,
        binding,
        id,
    });
}

fn attach_location(inspection: &mut SpirvInspection, id: Word, location: Option<u32>) {
    if inspection.locations.iter().any(|l| l.id == id) {
        return;
    }
    inspection.locations.push(Location {
        name: None,
        location,
        id,
    });
}

fn attach_builtin(inspection: &mut SpirvInspection, id: Word, name: Option<String>) {
    if inspection.builtins.iter().any(|b| b.id == id) {
        return;
    }
    inspection.builtins.push(BuiltIn { name, id });
}

/// Splits the entry-point interface variables into inputs and outputs based on
/// each variable's storage class, annotated with whatever interface info we
/// have gathered (location and/or built-in).
fn collect_inputs_outputs(inspection: &mut SpirvInspection, module: &dr::Module) {
    // Map variable id -> storage class string, from OpVariable instructions.
    let mut storage: Vec<(Word, String)> = Vec::new();
    for instruction in &module.types_global_values {
        if instruction.class.opcode != Op::Variable {
            continue;
        }
        let Some(id) = instruction.result_id else {
            continue;
        };
        let storage_class = instruction
            .operands
            .first()
            .map(|o| format!("{o:?}"))
            .unwrap_or_default();
        storage.push((id, storage_class));
    }

    for entry in &inspection.entry_points {
        for id in &entry.interface {
            let is_input = storage
                .iter()
                .any(|(var_id, sc)| var_id == id && sc == "StorageClass(Input)");
            let is_output = storage
                .iter()
                .any(|(var_id, sc)| var_id == id && sc == "StorageClass(Output)");
            if !is_input && !is_output {
                continue;
            }

            let location = inspection
                .locations
                .iter()
                .find(|l| l.id == *id)
                .and_then(|l| l.location);
            let builtin = inspection
                .builtins
                .iter()
                .find(|b| b.id == *id)
                .and_then(|b| b.name.clone());

            let var = IoVar {
                name: None,
                location,
                builtin,
                id: *id,
            };
            if is_input {
                inspection.inputs.push(var);
            } else {
                inspection.outputs.push(var);
            }
        }
    }
}

// --------------------------------------------------------------------------
// Text rendering
// --------------------------------------------------------------------------

/// Renders a [`SpirvInspection`] as human-readable text (for `--dump-spirv`).
pub fn dump_inspection(inspection: &SpirvInspection) -> String {
    let mut out = String::new();

    if let Some((major, minor)) = inspection.version {
        let _ = std::fmt::write(&mut out, format_args!("SPIR-V version: {major}.{minor}\n"));
    }
    if let Some(generator) = inspection.generator {
        let _ = std::fmt::write(&mut out, format_args!("generator: 0x{generator:08x}\n"));
    }
    if let Some(bound) = inspection.bound {
        let _ = std::fmt::write(&mut out, format_args!("bound: {bound}\n"));
    }

    let _ = writeln_str(
        &mut out,
        &format!("capabilities: {:?}", inspection.capabilities),
    );
    let _ = writeln_str(
        &mut out,
        &format!("instruction_count: {}", inspection.instruction_count),
    );

    let _ = writeln_str(&mut out, "\n== Entry points ==");
    for entry in &inspection.entry_points {
        let name = &entry.name;
        let id = entry.id;
        let model = entry.execution_model.as_deref().unwrap_or("unknown");
        let interface = entry
            .interface
            .iter()
            .map(|w| format!("%{w}"))
            .collect::<Vec<_>>()
            .join(", ");
        let _ = writeln_str(
            &mut out,
            &format!("  {name} id=%{id} model={model} interface=[{interface}]"),
        );
    }

    let _ = writeln_str(&mut out, "\n== Descriptor sets ==");
    for binding in &inspection.descriptor_sets {
        let _ = writeln_str(
            &mut out,
            &format!("  id=%{} set={}", binding.id, fmt_opt(binding.set)),
        );
    }

    let _ = writeln_str(&mut out, "\n== Bindings ==");
    for binding in &inspection.bindings {
        let _ = writeln_str(
            &mut out,
            &format!("  id=%{} binding={}", binding.id, fmt_opt(binding.binding)),
        );
    }

    let _ = writeln_str(&mut out, "\n== Locations ==");
    for location in &inspection.locations {
        let _ = writeln_str(
            &mut out,
            &format!(
                "  id=%{} location={}",
                location.id,
                fmt_opt(location.location)
            ),
        );
    }

    let _ = writeln_str(&mut out, "\n== Built-ins ==");
    for builtin in &inspection.builtins {
        let _ = writeln_str(
            &mut out,
            &format!(
                "  id=%{} BuiltIn={}",
                builtin.id,
                builtin.name.as_deref().unwrap_or("unknown")
            ),
        );
    }

    let _ = writeln_str(&mut out, "\n== Inputs ==");
    for input in &inspection.inputs {
        let _ = writeln_str(&mut out, &render_io(input, "input"));
    }

    let _ = writeln_str(&mut out, "\n== Outputs ==");
    for output in &inspection.outputs {
        let _ = writeln_str(&mut out, &render_io(output, "output"));
    }

    let _ = writeln_str(&mut out, "\n== Decorations ==");
    for decoration in &inspection.decorations {
        let _ = writeln_str(
            &mut out,
            &format!(
                "  target=%{} kind={} args={:?}",
                decoration.target,
                decoration.kind.as_deref().unwrap_or("unknown"),
                decoration.args
            ),
        );
    }

    let _ = writeln_str(&mut out, "\n== Types ==");
    for ty in &inspection.types {
        let _ = writeln_str(&mut out, &format!("  {ty}"));
    }

    out
}

fn fmt_opt(value: Option<u32>) -> String {
    value
        .map(|v| v.to_string())
        .unwrap_or_else(|| "-".to_string())
}

/// Renders a single input/output variable line, e.g.
/// `  id=%3 location=0` or `  id=%2 BuiltIn=FragCoord`.
fn render_io(var: &IoVar, kind: &str) -> String {
    let mut parts: Vec<String> = Vec::new();
    match var.location {
        Some(location) => {
            parts.push(format!("location={location}"));
        }
        None => {}
    }
    match &var.builtin {
        Some(name) => {
            parts.push(format!("BuiltIn={name}"));
        }
        None => {}
    }
    let annotation = if parts.is_empty() {
        String::new()
    } else {
        format!(" {}", parts.join(" "))
    };
    format!("  id=%{}{} ({kind})", var.id, annotation)
}

fn writeln_str(out: &mut String, line: &str) -> std::fmt::Result {
    out.push_str(line);
    out.push('\n');
    Ok(())
}
