# ps5-shader

PSSL → Vulkan HLSL → SPIR-V shader compiler in Rust.

The crate takes PSSL shader source through preprocessing, parsing,
semantic analysis, and lowering to a backend-agnostic Shader IR, then
emits Vulkan-compatible HLSL and compiles it to SPIR-V with DXC
(validated with SPIRV-Tools). No `unsafe` code (`#![forbid(unsafe_code)]`).

## Pipeline

```text
.pssl source
  → preprocess (`#include` / `#define` / conditional blocks, BOM stripping)
  → parse (PSSL grammar → AST)
  → semantic analysis (stage/entry inference, types, resources)
  → lower (AST → Shader IR)
  → emit (Shader IR → Vulkan HLSL)
  → DXC (HLSL → SPIR-V)
  → validate (SPIRV-Tools) / inspect (rspirv)
```

Framework vocabulary (`sce::` types, `sce::Agc::` runtime objects,
ray-tracing descriptors, wave intrinsics, …) has no HLSL counterpart
and is mapped centrally in `src/framework.rs` — never patched
per-shader. Unknown framework names fail loudly at DXC instead of
being guessed.

## Requirements

- Rust (stable) — `cargo build` / `cargo test`
- The vendored DXC toolchain in `dxc/` (used at *runtime* for HLSL →
  SPIR-V; intentionally tracked so a fresh clone can validate)

## Usage

```sh
cargo build
ps5-shader compile <shader.pssl> [options]

Options:
  --emit-hlsl          Print the generated Vulkan HLSL.
  --emit-spirv <file>  Write the SPIR-V binary to <file>.
  --validate           Validate the SPIR-V with SPIRV-Tools (Vulkan 1.3).
  --dump-ast           Print the parsed AST (Debug form).
  --dump-ir            Print the Shader IR (Debug form).
  --dump-spirv         Inspect the SPIR-V structurally via rspirv.
  -h, --help           Print help.
```

Single-shader and corpus tooling (all `cargo run --bin <name> -- …`):

| Binary           | Purpose                                                        |
| ---------------- | -------------------------------------------------------------- |
| `validate_one`   | Full pipeline for one shader with per-stage diagnostics.       |
| `bulk_validate`  | Validate shaders and update `shader-DXC-validated.md`.         |
| `dump_semantic`  | Dump per-shader semantic snapshots (`analysis/semantic.json`). |
| `regression`     | Re-run the corpus and diff HLSL/SPIR-V against a baseline.     |
| `dump_preproc` / `dump_expanded` | Inspect preprocessor output.                     |
| `dxc_probe`      | Compile an HLSL file directly (`dxc_probe <f.hlsl> <entry> <vs\|ps\|cs>`). |

## Configuration

Include roots are supplied by the caller — nothing proprietary is
hardcoded:

```rust
let mut cfg = PreprocessorConfig::platform_defaults();
cfg.include_dirs.push(my_project_root);
let out = preprocess_reported(&src, Some(path), &cfg)?;
```

- `PreprocessorConfig::platform_defaults()` carries the standard
  predefined macros and an (empty-by-default) platform root hook.
- Translated virtual headers are written best-effort to `cache/`
  (gitignored, debug only); the header source plus the translator
  rules version remain the source of truth.

## Layout

```text
src/
  lexer.rs / token.rs / parser/   Lexing + PSSL grammar → AST
  preprocess/                     Include resolution, classification, directives
  semantic/                       Stage/entry inference, type checking
  lower/                          AST → Shader IR (SRT flattening, hoisting, …)
  emit/                           Shader IR → Vulkan HLSL
  dxc/ / spirv/                   DXC compile, SPIR-V validate/inspect
  psfx/                           Virtual-header translation pipeline + debug cache
  framework.rs                    Central framework-type/constant mappings
  diagnostics.rs / error.rs       Source maps, phased errors
tests/                            Integration tests (unit + corpus matrix)
dxc/                              Vendored DXC toolchain binaries
```

## Status

Corpus results (1000 sample shaders):

| Stage     | Pass        |
| --------- | ----------- |
| Parse     | 1000/1000   |
| Semantic  | 1000/1000   |
| IR        | 1000/1000   |
| HLSL      | 1000/1000   |
| DXC       | 821/1000    |
| Validated | 821/1000    |

Every change is tracked against this baseline with zero regressions.
This is still research-stage software: when the toolchain cannot model
something it fails loudly (`cargo test` guards the baseline) rather
than guessing.

## Example

Input:

```cpp
struct V2P
{
    float4 pos : S_POSITION;
    float3 color : COLOR;
};

[CxxSymbol("Shader::ps")]
float4 main(V2P input) : S_TARGET_OUTPUT0
{
    return float4(input.color, 1.0);
}
```

Emitted HLSL (semantics lowered to Vulkan spellings):

```hlsl
struct V2P
{
    float4 pos : SV_Position;
    float3 color : TEXCOORD0;
};

// PSSL attribute: CxxSymbol
float4 main(V2P input) : SV_Target0
{
    return float4(input.color, 1.0);
}
```
