use ps5_shader::parse_shader;

// Regression tests for mesh-shader entry prerequisites:
// - constant ternary folding in `[num_threads]` bounds
//   (`((64 > 64) ? 64 : 64)` → `64`; macro-expanded mesh subgroup sizes);
// - `[output_primitive_type("tri")]` → HLSL `[outputtopology("triangle")]`.
// Both are provably-correct lowerings (literal evaluation; direct kind
// mapping verified against DXC mesh profiles). Full NGG-output lowering
// (per-lane scattering) remains future work; these unblock the profile
// path with truthful diagnostics (fail-closed, never miscompiled).

fn emit(source: &str) -> String {
    let module = parse_shader(source).expect("parser accepted shader");
    let model = ps5_shader::semantic::analyze(&module).expect("semantic analysis succeeded");
    let ir = ps5_shader::lower::lower_module(&module, &model).expect("lowering succeeded");
    ps5_shader::emit::emit_vulkan_hlsl(&ir)
}

const MESH_ATTRS: &str = "struct OutV { float4 p : S_POSITION; };\n\
     [num_threads(((64 > 64) ? 64 : 64), 1, 1)]\n\
     [output_primitive_type(\"tri\")]\n\
     [CxxSymbol(\"Shader::test_ms\")]\n\
     OutV main(uint tid : S_GROUP_THREAD_ID) {\n\
     OutV o; o.p = float4(tid, 0, 0, 1);\n\
     return o;\n\
     }\n";

#[test]
fn mesh_numthreads_ternary_folds() {
    let hlsl = emit(MESH_ATTRS);
    assert!(
        hlsl.contains("[numthreads(64, 1, 1)]"),
        "ternary bound must fold to 64, got:\n{hlsl}"
    );
}

#[test]
fn mesh_output_topology_maps() {
    let hlsl = emit(MESH_ATTRS);
    assert!(
        hlsl.contains("[outputtopology(\"triangle\")]"),
        "tri must map to triangle topology, got:\n{hlsl}"
    );
}

// When a mesh library declares both a bare helper-named `main` (NGG
// markers, no attributes) and the real entry (`main_common` with
// `[num_threads(...), profile("ms")]`), the attributed candidate must win.
// The bare `main` loses source order but carries no num_threads, so DXC
// rejects it (`mesh entry point must have a valid numthreads attribute`).
const ATTRIBUTED_ENTRY_OUTRANKS_BARE_MAIN: &str = "struct OutV { float4 p : S_POSITION; };\n\
     void main(uint tid : S_GROUP_THREAD_ID, out OutV o, out uint op : S_NGG_OUTPUT_PRIMITIVE)\n\
     { o.p = float4(tid, 0, 0, 1); }\n\
     [num_threads(64, 1, 1)]\n\
     [output_primitive_type(\"tri\")]\n\
     [profile(\"ms\")]\n\
     void main_common(uint tid : S_GROUP_THREAD_ID, out OutV o, out uint op : S_NGG_OUTPUT_PRIMITIVE)\n\
     { o.p = float4(tid, 0, 0, 1); }\n";

#[test]
fn mesh_entry_prefers_attributed_candidate() {
    let module = parse_shader(ATTRIBUTED_ENTRY_OUTRANKS_BARE_MAIN).expect("parser accepted shader");
    let model = ps5_shader::semantic::analyze(&module).expect("semantic analysis succeeded");
    let entry = model.entry_point.expect("an entry point is selected");
    assert_eq!(
        entry.function, "main_common",
        "attributed entry must outrank a bare helper-named main"
    );
}

// NGG-output mesh entries lower to DX mesh objects: `out vertices` /
// `out indices` sized by num_verts/num_prims, `SetMeshOutputCounts`,
// and predicated per-lane exports with packed-prim unpacking.
const MESH_NGG_RETURN: &str = "struct GSOut { float4 p : S_POSITION; };\n\
     struct NGG_OUTPUT { GSOut output; uint nggPrimExport : S_NGG_OUTPUT_PRIMITIVE; uint nggVertCount : S_NGG_OUTPUT_VERTEX_COUNT; uint nggPrimCount : S_NGG_OUTPUT_PRIMITIVE_COUNT; };\n\
     [num_threads(64, 1, 1)]\n\
     [num_verts(64)]\n\
     [num_prims(64)]\n\
     [output_primitive_type(\"tri\")]\n\
     [profile(\"ms\")]\n\
     NGG_OUTPUT main(uint tid : S_GROUP_THREAD_ID) {\n\
     NGG_OUTPUT ret;\n\
     ret.nggVertCount = 1;\n\
     ret.nggPrimCount = 0;\n\
     ret.nggPrimExport = 0;\n\
     ret.output.p = float4(tid, 0, 0, 1);\n\
     return ret;\n\
     }\n";

#[test]
fn mesh_ngg_return_lowers_to_objects() {
    let hlsl = emit(MESH_NGG_RETURN);
    assert!(
        hlsl.contains("out vertices GSOut __meshVerts[64]"),
        "vertices object must emit, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("out indices uint3 __meshTris[64]"),
        "indices object must emit, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("SetMeshOutputCounts("),
        "counts call must emit, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("NGG_OUTPUT main("),
        "NGG return must not survive, got:\n{hlsl}"
    );
}

// Out-parameter mesh entries (forwarding workers) gain mesh objects
// while the NGG out-params become locals: the worker call keeps
// matching and the epilogue exports through the objects.
const MESH_OUTPARAM_ENTRY: &str = "struct OutV { float4 p : S_POSITION; };\n\
     void worker(uint tid, out OutV o, out uint pc, out uint vc) { o.p = float4(tid, 0, 0, 1); vc = 1; pc = 0; }\n\
     [num_threads(64, 1, 1)]\n\
     [output_primitive_type(\"tri\")]\n\
     [profile(\"ms\")]\n\
     void main(uint tid : S_GROUP_THREAD_ID, out OutV o, out uint op : S_NGG_OUTPUT_PRIMITIVE, out uint vc : S_NGG_OUTPUT_VERTEX_COUNT, out uint pc : S_NGG_OUTPUT_PRIMITIVE_COUNT)\n\
     { worker(tid, o, pc, vc); }\n";

#[test]
fn mesh_outparam_entry_lowers() {
    let hlsl = emit(MESH_OUTPARAM_ENTRY);
    assert!(
        hlsl.contains("out vertices OutV __meshVerts[64]"),
        "vertices object must emit, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("S_NGG_OUTPUT_PRIMITIVE)"),
        "NGG params must leave the entry signature, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("SetMeshOutputCounts(vc, pc);"),
        "epilogue must export counts, got:\n{hlsl}"
    );
}

// Entries with early bare returns terminate lowering (a splice loop
// once re-expanded its own inserted `return`, growing memory without
// bound on asset-streaming indirect mesh shaders).
const MESH_EARLY_RETURN: &str = "struct OutV { float4 p : S_POSITION; };\n\
     void worker(uint tid, out OutV o, out uint pc, out uint vc, uint cull) { if (cull) { vc = 0; pc = 0; return; } o.p = float4(tid, 0, 0, 1); vc = 1; pc = 0; }\n\
     [num_threads(64, 1, 1)]\n\
     [output_primitive_type(\"tri\")]\n\
     [profile(\"ms\")]\n\
     void main(uint tid : S_GROUP_THREAD_ID, uint cull, out OutV o, out uint op : S_NGG_OUTPUT_PRIMITIVE, out uint vc : S_NGG_OUTPUT_VERTEX_COUNT, out uint pc : S_NGG_OUTPUT_PRIMITIVE_COUNT)\n\
     { if (cull) { vc = 0; pc = 0; return; } worker(tid, o, pc, vc, cull); }\n";

#[test]
fn mesh_early_bare_return_terminates() {
    let hlsl = emit(MESH_EARLY_RETURN);
    assert!(
        hlsl.contains("SetMeshOutputCounts("),
        "epilogue must emit, got:\n{hlsl}"
    );
}

// Whole-struct zero assigns (`s = 0`) desugar memberwise (meshlet
// `primOutput.output = 0`); resource/array members stay loud.
const STRUCT_ZERO_ASSIGN: &str = "struct Inner { float x; float y; };\n\
     struct Outer { Inner inner; float z; };\n\
     RWBuffer<float> outBuf;\n\
     [NUM_THREADS(1,1,1)]\n\
     void main(uint i : S_DISPATCH_THREAD_ID) {\n\
     Outer o;\n\
     o = 0;\n\
     outBuf[0] = o.inner.x + o.z;\n\
     }\n";

#[test]
fn struct_zero_assign_desugars_memberwise() {
    let hlsl = emit(STRUCT_ZERO_ASSIGN);
    assert!(
        !hlsl.contains("o = 0;"),
        "scalar-to-struct assign must not survive, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("o.inner.x = 0;") && hlsl.contains("o.z = 0;"),
        "members must zero individually, got:\n{hlsl}"
    );
}
