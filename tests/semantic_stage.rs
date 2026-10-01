use ps5_shader::ir::Stage;
use ps5_shader::{parse_shader, semantic};

fn stage_of(source: &str) -> Stage {
    let module = parse_shader(source).expect("parser accepted shader");
    let model = semantic::analyze(&module).expect("semantic analysis succeeded");
    model.stage.expect("a stage was inferred")
}

// Bare `CXX_SYMBOL("..._vv")` markers (imgui) resolve to vertex.
#[test]
fn bare_cxx_symbol_vv_is_vertex() {
    let stage = stage_of(
        "struct Varying { float4 pos : S_POSITION; };\n\
         CXX_SYMBOL(\"Shader::imgui_impl_ps_shader_vv\")\n\
         Varying main() { Varying v; v.pos = float4(0, 0, 0, 1); return v; }",
    );
    assert_eq!(stage, Stage::Vertex);
}

// Raytracing `[shader("kind")]` attributes map to DXR-equivalent stages.
#[test]
fn shader_kind_attributes_map_to_raytracing_stages() {
    assert_eq!(
        stage_of("[shader(\"anyhit\")] void main() { }"),
        Stage::AnyHit
    );
    assert_eq!(
        stage_of("[shader(\"closesthit\")] void main() { }"),
        Stage::ClosestHit
    );
    assert_eq!(stage_of("[shader(\"miss\")] void main() { }"), Stage::Miss);
    assert_eq!(
        stage_of("[shader(\"intersection\")] void main() { }"),
        Stage::Intersection
    );
    assert_eq!(
        stage_of("[shader(\"instance\")] void main() { }"),
        Stage::Intersection
    );
}

// `[NUM_THREADS(...)]` marks compute entries.
#[test]
fn num_threads_is_compute() {
    let stage = stage_of("[NUM_THREADS(8, 8, 1)] void main() { }");
    assert_eq!(stage, Stage::Compute);
}

// NGG outputs mark mesh entries; hull outputs mark hull entries.
#[test]
fn ngg_and_hull_outputs_map_by_type_name() {
    assert_eq!(
        stage_of("struct NGG_OUTPUT { float4 p : S_POSITION; };\nNGG_OUTPUT main() { NGG_OUTPUT o; o.p = float4(0,0,0,1); return o; }"),
        Stage::Mesh
    );
    assert_eq!(
        stage_of("struct HS_OUTPUT { float4 p : S_POSITION; };\nHS_OUTPUT main() { HS_OUTPUT o; o.p = float4(0,0,0,1); return o; }"),
        Stage::Hull
    );
    // `LS_OUTPUT` (local/vertex-flavor) by-value returns are vertex:
    // the Sony reference compiles `agc_basic_tessellation__local_vl`
    // (the corpus's only `LS_OUTPUT` entry) as a vertex shader. Genuine
    // hull entries carry `InputPatch`/patch attributes instead.
    assert_eq!(
        stage_of("struct LS_OUTPUT { float4 p : S_POSITION; };\nLS_OUTPUT main() { LS_OUTPUT o; o.p = float4(0,0,0,1); return o; }"),
        Stage::Vertex
    );
}

// A bare scalar/vector return with no return semantic is a fragment output.
#[test]
fn bare_vector_return_is_fragment() {
    assert_eq!(stage_of("float4 main() { return float4(0, 0, 0, 1); }"), Stage::Fragment);
    assert_eq!(stage_of("float3 main() { return float3(0, 0, 0); }"), Stage::Fragment);
}

// Pixel/compute/geometry CxxSymbol suffixes resolve without other signals.
#[test]
fn cxx_symbol_suffixes_resolve() {
    assert_eq!(
        stage_of("[CxxSymbol(\"Shader::model_dummy_p\")] void main() { }"),
        Stage::Fragment
    );
    assert_eq!(
        stage_of("[CxxSymbol(\"shaders::reset_indirect_args_c\")] void main() { }"),
        Stage::Compute
    );
    assert_eq!(
        stage_of("[CxxSymbol(\"Shader::particle_blur_g\")] void main() { }"),
        Stage::Geometry
    );
    assert_eq!(
        stage_of("[CxxSymbol(\"Shader::hs\")] void main() { }"),
        Stage::Hull
    );
}

// `clip()` / `discard` only appear in fragment entries (void alpha-test mains).
#[test]
fn kill_intrinsics_are_fragment() {
    assert_eq!(
        stage_of("void main(float4 c) { clip(c.x); }"),
        Stage::Fragment
    );
    assert_eq!(stage_of("void main() { discard; }"), Stage::Fragment);
}

// An input struct with vertex-only system values marks a vertex entry.
#[test]
fn instance_id_input_is_vertex() {
    let stage = stage_of(
        "struct PackVertex { float3 pos; uint id : S_INSTANCE_ID; };\n\
         void main(PackVertex input) { }",
    );
    assert_eq!(stage, Stage::Vertex);
}

// User calls keep their argument lists, so member access on results resolves.
#[test]
fn user_call_result_member_resolves() {
    let module = parse_shader(
        "float3 g(float3 x) { return x; }\n\
         float4 main(float3 a) : S_TARGET_OUTPUT0 { return float4(g(a), 1.0); }",
    )
    .unwrap();
    semantic::analyze(&module).expect("semantic analysis succeeded");
}

// Per-component builtins keep vector types; well-known constants exist.
#[test]
fn builtin_math_models() {
    let module = parse_shader(
        "float4 main(float3 a) : S_TARGET_OUTPUT0 {\n\
            float3 c = cross(a, a);\n\
            float y = exp2(2.0) + log2(2.0) + PI;\n\
            return float4(c, y); }",
    )
    .unwrap();
    semantic::analyze(&module).expect("semantic analysis succeeded");
}

// Fixed-width integer aliases behave as their underlying scalar.
#[test]
fn fixed_width_int_aliases() {
    let module = parse_shader(
        "float4 main(int32_t a, uint32_t b) : S_TARGET_OUTPUT0 {\n\
            float x = a + b;\n\
            return float4(x, x, x, 1.0); }",
    )
    .unwrap();
    semantic::analyze(&module).expect("semantic analysis succeeded");
}

// Nested-type references (`Outer::Inner`) resolve to the nested struct.
#[test]
fn scoped_nested_struct_resolves() {
    let module = parse_shader(
        "struct Outer { struct Inner { float3 Position; }; };\n\
         float4 main(Outer::Inner v) : S_TARGET_OUTPUT0 { return float4(v.Position, 1.0); }",
    )
    .unwrap();
    let model = semantic::analyze(&module).expect("semantic analysis succeeded");
    assert_eq!(model.stage, Some(Stage::Fragment));
}

// Indexing a matrix yields a row vector; indexing a sized local array
// yields its element type.
#[test]
fn index_rules() {
    let module = parse_shader(
        "float4x4 g_m;\n\
         float4 main(float4 a[2]) : S_TARGET_OUTPUT0 {\n\
             float3 row = g_m[3].xyz;\n\
             return a[1] + float4(row, 1.0); }",
    )
    .unwrap();
    semantic::analyze(&module).expect("semantic analysis succeeded");
    let module = parse_shader(
        "[NUM_THREADS(8, 8, 1)]\n\
         void main() {\n\
             float4 boxes[4];\n\
             float2 v = boxes[2].xy; }",
    )
    .unwrap();
    semantic::analyze(&module).expect("semantic analysis succeeded");
}

// Scalar-literal splats (`1.0.xxx`) broadcast across the swizzle width.
#[test]
fn scalar_literal_splat() {
    let module = parse_shader(
        "float4 main() : S_TARGET_OUTPUT0 { return float4(1.0.xxx, 1.0); }",
    )
    .unwrap();
    semantic::analyze(&module).expect("semantic analysis succeeded");
}

// `auto` (undeducible) operands defer to the other side in arithmetic.
#[test]
fn auto_operand_defers() {
    let module = parse_shader("float f(auto a) { return a + 1.0; }").unwrap();
    semantic::analyze(&module).expect("semantic analysis succeeded");
}

// Wave/DXR builtins resolve with honest types.
#[test]
fn wave_dxr_builtins() {
    let module = parse_shader(
        "[NUM_THREADS(8, 8, 1)]\n\
         void main(float x) {\n\
             float a = exp2(x) + CrossLaneAdd(x);\n\
             bool h = ReportHit(0.0, 0u, float4(0, 0, 0, 1));\n\
             float3 o = WorldRayOrigin() + x; }",
    )
    .unwrap();
    semantic::analyze(&module).expect("semantic analysis succeeded");
}

// `[profile("cs")]` names the stage like the DXC profile does.
#[test]
fn profile_attribute_stage() {
    assert_eq!(
        stage_of("[profile(\"cs\")]\nvoid main() { }"),
        Stage::Compute
    );
}

// Enum-typed template value params (`template <StackType s>`) are visible.
#[test]
fn enum_template_value_param() {
    let module = parse_shader(
        "enum class StackType : uint { kA, kB };\n\
         template <StackType s>\n\
         uint f() { return s == StackType::kA ? 1u : 0u; }",
    )
    .unwrap();
    semantic::analyze(&module).expect("semantic analysis succeeded");
}

// Scalar splats broadcast (`fog.xxx`).
#[test]
fn scalar_splat_broadcast() {
    let module = parse_shader(
        "float4 main(float fog) : S_TARGET_OUTPUT0 {\n\
            return float4(1.0f - fog.xxx, 1.0); }",
    )
    .unwrap();
    semantic::analyze(&module).expect("semantic analysis succeeded");
}

// `bit_cast<T>(x)` keeps its target type.
#[test]
fn bit_cast_target_type() {
    let module = parse_shader(
        "uint4 f(uint x) { return bit_cast<uint4>(x); }",
    )
    .unwrap();
    semantic::analyze(&module).expect("semantic analysis succeeded");
}

// Lambdas assigned to `auto` are callable through the opaque-type rules.
#[test]
fn lambda_auto_callable() {
    let module = parse_shader(
        "float f() {\n\
            const auto g = [&](uint32_t idx, float weight) { return; };\n\
            g(0u, 1.0);\n\
            return 1.0; }",
    )
    .unwrap();
    semantic::analyze(&module).expect("semantic analysis succeeded");
}

// `sizeof(T)` and calls through `auto` callables resolve.
#[test]
fn sizeof_and_auto_callable() {
    let module = parse_shader(
        "uint f(auto pred, uint x) {\n\
            uint s = sizeof(x);\n\
            if (pred(x)) { return s; }\n\
            return 0u; }",
    )
    .unwrap();
    semantic::analyze(&module).expect("semantic analysis succeeded");
}

// Multi-level member chains on template-typed values resolve opaquely.
#[test]
fn template_member_chain() {
    let module = parse_shader(
        "template <typename T>\n\
         void accumulate(T const *srt) {\n\
             float v = srt->m_statistics[0].m_maxStackSizeCount; }",
    )
    .unwrap();
    semantic::analyze(&module).expect("semantic analysis succeeded");
}

// Wave/bit builtins resolve with honest types.
#[test]
fn wave_bit_builtins() {
    let module = parse_shader(
        "[NUM_THREADS(8, 8, 1)]\n\
         void main(uint x, float y) {\n\
             uint a = MaskBitCnt(x) + CountSetBits(x) + ReadLane(x);\n\
             float s; float c;\n\
             sincos(y, s, c);\n\
             float b = s + c + y; }",
    )
    .unwrap();
    semantic::analyze(&module).expect("semantic analysis succeeded");
}

// `(void)x;` discards its operand; `long` is a 32-bit int.
#[test]
fn void_cast_and_long() {
    let module = parse_shader(
        "[NUM_THREADS(8, 8, 1)]\n\
         void main(uint subgroupId : S_GROUP_ID, long x) {\n\
             (void)subgroupId;\n\
             int y = x + 1; }",
    )
    .unwrap();
    semantic::analyze(&module).expect("semantic analysis succeeded");
}

// Enum enumerators resolve as ints; `static int constexpr` declares.
#[test]
fn enum_and_constexpr_decl() {
    let module = parse_shader(
        "enum Options { kOptA = 1 << 0, kOptB };\n\
         uint f(uint opts) {\n\
             static int constexpr n = 1;\n\
             uint r = 0u;\n\
             for (int i = 0; i < n; i++) { r += opts & kOptB; }\n\
             return r; }",
    )
    .unwrap();
    semantic::analyze(&module).expect("semantic analysis succeeded");
}

// A `static constexpr VertexLayout vertexLayout = VertexLayout::Create(...)`
// global (Framework C++ construct) must be elided WITHOUT letting the elision
// cascade and eat the rest of the file (regression: `model_vv`/`shadow_vv`
// produced zero functions -> "no stage").
#[test]
fn vertex_layout_elision_keeps_following_functions() {
    let module = parse_shader(
        "struct VS_OUTPUT { float4 p : S_POSITION; };\n\
         static constexpr VertexLayout vertexLayout = VertexLayout::Create(\n\
             VertexLayout::Attribute(sce::Agc::Core::VertexTableIndex{0}, 0));\n\
         VS_OUTPUT main() { VS_OUTPUT o; o.p = float4(0,0,0,1); return o; }",
    )
    .expect("parser keeps the entry after eliding the layout");
    let f = module
        .functions
        .iter()
        .find(|f| f.name == "main")
        .expect("main survives the layout elision");
    assert_eq!(f.parameters.len(), 0);
}

// A mesh entry that returns `NGG_OUTPUT` and carries only compute dispatch
// IDs is still a mesh shader (`meshlet_draw_*_ms`).
#[test]
fn ngg_output_return_overrides_dispatch_ids() {
    let module = parse_shader(
        "struct NGG_OUTPUT { float4 p : S_POSITION; };\n\
         [NUM_THREADS(8, 8, 1)]\n\
         NGG_OUTPUT main(uint subgroupId : S_GROUP_ID, uint tid : S_GROUP_THREAD_ID) {\n\
             NGG_OUTPUT o; o.p = float4(0,0,0,1); return o; }",
    )
    .unwrap();
    let model = semantic::analyze(&module).unwrap();
    assert_eq!(model.stage, Some(ps5_shader::ir::Stage::Mesh));
}
