use super::parse_shader;
use crate::ast::{Expr, Semantic};
use crate::ast::{BinaryOp, ScalarType, Stmt, Type};

#[test]
fn top_level_enum_registers_values_as_constants() {
    // `enum class OutputEncoding { PQ, P3D65, LinearNits }` (auto
    // 0,1,2) and explicit/shifted values must land in
    // `module.constants` so `case` labels and value uses resolve.
    let source = "enum class OutputEncoding { PQ, P3D65, LinearNits }; enum Flags { A = 1, B, C = 1 << 3, D, E = 0x10 };";
    let module = parse_shader(source).unwrap();
    let val = |n: &str| match module.constants.get(n) {
        Some(Expr::Number(s)) => s.clone(),
        other => panic!("{n} must be a numeric constant, got {other:?}"),
    };
    assert_eq!(val("PQ"), "0");
    assert_eq!(val("P3D65"), "1");
    assert_eq!(val("LinearNits"), "2");
    assert_eq!(val("A"), "1");
    assert_eq!(val("B"), "2");
    assert_eq!(val("C"), "8");
    assert_eq!(val("D"), "9");
    assert_eq!(val("E"), "16");
}

#[test]
fn nested_struct_enum_registers_values_as_constants() {
    // `enum Flags` inside `MaterialSRT`: same constants as top-level.
    let source = "struct MaterialSRT { uint m_flags; enum Flags { DoubleSided = 1, Transparent = 2 }; };";
    let module = parse_shader(source).unwrap();
    assert_eq!(module.structs.len(), 1);
    assert_eq!(module.structs[0].fields.len(), 1);
    match module.constants.get("Transparent") {
        Some(Expr::Number(s)) => assert_eq!(s, "2"),
        other => panic!("Transparent must be numeric 2, got {other:?}"),
    }
}

#[test]
fn parses_simple_struct() {
    let source = "struct V2P { float4 pos : S_POSITION; float3 color : COLOR; };";
    let module = parse_shader(source).unwrap();
    assert_eq!(module.structs.len(), 1);
    let s = &module.structs[0];
    assert_eq!(s.name, "V2P");
    assert_eq!(s.fields.len(), 2);
    assert_eq!(s.fields[0].name, "pos");
    assert_eq!(
        s.fields[0].semantic.as_ref().map(Semantic::name),
        Some("S_POSITION")
    );
}

#[test]
fn parses_main_function() {
    let source = "[CxxSymbol(\"Shader::ps\")]\n\
        float4 main(V2P input) : S_TARGET_OUTPUT0 {\n\
        return float4(input.color, 1.0);\n\
        }";
    let module = parse_shader(source).unwrap();
    assert_eq!(module.functions.len(), 1);
    let f = &module.functions[0];
    assert_eq!(f.name, "main");
    assert_eq!(f.attributes.len(), 1);
    assert_eq!(f.attributes[0].name, "CxxSymbol");
    assert_eq!(f.parameters.len(), 1);
    assert_eq!(f.parameters[0].name, "input");
}

#[test]
fn parses_union_in_function() {
    let source = "void foo() { union { uint2 a; float *b; } u; u.a = uint2(1,2); }";
    let module = parse_shader(source).unwrap();
    assert_eq!(module.functions.len(), 1);
}

#[test]
fn parses_hash_in_attribute() {
    let source = "[CxxSymbol(\"Shader::\" #SYMBOL)]\nvoid foo() {}";
    let module = parse_shader(source).unwrap();
    assert_eq!(module.functions.len(), 1);
}

#[test]
fn parses_string_paste_in_attribute() {
    // Regression: `[cxxsymbol("A" #M "_vgprs")]` failed with
    // `expected `)`, found string literal` (adjacent literals paste).
    let source = "[cxxsymbol(\"Shader::\" #VGPR_COUNT \"_vgprs\")]\nvoid foo() {}";
    let module = parse_shader(source).unwrap();
    assert_eq!(module.functions.len(), 1);
    assert_eq!(
        module.functions[0].attributes[0].arguments.len(),
        1,
    );
}

#[test]
fn parses_token_paste_in_name() {
    // Regression: `closest_hit_caller_##VGPR_COUNT##_vgprs` failed
    // (`##` pasting from function-generation macros).
    let source = "void closest_hit_caller_##VGPR_COUNT##_vgprs() { }";
    let module = parse_shader(source).unwrap();
    assert_eq!(module.functions.len(), 1);
    assert_eq!(
        module.functions[0].name,
        "closest_hit_caller_VGPR_COUNT_vgprs",
    );
}

#[test]
fn parses_designated_aggregate_init() {
    // Regression: `T { .m_org = v, ... }` failed with
    // `expected expression, found `.``.
    let source = "struct VPL { float3 o; float i; }; void f() { VPL v = VPL { .m_org = get(), .m_direct = 0 }; float a[2] = { .5f, .6f }; }";
    let module = parse_shader(source).unwrap();
    assert_eq!(module.functions.len(), 1);
}

#[test]
fn parses_explicit_specialization() {
    // Regression: `template<> Vector3 shade<0>(...)` failed with
    // `expected `;`, found `<``.
    let source = "template <typename T> T shade(T a) { return a; } template<> uint shade<0>(uint a) { return a; }";
    let module = parse_shader(source).unwrap();
    assert_eq!(module.functions.len(), 2);
}

#[test]
fn parses_lambda_scoped_ptr_return() {
    // Regression: `[...](...) -> Scope::T const * {...}` failed with
    // `expected `;`, found `const`` (trailing return declarators).
    let source = "struct T { uint x; }; void f() { auto g = [x](uint i) -> T const * { return nullptr; }; }";
    let module = parse_shader(source).unwrap();
    assert_eq!(module.functions.len(), 1);
}

#[test]
fn parses_template_with_typename() {
    let source = "template <typename T> T const *bar(uint2 a) { union { uint2 aa; T const *ptr; } u; u.aa = a; return u.ptr; } void test() { bar<float>(uint2(1,2)); }";
    let module = parse_shader(source).unwrap();
    assert!(module.functions.len() >= 1);
}

#[test]
fn parses_brace_init_without_equal() {
    let source = "struct S { float3 c; }; void foo() { S s{}; S t{}; }";
    let module = parse_shader(source).unwrap();
    assert_eq!(module.functions.len(), 1);
}

#[test]
fn parses_isolate_attribute() {
    let source = "float3 foo(float4 c) { [isolate] { float2 x = float2(1,2); } return float3(0,0,0); }";
    let module = parse_shader(source).unwrap();
    assert_eq!(module.functions.len(), 1);
}

#[test]
fn parses_srt_signature_blocks() {
    let source = "SrtSignature MySig { DescriptorTable(NamedConstantBuffer(CB, flags=INLINED_CONSTANT_BUFFER)) }; void foo() {}";
    let module = parse_shader(source).unwrap();
    // SrtSignature should be skipped, leaving one function
    assert_eq!(module.functions.len(), 1);
}

#[test]
fn parses_struct_inside_function() {
    let source = "void foo() { struct Foo { float x; float y; }; Foo f; f.x = 1.0; }";
    let module = parse_shader(source).unwrap();
    assert_eq!(module.functions.len(), 1);
}

#[test]
fn parses_struct_def_followed_by_auto_decl() {
    // Regression: `struct Basis { ... };` must not consume the next
    // statement's first token (`auto`) as a variable instance.
    // (psr_bvh_update_heatmap_lib: `expected identifier, found `=``)
    let source = "void foo(float3 n) { struct Basis { float3 u, v, w; }; auto const w = n; auto tv = float3(0.0f); }";
    let module = parse_shader(source).unwrap();
    assert_eq!(module.functions.len(), 1);
    let body = &module.functions[0].body;
    // struct def (Block) + 2 VarDecls
    assert!(body.len() >= 3);
}

#[test]
fn parses_struct_with_instance_still_works() {
    let source = "void foo() { struct S { float x; } myS; myS.x = 1.0; }";
    let module = parse_shader(source).unwrap();
    assert_eq!(module.functions.len(), 1);
}

#[test]
fn parses_anonymous_union_followed_by_stmt() {
    let source = "void foo() { union { uint2 a; float b; }; float x = 1.0; }";
    let module = parse_shader(source).unwrap();
    assert_eq!(module.functions.len(), 1);
}

#[test]
fn parses_function_returning_sampler_state() {
    // Regression: `static constexpr SamplerState MakeClampLinearSampler()`
    // must parse as a function, not a resource (`expected `;`, found `(`).
    let source = "static constexpr SamplerState MakeClampLinearSampler() { SamplerState s; return s; }";
    let module = parse_shader(source).unwrap();
    assert_eq!(module.functions.len(), 1);
    assert_eq!(module.functions[0].name, "MakeClampLinearSampler");
    // Real resources must still parse as resources.
    let source = "SamplerState samp; ConstantBuffer CB : register(b0) { float4 c; };";
    let module = parse_shader(source).unwrap();
    assert_eq!(module.resources.len(), 2);
}

#[test]
fn parses_template_trait_with_scope_access() {
    // Regression: `wave::__is_same<A, B>::value` must parse `<...>` as
    // template args (followed by `::`), not as a comparison — otherwise the
    // `,` inside breaks `if (...)` with `expected `)`, found `,``.
    let source = "void foo() { if constexpr (wave::__is_same<uint, uint>::value) { float x = 1.0; } }";
    let module = parse_shader(source).unwrap();
    assert_eq!(module.functions.len(), 1);
}

#[test]
fn preserves_ngg_culldata_element_type() {
    // `Ngg::CullData<T>::Set/GetTriangle` is an NGG primitive-shader
    // builtin; the compat layer needs `T`, so the template arg survives
    // (every other scoped template keeps the drop behavior).
    let source = "void foo() { Ngg::CullData<MyCullVertex>::Set(v); auto t = Ngg::CullData<float4>::GetTriangle(); }";
    let module = parse_shader(source).unwrap();
    let body = &module.functions[0].body;
    assert_eq!(body.len(), 2);
    let Stmt::Expr(Expr::Call { function, .. }) = &body[0] else {
        panic!("expected Set call, got {:?}", body[0]);
    };
    let Expr::Variable(callee) = function.as_ref() else {
        panic!("expected variable callee, got {function:?}");
    };
    assert_eq!(callee, "Ngg::CullData<MyCullVertex>::Set");
}

#[test]
fn parses_struct_c_style_cast() {
    // `(PBRInfo)0` is a C-style cast of a known struct (zero-init), not a
    // parenthesized variable followed by a stray `0;` statement.
    let source = "struct PBRInfo { float x; }; void foo() { PBRInfo ret = (PBRInfo)0; }";
    let module = parse_shader(source).unwrap();
    let body = &module.functions[0].body;
    assert_eq!(body.len(), 1);
    let Stmt::VarDecl { init, .. } = &body[0] else {
        panic!("expected VarDecl, got {:?}", body[0]);
    };
    let Some(Expr::Constructor { ty, arguments }) = init else {
        panic!("expected Constructor init, got {init:?}");
    };
    assert_eq!(ty, &Type::Named("PBRInfo".to_string()));
    assert_eq!(arguments.len(), 1);
}

#[test]
fn parses_scoped_struct_c_style_cast() {
    // `(sce::Got::VisGsOut)0` is a C-style cast of the flattened
    // `VisGsOut` (namespaces flatten at parse), not a parenthesized
    // scoped expression followed by a stray `0;` (vis_draw_vv).
    let source = "namespace sce { namespace Got { struct VisGsOut { float x; }; } } void foo() { auto o = (sce::Got::VisGsOut)0; }";
    let module = parse_shader(source).unwrap();
    let body = &module.functions[0].body;
    assert_eq!(body.len(), 1);
    let Stmt::VarDecl { init, .. } = &body[0] else {
        panic!("expected VarDecl, got {:?}", body[0]);
    };
    let Some(Expr::Constructor { ty, arguments }) = init else {
        panic!("expected Constructor init, got {init:?}");
    };
    assert_eq!(ty, &Type::Named("VisGsOut".to_string()));
    assert_eq!(arguments.len(), 1);
}

#[test]
fn parses_captureless_lambda_with_body() {
    // `auto convertUInt16 = [](ulong v) { return ...; };` keeps params
    // and body (for beta-reduction); captures keep the opaque
    // placeholder (today's path).
    let source = "void foo() { auto f = [](ulong v) { return (v & 0xffff) / 65535.0f; }; auto g = [srt](float3 a) { return a; }; }";
    let module = parse_shader(source).unwrap();
    let body = &module.functions[0].body;
    assert_eq!(body.len(), 2);
    let Stmt::VarDecl { init: Some(Expr::Lambda { captures, params, body: lam }), .. } = &body[0] else {
        panic!("capture-less lambda must preserve body, got {:?}", body[0]);
    };
    assert!(captures.is_empty(), "captures must be empty, got {captures:?}");
    assert_eq!(params, &vec!["v".to_string()]);
    assert_eq!(lam.len(), 1);
    let Stmt::VarDecl { init: Some(Expr::Lambda { captures, params, .. }), .. } = &body[1] else {
        panic!("capturing lambda must preserve captures, got {:?}", body[1]);
    };
    assert_eq!(captures, &vec!["srt".to_string()]);
    assert_eq!(params, &vec!["a".to_string()]);
}

#[test]
fn parses_inline_struct_param_definition() {
    // `float4 main(struct UserData { float4 x; } p)` registers the
    // struct (clear shaders) instead of skipping its body.
    let source = "float4 main(struct UserData { float4 m_clearColor; } srt_data : S_SRT_DATA) : S_TARGET_OUTPUT { return srt_data.m_clearColor; }";
    let module = parse_shader(source).unwrap();
    assert!(
        module.structs.iter().any(|s| s.name == "UserData"),
        "inline struct must register, got {:?}",
        module.structs.iter().map(|s| &s.name).collect::<Vec<_>>()
    );
    let user = module.structs.iter().find(|s| s.name == "UserData").unwrap();
    assert_eq!(user.fields.len(), 1);
    assert_eq!(user.fields[0].name, "m_clearColor");
}

#[test]
fn explicit_literal_template_args_mangle_callee() {
    // `f<true>(x)` keeps its literal args in the callee spelling (for
    // monomorphization); identifier/type args (`f<T>(x)`,
    // `minT<uint2>(x)`) preserve as type-arg spellings for the same
    // machinery (`split_mangled_ty` in lowering binds them to typename
    // params).
    let source = "float f(float x) { return x; }\n\
         float g(float x) { return x + f<true>(x) + f<T>(x); }";
    let module = parse_shader(source).unwrap();
    let g = module.functions.iter().find(|f| f.name == "g").unwrap();
    let text = format!("{g:?}");
    assert!(
        text.contains("f<true>"),
        "literal args must mangle, got: {text}"
    );
    assert!(
        text.contains("f<T>"),
        "identifier args must mangle as type args, got: {text}"
    );
}

#[test]
fn template_id_spellings_recorded_per_function() {
    // `BcnEncodeSrt<uint2> g_srt` preserves its args (normalized) under
    // the enclosing function; plain params record nothing.
    let source = "struct BcnEncodeSrt { uint x; };\n\
         void entry(BcnEncodeSrt<uint2> g_srt, float y) {}\n\
         void plain(float z) {}";
    let module = parse_shader(source).unwrap();
    let entry = module.template_spellings.get("entry").unwrap();
    assert_eq!(entry.get("g_srt").map(String::as_str), Some("BcnEncodeSrt<uint2>"));
    assert!(!entry.contains_key("y"));
    assert!(!module.template_spellings.contains_key("plain"));
}

#[test]
fn template_params_recorded_per_function() {
    // `template<typename T, bool B>` prefixes drain ordered (name,
    // is_value) params into the next function; a following struct
    // clears rather than inherits them.
    let source = "template<typename T, bool B> float3 f(T x) { return float3(x); }\n\
         template<typename U> struct S { U m; };\n\
         float g(float y) { return y; }";
    let module = parse_shader(source).unwrap();
    let f = module.functions.iter().find(|f| f.name == "f").unwrap();
    assert_eq!(
        f.template_params,
        vec![("T".to_string(), false), ("B".to_string(), true)]
    );
    let g = module.functions.iter().find(|f| f.name == "g").unwrap();
    assert!(g.template_params.is_empty());
}

#[test]
fn scoped_struct_construction_folds_to_constructor() {
    // `shader::MaterialData(...)` with bare `MaterialData` declared is
    // a constructor call (parsed as Call since scoped names aren't
    // known structs at parse): it folds to a Constructor so downstream
    // construction handling applies (barycentric pixel shaders).
    let source = "namespace shader { struct MaterialData { float x; }; } float f(float a) { return shader::MaterialData(a).x; }";
    let module = parse_shader(source).unwrap();
    let body = &module.functions[0].body;
    assert_eq!(body.len(), 1);
    let Stmt::Return(Some(expr)) = &body[0] else {
        panic!("expected return, got {:?}", body[0]);
    };
    // Member access on the construction (`.x` postfix).
    let Expr::Member { object, member } = expr else {
        panic!("expected member, got {expr:?}");
    };
    assert_eq!(member, "x");
    let Expr::Constructor { ty, arguments } = object.as_ref() else {
        panic!("expected Constructor, got {:?}", object.as_ref());
    };
    assert_eq!(ty, &Type::Named("MaterialData".to_string()));
    assert_eq!(arguments.len(), 1);
}

#[test]
fn scoped_static_call_with_default_arg_folds() {
    // `sce::Got::heatmap(x)` must fold to bare `heatmap(x)` even though
    // the declaration has 2 params: the trailing one is defaulted, so
    // arity 1 is admitted and lowering fills the default (vis_draw_vv).
    let source = "namespace sce { namespace Got { float heat(float s, bool b = false) { return s; } } } float foo() { return sce::Got::heat(0.5); }";
    let module = parse_shader(source).unwrap();
    let foo = module.functions.iter().find(|f| f.name == "foo").unwrap();
    let body = &foo.body;
    assert_eq!(body.len(), 1);
    let Stmt::Return(Some(expr)) = &body[0] else {
        panic!("expected return, got {:?}", body[0]);
    };
    let Expr::Call { function, arguments } = expr else {
        panic!("expected call, got {expr:?}");
    };
    assert!(
        matches!(function.as_ref(), Expr::Variable(n) if n == "heat"),
        "scoped call with defaulted trailing param must fold, got {function:?}"
    );
    assert_eq!(arguments.len(), 1);
}

#[test]
fn parses_multidim_global_array() {
    // Regression: `thread_group_memory uint a[64][2];` — module globals
    // only handled one dimension (`expected `;`, found `[``).
    let source = "thread_group_memory uint s_itemQueue[64 * 2]; thread_group_memory uint s_outSSCTilePacking[64][2]; void foo() {}";
    let module = parse_shader(source).unwrap();
    assert_eq!(module.globals.len(), 2);
    assert_eq!(module.globals[1].name, "s_outSSCTilePacking");
}

#[test]
fn parses_multidim_struct_field() {
    let source = "struct S { float m[4][2]; }; void foo() {}";
    let module = parse_shader(source).unwrap();
    assert_eq!(module.structs.len(), 1);
    assert_eq!(module.structs[0].fields[0].name, "m");
}

#[test]
fn multidim_global_flattens_to_product_and_linearizes_uses() {
    // `float3 l_colors[Y][X]` flattens to Y*X elements; `a[i][j]`
    // becomes `a[i*X + j]` (row-major, matching PSSL layout).
    let source = "thread_group_memory float3 l_colors[2][3];\n\
         void f(uint y, uint x, float3 v) { l_colors[y][x] = v; }";
    let module = parse_shader(source).unwrap();
    assert_eq!(module.globals[0].array_size, Some(6));
    let Stmt::Expr(Expr::Assign { lhs, .. }) = &module.functions[0].body[0] else {
        panic!("expected indexed store, got {:?}", module.functions[0].body[0]);
    };
    let Expr::Index { object, index } = lhs.as_ref() else {
        panic!("expected index lhs, got {lhs:?}");
    };
    assert!(matches!(object.as_ref(), Expr::Variable(n) if n == "l_colors"));
    let Expr::Binary { op: BinaryOp::Add, left, right } = index.as_ref() else {
        panic!("expected linearized i*X+j, got {index:?}");
    };
    let Expr::Binary { op: BinaryOp::Mul, right: stride, .. } = left.as_ref() else {
        panic!("expected stride multiply, got {left:?}");
    };
    assert!(matches!(stride.as_ref(), Expr::Number(n) if n == "3"));
    let _ = right;
}

#[test]
fn multidim_3d_flattens_to_product_and_linearizes_uses() {
    // `float a[4][3][2]` flattens to 24 elements; `a[i][j][k]`
    // becomes `a[(i*3+j)*2+k]` (row-major, matching PSSL layout).
    let source = "thread_group_memory float a[4][3][2];\n\
         void f(uint i, uint j, uint k, float v) { a[i][j][k] = v; }";
    let module = parse_shader(source).unwrap();
    assert_eq!(module.globals[0].array_size, Some(24));
    let Stmt::Expr(Expr::Assign { lhs, .. }) = &module.functions[0].body[0] else {
        panic!("expected indexed store, got {:?}", module.functions[0].body[0]);
    };
    let Expr::Index { object, index } = lhs.as_ref() else {
        panic!("expected index lhs, got {lhs:?}");
    };
    assert!(matches!(object.as_ref(), Expr::Variable(n) if n == "a"));
    // Outermost op is `+ k`.
    let Expr::Binary { op: BinaryOp::Add, .. } = index.as_ref() else {
        panic!("expected linearized (i*D1+j)*D2+k, got {index:?}");
    };
}

#[test]
fn multidim_nested_init_keeps_legacy_shape() {
    // `= {{...}}` inits belong to the vector rule in lowering (float4[4]),
    // so nested-init locals keep first-dimension sizing.
    let source = "void f() { float M[4][4] = {{1,2,3,4},{5,6,7,8},{9,10,11,12},{13,14,15,16}}; }";
    let module = parse_shader(source).unwrap();
    let Stmt::VarDecl { array_size, init, .. } = &module.functions[0].body[0] else {
        panic!("expected vardecl, got {:?}", module.functions[0].body[0]);
    };
    assert_eq!(*array_size, Some(4));
    assert!(matches!(init, Some(Expr::InitializerList(_))));
}

#[test]
fn parses_for_with_comma_updates() {
    // Regression: `for (...; ...; i++, pos.y++)` multi-update.
    let source = "void foo() { float x = 0.0; for (uint i = 0; i < 4; i++, x++) { x = x + 1.0; } }";
    let module = parse_shader(source).unwrap();
    assert_eq!(module.functions.len(), 1);
}

#[test]
fn parses_lt_comparison_before_distant_gt_paren() {
    // Regression: `<` of a comparison must not be treated as template
    // args even when a `> (` appears much later (the `<` in
    // `dot(...) < 0.0f` scanned kilobytes to a far-away `>` and ate
    // whole functions → `expected type, found `{``).
    let source = "void foo(float3 a) { bool b = (dot(a, a) < 0.0f); float x = 1.0; float y = (x > (0.5)) ? x : 0.0; float z = x + y; }";
    let module = parse_shader(source).unwrap();
    assert_eq!(module.functions.len(), 1);
    // 4 separate declarations; the runaway merged them into fewer.
    assert_eq!(module.functions[0].body.len(), 4);
}

#[test]
fn parses_sample_interpolation_struct_fields() {
    // Regression: AGC shading samples use `sample float4 pos : S_POSITION;`
    // (29 shaders failed with `expected identifier, found `:`).
    let source = "struct V2P { sample float4 pos : S_POSITION; sample float4 nrm : TEXCOORD0; sample uint sampleIndex : S_SAMPLE_INDEX; };";
    let module = parse_shader(source).unwrap();
    assert_eq!(module.structs.len(), 1);
    assert_eq!(module.structs[0].fields.len(), 3);
    assert_eq!(module.structs[0].fields[0].name, "pos");
    // Combined modifiers + precise/unorm prefixes.
    let source = "struct S { linear centroid float4 a : TEXCOORD0; noperspective float2 b : TEXCOORD1; precise sample float4 c : TEXCOORD2; }; void foo() { sample float4 x = float4(0,0,0,0); }";
    let module = parse_shader(source).unwrap();
    assert_eq!(module.structs[0].fields.len(), 3);
    assert_eq!(module.functions.len(), 1);
}

#[test]
fn parses_interp_modifier_parameters() {
    // Regression: interpolation modifiers on entry-point parameters.
    let source = "struct V2P { float4 pos : S_POSITION; }; float4 main(linear centroid V2P input, sample float4 c : TEXCOORD0) : S_TARGET_OUTPUT0 { return c; }";
    let module = parse_shader(source).unwrap();
    assert_eq!(module.functions.len(), 1);
    assert_eq!(module.functions[0].parameters.len(), 2);
}

#[test]
fn parses_out_of_class_method_definitions() {
    // Regression: `uint4 MeshletSkinData::uncompressBoneIndices(...)`
    // failed with `expected identifier, found `:` (matl_* shaders).
    let source = "struct MeshletSkinData { uint x; }; uint4 MeshletSkinData::uncompressBoneIndices(const uint input, out uint numBonesUsed) { numBonesUsed = (input & 0x3) + 1; uint4 ret = uint4(0,0,0,0); return ret; }";
    let module = parse_shader(source).unwrap();
    assert_eq!(module.functions.len(), 1);
    assert_eq!(module.functions[0].name, "uncompressBoneIndices");
    // Constructor without return type.
    let source = "struct S { uint x; }; S::S() { } void foo() {}";
    let module = parse_shader(source).unwrap();
    assert!(module.functions.iter().any(|f| f.name == "S"));
}

#[test]
fn parses_lambda_with_trailing_return_type() {
    // Regression: `auto f = [cap](uint i) -> uint { ... };` — the `-> uint`
    // was read as member access (`expected identifier, found `uint``).
    let source = "void foo() { ulong vert_ballot = 1; auto getCompactIndex64 = [vert_ballot](uint index) -> uint { ulong m = (ulong(1) << index) - ulong(1); return popcnt(m & vert_ballot); }; uint i0 = getCompactIndex64(3); }";
    let module = parse_shader(source).unwrap();
    assert_eq!(module.functions.len(), 1);
}

#[test]
fn parses_resource_with_initializer() {
    // Regression: `static const SamplerState s = make_bilinear_sampler();`
    // failed with `expected `;`, found `=`` (psfx compute shaders).
    let source = "static const SamplerState tx_sampler = make_bilinear_sampler(); void foo() {}";
    let module = parse_shader(source).unwrap();
    assert_eq!(module.resources.len(), 1);
    assert_eq!(module.resources[0].name, "tx_sampler");
}

#[test]
fn parses_srt_buffer_macros() {
    // Regression: `REGULAR_BUFFER(T) name;` members failed with
    // `expected identifier, found `;`` (clustered-geometry SRT structs).
    let source = "struct InputVertex { float3 p; }; struct SRT_GeometryData { REGULAR_BUFFER(InputVertex) vertexBuffer; REGULAR_BUFFER(uint) indexBuffer; };";
    let module = parse_shader(source).unwrap();
    assert_eq!(module.structs.len(), 2);
    assert_eq!(module.structs[1].fields.len(), 2);
    assert_eq!(module.structs[1].fields[0].name, "vertexBuffer");
    // Module-scope macro declaration also desugars to a resource.
    let source = "struct T { float x; }; REGULAR_BUFFER(T) g_buf; void foo() {}";
    let module = parse_shader(source).unwrap();
    assert_eq!(module.resources.len(), 1);
    assert_eq!(module.resources[0].name, "g_buf");
}

#[test]
fn parses_multi_arg_resource_template() {
    // Regression: `RW_RegularBuffer<uint, CacheFlags::kGL2Only>` failed
    // with `expected `>`, found `,`` (AGC mesh shaders).
    let source = "struct Srt { RegularBuffer<uint> a; RW_RegularBuffer<unsigned int, CacheFlags::kGL2Only> b; Texture2D<float4> t; };";
    let module = parse_shader(source).unwrap();
    assert_eq!(module.structs.len(), 1);
    assert_eq!(module.structs[0].fields.len(), 3);
    assert_eq!(module.structs[0].fields[1].name, "b");
    // Function returning a multi-arg resource template.
    let source = "template <typename T, ulong cacheFlags> RW_RegularBuffer<T, cacheFlags> makeRwRegularBuffer(ulong address, uint elemCount) { return RW_RegularBuffer<T, cacheFlags>(); }";
    let module = parse_shader(source).unwrap();
    assert_eq!(module.functions.len(), 1);
    assert_eq!(module.functions[0].name, "makeRwRegularBuffer");
}

#[test]
fn parses_nested_resource_template() {
    // Regression: `RegularBuffer<Embedded<X>>* p;` failed with
    // `expected `>`, found `*`` (the inner `>>` starved the outer list).
    let source = "struct X { uint a; }; struct S { const RegularBuffer<X> a; RegularBuffer<Pack<X>> * b; };";
    let module = parse_shader(source).unwrap();
    assert_eq!(module.structs.len(), 2);
    assert_eq!(module.structs[1].fields.len(), 2);
}

#[test]
fn parses_shift_after_dot() {
    // Regression guard: speculative template skipping must not corrupt
    // shift expressions (`d.u32 >> 31`): skippers never mutate tokens,
    // so index-only rollbacks stay sound.
    let source = "struct D { uint u32; }; void f() { D d; uint32_t s = d.u32 >> 31; uint32_t e = (d.u32 >> 23) & 2u; }";
    let module = parse_shader(source).unwrap();
    assert_eq!(module.functions.len(), 1);
}

#[test]
fn parses_templated_type_brace_init_local() {
    // Regression: `RW_RegularBuffer<T, F> rb { buf };` was read as a
    // `<` comparison (`expected `;`, found `,`).
    let source = "template <typename T, ulong F> RW_RegularBuffer<T, F> mk() { RW_RegularBuffer<T, F> rb { 0 }; return rb; }";
    let module = parse_shader(source).unwrap();
    assert_eq!(module.functions.len(), 1);
}

#[test]
fn unsized_const_array_infers_size_from_braces() {
    // `static const float3 distinctColors[] = { ... }`: the top-level
    // brace-list count becomes the array dimension (DXC rejects `[]`
    // with `array type needs an explicit size`). Constructor commas and
    // a trailing comma are not elements.
    let source = "static const float3 c[] = {\nfloat3(1.0, 1.0, 0.0),\nfloat3(0.1, 0.9, 1.0),\nfloat3(1.0, 1.0, 0.0),\n};\nfloat3 f(uint id) { return c[id % 3]; }\n";
    let module = parse_shader(source).unwrap();
    let g = module.globals.iter().find(|g| g.name == "c").unwrap();
    assert_eq!(g.array_size, Some(3));
}

#[test]
fn parses_static_unsigned_constexpr_struct_member() {
    // Regression: `static unsigned constexpr k = 63;` inside a struct
    // failed with `expected identifier, found `=`` (PSR deferred shaders).
    let source = "struct IndirectArgSrt { static unsigned constexpr kRayTracingDispatchArg = 63; RegularBuffer<unsigned> m_occupancyBuffer; };";
    let module = parse_shader(source).unwrap();
    assert_eq!(module.structs.len(), 1);
    assert_eq!(module.structs[0].fields.len(), 2);
    assert_eq!(module.structs[0].fields[0].name, "kRayTracingDispatchArg");
}

#[test]
fn parses_braced_return_list() {
    // Regression: `return {pos, normal, view, uv};` failed with
    // `expected expression, found `{``.
    let source = "struct Hit { float3 pos; float3 nrm; }; Hit foo(float3 pos, float3 nrm) { return {pos, nrm}; }";
    let module = parse_shader(source).unwrap();
    assert_eq!(module.functions.len(), 1);
}

#[test]
fn parses_post_type_direction() {
    // Regression: `T out to` (direction after the type) failed with
    // `expected identifier, found `out`` (PSR shaders).
    let source = "template <typename T> void setAttenuation(T out to, float3 attenuation, unsigned lit) { to = attenuation; }";
    let module = parse_shader(source).unwrap();
    assert_eq!(module.functions.len(), 1);
    assert_eq!(module.functions[0].parameters.len(), 3);
}

#[test]
fn parses_module_paren_init_vs_function() {
    // Regression: `static constexpr uint2 k(8, 8);` failed with
    // `expected type, found number literal` (PSR denoiser shaders).
    let source = "namespace Shared { static constexpr uint2 copyForVisualizeGroupSize(8, 8); }";
    let module = parse_shader(source).unwrap();
    assert_eq!(module.globals.len(), 1);
    assert_eq!(module.globals[0].name, "copyForVisualizeGroupSize");
    // Real functions (incl. `void`-param and empty lists) still parse.
    let source = "uint foo(uint x, float y) { return x; } void bar(); void baz(void);";
    let module = parse_shader(source).unwrap();
    assert_eq!(module.functions.len(), 3);
}

#[test]
fn parses_module_bare_brace_init() {
    // Regression: `static const float3 k{...};` failed with
    // `expected type, found `{`` (tonemap shaders).
    let source = "typedef float3 Vector3Unaligned; static const Vector3Unaligned kLuminanceVector{ 0.2125f, 0.7154f, 0.0721f }; static const float kMiddleGray = 0.72f;";
    let module = parse_shader(source).unwrap();
    assert_eq!(module.globals.len(), 2);
}

#[test]
fn parses_comma_separated_struct_fields() {
    // Regression: `uint32_t index0, index1, index2;` failed with
    // `expected `;`, found `,`` (PSR denoiser shaders).
    let source = "struct T { uint32_t index0, index1, index2; float3 u, v, w; };";
    let module = parse_shader(source).unwrap();
    assert_eq!(module.structs.len(), 1);
    assert_eq!(module.structs[0].fields.len(), 6);
}

#[test]
fn parses_shared_namespace_types() {
    // Regression: the `Shared` framework namespace collided with the `shared`
    // interpolation-modifier skip (`expected type, found `:`).
    let source = "namespace Shared { struct SRT { int x; }; } void fitting_generic(Shared::SRT srt, int2 groupIdx) { }";
    let module = parse_shader(source).unwrap();
    assert_eq!(module.functions.len(), 1);
    assert_eq!(module.functions[0].parameters.len(), 2);
    // A real modifier before a scoped type still applies.
    let source = "namespace Shared { struct SRT { int x; }; } struct V2P { sample Shared::SRT s : TEXCOORD0; };";
    let module = parse_shader(source).unwrap();
    assert_eq!(module.structs[1].fields.len(), 1);
}

#[test]
fn parses_local_typedef() {
    // Regression: function-body `typedef Toolkit::...<...> Alias;`
    // failed with `expected `;`, found `:`.
    let source = "struct B { int x; }; void foo() { typedef B Alias; Alias a; a.x = 1; }";
    let module = parse_shader(source).unwrap();
    assert_eq!(module.functions.len(), 1);
}

#[test]
fn parses_pssl_semantic_parameters() {
    // Regression: `SRT srt PSSL_SEMANTIC(S_SRT_DATA)` params broke the
    // whole function (`expected `;`, found `[` downstream).
    let source = "struct SRT { int x; }; void f0(SRT srt PSSL_SEMANTIC(S_SRT_DATA), int2 groupIdx PSSL_SEMANTIC(S_GROUP_ID)) { g(srt); }";
    let module = parse_shader(source).unwrap();
    assert_eq!(module.functions.len(), 1);
    assert_eq!(module.functions[0].parameters.len(), 2);
}

#[test]
fn parses_stray_module_paren_expression() {
    // Regression: a bare `((a == 1) || (b == 2))` at module scope
    // (expanded macro fragment) failed with `expected type, found `(`.
    let source = "((1 == 1) || (2 == 2)); void foo() { }";
    let module = parse_shader(source).unwrap();
    assert_eq!(module.functions.len(), 1);
    assert!(module.globals.is_empty());
}

#[test]
fn parses_do_while() {
    // Regression: `do { ... } while (...);` failed with
    // `expected expression, found `const`` (hdr_workflow shaders).
    // Desugared to `while (true) { body; if (!cond) break; }`.
    let source = "void f(uint n) { uint s = 0; do { const uint x = s + n; s = x; } while (s < 10); }";
    let module = parse_shader(source).unwrap();
    assert_eq!(module.functions.len(), 1);
    assert!(matches!(module.functions[0].body.last(), Some(crate::ast::Stmt::While { .. })));
}

#[test]
fn parses_nointerp_modifier() {
    // Regression: PSSL `nointerp uint x : SEM;` failed with
    // `expected identifier, found `uint`` (asset_streaming shaders).
    // `custominterp` (taa shaders) rides the same path.
    let source = "struct V { float4 p : S_POSITION; nointerp uint entityIndex : TEXCOORD4; custominterp float2 uv : UV; };";
    let module = parse_shader(source).unwrap();
    assert_eq!(module.structs.len(), 1);
    assert_eq!(module.structs[0].fields.len(), 3);
    assert_eq!(module.structs[0].fields[1].name, "entityIndex");
}

#[test]
fn parses_anonymous_struct_pointer_member() {
    // Regression: `struct { ... }* transforms;` failed with
    // `expected type, found `*`` (lofsr/vr SRT structs).
    let source = "struct T { uint x; }; struct Srt { struct { T transforms[2]; }* transforms; T* textures; };";
    let module = parse_shader(source).unwrap();
    // T + Srt + synthesized Srt__anon0.
    assert_eq!(module.structs.len(), 3);
    let srt = module.structs.iter().find(|s| s.name == "Srt").unwrap();
    assert_eq!(srt.fields.len(), 2);
    assert_eq!(srt.fields[0].name, "transforms");
    // Anonymous member without declarator promotes its fields.
    let source = "struct Outer { uint a; struct { uint b; uint c; }; uint d; };";
    let module = parse_shader(source).unwrap();
    let outer = module.structs.iter().find(|s| s.name == "Outer").unwrap();
    assert_eq!(outer.fields.len(), 4);
}

#[test]
fn parses_elaborated_struct_use() {
    // Regression: `struct MeshData* mesh;` failed with
    // `expected `{`, found `*`` (alpha_tested_geometry SRT structs).
    let source = "struct MeshData { uint x; }; struct T { struct MeshData* mesh; struct MaterialData* material; };";
    let module = parse_shader(source).unwrap();
    let t = module.structs.iter().find(|s| s.name == "T").unwrap();
    assert_eq!(t.fields.len(), 2);
    assert_eq!(t.fields[0].name, "mesh");
}

#[test]
fn parses_legacy_constant_register() {
    // Regression: `register(c0)` failed with `invalid register 'c0'`
    // (font/gnmp samples).
    let source = "struct CameraLightingData { float4 c; }; ConstantBuffer SharedConstantBuffer : register(c0) { CameraLightingData cb_data; };";
    let module = parse_shader(source).unwrap();
    assert_eq!(module.resources.len(), 1);
    assert_eq!(module.resources[0].register, Some(('c', 0)));
}

#[test]
fn parses_pointer_typedefs() {
    // Regression: `typedef vec_double1* vec_double1_out;` failed with
    // `expected identifier, found `*`` (post_processing shaders).
    let source = "struct vec_double1 { double x; }; typedef vec_double1* vec_double1_out; typedef float FArr[4]; void foo() { vec_double1_out o; }";
    let module = parse_shader(source).unwrap();
    assert_eq!(module.functions.len(), 1);
}

#[test]
fn parses_unknown_prefix_macro() {
    // Regression: `SCE_VECTORMATH_ALWAYS_INLINE vec_ullong1 f(...)`
    // misaligned the declaration (`expected type, found `{`` downstream).
    let source = "SCE_VECTORMATH_ALWAYS_INLINE uint sce_f(uint a) { return a; }";
    let module = parse_shader(source).unwrap();
    assert_eq!(module.functions.len(), 1);
    assert_eq!(module.functions[0].name, "sce_f");
}

#[test]
fn parses_typedef_with_const_and_callconv() {
    // Regression: `typedef const floatInVec SCE_..._ARG floatInVec_arg;`
    // failed with `expected type, found `const``.
    let source = "struct floatInVec { float x; }; typedef const floatInVec SCE_VECTORMATH_SIMD_AOS_FLOATINVEC_ARG floatInVec_arg; void foo() { floatInVec_arg a; }";
    let module = parse_shader(source).unwrap();
    assert_eq!(module.functions.len(), 1);
}

#[test]
fn parses_operator_overloads() {
    // Regression: `Ret operator * (...);` failed with
    // `expected identifier, found `(` (vectormath headers).
    let source = "struct V { float x; }; V operator * (V a, V b); V operator + (V a, V b) { return a; } void foo() { V a; V b; V c = a; }";
    let module = parse_shader(source).unwrap();
    // Operator overloads are skipped (never entry points); only foo remains.
    assert_eq!(module.functions.len(), 1);
    assert_eq!(module.functions[0].name, "foo");
}

#[test]
fn parses_ctor_init_lists() {
    // Regression: `C::C(...) : mData(...) { }` failed with
    // `expected `{`, found `(`.
    let source = "struct C { uint m; }; C::C(uint v) : m(v) { } void foo() { C c; }";
    let module = parse_shader(source).unwrap();
    assert!(module.functions.iter().any(|f| f.name == "C"));
}

#[test]
fn parses_method_const_and_conversion_ops() {
    // Regression: `bool C::get() const` failed with
    // `expected `{`, found `const``; `C::operator bool()` needed the
    // conversion-target consumption.
    let source = "struct C { uint m; }; bool C::get() const { return true; } C::operator bool() const { return get(); }";
    let module = parse_shader(source).unwrap();
    assert!(module.functions.iter().any(|f| f.name == "get"));
}

#[test]
fn parses_scoped_unary_operator_const() {
    // Regression: `const B B::operator !() const` failed with
    // `expected `;`, found `const`` (vectormath headers).
    let source = "struct B { uint m; }; const B B::operator !() const { B r; return r; }";
    let module = parse_shader(source).unwrap();
    assert!(module.functions.is_empty());
}

#[test]
fn parses_in_struct_operators() {
    // Regression: `T &operator[](int idx) {...}` failed with
    // `expected `;`, found `(` (BlockReduce TempStorage).
    let source = "struct S { uint m[4]; TGet get; S &operator[](int idx) { return get; } operator float() const; };";
    let module = parse_shader(source).unwrap();
    assert_eq!(module.structs.len(), 1);
}

#[test]
fn parses_nested_template_params() {
    // Regression: `V2P<PackVertex<true, true, 1>> input` failed with
    // `expected `;`, found end of file` (nested `>>` confused the
    // paren-init lookahead).
    let source = "struct V2P { float4 p; }; float4 main(V2P<PackVertex<true, true, 1>> input) { return input.p; }";
    let module = parse_shader(source).unwrap();
    assert_eq!(module.functions.len(), 1);
    assert_eq!(module.functions[0].parameters.len(), 1);
}

#[test]
fn parses_gs_primitive_params() {
    // Regression: `Point ES_OUTPUT input[1]` failed with
    // `expected `)`, found identifier` (geometry shaders).
    let source = "struct ES_OUTPUT { uint x; }; void main(Point ES_OUTPUT input[1]) { }";
    let module = parse_shader(source).unwrap();
    assert_eq!(module.functions.len(), 1);
    assert_eq!(module.functions[0].parameters.len(), 1);
    assert_eq!(module.functions[0].parameters[0].name, "input");
    // With a direction qualifier (`in Point Vertex input[1]`).
    let source = "struct Vertex { float4 p; }; void main(in Point Vertex input[1]) { }";
    let module = parse_shader(source).unwrap();
    assert_eq!(module.functions[0].parameters.len(), 1);
}

#[test]
fn parses_unnamed_params() {
    // Regression: `void PSR_VALIDATE(bool cond, string, T0)` failed
    // with `expected `;`, found `[` downstream (debug helpers).
    let source = "template <typename T0> void PSR_VALIDATE(bool cond, string, T0) { }";
    let module = parse_shader(source).unwrap();
    assert_eq!(module.functions.len(), 1);
    assert_eq!(module.functions[0].parameters.len(), 3);
}

#[test]
fn parses_keyword_param_name() {
    // Regression: `void main(PrimaryPayload inout)` failed with
    // `expected `;`, found end of file` (anyhit shaders use `inout`
    // as the payload parameter's name).
    let source = "struct PrimaryPayload { uint x; }; void main(PrimaryPayload inout) { }";
    let module = parse_shader(source).unwrap();
    assert_eq!(module.functions.len(), 1);
    assert_eq!(module.functions[0].parameters.len(), 1);
    assert_eq!(module.functions[0].parameters[0].name, "inout");
}

#[test]
fn parses_empty_for_clauses() {
    // Regression: `for (;;)` failed with `expected `;`, found `)``
    // (the empty init consumed its own terminator).
    let source = "void f() { for (;;) { break; } for (; true;) { break; } }";
    let module = parse_shader(source).unwrap();
    assert_eq!(module.functions.len(), 1);
}

#[test]
fn parses_type_matching_modifier_name() {
    // Regression: `Sample get_sample(...)` failed with
    // `expected identifier, found `(` (`Sample` was eaten as the
    // `sample` interpolation modifier).
    let source = "struct Sample { float x; }; Sample get_sample(int pos) { Sample s; return s; }";
    let module = parse_shader(source).unwrap();
    assert_eq!(module.functions.len(), 1);
    assert_eq!(module.functions[0].name, "get_sample");
}

#[test]
fn parses_templated_const_local() {
    // Regression: `RegularBuffer<uint3> const indexBuffer(...);`
    // failed with `expected expression, found `const`` (the declaration
    // lookahead stopped at the template close).
    let source = "void f() { RegularBuffer<uint3> const indexBuffer; }";
    let module = parse_shader(source).unwrap();
    assert_eq!(module.functions.len(), 1);
}

#[test]
fn parses_structured_bindings() {
    // Regression: `auto [org, dir] = ...;` failed with
    // `expected `]`, found `,`` (the indexed-assignment guard in
    // declaration lookahead misfired on `auto [...]`).
    let source = "void f() { auto [org, dir] = get(); }";
    let module = parse_shader(source).unwrap();
    assert_eq!(module.functions.len(), 1);
}

#[test]
fn parses_double_const_pointer() {
    // Regression: `T const * const p = ...;` failed with
    // `expected `;`, found `const`` (the declaration lookahead stopped
    // at the first `const *`).
    let source = "struct T { uint x; }; void f() { T const * const p = get(); }";
    let module = parse_shader(source).unwrap();
    assert_eq!(module.functions.len(), 1);
}

#[test]
fn parses_array_constructor_expr() {
    // Regression: `float3 colors[] = float3[](...);` failed with
    // `expected expression, found `]``.
    let source = "void f() { float3 colors[] = float3[](float3(1, 0, 0), float3(0, 1, 0)); }";
    let module = parse_shader(source).unwrap();
    assert_eq!(module.functions.len(), 1);
}

#[test]
fn parses_short_suffix_literals() {
    // Regression: `(128us >> numBits)` failed with
    // `expected `)`, found identifier` (`128us` lexed as `128u` + `s`).
    let source = "void f(uint numBits) { uint q = (128us >> numBits); uint s = 4s + 1; }";
    let module = parse_shader(source).unwrap();
    assert_eq!(module.functions.len(), 1);
}

#[test]
fn parses_trailing_comma_calls() {
    // Regression: `uint2(a, b, )` failed with
    // `expected expression, found `)``.
    let source = "void f() { uint2 v = uint2(1, 2, ); foo(v, ); }";
    let module = parse_shader(source).unwrap();
    assert_eq!(module.functions.len(), 1);
}

#[test]
fn parses_comma_expression_stmts() {
    // Regression: `a[2] = X, a[3] = Y;` failed with
    // `expected `;`, found `,``.
    let source = "void f() { uint a[4]; a[2] = 1, a[3] = 2; }";
    let module = parse_shader(source).unwrap();
    assert_eq!(module.functions.len(), 1);
}

#[test]
fn parses_anonymous_enum_instance() {
    // Regression: `enum { ... } state;` failed with
    // `expected identifier, found `;``.
    let source = "struct T { uint x; enum { kInit, kClose, kOpen } state; T t; };";
    let module = parse_shader(source).unwrap();
    let t = module.structs.iter().find(|s| s.name == "T").unwrap();
    assert_eq!(t.fields.len(), 3);
    assert_eq!(t.fields[1].name, "state");
}

#[test]
fn parses_const_struct_instance() {
    // Regression: `struct S {...} const st = {...};` failed with
    // `expected identifier, found `=``.
    let source = "struct T { uint x; }; void f() { struct S { T a; T b; } const st = { getA(), getB() }; }";
    let module = parse_shader(source).unwrap();
    assert_eq!(module.functions.len(), 1);
}

#[test]
fn parses_struct_then_scoped_decl() {
    // Regression: `struct T {...}; sce::Psr::F const flags = ...;` ate
    // `sce` as a struct instance (`expected expression, found `:`).
    let source = "namespace sce { namespace Psr { struct F { uint x; }; } } void f() { struct T { uint a; }; sce::Psr::F const flags = get(); }";
    let module = parse_shader(source).unwrap();
    assert_eq!(module.functions.len(), 1);
}

#[test]
fn parses_macro_after_attributes() {
    // Regression: `[profile(..)] SCE_MACRO(..)` failed with
    // `expected identifier, found `(` (macro check ran before attrs).
    let source = "[profile(\"cs\")] SCE_PSR_WITH_TRINITY_SUPPORT([argument(target=trinity)])\nvoid f(uint idx) { }";
    let module = parse_shader(source).unwrap();
    assert_eq!(module.functions.len(), 1);
}

#[test]
fn parses_paren_comma_expr() {
    // Regression: `(a[i], b[j])` failed with `expected `)`, found `,``.
    let source = "void f() { float2 uv = (a[0], b[1]); }";
    let module = parse_shader(source).unwrap();
    assert_eq!(module.functions.len(), 1);
}

#[test]
fn parses_interposed_param_macro() {
    // Regression: `const MaterialGeom SCE_GEOMETRY_ARG target` failed
    // with `expected `)`, found identifier`.
    let source = "struct M { uint x; }; int hit(M s, const M SCE_GEOMETRY_ARG target) { return 1; }";
    let module = parse_shader(source).unwrap();
    assert_eq!(module.functions.len(), 1);
    assert_eq!(module.functions[0].parameters.len(), 2);
}

#[test]
fn parses_typename_typedef() {
    // Regression: `typedef typename ShuffleType<R>::type type;`
    // failed with `expected `;`, found `<``.
    let source = "template <typename R> struct ShuffleType { uint x; }; void f() { typedef typename ShuffleType<uint>::type type; type t; }";
    let module = parse_shader(source).unwrap();
    assert_eq!(module.functions.len(), 1);
}

#[test]
fn parses_binary_literals() {
    // Regression: `(0b100000 << 0)` failed with
    // `expected `)`, found identifier` (`0b100000` lexed as `0` + id).
    let source = "void f() { uint q = (0b100000 << 0) | (0B1010 >> 1); }";
    let module = parse_shader(source).unwrap();
    assert_eq!(module.functions.len(), 1);
}

#[test]
fn parses_multidim_array_params() {
    // Regression: `short3 endpoints[3][2]` failed with
    // `expected `)`, found `[``.
    let source = "uint4 encodeMode0(short3 endpoints[3][2]) { return uint4(0, 0, 0, 0); }";
    let module = parse_shader(source).unwrap();
    assert_eq!(module.functions.len(), 1);
    assert_eq!(module.functions[0].parameters.len(), 1);
}

#[test]
fn parses_arg_macro_params() {
    // Regression: `PSSL_ARG_OUT(float4, x) PSSL_SEMANTIC(...)` failed
    // (PSR entry-point output params).
    let source = "void main(PSSL_ARG_OUT(float4, gBuffer) PSSL_SEMANTIC(S_TARGET_OUTPUT0)) { gBuffer = float4(0, 0, 0, 0); }";
    let module = parse_shader(source).unwrap();
    assert_eq!(module.functions.len(), 1);
    assert_eq!(module.functions[0].parameters.len(), 1);
    assert_eq!(module.functions[0].parameters[0].name, "gBuffer");
}

#[test]
fn parses_template_after_attribute() {
    // Regression: `[attr] template <typename T0, typename T1> void f(...)`
    // lost its type params (only the loop-top template handler
    // recorded them).
    let source = "[call_site_debugloc] template <typename T0, typename T1> void PSR_VALIDATE(bool cond, string, T0, T1) { }";
    let module = parse_shader(source).unwrap();
    assert_eq!(module.functions.len(), 1);
    assert_eq!(module.functions[0].parameters.len(), 4);
}

#[test]
fn parses_shader_semantic_macro() {
    // Regression: `Varying vs_main(...) SHADER_SEMANTIC(S_POSITION)`
    // failed (any `*_SEMANTIC(...)` now works in param/return/field).
    let source = "struct Varying { float4 p; }; Varying vs_main(uint vID SHADER_SEMANTIC(S_VERTEX_ID)) SHADER_SEMANTIC(S_POSITION) { Varying v; return v; }";
    let module = parse_shader(source).unwrap();
    assert_eq!(module.functions.len(), 1);
    assert_eq!(module.functions[0].parameters.len(), 1);
}

#[test]
fn parses_typename_dependent_local() {
    // Regression: `typename conditional<...>::type traverser;` failed
    // with `expected `;`, found `<``.
    let source = "struct T { uint x; }; void foo() { typename conditional<true, T, T>::type traverser; }";
    let module = parse_shader(source).unwrap();
    assert_eq!(module.functions.len(), 1);
}

#[test]
fn parses_extern_const_global() {
    // Regression: `extern const uint32_t k[64];` failed with
    // `expected type, found `const`` (texture_encode shaders).
    let source = "extern const uint kDefaultPartitionOrder2[64]; void foo() { }";
    let module = parse_shader(source).unwrap();
    assert_eq!(module.globals.len(), 1);
    assert_eq!(module.globals[0].name, "kDefaultPartitionOrder2");
    // A `Shared::` scope must not be eaten as a storage modifier.
    let source = "namespace Shared { struct SRT { int x; }; } Shared::SRT g_srt; void foo() { }";
    let module = parse_shader(source).unwrap();
    assert_eq!(module.globals.len(), 1);
    assert_eq!(module.globals[0].name, "g_srt");
}

#[test]
fn parses_brace_member_init_list() {
    // Regression: `: x{vx}, y{vy} {}` failed with
    // `expected type, found `,`` (vectormath ctors).
    let source = "struct S { S(float vx, float vy) : x{vx}, y{vy} {} float x; float y; };";
    let module = parse_shader(source).unwrap();
    assert_eq!(module.structs.len(), 1);
}

#[test]
fn parses_resource_with_semantic() {
    // Regression: `TextureCube<float3> env : S_SRT_DATA;` failed with
    // `expected register, found `;``.
    let source = "TextureCube<float3> env_tx : S_SRT_DATA; void foo() { }";
    let module = parse_shader(source).unwrap();
    assert_eq!(module.resources.len(), 1);
    assert_eq!(module.resources[0].name, "env_tx");
}

#[test]
fn parses_alignment_macro_after_brace() {
    // Regression: `} SCE_VECTORMATH_POSTALIGN_16;` failed with
    // `expected identifier, found `;``.
    let source = "struct X { uint a; } SCE_VECTORMATH_POSTALIGN_16; void foo() { }";
    let module = parse_shader(source).unwrap();
    assert_eq!(module.structs.len(), 1);
    assert_eq!(module.functions.len(), 1);
    assert!(module.globals.is_empty());
}

#[test]
fn parses_keyword_typedef_alias() {
    // Regression: `typedef unsigned int uint;` failed with
    // `expected identifier, found `uint``.
    let source = "typedef unsigned int uint; void foo() { uint x = 1; }";
    let module = parse_shader(source).unwrap();
    assert_eq!(module.functions.len(), 1);
}

#[test]
fn parses_stray_semicolons() {
    // Regression: `int value[3]={...};;` failed with
    // `expected type, found `;``.
    let source = "struct T { int value[3] = {0, 0, 0};; }; uint k;; void foo() { }";
    let module = parse_shader(source).unwrap();
    assert_eq!(module.structs.len(), 1);
    assert_eq!(module.functions.len(), 1);
}

#[test]
fn parses_stray_backslash() {
    // Regression: `static_assert(\ + 0 == 128, ...)` failed with
    // `unrecognized token "\\ "` (low_level PSR headers).
    let source = "struct B { uint x; }; static_assert(\\ + 0 == 128, \"bad\"); void foo() { }";
    let module = parse_shader(source).unwrap();
    assert_eq!(module.functions.len(), 1);
}

#[test]
fn parses_align_macro_before_name() {
    // Regression: `struct RT_ALIGNAS(64) S64Byte {...}` failed with
    // `expected `{`, found `(`.
    let source = "struct RT_ALIGNAS(64) S64Byte { uint dummy[16]; }; void foo() { }";
    let module = parse_shader(source).unwrap();
    assert_eq!(module.structs.len(), 1);
    assert_eq!(module.structs[0].name, "S64Byte");
}

#[test]
fn parses_module_union() {
    // Regression: `union RT_ALIGNAS(64) TriangleFanNode {...}` failed
    // with `expected identifier, found `{``.
    let source = "union RT_ALIGNAS(64) TriangleFanNode { uint a; float b; }; void foo() { }";
    let module = parse_shader(source).unwrap();
    assert_eq!(module.functions.len(), 1);
}

#[test]
fn parses_templated_union() {
    // Regression: `union UnifiedResult;` + partial
    // `template <...> union UnifiedResult<A, B> {...}` failed with
    // `expected `;`, found `<``.
    let source = "union UnifiedResult; template <typename T> union UnifiedResult<T> { uint a; }; void foo() { }";
    let module = parse_shader(source).unwrap();
    assert_eq!(module.functions.len(), 1);
}

#[test]
fn parses_struct_inheritance() {
    // Regression: `struct VcoStatsSRT : BasicDrawSRT` failed with
    // `expected `{`, found `:` (blendshape_decoder shaders).
    // Bases flatten: derived sees base fields first (C++ layout order),
    // own declarations win on name clash; unknown bases stay loud.
    let source = "struct Base { uint a; }; struct Derived : Base { uint b; }; struct Multi : public Base, protected Base { uint c; }; struct Orphan : Missing { uint d; };";
    let module = parse_shader(source).unwrap();
    assert_eq!(module.structs.len(), 4);
    let names = |i: usize| {
        module.structs[i]
            .fields
            .iter()
            .map(|f| f.name.as_str())
            .collect::<Vec<_>>()
    };
    assert_eq!(names(1), vec!["a", "b"]);
    assert_eq!(names(2), vec!["a", "c"]);
    assert_eq!(names(3), vec!["d"]);
}

#[test]
fn parses_member_templates() {
    // Regression: `template <typename T> static T getInvalidValue();`
    // inside a struct failed with `expected identifier, found `static``.
    let source = "struct Min { template <typename T> static T getInvalidValue(); template <typename T> static T CrossLaneFunc(T value) { return value; } };";
    let module = parse_shader(source).unwrap();
    assert_eq!(module.structs.len(), 1);
}

#[test]
fn parses_attribute_before_class() {
    // Regression: `[ThisAsPointer] class PCG32RNG { ... }` with an attribute
    // block preceding the `class` keyword failed with
    // `expected identifier, found `{`` (psr_denoiser shaders sharing the
    // `PCG32RNG` header). The loop-top `class`/`union`/`enum`/`namespace`
    // handlers re-dispatch only when the keyword opens a loop iteration, so
    // a leading `[attr]` block stranded the `class` keyword into the
    // declaration path.
    let source = "[ThisAsPointer] class PCG32RNG { uint64_t state; public: PCG32RNG() {} }; void foo() { }";
    let module = parse_shader(source).unwrap();
    assert_eq!(module.functions.len(), 1);
}

#[test]
fn parses_attribute_before_union_and_enum() {
    // Same attribute-prefix dispatch for `union` and `enum` aggregates.
    let source = "[x] union U { uint a; float b; }; [y] enum class E { A, B }; void foo() { }";
    let module = parse_shader(source).unwrap();
    assert_eq!(module.functions.len(), 1);
}

#[test]
fn builtin_constructor_resolves_to_vector_type() {
    // Regression: `float3(1.0)` built `Constructor { ty: Named("float3") }`
    // instead of `Vector`, failing later binary-op checks with the
    // confusing `expected float3, found float3`.
    let source = "float3 ReinhardTonemap(float3 orig) { return orig / (float3(1.0) + orig); }";
    let module = parse_shader(source).unwrap();
    let body = &module.functions[0].body;
    let Stmt::Return(Some(Expr::Binary {
        left: _dividend,
        op: BinaryOp::Div,
        right: divisor,
    })) = &body[body.len() - 1]
    else {
        panic!("expected trailing `return a / b`");
    };
    let Expr::Binary {
        left: ctor,
        op: BinaryOp::Add,
        ..
    } = divisor.as_ref()
    else {
        panic!("expected `(float3(1.0) + orig)` divisor, got {divisor:?}");
    };
    let Expr::Constructor { ty, .. } = ctor.as_ref() else {
        panic!("expected `float3(1.0)` constructor, got {ctor:?}");
    };
    assert_eq!(
        *ty,
        Type::Vector {
            base: ScalarType::Float,
            width: 3
        }
    );
}

#[test]
fn bare_cxx_symbol_marker_attaches_to_next_function() {
    // A bare `CXX_SYMBOL("...")` embedding marker (imgui shaders) must
    // surface as a `CxxSymbol` attribute on the following function.
    let source = "CXX_SYMBOL(\"Shader::imgui_impl_ps_shader_vv\")\nVarying main() { Varying v; return v; }";
    let module = parse_shader(source).unwrap();
    assert_eq!(module.functions.len(), 1);
    let f = &module.functions[0];
    assert_eq!(f.name, "main");
    assert_eq!(f.attributes.len(), 1);
    assert_eq!(f.attributes[0].name, "CxxSymbol");
    assert_eq!(
        f.attributes[0].arguments,
        vec![Expr::String(
            "Shader::imgui_impl_ps_shader_vv".to_string()
        )]
    );
}

#[test]
fn clip_call_is_preserved_as_call() {
    // `clip(x)` is a real intrinsic, not an erasable unknown identifier.
    let source = "void main(float4 c) { clip(c.x); }";
    let module = parse_shader(source).unwrap();
    let body = &module.functions[0].body;
    assert_eq!(body.len(), 1);
    let Stmt::Expr(Expr::Call {
        function,
        arguments,
    }) = &body[0]
    else {
        panic!("expected clip call, got {:?}", body[0]);
    };
    assert_eq!(**function, Expr::Variable("clip".to_string()));
    assert_eq!(arguments.len(), 1);
}

#[test]
fn plain_user_call_is_preserved_as_call() {
    // User-function calls must survive parsing with their argument lists
    // (previously erased to `false`, breaking member access on results).
    let source = "float3 g(float3 x) { return x; } float f(float3 a) { return g(a).x; }";
    let module = parse_shader(source).unwrap();
    let f = module.functions.iter().find(|f| f.name == "f").unwrap();
    assert_eq!(f.body.len(), 1);
    let Stmt::Return(Some(Expr::Member { object, member })) = &f.body[0] else {
        panic!("expected member on call, got {:?}", f.body[0]);
    };
    assert_eq!(member, "x");
    let Expr::Call { function, arguments } = object.as_ref() else {
        panic!("expected call object, got {object:?}");
    };
    assert_eq!(**function, Expr::Variable("g".to_string()));
    assert_eq!(arguments.len(), 1);
}

#[test]
fn comparison_followed_by_shift_is_not_a_template_call() {
    // `if (index < 8) return (fmask.x >> (index*4)) & m;` — the `<` is a
    // comparison even though a `>>` + `(` follow within bounds.
    let source = "int getFptr(int index, int2 fmask) { if (index < 8) return (fmask.x >> (index*4)) & 15; else return 0; }";
    let module = parse_shader(source).unwrap();
    let f = &module.functions[0];
    let Stmt::If { condition, .. } = &f.body[0] else {
        panic!("expected if, got {:?}", f.body[0]);
    };
    let Expr::Binary { left, op, right } = condition.as_ref() else {
        panic!("expected comparison, got {condition:?}");
    };
    assert_eq!(**left, Expr::Variable("index".to_string()));
    assert_eq!(*op, BinaryOp::Less);
    assert_eq!(**right, Expr::Number("8".to_string()));
}

#[test]
fn bit_cast_preserves_target_type() {
    // `bit_cast<short2>(x)` parses as a constructor of the target type.
    let source = "short4 f(uint x) { short4 r; r.xz = bit_cast<short2>(x); return r; }";
    let module = parse_shader(source).unwrap();
    let f = &module.functions[0];
    let Stmt::Expr(Expr::Assign { rhs, .. }) = &f.body[1] else {
        panic!("expected assign, got {:?}", f.body[1]);
    };
    let Expr::Constructor { ty, .. } = rhs.as_ref() else {
        panic!("expected constructor, got {rhs:?}");
    };
    assert_eq!(
        *ty,
        Type::Vector {
            base: ScalarType::Int,
            width: 2
        }
    );
}

#[test]
fn enable_if_member_resolves_to_guarded_type() {
    // `typename enable_if<Cond, float4>::type m;` models the member as `T`.
    let source = "struct P { typename enable_if<true, float4>::type m; }; float4 f(P p) { return p.m; }";
    let module = parse_shader(source).unwrap();
    assert_eq!(
        module.structs[0].fields[0].ty,
        Type::Vector {
            base: ScalarType::Float,
            width: 4
        }
    );
}

#[test]
fn capitalized_union_name_is_not_a_keyword() {
    // A type named `Union` must parse `Union u;` as a declaration — the
    // `union` keyword match is case-sensitive.
    let source = "void f() { union Union { int a; int b; }; Union u; u.a = 1; }";
    let module = parse_shader(source).unwrap();
    let body = &module.functions[0].body;
    assert_eq!(body.len(), 3);
    let Stmt::VarDecl { name, .. } = &body[1] else {
        panic!("expected VarDecl, got {:?}", body[1]);
    };
    assert_eq!(name, "u");
}

#[test]
fn enum_enumerators_recorded_as_int_globals() {
    let source = "enum Options { kOptA = 1 << 0, kOptB };";
    let module = parse_shader(source).unwrap();
    let names: Vec<&str> = module.globals.iter().map(|g| g.name.as_str()).collect();
    assert!(names.contains(&"kOptA"), "globals: {names:?}");
    assert!(names.contains(&"kOptB"), "globals: {names:?}");
}

#[test]
fn parses_comma_separated_shared_globals() {
    // Regression: `thread_group_memory uint a, b, c, d;` (LDS block in
    // wave headers) — the global path only took one declarator
    // (`expected `;`, found `,`).
    let source = "thread_group_memory uint batchID, targetIndex, batchCount, found; void foo() {}";
    let module = parse_shader(source).unwrap();
    assert_eq!(module.globals.len(), 4);
    let names: Vec<&str> = module.globals.iter().map(|g| g.name.as_str()).collect();
    assert_eq!(names, vec!["batchID", "targetIndex", "batchCount", "found"]);
    assert!(
        module.globals.iter().all(|g| g.shared),
        "thread_group_memory globals must be shared"
    );
}

#[test]
fn parses_plain_comma_globals_with_inits() {
    let source = "uint x = 1, y = 2; void foo() {}";
    let module = parse_shader(source).unwrap();
    assert_eq!(module.globals.len(), 2);
    assert!(!module.globals[0].shared);
    assert!(module.constants.contains_key("x"));
    assert!(module.constants.contains_key("y"));
}

#[test]
fn skips_body_using_directive() {
    // `using namespace sce::Got;` inside a body is compile-time only —
    // it must not leak an `sce::Got;` expression statement.
    let source = "void foo() { using namespace sce::Got; float x = 1.0; }";
    let module = parse_shader(source).unwrap();
    assert_eq!(module.functions.len(), 1);
    let mut found_sce = false;
    for s in &module.functions[0].body {
        if let Stmt::Expr(Expr::Variable(v)) = s {
            if v.contains("sce") {
                found_sce = true;
            }
        }
    }
    assert!(!found_sce, "body: {:?}", module.functions[0].body);
}

#[test]
fn resolves_scoped_struct_type_to_declared() {
    // Namespaces flatten on declaration but uses keep scopes.
    let source = "namespace sce { namespace Got { struct Foo { float x; }; } } void bar(sce::Got::Foo v) {}";
    let module = parse_shader(source).unwrap();
    assert_eq!(module.structs.len(), 1);
    assert_eq!(module.functions.len(), 1);
    assert_eq!(
        module.functions[0].parameters[0].ty,
        Type::Named("Foo".to_string())
    );
}

#[test]
fn leaves_undeclared_scopes_untouched() {
    // Framework classes/enums with no declared struct must keep failing
    // loudly downstream, never silently misresolve.
    let source = "void bar(sce::Agc::VrsRate v) { sce::Agc::VrsRate r = sce::Agc::VrsRate::k1x1; }";
    let module = parse_shader(source).unwrap();
    assert_eq!(
        module.functions[0].parameters[0].ty,
        Type::Named("sce::Agc::VrsRate".to_string())
    );
}

#[test]
fn attribute_before_struct_redispatches() {
    // A bracket attribute before an aggregate (`[ThisAsPointer]` from
    // the `RT_PSSL_ONLY(...)` macro) must not derail the struct: the
    // loop re-enters so the struct handler owns the keyword.
    let module =
        parse_shader("[ThisAsPointer]\nstruct Foo { float x; };\n").unwrap();
    assert_eq!(module.structs.len(), 1);
    assert_eq!(module.structs[0].name, "Foo");
}

#[test]
fn scoped_indexed_store_is_assignment_not_destructure() {
    // `Lds::position[g * 4 + i] = v` stores through a flattened namespace
    // array: an expression, not a `Lds::position` structured-binding
    // declaration (the `[` disambiguation must look past `::` chains).
    let source = "namespace Lds { thread_group_memory float3 position[256]; }\n\
         void f(uint g, uint i, float3 v) { Lds::position[g * 4 + i] = v; }";
    let module = parse_shader(source).unwrap();
    let body = &module.functions[0].body;
    assert_eq!(body.len(), 1);
    let Stmt::Expr(Expr::Assign { lhs, .. }) = &body[0] else {
        panic!("expected indexed assignment, got {:?}", body[0]);
    };
    let Expr::Index { object, .. } = lhs.as_ref() else {
        panic!("expected index lhs, got {lhs:?}");
    };
    let Expr::Variable(base) = object.as_ref() else {
        panic!("expected variable base, got {object:?}");
    };
    assert_eq!(base, "position", "scoped use folds to the bare global");
}

#[test]
fn namespace_var_function_collision_qualifies_global() {
    // `namespace Lds { ... uint batchID; }` plus a free `uint batchID()`
    // cannot share the bare name (DXC `redefinition ... different kind of
    // symbol`), and folding `Lds::batchID` to bare would misbind to the
    // `inout` parameter. The global qualifies (`Lds_batchID`); bare uses
    // keep C++ scoping (parameter call).
    let source = "namespace Lds { thread_group_memory uint batchID; }\n\
         uint batchID(uint dispatchThreadID) { return dispatchThreadID; }\n\
         void f(uint dispatchThreadID, inout uint targetIndex) {\n\
         Lds::batchID = 0;\n\
         targetIndex = batchID(dispatchThreadID);\n}";
    let module = parse_shader(source).unwrap();
    assert!(
        module.globals.iter().any(|g| g.name == "Lds_batchID"),
        "global qualifies, got {:?}",
        module.globals.iter().map(|g| &g.name).collect::<Vec<_>>()
    );
    assert!(
        !module.globals.iter().any(|g| g.name == "batchID"),
        "bare name left for the function"
    );
    let body = &module.functions[1].body;
    let Stmt::Expr(Expr::Assign { lhs, .. }) = &body[0] else {
        panic!("expected assignment, got {:?}", body[0]);
    };
    assert!(
        matches!(lhs.as_ref(), Expr::Variable(v) if v == "Lds_batchID"),
        "LDS store binds the qualified global, got {lhs:?}"
    );
    let Stmt::Expr(Expr::Assign { rhs, .. }) = &body[1] else {
        panic!("expected assignment, got {:?}", body[1]);
    };
    assert!(
        matches!(rhs.as_ref(), Expr::Call { function, .. } if matches!(function.as_ref(), Expr::Variable(v) if v == "batchID")),
        "call still targets the free function, got {rhs:?}"
    );
}

#[test]
fn scoped_static_call_folds_to_free_function() {
    // `Ns::f(x)` calls a flattened free function when the bare name has
    // a matching-arity overload; struct heads and arity mismatches stay.
    let source = "struct S { uint m; };\n\
         float umToCm(int um) { return (float)um; }\n\
         float3 umToCm(int3 um) { return (float3)um; }\n\
         float f(S s, int i) { return S::m(s) + umToCm(i); }\n\
         void g() { sce::Got::Blnd::Gpu::umToCm(3); }\n";
    let module = parse_shader(source).unwrap();
    let body = &module.functions[3].body;
    let Stmt::Expr(Expr::Call { function, .. }) = &body[0] else {
        panic!("expected call, got {:?}", body[0]);
    };
    assert!(
        matches!(function.as_ref(), Expr::Variable(v) if v == "umToCm"),
        "scoped call must fold, got {function:?}"
    );
}

#[test]
fn template_empty_head_struct_specialization_parses() {
    // `template <> struct iBVH<false> {...}` (explicit specialization
    // after an attribute): the empty head is consumed and the struct
    // parses under its sanitized specialization name (`iBVH_false`)
    // so it no longer collides with the primary template; the
    // spelling is recorded for use-site matching.
    let module = parse_shader(
        "template <bool isTwoLevel>\nstruct iBVH;\n[ThisAsPointer]\ntemplate <>\nstruct iBVH<false> {\n float x;\n};\n",
    )
    .unwrap();
    assert!(module.structs.iter().any(|s| s.name == "iBVH"));
    assert!(module.structs.iter().any(|s| s.name == "iBVH_false"));
    let specs = module.template_site_spellings.get("@spec").unwrap();
    assert_eq!(specs.get("iBVH<false>").map(String::as_str), Some("iBVH_false"));
}

#[test]
fn template_value_struct_site_spellings_recorded() {
    // `Indices<3> x;` locals and `Indices<2> m;` members preserve
    // spellings per scope for value-template instantiation; params
    // keep flowing through `template_spellings`.
    let source = "struct Outer { Indices<2> m; };\n\
         float f(Indices<3> p) { Indices<3> x; return 1.0; }";
    let module = parse_shader(source).unwrap();
    let site = module.template_site_spellings.get("f").unwrap();
    assert_eq!(site.get("x").map(String::as_str), Some("Indices<3>"));
    let fields = module.template_site_spellings.get("Outer").unwrap();
    assert_eq!(fields.get("m").map(String::as_str), Some("Indices<2>"));
}
