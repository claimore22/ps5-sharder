use ps5_shader::parse_shader;

// Regression tests for the sce framework-mapping campaign, slice 1 (VRS):
// `sce::Agc::VrsRate` is a uint8-backed enum (Framework/target/include/agc/vrs.h)
// whose canonical HW encodings are stored into uint metadata textures.
// It must emit as `uint` with Framework enumerator values — never leak `sce::`
// into HLSL (DXC `use of undeclared identifier 'sce'`).
//
// Companion fix: `switch` statements were parsed-and-dropped
// (`Stmt::Block(vec![])`), silently deleting control flow. They now lower
// to native HLSL `switch`, which has C fallthrough semantics like PSSL.

fn emit(source: &str) -> String {
    let module = parse_shader(source).expect("parser accepted shader");
    let model = ps5_shader::semantic::analyze(&module).expect("semantic analysis succeeded");
    let ir = ps5_shader::lower::lower_module(&module, &model).expect("lowering succeeded");
    ps5_shader::emit::emit_vulkan_hlsl(&ir)
}

#[test]
fn vrs_rate_type_emits_uint_with_Framework_values() {
    let hlsl = emit(
        "RW_Texture2D<uint> vrsTex;\n\
         [NUM_THREADS(8,8,1)]\n\
         void main(uint2 position : S_DISPATCH_THREAD_ID) {\n\
         \tsce::Agc::VrsRate r = sce::Agc::VrsRate::k1x1;\n\
         \tswitch (int(position.x)) {\n\
         \t\tcase 0: r = sce::Agc::VrsRate::k4xSsaa; break;\n\
         \t\tcase 1: r = sce::Agc::VrsRate::k2x2; break;\n\
         \t}\n\
         \tvrsTex[position] = r;\n\
         }",
    );
    assert!(
        hlsl.contains("uint r = 0;"),
        "VrsRate decl must emit as uint k1x1=0, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("r = 15;"),
        "k4xSsaa must emit as 15 (agc/vrs.h), got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("r = 5;"),
        "k2x2 must emit as 5 (agc/vrs.h), got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("switch ("),
        "switch must survive lowering, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("sce::"),
        "no sce::-qualified names may leak into HLSL, got:\n{hlsl}"
    );
}

#[test]
fn framework_vrs_rate_values_match_Framework_header() {
    // Framework/target/include/agc/vrs.h `enum class VrsRate : uint8_t`.
    let cases = [
        ("sce::Agc::VrsRate::k1x1", 0x0),
        ("sce::Agc::VrsRate::k2x1", 0x1),
        ("sce::Agc::VrsRate::k2xSsaa", 0x3),
        ("sce::Agc::VrsRate::k1x2", 0x4),
        ("sce::Agc::VrsRate::k2x2", 0x5),
        ("sce::Agc::VrsRate::k4xSsaa", 0xf),
    ];
    for (path, value) in cases {
        assert_eq!(
            ps5_shader::framework::framework_const(path),
            Some(value),
            "wrong value for {path}"
        );
    }
    assert_eq!(
        ps5_shader::framework::framework_type("sce::Agc::VrsRate"),
        Some("uint")
    );
    assert_eq!(ps5_shader::framework::framework_const("sce::Psr::Ray"), None);
}

#[test]
fn switch_with_default_and_fallthrough_shape() {
    let hlsl = emit(
        "void main(uint i : S_DISPATCH_THREAD_ID) {\n\
         \tuint r = 0;\n\
         \tswitch (i) {\n\
         \t\tcase 0:\n\
         \t\t\tr = 1;\n\
         \t\tcase 1:\n\
         \t\t\tr = 2;\n\
         \t\t\tbreak;\n\
         \t\tdefault:\n\
         \t\t\tr = 3;\n\
         \t\t\tbreak;\n\
         \t}\n\
         }",
    );
    for want in ["switch (i)", "case 0:", "case 1:", "default:", "break;"] {
        assert!(hlsl.contains(want), "missing `{want}`, got:\n{hlsl}");
    }
    // Fallthrough: case 0 body must NOT gain a synthetic break.
    let c0 = hlsl.find("case 0:").expect("case 0 in output");
    let c1 = hlsl.find("case 1:").expect("case 1 in output");
    assert!(
        !hlsl[c0..c1].contains("break;"),
        "fallthrough from case 0 must be preserved, got:\n{hlsl}"
    );
}

#[test]
fn scoped_enum_case_label_resolves_to_literal() {
    let hlsl = emit(
        "void main(uint i : S_DISPATCH_THREAD_ID) {\n\
         \tsce::Agc::VrsRate r = sce::Agc::VrsRate::k1x1;\n\
         \tswitch (i) {\n\
         \t\tcase sce::Agc::VrsRate::k2x1:\n\
         \t\t\tr = sce::Agc::VrsRate::k2x2;\n\
         \t\t\tbreak;\n\
         \t}\n\
         }",
    );
    assert!(
        hlsl.contains("case 1:"),
        "scoped case label must resolve to literal 1, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("sce::"),
        "no sce::-qualified names may leak into HLSL, got:\n{hlsl}"
    );
}
