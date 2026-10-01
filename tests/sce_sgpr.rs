use ps5_shader::parse_shader;

// Regression tests for user-SGPR reads (`__read_user_sgpr(i)`): host-provided
// 32-bit root constants (`#pragma argument(reservedusersgpr=N)`). They lower
// to plain backing globals (`__user_sgprN`) — externally settable, never
// constants — instead of leaking the intrinsic call into HLSL
// (DXC `use of undeclared identifier '__read_user_sgpr'`).

fn emit(source: &str) -> String {
    let module = parse_shader(source).expect("parser accepted shader");
    let model = ps5_shader::semantic::analyze(&module).expect("semantic analysis succeeded");
    let ir = ps5_shader::lower::lower_module(&module, &model).expect("lowering succeeded");
    ps5_shader::emit::emit_vulkan_hlsl(&ir)
}

#[test]
fn user_sgpr_read_lowes_to_backing_global() {
    // (Consumed through a sink: unread locals drop in lowering.)
    let hlsl = emit(
        "void sink(uint x) {}\n\
         [NUM_THREADS(1,1,1)]\n\
         void main(uint2 position : S_DISPATCH_THREAD_ID) {\n\
         uint frame = __read_user_sgpr(0);\n\
         sink(frame);\n\
         }",
    );
    assert!(
        hlsl.contains("uint __user_sgpr0;"),
        "backing global must be declared, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("uint frame = __user_sgpr0;"),
        "read must use backing global, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("__read_user_sgpr"),
        "intrinsic must not leak into HLSL, got:\n{hlsl}"
    );
}

#[test]
fn multiple_sgpr_indices_each_get_a_global() {
    let hlsl = emit(
        "[NUM_THREADS(1,1,1)]\n\
         void main(uint2 position : S_DISPATCH_THREAD_ID) {\n\
         uint frame = __read_user_sgpr(0);\n\
         uint tri = __read_user_sgpr(1);\n\
         }",
    );
    assert!(
        hlsl.contains("uint __user_sgpr0;"),
        "sgpr0 global must be declared, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("uint __user_sgpr1;"),
        "sgpr1 global must be declared, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("__read_user_sgpr"),
        "intrinsic must not leak into HLSL, got:\n{hlsl}"
    );
}

#[test]
fn framework_user_sgpr_names_roundtrip() {
    assert_eq!(ps5_shader::framework::user_sgpr_name(0), "__user_sgpr0");
    assert_eq!(ps5_shader::framework::user_sgpr_name(7), "__user_sgpr7");
    assert_eq!(ps5_shader::framework::user_sgpr_index("__user_sgpr0"), Some(0));
    assert_eq!(ps5_shader::framework::user_sgpr_index("__user_sgpr12"), Some(12));
    assert_eq!(ps5_shader::framework::user_sgpr_index("frame"), None);
    assert_eq!(ps5_shader::framework::user_sgpr_index("__read_user_sgpr"), None);
}
