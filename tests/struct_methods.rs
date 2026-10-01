use ps5_shader::parse_shader;

// Regression tests for struct member functions (DXC `expected expression`
// via `<complex callee>` on `obj.method()`): methods parse with their
// scope recorded and emit as HLSL struct methods, so calls resolve
// natively. Unknown shapes keep failing loudly.

fn emit(source: &str) -> String {
    let module = parse_shader(source).expect("parser accepted shader");
    let model = ps5_shader::semantic::analyze(&module).expect("semantic analysis succeeded");
    let ir = ps5_shader::lower::lower_module(&module, &model).expect("lowering succeeded");
    ps5_shader::emit::emit_vulkan_hlsl(&ir)
}

const SHADER: &str = "struct XForm {\n\
     float3 offset;\n\
     float3 move(float3 p) const { return p + offset; }\n\
     };\n\
     [NUM_THREADS(1,1,1)]\n\
     void main(uint position : S_DISPATCH_THREAD_ID) {\n\
     XForm t;\n\
     float3 q = t.move(float3(position, 0, 0));\n\
     }\n";

#[test]
fn method_parses_with_struct_scope() {
    let module = parse_shader(SHADER).unwrap();
    let methods: Vec<_> = module
        .functions
        .iter()
        .filter(|f| f.scope.is_some())
        .collect();
    assert_eq!(methods.len(), 1);
    assert_eq!(methods[0].scope.as_deref(), Some("XForm"));
    assert_eq!(methods[0].name, "move");
}

#[test]
fn method_emits_inside_struct_body() {
    let hlsl = emit(SHADER);
    let struct_start = hlsl.find("struct XForm {").expect("struct emitted");
    let method_at = hlsl.find("float3 move(float3 p)").expect("method emitted");
    let struct_end = hlsl[struct_start..].find("};").expect("struct closes");
    assert!(
        method_at > struct_start && method_at < struct_start + struct_end,
        "method must be inside the struct body, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("complex callee"),
        "no complex-callee fallback, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("t.move("),
        "call must pass through, got:\n{hlsl}"
    );
}

#[test]
fn shim_precedes_struct_method_use() {
    // Intrinsic shims (here `M_PI`) must emit before struct definitions:
    // methods can reference them, and HLSL needs declarations before
    // use (psr_fast_trace_ao_primary_lib: `M_PI` inside a struct method
    // failed as undeclared).
    let hlsl = emit(
        "struct Accum { float x; float norm(float t) { return t / M_PI; } };\n\
         [NUM_THREADS(1,1,1)]\n\
         void main(uint position : S_DISPATCH_THREAD_ID) {\n\
         Accum a; a.x = 1.0f;\n\
         float y = a.norm(0.5f);\n\
         }",
    );
    let shim_at = hlsl.find("M_PI = ").expect("M_PI shim emitted");
    let use_at = hlsl.find("/ M_PI)").expect("method use emitted");
    assert!(
        shim_at < use_at,
        "shim must precede struct-method use, got:\n{hlsl}"
    );
}

#[test]
fn statically_called_pure_method_emits_static() {
    // `S::m(...)` calls need a `static` HLSL method (DXC rejects calls
    // to non-static member functions without an object). Pure methods
    // (no instance-state uses) emit `static` and resolve natively;
    // stateful ones keep today's loud path (MeshletSkinData helpers).
    let hlsl = emit(
        "struct Codec {\n\
         uint base;\n\
         uint4 pack(uint input) { return uint4(input & 3u) + 1u; }\n\
         uint get(uint input) { return (input >> 2) + base; }\n\
         };\n\
         [NUM_THREADS(1,1,1)]\n\
         void main(uint position : S_DISPATCH_THREAD_ID) {\n\
         uint4 a = Codec::pack(position);\n\
         Codec c; c.base = 2u;\n\
         uint b = c.get(position);\n\
         }",
    );
    assert!(
        hlsl.contains("static uint4 pack("),
        "pure statically-called method must emit static, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("static uint get("),
        "instance method must not emit static, got:\n{hlsl}"
    );
}

#[test]
fn member_access_uses_this_scope() {
    // The method body references the member bare; it must resolve via
    // the struct scope (semantic member scope), not leak or misresolve.
    let hlsl = emit(SHADER);
    assert!(
        hlsl.contains("return (p + offset);") || hlsl.contains("return p + offset;"),
        "member ref must survive, got:\n{hlsl}"
    );
}

#[test]
fn uncalled_methods_stay_dropped() {
    // Dead methods must not emit (their bodies may reference helpers
    // that break otherwise-passing shaders) nor pollute hoisting.
    let hlsl = emit(
        "struct S { float x; float dead(uint i) { return x + i + undefined_helper(); } };\n\
         [NUM_THREADS(1,1,1)]\n\
         void main(uint position : S_DISPATCH_THREAD_ID) {\n\
         S s; s.x = position;\n\
         }",
    );
    assert!(
        !hlsl.contains("dead("),
        "uncalled method must stay dropped, got:\n{hlsl}"
    );
}

#[test]
fn reversebits_lowercases_to_hlsl() {
    let hlsl = emit(
        "[NUM_THREADS(1,1,1)]\n\
         void main(uint position : S_DISPATCH_THREAD_ID) {\n\
         uint r = ReverseBits(position);\n\
         }",
    );
    assert!(
        hlsl.contains("reversebits(position)"),
        "ReverseBits must map exactly, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("ReverseBits("),
        "PSSL spelling must not leak, got:\n{hlsl}"
    );
}

#[test]
fn hoisted_resource_member_rewrites_inside_methods() {
    // A resource member hoisted out of its struct must also rewrite
    // `this`-relative uses inside struct methods; otherwise the field is
    // stripped while the method still references it (dangling reference
    // that crashes DXC instead of erroring cleanly).
    let hlsl = emit(
        "struct Inner { RegularBuffer<uint> arr; };\n\
         struct Outer { Inner m_g; uint m_i; uint get() { return m_g.arr[m_i]; } };\n\
         [NUM_THREADS(1,1,1)]\n\
         void main(uint position : S_DISPATCH_THREAD_ID) {\n\
         Outer o;\n\
         uint x = o.get();\n\
         }",
    );
    assert!(
        hlsl.contains("__hoist_"),
        "hoisted binding must exist, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("m_g.arr"),
        "no dangling member path may remain, got:\n{hlsl}"
    );
}
