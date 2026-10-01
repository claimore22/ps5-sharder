use ps5_shader::parse_shader;

// Regression tests for dead-code elimination in lowering: dead locals
// (materialization residue), dead stores into split members, and
// single-assignment resource aliases. All three remove code that is
// provably unread; passing shaders without such residue are
// byte-identical.

fn emit(source: &str) -> String {
    let module = parse_shader(source).expect("parser accepted shader");
    let model = ps5_shader::semantic::analyze(&module).expect("semantic analysis succeeded");
    let ir = ps5_shader::lower::lower_module(&module, &model).expect("lowering succeeded");
    ps5_shader::emit::emit_vulkan_hlsl(&ir)
}

#[test]
fn dead_struct_local_drops_with_stores() {
    // `s` is never read: the declaration and both member stores drop.
    let hlsl = emit(
        "struct S { float x; float y; };\n\
         float f(float a) { S s; s.x = a; s.y = 2.0; return a; }\n\
         [NUM_THREADS(1,1,1)]\n\
         void main(uint i : S_DISPATCH_THREAD_ID) {\n\
         float z = f(i);\n\
         }",
    );
    assert!(!hlsl.contains("S s;"), "dead local must drop, got:\n{hlsl}");
    assert!(!hlsl.contains("s.x ="), "dead stores must drop, got:\n{hlsl}");
    assert!(hlsl.contains("return a;"), "live code must stay, got:\n{hlsl}");
}

#[test]
fn live_struct_local_survives() {
    // `s` is read through `y`: everything stays.
    let hlsl = emit(
        "struct S { float x; float y; };\n\
         float f(float a) { S s; s.x = a; s.y = 2.0; return s.y; }\n\
         [NUM_THREADS(1,1,1)]\n\
         void main(uint i : S_DISPATCH_THREAD_ID) {\n\
         float z = f(i);\n\
         }",
    );
    assert!(hlsl.contains("S s;"), "live local must stay, got:\n{hlsl}");
    assert!(hlsl.contains("s.y = 2.0"), "live stores must stay, got:\n{hlsl}");
}

#[test]
fn dead_split_member_store_drops_and_strips() {
    // `x.slot` is written from a split buffer but never read: the store
    // drops and the template-nesting member strips from the struct,
    // while the buffer itself still splits (whole-value read keeps the
    // texture member alive to split time).
    let hlsl = emit(
        "struct T { float k; Texture2D<float4> tex; };\n\
         struct S { StructuredBuffer<T> slot; float j; };\n\
         StructuredBuffer<T> buf;\n\
         SamplerState smp;\n\
         float4 f(float2 uv, uint i) { S x; x.slot = buf; x.j = 1.0; T t = buf[i]; return t.tex.Sample(smp, uv) * x.j; }\n\
         [NUM_THREADS(1,1,1)]\n\
         void main(uint i : S_DISPATCH_THREAD_ID) {\n\
         float4 c = f(float2(0.5, 0.5), i);\n\
         }",
    );
    assert!(
        !hlsl.contains("x.slot ="),
        "dead member store must drop, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("StructuredBuffer<T> slot;"),
        "unread split member must strip, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("TScalars"),
        "buffer must still split (twin), got:\n{hlsl}"
    );
}

#[test]
fn resource_alias_erases_to_binding() {
    // `PbrAlias` is stored once from a module texture and only sampled:
    // reads re-root to the binding, the store and global vanish
    // (SPIR-V forbids the store).
    let hlsl = emit(
        "Texture2D<float4> PbrAlias;\n\
         Texture2D<float4> realTex;\n\
         SamplerState smp;\n\
         float4 f(float2 uv) { PbrAlias = realTex; return PbrAlias.Sample(smp, uv); }\n\
         [NUM_THREADS(1,1,1)]\n\
         void main(uint i : S_DISPATCH_THREAD_ID) {\n\
         float4 c = f(float2(0.5, 0.5));\n\
         }",
    );
    assert!(
        !hlsl.contains("PbrAlias"),
        "alias name must vanish entirely, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("realTex.Sample(smp,"),
        "reads must re-root to the binding, got:\n{hlsl}"
    );
}

#[test]
fn multi_store_alias_stays_loud() {
    // Two stores: not a single alias — the global and both stores stay
    // (today's loud path, never mis-resolved).
    let hlsl = emit(
        "Texture2D<float4> PbrAlias;\n\
         Texture2D<float4> realTex;\n\
         Texture2D<float4> otherTex;\n\
         SamplerState smp;\n\
         float4 f(float2 uv, bool c) { if (c) { PbrAlias = realTex; } else { PbrAlias = otherTex; } return PbrAlias.Sample(smp, uv); }\n\
         [NUM_THREADS(1,1,1)]\n\
         void main(uint i : S_DISPATCH_THREAD_ID) {\n\
         float4 c = f(float2(0.5, 0.5), true);\n\
         }",
    );
    assert!(
        hlsl.contains("Texture2D<float4> PbrAlias;"),
        "multi-store alias must stay, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("PbrAlias = realTex;"),
        "stores must stay, got:\n{hlsl}"
    );
}

#[test]
fn out_param_stores_survive() {
    // Stores through `out`/`inout` params are caller-visible and never
    // dead, even when the name is never read in-body: dropping `o = ...`
    // here silently emptied a helper (mesh `main` losing `outPrim`).
    let hlsl = emit(
        "struct Srt { RegularBuffer<uint> prims; uint k; };\n\
         Srt g_srt;\n\
         void helper(out uint o, uint tid) { o = g_srt.prims[tid]; }\n\
         RWBuffer<uint> outBuf;\n\
         [NUM_THREADS(1,1,1)]\n\
         void main(uint i : S_DISPATCH_THREAD_ID) {\n\
         uint o = 0;\n\
         helper(o, i);\n\
         outBuf[0] = o;\n\
         }",
    );
    assert!(
        hlsl.contains("o = g_srt_prims[tid];"),
        "out-param store must survive, got:\n{hlsl}"
    );
}
