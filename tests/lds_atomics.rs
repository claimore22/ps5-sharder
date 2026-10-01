use ps5_shader::parse_shader;

// Regression tests for compute-shader LDS + atomics: `thread_group_memory`
// emits writable `groupshared`, `ThreadGroupMemoryBarrierSync()` maps to
// the SPIR-V spelling, and `AtomicAdd` maps to `InterlockedAdd`
// (all match the Framework's own `#define`s in `cpu_gpu_types.h`).

fn emit(source: &str) -> String {
    let module = parse_shader(source).expect("parser accepted shader");
    let model = ps5_shader::semantic::analyze(&module).expect("semantic analysis succeeded");
    let ir = ps5_shader::lower::lower_module(&module, &model).expect("lowering succeeded");
    ps5_shader::emit::emit_vulkan_hlsl(&ir)
}

#[test]
fn thread_group_memory_emits_groupshared() {
    let hlsl = emit(
        "thread_group_memory uint EV_dist[128];\n\
         [NUM_THREADS(8,8,1)]\n\
         void main(uint i : S_GROUP_THREAD_ID) {\n\
         EV_dist[i] = 0;\n\
         }",
    );
    assert!(
        hlsl.contains("groupshared uint EV_dist[128];"),
        "LDS array must emit groupshared, got:\n{hlsl}"
    );
}

#[test]
fn workgroup_barrier_uses_spirv_spelling() {
    let hlsl = emit(
        "thread_group_memory uint EV_dist[128];\n\
         [NUM_THREADS(8,8,1)]\n\
         void main(uint i : S_GROUP_THREAD_ID) {\n\
         EV_dist[i] = 0;\n\
         ThreadGroupMemoryBarrierSync();\n\
         }",
    );
    assert!(
        hlsl.contains("GroupMemoryBarrierWithGroupSync()"),
        "barrier must use the SPIR-V spelling, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("ThreadGroupMemoryBarrierSync"),
        "PSSL spelling must be gone, got:\n{hlsl}"
    );
}

#[test]
fn atomic_add_maps_both_forms() {
    let hlsl = emit(
        "thread_group_memory uint EV_dist[128];\n\
         RW_ByteBuffer buckets;\n\
         [NUM_THREADS(8,8,1)]\n\
         void main(uint i : S_GROUP_THREAD_ID) {\n\
         AtomicAdd(EV_dist[i], 1);\n\
         buckets.AtomicAdd(i * 4, 2);\n\
         }",
    );
    assert!(
        hlsl.contains("InterlockedAdd(EV_dist[i], 1)"),
        "free AtomicAdd must map, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains(".InterlockedAdd("),
        "method AtomicAdd must map, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("AtomicAdd"),
        "PSSL spelling must be gone, got:\n{hlsl}"
    );
}

#[test]
fn bare_barrier_maps_to_group_memory_barrier() {
    // The non-sync form (`ThreadGroupMemoryBarrier`, no `Sync`) maps to
    // HLSL `GroupMemoryBarrier()` — memory + visibility, no execution sync.
    let hlsl = emit(
        "thread_group_memory uint EV_dist[128];\n\
         [NUM_THREADS(8,8,1)]\n\
         void main(uint i : S_GROUP_THREAD_ID) {\n\
         EV_dist[i] = 0;\n\
         ThreadGroupMemoryBarrier();\n\
         uint x = EV_dist[i];\n\
         }",
    );
    assert!(
        hlsl.contains("GroupMemoryBarrier()"),
        "bare barrier must map to GroupMemoryBarrier(), got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("ThreadGroupMemoryBarrier()"),
        "PSSL spelling must be gone, got:\n{hlsl}"
    );
}

#[test]
fn namespace_lds_scoped_uses_fold_to_bare() {
    // `namespace Lds { thread_group_memory uint batchID; }` flattens to a
    // `groupshared uint batchID;` global; `Lds::batchID` uses fold to the
    // bare identifier.
    let hlsl = emit(
        "namespace Lds { thread_group_memory uint batchID, targetIndex; }\n\
         [NUM_THREADS(8,8,1)]\n\
         void main(uint i : S_GROUP_THREAD_ID) {\n\
         Lds::batchID = 0;\n\
         Lds::targetIndex = i;\n\
         uint x = Lds::batchID + Lds::targetIndex;\n\
         }",
    );
    assert!(
        hlsl.contains("groupshared uint batchID;"),
        "namespace LDS must emit groupshared global, got:\n{hlsl}"
    );
    assert!(
        !hlsl.contains("Lds::"),
        "scoped uses must fold to bare, got:\n{hlsl}"
    );
    assert!(
        hlsl.contains("batchID = 0;"),
        "folded use must be present, got:\n{hlsl}"
    );
}
