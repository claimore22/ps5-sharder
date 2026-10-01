pub mod hlsl;

/// Emits Vulkan-compatible HLSL from a Shader IR module.
///
/// This is a thin wrapper over [`hlsl::emit_module`] for callers that work
/// with the IR directly.
pub fn emit_vulkan_hlsl(module: &crate::ir::ShaderModule) -> String {
    hlsl::emit_module(module)
}
