//! PSSL → Vulkan/HLSL semantic compatibility layer.
//!
//! PSSL shaders use `sce::` framework vocabulary (AGC runtime types,
//! PSR ray-tracing, Gnm constants, …) that has no DXC/HLSL counterpart and
//! must be mapped to legal HLSL at a defined layer — never hacked around
//! per-shader. This module is the single home for those mappings, sourced
//! from the external framework headers and cross-checked against official `.ags`/ISA
//! reference dumps (see `sce_mapping.md`).
//!
//! Rules:
//! * Types map to HLSL types (`framework_type`), used by the emitter.
//! * Enumerators map to integer literals (`framework_const`), resolved in
//!   lowering so every downstream walker sees plain literals.
//! * Resource-like framework objects (descriptors holding GPU resources)
//!   map to IR resource kinds (`framework_resource_kind`), so SRT
//!   flattening and nested-resource hoisting treat them exactly like
//!   their HLSL counterparts — never as cbuffer scalars.
//! * Unknown framework names return `None` and keep failing loudly at DXC
//!   (`use of undeclared identifier 'sce'`): a loud error beats a guessed
//!   constant.

use crate::ir::{IrAccess, IrScalar, IrType};

/// HLSL type for a known `sce::` framework type name.
pub fn framework_type(name: &str) -> Option<&'static str> {
    match name {
        // `enum class VrsRate : uint8_t` (framework VRS header):
        // a uint8-backed shading-rate code stored into uint metadata
        // textures and consumed by VRS hardware. Emits as `uint`.
        "sce::Agc::VrsRate" => Some("uint"),
        // `sce::Gnm::Sampler` is the GNM sampler descriptor (filter/wrap/
        // anisotropy configuration). Shaders build builtin sets from it
        // (`getBuiltinSamplers()` helper: a local
        // `sce::Gnm::Sampler[9]`, per-element mutators, `SamplerState(x)`
        // conversions). Vulkan samplers are immutable and externally
        // bound, so the descriptor lowers to a plain `SamplerState`;
        // configuration mutators are already elided as no-ops.
        "sce::Gnm::Sampler" => Some("SamplerState"),
        // Bare `Sampler` is `sce::Agc::Core::Sampler` under
        // `using namespace sce::Agc::Core` (corpus-wide idiom, e.g.
        // `const Sampler samplerDesc = Sampler().init().set...()` in
        // sampler shadow helpers): the same
        // object as the qualified name, so the same `SamplerState`
        // lowering (builder chains elided, conversions identity).
        "Sampler" => Some("SamplerState"),
        // `sce::Agc::Core::Texture` is the AGC 2D-texture object (e.g.
        // upscaling-pass LUT tables). It is a
        // GPU resource, not a scalar: emits as `Texture2D<float4>` wherever
        // a type name is needed and lowers to a top-level texture binding
        // wherever a resource is needed (see `framework_resource_kind`).
        "sce::Agc::Core::Texture" => Some("Texture2D<float4>"),
        // `sce::Psr::TopLevelBvhDescriptor` is the PSR ray-tracing
        // acceleration-structure descriptor (`m_bvh` in GBUFFER lighting
        // SRTs). The DXR counterpart is `RaytracingAccelerationStructure`.
        "sce::Psr::TopLevelBvhDescriptor" => Some("RaytracingAccelerationStructure"),
        // `sce::Psr::BottomLevelBvhDescriptor` is the bottom-level
        // counterpart (per-geometry BVH input, e.g. `m_instance` in
        // api_psr BVH-build shaders). HLSL/DXR has a single
        // acceleration-structure type covering both levels.
        "sce::Psr::BottomLevelBvhDescriptor" => Some("RaytracingAccelerationStructure"),
        // `sce::Agc::Core::Buffer` is the AGC untyped GPU buffer object
        // (mesh-pipeline `GeometryBuffers::{vertexBuffer,meshletInfo,…}`
        // in image-based-lighting shaders, `m_instance` in api_psr BVH builders).
        // Corpus uses are descriptor carriage (members are hoisted or
        // stripped; no direct indexing observed): the HLSL counterpart
        // is the untyped `ByteAddressBuffer` (explicit `Load<T>`,
        // externally bound).
        "sce::Agc::Core::Buffer" => Some("ByteAddressBuffer"),
        _ => None,
    }
}

/// Whether a type name denotes a sampler object: HLSL `SamplerState`,
/// the framework `sce::Agc::Core::Sampler`, or a framework type mapped
/// to `SamplerState` (see [`framework_type`]). Single home for the
/// sampler set so lowering and emission agree.
pub(crate) fn is_sampler_type_name(name: &str) -> bool {
    name == "SamplerState"
        || name == "sce::Agc::Core::Sampler"
        || framework_type(name) == Some("SamplerState")
}

/// IR resource kind for a resource-like `sce::` framework object name
/// (`None` for non-resources). The kind strings are the same vocabulary
/// the IR uses for HLSL resources (`texture2d`, `byteaddressbuffer`, …),
/// plus `raytracingaccelerationstructure` for the PSR BVH descriptors.
pub(crate) fn framework_resource_kind(name: &str) -> Option<&'static str> {
    match name {
        "sce::Agc::Core::Texture" => Some("texture2d"),
        "sce::Agc::Core::Buffer" => Some("byteaddressbuffer"),
        "sce::Psr::TopLevelBvhDescriptor" | "sce::Psr::BottomLevelBvhDescriptor" => {
            Some("raytracingaccelerationstructure")
        }
        _ => None,
    }
}

/// Synthesized `IrType::Resource` for a resource-like framework object:
/// the single home for the lowering shape so SRT flattening and hoisting
/// agree. Textures are `texture2d` of `float4` (matching the
/// `framework_type` spelling); the BVH descriptor and the untyped buffer
/// carry a `bool` element that is never emitted (acceleration structures
/// and byte-address buffers are untyped in HLSL — see the emitter's
/// kind handling).
pub(crate) fn framework_resource_ir(name: &str) -> Option<IrType> {
    let kind = framework_resource_kind(name)?;
    let element = if kind == "texture2d" {
        IrType::Vector {
            base: IrScalar::Float,
            width: 4,
        }
    } else {
        IrType::Bool
    };
    Some(IrType::Resource {
        kind: kind.to_string(),
        access: IrAccess::ReadOnly,
        element: Box::new(element),
    })
}

/// Value of a known `sce::` framework enumerator, by fully-scoped path.
////// `VrsRate` values are the canonical HW encodings from
/// the framework VRS header (`k1x1 = 0x0`, `k2x1 = 0x1`,
/// `k2xSsaa = 0x3`, `k1x2 = 0x4`, `k2x2 = 0x5`, `k4xSsaa = 0xf`),
/// confirmed against the official ISA for
/// `vrs_generate_rate_texture_c` (switch lowers to selects over
/// `{0, 1, 3, 4, 5, 15}` with default `0`).
pub fn framework_const(path: &str) -> Option<u32> {
    match path {
        "sce::Agc::VrsRate::k1x1" => Some(0x0),
        "sce::Agc::VrsRate::k2x1" => Some(0x1),
        "sce::Agc::VrsRate::k2xSsaa" => Some(0x3),
        "sce::Agc::VrsRate::k1x2" => Some(0x4),
        "sce::Agc::VrsRate::k2x2" => Some(0x5),
        "sce::Agc::VrsRate::k4xSsaa" => Some(0xf),
        // HTILE metadata tile dimensions
        // (framework base-layer header:
        // `static const uint32_t kHtileWidthAndHeight = 8`).
        "sce::Agc::Constants::kHtileWidthAndHeight" => Some(8),
        _ => None,
    }
}

/// Value of a known bare PSSL builtin constant, as HLSL source text.
/// Resolved in lowering only when no variable of that name is in scope
/// (locals, params, and globals shadow it), so shader-side declarations
/// always win.
///
/// `PI`: The PSSL compiler predefines single-precision π (used bare
/// corpus-wide with no sample-side definition; the only PSSL-side
/// declaration is a function-local `const float PI` in
/// one GPU helper header, which shadows via the environment, plus
/// case-differing `Pi`/`pi` and a host-only C++ `constexpr` that never
/// reaches PSSL). Spelling matches the framework's own `M_PI`
/// (`3.1415926536f`).
pub fn framework_float_const(name: &str) -> Option<&'static str> {
    match name {
        "PI" => Some("3.1415926536f"),
        // AMD wave-mode marker constants (`__get_wavemode()` return values):
        // the wave is 64- or 32-wide. The HLSL equivalent (via the
        // `__get_wavemode` → `WaveGetLaneCount` lowering) compares against
        // these literals.
        "__kWaveModeWave64" => Some("64"),
        "__kWaveModeWave32" => Some("32"),
        _ => None,
    }
}

/// Global HLSL variable backing user SGPR `index` (`__read_user_sgpr(i)`).
///
/// User SGPRs are host-provided 32-bit root constants (`#pragma
/// argument(reservedusersgpr=N)` reserves them). In HLSL they lower to
/// plain globals — externally settable, never constants — so validation
/// (DXC + SPIR-V) sees a legal declaration and the host feeds the value
/// at dispatch time, exactly like any other uniform.
pub fn user_sgpr_name(index: u32) -> String {
    format!("__user_sgpr{index}")
}

/// Parses a `__user_sgpr{N}` global name back to `N`.
pub fn user_sgpr_index(name: &str) -> Option<u32> {
    name.strip_prefix("__user_sgpr")?.parse().ok()
}
