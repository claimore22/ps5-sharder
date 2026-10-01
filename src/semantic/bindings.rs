//! Target -> Vulkan semantic mapping layer.
//!
//! This module is intentionally separate from the lexer and parser: the parser
//! only treats semantic names (e.g. `S_POSITION`, `COLOR`) as opaque
//! identifiers, and this layer decides what each PSSL semantic means and how
//! it maps to a Vulkan-compatible HLSL semantic.

use super::Stage;
use crate::ast::Semantic;

/// The Vulkan HLSL semantic string for a PSSL semantic, if one is known.
///
/// Returns `None` when the semantic name is unrecognised (the emitter passes
/// the raw name through in that case, mirroring PSSL behaviour).
pub fn map_semantic(semantic: &Semantic) -> Option<String> {
    let name = semantic.name();
    let upper = name.to_ascii_uppercase();

    match upper.as_str() {
        "S_POSITION" | "SV_POSITION" => return Some("SV_Position".to_string()),
        "COLOR" | "COLOR0" => return Some("TEXCOORD0".to_string()),
        "COLOR1" => return Some("TEXCOORD1".to_string()),
        "COLOR2" => return Some("TEXCOORD2".to_string()),
        "COLOR3" => return Some("TEXCOORD3".to_string()),
        "NORMAL" => return Some("NORMAL".to_string()),
        "SV_VERTEXID" | "S_VERTEX_ID" => return Some("SV_VertexID".to_string()),
        "SV_INSTANCEID" | "S_INSTANCE_ID" => return Some("SV_InstanceID".to_string()),
        "SV_PRIMITIVEID" | "S_PRIMITIVE_ID" => return Some("SV_PrimitiveID".to_string()),
        "SV_DISPATCHTHREADID" | "S_DISPATCH_THREAD_ID" => {
            return Some("SV_DispatchThreadID".to_string());
        }
        "S_GROUP_ID" | "SV_GROUPID" => return Some("SV_GroupID".to_string()),
        "S_GROUP_THREAD_ID" | "SV_GROUPTHREADID" => {
            return Some("SV_GroupThreadID".to_string());
        }
        "S_GROUP_INDEX" | "SV_GROUPINDEX" => return Some("SV_GroupIndex".to_string()),
        "S_DOMAIN_LOCATION" | "SV_DOMAINLOCATION" => {
            return Some("SV_DomainLocation".to_string())
        }
        // Hull patch-constant outputs (edge/inside tessellation factors
        // feed the fixed-function tessellator, not varyings).
        "S_EDGE_TESS_FACTOR" | "SV_TESSFACTOR" => {
            return Some("SV_TessFactor".to_string())
        }
        "S_INSIDE_TESS_FACTOR" | "SV_INSIDETESSFACTOR" => {
            return Some("SV_InsideTessFactor".to_string())
        }
        "S_SRT_DATA" => return None,
        _ => {}
    }

    if let Some(rest) = upper.strip_prefix("COLOR") {
        if rest.is_empty() {
            return Some("TEXCOORD0".to_string());
        }
        if let Ok(index) = rest.parse::<u32>() {
            return Some(format!("TEXCOORD{index}"));
        }
    }
    if let Some(rest) = upper.strip_prefix("S_TARGET_OUTPUT") {
        if rest.is_empty() {
            return Some("SV_Target0".to_string());
        }
        if let Ok(index) = rest.parse::<u32>() {
            return Some(format!("SV_Target{index}"));
        }
    }
    if upper == "S_TARGET_OUTPUT" {
        return Some("SV_Target0".to_string());
    }
    if let Some(rest) = upper.strip_prefix("TEXCOORD") {
        if rest.is_empty() {
            return Some("TEXCOORD0".to_string());
        }
        if let Ok(index) = rest.parse::<u32>() {
            return Some(format!("TEXCOORD{index}"));
        }
    }

    None
}

/// Infers a shader stage hint from a function return semantic.
pub fn stage_hint_from_return(semantic: Option<&Semantic>) -> Option<Stage> {
    let name = semantic?.name();
    let upper = name.to_ascii_uppercase();

    if upper == "S_POSITION" || upper == "SV_POSITION" {
        return Some(Stage::Vertex);
    }
    if upper.starts_with("S_TARGET_OUTPUT") || upper.starts_with("SV_TARGET") {
        return Some(Stage::Fragment);
    }
    None
}
