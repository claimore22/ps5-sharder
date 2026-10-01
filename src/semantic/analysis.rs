use crate::Module;
use crate::ast::{Attribute, Expr};
use crate::error::{Phase, Ps5ShaderError, Result};
use crate::ir::Stage;

use super::bindings::stage_hint_from_return;

/// The inferred entry point: which function is the shader entry, and which
/// stage it represents.
#[derive(Debug, Clone, PartialEq)]
pub struct EntryPoint {
    pub function: String,
    pub stage: Stage,
    pub attributes: Vec<Attribute>,
}

/// Identifies the shader entry point and its stage.
///
/// We deliberately do NOT assume every `main` is a pixel shader. Precedence:
/// 1. A function carrying a `CxxSymbol` attribute, whose symbol suffix
///    (`Shader::vs` / `Shader::ps` / `Shader::cs`) reveals the stage.
/// 2. Otherwise a function named `main`, with the stage inferred from its
///    return semantic (`S_POSITION` -> vertex, `S_TARGET_OUTPUTn` -> fragment).
/// Whether `function` is an entry candidate: CxxSymbol-tagged or named
/// `main`. Only candidates can become entries; helpers never should.
fn is_entry_candidate(function: &crate::ast::Function) -> bool {
    // Scoped methods (`Ret Scope::m()`, in-struct methods) are never
    // entries, even if named `main` or carrying attributes.
    // A function is a candidate when it is named `main`, or when it
    // carries an explicit entry attribute: `CxxSymbol` (the usual
    // `Shader::name` tag), `profile(...)`, `shader("kind")`, or one of
    // the mesh/entry group-size markers (`NUM_THREADS`/`num_threads`).
    // This admits real entries like `agc_mesh_shader`'s `main_common`
    // (carries `cxxsymbol(...)` + `profile("ms")`) that are not named
    // `main`, so they can outrank a bare helper-named `main`.
    function.scope.is_none()
        && (function.name == "main"
            || function.attributes.iter().any(|a| {
                a.name.eq_ignore_ascii_case("CxxSymbol")
                    || a.name.eq_ignore_ascii_case("profile")
                    || a.name.eq_ignore_ascii_case("shader")
                    || a.name.eq_ignore_ascii_case("NUM_THREADS")
                    || a.name.eq_ignore_ascii_case("num_threads")
            }))
}

/// Definitive HLSL-exclusive signature markers, checked before any naming
/// heuristics (a `Shader::ps` tag never overrides dispatch dimensions).
/// Order: NGG (mesh-only) > stream topology (GS-only) > NUM_THREADS
/// (compute-only) > dispatch/group IDs (compute-only) > domain location
/// (domain-only).
fn definitive_stage(function: &crate::ast::Function) -> Option<Stage> {
    // Pass 1: NGG outputs exist ONLY on mesh entries, and must outrank
    // compute dispatch IDs. Mesh shaders carry BOTH `S_NGG_OUTPUT_*` /
    // `NGG_OUTPUT` returns AND `S_GROUP_ID`/`S_GROUP_THREAD_ID`; the NGG
    // markers settle the stage before the dispatch IDs (which mesh shaders
    // also have) can misclassify them as compute.
    if let crate::ast::Type::Named(type_name) = &function.return_type {
        if type_name.to_ascii_uppercase() == "NGG_OUTPUT" {
            return Some(Stage::Mesh);
        }
    }
    for param in &function.parameters {
        if let Some(sem) = &param.semantic {
            let upper = sem.name().to_ascii_uppercase();
            if upper.starts_with("S_NGG_OUTPUT_") {
                return Some(Stage::Mesh);
            }
        }
    }
    // Pass 2: domain-location parameter (domain shader takes the hull
    // output patch plus barycentric/domain coordinates).
    for param in &function.parameters {
        if let Some(sem) = &param.semantic {
            let upper = sem.name().to_ascii_uppercase();
            if upper == "S_DOMAIN_LOCATION" || upper == "SV_DOMAINLOCATION" {
                return Some(Stage::Domain);
            }
        }
    }
    // Pass 3: stream/topology parameters => geometry shader.
    // (`TriangleBuffer<T>` lowers to a Resource; bare spellings stay Named.)
    for param in &function.parameters {
        match &param.ty {
            crate::ast::Type::Named(type_name) => {
                let upper = type_name.to_ascii_uppercase();
                if upper.contains("TRIANGLEBUFFER")
                    || upper.contains("TRIANGLESTREAM")
                    || upper.contains("POINTSTREAM")
                    || upper.contains("LINESTREAM")
                {
                    return Some(Stage::Geometry);
                }
            }
            crate::ast::Type::Resource { kind, .. }
                if kind.eq_ignore_ascii_case("trianglestream") =>
            {
                return Some(Stage::Geometry);
            }
            _ => {}
        }
        // `Point T x` input topology markers only occur on GS entries.
        if param.gs_primitive.is_some() {
            return Some(Stage::Geometry);
        }
    }
    // Pass 3b: `[MAX_VERTEX_COUNT(N)]` only occurs on geometry shaders.
    if has_attribute(&function.attributes, "MAX_VERTEX_COUNT") {
        return Some(Stage::Geometry);
    }
    // Pass 4: compute dispatch IDs.
    for param in &function.parameters {
        if let Some(sem) = &param.semantic {
            let upper = sem.name().to_ascii_uppercase();
            if upper == "S_DISPATCH_THREAD_ID"
                || upper == "SV_DISPATCHTHREADID"
                || upper == "S_GROUP_ID"
                || upper == "SV_GROUPID"
                || upper == "S_GROUP_THREAD_ID"
                || upper == "SV_GROUPTHREADID"
                || upper == "S_GROUP_INDEX"
                || upper == "SV_GROUPINDEX"
            {
                return Some(Stage::Compute);
            }
        }
    }
    // Compute dispatch dimensions never appear on non-compute entries
    // (mesh carries NGG markers instead, checked above).
    if has_attribute(&function.attributes, "NUM_THREADS") {
        return Some(Stage::Compute);
    }
    None
}

pub fn find_entry_point(module: &Module) -> Result<Option<EntryPoint>> {
    let functions = &module.functions;

    // 0. Definitive signature markers outrank naming heuristics.
    // Multiple candidates can share a definitive stage (a mesh library has
    // both a bare helper-named `main` and the real `main_common` with
    // `[num_threads(...), profile("ms")]`). The bare `main` wins source
    // order, then fails at DXC (`mesh entry point must have a valid
    // numthreads attribute`) because it carries no num_threads. Prefer the
    // candidate that is most explicitly attributed: an explicit stage
    // attribute (`profile`/`CxxSymbol`/`shader`) or `NUM_THREADS` outranks
    // a bare `main`; only when every applicant is equally bare does the
    // first `main` win as before.
    let mut found: Option<(&crate::ast::Function, Stage)> = None;
    let mut found_score: (bool, bool) = (false, false);
    for function in functions {
        if !is_entry_candidate(function) {
            continue;
        }
        let Some(stage) = definitive_stage(function) else {
            continue;
        };
        let numpy = function
            .attributes
            .iter()
            .any(|a| a.name.eq_ignore_ascii_case("NUM_THREADS") || a.name.eq_ignore_ascii_case("num_threads"));
        let explicit = stage_from_attributes(&function.attributes).is_some();
        let score = (explicit, numpy);
        let better = match &found {
            None => true,
            Some((_, prev_stage)) if *prev_stage != stage => false,
            Some(_) => score > found_score,
        };
        if better {
            found = Some((function, stage));
            found_score = score;
        }
    }
    if let Some((function, stage)) = found {
        return Ok(Some(EntryPoint {
            function: function.name.clone(),
            stage,
            attributes: function.attributes.clone(),
        }));
    }

    // 1. CxxSymbol-tagged functions. (A raytracing-kind suffix guess such
    // as `_instance` never reaches a NUM_THREADS carrier: the definitive
    // pre-pass above already claimed those as compute.)
    for function in functions {
        if let Some(stage) = stage_from_attributes(&function.attributes) {
            return Ok(Some(EntryPoint {
                function: function.name.clone(),
                stage,
                attributes: function.attributes.clone(),
            }));
        }
    }

    // 2. A function named `main` with an inferable stage.
    for function in functions {
        if function.name == "main" && function.scope.is_none() {
            let stage = stage_hint_from_function(function, module).ok_or_else(|| {
                Ps5ShaderError::at(
                    "cannot determine shader stage for `main`; \
                         annotate it with [CxxSymbol(\"Shader::<stage>\")] \
                         or a stage semantic",
                    Phase::Semantic,
                    function.span,
                )
            })?;
            return Ok(Some(EntryPoint {
                function: function.name.clone(),
                stage,
                attributes: function.attributes.clone(),
            }));
        }
    }

    Ok(None)
}

/// Determines a shader stage from a set of `CxxSymbol` attributes, if one is
/// interpretable. Also honours the raytracing `[shader("kind")]` attribute,
/// which names the DXR-equivalent stage directly.
fn stage_from_attributes(attributes: &[Attribute]) -> Option<Stage> {
    for attribute in attributes {
        if attribute.name != "CxxSymbol" {
            continue;
        }
        let Some(arg) = attribute.arguments.first() else {
            continue;
        };
        let Expr::String(symbol) = arg else {
            continue;
        };
        if let Some(stage) = stage_from_symbol(symbol) {
            return Some(stage);
        }
    }
    for attribute in attributes {
        if !attribute.name.eq_ignore_ascii_case("shader") {
            continue;
        }
        let Some(arg) = attribute.arguments.first() else {
            continue;
        };
        let Expr::String(kind) = arg else {
            continue;
        };
        if let Some(stage) = stage_from_raytracing_kind(kind) {
            return Some(stage);
        }
    }
    // `[profile("cs")]` / `[profile("vs")]` / ... names the pipeline stage.
    for attribute in attributes {
        if !attribute.name.eq_ignore_ascii_case("profile") {
            continue;
        }
        let Some(arg) = attribute.arguments.first() else {
            continue;
        };
        let Expr::String(profile) = arg else {
            continue;
        };
        let p = profile.to_ascii_lowercase();
        // Prefix match: `vs_vs`, `vs`, `ps_5_0`, ... all start with the
        // stage token. This covers framework multi-token profiles like
        // `profile("vs_vs")` (vertex shader).
        if p == "vs" || p.starts_with("vs_") || p.starts_with("vs.") {
            return Some(Stage::Vertex);
        }
        if p == "ps" || p == "fs" || p.starts_with("ps_") || p.starts_with("fs_") {
            return Some(Stage::Fragment);
        }
        if p == "cs" || p.starts_with("cs_") {
            return Some(Stage::Compute);
        }
        if p == "gs" || p.starts_with("gs_") {
            return Some(Stage::Geometry);
        }
        if p == "hs" || p.starts_with("hs_") {
            return Some(Stage::Hull);
        }
        if p == "ms" || p.starts_with("ms_") {
            return Some(Stage::Mesh);
        }
    }
    // `[ABI(g_closestHitAbi)]` — the ABI name carries the raytracing kind
    // when neither `CxxSymbol` nor `[shader]` names it (e.g. `Shader::solid`
    // with a closest-hit ABI).
    for attribute in attributes {
        if !attribute.name.eq_ignore_ascii_case("abi") {
            continue;
        }
        let Some(arg) = attribute.arguments.first() else {
            continue;
        };
        let abi_name = match arg {
            Expr::String(s) => s.to_ascii_lowercase(),
            Expr::Variable(v) => v.to_ascii_lowercase(),
            _ => continue,
        };
        if abi_name.contains("anyhit") {
            return Some(Stage::AnyHit);
        }
        if abi_name.contains("closesthit") || abi_name.contains("closest_hit") {
            return Some(Stage::ClosestHit);
        }
        if abi_name.contains("miss") {
            return Some(Stage::Miss);
        }
        if abi_name.contains("intersection") || abi_name.contains("instance") {
            return Some(Stage::Intersection);
        }
    }
    None
}

/// Maps a raytracing `[shader("kind")]` name to its DXR-equivalent stage.
///
/// `instance` is the PSR per-instance ray callback (intersection-style), so
/// it maps to [`Stage::Intersection`].
fn stage_from_raytracing_kind(kind: &str) -> Option<Stage> {
    match kind.to_ascii_lowercase().as_str() {
        "anyhit" => Some(Stage::AnyHit),
        "closesthit" => Some(Stage::ClosestHit),
        "miss" => Some(Stage::Miss),
        "intersection" | "instance" => Some(Stage::Intersection),
        _ => None,
    }
}

/// True when `tag` (e.g. `"::ps"`) occurs in `lower` followed by a
/// non-alphanumeric boundary (end of string or separator). Prevents
/// framework-namespace false positives such as `Shader::psr_*` (PSR =
/// raytracing framework, not a pixel shader).
fn has_tag(lower: &str, tag: &str) -> bool {
    let mut rest = lower;
    while let Some(pos) = rest.find(tag) {
        let after = rest[pos + tag.len()..].chars().next();
        match after {
            None => return true,
            Some(c) if !c.is_ascii_alphanumeric() => return true,
            _ => {}
        }
        rest = &rest[pos + 1..];
    }
    false
}

fn stage_from_symbol(symbol: &str) -> Option<Stage> {
    let lower = symbol.to_ascii_lowercase();
    // Direct suffix like "vs", "ps", "cs"
    let suffix = lower.rsplit([':', '/', '.']).next().unwrap_or(&lower);
    // Handle framework naming like `Shader::triangle_gs` / `Shader::gbuffer_ps`
    // where the suffix is `triangle_gs` — check for embedded stage hints.
    let s = suffix;
    if s == "vs"
        || s == "vv"
        || s == "vertex"
        || s.ends_with("_vs")
        || s.ends_with("_vv")
        || s.ends_with("_gs")
        || s == "gs"
        || s == "geometry"
    {
        return Some(Stage::Vertex);
    }
    if s == "ps"
        || s == "fs"
        || s == "pixel"
        || s == "fragment"
        || s.ends_with("_ps")
        || s.ends_with("_fs")
        || s.ends_with("_p")
    {
        return Some(Stage::Fragment);
    }
    if s == "cs" || s == "c" || s == "compute" || s.ends_with("_cs") || s.ends_with("_c") {
        return Some(Stage::Compute);
    }
    if s == "hs"
        || s == "hull"
        || s.ends_with("_hs")
        || s.ends_with("_hull")
    {
        return Some(Stage::Hull);
    }
    // Mesh shaders (`Shader::gs_ms`, `vco_stats_ms`, ...). Checked before
    // the compute `_c` rule: `_ms` never means compute.
    if s == "ms" || s == "mesh" || s.ends_with("_ms") {
        return Some(Stage::Mesh);
    }
    // Domain shaders (`Shader::ds`, `domain_dv`).
    if s == "ds" || s == "domain" || s.ends_with("_ds") {
        return Some(Stage::Domain);
    }
    // Raytracing CxxSymbol suffixes (`Shader::anyhit`, `Shader::miss`).
    if s == "anyhit" || s == "any_hit" || s.ends_with("_anyhit") {
        return Some(Stage::AnyHit);
    }
    if s == "closesthit"
        || s == "closest_hit"
        || s.ends_with("_closesthit")
        || s.ends_with("_closest_hit")
    {
        return Some(Stage::ClosestHit);
    }
    if s == "miss" || s.ends_with("_miss") {
        return Some(Stage::Miss);
    }
    if s == "intersection"
        || s == "instance"
        || s.ends_with("_intersection")
        || s.ends_with("_instance")
    {
        return Some(Stage::Intersection);
    }
    // Note: `gs` / `_gs` map to Vertex above (legacy GS-flavored vertex
    // stage); a lone `_g` suffix denotes a true geometry shader.
    if s == "geometry" || s.ends_with("_g") {
        return Some(Stage::Geometry);
    }
    // Fallback: check if the whole symbol contains a stage hint. The tag
    // must sit on a word boundary (`Shader::psr_*` is the raytracing
    // framework, not a pixel shader).
    if has_tag(&lower, "_gs") || has_tag(&lower, "::gs") {
        return Some(Stage::Vertex);
    }
    if has_tag(&lower, "_ps")
        || has_tag(&lower, "::ps")
        || has_tag(&lower, "_fs")
        || has_tag(&lower, "::fs")
    {
        return Some(Stage::Fragment);
    }
    if has_tag(&lower, "_cs") || has_tag(&lower, "::cs") {
        return Some(Stage::Compute);
    }
    None
}

fn stage_hint_from_function(function: &crate::ast::Function, module: &Module) -> Option<Stage> {
    // 1. Direct return semantic
    if let Some(stage) = stage_hint_from_return(function.return_semantic.as_ref()) {
        return Some(stage);
    }
    // 2. Return-type name carries the pipeline family: NGG (mesh), hull
    // outputs, legacy GS outputs, deferred pixel outputs. This precedes
    // struct-field inspection because those structs also carry `S_POSITION`
    // fields that would otherwise misclassify them as vertex.
    //
    // `GS_OUTPUT` / `LS_OUTPUT` by-value returns are vertex-flavored (the toolchain
    // compiles `regular_vv`, `local_vl` as vertex): true geometry/hull
    // entries are already claimed by the stream/patch rules above, and a
    // by-value struct return cannot drive rasterization as GS/HS output.
    if let crate::ast::Type::Named(type_name) = &function.return_type {
        let upper = type_name.to_ascii_uppercase();
        if upper == "NGG_OUTPUT" {
            return Some(Stage::Mesh);
        }
        if upper.contains("HS_OUTPUT") {
            return Some(Stage::Hull);
        }
        if upper.contains("GS_OUTPUT") || upper.contains("LS_OUTPUT") {
            return Some(Stage::Vertex);
        }
        if upper.contains("PS_OUTPUT") {
            return Some(Stage::Fragment);
        }
    }
    // 3. Return type is a struct — inspect its fields for stage semantics
    if let crate::ast::Type::Named(struct_name) = &function.return_type {
        if let Some(decl) = module.structs.iter().find(|s| &s.name == struct_name) {
            for field in &decl.fields {
                if let Some(sem) = &field.semantic {
                    if let Some(stage) = stage_hint_from_return(Some(sem)) {
                        return Some(stage);
                    }
                }
            }
        }
    }
    // 4. A bare scalar/vector/matrix return with no return semantic can only
    // be a fragment output (vertex outputs need `S_POSITION`, compute and
    // raytracing entries return void, geometry/hull/mesh return structs).
    if function.return_semantic.is_none()
        && matches!(
            function.return_type,
            crate::ast::Type::Bool
                | crate::ast::Type::Int
                | crate::ast::Type::UInt
                | crate::ast::Type::Float
                | crate::ast::Type::Half
                | crate::ast::Type::Double
                | crate::ast::Type::Vector { .. }
                | crate::ast::Type::Matrix { .. }
        )
    {
        return Some(Stage::Fragment);
    }
    // 5. Compute dispatch dimensions — compute shaders always carry it.
    if has_attribute(&function.attributes, "NUM_THREADS") {
        return Some(Stage::Compute);
    }
    // 6. Raytracing `[shader("kind")]` names the DXR-equivalent stage.
    for attribute in &function.attributes {
        if !attribute.name.eq_ignore_ascii_case("shader") {
            continue;
        }
        if let Some(arg) = attribute.arguments.first()
            && let Expr::String(kind) = arg
            && let Some(stage) = stage_from_raytracing_kind(kind)
        {
            return Some(stage);
        }
    }
    // 6b. `[ABI(g_closestHitAbi)]` carries the kind when `[shader]` is absent.
    for attribute in &function.attributes {
        if !attribute.name.eq_ignore_ascii_case("abi") {
            continue;
        }
        let Some(arg) = attribute.arguments.first() else {
            continue;
        };
        let abi_name = match arg {
            Expr::String(s) => s.to_ascii_lowercase(),
            Expr::Variable(v) => v.to_ascii_lowercase(),
            _ => continue,
        };
        if abi_name.contains("anyhit") {
            return Some(Stage::AnyHit);
        }
        if abi_name.contains("closesthit") || abi_name.contains("closest_hit") {
            return Some(Stage::ClosestHit);
        }
        if abi_name.contains("miss") {
            return Some(Stage::Miss);
        }
        if abi_name.contains("intersection") || abi_name.contains("instance") {
            return Some(Stage::Intersection);
        }
    }
    // 7. Parameter semantics hint (e.g. S_VERTEX_ID → vertex, S_DISPATCH_THREAD_ID → compute)
    for param in &function.parameters {
        if let Some(sem) = &param.semantic {
            let upper = sem.name().to_ascii_uppercase();
            if upper == "S_VERTEX_ID" || upper == "SV_VERTEXID" {
                return Some(Stage::Vertex);
            }
            if upper == "S_DISPATCH_THREAD_ID" || upper == "SV_DISPATCHTHREADID" {
                return Some(Stage::Compute);
            }
            if upper == "S_GROUP_ID"
                || upper == "SV_GROUPID"
                || upper == "S_GROUP_THREAD_ID"
                || upper == "SV_GROUPTHREADID"
                || upper == "S_GROUP_INDEX"
                || upper == "SV_GROUPINDEX"
            {
                return Some(Stage::Compute);
            }
            if upper.starts_with("S_TARGET_OUTPUT") || upper.starts_with("SV_TARGET") {
                // Render-target outputs only exist on fragment entries
                // (e.g. `PSSL_ARG_OUT(...) PSSL_SEMANTIC(S_TARGET_OUTPUT0)`).
                return Some(Stage::Fragment);
            }
            if upper == "S_SRT_DATA" {
                // S_SRT_DATA is not stage-specific, skip
                continue;
            }
        }
        // 8. Tessellation / stream parameter types imply their stage.
        if let crate::ast::Type::Named(type_name) = &param.ty {
            let upper = type_name.to_ascii_uppercase();
            if upper.starts_with("INPUTPATCH") || upper.starts_with("OUTPUTPATCH") {
                return Some(Stage::Hull);
            }
            if upper.contains("TRIANGLEBUFFER")
                || upper.contains("TRIANGLESTREAM")
                || upper.contains("POINTSTREAM")
                || upper.contains("LINESTREAM")
            {
                return Some(Stage::Geometry);
            }
        }
    }
    // 9. An input struct carrying vertex-only system values (vertex/instance
    // IDs) marks a vertex entry. `S_POSITION` is deliberately excluded: it
    // also appears on interpolated fragment inputs.
    for param in &function.parameters {
        if let crate::ast::Type::Named(type_name) = &param.ty
            && let Some(decl) = module.structs.iter().find(|s| &s.name == type_name)
        {
            for field in &decl.fields {
                if let Some(sem) = &field.semantic {
                    let upper = sem.name().to_ascii_uppercase();
                    if upper == "S_VERTEX_ID"
                        || upper == "SV_VERTEXID"
                        || upper == "S_INSTANCE_ID"
                        || upper == "SV_INSTANCEID"
                    {
                        return Some(Stage::Vertex);
                    }
                }
            }
        }
    }
    // 10. Tessellation attributes / legacy geometry vertex-count attributes.
    // (NGG mesh entries carry `max_vertex_count` too but return NGG_OUTPUT,
    // which rule 5 already maps to Mesh.)
    if has_attribute(&function.attributes, "OUTPUT_CONTROL_POINTS")
        || has_attribute(&function.attributes, "PATCH_CONSTANT_FUNC")
        || has_attribute(&function.attributes, "PARTITIONING_TYPE")
        || has_attribute(&function.attributes, "DOMAIN_PATCH_TYPE")
        || has_attribute(&function.attributes, "OUTPUT_TOPOLOGY_TYPE")
        || has_attribute(&function.attributes, "MAX_TESS_FACTOR")
    {
        return Some(Stage::Hull);
    }
    if has_attribute(&function.attributes, "MAX_VERTEX_COUNT") {
        return Some(Stage::Geometry);
    }
    // 11. `clip()` / `discard` only exist in fragment entries (alpha-tested
    // pixel shaders with `void main`, e.g. `matl_alpha_test_p`).
    if body_kills_fragments(&function.body) {
        return Some(Stage::Fragment);
    }
    None
}

/// Whether `attributes` contains an attribute named `name` (case-insensitive).
fn has_attribute(attributes: &[Attribute], name: &str) -> bool {
    attributes
        .iter()
        .any(|a| a.name.eq_ignore_ascii_case(name))
}

/// Whether a function body calls `clip()` or executes `discard` (in any
/// nested statement or expression), which only fragment shaders do.
fn body_kills_fragments(body: &[crate::ast::Stmt]) -> bool {
    body.iter().any(stmt_kills_fragments)
}

fn stmt_kills_fragments(stmt: &crate::ast::Stmt) -> bool {
    use crate::ast::Stmt;
    match stmt {
        Stmt::Return(expr) => expr.as_ref().is_some_and(expr_kills_fragments),
        Stmt::If {
            condition,
            then_branch,
            else_branch,
        } => {
            expr_kills_fragments(condition)
                || then_branch.iter().any(stmt_kills_fragments)
                || else_branch
                    .as_ref()
                    .is_some_and(|b| b.iter().any(stmt_kills_fragments))
        }
        Stmt::Block(stmts) => stmts.iter().any(stmt_kills_fragments),
        Stmt::For {
            init,
            condition,
            update,
            body,
        } => {
            stmt_kills_fragments(init)
                || condition.as_ref().is_some_and(expr_kills_fragments)
                || update.as_ref().is_some_and(expr_kills_fragments)
                || body.iter().any(stmt_kills_fragments)
        }
        Stmt::While { condition, body } => {
            expr_kills_fragments(condition) || body.iter().any(stmt_kills_fragments)
        }
        Stmt::Switch {
            scrutinee,
            cases,
        } => {
            expr_kills_fragments(scrutinee)
                || cases.iter().any(|c| {
                    c.label.as_ref().is_some_and(expr_kills_fragments)
                        || c.body.iter().any(stmt_kills_fragments)
                })
        }
        Stmt::VarDecl { init, .. } => init.as_ref().is_some_and(expr_kills_fragments),
        Stmt::Destructure { init, .. } => {
            init.as_ref().is_some_and(expr_kills_fragments)
        }
        Stmt::Expr(expr) => expr_kills_fragments(expr),
    }
}

fn expr_kills_fragments(expr: &crate::ast::Expr) -> bool {
    use crate::ast::Expr;
    match expr {
        Expr::Variable(name) => name.eq_ignore_ascii_case("discard"),
        Expr::Call { function, arguments } => {
            if let Expr::Variable(name) = function.as_ref()
                && name.eq_ignore_ascii_case("clip")
            {
                return true;
            }
            expr_kills_fragments(function)
                || arguments.iter().any(expr_kills_fragments)
        }
        Expr::Constructor { arguments, .. } => arguments.iter().any(expr_kills_fragments),
        Expr::Member { object, .. } => expr_kills_fragments(object),
        Expr::Index { object, index } => {
            expr_kills_fragments(object) || expr_kills_fragments(index)
        }
        Expr::Assign { lhs, rhs } => expr_kills_fragments(lhs) || expr_kills_fragments(rhs),
        Expr::Unary { expr, .. } => expr_kills_fragments(expr),
        Expr::Binary { left, right, .. } => {
            expr_kills_fragments(left) || expr_kills_fragments(right)
        }
        Expr::Conditional {
            condition,
            then_branch,
            else_branch,
        } => {
            expr_kills_fragments(condition)
                || expr_kills_fragments(then_branch)
                || expr_kills_fragments(else_branch)
        }
        Expr::InitializerList(exprs) => exprs.iter().any(expr_kills_fragments),
        Expr::Lambda { .. } => false,
        Expr::Number(_) | Expr::String(_) | Expr::Bool(_) => false,
    }
}
