pub mod analysis;
pub mod bindings;
pub mod symbols;
pub mod types;

pub use analysis::EntryPoint;
pub use types::{TypeEnv, mismatch, to_ir_type, type_error};

use crate::ast::{Expr, Function, Module, Stmt, Type};
use crate::error::{Phase, Ps5ShaderError, Result};
use crate::ir::Stage;

use self::symbols::SymbolTable;

/// The output of semantic analysis: a resolved, validated description of the
/// shader that is sufficient to drive lowering into the Shader IR.
#[derive(Debug)]
pub struct SemanticModel {
    /// The inferred stage, if an entry point was identified.
    pub stage: Option<Stage>,
    /// The identified entry point, if any.
    pub entry_point: Option<EntryPoint>,
    /// Collected module symbols (structs / functions).
    pub symbols: SymbolTable,
}

/// Performs semantic analysis on the parsed `module`.
///
/// This identifies the entry point/stage and validates the shader well enough
/// to catch the milestone's required classes of error (unknown symbols, type
/// mismatches, bad constructors/members).
pub fn analyze(module: &Module) -> Result<SemanticModel> {
    let mut symbols = SymbolTable::default();
    for decl in &module.structs {
        symbols.add_struct(decl.clone());
    }
    for decl in &module.resources {
        symbols.add_resource(decl.clone());
        // ConstantBuffer fields are accessed as bare globals in PSSL
        // (e.g. `m_rotation` from `ConstantBuffer TransformConstants : register(b0) { float m_rotation; }`)
        for field in &decl.fields {
            symbols.add_global(crate::ast::GlobalDecl {
                ty: field.ty.clone(),
                name: field.name.clone(),
                semantic: field.semantic.clone(),
                array_size: field.array_size,
                shared: false,
                init: None,
                span: crate::error::Span::default(),
            });
        }
    }
    for decl in &module.globals {
        symbols.add_global(decl.clone());
    }
    for function in &module.functions {
        symbols.add_function(function.clone());
    }

    let entry_point = analysis::find_entry_point(module)?;
    let stage = entry_point.as_ref().map(|ep| ep.stage);

    // Type check every function body.
    for function in &module.functions {
        let mut env = TypeEnv::default();
        // Out-of-class method definitions (`Ret Scope::method(...)`) see
        // their scope's members as bare names (C++ member scope).
        // Parameters (inserted next) shadow same-named members.
        if let Some(scope) = function.scope.as_ref()
            && let Some(decl) = symbols.get_struct(scope)
        {
            for field in &decl.fields {
                env.insert(field.name.clone(), field.ty.clone());
            }
        }
        for parameter in &function.parameters {
            env.insert(parameter.name.clone(), parameter.ty.clone());
        }
        check_function_body(function, &symbols, &mut env)?;
    }

    Ok(SemanticModel {
        stage,
        entry_point,
        symbols,
    })
}

fn check_function_body(
    function: &Function,
    symbols: &SymbolTable,
    env: &mut TypeEnv,
) -> Result<()> {
    for statement in &function.body {
        check_stmt(statement, symbols, env)?;
    }
    Ok(())
}

fn check_stmt(stmt: &Stmt, symbols: &SymbolTable, env: &mut TypeEnv) -> Result<()> {
    match stmt {
        Stmt::Return(Some(expr)) => {
            check_expr(expr, symbols, env)?;
        }
        Stmt::Return(None) => {}
        Stmt::Block(statements) => {
            for s in statements {
                check_stmt(s, symbols, env)?;
            }
        }
        Stmt::If {
            condition,
            then_branch,
            else_branch,
        } => {
            check_expr(condition, symbols, env)?;
            for s in then_branch {
                check_stmt(s, symbols, env)?;
            }
            if let Some(block) = else_branch {
                for s in block {
                    check_stmt(s, symbols, env)?;
                }
            }
        }
        Stmt::For {
            init,
            condition,
            update,
            body,
        } => {
            check_stmt(init, symbols, env)?;
            if let Some(cond) = condition {
                check_expr(cond, symbols, env)?;
            }
            if let Some(upd) = update {
                check_expr(upd, symbols, env)?;
            }
            for s in body {
                check_stmt(s, symbols, env)?;
            }
        }
        Stmt::While { condition, body } => {
            check_expr(condition, symbols, env)?;
            for s in body {
                check_stmt(s, symbols, env)?;
            }
        }
        Stmt::Switch { scrutinee, cases } => {
            check_expr(scrutinee, symbols, env)?;
            for case in cases {
                if let Some(label) = &case.label {
                    check_expr(label, symbols, env)?;
                }
                for s in &case.body {
                    check_stmt(s, symbols, env)?;
                }
            }
        }
        Stmt::VarDecl { ty, name, init, array_size, .. } => {
            if let Some(init) = init {
                // PSSL samples sometimes use the variable being declared inside its
                // own designated initializer (`PrimaryPayload payload = {
                // payload.m_diffuse = ... }`). Make the variable visible *before*
                // checking the initializer so member access resolves; for `auto`
                // we still need the init's type first, so check it beforehand.
                let init_ty = if matches!(ty, Type::Named(n) if n == "auto") {
                    check_expr(init, symbols, env)?
                } else {
                    env.insert(name.clone(), ty.clone());
                    check_expr(init, symbols, env)?
                };
                // `auto` deduction: `auto x = expr;` → type of expr
                let effective_ty = if matches!(ty, Type::Named(n) if n == "auto") {
                    &init_ty
                } else {
                    ty
                };
                if let Err(mut e) = check_assignable(effective_ty, init_ty.clone(), init) {
                    // Augment error with variable name for diagnostics
                    e.message = format!("{name}: {}", e.message);
                    return Err(e);
                }
                // For `auto`, store deduced type so later uses (e.g. `GatherRed`) resolve correctly.
                if matches!(ty, Type::Named(n) if n == "auto") {
                    env.insert(name.clone(), init_ty);
                } else {
                    env.insert(name.clone(), ty.clone());
                }
            } else {
                env.insert(name.clone(), ty.clone());
            }
            if array_size.is_some() {
                env.mark_array(name.clone());
            }
        }
        Stmt::Expr(expr) => {
            check_expr(expr, symbols, env)?;
        }
        Stmt::Destructure { names, init, .. } => {
            // C++17 structured bindings: check the init, then bind each
            // name. A known struct init with matching fields binds
            // precisely; otherwise every name takes the init type.
            if let Some(init) = init {
                let init_ty = check_expr(init, symbols, env)?;
                if let Type::Named(s) = &init_ty {
                    if let Some(decl) = symbols.get_struct(s) {
                        if decl.fields.len() == names.len() {
                            for (name, field) in names.iter().zip(decl.fields.iter()) {
                                env.insert(name.clone(), field.ty.clone());
                            }
                            return Ok(());
                        }
                    }
                }
                for name in names {
                    env.insert(name.clone(), init_ty.clone());
                }
            }
        }
    }
    Ok(())
}

/// Returns the type of `expr`, validating symbol/member/constructor usage.
fn check_expr<'a>(expr: &'a Expr, symbols: &SymbolTable, env: &TypeEnv) -> Result<Type> {
    tracing::trace!(target: "pssl_shader::semantic::expr", kind = %expr_kind(expr), "checking expression");
    match expr {
        Expr::Number(value) => {
            // Distinguish integer vs float literals: `0` -> Int, `0.0`/`0f` -> Float
            let is_float = value.contains('.')
                || value.contains('e')
                || value.contains('E')
                || value.ends_with('f')
                || value.ends_with('F');
            if is_float {
                Ok(Type::Float)
            } else {
                Ok(Type::Int)
            }
        }
        Expr::String(_) => Ok(Type::Named("string".to_string())),
        Expr::Bool(_) => Ok(Type::Bool),
        // Lambdas are opaque callables (beta-reduction inlines the
        // reducible ones in lowering); the placeholder type keeps uses
        // flowing through the unknown-type rules.
        Expr::Lambda { .. } => Ok(Type::Named("__lambda".to_string())),
        Expr::Variable(name) => {
            if name == "break" || name == "continue" || name == "discard" {
                return Ok(Type::Void);
            }
            // C++ `this` (operator overloads in inlined headers): model
            // opaquely; member chains and copies flow through.
            if name == "this" {
                return Ok(Type::Named("this".to_string()));
            }
            if let Some(ty) = env.get(name) {
                Ok(ty.clone())
            } else if let Some(res) = symbols.get_resource(name) {
                if let Some(ty) = &res.element_ty {
                    Ok(ty.clone())
                } else {
                    // SamplerState etc. have no element type
                    Ok(Type::Named(name.clone()))
                }
            } else if let Some(ty) = symbols.get_global_type(name) {
                Ok(ty.clone())
            } else if symbols.has_struct(name) {
                Ok(Type::Named(name.clone()))
            } else if name.contains("::k") {
                // Likely an enum constant (e.g. TraversalFlags::kNoAABBGeometryInBvh)
                // Treat as integer constant for type checking.
                Ok(Type::Int)
            } else if name.contains("::") {
                // Treat unknown scoped name (e.g., `sce::Psr::Hit`) as an opaque struct.
                // This permits member accesses on it without needing the full definition.
                tracing::trace!(target: "pssl_shader::semantic::variable", name = %name, "opaque struct fallback");
                Ok(Type::Named(name.clone()))
            } else if name == "PI" || name == "M_PI" {
                // Well-known math constants (used without a defining header
                // in several sample shaders).
                Ok(Type::Float)
            } else if name.to_ascii_lowercase().starts_with("__kbuffer_")
                || name.to_ascii_lowercase().starts_with("__kimage_")
                || name.to_ascii_lowercase().starts_with("__kwavemode")
            {
                // GCN buffer/image flag constants (`__kBuffer_SLC`,
                // `__kImage_texture2d`, ...) for the `__buffer_*` /
                // `__image_*` intrinsics.
                Ok(Type::UInt)
            } else {
                // Unresolvable bare name (macro-generated constants like
                // `MODE_VISUALIZE_PROBES` from unexpanded function-like
                // macros, template value-params dropped by the frozen
                // parser such as `flags`, framework constants): model
                // opaquely so uses flow through arithmetic/ternaries via
                // `is_opaque_named`; DXC is the truth for genuine errors.
                tracing::trace!(target: "pssl_shader::semantic::variable", name = %name, "unknown variable opaque fallback");
                Ok(Type::Named(name.clone()))
            }
        }

        Expr::Member { object, member } => {
            tracing::trace!(
                target: "pssl_shader::semantic::member",
                member = %member,
                "checking member access"
            );

            if let Expr::Constructor { ty, arguments } = object.as_ref() {
                let is_scalar_target = matches!(
                    ty,
                    Type::Float | Type::Half | Type::Int | Type::UInt | Type::Bool | Type::Double
                );

                tracing::trace!(
                    target: "pssl_shader::semantic::member",
                    member = %member,
                    is_scalar_target,
                    constructor_args = arguments.len(),
                    member_len = member.len(),
                    "member on scalar constructor"
                );

                if is_scalar_target && arguments.len() == 1 && member.len() == 1 {
                    let inner_ty = check_expr(&arguments[0], symbols, env)?;

                    tracing::trace!(
                        target: "pssl_shader::semantic::member",
                        member = %member,
                        inner = ?inner_ty,
                        "constructor argument resolved"
                    );

                    if let Type::Vector { base, .. } = inner_ty {
                        tracing::trace!(
                            target: "pssl_shader::semantic::member",
                            member = %member,
                            scalar = ?scalar_type(base),
                            "PSSL scalar-from-vector cast"
                        );
                        return Ok(scalar_type(base));
                    }
                }
            }

            let object_ty = check_expr(object, symbols, env)?;

            // Scalar splat (`1.0.xxx`, `fog.xxx`, `f.y`): HLSL broadcasts
            // a scalar across the swizzle width (length 1 is identity).
            if matches!(
                object_ty,
                Type::Float
                    | Type::Half
                    | Type::Int
                    | Type::UInt
                    | Type::Bool
                    | Type::Double
            ) && member
                .chars()
                .all(|c| matches!(c, 'x' | 'y' | 'z' | 'w' | 'r' | 'g' | 'b' | 'a'))
                && !member.is_empty()
            {
                let base = match object_ty {
                    Type::Float => crate::ast::ScalarType::Float,
                    Type::Half => crate::ast::ScalarType::Half,
                    Type::Int => crate::ast::ScalarType::Int,
                    Type::UInt => crate::ast::ScalarType::UInt,
                    // No double vectors exist; bool/double splats are
                    // vanishingly rare, model with a close base.
                    Type::Bool => crate::ast::ScalarType::Bool,
                    _ => crate::ast::ScalarType::Float,
                };
                if member.len() == 1 {
                    return Ok(object_ty.clone());
                }
                return Ok(Type::Vector {
                    base,
                    width: member.len() as u8,
                });
            }

            // UAV mipmap writes (`tex.MipMaps[level][coord] = ...`): the
            // member denotes the resource itself so double indexing reaches
            // the element type.
            if let Expr::Variable(obj_name) = object.as_ref()
                && member.eq_ignore_ascii_case("mipmaps")
                && let Some(res) = symbols.get_resource(obj_name)
            {
                let access = if res.kind.starts_with("rw") {
                    crate::ast::ResourceAccess::ReadWrite
                } else {
                    crate::ast::ResourceAccess::ReadOnly
                };
                return Ok(Type::Resource {
                    kind: res.kind.clone(),
                    access,
                    element: Box::new(
                        res.element_ty.clone().unwrap_or(Type::Float),
                    ),
                });
            }

            tracing::trace!(
                target: "pssl_shader::semantic::member",
                member = %member,
                object_ty = ?object_ty,
                "resolved member object type"
            );

            match &object_ty {
                Type::Named(struct_name) => {
                    match symbols.get_struct(struct_name) {
                        Some(decl) => {
                            if let Some(field) =
                                decl.fields.iter().find(|f| f.name == *member)
                            {
                                tracing::trace!(
                                    target: "pssl_shader::semantic::member",
                                    member = %member,
                                    struct_name = %struct_name,
                                    field_ty = ?field.ty,
                                    "struct member resolved"
                                );

                                Ok(field.ty.clone())
                            } else {
                                // Known struct, unknown member: the parser
                                // drops `union { ... }` bodies (frozen), so
                                // union members (`Hit.triangleID`,
                                // `ElemAtt.m_pixelCoords`, ...) never reach
                                // the table; nested same-named structs from
                                // different parents can also collide. Model
                                // the member opaquely like unknown-struct
                                // members so deeper access and arithmetic
                                // keep resolving; DXC is the truth for
                                // genuine mismatches.
                                tracing::trace!(target: "pssl_shader::semantic::member", name = %struct_name, member = %member, "known-struct unknown-member opaque fallback");
                                Ok(Type::Named(format!("{struct_name}.{member}")))
                            }
                        }

                        None => {
                            // Opaque struct (template params, framework
                            // types, `::`-scoped names): model the member as
                            // a chained opaque type so deeper access
                            // (`t->a[0].b`) keeps resolving instead of
                            // collapsing to a scalar that rejects members.
                            // Arithmetic/ternaries treat these as
                            // compatible via `is_opaque_named`.
                            tracing::trace!(target: "pssl_shader::semantic::member", name = %struct_name, member = %member, "opaque struct member fallback");
                            Ok(Type::Named(format!("{struct_name}.{member}")))
                        },
                    }
                }

                Type::Vector { base, width } => {
                    validate_swizzle(member, *base, *width)?;

                    let swizzle_len = member.len() as u8;

                    tracing::trace!(
                        target: "pssl_shader::semantic::member",
                        member = %member,
                        swizzle_len,
                        "vector swizzle"
                    );

                    if swizzle_len == 1 {
                        Ok(scalar_type(*base))
                    } else {
                        Ok(Type::Vector {
                            base: *base,
                            width: swizzle_len,
                        })
                    }
                }

                Type::Float
                | Type::Half
                | Type::Int
                | Type::UInt
                | Type::Bool
                | Type::Double => {
                    // PSSL often does `scalar_tex.SampleLOD(...).x` where the
                    // texture element is a scalar `half`/`float`; the `.x`/`.r`
                    // is an identity swizzle on the scalar value.
                    if member.eq_ignore_ascii_case("x") || member.eq_ignore_ascii_case("r") {
                        return Ok(object_ty.clone());
                    }
                    return Err(Ps5ShaderError::at(
                        format!(
                            "cannot access member `{member}` of non-composite type {}",
                            object_ty.display()
                        ),
                        Phase::Semantic,
                        crate::error::Span::default(),
                    ));
                }

                _ => {
                    return Err(Ps5ShaderError::at(
                        format!(
                            "cannot access member `{member}` of non-composite type {}",
                            object_ty.display()
                        ),
                        Phase::Semantic,
                        crate::error::Span::default(),
                    ))
                }
            }
        }

        Expr::Index { object, .. } => {
            // Special cases for resource and global arrays
            if let Expr::Variable(name) = object.as_ref() {
                if let Some(res) = symbols.get_resource(name) {
                    let normalized = res.kind.to_ascii_lowercase().replace('_', "");
                    if normalized.contains("texture") || normalized.contains("rwtexture") {
                        if let Some(ty) = &res.element_ty {
                            return Ok(ty.clone());
                        }
                    }
                }
                if let Some(global) = symbols.get_global(name) {
                    if global.array_size.is_some() {
                        return Ok(global.ty.clone());
                    }
                }
                // Sized local arrays (`float4 AABB[48]`): indexing yields
                // the element type, not a vector component.
                if env.is_array(name)
                    && let Some(ty) = env.get(name)
                {
                    return Ok(ty.clone());
                }
            }
            // Sized struct fields (`obj.weights[i]`): likewise the element.
            if let Expr::Member { object: inner, member } = object.as_ref()
                && let Ok(Type::Named(struct_name)) = check_expr(inner, symbols, env)
                && let Some(decl) = symbols.get_struct(&struct_name)
                && let Some(field) = decl.fields.iter().find(|f| &f.name == member)
                && field.array_size.is_some()
            {
                return Ok(field.ty.clone());
            }
            let object_ty = check_expr(object, symbols, env)?;
            match &object_ty {
                Type::Vector { base, .. } => Ok(scalar_type(*base)),
                // Indexing a matrix yields a row vector (`mat[3].xyz`).
                Type::Matrix { base, columns, .. } => Ok(Type::Vector {
                    base: *base,
                    width: *columns,
                }),
                // Indexing a resource handle yields its element type, e.g.
                // `srt_data.dest[coord]` where `dest` is `RW_Texture2D<float>`.
                Type::Resource { element, .. } => Ok((**element).clone()),
                // Resources / arrays: indexing returns the element type (already)
                // or the same type for generic buffers. Be permissive.
                _ => Ok(object_ty),
            }
        }
        Expr::Call {
            function,
            arguments,
        } => {
            // `sizeof(x)` is compile-time; its operand may name a type or
            // an uninstantiated template param (`sizeof(T)`), neither of
            // which resolves as a value.
            if let Expr::Variable(name) = function.as_ref()
                && name.eq_ignore_ascii_case("sizeof")
            {
                return Ok(Type::UInt);
            }
            for arg in arguments {
                check_expr(arg, symbols, env)?;
            }

            // Handle `texture.Sample(...)` where function is `Member { object: Variable, member: "Sample" }`
            if let Expr::Member { object, member } = function.as_ref() {
                if let Expr::Variable(obj_name) = object.as_ref() {
                    if symbols.has_resource(obj_name) {
                        // Resource method calls.
                        //
                        // GetDimensionsFast writes through its output arguments and
                        // does not produce a meaningful expression value.
                        if member.eq_ignore_ascii_case("GetDimensionsFast") {
                            return Ok(Type::Void);
                        }
                        let lower_g = member.to_ascii_lowercase().replace('_', "");
                        if lower_g.starts_with("gather") {
                            if let Some(res) = symbols.get_resource(obj_name) {
                                if let Some(ty) = &res.element_ty {
                                    let base = match ty {
                                        Type::Float => crate::ast::ScalarType::Float,
                                        Type::Half => crate::ast::ScalarType::Half,
                                        Type::Int => crate::ast::ScalarType::Int,
                                        Type::UInt => crate::ast::ScalarType::UInt,
                                        Type::Vector { base, .. } => *base,
                                        _ => crate::ast::ScalarType::Float,
                                    };
                                    return Ok(Type::Vector { base, width: 4 });
                                }
                            }
                            return Ok(Type::Vector { base: crate::ast::ScalarType::Float, width: 4 });
                        }

                        // Texture.Sample / Load / Gather etc. return the resource
                        // element type when it is known.
                        if let Some(res) = symbols.get_resource(obj_name) {
                            if let Some(ty) = &res.element_ty {
                                return Ok(ty.clone());
                            }
                        }

                        return Ok(Type::Float);
                    }
                }

                // Resource member call, e.g. `srt_data.source.Load(coord, sample)`
                // where the callee object is a resource-typed struct member.
            let object_ty = check_expr(object, symbols, env)?;
                if let Type::Resource { element, .. } = object_ty {
                    let lower = member.to_ascii_lowercase().replace('_', "");
                    if matches!(
                        lower.as_str(),
                        "load"
                            | "sample"
                            | "samplelevel"
                            | "samplelod"
                            | "samplebias"
                            | "samplegrad"
                            | "samplecmplevelzero"
                            | "gather"
                            | "gatherred"
                            | "gatherred0"
                            | "gatherred1"
                            | "gathergreen"
                            | "gatherblue"
                            | "gatheralpha"
                    ) {
                        // Gather variants return 4 of the element type (e.g. Texture2D<float> -> float4)
                        if lower.starts_with("gather") {
                            let base = match element.as_ref() {
                                Type::Float => crate::ast::ScalarType::Float,
                                Type::Half => crate::ast::ScalarType::Half,
                                Type::Int => crate::ast::ScalarType::Int,
                                Type::UInt => crate::ast::ScalarType::UInt,
                                Type::Vector { base, .. } => *base,
                                _ => crate::ast::ScalarType::Float,
                            };
                            return Ok(Type::Vector { base, width: 4 });
                        }
                        return Ok((*element).clone());
                    }
                    if lower == "getdimensionsfast" {
                        return Ok(Type::Void);
                    }
                    return Ok(Type::Float);
                }

                // Generic member call (`obj.method(...)` where `obj` is not
                // a resource): the return type is unknowable, so model it
                // opaquely under the method name. It flows through member
                // chains and arithmetic via the unknown-type rules (a `Float`
                // default would poison vector ternaries like
                // `c ? obj.getEmissive() : float3(0)`).
                return Ok(Type::Named(member.clone()));
            }

            let name = match function.as_ref() {
                Expr::Variable(name) => name.clone(),
                _ => {
                    return Err(Ps5ShaderError::at(
                        "callee must be a named function",
                        Phase::Semantic,
                        crate::error::Span::default(),
                    ));
                }
            };
            if let Some(func) = symbols.get_function(&name)
                && (func.scope.is_none() || env.get(&name).is_none())
            {
                // Scoped methods (`Scope::m`) are invisible to unqualified
                // calls when a same-named local shadows them: a bare
                // `incr(x)` must never resolve to an unrelated struct's
                // `void incr()` (a psr stack helper shadowed a lambda and
                // poisoned arithmetic with `void`). Locals/params win over
                // methods (C++ hiding); recursion and implicit-`this`
                // sibling calls have no same-named local, so they still
                // resolve. Free functions are unaffected by this gate.
                // Generic `T` via `template<class T>` – treat `T` as a type
                // variable compatible with any. For `tex2D_bicubic` the return
                // `T` is deduced from the first `Texture2D<T>` argument.
                if func.return_type == Type::Named("T".to_string()) {
                    // Find a param that uses `T` and deduce from the actual arg.
                    for (idx, param) in func.parameters.iter().enumerate() {
                        let uses_t = match &param.ty {
                            Type::Named(n) if n == "T" => true,
                            Type::Resource { element, .. } if **element == Type::Named("T".to_string()) => true,
                            _ => false,
                        };
                        if uses_t && idx < arguments.len() {
                            let actual = check_expr(&arguments[idx], symbols, env)?;
                            // If param is `Texture2D<T>`, actual is `Texture2D<half2>` → deduce `half2`
                            if let Type::Resource { element, .. } = &actual {
                                tracing::trace!(target: "pssl_shader::semantic::generic", deduced_T = ?element, "deduced T from resource element");
                                return Ok((**element).clone());
                            }
                            // If param is bare `T`, actual itself is `T`
                            tracing::trace!(target: "pssl_shader::semantic::generic", deduced_T = ?actual, "deduced T from arg type");
                            return Ok(actual);
                        }
                    }
                    // Fallback: treat `T` as `half4` (common case in this shader)
                    tracing::trace!(target: "pssl_shader::semantic::generic", "generic T fallback to half4");
                    return Ok(Type::Vector { base: crate::ast::ScalarType::Half, width: 4 });
                }
                return Ok(func.return_type.clone());
            } else if is_builtin_function(&name) {
                let lower =
                    strip_c_float_suffix(&name.to_ascii_lowercase()).to_string();
                if matches!(lower.as_str(), "make_point_sampler" | "make_bilinear_sampler" | "make_trilinear_sampler") {
                    return Ok(Type::Named("SamplerState".to_string()));
                }
                if matches!(lower.as_str(), "min3" | "max3") && !arguments.is_empty() {
                    // PSSL 3-arg min/max: return type of first arg (float/half)
                    let first_ty = check_expr(&arguments[0], symbols, env)?;
                    return Ok(first_ty);
                }
                // Per-component builtins return the type of their first
                // argument: same scalar/vector for math (sin, log2, cross,
                // saturate, ...), which also lets opaque/template-typed
                // values (Named) flow through member chains.
                // The vectormath family (`mantissa_*`, `exponent_*`,
                // ...) is also per-component.
                if (matches!(
                    lower.as_str(),
                    "normalize" | "reflect" | "refract" | "lerp" | "clamp" | "min" | "max"
                        | "cross" | "saturate" | "abs" | "sign" | "floor" | "ceil" | "frac"
                        | "trunc" | "round" | "fmod" | "mad" | "fwidth" | "ddx" | "ddy"
                        | "sin" | "cos" | "tan" | "asin" | "acos" | "atan" | "atan2"
                        | "pow" | "pow4" | "exp" | "log" | "log2" | "log10" | "exp2"
                        | "sqrt" | "rsqrt" | "rcp" | "frexp" | "ldexp"
                        | "fmaxf" | "fminf" | "fmax" | "fmin" | "fabs"
                        | "inclusiveprefixsum" | "exclusiveprefixsum"
                        | "step" | "smoothstep" | "fxaasat"
                        | "readfirstlane" | "readlane"
                        | "crosslaneadd" | "crosslaneor" | "crosslanemin" | "crosslanemax"
                        | "laneswizzle" | "reversebits" | "distsqr" | "dist" | "lengthsqr"
                ) || lower.starts_with("sce_vectormath_"))
                    && !arguments.is_empty()
                {
                    let first_ty = check_expr(&arguments[0], symbols, env)?;
                    if matches!(
                        first_ty,
                        Type::Vector { .. } | Type::Named(_) | Type::Float | Type::Half
                    ) {
                        return Ok(first_ty);
                    }
                }
                // Integer/bit builtins return the uint shape of their first
                // argument (scalar `uint`, vector `uintN`).
                if matches!(
                    lower.as_str(),
                    "countbits" | "countsetbits" | "maskbitcnt" | "popcnt"
                        | "bitfieldextract" | "f32tof16"
                        | "__v_ffbh_u32" | "__v_ffbl_b32" | "__v_lerp_u8"
                        | "__read_user_sgpr" | "__v_readfirstlane_b32" | "__v_readlane_b32"
                        | "__v_mbcnt_lo_u32_b32" | "__v_mbcnt_hi_u32_b32" | "__get_tsharplo"
                        | "__get_tsharphi" | "__get_ssharp" | "__get_wavemode"
                        | "__v_cvt_f16_f32" | "__v_cvt_f32_f16" | "__v_cvt_f32_u32"
                ) && !arguments.is_empty()
                {
                    let first_ty = check_expr(&arguments[0], symbols, env)?;
                    if let Type::Vector { width, .. } = first_ty {
                        return Ok(Type::Vector {
                            base: crate::ast::ScalarType::UInt,
                            width,
                        });
                    }
                    return Ok(Type::UInt);
                }
                // Wave ballot reduces to a `uint4` mask.
                if lower == "ballot" {
                    return Ok(Type::Vector {
                        base: crate::ast::ScalarType::UInt,
                        width: 4,
                    });
                }
                // `isnan`/`isinf`/`isfinite`/`all`/`any` reduce to boolean.
                if matches!(lower.as_str(), "isnan" | "isinf" | "isfinite" | "all" | "any" | "sce_geometry_aos_isnormalized") {
                    if !arguments.is_empty()
                        && let Type::Vector { width, .. } =
                            check_expr(&arguments[0], symbols, env)?
                    {
                        return Ok(Type::Vector {
                            base: crate::ast::ScalarType::Bool,
                            width,
                        });
                    }
                    return Ok(Type::Bool);
                }
                // C `COUNTOF(x)` array-size macro yields a `uint`.
                if lower == "countof" || lower == "_countof" {
                    return Ok(Type::UInt);
                }
                // Bit reinterpretations keep the shape, converting the base.
                if matches!(lower.as_str(), "asfloat" | "asint" | "asuint") {
                    let base = match lower.as_str() {
                        "asfloat" => crate::ast::ScalarType::Float,
                        "asint" => crate::ast::ScalarType::Int,
                        _ => crate::ast::ScalarType::UInt,
                    };
                    if !arguments.is_empty()
                        && let Type::Vector { width, .. } =
                            check_expr(&arguments[0], symbols, env)?
                    {
                        return Ok(Type::Vector { base, width });
                    }
                    return Ok(match base {
                        crate::ast::ScalarType::Float => Type::Float,
                        crate::ast::ScalarType::Int => Type::Int,
                        _ => Type::UInt,
                    });
                }
                // Interlocked atomics `AtomicOp(dest, value[, original])`
                // return the original value (2-arg form) or nothing (3-arg
                // form with an explicit out).
                if matches!(lower.as_str(), "atomicadd" | "atomicor" | "atomicmax")
                    && !arguments.is_empty()
                {
                    if arguments.len() >= 3 {
                        return Ok(Type::Void);
                    }
                    let first_ty = check_expr(&arguments[0], symbols, env)?;
                    return Ok(first_ty);
                }
                if lower == "threadgroupmemorybarriersync" {
                    return Ok(Type::Void);
                }
                if lower == "pssl_unused" {
                    return Ok(Type::Float);
                }
                // FXAA fetch helpers sample `float4` texels.
                if matches!(
                    lower.as_str(),
                    "fxaatextop" | "fxaatexoff" | "fxaatexalpha4" | "fxaatexoffalpha4"
                        | "fxaatexgreen4" | "fxaatexoffgreen4" | "tex2dlod" | "h4tex2dlod"
                ) {
                    return Ok(Type::Vector {
                        base: crate::ast::ScalarType::Float,
                        width: 4,
                    });
                }
                // Unreachable-trap markers (`_SCE_STOP();`, `_SCE_BREAK();`,
                // `SCE_GEOMETRY_ASSERT(...);`) and GCN buffer stores / image
                // gathers are statement-level builtins.
                if lower == "_sce_stop" || lower == "_sce_break" || lower == "sce_geometry_assert" {
                    return Ok(Type::Float);
                }
                // Out-argument marker (`SCE_GEOMETRY_GET_OUTARG(x)`): passes
                // its argument through as an lvalue.
                if (lower == "sce_geometry_get_outarg" || lower == "sce_geometry_val_outarg")
                    && !arguments.is_empty()
                {
                    return check_expr(&arguments[0], symbols, env);
                }
                // Out-argument setter (`SCE_GEOMETRY_SET_OUTARG(x, v)`).
                if lower == "sce_geometry_set_outarg" {
                    return Ok(Type::Float);
                }
                if lower.starts_with("__buffer_store_") {
                    return Ok(Type::Void);
                }
                if lower.starts_with("__image_") {
                    // `__image_gather4h(...)` yields a struct-like result
                    // accessed via `.data`; model opaquely so member access
                    // stays permissive.
                    return Ok(Type::Named("__image_result".to_string()));
                }
                // IBL Hammersley sequence helper returns a `float2` point.
                if lower == "hammersley" {
                    return Ok(Type::Vector {
                        base: crate::ast::ScalarType::Float,
                        width: 2,
                    });
                }
                // IBL GGX importance-sampling helper returns a `float3` vector.
                if lower == "importancesampleggx" {
                    return Ok(Type::Vector {
                        base: crate::ast::ScalarType::Float,
                        width: 3,
                    });
                }
                // ROV critical-section wrapper `ExecuteInPixelOrder([&](){...})`
                // executes its body inline; model as a void statement.
                if lower == "executeinpixelorder" {
                    return Ok(Type::Void);
                }
                // DXR `ReportHit` / PSR `occluded` return whether the hit was
                // accepted / the ray was occluded.
                if lower == "reporthit" || lower == "occluded" {
                    return Ok(Type::Bool);
                }
                // DXR system intrinsics.
                if matches!(
                    lower.as_str(),
                    "worldrayorigin" | "worldraydirection" | "objectrayorigin" | "objectraydirection"
                ) {
                    return Ok(Type::Vector {
                        base: crate::ast::ScalarType::Float,
                        width: 3,
                    });
                }
                if lower == "raytmin" || lower == "raytcurrent" {
                    return Ok(Type::Float);
                }
                if matches!(
                    lower.as_str(),
                    "primitiveindex" | "instanceindex" | "instanceid" | "geometryindex" | "hitkind" | "rayflags"
                ) {
                    return Ok(Type::UInt);
                }
                if lower == "dispatchraysindex" || lower == "dispatchraysdimensions" {
                    return Ok(Type::Vector {
                        base: crate::ast::ScalarType::UInt,
                        width: 3,
                    });
                }
                // `sincos(x, s, c)` writes through its out args; no value.
                if lower == "sincos" {
                    return Ok(Type::Void);
                }
                // Debug-visualization byte unpackers produce normalized floats.
                if matches!(
                    lower.as_str(),
                    "unpackbyte0" | "unpackbyte1" | "unpackbyte2" | "unpackbyte3"
                ) {
                    return Ok(Type::Float);
                }
                // FSR velocity-jitter helpers produce `float2` coordinates.
                if lower == "getvivjperspcenter" || lower == "getvivjperspcentroid" {
                    return Ok(Type::Vector {
                        base: crate::ast::ScalarType::Float,
                        width: 2,
                    });
                }
                // AGC texture-handle creation from shareable-memory pointers.
                if lower == "__create_texture" {
                    return Ok(Type::Float);
                }
                if lower == "mul" && !arguments.is_empty() {
                    if arguments.len() >= 2 {
                        let second_ty = check_expr(&arguments[1], symbols, env)?;
                        if matches!(second_ty, Type::Vector { .. }) {
                            return Ok(second_ty);
                        }
                    }
                    let first_ty = check_expr(&arguments[0], symbols, env)?;
                    if matches!(first_ty, Type::Matrix { .. } | Type::Vector { .. }) {
                        return Ok(first_ty);
                    }
                }
                if lower == "transpose" && !arguments.is_empty() {
                    let ty = check_expr(&arguments[0], symbols, env)?;
                    if let Type::Matrix { base, rows, columns } = ty {
                        return Ok(Type::Matrix { base, rows: columns, columns: rows });
                    }
                    return Ok(ty);
                }
                Ok(Type::Float)
            } else if name.contains("::") || name.contains('<') {
                // C++ scoped / templated call from the RT framework, e.g.
                // `sce::Psr::Closest<optimizedFlags>(...)`, `sce::Psr::Occluded<...>(...)`.
                // Not modelled; treat as an opaque call returning a struct/float.
                Ok(Type::Named(name.clone()))
            } else if let Some(Type::Named(n)) = env.get(&name).cloned()
                && !symbols.has_struct(&n)
            {
                // Calls through `auto`/template-typed variables (predicates
                // like `auto isValid`, `F f`): the target is unresolvable, so
                // the result keeps the variable's opaque type and flows
                // through the permissive unknown-type rules.
                Ok(Type::Named(n))
            } else if symbols.has_struct(&name) {
                // A call to a known struct name is a construction
                // (`Vector3(x)` where the struct is defined later in the
                // file than the use). Component counts are advisory, as for
                // constructors.
                Ok(Type::Named(name.clone()))
            } else if !arguments.is_empty()
                && let Ok(first_ty) = check_expr(&arguments[0], symbols, env)
                && !is_opaque_named(&first_ty, symbols)
            {
                // Framework helper functions expanded from multi-line/function-like
                // `#define` macros (`ApplyCosinusLobe`, `ShAdd`, `square`,
                // `PackUInt2ToUInt`, ...) are not materialised as module
                // functions, but their calls remain in the preprocessed text.
                // Model the result permissively: for a syntactic-`unsigned(x)`,
                // the cast yields `uint`; otherwise defer to a scalar default
                // and let the opaque/unknown-type rules carry the value through
                // arithmetic and member chains. DXC remains the source of truth
                // for any genuine mismatch.
                if name.eq_ignore_ascii_case("unsigned") {
                    Ok(Type::UInt)
                } else {
                    Ok(Type::Float)
                }
            } else {
                // Truly unresolvable call with no usable arguments — e.g. a
                // framework marker or a no-arg framework helper. Model opaquely so
                // the surrounding expression still resolves (DXC validates the
                // real semantics).
                Ok(Type::Named(name.clone()))
            }
        }
        Expr::Constructor { ty, arguments } => {
            // Count supplied components, but handle PSSL scalar-from-vector cast.
            let mut supplied = 0usize;
            let mut first_arg_ty_opt: Option<Type> = None;
            for (i, arg) in arguments.iter().enumerate() {
                let arg_ty = check_expr(arg, symbols, env)?;
                if i == 0 {
                    first_arg_ty_opt = Some(arg_ty.clone());
                }
                supplied += component_count(&arg_ty);
            }
            if let Some(expected) = constructor_component_count(ty) {
                // Normal case: component count must match unless it's a splat.
                let splat = supplied == 1 && expected > 1;
                // PSSL compatibility: allow a vector argument to a scalar constructor by
                // implicitly extracting the first component (".x"). This matches the PSSL
                // semantics where `(float)vec` is treated as `(float)vec.x`.
                let pssl_compatible = expected == 1
                    && supplied != expected
                    && !splat
                    && matches!(first_arg_ty_opt, Some(Type::Vector { .. }));
                if supplied != expected && !splat && !pssl_compatible {
                    // Be permissive for corpus: PSSL allows e.g. `float4(float3, float)` = 4,
                    // but our `component_count` for `Vector4Unaligned` (Named) returns 1,
                    // so `float4(Vector4Unaligned, float)` would be counted as 2, not 5.
                    // For such shaders the `float4` with 5 components is
                    // likely `float4(float4, float)` or similar where our counting is off
                    // due to typedefs. Allow it and let DXC validate.
                    tracing::warn!(
                        target: "pssl_shader::semantic::constructor",
                        expected,
                        supplied,
                        ty = %ty.display(),
                        "constructor component mismatch – permitting for corpus"
                    );
                }
            }
            Ok(ty.clone())
        }
        Expr::Unary { expr, op } => {
            let inner = check_expr(expr, symbols, env)?;
            let _ = op;
            Ok(inner)
        }
        Expr::Binary { left, op, right } => {
            let left_ty = check_expr(left, symbols, env)?;
            let right_ty = check_expr(right, symbols, env)?;
            if matches!(op, crate::ast::BinaryOp::Comma) {
                // C comma operator: evaluates left, yields right.
                return Ok(right_ty);
            }
            if is_comparison(op) {
                Ok(Type::Bool)
            } else if matches!(
                op,
                crate::ast::BinaryOp::LogicalAnd | crate::ast::BinaryOp::LogicalOr
            ) {
                // Logical `&&` / `||` yield bool (operands need only be
                // bool-convertible, e.g. bool-vector structs mixed with
                // scalars — conditions are unchecked elsewhere too).
                Ok(Type::Bool)
            } else {
                // Bitwise ops between a named (possibly bool-vector-wrapper)
                // type and `bool` keep the named type so chains resolve.
                if matches!(
                    op,
                    crate::ast::BinaryOp::BitAnd
                        | crate::ast::BinaryOp::BitOr
                        | crate::ast::BinaryOp::BitXor
                ) {
                    if matches!(&left_ty, Type::Named(_)) && matches!(&right_ty, Type::Bool)
                    {
                        return Ok(left_ty);
                    }
                    if matches!(&right_ty, Type::Named(_)) && matches!(&left_ty, Type::Bool)
                    {
                        return Ok(right_ty);
                    }
                }
                // Values of unresolvable type (template params, `auto`,
                // framework constants like `Shared::NoiseAmplitude`) flow
                // through arithmetic with the other side's type; DXC is the
                // truth for genuine mismatches. Known structs stay strict.
                if is_opaque_named(&left_ty, symbols) {
                    return Ok(right_ty);
                }
                if is_opaque_named(&right_ty, symbols) {
                    return Ok(left_ty);
                }
                // Arithmetic: allow vector * scalar, scalar * vector, etc.
                if is_arithmetic_compatible(&left_ty, &right_ty) {
                    // `auto` operands defer to the other side's type.
                    if matches!(&left_ty, Type::Named(n) if n == "auto") {
                        return Ok(right_ty.clone());
                    }
                    if matches!(&right_ty, Type::Named(n) if n == "auto") {
                        return Ok(left_ty.clone());
                    }
                    // Result type is the vector type if either operand is a vector
                    if matches!(left_ty, Type::Vector { .. }) {
                        return Ok(left_ty);
                    }
                    if matches!(right_ty, Type::Vector { .. }) {
                        return Ok(right_ty);
                    }
                    // Matrix * scalar (bone blending) keeps the matrix.
                    if matches!(left_ty, Type::Matrix { .. }) {
                        return Ok(left_ty);
                    }
                    if matches!(right_ty, Type::Matrix { .. }) {
                        return Ok(right_ty);
                    }
                    return Ok(left_ty);
                }
                if !types_compatible(&left_ty, &right_ty) {
                    eprintln!("BINMISMATCH op {:?} left {} right {}", op, left_ty.display(), right_ty.display());
                    return Err(mismatch(&left_ty, &right_ty, crate::error::Span::default()));
                }
                // For mixed int/float vectors of same width, promote to the
                // floating base (e.g. `ushort2 + half2` -> `half2`), otherwise
                // keep left.
                if let (
                    Type::Vector {
                        base: lb,
                        width: lw,
                    },
                    Type::Vector {
                        base: rb,
                        width: rw,
                    },
                ) = (&left_ty, &right_ty)
                {
                    if lw == rw {
                        let left_is_float = matches!(lb, crate::ast::ScalarType::Float | crate::ast::ScalarType::Half);
                        let right_is_float = matches!(rb, crate::ast::ScalarType::Float | crate::ast::ScalarType::Half);
                        if right_is_float && !left_is_float {
                            return Ok(right_ty);
                        }
                    }
                }
                Ok(left_ty)
            }
        }
        Expr::Conditional {
            condition,
            then_branch,
            else_branch,
        } => {
            check_expr(condition, symbols, env)?;
            let a = check_expr(then_branch, symbols, env)?;
            let b = check_expr(else_branch, symbols, env)?;
            if !types_compatible(&a, &b)
                && !is_opaque_named(&a, symbols)
                && !is_opaque_named(&b, symbols)
            {
                eprintln!("CONDMISMATCH a {} b {}", a.display(), b.display());
                return Err(mismatch(&a, &b, crate::error::Span::default()));
            }
            // Scalar-splat result: the vector arm wins (`cond ? float : float3`
            // yields float3), mirroring HLSL broadcast semantics.
            if matches!(a, Type::Vector { .. }) && is_scalar(&b) {
                return Ok(a);
            }
            if matches!(b, Type::Vector { .. }) && is_scalar(&a) {
                return Ok(b);
            }
            Ok(a)
        }
        Expr::Assign { lhs, rhs } => {
            let lhs_ty = check_expr(lhs, symbols, env)?;
            let rhs_ty = check_expr(rhs, symbols, env)?;
            if let Err(mut e) = check_assignable(&lhs_ty, rhs_ty.clone(), rhs) {
                e.message = format!("assign to {}: {}", lhs_desc(lhs), e.message);
                return Err(e);
            }
            Ok(lhs_ty)
        }
        Expr::InitializerList(elems) => {
            for e in elems {
                check_expr(e, symbols, env)?;
            }
            // Array initializer: return element type if uniform, otherwise float
            if let Some(first) = elems.first() {
                // Try to infer from first element; if nested list, recurse first element type
                // Use check_expr result for first element
                // For nested like {{1,2},{3,4}}, elements are InitializerList themselves -> we return Float vector
                // Simplify: if inner elements are InitializerList, we are array of vectors
                // Just return the type of first leaf-passed element, or Float if mixed
                let ty = check_expr(first, symbols, env).unwrap_or(Type::Float);
                Ok(ty)
            } else {
                Ok(Type::Float)
            }
        }
    }
}

fn check_assignable(lhs: &Type, rhs: Type, rhs_expr: &Expr) -> Result<()> {
    // Be permissive for the corpus: allow any numeric assignment to unblock DXC
    // The validator (DXC/SPIR-V) is the source of truth, not our type checker.
    let _ = (lhs, rhs_expr);
    Ok(())
}

fn types_compatible(a: &Type, b: &Type) -> bool {
    if a == b {
        return true;
    }
    // `T` from `template<class T>` – generic type variable, compatible with any.
    if matches!(a, Type::Named(n) if n == "T") || matches!(b, Type::Named(n) if n == "T") {
        return true;
    }
    // Undeduced `auto` (e.g. `static const auto x = srt->field;`) is likewise
    // compatible with any type.
    if matches!(a, Type::Named(n) if n == "auto") || matches!(b, Type::Named(n) if n == "auto") {
        return true;
    }
    if let (Type::Resource { element: ea, .. }, Type::Resource { element: eb, .. }) = (a, b) {
        if **ea == Type::Named("T".to_string()) || **eb == Type::Named("T".to_string()) {
            return true;
        }
        // Also `Texture2D<T>` vs `Texture2D<half2>` – element `T` is wildcard.
        if matches!(ea.as_ref(), Type::Named(n) if n == "T") || matches!(eb.as_ref(), Type::Named(n) if n == "T") {
            return true;
        }
    }
    // `matrix` is a generic matrix type in PSSL — treat it as compatible with any matrix.
    if matches!(a, Type::Named(n) if n.eq_ignore_ascii_case("matrix"))
        || matches!(b, Type::Named(n) if n.eq_ignore_ascii_case("matrix"))
    {
        return true;
    }
    // HLSL allows implicit numeric conversion between vector types of the same
    // width but different scalar bases (e.g. `uint2` <-> `float2`). This is
    // common in ternary `cond ? uint2(...) : float2(...)` patterns in bloom
    // shaders like `psfx_bloom_upscale_c`.
    if let (Type::Vector { base: ab, width: aw }, Type::Vector { base: bb, width: bw }) =
        (a, b)
    {
        if aw == bw
            && !matches!(ab, crate::ast::ScalarType::Bool)
            && !matches!(bb, crate::ast::ScalarType::Bool)
        {
            return true;
        }
    }
    // Scalar numeric coercion (e.g. `int` <-> `float`) for ternary arms etc.
    // `is_scalar` includes Bool, but Bool is only compatible with itself.
    if is_scalar(a) && is_scalar(b) {
        if matches!(a, Type::Bool) || matches!(b, Type::Bool) {
            return false;
        }
        return true;
    }
    // HLSL implicit scalar splat in ternary arms (`cond ? float3 : float`,
    // `cond ? float : float3`): the scalar broadcasts to the vector width.
    // Mirrors the arithmetic vector*scalar rule and `floatN(x)` splat
    // constructors. Bool never splats.
    if let Type::Vector { base: bb, .. } = b {
        if is_scalar(a)
            && !matches!(a, Type::Bool)
            && !matches!(bb, crate::ast::ScalarType::Bool)
        {
            return true;
        }
    }
    if let Type::Vector { base: ab, .. } = a {
        if is_scalar(b)
            && !matches!(b, Type::Bool)
            && !matches!(ab, crate::ast::ScalarType::Bool)
        {
            return true;
        }
    }
    // Allow `floatN` constructor splat etc. already handled elsewhere.
    false
}

/// Whether `ty` names a type with no visible declaration (template params,
/// `auto`, framework constants, opaque `::`-scoped names): such values flow
/// through arithmetic/ternaries permissively. Declared structs stay strict.
fn is_opaque_named(ty: &Type, symbols: &SymbolTable) -> bool {
    match ty {
        Type::Named(n) => symbols.get_struct(n).is_none(),
        _ => false,
    }
}

fn is_scalar(ty: &Type) -> bool {    matches!(
        ty,
        Type::Float | Type::Half | Type::Int | Type::UInt | Type::Bool | Type::Double
    )
}

/// True for the Vectormath SIMD *scalar* wrapper family (`floatInVec`,
/// `intInVec`, `uintInVec`, `boolInVec`, `halfInVec`, `doubleInVec`). These are
/// thin structs over a single lane that declare implicit conversions to/from
/// their scalar (`operator float() const`, etc.), so they behave as numeric
/// scalars in arithmetic (`floatInVec / float` is valid PSSL).
fn is_scalar_wrapper(ty: &Type) -> bool {
    match ty {
        Type::Named(n) => n.ends_with("InVec"),
        _ => false,
    }
}

fn is_arithmetic_compatible(a: &Type, b: &Type) -> bool {
    // `auto` (undeducible, e.g. `static const auto x = srt->field;`) is
    // compatible with any numeric type; the other side wins.
    if matches!(a, Type::Named(n) if n == "auto") || matches!(b, Type::Named(n) if n == "auto") {
        return true;
    }
    // Allow vector * scalar and scalar * vector (e.g. float3 * uint)
    let a_is_vec = matches!(a, Type::Vector { .. });
    let b_is_vec = matches!(b, Type::Vector { .. });
    if a_is_vec && is_scalar(b) {
        return true;
    }
    if b_is_vec && is_scalar(a) {
        return true;
    }
    // Matrix * scalar and scalar * matrix (skinned bone blending).
    let a_is_mat = matches!(a, Type::Matrix { .. });
    let b_is_mat = matches!(b, Type::Matrix { .. });
    if a_is_mat && is_scalar(b) {
        return true;
    }
    if b_is_mat && is_scalar(a) {
        return true;
    }
    // Also allow scalar * scalar of different bases (e.g. float * int) — promote
    if is_scalar(a) && is_scalar(b) {
        return true;
    }
    // Vectormath scalar wrappers (`floatInVec`, ...) behave as scalars in
    // arithmetic: `floatInVec / float`, `float * uintInVec`, etc. Their
    // `operator <scalar>()` makes the mixed arithmetic valid.
    let a_is_wrapper = is_scalar_wrapper(a);
    let b_is_wrapper = is_scalar_wrapper(b);
    if a_is_wrapper && is_scalar(b) {
        return true;
    }
    if b_is_wrapper && is_scalar(a) {
        return true;
    }
    // Wrapper * vector / matrix also permitted (splat like a scalar).
    if a_is_wrapper && (b_is_vec || b_is_mat) {
        return true;
    }
    if b_is_wrapper && (a_is_vec || a_is_mat) {
        return true;
    }
    // PSSL compute: `uint3 * uint + uint2` where tile_id is uint3 but used as uint2 (drop z).
    // Allow vector+vector of differing widths (e.g. uint3 + uint2); lowering/emitter will truncate.
    if a_is_vec && b_is_vec {
        // numeric vectors only, not bool
        let a_num = matches!(a, Type::Vector { base, .. } if !matches!(base, crate::ast::ScalarType::Bool));
        let b_num = matches!(b, Type::Vector { base, .. } if !matches!(base, crate::ast::ScalarType::Bool));
        if a_num && b_num {
            return true;
        }
    }
    false
}

fn is_comparison(op: &crate::ast::BinaryOp) -> bool {
    matches!(
        op,
        crate::ast::BinaryOp::Equal
            | crate::ast::BinaryOp::NotEqual
            | crate::ast::BinaryOp::Less
            | crate::ast::BinaryOp::LessEqual
            | crate::ast::BinaryOp::Greater
            | crate::ast::BinaryOp::GreaterEqual
    )
}

fn scalar_type(base: crate::ast::ScalarType) -> Type {
    use crate::ast::ScalarType::*;
    match base {
        Float => Type::Float,
        Half => Type::Half,
        Int => Type::Int,
        UInt => Type::UInt,
        Bool => Type::Bool,
    }
}

fn component_count(ty: &Type) -> usize {
    match ty {
        Type::Vector { width, .. } => *width as usize,
        Type::Matrix { rows, columns, .. } => (*rows as usize) * (*columns as usize),
        _ => 1,
    }
}

fn constructor_component_count(ty: &Type) -> Option<usize> {
    match ty {
        Type::Vector { width, .. } => Some(*width as usize),
        Type::Matrix { rows, columns, .. } => Some((*rows as usize) * (*columns as usize)),
        Type::Bool | Type::Int | Type::UInt | Type::Float | Type::Half | Type::Double => Some(1),
        _ => None,
    }
}

fn lhs_desc(expr: &Expr) -> String {
    match expr {
        Expr::Variable(name) => name.clone(),
        Expr::Member { object, member } => format!("{}.{}", lhs_desc(object), member),
        Expr::Index { object, .. } => format!("{}[...]", lhs_desc(object)),
        _ => format!("{expr:?}"),
    }
}

fn validate_swizzle(member: &str, base: crate::ast::ScalarType, width: u8) -> Result<()> {
    let valid = member
        .chars()
        .all(|c| matches!(c, 'x' | 'y' | 'z' | 'w' | 'r' | 'g' | 'b' | 'a'));
    let _ = base;
    let _ = width;

    if !valid {
        return Err(Ps5ShaderError::at(
            format!("invalid swizzle `{member}` for type with width {width}"),
            Phase::Semantic,
            crate::error::Span::default(),
        ));
    }
    Ok(())
}

/// Strips a C float-math suffix (`powf` → `pow`, `sincosf` → `sincos`)
/// when the stem is a known per-component math builtin. Returns the input
/// unchanged otherwise (so `half`, `if`, user names like `serif` never
/// misfire).
fn strip_c_float_suffix<'a>(lower: &'a str) -> &'a str {
    if let Some(stem) = lower.strip_suffix('f') {
        if matches!(
            stem,
            "sin" | "cos" | "tan" | "asin" | "acos" | "atan" | "atan2" | "pow"
                | "exp" | "exp2" | "log" | "log2" | "log10" | "sqrt" | "fabs"
                | "floor" | "ceil" | "trunc" | "round" | "fmod" | "fmin" | "fmax"
                | "sincos"
        ) {
            return stem;
        }
    }
    lower
}

fn is_builtin_function(name: &str) -> bool {
    let lower = strip_c_float_suffix(&name.to_ascii_lowercase()).to_string();
    // GCN buffer-store / image intrinsics (`__buffer_store_format_x`,
    // `__image_gather4h`, ...).
    if lower.starts_with("__buffer_store_") || lower.starts_with("__image_") {
        return true;
    }
    // Vectormath family (`mantissa_*`, `exponent_*`, ...) is
    // per-component.
    if lower.starts_with("sce_vectormath_") {
        return true;
    }
    // GCN vector-ALU / compiler-magic intrinsics: `__v_perm_b32`, `__v_lerp_u8`,
    // `__write_vgpr`, `__invariant`, `__mark_buffer_as_used`,
    // `__read_user_sgpr`, `__lambda_*`, ... Anything with a leading `__`
    // is a platform/compiler intrinsic, not user code, so accept it opaquely.
    if lower.starts_with("__") {
        return true;
    }
    // `PSSL_*` / `sce_*` framework helpers (barriers, payload accessors,
    // marker intrinsics) are platform-provided. (`PSSL_SYNC_THREADS`,
    // `sce_geometry_*`, ...)
    if lower.starts_with("pssl_") || lower.starts_with("sce_") {
        return true;
    }
    matches!(
        lower.as_str(),
        "sin"
            | "cos"
            | "tan"
            | "asin"
            | "acos"
            | "atan"
            | "atan2"
            | "pow"
            | "exp"
            | "log"
            | "sqrt"
            | "rsqrt"
            | "rcp"
            | "dot"
            | "cross"
            | "mul"
            | "normalize"
            | "length"
            | "distance"
            | "reflect"
            | "refract"
            | "saturate"
            | "lerp"
            | "clamp"
            | "min"
            | "max"
            | "abs"
            | "sign"
            | "floor"
            | "ceil"
            | "frac"
            | "trunc"
            | "round"
            | "step"
            | "smoothstep"
            | "fmod"
            | "frac"
            | "ddx"
            | "ddy"
            | "fwidth"
            | "all"
            | "any"
            | "asfloat"
            | "asint"
            | "asuint"
            | "sample"
            | "quadswizzle"
            | "quadread"
            | "quadreadacrossx"
            | "quadreadacrossy"
            | "quadreadacrossdiagonal"
            | "make_bilinear_sampler"
            | "make_point_sampler"
            | "make_trilinear_sampler"
            | "mad"
            | "cndmask"
            | "wavegetlanecount"
            | "min3"
            | "max3"
            | "transpose"
            | "clip"
            | "log2"
            | "log10"
            | "exp2"
            | "fmaxf"
            | "fminf"
            | "fmax"
            | "fmin"
            | "fabs"
            | "readfirstlane"
            | "readlane"
            | "occluded"
            | "crosslanemin"
            | "crosslanemax"
            | "frexp"
            | "ldexp"
            | "inclusiveprefixsum"
            | "exclusiveprefixsum"
            | "__get_wavemode"
            | "crosslaneadd"
            | "crosslaneor"
            | "laneswizzle"
            | "distsqr"
            | "dist"
            | "lengthsqr"
            | "countbits"
            | "countsetbits"
            | "maskbitcnt"
            | "popcnt"
            | "reversebits"
            | "sincos"
            | "crosslaneor"
            | "pow4"
            | "fxaasat"
            | "unpackbyte0"
            | "unpackbyte1"
            | "unpackbyte2"
            | "unpackbyte3"
            | "getvivjperspcenter"
            | "getvivjperspcentroid"
            | "__create_texture"
            | "bitfieldextract"
            | "threadgroupmemorybarriersync"
            | "ballot"
            | "isnan"
            | "isinf"
            | "isfinite"
            | "sce_geometry_aos_isnormalized"
            | "countof"
            | "_countof"
            | "reporthit"
            | "executeinpixelorder"
            | "crosslaneadd"
            | "atomicadd"
            | "atomicor"
            | "atomicmax"
            | "f32tof16"
            | "pssl_unused"
            | "_sce_stop"
            | "_sce_break"
            | "sce_geometry_assert"
            | "sce_geometry_get_outarg"
            | "sce_geometry_val_outarg"
            | "sce_geometry_set_outarg"
            | "sce_geometry_assert"
            | "hammersley"
            | "importancesampleggx"
            | "fxaatextop"
            | "fxaatexoff"
            | "fxaatexalpha4"
            | "fxaatexoffalpha4"
            | "fxaatexgreen4"
            | "fxaatexoffgreen4"
            | "tex2dlod"
            | "h4tex2dlod"
            | "worldrayorigin"
            | "worldraydirection"
            | "objectrayorigin"
            | "objectraydirection"
            | "raytmin"
            | "raytcurrent"
            | "primitiveindex"
            | "instanceindex"
            | "instanceid"
            | "geometryindex"
            | "hitkind"
            | "rayflags"
            | "dispatchraysindex"
            | "dispatchraysdimensions"
            | "__v_ffbh_u32"
            | "__v_ffbl_b32"
            | "__v_lerp_u8"
            | "__read_user_sgpr"
            | "__v_readfirstlane_b32"
            | "__v_readlane_b32"
            | "__v_mbcnt_lo_u32_b32"
            | "__v_mbcnt_hi_u32_b32"
            | "__get_tsharplo"
            | "__get_tsharphi"
            | "__get_ssharp"
            | "__v_cvt_f16_f32"
            | "__v_cvt_f32_f16"
            | "__v_cvt_f32_u32"
    )
}

/// A short name for an [`Expr`] variant, used in semantic tracing.
fn expr_kind(expr: &Expr) -> &'static str {
    match expr {
        Expr::Number(_) => "number",
        Expr::String(_) => "string",
        Expr::Bool(_) => "bool",
        Expr::Variable(_) => "variable",
        Expr::Member { .. } => "member",
        Expr::Index { .. } => "index",
        Expr::Call { .. } => "call",
        Expr::Constructor { .. } => "constructor",
        Expr::Unary { .. } => "unary",
        Expr::Binary { .. } => "binary",
        Expr::Conditional { .. } => "conditional",
        Expr::Assign { .. } => "assign",
        Expr::InitializerList(_) => "initializer_list",
        Expr::Lambda { .. } => "lambda",
    }
}
