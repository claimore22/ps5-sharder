//! Statement lowering.

use std::collections::HashMap;

use crate::ast::Type;
use crate::error::Result;
use crate::ir::{IrExpr, IrScalar, IrStmt, IrType};
use crate::semantic::to_ir_type;

use super::expr::lower_expr;

pub(crate) fn lower_stmt(
    stmt: &crate::ast::Stmt,
    env: &HashMap<String, crate::ast::Type>,
    resource_fields: &HashMap<(String, String), Type>,
    var_struct: &HashMap<String, String>,
) -> Result<IrStmt> {
    match stmt {
        crate::ast::Stmt::Return(None) => Ok(IrStmt::Return(None)),

        crate::ast::Stmt::Return(Some(expr)) => Ok(IrStmt::Return(Some(lower_expr(
            expr,
            env,
            resource_fields,
            var_struct,
        )))),

        crate::ast::Stmt::Block(statements) => {
            let mut lowered = Vec::new();

            for s in statements {
                lowered.push(lower_stmt(s, env, resource_fields, var_struct)?);
            }

            Ok(IrStmt::Block(lowered))
        }

        crate::ast::Stmt::VarDecl {
            storage: _,
            ty,
            name,
            array_size,
            init,
        } => {
            let mut ir_ty = to_ir_type(ty);
            // `auto` deduction – use initializer type for SamplerState etc.
            if matches!(ty, crate::ast::Type::Named(n) if n == "auto") {
                if let Some(init_expr) = init {
                    if let crate::ast::Expr::Call { function, .. } = init_expr {
                        if let crate::ast::Expr::Variable(fname) = function.as_ref() {
                            let lower = fname.to_ascii_lowercase();
                            if matches!(lower.as_str(), "make_point_sampler" | "make_bilinear_sampler" | "make_trilinear_sampler") {
                                ir_ty = crate::ir::IrType::Named("SamplerState".to_string());
                            }
                        }
                    }
                }
            }
            // Framework sampler construction (`sce::Agc::Core::Sampler()`
            // builder chains, `= {}`, `SamplerState(otherSampler)`) has no
            // HLSL runtime meaning — Vulkan samplers are immutable and
            // externally bound (same rationale as the elided void mutators
            // in `Stmt::Expr`). Emit a bare declaration instead of
            // degrading to `<complex callee>`.
            //
            // A chain selecting a depth-compare function configures a
            // COMPARISON sampler: HLSL has no
            // `SamplerState`→`SamplerComparisonState` conversion
            // (`SamplerComparisonState(samplerDesc)` is `type mismatch`),
            // so declare it as `SamplerComparisonState` directly (the
            // chain content predicts the single downstream use — framework
            // shadow helpers pair compare-chains with `SampleCmp*`).
            let init = match init {
                Some(e) if is_sampler_ir_type(&ir_ty) && is_sampler_builder_init(e) => {
                    if builder_selects_comparison(e) {
                        ir_ty = IrType::Named("SamplerComparisonState".to_string());
                    }
                    None
                }
                other => other
                    .as_ref()
                    .map(|e| lower_expr(e, env, resource_fields, var_struct)),
            };
            // Multi-dimensional scalar locals (`const float M[4][4] =
            // {{...}, ...}` — the parser keeps one dimension) become
            // arrays of vectors (`float4 M[4] = {float4(...), ...}`):
            // `M[a][b]` then indexes a vector component, valid HLSL.
            // Only complete, uniform, scalar-element nests of width
            // 2..=4 qualify; anything else keeps today's loud path.
            let (ir_ty, array_size, init) = match (&ir_ty, array_size, &init) {
                // `double` has no IR scalar spelling: excluded above, so
                // `ir_scalar_of` cannot fail here; the fallback is
                // defensive only.
                (IrType::Float | IrType::Half | IrType::Int | IrType::UInt | IrType::Bool, Some(outer), Some(IrExpr::InitializerList(rows)))
                    if rows.len() as u32 == *outer
                        && row_vector_width(&rows).is_some_and(|w| (2..=4).contains(&w)) =>
                {
                    let width = row_vector_width(&rows).unwrap_or(0);
                    let vec_ty = IrType::Vector {
                        base: ir_scalar_of(&ir_ty).unwrap_or(IrScalar::Float),
                        width,
                    };
                    let vec_rows = rows
                        .iter()
                        .map(|row| match row {
                            IrExpr::InitializerList(elems) => IrExpr::Construct {
                                ty: vec_ty.clone(),
                                arguments: elems.clone(),
                            },
                            other => other.clone(),
                        })
                        .collect();
                    (vec_ty, Some(*outer), Some(IrExpr::InitializerList(vec_rows)))
                }
                _ => (ir_ty, *array_size, init),
            };
            Ok(IrStmt::VarDecl {
                ty: ir_ty,
                // Keep the declaration consistent with the identifier
                // rewriting performed by `rename_sampler_in_stmt`
                // (mirrors the parameter rename in `lower_function`):
                // PSSL may use HLSL-reserved `sampler`/`texture`, so they
                // become `sampler_`/`texture_` everywhere.
                name: if let Some(renamed) = super::rename::reserved_hlsl_rename(name) {
                    renamed.to_string()
                } else {
                    name.clone()
                },
                array_size,
                init,
            })
        }

        crate::ast::Stmt::Expr(expr) => {
            if is_void_sampler_config_call(expr) {
                // `sce::Agc::Core::Sampler` configuration mutators
                // (`sampler.init()`, `sampler.setMipFilterMode(...)`, ...)
                // return void and have no effect on an externally-bound
                // Vulkan `SamplerState`; elide them rather than degrading
                // to `<complex callee>()` (DXC `expected expression`).
                return Ok(IrStmt::Block(Vec::new()));
            }
            if is_assert_call(expr) {
                // PSSL `assert(cond)` is a debug-only check (C release
                // semantics: `NDEBUG` removes it entirely); HLSL has no
                // shader assert intrinsic (DXC `use of undeclared
                // identifier 'assert'`). Elide to an empty block.
                return Ok(IrStmt::Block(Vec::new()));
            }
            // `(void)x` unused-suppressors (`(void)geometry;`) have no runtime
            // effect; HLSL rejects `void(x)` (`variable has incomplete type
            // 'void'`). Drop side-effect-free shapes, keep inner side effects
            // (`(void)foo()` -> `foo();`).
            if let crate::ast::Expr::Constructor { ty, arguments } = expr {
                if matches!(ty, crate::ast::Type::Void) && arguments.len() == 1 {
                    let inner = &arguments[0];
                    let pure = matches!(
                        inner,
                        crate::ast::Expr::Variable(_)
                            | crate::ast::Expr::Number(_)
                            | crate::ast::Expr::Bool(_)
                    ) || matches!(
                        inner,
                        crate::ast::Expr::Member { .. } | crate::ast::Expr::Index { .. }
                    );
                    if pure {
                        return Ok(IrStmt::Block(Vec::new()));
                    }
                    return Ok(IrStmt::Expr(lower_expr(
                        inner,
                        env,
                        resource_fields,
                        var_struct,
                    )));
                }
            }
            if let Some(expanded) =
                lower_getdimensions_fast_stmt(expr, env, resource_fields, var_struct)
            {
                Ok(expanded)
            } else {
                Ok(IrStmt::Expr(lower_expr(
                    expr,
                    env,
                    resource_fields,
                    var_struct,
                )))
            }
        }
        crate::ast::Stmt::Destructure {
            storage: _,
            ty,
            names,
            init,
        } => {
            // Un-desugared leftovers (non-call inits the pre-pass skips):
            // today's block-of-autos behavior, verbatim.
            let stmts = names
                .iter()
                .map(|name| {
                    Ok(IrStmt::VarDecl {
                        ty: to_ir_type(ty),
                        name: name.clone(),
                        array_size: None,
                        init: None,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            // NOTE: `init` is dropped here exactly as the parser used to;
            // reachable through desugar only for call inits.
            let _ = init;
            Ok(IrStmt::Block(stmts))
        }

        crate::ast::Stmt::If {
            condition,
            then_branch,
            else_branch,
        } => {
            let mut then_lowered = Vec::new();

            for s in then_branch {
                then_lowered.push(lower_stmt(s, env, resource_fields, var_struct)?);
            }

            let else_lowered = if let Some(branch) = else_branch {
                let mut v = Vec::new();

                for s in branch {
                    v.push(lower_stmt(s, env, resource_fields, var_struct)?);
                }

                Some(v)
            } else {
                None
            };

            Ok(IrStmt::If {
                condition: lower_expr(condition, env, resource_fields, var_struct),
                then_branch: then_lowered,
                else_branch: else_lowered,
            })
        }

        crate::ast::Stmt::For {
            init,
            condition,
            update,
            body,
        } => {
            let init_lowered = Box::new(lower_stmt(init, env, resource_fields, var_struct)?);
            let condition_lowered = condition
                .as_ref()
                .map(|e| lower_expr(e, env, resource_fields, var_struct));
            let update_lowered = update
                .as_ref()
                .map(|e| lower_expr(e, env, resource_fields, var_struct));

            let mut body_lowered = Vec::new();

            for s in body {
                body_lowered.push(lower_stmt(s, env, resource_fields, var_struct)?);
            }

            Ok(IrStmt::For {
                init: init_lowered,
                condition: condition_lowered,
                update: update_lowered,
                body: body_lowered,
            })
        }

        crate::ast::Stmt::While { condition, body } => {
            let mut body_lowered = Vec::new();

            for s in body {
                body_lowered.push(lower_stmt(s, env, resource_fields, var_struct)?);
            }

            Ok(IrStmt::While {
                condition: lower_expr(condition, env, resource_fields, var_struct),
                body: body_lowered,
            })
        }

        crate::ast::Stmt::Switch { scrutinee, cases } => {
            let mut lowered = Vec::new();
            for case in cases {
                let mut body_lowered = Vec::new();
                for s in &case.body {
                    body_lowered.push(lower_stmt(s, env, resource_fields, var_struct)?);
                }
                lowered.push(crate::ir::SwitchCase {
                    label: case
                        .label
                        .as_ref()
                        .map(|e| lower_expr(e, env, resource_fields, var_struct)),
                    body: body_lowered,
                });
            }
            Ok(IrStmt::Switch {
                scrutinee: lower_expr(scrutinee, env, resource_fields, var_struct),
                cases: lowered,
            })
        }
    }
}

/// Whether an IR type denotes a sampler object (see
/// `crate::framework::is_sampler_type_name` for the set).
fn is_sampler_ir_type(ty: &IrType) -> bool {
    match ty {
        IrType::Named(n) => crate::framework::is_sampler_type_name(n),
        _ => false,
    }
}

/// Uniform row width of a nested initializer list (`{{...}, ...}`):
/// `Some(n)` when every element is a list of the same length `n`.
fn row_vector_width(rows: &[IrExpr]) -> Option<u8> {
    if rows.is_empty() {
        return None;
    }
    let mut width = None;
    for row in rows {
        let IrExpr::InitializerList(elems) = row else {
            return None;
        };
        match width {
            None => width = Some(elems.len()),
            Some(w) if w == elems.len() => {}
            _ => return None,
        }
    }
    width.map(|w| w as u8)
}

/// Scalar base of a scalar IR type for row vectorization.
fn ir_scalar_of(ty: &IrType) -> Option<IrScalar> {
    match ty {
        IrType::Float => Some(IrScalar::Float),
        IrType::Half => Some(IrScalar::Half),
        IrType::Int => Some(IrScalar::Int),
        IrType::UInt => Some(IrScalar::UInt),
        IrType::Bool => Some(IrScalar::Bool),
        // `double` has no IR scalar (and no HLSL double-vector spelling
        // here): multi-dim double arrays keep today's loud path.
        _ => None,
        _ => None,
    }
}

/// Whether an AST type denotes a sampler object (same set as above).
fn is_sampler_ast_type(ty: &Type) -> bool {
    match ty {
        Type::Named(n) => crate::framework::is_sampler_type_name(n),
        _ => false,
    }
}

/// Whether `init` builds a framework sampler object: `= {}`,
/// a `Sampler()` / `SamplerState(x)` constructor, or a fluent
/// `.init().setX().setY()` chain rooted in one. Anything else
/// (notably chains rooted in ordinary variables) is left alone.
fn is_sampler_builder_init(init: &crate::ast::Expr) -> bool {
    use crate::ast::Expr;
    let mut e = init;
    loop {
        match e {
            Expr::InitializerList(elems) => return elems.is_empty(),
            Expr::Constructor { ty, .. } => return is_sampler_ast_type(ty),
            Expr::Call { function, .. } => e = function,
            Expr::Member { object, .. } => e = object,
            // Chain root: the qualified object or the bare `Sampler`
            // (`using namespace sce::Agc::Core`).
            Expr::Variable(name) => {
                return name == "sce::Agc::Core::Sampler" || name == "Sampler"
            }
            _ => return false,
        }
    }
}

/// Whether a sampler builder chain selects a depth-compare function
/// (`.setDepthCompareFunction(...)` anywhere along the fluent spine).
/// Such a chain configures a comparison sampler, not a plain one.
fn builder_selects_comparison(init: &crate::ast::Expr) -> bool {
    use crate::ast::Expr;
    let mut e = init;
    loop {
        match e {
            Expr::Call { function, .. } => {
                if let Expr::Member { object, member } = function.as_ref() {
                    let l = member.to_ascii_lowercase().replace('_', "");
                    if l == "setdepthcomparefunction" {
                        return true;
                    }
                    e = object;
                } else {
                    e = function;
                }
            }
            Expr::Member { object, .. } => e = object,
            Expr::Constructor { .. }
            | Expr::InitializerList(_)
            | Expr::Variable(_) => return false,
            _ => return false,
        }
    }
}

/// Lowers `GetDimensionsFast` / single-out `GetDimensions` call
/// statements into the equivalent HLSL form using explicit integer
/// temporaries, since HLSL's `Texture2D::GetDimensions` requires
/// integer output parameters:
///
/// ```rust
/// uint vec_x;
/// uint vec_y;
/// tex.GetDimensions(vec_x, vec_y);
/// vec = float2(vec_x, vec_y);
/// ```
///
/// Buffers and 1D textures take a single dimension
/// (`buf.GetDimensions(dim)`); an `int` destination goes through a
/// `uint` temporary with an explicit `int()` conversion back (HLSL
/// rejects `int&` out-params: `no matching member function`). A
/// `uint` destination passes through untouched.
///
/// Returns `None` for any other shape (today's loud path).
fn lower_getdimensions_fast_stmt(
    expr: &crate::ast::Expr,
    env: &std::collections::HashMap<String, crate::ast::Type>,
    resource_fields: &std::collections::HashMap<(String, String), crate::ast::Type>,
    var_struct: &std::collections::HashMap<String, String>,
) -> Option<IrStmt> {
    let crate::ast::Expr::Call {
        function,
        arguments,
    } = expr
    else {
        return None;
    };
    let crate::ast::Expr::Member { object, member } = function.as_ref() else {
        return None;
    };
    // Single-dimension form (`buf.GetDimensions(dim)`, either spelling):
    // only for buffer-like / 1D-texture objects (bare variables via
    // the environment, member paths via resource fields).
    if (member.eq_ignore_ascii_case("GetDimensions")
        || member.eq_ignore_ascii_case("GetDimensionsFast"))
        && arguments.len() == 1
    {
        if let Some(kind) = object_kind(object, env, resource_fields, var_struct) {
            let l = kind.to_ascii_lowercase();
            if l.contains("buffer") || l == "texture1d" || l == "rwtexture1d" {
                // Structured buffers take `(count, stride)`; plain buffers
                // and 1D textures take `(dim)` alone.
                let structured = l.contains("structured")
                    || l == "regularbuffer"
                    || l == "rwregularbuffer"
                    || l.starts_with("append")
                    || l.starts_with("consume");
                if let Some(expanded) =
                    lower_single_dimension(object, &arguments[0], env, structured)
                {
                    return Some(expanded);
                }
            }
        }
    }
    if !member.eq_ignore_ascii_case("GetDimensionsFast") {
        return None;
    }
    let crate::ast::Expr::Member { object, member } = function.as_ref() else {
        return None;
    };
    if !member.eq_ignore_ascii_case("GetDimensionsFast") {
        return None;
    }
    let crate::ast::Expr::Variable(tex_name) = object.as_ref() else {
        return None;
    };
    if arguments.len() != 2 {
        return None;
    }

    // Both arguments must be `.x` / `.y` component accesses on the same base
    // variable, e.g. `dims.x` and `dims.y`.
    let x_a = component_of(&arguments[0], "x")?;
    let y_a = component_of(&arguments[1], "y")?;
    if x_a != y_a {
        return None;
    }
    let base = x_a;

    tracing::trace!(
        target: "pssl_shader::lower::call",
        tex = %tex_name,
        base = %base,
        "lowering GetDimensionsFast -> GetDimensions with integer temporaries"
    );

    let width_name = format!("{base}_x");
    let height_name = format!("{base}_y");
    let call_fn = format!(
        "{}.GetDimensions",
        super::rename::rename_member_root(tex_name)
    );

    // `vec = float2(vec_x, vec_y)` — reconstruct the (float2) vector from the
    // integer dimensions. HLSL implicitly converts uint -> float on assignment.
    let dims_construct = IrExpr::Construct {
        ty: IrType::Vector {
            base: IrScalar::Float,
            width: 2,
        },
        arguments: vec![
            IrExpr::Variable(width_name.clone()),
            IrExpr::Variable(height_name.clone()),
        ],
    };
    let assign_back = IrExpr::Assign {
        lhs: Box::new(IrExpr::Variable(base)),
        rhs: Box::new(dims_construct),
    };

    Some(IrStmt::Block(vec![
        IrStmt::VarDecl {
            ty: IrType::UInt,
            name: width_name.clone(),
            array_size: None,
            init: None,
        },
        IrStmt::VarDecl {
            ty: IrType::UInt,
            name: height_name.clone(),
            array_size: None,
            init: None,
        },
        IrStmt::Expr(IrExpr::Call {
            function: call_fn,
            arguments: vec![IrExpr::Variable(width_name), IrExpr::Variable(height_name)],
        }),
        IrStmt::Expr(assign_back),
    ]))
}

/// Resource kind of a call object: bare variables from the environment,
/// `root.member` paths from resource fields (struct of the root via
/// `var_struct`). `None` for anything else.
fn object_kind<'a>(
    object: &crate::ast::Expr,
    env: &'a std::collections::HashMap<String, crate::ast::Type>,
    resource_fields: &'a std::collections::HashMap<(String, String), crate::ast::Type>,
    var_struct: &std::collections::HashMap<String, String>,
) -> Option<&'a str> {
    match object {
        crate::ast::Expr::Variable(name) => match env.get(name) {
            Some(crate::ast::Type::Resource { kind, .. }) => Some(kind.as_str()),
            _ => None,
        },
        crate::ast::Expr::Member { object, member } => {
            if let crate::ast::Expr::Variable(root) = object.as_ref() {
                if let Some(struct_name) = var_struct.get(root) {
                    if let Some(crate::ast::Type::Resource { kind, .. }) =
                        resource_fields.get(&(struct_name.clone(), member.clone()))
                    {
                        return Some(kind.as_str());
                    }
                }
            }
            None
        }
        _ => None,
    }
}

/// Dotted path of a variable/member chain (`srt.m_histogram`), for
/// rebuilding method-call callees that later passes rewrite.
fn dotted_path(object: &crate::ast::Expr) -> Option<String> {
    match object {
        // Reserved renames apply to bare roots (`texture` → `texture_`),
        // matching declarations; struct fields keep spelling (handled by
        // the recursive member append below, which never renames fields).
        crate::ast::Expr::Variable(name) => Some(
            super::rename::reserved_hlsl_rename(name)
                .map(|r| r.to_string())
                .unwrap_or_else(|| name.clone()),
        ),
        crate::ast::Expr::Member { object, member } => {
            dotted_path(object).map(|p| format!("{p}.{member}"))
        }
        _ => None,
    }
}

/// Single-dimension `GetDimensions` (`buf.GetDimensions(dim)`): `uint`
/// destinations pass through; anything else goes through a `uint`
/// temporary with an explicit `int()` conversion back. `None` for
/// non-lvalue destinations or unknown shapes (loud).
fn lower_single_dimension(
    object: &crate::ast::Expr,
    arg: &crate::ast::Expr,
    env: &std::collections::HashMap<String, crate::ast::Type>,
    structured: bool,
) -> Option<IrStmt> {
    use crate::ast::{Expr, Type};
    // Lvalue destinations only (literals and other shapes are not
    // lvalues: loud). A `uint` variable binds directly with no
    // temporary; `int` variables and member paths go through a `uint`
    // temporary with an explicit `int()` conversion back (`uint m = ...`
    // through a member would also work, but the uniform temp shape
    // keeps one code path).
    match arg {
        Expr::Variable(name) => match env.get(name) {
            // `uint` binds directly: no temporary needed. Structured
            // buffers still need the `(count, stride)` pair, so a stride
            // dummy joins the call.
            Some(Type::UInt) => {
                let obj_path = dotted_path(object)?;
                if structured {
                    let stride = format!("{name}_stride");
                    return Some(IrStmt::Block(vec![
                        IrStmt::VarDecl {
                            ty: IrType::UInt,
                            name: stride.clone(),
                            array_size: None,
                            init: None,
                        },
                        IrStmt::Expr(IrExpr::Call {
                            function: format!("{obj_path}.GetDimensions"),
                            arguments: vec![
                                IrExpr::Variable(name.clone()),
                                IrExpr::Variable(stride),
                            ],
                        }),
                    ]));
                }
                return Some(IrStmt::Expr(IrExpr::Call {
                    function: format!("{obj_path}.GetDimensions"),
                    arguments: vec![IrExpr::Variable(name.clone())],
                }));
            }
            Some(Type::Int) => {}
            _ => return None,
        },
        Expr::Member { .. } => {}
        _ => return None,
    }
    // Temporary name from the destination root (mirrors the `_x`/`_y`
    // convention of the 2D path above).
    let root = match arg {
        Expr::Variable(name) => name.clone(),
        Expr::Member { object, .. } => match object.as_ref() {
            Expr::Variable(name) => name.clone(),
            _ => return None,
        },
        _ => return None,
    };
    let tmp = format!("{root}_dim");
    let obj_path = dotted_path(object)?;
    let call_fn = format!("{obj_path}.GetDimensions");
    // `{arg} = int({tmp});` — explicit narrowing (warning-free).
    let assign_back = IrExpr::Assign {
        lhs: Box::new(match arg {
            Expr::Variable(name) => IrExpr::Variable(name.clone()),
            Expr::Member { object, member } => IrExpr::Member {
                object: Box::new(match object.as_ref() {
                    Expr::Variable(name) => IrExpr::Variable(name.clone()),
                    _ => return None,
                }),
                member: member.clone(),
            },
            _ => return None,
        }),
        rhs: Box::new(IrExpr::Construct {
            ty: IrType::Int,
            arguments: vec![IrExpr::Variable(tmp.clone())],
        }),
    };
    // Structured buffers report `(count, stride)`; the stride is discarded
    // via a dummy temp. Plain buffers take the count alone.
    if structured {
        let stride = format!("{root}_stride");
        return Some(IrStmt::Block(vec![
            IrStmt::VarDecl {
                ty: IrType::UInt,
                name: tmp.clone(),
                array_size: None,
                init: None,
            },
            IrStmt::VarDecl {
                ty: IrType::UInt,
                name: stride.clone(),
                array_size: None,
                init: None,
            },
            IrStmt::Expr(IrExpr::Call {
                function: call_fn,
                arguments: vec![IrExpr::Variable(tmp), IrExpr::Variable(stride)],
            }),
            IrStmt::Expr(assign_back),
        ]));
    }
    Some(IrStmt::Block(vec![
        IrStmt::VarDecl {
            ty: IrType::UInt,
            name: tmp.clone(),
            array_size: None,
            init: None,
        },
        IrStmt::Expr(IrExpr::Call {
            function: call_fn,
            arguments: vec![IrExpr::Variable(tmp)],
        }),
        IrStmt::Expr(assign_back),
    ]))
}

/// Whether `expr` is a bare `assert(...)` debug-check call.
fn is_assert_call(expr: &crate::ast::Expr) -> bool {
    let crate::ast::Expr::Call { function, .. } = expr else {
        return false;
    };
    match function.as_ref() {
        crate::ast::Expr::Variable(name) => name.eq_ignore_ascii_case("assert"),
        _ => false,
    }
}

/// Whether `expr` is a void-returning `sce::Agc::Core::Sampler` configuration
/// mutator call (`sampler.init()`, `sampler.setMipFilterMode(...)`,
/// `sampler.setWrapMode(...)`, ...). These configure a PSSL sampler object
/// that has no runtime counterpart in Vulkan (samplers are immutable and
/// externally bound), so the statement is a no-op in the emitted HLSL.
fn is_void_sampler_config_call(expr: &crate::ast::Expr) -> bool {
    let crate::ast::Expr::Call { function, .. } = expr else {
        return false;
    };
    let crate::ast::Expr::Member { member, .. } = function.as_ref() else {
        return false;
    };
    let l = member.to_ascii_lowercase().replace('_', "");
    matches!(
        l.as_str(),
        "init"
            | "setmipfiltermode"
            | "setxyfiltermode"
            | "setzfiltermode"
            | "setwrapmode"
            | "setwrapmodex"
            | "setwrapmodey"
            | "setcubewrap"
            | "settexturecoordinates"
            | "setbordercolor"
            | "setdepthcomparefunction"
            | "setanisotropy"
            | "setanisotropyratio"
            | "setanisotropythreshold"
            | "setfilterreductionmode"
            | "setlodbias"
            | "setminlod"
            | "setmaxlod"
    )
}

/// Retypes locals/params configured as depth-compare samplers by a
/// standalone builder chain (`sampler.init()...setDepthCompareFunction()`
/// as its own statement, as in the cross_gen shadow helper) to
/// `SamplerComparisonState`. Builder chains inside initializers already
/// select comparison at the declaration above; a split declaration like
/// `sce::Agc::Core::Sampler sampler; sampler.init()...` needs this
/// whole-body pass instead (`SampleCmp*` requires the comparison type).
/// Only sampler-typed declarations are touched; anything else keeps
/// today's loud path.
pub(crate) fn retype_comparison_samplers(function: &mut crate::ast::Function) {
    use std::collections::HashSet;
    fn chain_root(expr: &crate::ast::Expr) -> Option<String> {
        let mut found = false;
        let mut cur = expr;
        loop {
            match cur {
                crate::ast::Expr::Call { function, .. } => {
                    let crate::ast::Expr::Member { object, member } = function.as_ref() else {
                        return None;
                    };
                    if member.to_ascii_lowercase().replace('_', "") == "setdepthcomparefunction" {
                        found = true;
                    }
                    cur = object;
                }
                crate::ast::Expr::Variable(v) => {
                    return if found { Some(v.clone()) } else { None };
                }
                _ => return None,
            }
        }
    }
    fn collect(stmt: &crate::ast::Stmt, out: &mut HashSet<String>) {
        match stmt {
            crate::ast::Stmt::Expr(e) => {
                if let Some(root) = chain_root(e) {
                    out.insert(root);
                }
            }
            crate::ast::Stmt::Block(ss) => {
                for s in ss {
                    collect(s, out);
                }
            }
            crate::ast::Stmt::If {
                condition: _,
                then_branch,
                else_branch,
            } => {
                for s in then_branch {
                    collect(s, out);
                }
                if let Some(b) = else_branch {
                    for s in b {
                        collect(s, out);
                    }
                }
            }
            crate::ast::Stmt::For {
                init,
                body,
                ..
            } => {
                collect(init, out);
                for s in body {
                    collect(s, out);
                }
            }
            crate::ast::Stmt::While { body, .. } => {
                for s in body {
                    collect(s, out);
                }
            }
            crate::ast::Stmt::Switch { cases, .. } => {
                for c in cases {
                    for s in &c.body {
                        collect(s, out);
                    }
                }
            }
            _ => {}
        }
    }
    fn patch(stmt: &mut crate::ast::Stmt, names: &HashSet<String>) {
        match stmt {
            crate::ast::Stmt::VarDecl { ty, name, .. } => {
                if names.contains(name)
                    && let crate::ast::Type::Named(n) = ty
                    && crate::framework::is_sampler_type_name(n)
                {
                    *ty = crate::ast::Type::Named("SamplerComparisonState".to_string());
                }
            }
            crate::ast::Stmt::Block(ss) => {
                for s in ss {
                    patch(s, names);
                }
            }
            crate::ast::Stmt::If {
                then_branch,
                else_branch,
                ..
            } => {
                for s in then_branch {
                    patch(s, names);
                }
                if let Some(b) = else_branch {
                    for s in b {
                        patch(s, names);
                    }
                }
            }
            crate::ast::Stmt::For { init, body, .. } => {
                patch(init, names);
                for s in body {
                    patch(s, names);
                }
            }
            crate::ast::Stmt::While { body, .. } => {
                for s in body {
                    patch(s, names);
                }
            }
            crate::ast::Stmt::Switch { cases, .. } => {
                for c in cases {
                    for s in &mut c.body {
                        patch(s, names);
                    }
                }
            }
            _ => {}
        }
    }
    let mut names = HashSet::new();
    for s in &function.body {
        collect(s, &mut names);
    }
    if names.is_empty() {
        return;
    }
    for p in &mut function.parameters {
        if names.contains(&p.name)
            && let crate::ast::Type::Named(n) = &p.ty
            && crate::framework::is_sampler_type_name(n)
        {
            p.ty = crate::ast::Type::Named("SamplerComparisonState".to_string());
        }
    }
    for s in &mut function.body {
        patch(s, &names);
    }
}

/// If `expr` is `object.member` where `object` is a variable, return the
/// variable name when `member` equals `want`. Otherwise return `None`.
fn component_of(expr: &crate::ast::Expr, want: &str) -> Option<String> {
    let crate::ast::Expr::Member { object, member } = expr else {
        return None;
    };
    if member != want {
        return None;
    }
    let crate::ast::Expr::Variable(name) = object.as_ref() else {
        return None;
    };
    Some(name.clone())
}
