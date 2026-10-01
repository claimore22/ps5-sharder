//! Struct-constructor returns (`return S(a, b, ...)`).
//!
//! HLSL has no user-struct constructors (DXC: "constructors only defined
//! for numeric base types"), so a returned struct construction is
//! desugared into a temporary plus memberwise assignments — the same
//! shape the emitter already uses for struct brace-init declarations
//! (`emit_struct_init`), but returns never reach that path:
//! ```text
//! return S(a0, a1);
//! // →
//! S __retinit_0;
//! __retinit_0.f0 = a0;
//! __retinit_0.f1 = a1;
//! return __retinit_0;
//! ```
//! Missing trailing fields zero-fill numerics (recursing into structs),
//! mirroring the emitter; resources and unknown members stay unassigned.
//! Only complete, flat argument lists qualify — anything else keeps
//! failing loudly at DXC.

use crate::ir::{IrExpr, IrLiteral, IrStmt, IrType, ShaderModule, StructDecl};

/// Desugars struct-constructor returns in every function body. Shaders
/// without one are untouched.
pub(crate) fn lower_struct_returns(ir: &mut ShaderModule) {
    let mut counter = 0u32;
    for function in &mut ir.functions {
        let ret_struct = match &function.return_type {
            IrType::Named(n) => Some(n.clone()),
            _ => None,
        };
        lower_returns_in_stmts(&mut function.body, &ir.structs, &ret_struct, &mut counter);
    }
}

fn lower_returns_in_stmts(
    stmts: &mut Vec<IrStmt>,
    structs: &[StructDecl],
    ret_struct: &Option<String>,
    counter: &mut u32,
) {
    let mut i = 0;
    while i < stmts.len() {
        // Struct-constructor returns and zero-struct declarations both
        // expand to several statements, spliced into the parent list.
        // Empty-list array inits fill with zeros in place.
        let expanded = expand_struct_return(&stmts[i], structs, ret_struct, counter)
            .or_else(|| expand_zero_struct(&stmts[i], structs));
        if expanded.is_none() {
            expand_empty_array_init(&mut stmts[i]);
        }
        match expanded {
            Some(expanded) => {
                let n = expanded.len();
                stmts.splice(i..=i, expanded);
                i += n;
            }
            None => {
                recurse_returns(&mut stmts[i], structs, ret_struct, counter);
                i += 1;
            }
        }
    }
}

/// Expands `return S(args...)` (or a typeless `return {...}` when the
/// function returns known struct `S`) into temp + assigns + return.
/// `None` for any other statement.
fn expand_struct_return(
    stmt: &IrStmt,
    structs: &[StructDecl],
    ret_struct: &Option<String>,
    counter: &mut u32,
) -> Option<Vec<IrStmt>> {
    let IrStmt::Return(Some(value)) = stmt else {
        return None;
    };
    // Resolve the struct and its arguments: either an explicit `S(...)`
    // construction of the returned struct, or a typeless `{...}` list.
    // A third shape initializes a SINGLE-field struct from one numeric
    // value (`return float4(1,1,1,1);` in a `PixelOutputForward`
    // function whose only field is `float4 m_color`): PSSL accepts the
    // obvious memberwise construction, HLSL has no such conversion, so
    // it becomes the one member init. Multi-field structs stay loud
    // (ambiguous), as do non-numeric values (a same-typed struct
    // variable must pass through untouched).
    fn is_numeric_ty(ty: &IrType) -> bool {
        matches!(
            ty,
            IrType::Bool
                | IrType::Int
                | IrType::UInt
                | IrType::Float
                | IrType::Half
                | IrType::Double
                | IrType::Vector { .. }
                | IrType::Matrix { .. }
        )
    }
    let (struct_name, args) = match value {
        IrExpr::Construct {
            ty: IrType::Named(s),
            arguments,
        } if Some(s) == ret_struct.as_ref() => (s.clone(), arguments.clone()),
        IrExpr::InitializerList(arguments)
            if matches!(ret_struct, Some(s) if structs.iter().any(|d| &d.name == s)) =>
        {
            (ret_struct.clone().unwrap_or_default(), arguments.clone())
        }
        _ => {
            let s = ret_struct.as_ref()?;
            let decl = structs.iter().find(|d| &d.name == s)?;
            let [field] = decl.fields.as_slice() else {
                return None;
            };
            if field.array_size.is_some() || !is_numeric_ty(&field.ty) {
                return None;
            }
            let value_is_numeric = match value {
                IrExpr::Construct { ty, .. } => is_numeric_ty(ty),
                IrExpr::Literal(IrLiteral::Number(_)) | IrExpr::Literal(IrLiteral::Bool(_)) => true,
                _ => false,
            };
            if !value_is_numeric {
                return None;
            }
            (s.clone(), vec![value.clone()])
        }
    };
    let decl = structs.iter().find(|d| &d.name == &struct_name)?;
    if args.len() > decl.fields.len()
        || args
            .iter()
            .any(|a| matches!(a, IrExpr::InitializerList(_)))
    {
        return None;
    }
    let temp = format!("__retinit_{counter}");
    *counter += 1;
    let mut expanded = vec![IrStmt::VarDecl {
        ty: IrType::Named(struct_name),
        name: temp.clone(),
        array_size: None,
        init: None,
    }];
    for (i, field) in decl.fields.iter().enumerate() {
        let target = IrExpr::Member {
            object: Box::new(IrExpr::Variable(temp.clone())),
            member: field.name.clone(),
        };
        match args.get(i) {
            Some(init) => expanded.push(IrStmt::Expr(IrExpr::Assign {
                lhs: Box::new(target),
                rhs: Box::new(init.clone()),
            })),
            None => push_zero_fill(&mut expanded, target, &field.ty, structs, field.array_size, 0),
        }
    }
    expanded.push(IrStmt::Return(Some(IrExpr::Variable(temp))));
    Some(expanded)
}

/// Whether a literal is a zero (`0`, `0.0`, `0f`, ...).
fn is_zero_literal(expr: &IrExpr) -> bool {
    if let IrExpr::Literal(IrLiteral::Number(lex)) = expr {
        let t = lex.trim().to_ascii_lowercase();
        let t = t.strip_suffix('f').unwrap_or(&t);
        let t = t.strip_suffix('u').unwrap_or(t);
        let t = t.strip_suffix('l').unwrap_or(t);
        if let Ok(v) = t.parse::<f64>() {
            return v == 0.0;
        }
    }
    false
}

/// Expands `S name = (S)0;` (C-style struct zero-cast, e.g.
/// `PBRInfo ret = (PBRInfo)0;`), the bare form `S name = 0;` (e.g.
/// `VS_OUT output = 0;`, which DXC rejects as `literal int` init), and
/// the empty-list form `S name = {};` into a bare declaration plus
/// memberwise zero assignments (sized arrays fill per element). An
/// `auto` declaration takes the cast's struct (`auto x = 0` and
/// `auto x = {}` stay loud: the struct is unknowable). `None` for any
/// other statement.
fn expand_zero_struct(stmt: &IrStmt, structs: &[StructDecl]) -> Option<Vec<IrStmt>> {
    let IrStmt::VarDecl {
        ty,
        name,
        array_size,
        init: Some(init),
        ..
    } = stmt
    else {
        return None;
    };
    let struct_name: &String = match (ty, init) {
        (
            IrType::Named(n),
            IrExpr::Construct {
                ty: IrType::Named(s),
                arguments,
            },
        ) if (n == "auto" || n == s)
            && arguments.len() == 1
            && is_zero_literal(&arguments[0]) =>
        {
            s
        }
        (IrType::Named(n), init) if is_zero_literal(init) => n,
        (IrType::Named(n), IrExpr::InitializerList(elems)) if elems.is_empty() => n,
        _ => return None,
    };
    if structs.iter().find(|d| &d.name == struct_name).is_none() {
        return None;
    }
    let mut expanded = vec![IrStmt::VarDecl {
        ty: IrType::Named(struct_name.clone()),
        name: name.clone(),
        array_size: *array_size,
        init: None,
    }];
    push_zero_fill(
        &mut expanded,
        IrExpr::Variable(name.clone()),
        &IrType::Named(struct_name.clone()),
        structs,
        *array_size,
        0,
    );
    Some(expanded)
}

/// Fills an empty-list numeric array init in place (`uint a[2] = {};`
/// → `uint a[2] = { 0, 0 };`): DXC rejects partial/empty aggregate
/// inits, while missing elements would default to zero anyway. Scalars
/// and vectors keep the single `{ 0 }` the emitter produces; matrices,
/// structs (memberwise path above), and resources stay loud.
fn expand_empty_array_init(stmt: &mut IrStmt) {
    let IrStmt::VarDecl {
        ty,
        array_size: Some(n),
        init: Some(IrExpr::InitializerList(elems)),
        ..
    } = stmt
    else {
        return;
    };
    if !elems.is_empty() {
        return;
    }
    let width = match ty {
        IrType::Int
        | IrType::UInt
        | IrType::Float
        | IrType::Half
        | IrType::Bool
        | IrType::Double => 1,
        IrType::Vector { width, .. } => *width as usize,
        _ => return,
    };
    let total = (*n as usize).saturating_mul(width);
    if total == 0 {
        return;
    }
    *elems = (0..total)
        .map(|_| IrExpr::Literal(IrLiteral::Number("0".to_string())))
        .collect();
}

/// Appends `target = 0` fills for numeric types (HLSL splats the scalar
/// into vectors/matrices), recursing into known structs and arrays.
/// Resources and unknown members stay unassigned: nothing valid can
/// initialize them here.
fn push_zero_fill(
    out: &mut Vec<IrStmt>,
    target: IrExpr,
    ty: &IrType,
    structs: &[StructDecl],
    array_size: Option<u32>,
    depth: usize,
) {
    if let Some(n) = array_size {
        for i in 0..n {
            push_zero_fill(
                out,
                IrExpr::Index {
                    object: Box::new(target.clone()),
                    index: Box::new(IrExpr::Literal(IrLiteral::Number(i.to_string()))),
                },
                ty,
                structs,
                None,
                depth,
            );
        }
        return;
    }
    match ty {
        IrType::Int
        | IrType::UInt
        | IrType::Float
        | IrType::Half
        | IrType::Bool
        | IrType::Double
        | IrType::Vector { .. }
        | IrType::Matrix { .. } => out.push(IrStmt::Expr(IrExpr::Assign {
            lhs: Box::new(target),
            rhs: Box::new(IrExpr::Literal(IrLiteral::Number("0".to_string()))),
        })),
        IrType::Named(n) => {
            if depth >= 8 {
                return;
            }
            if let Some(sdecl) = structs.iter().find(|s| &s.name == n) {
                for field in &sdecl.fields {
                    push_zero_fill(
                        out,
                        IrExpr::Member {
                            object: Box::new(target.clone()),
                            member: field.name.clone(),
                        },
                        &field.ty,
                        structs,
                        field.array_size,
                        depth + 1,
                    );
                }
            }
        }
        _ => {}
    }
}

fn recurse_returns(
    stmt: &mut IrStmt,
    structs: &[StructDecl],
    ret_struct: &Option<String>,
    counter: &mut u32,
) {
    match stmt {
        IrStmt::Block(inner) => lower_returns_in_stmts(inner, structs, ret_struct, counter),
        IrStmt::If {
            then_branch,
            else_branch,
            ..
        } => {
            lower_returns_in_stmts(then_branch, structs, ret_struct, counter);
            if let Some(else_branch) = else_branch {
                lower_returns_in_stmts(else_branch, structs, ret_struct, counter);
            }
        }
        IrStmt::For { init, body, .. } => {
            recurse_returns(init, structs, ret_struct, counter);
            lower_returns_in_stmts(body, structs, ret_struct, counter);
        }
        IrStmt::While { body, .. } => lower_returns_in_stmts(body, structs, ret_struct, counter),
        IrStmt::Switch { cases, .. } => {
            for case in cases {
                lower_returns_in_stmts(&mut case.body, structs, ret_struct, counter);
            }
        }
        _ => {}
    }
}
