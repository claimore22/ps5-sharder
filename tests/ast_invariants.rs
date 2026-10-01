//! Generic AST structural-invariant tests.
//!
//! The parser must not merely *succeed*; the resulting [`Module`] must be
//! internally well-formed and preserve the source construct's structure
//! (callee + arguments, member base + member, index base + index, etc.).
//! These tests are independent of semantics and make no reference to any
//! particular shader in the corpus.
//!
//! Three layers:
//!   * [`validate_module_invariants`] — walks any AST and asserts it is
//!     structurally well-formed (no erased/malformed nodes).
//!   * structural regression tests — parse a small snippet and assert the
//!     exact AST shape.
//!   * shape-preservation tests — confirm the parser did not silently drop
//!     sub-expressions (arguments, operands, indices, members).

use ps5_shader::ast::{BinaryOp, Expr, Module, Stmt, Type, UnaryOp};
use ps5_shader::parse_shader;

// ---------------------------------------------------------------------------
// Generic AST well-formedness validator
// ---------------------------------------------------------------------------

/// Validate that every expression node in a parsed module is internally
/// well-formed. Returns `Vec<String>` of violations (empty == OK).
fn validate_module_invariants(module: &Module) -> Vec<String> {
    let mut violations = Vec::new();
    for func in &module.functions {
        if func.name.is_empty() {
            violations.push("function with empty name".to_string());
        }
        for stmt in &func.body {
            validate_stmt(stmt, &mut violations);
        }
    }
    for global in &module.globals {
        if global.name.is_empty() {
            violations.push("global with empty name".to_string());
        }
    }
    for resource in &module.resources {
        if resource.name.is_empty() {
            violations.push("resource with empty name".to_string());
        }
    }
    for s in &module.structs {
        if s.name.is_empty() {
            violations.push("struct with empty name".to_string());
        }
        for field in &s.fields {
            validate_type(&field.ty, "struct field", &mut violations);
        }
    }
    violations
}

fn validate_stmt(stmt: &Stmt, violations: &mut Vec<String>) {
    match stmt {
        Stmt::Return(expr) => {
            if let Some(e) = expr {
                validate_expr(e, "return", violations);
            }
        }
        Stmt::If { condition, then_branch, else_branch } => {
            validate_expr(condition, "if-condition", violations);
            for s in then_branch {
                validate_stmt(s, violations);
            }
            if let Some(else_blk) = else_branch {
                for s in else_blk {
                    validate_stmt(s, violations);
                }
            }
        }
        Stmt::Block(stmts) => {
            for s in stmts {
                validate_stmt(s, violations);
            }
        }
        Stmt::For { init, condition, update, body } => {
            validate_stmt(init, violations);
            if let Some(c) = condition {
                validate_expr(c, "for-condition", violations);
            }
            if let Some(u) = update {
                validate_expr(u, "for-update", violations);
            }
            for s in body {
                validate_stmt(s, violations);
            }
        }
        Stmt::While { condition, body } => {
            validate_expr(condition, "while-condition", violations);
            for s in body {
                validate_stmt(s, violations);
            }
        }
        Stmt::Switch { scrutinee, cases } => {
            validate_expr(scrutinee, "switch-scrutinee", violations);
            for case in cases {
                if let Some(label) = &case.label {
                    validate_expr(label, "switch-case-label", violations);
                }
                for s in &case.body {
                    validate_stmt(s, violations);
                }
            }
        }
        Stmt::VarDecl { ty, name, init, .. } => {
            validate_type(ty, "var-decl", violations);
            if name.is_empty() {
                violations.push("var-decl with empty name".to_string());
            }
            if let Some(e) = init {
                validate_expr(e, "var-init", violations);
            }
        }
        Stmt::Destructure { ty, names, init, .. } => {
            validate_type(ty, "destructure", violations);
            if names.is_empty() {
                violations.push("destructure with no names".to_string());
            }
            if let Some(e) = init {
                validate_expr(e, "destructure-init", violations);
            }
        }
        Stmt::Expr(e) => validate_expr(e, "expr-stmt", violations),
    }
}

fn validate_expr(expr: &Expr, ctx: &str, violations: &mut Vec<String>) {
    match expr {
        Expr::Call { function, arguments } => {
            // A call must carry a callee that is itself a well-formed
            // expression (never an erased/placeholder).
            validate_expr(function, "callee", violations);
            for (i, arg) in arguments.iter().enumerate() {
                validate_expr(arg, &format!("{ctx} arg {i}"), violations);
            }
        }
        Expr::Constructor { ty, arguments } => {
            validate_type(ty, "constructor", violations);
            for (i, arg) in arguments.iter().enumerate() {
                validate_expr(arg, &format!("{ctx} ctor-arg {i}"), violations);
            }
        }
        Expr::Member { object, member } => {
            validate_expr(object, "member-base", violations);
            if member.is_empty() {
                violations.push(format!("{ctx}: member access with empty member name"));
            }
        }
        Expr::Index { object, index } => {
            validate_expr(object, "index-base", violations);
            validate_expr(index, "index", violations);
        }
        Expr::Assign { lhs, rhs } => {
            validate_expr(lhs, "assign-lhs", violations);
            validate_expr(rhs, "assign-rhs", violations);
        }
        Expr::Unary { expr, .. } => validate_expr(expr, "unary-operand", violations),
        Expr::Binary { left, right, .. } => {
            validate_expr(left, "binary-lhs", violations);
            validate_expr(right, "binary-rhs", violations);
        }
        Expr::Conditional { condition, then_branch, else_branch } => {
            validate_expr(condition, "cond-condition", violations);
            validate_expr(then_branch, "cond-then", violations);
            validate_expr(else_branch, "cond-else", violations);
        }
        Expr::InitializerList(elems) => {
            for (i, e) in elems.iter().enumerate() {
                validate_expr(e, &format!("{ctx} init-list {i}"), violations);
            }
        }
        Expr::Lambda { body, .. } => {
            for s in body {
                validate_stmt(s, violations);
            }
        }
        Expr::Number(_) | Expr::String(_) | Expr::Bool(_) | Expr::Variable(_) => {}
    }
}

fn validate_type(ty: &Type, ctx: &str, violations: &mut Vec<String>) {
    match ty {
        Type::Named(name) if name.is_empty() => {
            violations.push(format!("{ctx}: empty named type"));
        }
        Type::Vector { width, .. } if *width == 0 => {
            violations.push(format!("{ctx}: zero-width vector type"));
        }
        Type::Matrix { rows, columns, .. } if *rows == 0 || *columns == 0 => {
            violations.push(format!("{ctx}: zero-dimension matrix type"));
        }
        Type::Resource { element, .. } => validate_type(element, "resource-element", violations),
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

/// Parse and return the single function's body (panicking on mis-shape).
fn parse_single_body(source: &str) -> Vec<Stmt> {
    let module = parse_shader(source).expect("parse failed");
    assert_eq!(
        module.functions.len(),
        1,
        "expected exactly one function, got {}",
        module.functions.len()
    );
    module.functions[0].body.clone()
}

/// The statement inside a `return <expr>;`.
fn return_expr(stmts: &[Stmt]) -> &Expr {
    match stmts {
        [Stmt::Return(Some(e))] => e,
        [Stmt::Return(None)] => panic!("expected a return value, got bare return"),
        other => panic!("expected a single return statement, got {other:?}"),
    }
}

fn as_call(e: &Expr) -> (&Expr, &[Expr]) {
    match e {
        Expr::Call { function, arguments } => (function, arguments),
        other => panic!("expected a Call, got {other:?}"),
    }
}

fn as_var(e: &Expr) -> &str {
    match e {
        Expr::Variable(name) => name,
        other => panic!("expected a Variable, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Structural regression tests
// ---------------------------------------------------------------------------

#[test]
fn call_preserves_callee_and_arguments() {
    let body = parse_single_body("void f() { return foo(a, b); }");
    let e = return_expr(&body);
    let (callee, args) = as_call(e);
    assert_eq!(as_var(callee), "foo");
    assert_eq!(args.len(), 2);
    assert_eq!(as_var(&args[0]), "a");
    assert_eq!(as_var(&args[1]), "b");
}

#[test]
fn nested_calls_are_nested() {
    let body = parse_single_body("void f() { return foo(bar(c)); }");
    let e = return_expr(&body);
    let (callee, args) = as_call(e);
    assert_eq!(as_var(callee), "foo");
    assert_eq!(args.len(), 1);
    let (inner_callee, inner_args) = as_call(&args[0]);
    assert_eq!(as_var(inner_callee), "bar");
    assert_eq!(as_var(&inner_args[0]), "c");
}

#[test]
fn member_access_preserves_base_and_member() {
    let body = parse_single_body("void f() { return obj.field; }");
    match return_expr(&body) {
        Expr::Member { object, member } => {
            assert_eq!(as_var(object), "obj");
            assert_eq!(member, "field");
        }
        other => panic!("expected Member, got {other:?}"),
    }
}

#[test]
fn chained_member_access_preserves_chain() {
    let body = parse_single_body("void f() { return a.b.c; }");
    match return_expr(&body) {
        Expr::Member { object, member } => {
            assert_eq!(member, "c");
            match object.as_ref() {
                Expr::Member { object: inner, member: mid } => {
                    assert_eq!(mid, "b");
                    assert_eq!(as_var(inner), "a");
                }
                other => panic!("expected inner Member, got {other:?}"),
            }
        }
        other => panic!("expected Member, got {other:?}"),
    }
}

#[test]
fn swizzle_is_member_access() {
    let body = parse_single_body("void f() { return v.xy; }");
    match return_expr(&body) {
        Expr::Member { object, member } => {
            assert_eq!(as_var(object), "v");
            assert_eq!(member, "xy");
        }
        other => panic!("expected Member swizzle, got {other:?}"),
    }
}

#[test]
fn index_preserves_base_and_index() {
    let body = parse_single_body("void f() { return arr[i]; }");
    match return_expr(&body) {
        Expr::Index { object, index } => {
            assert_eq!(as_var(object), "arr");
            assert_eq!(as_var(index), "i");
        }
        other => panic!("expected Index, got {other:?}"),
    }
}

#[test]
fn index_of_member_preserves_structure() {
    let body = parse_single_body("void f() { return obj.buf[3]; }");
    match return_expr(&body) {
        Expr::Index { object, index } => {
            match object.as_ref() {
                Expr::Member { object: base, member } => {
                    assert_eq!(as_var(base), "obj");
                    assert_eq!(member, "buf");
                }
                other => panic!("expected Member base, got {other:?}"),
            }
            assert!(matches!(index.as_ref(), Expr::Number(_)));
        }
        other => panic!("expected Index, got {other:?}"),
    }
}

#[test]
fn binary_preserves_operands() {
    let body = parse_single_body("void f() { return lhs + rhs; }");
    match return_expr(&body) {
        Expr::Binary { left, op, right } => {
            assert_eq!(op, &BinaryOp::Add);
            assert_eq!(as_var(left), "lhs");
            assert_eq!(as_var(right), "rhs");
        }
        other => panic!("expected Binary, got {other:?}"),
    }
}

#[test]
fn unary_preserves_operand() {
    let body = parse_single_body("void f() { return -x; }");
    match return_expr(&body) {
        Expr::Unary { op, expr } => {
            assert_eq!(op, &UnaryOp::Minus);
            assert_eq!(as_var(expr), "x");
        }
        other => panic!("expected Unary, got {other:?}"),
    }
}

#[test]
fn cast_preserves_inner_expression() {
    let body = parse_single_body("void f() { return (float)x; }");
    match return_expr(&body) {
        Expr::Constructor { ty, arguments } => {
            assert_eq!(ty, &Type::Float);
            assert_eq!(arguments.len(), 1);
            assert_eq!(as_var(&arguments[0]), "x");
        }
        other => panic!("expected Constructor cast, got {other:?}"),
    }
}

#[test]
fn constructor_preserves_type_and_arguments() {
    let body = parse_single_body("void f() { return float3(a, 1.0, z); }");
    match return_expr(&body) {
        Expr::Constructor { ty, arguments } => {
            assert_eq!(
                ty,
                &Type::Vector { base: ps5_shader::ast::ScalarType::Float, width: 3 }
            );
            assert_eq!(arguments.len(), 3);
            assert_eq!(as_var(&arguments[0]), "a");
            assert_eq!(as_var(&arguments[2]), "z");
        }
        other => panic!("expected Constructor, got {other:?}"),
    }
}

#[test]
fn comma_expression_preserves_both_sides() {
    // The C comma operator is legal in assignment-statement context
    // (`a = 1, b = 2;`), not in a bare `return a, b;`.
    let body = parse_single_body("void f() { a = 1, b = 2; }");
    match body.as_slice() {
        [Stmt::Expr(Expr::Binary { left, op, right })] => {
            assert_eq!(op, &BinaryOp::Comma);
            assert!(matches!(left.as_ref(), Expr::Assign { .. }));
            assert!(matches!(right.as_ref(), Expr::Assign { .. }));
        }
        other => panic!("expected a single comma-assignment expr stmt, got {other:?}"),
    }
}

#[test]
fn conditional_preserves_three_branches() {
    let body = parse_single_body("void f() { return c ? t : e; }");
    match return_expr(&body) {
        Expr::Conditional { condition, then_branch, else_branch } => {
            assert_eq!(as_var(condition), "c");
            assert_eq!(as_var(then_branch), "t");
            assert_eq!(as_var(else_branch), "e");
        }
        other => panic!("expected Conditional, got {other:?}"),
    }
}

#[test]
fn assignment_preserves_lhs_and_rhs() {
    let body = parse_single_body("void f() { return x = y; }");
    match return_expr(&body) {
        Expr::Assign { lhs, rhs } => {
            assert_eq!(as_var(lhs), "x");
            assert_eq!(as_var(rhs), "y");
        }
        other => panic!("expected Assign, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Declarations / parameters / resources
// ---------------------------------------------------------------------------

#[test]
fn function_parameters_are_preserved() {
    let module = parse_shader(
        "float4 main(float3 pos : S_POSITION, out uint idx) { return float4(0,0,0,1); }",
    )
    .expect("parse failed");
    let func = &module.functions[0];
    assert_eq!(func.name, "main");
    assert_eq!(func.parameters.len(), 2);
    assert_eq!(func.parameters[0].name, "pos");
    assert_eq!(func.parameters[1].name, "idx");
    assert!(func.parameters[1].direction.is_some());
}

#[test]
fn var_decl_preserves_name_type_and_init() {
    let body = parse_single_body("void f() { float scale = 2.0; return scale; }");
    match &body[0] {
        Stmt::VarDecl { ty, name, init, .. } => {
            assert_eq!(ty, &Type::Float);
            assert_eq!(name, "scale");
            assert!(init.is_some());
        }
        other => panic!("expected VarDecl, got {other:?}"),
    }
}

#[test]
fn resource_decl_preserves_kind_and_name() {
    let module = parse_shader("Texture2D<float4> tex : register(t0);").expect("parse failed");
    assert_eq!(module.resources.len(), 1);
    let res = &module.resources[0];
    assert_eq!(res.name, "tex");
    assert_eq!(res.kind.to_ascii_lowercase(), "texture2d");
}

// ---------------------------------------------------------------------------
// Invariant validation regressions (shape must be well-formed)
// ---------------------------------------------------------------------------

#[test]
fn all_constructs_produce_wellformed_ast() {
    // A single function exercising every expression form; after parsing, the
    // generic invariant validator must report no violations.
    let source = r#"
struct S { float x; float4 arr[4]; };
Texture2D<float4> tex : register(t0);
float g_scale;

float4 mix(float3 a, uint i) {
    float s = g_scale;
    S obj;
    obj.x = s;
    float4 v = tex.Load(int3(i, 0, 0));
    float m = obj.arr[i].w;
    float neg = -s;
    float combined = s + m * 2.0;
    float chosen = (i > 0) ? s : m;
    for (uint k = 0; k < 4; ++k) { s += k; }
    while (s < 10.0) { s += 1.0; }
    if (s > 5.0) { s = 5.0; } else { s = 0.0; }
    return float4(a, s, chosen, 1.0);
}
"#;
    let module = parse_shader(source).expect("parse failed");
    let violations = validate_module_invariants(&module);
    assert!(violations.is_empty(), "AST invariant violations: {violations:?}");
}

#[test]
fn deep_nesting_is_wellformed() {
    // Heavily nested calls / members / indices must still stay well-formed.
    let source = "void f() { return a.b[ c.d(e) ].x( g.h(i) ); }";
    let module = parse_shader(source).expect("parse failed");
    let violations = validate_module_invariants(&module);
    assert!(violations.is_empty(), "AST invariant violations: {violations:?}");
}

#[test]
fn deep_nesting_preserves_innermost_leaf() {
    // The innermost node of the nested chain must not be erased. Walk:
    // a.b[c.d(e)].x(g.h(i)) is a Call(callee=Member(Index(Member(a,b), Call(c.d, e)), x), args=[Call(g.h, i)]).
    let body = parse_single_body("void f() { return a.b[ c.d(e) ].x( g.h(i) ); }");
    let outer = return_expr(&body);
    let (_callee, args) = as_call(outer);
    // The single argument is the inner call g.h(i).
    let (inner_callee, inner_args) = as_call(&args[0]);
    match inner_callee {
        Expr::Member { object: g, member } => {
            assert_eq!(as_var(g), "g");
            assert_eq!(member, "h");
        }
        other => panic!("expected Member callee g.h, got {other:?}"),
    }
    assert_eq!(as_var(&inner_args[0]), "i");
}