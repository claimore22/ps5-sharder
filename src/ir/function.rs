use super::types::IrType;
use serde::{Deserialize, Serialize};

/// A function in the Shader IR.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Function {
    pub name: String,
    pub return_type: IrType,
    /// Raw PSSL return semantic (e.g. "S_TARGET_OUTPUT0").
    pub return_semantic: Option<String>,
    pub parameters: Vec<Parameter>,
    pub body: Vec<IrStmt>,
    pub num_threads: Option<(u32, u32, u32)>,
    /// Out-of-class / in-struct scope (`Scope` in `Ret Scope::method`
    /// or methods declared inside `struct Scope`): emitted as an HLSL
    /// struct method, not a free function. Absent in older snapshots.
    #[serde(default)]
    pub scope: Option<String>,
    /// Geometry-shader `[maxvertexcount(N)]` bound. Absent in older
    /// snapshots.
    #[serde(default)]
    pub max_vertex_count: Option<u32>,
    /// Mesh-shader `[outputtopology("triangle")]` (from PSSL
    /// `[output_primitive_type("tri")]`). Absent in older snapshots.
    #[serde(default)]
    pub output_topology: Option<String>,
    /// Mesh-shader `[num_verts(N)]` bound (max vertices the entry can
    /// export; sizes the `out vertices` object). Absent in older
    /// snapshots.
    #[serde(default)]
    pub mesh_vert_count: Option<u32>,
    /// Mesh-shader `[num_prims(N)]` bound (max primitives the entry can
    /// export; sizes the `out indices` object). Absent in older
    /// snapshots.
    #[serde(default)]
    pub mesh_prim_count: Option<u32>,
    /// Hull-shader `[domain("quad")]` (from PSSL
    /// `[DOMAIN_PATCH_TYPE("quad")]`). Absent in older snapshots.
    #[serde(default)]
    pub hull_domain: Option<String>,
    /// Hull-shader `[partitioning("integer")]` (from PSSL
    /// `[PARTITIONING_TYPE("integer")]`). Absent in older snapshots.
    #[serde(default)]
    pub hull_partitioning: Option<String>,
    /// Hull-shader `[outputcontrolpoints(N)]` (from PSSL
    /// `[OUTPUT_CONTROL_POINTS(N)]`). Absent in older snapshots.
    #[serde(default)]
    pub hull_control_points: Option<u32>,
    /// Hull-shader `[patchconstantfunc("F")]` (from PSSL
    /// `[PATCH_CONSTANT_FUNC("F")]`): the patch-constant function name,
    /// pinned against dead-code stripping. Absent in older snapshots.
    #[serde(default)]
    pub hull_patch_func: Option<String>,
    /// Hull-shader `[maxtessfactor(N)]` (from PSSL
    /// `[MAX_TESS_FACTOR(N)]`). Absent in older snapshots.
    #[serde(default)]
    pub hull_max_tess: Option<u32>,
}

/// One `case label:` / `default:` group inside an IR `switch`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SwitchCase {
    /// `None` is `default`.
    pub label: Option<super::IrExpr>,
    pub body: Vec<IrStmt>,
}

/// A function parameter in the IR.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Parameter {
    pub ty: IrType,
    pub name: String,
    pub semantic: Option<String>,
    /// Array dimension (`float3 L[4]` => Some(4), Some(0) = unsized).
    pub array_size: Option<u32>,
    /// Direction qualifier (`in`/`out`/`inout`, e.g. GS stream params).
    /// Absent in older snapshots.
    #[serde(default)]
    pub direction: Option<String>,
    /// Geometry-shader input primitive (`point`, `line`, …): `Point T x`.
    /// Absent in older snapshots.
    #[serde(default)]
    pub gs_primitive: Option<String>,
    /// Lowered C++ default argument (see `crate::ast::Parameter`): call
    /// sites with fewer arguments are filled during lowering. Absent in
    /// older snapshots.
    #[serde(default)]
    pub default: Option<super::IrExpr>,
    /// Explicit `uniform` qualifier carried from the AST (host-provided
    /// entry constants, cbuffer-hoisted). Absent in older snapshots.
    #[serde(default)]
    pub is_uniform: bool,
    /// Mesh-shader output object (`out vertices` / `out indices`): the
    /// HLSL object kind when this parameter is a mesh output array
    /// (e.g. `Some("vertices")`). Absent in older snapshots.
    #[serde(default)]
    pub mesh_object: Option<String>,
}

/// A statement in the IR function body.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum IrStmt {
    /// `return [value];`
    Return(Option<super::IrExpr>),
    /// `{ ... }`
    Block(Vec<IrStmt>),
    /// `type name [array_size] [= init];`
    VarDecl {
        ty: super::types::IrType,
        name: String,
        array_size: Option<u32>,
        init: Option<super::IrExpr>,
    },
    /// `if (cond) { ... } else { ... }`
    If {
        condition: super::IrExpr,
        then_branch: Vec<IrStmt>,
        else_branch: Option<Vec<IrStmt>>,
    },
    /// `for (init; cond; update) { body }`
    For {
        init: Box<IrStmt>,
        condition: Option<super::IrExpr>,
        update: Option<super::IrExpr>,
        body: Vec<IrStmt>,
    },
    /// `while (cond) { body }`
    While {
        condition: super::IrExpr,
        body: Vec<IrStmt>,
    },
    /// `switch (scrutinee) { case label: ...; default: ...; }`
    /// (`label: None` is `default`). HLSL `switch` has C fallthrough
    /// semantics, matching PSSL exactly, so bodies lower verbatim.
    Switch {
        scrutinee: super::IrExpr,
        cases: Vec<SwitchCase>,
    },
    /// A bare expression statement (including assignments).
    Expr(super::IrExpr),
}

/// Collects callee name strings (`foo`, `obj.method`, `Scope::method`)
/// called anywhere in a function body. Shared by lowering (method
/// stripping) and emission (method gating).
pub fn function_calls(f: &Function) -> Vec<String> {
    fn expr_calls(e: &super::IrExpr, out: &mut Vec<String>) {
        use super::IrExpr;
        match e {
            IrExpr::Call { function, arguments } => {
                out.push(function.clone());
                for a in arguments {
                    expr_calls(a, out);
                }
            }
            IrExpr::Variable(_) | IrExpr::Literal(_) => {}
            IrExpr::Member { object, .. } => expr_calls(object, out),
            IrExpr::Index { object, index } | IrExpr::ResourceIndex { resource: object, index } => {
                expr_calls(object, out);
                expr_calls(index, out);
            }
            IrExpr::ResourceLoad {
                resource,
                coordinates,
                sample,
            } => {
                expr_calls(resource, out);
                expr_calls(coordinates, out);
                if let Some(s) = sample {
                    expr_calls(s, out);
                }
            }
            IrExpr::Construct { arguments, .. } => {
                for a in arguments {
                    expr_calls(a, out);
                }
            }
            IrExpr::Unary { expr: inner, .. } => expr_calls(inner, out),
            IrExpr::Binary { left, right, .. } => {
                expr_calls(left, out);
                expr_calls(right, out);
            }
            IrExpr::Conditional {
                condition,
                then_branch,
                else_branch,
            } => {
                expr_calls(condition, out);
                expr_calls(then_branch, out);
                expr_calls(else_branch, out);
            }
            IrExpr::Assign { lhs, rhs } => {
                expr_calls(lhs, out);
                expr_calls(rhs, out);
            }
            IrExpr::InitializerList(elems) => {
                for x in elems {
                    expr_calls(x, out);
                }
            }
        }
    }
    fn stmt_calls(s: &IrStmt, out: &mut Vec<String>) {
        match s {
            IrStmt::Return(Some(e)) | IrStmt::Expr(e) => expr_calls(e, out),
            IrStmt::Return(None) => {}
            IrStmt::VarDecl { init: Some(e), .. } => expr_calls(e, out),
            IrStmt::VarDecl { init: None, .. } => {}
            IrStmt::If {
                condition,
                then_branch,
                else_branch,
            } => {
                expr_calls(condition, out);
                for s in then_branch {
                    stmt_calls(s, out);
                }
                if let Some(b) = else_branch {
                    for s in b {
                        stmt_calls(s, out);
                    }
                }
            }
            IrStmt::For {
                init,
                condition,
                update,
                body,
            } => {
                stmt_calls(init, out);
                if let Some(c) = condition {
                    expr_calls(c, out);
                }
                if let Some(u) = update {
                    expr_calls(u, out);
                }
                for s in body {
                    stmt_calls(s, out);
                }
            }
            IrStmt::While { condition, body } => {
                expr_calls(condition, out);
                for s in body {
                    stmt_calls(s, out);
                }
            }
            IrStmt::Switch { scrutinee, cases } => {
                expr_calls(scrutinee, out);
                for c in cases {
                    if let Some(label) = &c.label {
                        expr_calls(label, out);
                    }
                    for s in &c.body {
                        stmt_calls(s, out);
                    }
                }
            }
            IrStmt::Block(ss) => {
                for s in ss {
                    stmt_calls(s, out);
                }
            }
        }
    }
    let mut out = Vec::new();
    for s in &f.body {
        stmt_calls(s, &mut out);
    }
    out
}

/// Struct methods reachable from entry + free functions, as (struct,
/// method) pairs (fixpoint over the call graph). Uncalled methods stay
/// dropped: emitting dead methods regresses passing shaders on latent
/// body issues, and their references pollute hoisting decisions.
pub fn reachable_methods(
    ir: &super::ShaderModule,
) -> std::collections::HashSet<(String, String)> {
    use std::collections::{HashMap, HashSet};
    // Declared structs' method names.
    let mut by_struct: HashMap<&str, HashSet<&str>> = HashMap::new();
    for f in &ir.functions {
        if let Some(s) = f.scope.as_deref() {
            by_struct.entry(s).or_default().insert(f.name.as_str());
        }
    }
    let mut reachable: HashSet<(String, String)> = HashSet::new();
    // Roots: every free function body (all emit unconditionally), plus
    // scoped-into-unknown ones (same free behavior).
    let mut queue: Vec<&Function> = ir
        .functions
        .iter()
        .filter(|f| {
            f.scope.is_none()
                || f.scope
                    .as_deref()
                    .is_some_and(|s| !by_struct.contains_key(s))
        })
        .collect();
    // A dotted `x.y` includes method `y` in every struct declaring it
    // (object types are not tracked here); `S::y` pins struct `S`.
    // Bare calls inside a method resolve to the same struct (C++ member
    // scope); bare calls in free functions never resolve to methods.
    let mut changed = true;
    while changed {
        changed = false;
        let mut next = Vec::new();
        for f in queue.drain(..) {
            for target in function_calls(f) {
                if let Some(dot) = target.rfind('.') {
                    let method = &target[dot + 1..];
                    if method.is_empty() {
                        continue;
                    }
                    for (s, methods) in &by_struct {
                        if methods.contains(method)
                            && reachable.insert((s.to_string(), method.to_string()))
                        {
                            changed = true;
                        }
                    }
                } else if let Some(scope_pos) = target.rfind("::") {
                    let (scope, method) = target.split_at(scope_pos);
                    let method = &method[2..];
                    if by_struct
                        .get(scope)
                        .is_some_and(|methods| methods.contains(method))
                        && reachable.insert((scope.to_string(), method.to_string()))
                    {
                        changed = true;
                    }
                } else if let Some(scope) = f.scope.as_deref() {
                    if by_struct
                        .get(scope)
                        .is_some_and(|methods| methods.contains(target.as_str()))
                        && reachable.insert((scope.to_string(), target.clone()))
                    {
                        changed = true;
                    }
                }
            }
        }
        // Newly included methods may call further methods.
        for f in &ir.functions {
            if let Some(scope) = f.scope.as_deref() {
                if reachable.contains(&(scope.to_string(), f.name.clone())) {
                    next.push(f);
                }
            }
        }
        queue = next;
    }
    reachable
}
