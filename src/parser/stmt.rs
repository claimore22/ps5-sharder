//! Statement parsing.

use super::Parser;
use crate::ast::{Expr, StorageClass, Stmt};
use crate::error::Result;
use crate::token::Token;

impl Parser {
    /// True when a `case`/`default` identifier at the cursor is really a
    /// plain identifier use (a variable literally named `case`/`default`),
    /// not a switch label: the next token can only continue an
    /// expression statement (`=`, `(`, `;`, `,`, …), never a label.
    fn label_is_plain_identifier(&self) -> bool {
        let lex = self.current_lexeme();
        let next = self.tokens.get(self.index + 1).map(|t| &t.token);
        if lex.eq_ignore_ascii_case("default") {
            // A `default:` label is always followed by a single colon.
            return !matches!(next, Some(Token::Colon));
        }
        // `case <expr> :` — an expression statement on a variable named
        // `case` shows `=`/`(`/`;`/`,`/`)`/`]`/`}` next instead.
        matches!(
            next,
            Some(
                Token::Equal
                    | Token::LParen
                    | Token::Semi
                    | Token::Comma
                    | Token::RParen
                    | Token::RBracket
                    | Token::RBrace
            )
        )
    }

    pub(crate) fn parse_block(&mut self) -> Result<Vec<Stmt>> {
        self.expect(Token::LBrace)?;
        let mut statements = Vec::new();
        while !self.peek_is(Token::RBrace) {
            statements.push(self.parse_stmt()?);
        }
        self.expect(Token::RBrace)?;
        Ok(statements)
    }

    pub(crate) fn parse_stmt(&mut self) -> Result<Stmt> {
        // Skip leading HLSL statement-control attributes, e.g. `[unroll]`,
        // `[loop]`, `[branch]`, `[flatten]` before a statement (esp. loops).
        loop {
            if !self.peek_is(Token::LBracket) {
                break;
            }
            let attr_name = self.lexeme_after(1);
            if matches!(
                attr_name.to_lowercase().as_str(),
                "unroll"
                    | "loop"
                    | "branch"
                    | "flatten"
                    | "fastopt"
                    | "nofastopt"
                    | "optimize"
                    | "nooptimize"
                    | "forcecase"
                    | "call"
                    | "allow_uav_condition"
                    | "noextrapolation"
                    | "isolate"
            ) {
                self.advance(); // '['
                self.advance(); // name
                self.expect(Token::RBracket)?;
            } else {
                break;
            }
        }

        if self.peek_is(Token::Return) {
            self.advance();
            if self.peek_is(Token::Semi) {
                self.advance();
                return Ok(Stmt::Return(None));
            }
            let expr = self.parse_assignment()?;
            self.expect_semi()?;
            return Ok(Stmt::Return(Some(expr)));
        }

        // C++ `using namespace X;` / `using X = Y;` inside a function body
        // (e.g. `using namespace sce::Got;`) — compile-time only, no runtime
        // statement. Consume through `;` like the module-scope handling.
        // Guarded to a following identifier so a call/assignment to something
        // literally named `using` still parses as an expression statement.
        if self.peek_is(Token::Identifier)
            && self.current_lexeme().eq_ignore_ascii_case("using")
            && self.peek_next_is(Token::Identifier)
        {
            while !self.eof() && !self.peek_is(Token::Semi) {
                self.advance();
            }
            self.expect_semi()?;
            return Ok(Stmt::Block(vec![]));
        }

        if self.peek_is(Token::Break) {
            self.advance();
            self.expect_semi()?;
            return Ok(Stmt::Expr(Expr::Variable("break".to_string())));
        }

        if self.peek_is(Token::Continue) {
            self.advance();
            self.expect_semi()?;
            return Ok(Stmt::Expr(Expr::Variable("continue".to_string())));
        }

        if self.peek_is(Token::Discard) {
            self.advance();
            self.expect_semi()?;
            return Ok(Stmt::Expr(Expr::Variable("discard".to_string())));
        }

        if self.peek_is(Token::Identifier)
            && self.current_lexeme().eq_ignore_ascii_case("switch")
        {
            self.advance(); // 'switch'
            self.expect(Token::LParen)?;
            let scrutinee = self.parse_assignment()?;
            self.expect(Token::RParen)?;
            if !self.peek_is(Token::LBrace) {
                // Braceless `switch (x) stmt;` — the body runs once.
                let body = self.parse_stmt()?;
                return Ok(Stmt::Block(vec![body]));
            }
            self.advance(); // '{'
            let mut cases: Vec<crate::ast::SwitchCase> = Vec::new();
            let mut current: Option<crate::ast::SwitchCase> = None;
            // Statements before the first label (unreachable in C, legal
            // grammar): run once, so they prepend the first case body; a
            // labelless switch body degrades to a plain block.
            let mut pending: Vec<Stmt> = Vec::new();
            while !self.peek_is(Token::RBrace) && !self.eof() {
                if self.peek_is(Token::Identifier)
                    && (self.current_lexeme().eq_ignore_ascii_case("case")
                        || self.current_lexeme().eq_ignore_ascii_case("default"))
                    && !self.label_is_plain_identifier()
                {
                    let is_default =
                        self.current_lexeme().eq_ignore_ascii_case("default");
                    self.advance(); // 'case' / 'default'
                    let label = if is_default {
                        None
                    } else {
                        Some(self.parse_assignment()?)
                    };
                    self.expect(Token::Colon)?;
                    if let Some(c) = current.take() {
                        cases.push(c);
                    }
                    current = Some(crate::ast::SwitchCase {
                        label,
                        body: Vec::new(),
                    });
                    continue;
                }
                let s = self.parse_stmt()?;
                if let Some(c) = current.as_mut() {
                    c.body.push(s);
                } else {
                    pending.push(s);
                }
            }
            self.expect(Token::RBrace)?;
            if let Some(c) = current.take() {
                cases.push(c);
            }
            if cases.is_empty() {
                return Ok(Stmt::Block(pending));
            }
            if !pending.is_empty() {
                let mut first = pending;
                first.extend(cases[0].body.drain(..));
                cases[0].body = first;
            }
            return Ok(Stmt::Switch {
                scrutinee: Box::new(scrutinee),
                cases,
            });
        }

        if self.peek_is(Token::Identifier)
            && self.current_lexeme().eq_ignore_ascii_case("asm")
        {
            self.advance(); // 'asm'
            if self.peek_is(Token::LBrace) {
                self.skip_brace_block()?;
            } else {
                // `asm` with single instruction/semicolon — skip until `;`
                while !self.eof() && !self.peek_is(Token::Semi) {
                    self.advance();
                }
                if self.peek_is(Token::Semi) {
                    self.advance();
                }
            }
            if self.peek_is(Token::Semi) {
                self.advance();
            }
            return Ok(Stmt::Block(vec![]));
        }

        if self.peek_is(Token::If) {
            return self.parse_if();
        }

        if self.peek_is(Token::For) {
            return self.parse_for();
        }

        // `do { ... } while (...);` — `do` lexes as a plain identifier
        // (there is no `Do` token). Desugared to
        // `while (true) { body; if (!cond) break; }` through existing
        // stages (semantic models `break` as an expression statement).
        if self.peek_is(Token::Identifier) && self.current_lexeme().eq_ignore_ascii_case("do") {
            return self.parse_do();
        }

        if self.peek_is(Token::While) {
            return self.parse_while();
        }

        if self.peek_is(Token::Semi) {
            // Empty statement `;` or stray `;` after `}` (e.g. `for (...) { ... };`)
            self.advance();
            return Ok(Stmt::Block(vec![]));
        }

        // Exact lowercase match: a type named `Union` must not take this path.
        if self.peek_is(Token::Identifier) && self.current_lexeme() == "union" {
            self.advance();
            if self.peek_is(Token::LBrace) {
                self.skip_brace_block()?;
            } else if self.peek_is(Token::Identifier) {
                // Record the union name so a later `Union u;` parses as a
                // declaration (members stay opaque via unknown-struct rules).
                self.struct_names.insert(self.current_lexeme());
                self.advance();
                if self.peek_is(Token::LBrace) {
                    self.skip_brace_block()?;
                }
            }
            let mut var_name: Option<String> = None;
            if self.peek_is(Token::Identifier) {
                // Same guard as for `struct`: only consume as instance when
                // followed by `;`, `[`, `=`, `,` or `:` — never a `::` scope.
                let after = self.tokens.get(self.index + 1).map(|t| t.token.clone());
                let is_scope = matches!(after, Some(Token::Colon))
                    && self
                        .tokens
                        .get(self.index + 2)
                        .map(|t| t.token == Token::Colon)
                        .unwrap_or(false);
                let is_instance = !is_scope
                    && matches!(
                        after,
                        Some(Token::Semi)
                            | Some(Token::LBracket)
                            | Some(Token::Equal)
                            | Some(Token::Comma)
                            | Some(Token::Colon)
                    );
                if is_instance {
                    var_name = Some(self.current_lexeme());
                    self.advance();
                    while self.peek_is(Token::LBracket) {
                        self.advance();
                        if !self.peek_is(Token::RBracket) {
                            let _ = self.parse_assignment();
                        }
                        self.expect(Token::RBracket)?;
                    }
                }
            }
            if self.peek_is(Token::Semi) {
                self.advance();
            }
            if let Some(name) = var_name {
                return Ok(Stmt::VarDecl {
                    storage: None,
                    ty: crate::ast::Type::Named("union".to_string()),
                    name,
                    array_size: None,
                    init: None,
                });
            } else {
                return Ok(Stmt::Block(vec![]));
            }
        }

        if self.peek_is(Token::Struct) {
            // Function-local struct declaration: parse for real and lift
            // to module scope under a counter-mangled name (locals never
            // escape their function; bare `output` would collide across
            // functions). Uses in this body resolve through
            // `local_struct_renames` (reset per function in
            // `parse_function`); the declaration drains via
            // `pending_structs`.
            let saved = self.index;
            let saved_names = self.struct_names.clone();
            let saved_pending = self.pending_structs.len();
            let saved_template_params = self.template_type_params.clone();
            // Mangled name of the struct just defined, if any (forward
            // declarations and legacy-skipped shapes define nothing).
            let mut lifted: Option<String> = None;
            match self.parse_struct() {
                Ok(mut decl) => {
                    if !decl.fields.is_empty() {
                        let mangled =
                            format!("{}__local{}", decl.name, self.local_struct_counter);
                        self.local_struct_counter += 1;
                        self.struct_names.insert(mangled.clone());
                        self.local_struct_renames
                            .insert(decl.name.clone(), mangled.clone());
                        decl.name = mangled.clone();
                        self.pending_structs.push(decl);
                        lifted = Some(mangled);
                    }
                }
                Err(_) => {
                    // Unparseable shape: keep the old skip behavior so
                    // Parse coverage never regresses.
                    self.index = saved;
                    self.struct_names = saved_names;
                    self.pending_structs.truncate(saved_pending);
                    self.template_type_params = saved_template_params;
                    self.advance(); // 'struct'
                    if self.peek_is(Token::Identifier) {
                        self.advance();
                    }
                    if self.peek_is(Token::LBrace) {
                        self.skip_brace_block()?;
                    } else if self.peek_is(Token::Semi) {
                        self.advance();
                        return Ok(Stmt::Block(vec![]));
                    }
                }
            }
            if self.peek_is(Token::Semi) {
                self.advance();
                return Ok(Stmt::Block(vec![]));
            }
            // Optional storage between `}` and the instance name
            // (`struct S {...} const st = {...};`).
            if self.peek_is(Token::Const) || self.peek_is(Token::Static) {
                self.advance();
            } else if self.peek_is(Token::Identifier)
                && matches!(
                    self.current_lexeme().to_ascii_lowercase().as_str(),
                    "const" | "constexpr" | "static" | "inline"
                )
            {
                self.advance();
            }
            if self.peek_is(Token::Identifier) {
                // Only treat a trailing identifier as a variable instance
                // (`struct S { ... } var;`) when followed by `;`, `[`, `=`,
                // `,` or `:` — otherwise it is the start of the next
                // statement (e.g. `struct Basis {...}; auto const w = n;`
                // where `auto` must not be consumed here). A `::` scope
                // (`struct S {...} sce::Psr::T ...`) is never an instance.
                let after = self.tokens.get(self.index + 1).map(|t| t.token.clone());
                let is_scope = matches!(after, Some(Token::Colon))
                    && self
                        .tokens
                        .get(self.index + 2)
                        .map(|t| t.token == Token::Colon)
                        .unwrap_or(false);
                let is_instance = !is_scope
                    && matches!(
                        after,
                        Some(Token::Semi)
                            | Some(Token::LBracket)
                            | Some(Token::Equal)
                            | Some(Token::Comma)
                            | Some(Token::Colon)
                    );
                if is_instance {
                    let var_name = self.current_lexeme();
                    self.advance();
                    while self.peek_is(Token::LBracket) {
                        self.advance();
                        if !self.peek_is(Token::RBracket) {
                            let _ = self.parse_assignment();
                        }
                        self.expect(Token::RBracket)?;
                    }
                    if self.peek_is(Token::Equal) {
                        self.advance();
                        let _ = self.parse_assignment();
                    }
                    if self.peek_is(Token::Semi) {
                        self.advance();
                    }
                    // Struct definition with variable instance `struct S { ... } var;`
                    // uses the lifted type when this declaration defined one.
                    let ty = lifted
                        .clone()
                        .map(crate::ast::Type::Named)
                        .unwrap_or(crate::ast::Type::Named("struct".to_string()));
                    return Ok(Stmt::VarDecl {
                        storage: None,
                        ty,
                        name: var_name,
                        array_size: None,
                        init: None,
                    });
                }
            }
            if self.peek_is(Token::Semi) {
                self.advance();
            }
            return Ok(Stmt::Block(vec![]));
        }

        if self.peek_is(Token::LBrace) {
            return Ok(Stmt::Block(self.parse_block()?));
        }

        // FXAA swizzle re-declaration `float luma4B.w = ...` / `FxaaFloat luma4B.w = ...`
        // looks like a declaration but is really an assignment to a member.
        // Detect `Type Ident Dot` and discard the redundant leading type.
        if self.is_type_start() {
            if let Some(name_tok) = self.tokens.get(self.index + 1) {
                if name_tok.token == Token::Identifier {
                    if let Some(dot_tok) = self.tokens.get(self.index + 2) {
                        if dot_tok.token == Token::Dot {
                            // Discard the type token and parse as ordinary assignment
                            self.advance(); // discard type (float / FxaaFloat)
                            let expr = self.parse_assignment()?;
                            self.expect_semi()?;
                            return Ok(Stmt::Expr(expr));
                        }
                    }
                }
            }
        }

        // C++ `typedef` inside a function body (e.g.
        // `typedef Toolkit::BlockReduce<float, W, W> BlockReduce;`):
        // record the alias; it has no runtime effect.
        if self.peek_is(Token::Identifier)
            && self.current_lexeme().eq_ignore_ascii_case("typedef")
        {
            return self.parse_local_typedef();
        }

        // Variable declaration: [storage] type name [= expr];
        if self.starts_vardecl() {
            return self.parse_vardecl();
        }

        let mut expr = self.parse_assignment()?;
        // C comma expressions (`a = X, b = Y;`) chaining assignments.
        while self.peek_is(Token::Comma) {
            self.advance();
            let rhs = self.parse_assignment()?;
            expr = crate::ast::Expr::Binary {
                left: Box::new(expr),
                op: crate::ast::BinaryOp::Comma,
                right: Box::new(rhs),
            };
        }
        self.expect_semi()?;
        Ok(Stmt::Expr(expr))
    }

    fn parse_if(&mut self) -> Result<Stmt> {
        self.expect(Token::If)?;
        // `if constexpr(...)` — C++17 constexpr if
        if self.peek_is(Token::Identifier)
            && self.current_lexeme().eq_ignore_ascii_case("constexpr")
        {
            self.advance();
        } else if self.peek_is(Token::Const) {
            self.advance();
        }
        self.expect(Token::LParen)?;
        let condition = self.parse_assignment()?;
        self.expect(Token::RParen)?;
        let then_branch = if self.peek_is(Token::LBrace) {
            self.parse_block()?
        } else {
            vec![self.parse_stmt()?]
        };
        let else_branch = if self.peek_is(Token::Else) {
            self.advance();
            let block = if self.peek_is(Token::If) {
                // `else if` chains as a single nested if statement.
                vec![self.parse_if()?]
            } else if self.peek_is(Token::LBrace) {
                self.parse_block()?
            } else {
                vec![self.parse_stmt()?]
            };
            Some(block)
        } else {
            None
        };
        Ok(Stmt::If {
            condition: Box::new(condition),
            then_branch,
            else_branch,
        })
    }

    fn parse_for(&mut self) -> Result<Stmt> {
        self.expect(Token::For)?;
        self.expect(Token::LParen)?;

        let init = if self.peek_is(Token::Semi) {
            // Empty init (`for (;;)`, `for (; cond;)`): don't consume —
            // the `;` below terminates the (empty) init clause.
            None
        } else if self.starts_vardecl() {
            Some(self.parse_vardecl_internal(false)?)
        } else {
            // C allows comma-separated init expressions (`for (i = 0, j = 0; ...)`).
            let mut exprs = vec![self.parse_assignment()?];
            while self.peek_is(Token::Comma) {
                self.advance();
                exprs.push(self.parse_assignment()?);
            }
            let mut iter = exprs.into_iter();
            let first = iter.next().unwrap();
            let combined = iter.fold(first, |acc, e| {
                crate::ast::Expr::Binary {
                    left: Box::new(acc),
                    op: crate::ast::BinaryOp::Comma,
                    right: Box::new(e),
                }
            });
            Some(Stmt::Expr(combined))
        };

        self.expect_semi()?;

        let condition = if self.peek_is(Token::Semi) {
            None
        } else if self.peek_is(Token::RParen) {
            None
        } else {
            Some(self.parse_assignment()?)
        };

        self.expect_semi()?;

        let update = if self.peek_is(Token::RParen) {
            None
        } else {
            // C allows comma-separated updates (`for (...; ...; i++, pos.y++)`).
            let mut expr = self.parse_assignment()?;
            while self.peek_is(Token::Comma) {
                self.advance();
                let rhs = self.parse_assignment()?;
                expr = crate::ast::Expr::Binary {
                    left: Box::new(expr),
                    op: crate::ast::BinaryOp::Comma,
                    right: Box::new(rhs),
                };
            }
            Some(expr)
        };

        self.expect(Token::RParen)?;
        let body = if self.peek_is(Token::LBrace) {
            self.parse_block()?
        } else {
            vec![self.parse_stmt()?]
        };

        Ok(Stmt::For {
            init: Box::new(init.unwrap_or_else(|| Stmt::Expr(Expr::Bool(false)))),
            condition,
            update,
            body,
        })
    }

    fn parse_while(&mut self) -> Result<Stmt> {
        self.expect(Token::While)?;
        self.expect(Token::LParen)?;
        let condition = self.parse_assignment()?;
        self.expect(Token::RParen)?;
        let body = if self.peek_is(Token::LBrace) {
            self.parse_block()?
        } else {
            vec![self.parse_stmt()?]
        };
        Ok(Stmt::While {
            condition: Box::new(condition),
            body,
        })
    }

    fn parse_do(&mut self) -> Result<Stmt> {
        self.advance(); // 'do'
        let mut body = if self.peek_is(Token::LBrace) {
            self.parse_block()?
        } else {
            vec![self.parse_stmt()?]
        };
        self.expect(Token::While)?;
        self.expect(Token::LParen)?;
        let condition = self.parse_assignment()?;
        self.expect(Token::RParen)?;
        self.expect_semi()?;
        // `do B while (C);` ≡ `while (true) { B; if (!C) break; }`.
        // `continue` inside B still checks the condition (it flows into
        // the trailing `if`), matching do-while semantics.
        body.push(Stmt::If {
            condition: Box::new(Expr::Unary {
                op: crate::ast::UnaryOp::Not,
                expr: Box::new(condition),
            }),
            then_branch: vec![Stmt::Expr(Expr::Variable("break".to_string()))],
            else_branch: None,
        });
        Ok(Stmt::While {
            condition: Box::new(Expr::Bool(true)),
            body,
        })
    }

    /// Parses a variable declaration statement, assuming `starts_vardecl()`
    /// was true and that a trailing `;` should be consumed.
    fn parse_vardecl(&mut self) -> Result<Stmt> {
        self.parse_vardecl_internal(true)
    }

    /// Parses `row_major|column_major matrix-type(args)` construction in
    /// initializer position (`= row_major matrix(…)`), producing a
    /// column-major `Constructor` with identical values (transpose of the
    /// row-major fill order for `row_major`; identity for
    /// `column_major`). Square float/half/int/uint matrices only (2×2,
    /// 3×3, 4×4; bare `matrix` infers from 4/9/16 args); rectangular,
    /// mismatched counts, or non-matrix shapes return `None` (index
    /// rolled back) so normal parsing stays loud. This keeps the
    /// qualifier+construction together (previously split into a stray
    /// `row_major_` value plus a detached statement) while preserving
    /// exact matrix values under HLSL's column-major default (locals;
    /// host follows our emitted cbuffer layout regardless).
    fn try_parse_qualified_matrix_construct(
        &mut self,
        decl_ty: &crate::ast::Type,
    ) -> Result<Option<Expr>> {
        use crate::ast::{ScalarType, Type};
        // Qualifier.
        let qual = self.current_lexeme().to_ascii_lowercase();
        let is_row = if qual == "row_major" {
            true
        } else if qual == "column_major" {
            false
        } else {
            return Ok(None);
        };
        // Must be an identifier-like qualifier (not e.g. a variable named
        // `row_major` followed by unrelated tokens — the matrix-type and
        // paren checks below disambiguate; a bare `row_major` value with
        // no matrix construct falls through to normal parsing).
        if !self.peek_is(Token::Identifier) {
            return Ok(None);
        }
        let saved = self.index;
        self.advance(); // qualifier
        // Matrix type (square only; bare `matrix` infers later).
        let mty = match self.parse_type() {
            Ok(t) => t,
            Err(_) => {
                self.index = saved;
                return Ok(None);
            }
        };
        let (base, n) = match &mty {
            Type::Matrix { base, rows, columns } if rows == columns => (*base, *rows as usize),
            Type::Named(n) if n.eq_ignore_ascii_case("matrix") => {
                // Infer below from arg count.
                (ScalarType::Float, 0usize)
            }
            _ => {
                self.index = saved;
                return Ok(None);
            }
        };
        // Base must be a plain scalar (float/half/int/uint/bool).
        match base {
            ScalarType::Float
            | ScalarType::Half
            | ScalarType::Int
            | ScalarType::UInt
            | ScalarType::Bool => {}
        }
        if !self.peek_is(Token::LParen) {
            self.index = saved;
            return Ok(None);
        }
        self.advance(); // '('
        let mut args = Vec::new();
        if !self.peek_is(Token::RParen) {
            loop {
                let a = match self.parse_assignment() {
                    Ok(x) => x,
                    Err(_) => {
                        self.index = saved;
                        return Ok(None);
                    }
                };
                args.push(a);
                if self.peek_is(Token::Comma) {
                    self.advance();
                } else {
                    break;
                }
                if self.peek_is(Token::RParen) {
                    break;
                }
                if self.eof() {
                    self.index = saved;
                    return Ok(None);
                }
            }
        }
        if !self.peek_is(Token::RParen) {
            self.index = saved;
            return Ok(None);
        }
        self.advance(); // ')'
        // Dimension + count validation.
        let n = if n == 0 {
            match args.len() {
                4 => 2usize,
                9 => 3usize,
                16 => 4usize,
                _ => {
                    self.index = saved;
                    return Ok(None);
                }
            }
        } else {
            if args.len() != n * n {
                self.index = saved;
                return Ok(None);
            }
            n
        };
        // Decl-type agreement (when the decl names concrete dims).
        if let Type::Matrix { rows, columns, .. } = decl_ty {
            if *rows as usize != n || *columns as usize != n {
                self.index = saved;
                return Ok(None);
            }
        }
        let ty = Type::Matrix {
            base,
            rows: n as u8,
            columns: n as u8,
        };
        // Row-major fill → column-major order (transpose); column-major
        // keeps order. Values identical; HLSL default layout applies.
        let arguments = if is_row {
            let mut t = Vec::with_capacity(args.len());
            for c in 0..n {
                for r in 0..n {
                    t.push(args[r * n + c].clone());
                }
            }
            t
        } else {
            args
        };
        Ok(Some(Expr::Constructor { ty, arguments }))
    }

    /// Parses a variable declaration body. When `consume_semi` is true a
    /// trailing `;` is consumed (statement context); when false the caller
    /// (e.g. the `for` initializer) is responsible for the terminator.
    pub(crate) fn parse_vardecl_internal(&mut self, consume_semi: bool) -> Result<Stmt> {
        let mut storage = None;
        // Handle `static const` / `const static` / `constexpr` / `inline` / `thread_group_memory` etc. in any order.
        for _ in 0..5 {
            if self.peek_is(Token::Const) {
                storage = Some(StorageClass::Const);
                self.advance();
            } else if self.peek_is(Token::Static) {
                if storage.is_none() {
                    storage = Some(StorageClass::Static);
                }
                self.advance();
            } else if self.peek_is(Token::Identifier) {
                let lex = self.current_lexeme();
                if lex.eq_ignore_ascii_case("constexpr")
                    || lex.eq_ignore_ascii_case("const")
                    || lex.eq_ignore_ascii_case("inline")
                    || lex.eq_ignore_ascii_case("thread_group_memory")
                    || lex.eq_ignore_ascii_case("groupshared")
                    || lex.eq_ignore_ascii_case("groupShared")
                {
                    storage = Some(StorageClass::Static);
                    self.advance();
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        self.skip_interp_modifiers();

        // C++ `typename` prefix on dependent types
        // (`typename conditional<...>::type traverser;`).
        if self.peek_is(Token::Identifier)
            && self.current_lexeme().eq_ignore_ascii_case("typename")
        {
            self.advance();
        }

        // Template-id local types (`Indices<3> x;`): record the spelling
        // window for value-template instantiation (hooked per
        // declarator below; the window covers type and name but not
        // dims/initializers).
        let local_ty_start = self.index;
        // SRT buffer macros `REGULAR_BUFFER(T) name;` in local position.
        let ty = match self.try_parse_macro_resource_type()? {
            Some(t) => t,
            None => self.parse_type()?,
        };
        // Function-local struct uses (`output const r = ...`): rewrite to
        // the mangled module name recorded when the local declaration was
        // lifted.
        let ty = match ty {
            crate::ast::Type::Named(name) => self
                .resolve_local_struct(&name)
                .map(crate::ast::Type::Named)
                .unwrap_or(crate::ast::Type::Named(name)),
            other => other,
        };
        // Trailing `::type` after a template (`enable_if<...>::type`,
        // `conditional<...>::type`) — part of the same declaration.
        while self.peek_is(Token::Colon)
            && self
                .tokens
                .get(self.index + 1)
                .map(|t| t.token == Token::Colon)
                .unwrap_or(false)
        {
            self.advance(); // ':'
            self.advance(); // ':'
            if self.peek_is(Token::Identifier) {
                self.advance();
            } else {
                break;
            }
        }
        // Handle trailing `const`/`constexpr` after type: `AnyHitSrt const`,
        // `auto const`, or `static int constexpr ntheta`.
        if self.peek_is(Token::Const) {
            self.advance();
        } else if self.peek_is(Token::Identifier)
            && (matches!(
                self.current_lexeme().to_ascii_lowercase().as_str(),
                "const" | "constexpr"
            ) || self.current_lexeme() == "restrict")
        {
            self.advance();
        }
        // Handle `&` / `*` reference/pointer declarators (e.g. `float4 &color`)
        if self.peek_is(Token::Ampersand) || self.peek_is(Token::Star) {
            self.advance();
            // `auto const &` — optional trailing const after `&`
            if self.peek_is(Token::Const) {
                self.advance();
            } else if self.peek_is(Token::Identifier)
                && self.current_lexeme().eq_ignore_ascii_case("const")
            {
                self.advance();
            }
        }
        // C++17 structured binding `auto const [a, b] = expr;` or `auto [a,b] = expr;`
        if self.peek_is(Token::LBracket) {
            self.advance(); // '['
            let mut bound_names: Vec<String> = Vec::new();
            while !self.peek_is(Token::RBracket) && !self.eof() {
                if self.peek_is(Token::Identifier) {
                    bound_names.push(self.current_lexeme());
                    self.advance();
                } else if self.peek_is(Token::Comma) {
                    self.advance();
                } else {
                    self.advance();
                }
            }
            self.expect(Token::RBracket)?;
            // `= expr`
            let init = if self.peek_is(Token::Equal) {
                self.advance();
                Some(self.parse_assignment()?)
            } else {
                None
            };
            if consume_semi {
                self.expect_semi()?;
            }
            // Emit as a block of VarDecls for each bound name (all share type `auto`).
            if bound_names.is_empty() {
                return Ok(Stmt::Block(vec![]));
            }
            if bound_names.len() == 1 {
                return Ok(Stmt::VarDecl {
                    storage: storage.clone(),
                    ty: ty.clone(),
                    name: bound_names.into_iter().next().unwrap(),
                    array_size: None,
                    init,
                });
            }
            // Multi-name bindings keep their init for lowering, which
            // splats once the init type is known (C++17 structured
            // bindings; storage/ty travel along for faithful fallback).
            return Ok(Stmt::Destructure {
                storage: storage.clone(),
                ty,
                names: bound_names,
                init,
            });
        }
        let mut declarators: Vec<(String, Option<u32>, Option<Expr>)> = Vec::new();
        loop {
            // Also handle `&`/`*` per-declarator (e.g. `float x, &y`)
            if self.peek_is(Token::Ampersand) || self.peek_is(Token::Star) {
                self.advance();
            }
            let name = self.expect_identifier()?;
            // Template-id spelling for this declarator's type (shares
            // the local window above).
            self.note_site_spelling(local_ty_start, self.index, &name);
            // Array size: `name[4]` or `name[64][2]` (possibly a named constant like
            // `name[k_sample_count]`)
            let mut array_size = None;
            let mut is_array = false;
            // Support multi-dimensional arrays by looping over `[...][...]...`
            let mut first_dim = true;
            let mut all_dims: Vec<Option<u32>> = Vec::new();
            while self.peek_is(Token::LBracket) {
                is_array = true;
                self.advance();
                if !self.peek_is(Token::RBracket) {
                    let size_expr = self.parse_assignment()?;
                    let resolved = self.resolve_array_size(&size_expr);
                    if first_dim {
                        if let Some(n) = resolved {
                            array_size = Some(n);
                        }
                    }
                    all_dims.push(resolved);
                } else {
                    // `T x[]` – unsized array, sentinel 0 distinguishes from non-array (None)
                    if first_dim {
                        array_size = Some(0);
                    }
                    all_dims.push(None);
                }
                self.expect(Token::RBracket)?;
                first_dim = false;
            }
            // Keep sentinel for unsized so `uint2 offsets[]` is not confused with `uint2 x`
            if is_array && array_size.is_none() {
                // Sized with non-constant expression or unknown size – treat as unsized for now
                array_size = Some(0);
            }
            // Exactly-2D all-constant shapes flatten to product size (uses
            // linearize inline) — except nested braced inits (`= {{...}}`),
            // which belong to the vector rule in lowering.
            let nested_brace_init = self.peek_is(Token::Equal)
                && matches!(
                    self.tokens.get(self.index + 1).map(|t| &t.token),
                    Some(Token::LBrace)
                )
                && matches!(
                    self.tokens.get(self.index + 2).map(|t| &t.token),
                    Some(Token::LBrace)
                );
            // (Bare `= {` without nesting keeps product sizing: flat
            // initializers fill row-major, matching PSSL.)
            if let Some(product) =
                self.note_multidim(&name, &all_dims, nested_brace_init)
            {
                array_size = Some(product);
            }
            let init = if self.peek_is(Token::Equal) {
                self.advance();
                if self.peek_is(Token::LBrace) {
                    Some(self.parse_initializer_list()?)
                } else if let Some(q) = self.try_parse_qualified_matrix_construct(&ty)? {
                    Some(q)
                } else {
                    Some(self.parse_assignment()?)
                }
            } else if self.peek_is(Token::LBrace) {
                Some(self.parse_initializer_list()?)
            } else if self.peek_is(Token::LParen) {
                // C++ paren initialization `float2 tex(a, b)`: parse as a
                // constructor init (exact values) for builtin types.
                // Unparseable contents (a local function declaration such
                // as `float2 f(float x);`) and void/named types keep the
                // legacy skip: named (possibly struct) types may carry
                // explicit constructors whose bodies need the method
                // system (parsed separately), so they never become
                // constructor expressions here.
                let parsed = if matches!(ty, crate::ast::Type::Void | crate::ast::Type::Named(_)) {
                    None
                } else {
                    self.try_parse_paren_init(&ty, false)
                };
                match parsed {
                    Some(init) => Some(init),
                    None => {
                        self.skip_paren_block()?;
                        Some(Expr::Number("0".to_string()))
                    }
                }
            } else {
                None
            };
            // Track integer constants for resolving later array sizes
            // (e.g. `const int k_sample_count = 4;` then `samples[k_sample_count]`).
            if matches!(storage, Some(StorageClass::Const) | Some(StorageClass::Static))
                && matches!(ty, crate::ast::Type::Int | crate::ast::Type::UInt)
            {
                if let Some(Expr::Number(v)) = &init {
                    let num_str = v.trim_end_matches(|c| c == 'f' || c == 'F');
                    if num_str.parse::<u32>().is_ok() {
                        self.int_consts.insert(name.clone(), num_str.to_string());
                    }
                }
            }
            declarators.push((name, array_size, init));
            if !self.peek_is(Token::Comma) {
                break;
            }
            self.advance();
        }

        if consume_semi {
            self.expect_semi()?;
        }

        if declarators.len() == 1 {
            let (name, array_size, init) = declarators.into_iter().next().unwrap();
            Ok(Stmt::VarDecl {
                storage,
                ty,
                name,
                array_size,
                init,
            })
        } else {
            let stmts = declarators
                .into_iter()
                .map(|(name, array_size, init)| Stmt::VarDecl {
                    storage: storage.clone(),
                    ty: ty.clone(),
                    name,
                    array_size,
                    init,
                })
                .collect();
            Ok(Stmt::Block(stmts))
        }
    }

    /// Parses a function-body `typedef` (e.g.
    /// `typedef Toolkit::BlockReduce<float, W, W> BlockReduce;`): records
    /// the alias like the module-scope form; has no runtime effect.
    fn parse_local_typedef(&mut self) -> Result<Stmt> {
        self.advance(); // 'typedef'
        // Optional storage modifiers (`row_major`, `const`, ...) and the
        // C++ `typename` prefix on dependent types
        // (`typedef typename ShuffleType<renderMode>::type type;`).
        loop {
            if self.peek_is(Token::Const) || self.peek_is(Token::Static) {
                self.advance();
                continue;
            }
            if self.peek_is(Token::Identifier)
                && matches!(
                    self.current_lexeme().to_ascii_lowercase().as_str(),
                    "row_major"
                        | "column_major"
                        | "const"
                        | "constexpr"
                        | "static"
                        | "inline"
                        | "typename"
                )
            {
                self.advance();
                continue;
            }
            break;
        }
        // Function-pointer declarators (`typedef void (*Cb)(int);`) carry
        // no nameable alias — skip to `;` unrecorded (uses of the name
        // still parse via the `Named` fallback).
        if self.peek_is(Token::LParen) || self.peek_is(Token::Star) {
            while !self.peek_is(Token::Semi) && !self.eof() {
                if self.peek_is(Token::LParen) {
                    self.skip_paren_block()?;
                } else {
                    self.advance();
                }
            }
            self.expect_semi()?;
            return Ok(Stmt::Block(vec![]));
        }
        let base_ty = self.parse_type()?;
        // Trailing `::type` after a template (`ShuffleType<...>::type`).
        while self.peek_is(Token::Colon)
            && self
                .tokens
                .get(self.index + 1)
                .map(|t| t.token == Token::Colon)
                .unwrap_or(false)
        {
            self.advance(); // ':'
            self.advance(); // ':'
            if self.peek_is(Token::Identifier) {
                self.advance();
            } else {
                break;
            }
        }
        // Calling-convention macros between type and alias.
        self.skip_interposed_macro();
        loop {
            while self.peek_is(Token::Star) || self.peek_is(Token::Ampersand) {
                self.advance();
            }
            // The alias may re-spell a keyword scalar.
            let alias = self.expect_alias_name()?;
            while self.peek_is(Token::LBracket) {
                self.advance();
                while !self.peek_is(Token::RBracket) && !self.eof() {
                    self.advance();
                }
                self.expect(Token::RBracket)?;
            }
            self.typedefs.insert(alias, base_ty.clone());
            if !self.peek_is(Token::Comma) {
                break;
            }
            self.advance();
        }
        self.expect_semi()?;
        Ok(Stmt::Block(vec![]))
    }

    pub(crate) fn parse_initializer_list(&mut self) -> Result<Expr> {
        self.expect(Token::LBrace)?;
        let mut elements = Vec::new();
        while !self.peek_is(Token::RBrace) {
            if self.eof() {
                return Err(self.eof_error("'}'"));
            }
            // C99/PSSL designated initializer `.m_org = org`.
            // Guarded by `Ident =` so leading-dot floats (`.5`)
            // still parse as numbers.
            if self.peek_is(Token::Dot)
                && matches!(
                    self.tokens.get(self.index + 1).map(|t| &t.token),
                    Some(Token::Identifier)
                )
                && matches!(
                    self.tokens.get(self.index + 2).map(|t| &t.token),
                    Some(Token::Equal)
                )
            {
                self.advance(); // '.'
                let _field = self.expect_identifier()?;
                self.expect(Token::Equal)?;
            }
            let elem = if self.peek_is(Token::LBrace) {
                self.parse_initializer_list()?
            } else {
                self.parse_assignment()?
            };
            elements.push(elem);
            if self.peek_is(Token::Comma) {
                self.advance();
                // allow trailing comma
                if self.peek_is(Token::RBrace) {
                    break;
                }
            } else {
                break;
            }
        }
        self.expect(Token::RBrace)?;
        Ok(Expr::InitializerList(elements))
    }

    /// Non-consuming lookahead: does the next input begin a variable
    /// declaration? We treat it as a declaration when:
    /// - it starts with a storage keyword (`const` / `static`), or
    /// - a type-start token is immediately followed by an identifier
    ///   (i.e. `type name ...`), since an expression such as `float4(...)`
    ///   or `x * y` has a non-identifier token after its first symbol.
    pub(crate) fn starts_vardecl(&self) -> bool {
        if self.peek_is(Token::Const) || self.peek_is(Token::Static) {
            return true;
        }
        if self.peek_is(Token::Identifier) {
            let lex = self.current_lexeme();
            if lex.eq_ignore_ascii_case("constexpr")
                || lex.eq_ignore_ascii_case("inline")
                || lex.eq_ignore_ascii_case("thread_group_memory")
                || lex.eq_ignore_ascii_case("groupshared")
                || lex.eq_ignore_ascii_case("groupShared")
            {
                return true;
            }
            // HLSL interpolation prefix (`sample float4 x;`, `linear centroid ...`):
            // skip past modifiers, then require `Type Ident`.
            if Self::is_decl_modifier_word(&lex) {
                let mut look = self.index + 1;
                while look < self.tokens.len() {
                    if let Some(tok) = self.tokens.get(look) {
                        if tok.token == Token::Identifier
                            && Self::is_decl_modifier_word(&tok.lexeme)
                        {
                            look += 1;
                            continue;
                        }
                    }
                    break;
                }
                // After modifiers: `Type Ident` (type may be keyword or identifier).
                let ty_ok = matches!(
                    self.tokens.get(look).map(|t| &t.token),
                    Some(
                        Token::Identifier
                            | Token::Void
                            | Token::Bool
                            | Token::Int
                            | Token::UInt
                            | Token::Float
                            | Token::Half
                            | Token::Double
                    )
                );
                let name_ok = matches!(
                    self.tokens.get(look + 1).map(|t| &t.token),
                    Some(Token::Identifier)
                );
                if ty_ok && name_ok {
                    return true;
                }
            }
            // `Type<Args...> name` declarations (e.g.
            // `RW_RegularBuffer<T, cacheFlags> rb { buf };`): a template list
            // after a *known type name*, followed by an identifier that is
            // not a call, is a declaration — not a `<` comparison expression.
            if self.peek_next_is(Token::Less) && self.is_known_type_name(&lex) {
                let mut k = self.index + 1;
                let mut depth = 0usize;
                let mut closed = None;
                while k < self.tokens.len() {
                    if self.tokens[k].token == Token::Less {
                        depth += 1;
                    } else if self.tokens[k].token == Token::Greater {
                        depth = depth.saturating_sub(1);
                        if depth == 0 {
                            closed = Some(k);
                            break;
                        }
                    } else if self.tokens[k].token == Token::ShiftRight {
                        if depth <= 2 {
                            closed = Some(k);
                            break;
                        }
                        depth -= 2;
                    } else if self.tokens[k].token == Token::LParen {
                        // Balanced parens inside args (e.g. `(kUvCount > 0)`).
                        let mut pd = 0usize;
                        while k < self.tokens.len() {
                            if self.tokens[k].token == Token::LParen {
                                pd += 1;
                            } else if self.tokens[k].token == Token::RParen {
                                pd = pd.saturating_sub(1);
                                if pd == 0 {
                                    break;
                                }
                            }
                            k += 1;
                        }
                    } else if matches!(
                        self.tokens[k].token,
                        Token::Semi | Token::LBrace | Token::RBrace
                    ) {
                        break; // no closer before terminator: not a template list
                    }
                    k += 1;
                    if k - (self.index + 1) > 200 {
                        break;
                    }
                }
                if let Some(c) = closed {
                    // Skip trailing qualifiers/pointers between the list
                    // and the name (`RegularBuffer<uint3> const *p`,
                    // `Type<...> &r`).
                    let mut j = c + 1;
                    for _ in 0..4 {
                        let Some(tok) = self.tokens.get(j) else {
                            break;
                        };
                        if tok.token == Token::Const
                            || tok.token == Token::Star
                            || tok.token == Token::Ampersand
                        {
                            j += 1;
                            continue;
                        }
                        if tok.token == Token::Identifier
                            && matches!(
                                tok.lexeme.to_ascii_lowercase().as_str(),
                                "const" | "constexpr"
                            )
                        {
                            j += 1;
                            continue;
                        }
                        break;
                    }
                    let name_ok = matches!(
                        self.tokens.get(j).map(|t| &t.token),
                        Some(Token::Identifier)
                    );
                    // `Type<...>(...)` is a construction/call (expression),
                    // `Type<...> name...` is a declaration.
                    let is_call = matches!(
                        self.tokens.get(j).map(|t| &t.token),
                        Some(Token::LParen)
                    );
                    if name_ok && !is_call {
                        return true;
                    }
                }
            }
            // SRT buffer macro declaration `REGULAR_BUFFER(T) name;` — the
            // `MACRO(...)` is followed by the declared name, not `;`, so it
            // is a declaration, not an expression-statement call.
            if self.peek_next_is(Token::LParen)
                && super::types::is_resource_kind(&lex)
            {
                let mut j = self.index + 1;
                let mut depth = 0usize;
                let mut rparen = None;
                while j < self.tokens.len() {
                    if self.tokens[j].token == Token::LParen {
                        depth += 1;
                    } else if self.tokens[j].token == Token::RParen {
                        depth = depth.saturating_sub(1);
                        if depth == 0 {
                            rparen = Some(j);
                            break;
                        }
                    }
                    j += 1;
                }
                if matches!(
                    rparen.and_then(|k| self.tokens.get(k + 1)),
                    Some(t) if t.token == Token::Identifier
                ) {
                    return true;
                }
            }
        }
        if !self.is_type_start() {
            return false;
        }
        // `a[0].x =` / `a[0][0] =` / `M[0][0] =` where `a`/`M` is variable with index/member, not a type.
        // Detect `Ident [ ... ] [ ... ] .` / `Ident [ ... ] =` immediately after the supposed type.
        // (`auto [a, b] = ...` structured bindings are declarations, not
        // indexed assignments — `auto` never names a variable.)
        //
        // The head may be `::`-qualified (`Lds::position[i] = v` stores
        // through a flattened namespace array — an expression, not a
        // `Lds::position` declaration): skip the qualifier chain first so
        // scoped names get the same disambiguation as bare ones.
        let mut head_end = self.index + 1;
        while head_end + 1 < self.tokens.len()
            && self.tokens[head_end].token == Token::Colon
            && self.tokens[head_end + 1].token == Token::Colon
            && matches!(
                self.tokens.get(head_end + 2).map(|t| &t.token),
                Some(Token::Identifier)
            )
        {
            head_end += 3;
        }
        if let Some(t1) = self.tokens.get(head_end) {
            if t1.token == Token::LBracket
                && !self.current_lexeme().eq_ignore_ascii_case("auto")
            {
                let mut j = head_end;
                // Consume all consecutive `[...]` blocks (handles `a[0][0]`)
                loop {
                    let mut depth = 0usize;
                    let mut k = j;
                    let mut found_close = false;
                    while k < self.tokens.len() {
                        if self.tokens[k].token == Token::LBracket {
                            depth += 1;
                        } else if self.tokens[k].token == Token::RBracket {
                            depth = depth.saturating_sub(1);
                            if depth == 0 {
                                j = k + 1; // position after `]`
                                found_close = true;
                                break;
                            }
                        }
                        k += 1;
                        if k - j > 30 {
                            break;
                        }
                    }
                    if !found_close {
                        break;
                    }
                    // If next is another `[` then continue for multi-dim `M[0][0]`
                    if j < self.tokens.len() && self.tokens[j].token == Token::LBracket {
                        continue;
                    }
                    break;
                }
                if j < self.tokens.len() {
                    if let Some(after) = self.tokens.get(j) {
                        if matches!(
                            after.token,
                            Token::Dot
                                | Token::Equal
                                | Token::PlusEqual
                                | Token::MinusEqual
                                | Token::StarEqual
                                | Token::SlashEqual
                                | Token::PercentEqual
                                | Token::CaretEqual
                                | Token::PipeEqual
                                | Token::AmpersandEqual
                                | Token::PlusPlus
                                | Token::MinusMinus
                        ) {
                            return false;
                        }
                    }
                }
            }
        }
        // `unsigned int i` / `long l` — C integer types with `unsigned`/`signed`/`long` prefix
        {
            let lower = self.current_lexeme().to_ascii_lowercase();
            if lower == "unsigned" || lower == "signed" || lower == "long" {
                if let Some(next) = self.tokens.get(self.index + 1) {
                    let is_next_type = matches!(
                        next.token,
                        Token::Int | Token::UInt | Token::Float | Token::Half | Token::Double | Token::Identifier
                    ) || next.lexeme.eq_ignore_ascii_case("long");
                    if is_next_type {
                        if let Some(name_tok) = self.tokens.get(self.index + 2) {
                            if name_tok.token == Token::Identifier {
                                return true;
                            }
                            // `unsigned long long i` — two `long`s
                            if name_tok.lexeme.eq_ignore_ascii_case("long") {
                                if let Some(name2) = self.tokens.get(self.index + 3) {
                                    if name2.token == Token::Identifier {
                                        return true;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        // `auto const [a, b]` / `auto [a,b]` structured binding — type `auto`
        // followed by optional const/&/* and then `[`.
        if self.current_lexeme().eq_ignore_ascii_case("auto") {
            let mut look = self.index + 1;
            // skip optional `const` / `constexpr` / `&` / `*`
            for _ in 0..3 {
                if let Some(tok) = self.tokens.get(look) {
                    if tok.token == Token::Const {
                        look += 1;
                        continue;
                    }
                    if tok.token == Token::Identifier
                        && tok.lexeme.eq_ignore_ascii_case("const")
                    {
                        look += 1;
                        continue;
                    }
                    if tok.token == Token::Ampersand || tok.token == Token::Star {
                        look += 1;
                        // optional trailing const after &
                        if let Some(nt) = self.tokens.get(look) {
                            if nt.token == Token::Const
                                || (nt.token == Token::Identifier
                                    && nt.lexeme.eq_ignore_ascii_case("const"))
                            {
                                look += 1;
                            }
                        }
                        continue;
                    }
                }
                break;
            }
            if let Some(tok) = self.tokens.get(look) {
                if tok.token == Token::LBracket {
                    return true;
                }
            }
        }
        // Handle `Type const name` and `Type::Scope const name` — need to find end of type
        // including `::` chains: `sce::Psr::Ray const ray` or `sce::Psr::Ray psrRay`
        {
            let mut ty_end = self.index + 1;
            // Consume `::Ident` chains after the initial type
            while ty_end + 1 < self.tokens.len()
                && self.tokens[ty_end].token == Token::Colon
                && self.tokens[ty_end + 1].token == Token::Colon
            {
                if let Some(id_tok) = self.tokens.get(ty_end + 2) {
                    if id_tok.token == Token::Identifier {
                        ty_end += 3;
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            }
            // Check for `const` after the type end and then an identifier
            if ty_end < self.tokens.len() {
                let next = &self.tokens[ty_end];
                let is_const = next.token == Token::Const
                    || (next.token == Token::Identifier
                        && next.lexeme.eq_ignore_ascii_case("const"));
                if is_const {
                    if let Some(name_tok) = self.tokens.get(ty_end + 1) {
                        if name_tok.token == Token::Identifier {
                            return true;
                        }
                    }
                    // `Type const *name` or `Type const &name`, with an
                    // optional second `const` (`T const * const p`).
                    if let Some(pt) = self.tokens.get(ty_end + 1) {
                        if pt.token == Token::Star || pt.token == Token::Ampersand {
                            let mut k = ty_end + 2;
                            while k < self.tokens.len() {
                                let Some(tk) = self.tokens.get(k) else {
                                    break;
                                };
                                if tk.token == Token::Const
                                    || (tk.token == Token::Identifier
                                        && (tk.lexeme.eq_ignore_ascii_case("const")
                                            || tk.lexeme.eq_ignore_ascii_case("constexpr")))
                                {
                                    k += 1;
                                    continue;
                                }
                                break;
                            }
                            if let Some(name_tok2) = self.tokens.get(k) {
                                if name_tok2.token == Token::Identifier {
                                    return true;
                                }
                            }
                        }
                    }
                } else if next.token == Token::Identifier {
                    // `Type name` with scoped type: `sce::Psr::Ray psrRay`
                    // Ensure next after name is not '(' (which would be a call, not vardecl)
                    if let Some(after) = self.tokens.get(ty_end + 1) {
                        if after.token != Token::LParen {
                            return true;
                        }
                    } else {
                        return true;
                    }
                } else if next.token == Token::Star || next.token == Token::Ampersand {
                    // `Type *name` / `Type &name`
                    if let Some(name_tok) = self.tokens.get(ty_end + 1) {
                        if name_tok.token == Token::Identifier {
                            return true;
                        }
                    }
                } else if next.token == Token::LBracket {
                    // `Type [a,b]` structured binding handled above, but also `Type name[sz]`
                    return true;
                }
            }
        }
        // `constexpr Type name` — the type is at +1 when leading qualifier present.
        let offset = if self.peek_is(Token::Identifier)
            && {
                let lex = self.current_lexeme();
                lex.eq_ignore_ascii_case("constexpr") || lex.eq_ignore_ascii_case("inline")
            } {
            1
        } else {
            0
        };
        // Need to check if at offset the type start is followed by identifier.
        // For `constexpr half2 delta_step` -> offset 1 -> `half2` Identifier + `delta_step` Identifier
        if offset == 1 {
            // Check token at index+1 is type start and index+2 is Identifier
            let ty_tok = self.tokens.get(self.index + 1).map(|t| &t.token);
            let name_tok = self.tokens.get(self.index + 2).map(|t| &t.token);
            return matches!(ty_tok, Some(Token::Identifier) | Some(Token::Void) | Some(Token::Bool) | Some(Token::Int) | Some(Token::UInt) | Some(Token::Float) | Some(Token::Half) | Some(Token::Double))
                && matches!(name_tok, Some(Token::Identifier));
        }
        // `Type const name` like `uint3 const tri` — type, const, identifier
        if let Some(next) = self.tokens.get(self.index + 1) {
            let is_const = next.token == Token::Const
                || (next.token == Token::Identifier && next.lexeme.eq_ignore_ascii_case("const"));
            if is_const {
                if let Some(name_tok) = self.tokens.get(self.index + 2) {
                    if name_tok.token == Token::Identifier {
                        // Reject `float x.w` swizzle assignments (FXAA) — they look like
                        // declarations but are assignments to a member.
                        if let Some(dot) = self.tokens.get(self.index + 3) {
                            if dot.token == Token::Dot {
                                return false;
                            }
                        }
                        return true;
                    }
                }
            }
        }
        // Final fallback `Type name` — reject swizzle assignments `Type ident .`
        if let Some(dot) = self.tokens.get(self.index + 2) {
            if dot.token == Token::Dot {
                return false;
            }
        }
        matches!(
            self.tokens.get(self.index + 1).map(|t| &t.token),
            Some(Token::Identifier)
        )
    }

    pub(crate) fn is_type_start(&self) -> bool {
        match self.peek() {
            Some(Token::Void)
            | Some(Token::Bool)
            | Some(Token::Int)
            | Some(Token::UInt)
            | Some(Token::Float)
            | Some(Token::Half)
            | Some(Token::Double)
            | Some(Token::Identifier) => true,
            _ => false,
        }
    }

    /// Whether `name` is a known type: a resource kind, a builtin
    /// scalar/vector/matrix spelling, a seen struct name, a typedef
    /// alias, a `template<typename T>` type param, or HLSL's `string`
    /// type. Used to disambiguate `Type<...> name` declarations from `<`
    /// comparison expressions (and lone-identifier parameter shapes).
    pub(crate) fn is_known_type_name(&self, name: &str) -> bool {
        if super::types::is_resource_kind(name) {
            return true;
        }
        if super::types::builtin_type(name).is_some() {
            return true;
        }
        if name.eq_ignore_ascii_case("string") {
            return true;
        }
        if self.struct_names.contains(name) {
            return true;
        }
        if self.typedefs.contains_key(name) {
            return true;
        }
        if self.template_type_params.contains(name) {
            return true;
        }
        false
    }
}
