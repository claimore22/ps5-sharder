//! Expression parsing: precedence climbing.

use super::Parser;
use crate::ast::{BinaryOp, Expr, Type, UnaryOp};
use crate::error::Result;
use crate::token::Token;

use super::types::{builtin_type, unquote};

/// Strips HLSL-illegal short-integer suffixes from a numeric literal
/// spelling (`128us`/`128US`/`7s` → `128`/`7`), preserving value.
/// Unsigned-short range always fits `int`, so plain form is exact.
/// `s`/`S` are never hex digits, so stripping is safe for `0x…` forms.
/// `u`/`U`/`l`/`L`/`f`/`F`/`h`/`H` suffixes (HLSL-legal) and hex digits
/// pass through untouched. Non-suffixed spellings return unchanged.
fn normalize_int_suffix(spelling: &str) -> String {
    // Two-char `us`/`US` (case combinations) first, then single `s`/`S`.
    // (Hex checked implicitly: `s` never a hex digit, so no ambiguity.)
    if spelling.len() > 2 {
        let (head, tail) = spelling.split_at(spelling.len() - 2);
        if tail.eq_ignore_ascii_case("us")
            && head.chars().last().is_some_and(|c| {
                c.is_ascii_hexdigit() || c == 'x' || c == 'X'
            })
        {
            // Ensure it's really a suffix (preceded by digit/x), not part
            // of a longer identifier-like tail (numbers can't have those;
            // defensive: require head non-empty and numeric-looking).
            // (Lexemes reaching here are `Token::Number`, so numeric.)
            return head.to_string();
        }
    }
    if spelling.len() > 1 {
        let (head, tail) = spelling.split_at(spelling.len() - 1);
        if (tail == "s" || tail == "S")
            && head.chars().last().is_some_and(|c| {
                c.is_ascii_alphanumeric()
            })
        {
            return head.to_string();
        }
    }
    spelling.to_string()
}

impl Parser {
    // ------------------------------------------------------------------
    // Expressions (precedence climbing)
    // ------------------------------------------------------------------

    pub(crate) fn parse_assignment(&mut self) -> Result<Expr> {
        let left = self.parse_conditional()?;
        if self.peek_is(Token::Equal) {
            self.advance();
            let right = self.parse_assignment()?;
            return Ok(Expr::Assign {
                lhs: Box::new(left),
                rhs: Box::new(right),
            });
        }
        let compound = if self.peek_is(Token::PlusEqual) {
            Some(BinaryOp::Add)
        } else if self.peek_is(Token::MinusEqual) {
            Some(BinaryOp::Sub)
        } else if self.peek_is(Token::StarEqual) {
            Some(BinaryOp::Mul)
        } else if self.peek_is(Token::SlashEqual) {
            Some(BinaryOp::Div)
        } else if self.peek_is(Token::PercentEqual) {
            Some(BinaryOp::Mod)
        } else if self.peek_is(Token::ShiftLeftEqual) {
            Some(BinaryOp::ShiftLeft)
        } else if self.peek_is(Token::ShiftRightEqual) {
            Some(BinaryOp::ShiftRight)
        } else if self.peek_is(Token::CaretEqual) {
            Some(BinaryOp::BitXor)
        } else if self.peek_is(Token::PipeEqual) {
            Some(BinaryOp::BitOr)
        } else if self.peek_is(Token::AmpersandEqual) {
            Some(BinaryOp::BitAnd)
        } else {
            None
        };
        if let Some(op) = compound {
            self.advance();
            let right = self.parse_assignment()?;
            let rhs = Expr::Binary {
                left: Box::new(left.clone()),
                op,
                right: Box::new(right),
            };
            return Ok(Expr::Assign {
                lhs: Box::new(left),
                rhs: Box::new(rhs),
            });
        }
        Ok(left)
    }

    fn parse_conditional(&mut self) -> Result<Expr> {
        let condition = self.parse_logical_or()?;
        if self.peek_is(Token::Question) {
            self.advance();
            let then_branch = self.parse_assignment()?;
            self.expect(Token::Colon)?;
            let else_branch = self.parse_assignment()?;
            return Ok(Expr::Conditional {
                condition: Box::new(condition),
                then_branch: Box::new(then_branch),
                else_branch: Box::new(else_branch),
            });
        }
        Ok(condition)
    }

    fn parse_logical_or(&mut self) -> Result<Expr> {
        let mut expr = self.parse_logical_and()?;
        while self.peek_is(Token::OrOr) {
            self.advance();
            let rhs = self.parse_logical_and()?;
            expr = Expr::Binary {
                left: Box::new(expr),
                op: BinaryOp::LogicalOr,
                right: Box::new(rhs),
            };
        }
        Ok(expr)
    }

    fn parse_logical_and(&mut self) -> Result<Expr> {
        let mut expr = self.parse_bitwise_or()?;
        while self.peek_is(Token::AndAnd) {
            self.advance();
            let rhs = self.parse_bitwise_or()?;
            expr = Expr::Binary {
                left: Box::new(expr),
                op: BinaryOp::LogicalAnd,
                right: Box::new(rhs),
            };
        }
        Ok(expr)
    }

    fn parse_bitwise_or(&mut self) -> Result<Expr> {
        let mut expr = self.parse_bitwise_xor()?;
        while self.peek_is(Token::Pipe) {
            self.advance();
            let rhs = self.parse_bitwise_xor()?;
            expr = Expr::Binary {
                left: Box::new(expr),
                op: BinaryOp::BitOr,
                right: Box::new(rhs),
            };
        }
        Ok(expr)
    }

    fn parse_bitwise_xor(&mut self) -> Result<Expr> {
        let mut expr = self.parse_bitwise_and()?;
        while self.peek_is(Token::Caret) {
            self.advance();
            let rhs = self.parse_bitwise_and()?;
            expr = Expr::Binary {
                left: Box::new(expr),
                op: BinaryOp::BitXor,
                right: Box::new(rhs),
            };
        }
        Ok(expr)
    }

    fn parse_bitwise_and(&mut self) -> Result<Expr> {
        let mut expr = self.parse_equality()?;
        while self.peek_is(Token::Ampersand) {
            self.advance();
            let rhs = self.parse_equality()?;
            expr = Expr::Binary {
                left: Box::new(expr),
                op: BinaryOp::BitAnd,
                right: Box::new(rhs),
            };
        }
        Ok(expr)
    }

    fn parse_equality(&mut self) -> Result<Expr> {
        let mut expr = self.parse_relational()?;
        loop {
            let op = if self.peek_is(Token::EqualEqual) {
                BinaryOp::Equal
            } else if self.peek_is(Token::NotEqual) {
                BinaryOp::NotEqual
            } else {
                break;
            };
            self.advance();
            let rhs = self.parse_relational()?;
            expr = Expr::Binary {
                left: Box::new(expr),
                op,
                right: Box::new(rhs),
            };
        }
        Ok(expr)
    }

    fn parse_relational(&mut self) -> Result<Expr> {
        let mut expr = self.parse_shift()?;
        loop {
            let op = if self.peek_is(Token::Less) {
                BinaryOp::Less
            } else if self.peek_is(Token::LessEqual) {
                BinaryOp::LessEqual
            } else if self.peek_is(Token::Greater) {
                BinaryOp::Greater
            } else if self.peek_is(Token::GreaterEqual) {
                BinaryOp::GreaterEqual
            } else {
                break;
            };
            self.advance();
            let rhs = self.parse_shift()?;
            expr = Expr::Binary {
                left: Box::new(expr),
                op,
                right: Box::new(rhs),
            };
        }
        Ok(expr)
    }

    fn parse_shift(&mut self) -> Result<Expr> {
        let mut expr = self.parse_additive()?;
        loop {
            let op = if self.peek_is(Token::ShiftLeft) {
                BinaryOp::ShiftLeft
            } else if self.peek_is(Token::ShiftRight) {
                BinaryOp::ShiftRight
            } else {
                break;
            };
            self.advance();
            let rhs = self.parse_additive()?;
            expr = Expr::Binary {
                left: Box::new(expr),
                op,
                right: Box::new(rhs),
            };
        }
        Ok(expr)
    }

    fn parse_additive(&mut self) -> Result<Expr> {
        let mut expr = self.parse_multiplicative()?;
        loop {
            let op = if self.peek_is(Token::Plus) {
                BinaryOp::Add
            } else if self.peek_is(Token::Minus) {
                BinaryOp::Sub
            } else {
                break;
            };
            self.advance();
            let rhs = self.parse_multiplicative()?;
            expr = Expr::Binary {
                left: Box::new(expr),
                op,
                right: Box::new(rhs),
            };
        }
        Ok(expr)
    }

    fn parse_multiplicative(&mut self) -> Result<Expr> {
        let mut expr = self.parse_unary()?;
        loop {
            let op = if self.peek_is(Token::Star) {
                BinaryOp::Mul
            } else if self.peek_is(Token::Slash) {
                BinaryOp::Div
            } else if self.peek_is(Token::Percent) {
                BinaryOp::Mod
            } else {
                break;
            };
            self.advance();
            let rhs = self.parse_unary()?;
            expr = Expr::Binary {
                left: Box::new(expr),
                op,
                right: Box::new(rhs),
            };
        }
        Ok(expr)
    }

    fn parse_unary(&mut self) -> Result<Expr> {
        // C++ dependent template disambiguator `template T::value` / `template T::kHasNormal`
        if self.peek_is(Token::Identifier)
            && self.current_lexeme().eq_ignore_ascii_case("template")
        {
            self.advance();
            // Also skip optional `<...>` after template keyword if present (e.g. `template <typename T>`)
            if self.peek_is(Token::Less) {
                let _ = self.skip_angle_block();
            }
            return self.parse_unary();
        }
        if self.peek_is(Token::PlusPlus) {
            self.advance();
            let inner = self.parse_unary()?;
            // Desugar `++i` to `i = i + 1`
            let one = Expr::Number("1".to_string());
            let rhs = Expr::Binary {
                left: Box::new(inner.clone()),
                op: BinaryOp::Add,
                right: Box::new(one),
            };
            return Ok(Expr::Assign {
                lhs: Box::new(inner),
                rhs: Box::new(rhs),
            });
        }
        if self.peek_is(Token::MinusMinus) {
            self.advance();
            let inner = self.parse_unary()?;
            let one = Expr::Number("1".to_string());
            let rhs = Expr::Binary {
                left: Box::new(inner.clone()),
                op: BinaryOp::Sub,
                right: Box::new(one),
            };
            return Ok(Expr::Assign {
                lhs: Box::new(inner),
                rhs: Box::new(rhs),
            });
        }
        if self.peek_is(Token::Tilde) {
            self.advance();
            return Ok(Expr::Unary {
                op: UnaryOp::BitNot,
                expr: Box::new(self.parse_unary()?),
            });
        }
        // C-style cast: `(float)expr` , `(float2)expr` etc.
        if self.peek_is(Token::LParen) {
            let saved = self.index;
            self.advance();
            if self.is_type_start() {
                // Peek if this looks like a cast type: next tokens should be `Type` then `)`
                // and the `Type` should be a known builtin or struct, not an arbitrary variable.
                // Speculative: token-stream surgery is forbidden inside
                // (only the index rolls back), so `>>` splits are disabled.
                let ty_start = self.index;
                self.speculation_depth += 1;
                let ty_result = self.parse_type();
                self.speculation_depth -= 1;
                if let Ok(mut ty) = ty_result {
                    // Check if ty is a plausible cast type (builtin scalar/vector/matrix,
                    // or a known struct/typedef/template param — e.g. `(PBRInfo)0`
                    // zero-initializes a struct). Resource-kind spellings are
                    // deliberately excluded: `(buffer)`-style grouping around a
                    // variable must keep parsing as grouping, not as a cast.
                    // Scoped names fold to their bare tail when (and only
                    // when) the tail is a known type and the full name is
                    // not — same rule as the post-parse scoped resolution
                    // (`(sce::Got::VisGsOut)0` zero-initializes the
                    // flattened `VisGsOut`; namespaces are flattened at
                    // parse). Scoped variables/enumerators keep grouping.
                    if let Type::Named(n) = &ty {
                        if n.contains("::")
                            && !self.struct_names.contains(n)
                            && !self.typedefs.contains_key(n)
                            && !self.template_type_params.contains(n)
                        {
                            if let Some(bare) = n.rsplit("::").next() {
                                if self.struct_names.contains(bare)
                                    || self.typedefs.contains_key(bare)
                                    || self.template_type_params.contains(bare)
                                {
                                    ty = Type::Named(bare.to_string());
                                }
                            }
                        }
                    }
                    let is_builtin = matches!(
                        ty,
                        Type::Void
                            | Type::Float
                            | Type::Half
                            | Type::Int
                            | Type::UInt
                            | Type::Bool
                            | Type::Double
                            | Type::Vector { .. }
                            | Type::Matrix { .. }
                    ) || matches!(ty, Type::Named(ref n) if {
                        let lower = n.to_ascii_lowercase();
                        matches!(
                            lower.as_str(),
                            "float" | "half" | "int" | "uint" | "bool" | "double"
                        ) || lower.starts_with("float")
                            || lower.starts_with("int")
                            || lower.starts_with("uint")
                            || lower.starts_with("half")
                    }) || matches!(ty, Type::Named(ref n) if {
                        self.struct_names.contains(n)
                            || self.typedefs.contains_key(n)
                            || self.template_type_params.contains(n)
                    });
                    if is_builtin && self.peek_is(Token::RParen) {
                        self.advance();
                        let inner = self.parse_unary()?;
                        return Ok(Expr::Constructor {
                            ty,
                            arguments: vec![inner],
                        });
                    }
                    // Not a valid cast — rollback to try `(expr)` grouping
                    self.index = saved;
                } else {
                    self.index = saved;
                }
                // Also reset ty_start
                let _ = ty_start;
            } else {
                self.index = saved;
            }
        }
        if self.peek_is(Token::Plus) {
            self.advance();
            return Ok(Expr::Unary {
                op: UnaryOp::Plus,
                expr: Box::new(self.parse_unary()?),
            });
        }
        if self.peek_is(Token::Minus) {
            self.advance();
            return Ok(Expr::Unary {
                op: UnaryOp::Minus,
                expr: Box::new(self.parse_unary()?),
            });
        }
        if self.peek_is(Token::Bang) {
            self.advance();
            return Ok(Expr::Unary {
                op: UnaryOp::Not,
                expr: Box::new(self.parse_unary()?),
            });
        }
        if self.peek_is(Token::Star) {
            // C++ pointer dereference `*p` / `*viewData` in `ConvertDepth01(..., *viewData)`
            // HLSL has no pointers — desugar to the underlying value.
            self.advance();
            return self.parse_unary();
        }
        if self.peek_is(Token::Ampersand) {
            // Address-of `&x` — desugar to `x` for HLSL.
            self.advance();
            return self.parse_unary();
        }
        self.parse_postfix()
    }

    fn parse_postfix(&mut self) -> Result<Expr> {
        let mut expr = self.parse_primary()?;

        loop {
            if self.peek_is(Token::Colon) {
                let next_is_colon = self
                    .tokens
                    .get(self.index + 1)
                    .map(|t| t.token == Token::Colon)
                    .unwrap_or(false);
                if next_is_colon {
                    self.advance(); // ':'
                    self.advance(); // ':'
                    let member = self.expect_identifier()?;
                    // `FunctionABI::Create` — treat as scoped variable `FunctionABI::Create`.
                    let combined = match expr {
                        Expr::Variable(name) => format!("{name}::{member}"),
                        _ => member.clone(),
                    };
                    expr = Expr::Variable(combined);
                    continue;
                }
            }

            // Template args `Closest<optimizedFlags>(...)` — skip `<...>`
            // when followed by `(` (call) or `::` (scoped access like
            // `wave::__is_same<A, B>::value`). A bare `<` from a comparison
            // (e.g. `dot(...) < 0.0f`) never leads to either within bounds.
            if self.peek_is(Token::Less) {
                let saved = self.index;
                self.advance(); // '<'
                let mut depth = 1;
                let mut is_template = false;
                let mut look = self.index;
                // Bound the scan: real template arg lists are short and never
                // contain `;`, `{` or `}`. Without this bound, a `<` from a
                // comparison (e.g. `dot(...) < 0.0f`) scans arbitrarily far
                // for a `>` that happens to precede `(`, swallowing whole
                // functions (a `<` in a prior function ate kilobytes to a
                // far-away call).
                let mut steps = 0;
                // Paren/bracket depth inside the candidate arg list: a `>`
                // inside `(...)` (e.g. `enable_if<(kUvCount > 0), ...>`) does
                // not close the list, while an unmatched `)`/`]` (e.g. the
                // `)` of `if (index < 8)`) proves this `<` is a comparison.
                let mut paren = 0usize;
                let mut bracket = 0usize;
                let mut invalid = false;
                while look < self.tokens.len() && depth > 0 && steps < 64 {
                    let tok = &self.tokens[look].token;
                    if *tok == Token::Semi || *tok == Token::LBrace || *tok == Token::RBrace {
                        break;
                    }
                    if *tok == Token::LParen {
                        paren += 1;
                    } else if *tok == Token::RParen {
                        if paren == 0 {
                            invalid = true;
                            break;
                        }
                        paren -= 1;
                    } else if *tok == Token::LBracket {
                        bracket += 1;
                    } else if *tok == Token::RBracket {
                        if bracket == 0 {
                            invalid = true;
                            break;
                        }
                        bracket -= 1;
                    } else if paren == 0 && bracket == 0 {
                        if *tok == Token::Less {
                            depth += 1;
                        } else if *tok == Token::Greater || *tok == Token::ShiftRight {
                            // `>>` is lexed as ShiftRight inside `A<B<C>>`; treat as `>`
                            depth -= 1;
                            if *tok == Token::ShiftRight {
                                // Split `>>` into `>` `>` — handled by expect_greater elsewhere,
                                // but for lookahead we treat it as one `>`
                                // Need to handle second `>`
                                if depth > 0 {
                                    depth -= 1;
                                }
                            }
                        }
                    }
                    look += 1;
                    steps += 1;
                    if depth == 0 {
                        let next_is_paren = self
                            .tokens
                            .get(look)
                            .map(|t| t.token == Token::LParen)
                            .unwrap_or(false);
                        // `Trait<A, B>::value` / `::type` — `::` is two Colons
                        let next_is_scope = matches!(
                            self.tokens.get(look).map(|t| &t.token),
                            Some(Token::Colon)
                        ) && matches!(
                            self.tokens.get(look + 1).map(|t| &t.token),
                            Some(Token::Colon)
                        );
                        is_template = next_is_paren || next_is_scope;
                        break;
                    }
                }
                if invalid {
                    is_template = false;
                }
                self.index = saved;
                if is_template {
                    // `Ngg::CullData<T>::...` — preserve the element type:
                    // the Vulkan/HLSL compat layer lowers this NGG
                    // primitive-shader builtin (`Set` → drop,
                    // `GetTriangle` → zero-initialized `T[3]`) and needs
                    // `T`. Every other template keeps the existing drop
                    // behavior. Only single-identifier args qualify;
                    // anything else falls through to the generic skip.
                    if let Expr::Variable(name) = &expr
                        && (name == "CullData" || name.ends_with("::CullData"))
                        && matches!(
                            self.tokens.get(saved + 1).map(|t| &t.token),
                            Some(Token::Identifier)
                        )
                        && matches!(
                            self.tokens.get(saved + 2).map(|t| &t.token),
                            Some(Token::Greater)
                        )
                    {
                        let tname = self.tokens[saved + 1].lexeme.clone();
                        expr = Expr::Variable(format!("{name}<{tname}>"));
                        self.index = saved + 3;
                        continue;
                    }
                    self.advance(); // '<'
                    let args_start = self.index;
                    let mut d2 = 1;
                    let mut p2 = 0usize;
                    let mut b2 = 0usize;
                    let mut aborted = false;
                    while d2 > 0 && !self.eof() {
                        if self.peek_is(Token::LParen) {
                            p2 += 1;
                        } else if self.peek_is(Token::RParen) {
                            if p2 == 0 {
                                aborted = true;
                                break;
                            }
                            p2 -= 1;
                        } else if self.peek_is(Token::LBracket) {
                            b2 += 1;
                        } else if self.peek_is(Token::RBracket) {
                            if b2 == 0 {
                                aborted = true;
                                break;
                            }
                            b2 -= 1;
                        } else if p2 == 0 && b2 == 0 {
                            if self.peek_is(Token::Less) {
                                d2 += 1;
                            } else if self.peek_is(Token::Greater) {
                                d2 -= 1;
                            } else if self.peek_is(Token::ShiftRight) {
                                // `>>` as one token counts as two `>`
                                d2 -= 1;
                                if d2 > 0 {
                                    d2 -= 1;
                                }
                            }
                        }
                        self.advance();
                        if d2 == 0 {
                            break;
                        }
                    }
                    if aborted {
                        // Lookahead/execution disagreed (should not happen):
                        // rewind and treat `<` as a comparison.
                        self.index = saved;
                    } else if let Expr::Variable(name) = &expr {
                        // Explicit template args (`f<true>(x)`, `f<0, 2>(x)`,
                        // `minT<uint2>(x)`): preserve them in the callee
                        // spelling for monomorphization. Literals keep the
                        // existing value-arg behavior; identifiers (type
                        // names, possibly `::`-scoped) preserve as type-arg
                        // spellings. Anything else (expressions) keeps the
                        // existing drop behavior. Only when the callee is
                        // not already mangled.
                        if !name.contains('<') {
                            // args range excludes the closing '>'.
                            let args_end = self.index - 1;
                            let mut spelling = String::new();
                            let mut ok = args_start < args_end;
                            // Inside `<...>` we expect an operand
                            // (literal/identifier) first, then
                            // comma-separated continuations; `::` only
                            // continues a scoped name.
                            let mut expect_operand = true;
                            let mut k = args_start;
                            while ok && k < args_end {
                                if expect_operand {
                                    if matches!(
                                        self.tokens.get(k).map(|t| &t.token),
                                        Some(Token::True)
                                            | Some(Token::False)
                                            | Some(Token::Number)
                                            | Some(Token::Identifier)
                                    ) {
                                        spelling.push_str(&self.tokens[k].lexeme);
                                        k += 1;
                                        expect_operand = false;
                                    } else {
                                        ok = false;
                                    }
                                } else if matches!(
                                    self.tokens.get(k).map(|t| &t.token),
                                    Some(Token::Comma)
                                ) {
                                    spelling.push(',');
                                    k += 1;
                                    expect_operand = true;
                                } else if matches!(
                                    self.tokens.get(k).map(|t| &t.token),
                                    Some(Token::Colon)
                                ) && matches!(
                                    self.tokens.get(k + 1).map(|t| &t.token),
                                    Some(Token::Colon)
                                ) && matches!(
                                    self.tokens.get(k + 2).map(|t| &t.token),
                                    Some(Token::Identifier)
                                ) {
                                    spelling.push_str("::");
                                    spelling.push_str(&self.tokens[k + 2].lexeme);
                                    k += 3;
                                } else {
                                    ok = false;
                                }
                            }
                            if ok && !expect_operand && !spelling.is_empty() {
                                expr = Expr::Variable(format!("{}<{}>", name, spelling));
                            }
                        }
                    }
                    continue;
                }
            }

            if self.peek_is(Token::Dot) {
                self.advance();
                let member = self.expect_identifier()?;
                expr = Expr::Member {
                    object: Box::new(expr),
                    member,
                };
                continue;
            }

            if self.peek_is(Token::Arrow) {
                self.advance();
                let member = self.expect_identifier()?;
                expr = Expr::Member {
                    object: Box::new(expr),
                    member,
                };
                continue;
            }

            if self.peek_is(Token::LBracket) {
                self.advance();
                let index = self.parse_assignment()?;
                self.expect(Token::RBracket)?;
                expr = Expr::Index {
                    object: Box::new(expr),
                    index: Box::new(index),
                };
                // Linearize flattened 2D indexing: `a[i][j]` with recorded
                // dims `[Y, X]` becomes `a[i * X + j]` (row-major, matching
                // PSSL layout). Only exact Variable bases qualify; member
                // chains, deeper nesting, and unknown names keep today's
                // path.
                if let Expr::Index { object: inner, index: outer } = &expr {
                    if let Expr::Index { object: base, index: inner_idx } = inner.as_ref() {
                        if let Expr::Variable(name) = base.as_ref() {
                            if let Some(stride) = self.multidim_stride(name) {
                                let flat = Expr::Binary {
                                    left: Box::new(Expr::Binary {
                                        left: Box::new(inner_idx.as_ref().clone()),
                                        op: crate::ast::BinaryOp::Mul,
                                        right: Box::new(Expr::Number(stride.to_string())),
                                    }),
                                    op: crate::ast::BinaryOp::Add,
                                    right: Box::new(outer.as_ref().clone()),
                                };
                                expr = Expr::Index {
                                    object: Box::new(Expr::Variable(name.clone())),
                                    index: Box::new(flat),
                                };
                            }
                        }
                    }
                }
                continue;
            }

            if self.peek_is(Token::LParen) {
                self.advance();
                let mut arguments = Vec::new();
                if !self.peek_is(Token::RParen) {
                    loop {
                        arguments.push(self.parse_assignment()?);
                        if !self.peek_is(Token::Comma) {
                            break;
                        }
                        self.advance();
                        // Trailing comma (`f(a, b,)`).
                        if self.peek_is(Token::RParen) {
                            break;
                        }
                    }
                }
                self.expect(Token::RParen)?;

                expr = match expr {
                    Expr::Variable(name) => {
                        if let Some(ty) = builtin_type(&name) {
                            Expr::Constructor { ty, arguments }
                        } else {
                            Expr::Call {
                                function: Box::new(Expr::Variable(name)),
                                arguments,
                            }
                        }
                    }
                    other => Expr::Call {
                        function: Box::new(other),
                        arguments,
                    },
                };
                continue;
            }

            if self.peek_is(Token::LBrace) {
                // Aggregate init `VgprCount {40}` inside `Create(...)` args.
                // Treat `Type { args }` as a constructor.
                let is_type_like = matches!(expr, Expr::Variable(_));
                if is_type_like {
                    self.advance(); // '{'
                    let mut arguments = Vec::new();
                    if !self.peek_is(Token::RBrace) {
                        loop {
                            // C99/PSSL designated initializer `.m_org = ...`
                            // (field names don't affect positional modeling).
                            // Guarded by `Ident =` so leading-dot floats
                            // (`.5`) still parse as numbers.
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
                            arguments.push(self.parse_assignment()?);
                            if !self.peek_is(Token::Comma) {
                                break;
                            }
                            self.advance();
                            // Trailing comma (`T{a, b,}`).
                            if self.peek_is(Token::RBrace) {
                                break;
                            }
                        }
                    }
                    self.expect(Token::RBrace)?;
                    expr = match expr {
                        Expr::Variable(name) => {
                            let ty = builtin_type(&name).unwrap_or(Type::Named(name.clone()));
                            Expr::Constructor { ty, arguments }
                        }
                        other => other,
                    };
                    continue;
                }
            }

            if self.peek_is(Token::PlusPlus) {
                self.advance();
                let one = Expr::Number("1".to_string());
                let rhs = Expr::Binary {
                    left: Box::new(expr.clone()),
                    op: BinaryOp::Add,
                    right: Box::new(one),
                };
                expr = Expr::Assign {
                    lhs: Box::new(expr),
                    rhs: Box::new(rhs),
                };
                continue;
            }

            if self.peek_is(Token::MinusMinus) {
                self.advance();
                let one = Expr::Number("1".to_string());
                let rhs = Expr::Binary {
                    left: Box::new(expr.clone()),
                    op: BinaryOp::Sub,
                    right: Box::new(one),
                };
                expr = Expr::Assign {
                    lhs: Box::new(expr),
                    rhs: Box::new(rhs),
                };
                continue;
            }

            break;
        }

        // Complete N-index chains on flattened 3D+ arrays fold here
        // (postfix-complete: no further index can follow in this call).
        Ok(self.fold_multidim_chain(expr))
    }

    fn parse_primary(&mut self) -> Result<Expr> {
        let token = self.current().ok_or_else(|| self.eof_error("expression"))?;

        match &token.token {
            Token::Number => {
                let value = self.current_lexeme();
                self.advance();
                Ok(Expr::Number(normalize_int_suffix(&value)))
            }
            Token::String => {
                // C adjacent-string concatenation (`"A" "B"` ≡ `"AB"`,
                // used by psr `__error(...)` diagnostics): fold following
                // string literals into one value.
                let mut value = unquote(&self.current_lexeme());
                self.advance();
                while self.peek_is(Token::String) {
                    value.push_str(&unquote(&self.current_lexeme()));
                    self.advance();
                }
                Ok(Expr::String(value))
            }
            Token::True => {
                self.advance();
                Ok(Expr::Bool(true))
            }
            Token::False => {
                self.advance();
                Ok(Expr::Bool(false))
            }
            Token::Identifier
            | Token::Float
            | Token::Half
            | Token::Int
            | Token::UInt
            | Token::Double
            | Token::Bool
            | Token::Void => {
                // Look ahead: if a type name is immediately followed by '(' it could be a
                // function‑type cast or a lambda without capture list (e.g. `float4(`).
                // In PSSL this is legal and should be treated as a placeholder expression.
                let name = self.current_lexeme();
                self.advance();
                // `bit_cast<T>(x)` / C++ `static_cast<T>(x)` (etc.)
                // reinterpret-cast: preserve `T` by modeling the call as a
                // constructor of `T` (component counts are advisory
                // downstream, so width mismatches still pass).
                if (name.eq_ignore_ascii_case("bit_cast")
                    || name.eq_ignore_ascii_case("static_cast")
                    || name.eq_ignore_ascii_case("reinterpret_cast")
                    || name.eq_ignore_ascii_case("const_cast")
                    || name.eq_ignore_ascii_case("dynamic_cast"))
                    && self.peek_is(Token::Less)
                {
                    let saved = self.index;
                    self.advance(); // '<'
                    // Leading qualifiers on the target
                    // (`static_cast<const void*>`).
                    while self.peek_is(Token::Const)
                        || (self.peek_is(Token::Identifier)
                            && matches!(
                                self.current_lexeme().to_ascii_lowercase().as_str(),
                                "const" | "constexpr"
                            ))
                    {
                        self.advance();
                    }
                    if let Ok(ty) = self.parse_type() {
                        // Trailing qualifiers/pointers on the target
                        // (`static_cast<const void*>`).
                        loop {
                            if self.peek_is(Token::Const) {
                                self.advance();
                                continue;
                            }
                            if self.peek_is(Token::Identifier) {
                                let l = self.current_lexeme().to_ascii_lowercase();
                                if l == "const" || l == "constexpr" {
                                    self.advance();
                                    continue;
                                }
                            }
                            if self.peek_is(Token::Star) || self.peek_is(Token::Ampersand) {
                                self.advance();
                                continue;
                            }
                            break;
                        }
                        if self.peek_is(Token::Greater) {
                            self.advance();
                            if self.peek_is(Token::LParen) {
                                self.advance(); // '('
                                let mut args = Vec::new();
                                if !self.peek_is(Token::RParen) {
                                    loop {
                                        let expr = self.parse_assignment()?;
                                        args.push(expr);
                                        if self.peek_is(Token::Comma) {
                                            self.advance();
                                        } else {
                                            break;
                                        }
                                        if self.peek_is(Token::RParen) {
                                            break;
                                        }
                                    }
                                }
                                self.expect(Token::RParen)?;
                                return Ok(Expr::Constructor { ty, arguments: args });
                            }
                        }
                    }
                    // Not `<Type>(args)` after all — rewind and fall through
                    // to the generic handling below.
                    self.index = saved;
                }
                // Array-constructor sugar `float3[](...)` / `T[N](...)`:
                // an array init list whose size the declaration (or the
                // argument count) provides. Only for known type names —
                // `x[0](...)` on a variable stays a postfix call below.
                // (Rollback on any confusion: index-only, no token
                // surgery, always sound.)
                if self.peek_is(Token::LBracket) {
                    let is_type_name = builtin_type(&name).is_some()
                        || self.is_known_type_name(&name);
                    if is_type_name {
                        let saved = self.index;
                        let mut good = true;
                        while self.peek_is(Token::LBracket) {
                            self.advance(); // '['
                            let mut depth = 1usize;
                            while depth > 0 && !self.eof() {
                                if self.peek_is(Token::LBracket) {
                                    depth += 1;
                                } else if self.peek_is(Token::RBracket) {
                                    depth -= 1;
                                } else if self.peek_is(Token::Semi)
                                    || self.peek_is(Token::LBrace)
                                    || self.peek_is(Token::RBrace)
                                {
                                    good = false;
                                    break;
                                }
                                self.advance();
                            }
                            if !good || depth != 0 {
                                good = false;
                                break;
                            }
                        }
                        if good && self.peek_is(Token::LParen) {
                            self.advance(); // '('
                            let mut args = Vec::new();
                            let mut args_good = true;
                            if !self.peek_is(Token::RParen) {
                                loop {
                                    match self.parse_assignment() {
                                        Ok(expr) => args.push(expr),
                                        Err(_) => {
                                            args_good = false;
                                            break;
                                        }
                                    }
                                    if !self.peek_is(Token::Comma) {
                                        break;
                                    }
                                    self.advance();
                                    if self.peek_is(Token::RParen) {
                                        break;
                                    }
                                }
                            }
                            if args_good && self.peek_is(Token::RParen) {
                                self.advance(); // ')'
                                return Ok(Expr::InitializerList(args));
                            }
                        }
                        self.index = saved;
                    }
                }
                if self.peek_is(Token::LParen) {
                    // Determine if this is a constructor call for a known type.
                    let is_builtin_type = matches!(name.to_ascii_lowercase().as_str(),
                        "float" | "float2" | "float3" | "float4" |
                        "half" | "half2" | "half3" | "half4" |
                        "int" | "int2" | "int3" | "int4" |
                        "uint" | "uint2" | "uint3" | "uint4" |
                        "bool" | "double" | "double2" | "double3" | "double4"
                    );
                    // C functional-cast spellings (`unsigned(x)`,
                    // `signed(x)` — framework `Hit::isHit`, `BvhNodeKind`
                    // inits): HLSL has no bare `unsigned` type, so model
                    // as the corresponding scalar constructor directly
                    // instead of an ordinary (undeclared) call.
                    let c_cast_scalar = match name.to_ascii_lowercase().as_str() {
                        "unsigned" | "ulong" | "dword" => Some(Type::UInt),
                        "signed" | "long" | "short" => Some(Type::Int),
                        _ => None,
                    };
                    let is_known_struct = self.is_known_type_name(&name);
                    if is_builtin_type || is_known_struct || c_cast_scalar.is_some() {
                        // Consume '(' and parse arguments.
                        self.advance(); // '('
                        let mut args = Vec::new();
                        if !self.peek_is(Token::RParen) {
                            loop {
                                let expr = self.parse_assignment()?;
                                args.push(expr);
                                if self.peek_is(Token::Comma) {
                                    self.advance();
                                } else {
                                    break;
                                }
                                // Trailing comma (`T(a, b,)`).
                                if self.peek_is(Token::RParen) {
                                    break;
                                }
                            }
                        }
                        self.expect(Token::RParen)?;
                        // Build a type placeholder. User typedefs (including
                        // int-backed enums like `Weapon`) resolve here so
                        // the constructor lowers as the underlying type
                        // instead of degrading to an undeclared call
                        // (`Weapon(x)` → `uint(x)`); structs keep their
                        // `Named` spelling for the struct paths below.
                        // Only value types qualify: resource (or other
                        // exotic) typedef targets keep the `Named`
                        // spelling and stay loud downstream.
                        let ty = c_cast_scalar.or_else(|| builtin_type(&name)).or_else(|| {
                            match self.resolve_typedef(&name) {
                                Some(
                                    t @ (Type::Bool
                                    | Type::Int
                                    | Type::UInt
                                    | Type::Float
                                    | Type::Half
                                    | Type::Double
                                    | Type::Vector { .. }
                                    | Type::Matrix { .. }),
                                ) => Some(t),
                                _ => None,
                            }
                        }).unwrap_or(Type::Named(name.to_string()));
                        return Ok(Expr::Constructor { ty, arguments: args });
                    } else {
                        // An ordinary call `name(args)` — user function,
                        // builtin intrinsic, or macro. Parse it as a real
                        // call so later stages resolve its type; unknown
                        // names report `unknown function` in semantic.
                        self.advance(); // '('
                        let mut args = Vec::new();
                        if !self.peek_is(Token::RParen) {
                            loop {
                                let expr = self.parse_assignment()?;
                                args.push(expr);
                                if self.peek_is(Token::Comma) {
                                    self.advance();
                                } else {
                                    break;
                                }
                                if self.peek_is(Token::RParen) {
                                    break;
                                }
                            }
                        }
                        self.expect(Token::RParen)?;
                        return Ok(Expr::Call {
                            function: Box::new(Expr::Variable(name)),
                            arguments: args,
                        });
                    }
                }
                Ok(Expr::Variable(name))
            }

            Token::LParen => {
                self.advance();
                let mut expr = self.parse_assignment()?;
                // Parenthesized comma expression `(a, b)` evaluates to `b`.
                while self.peek_is(Token::Comma) {
                    self.advance();
                    let rhs = self.parse_assignment()?;
                    expr = Expr::Binary {
                        left: Box::new(expr),
                        op: crate::ast::BinaryOp::Comma,
                        right: Box::new(rhs),
                    };
                }
                self.expect(Token::RParen)?;
                Ok(expr)
            }
            Token::LBrace => {
                // Braced initializer list as an expression: `return {a, b};`,
                // `x = {1, 2};`, `foo({1, 2})`. Downstream models
                // `InitializerList` end to end (semantic/lower/emit).
                self.parse_initializer_list()
            }
            Token::LBracket => {
                // C++ lambda capture `[srt]` / `[&]` / `[=]` followed by `(params)` and `{body}`
                // e.g. `auto const anyHit = [srt] (float3 a, Hit h) { ... };`
                self.advance(); // '['
                let cap_start = self.index;
                let mut depth = 1;
                while depth > 0 && !self.eof() {
                    if self.peek_is(Token::LBracket) {
                        depth += 1;
                    } else if self.peek_is(Token::RBracket) {
                        depth -= 1;
                    }
                    self.advance();
                    if depth == 0 {
                        break;
                    }
                }
                // Real parse (params + body preserved for beta-reduction
                // and immediate inlining); confusion falls back to the
                // legacy opaque path below. Capture items recorded in
                // order (identifiers verbatim; `,`/`&`/`=` as symbols)
                // for the capture-less gate.
                let mut captures: Vec<String> = Vec::new();
                for t in &self.tokens[cap_start..self.index - 1] {
                    match &t.token {
                        Token::Comma => captures.push(",".to_string()),
                        Token::Ampersand => captures.push("&".to_string()),
                        Token::Equal => captures.push("=".to_string()),
                        Token::Identifier => captures.push(t.lexeme.clone()),
                        // `&x` lexes as one identifier in some positions;
                        // anything else here means confusion (opaque path
                        // via the failed real parse below).
                        _ => captures.push(t.lexeme.clone()),
                    }
                }
                if let Some(lambda) = self.try_parse_lambda(captures) {
                    return Ok(lambda);
                }
                if self.peek_is(Token::LParen) {
                    self.skip_paren_block()?;
                }
                // Optional `mutable` specifier and trailing return type:
                // `[cap](uint i) mutable -> uint { ... }`.
                if self.peek_is(Token::Identifier)
                    && self.current_lexeme().eq_ignore_ascii_case("mutable")
                {
                    self.advance();
                }
                if self.peek_is(Token::Arrow) {
                    self.advance(); // '->'
                    let _ = self.parse_type()?;
                    // Trailing `const`/`&`/`*` on the return type
                    // (`-> sce::Psr::RunTimeInstance const *`).
                    loop {
                        if self.peek_is(Token::Const) {
                            self.advance();
                            continue;
                        }
                        if self.peek_is(Token::Identifier) {
                            let l = self.current_lexeme().to_ascii_lowercase();
                            if l == "const" || l == "constexpr" {
                                self.advance();
                                continue;
                            }
                        }
                        if self.peek_is(Token::Star) || self.peek_is(Token::Ampersand) {
                            self.advance();
                            continue;
                        }
                        break;
                    }
                }
                if self.peek_is(Token::LBrace) {
                    // Do not consume trailing `;` — that's the vardecl terminator.
                    self.expect(Token::LBrace)?;
                    let mut brace_depth = 1;
                    while brace_depth > 0 && !self.eof() {
                        if self.peek_is(Token::LBrace) {
                            brace_depth += 1;
                        } else if self.peek_is(Token::RBrace) {
                            brace_depth -= 1;
                        }
                        self.advance();
                    }
                }
                // Placeholder that won't trigger semantic "unknown variable":
                // lambdas are callbacks desugared to an opaque callable value
                // (so `auto g = [&](...){...}; g(...)` typechecks through the
                // unknown-type rules). A bool literal would poison member
                // access and arithmetic on results.
                Ok(Expr::Constructor {
                    ty: Type::Named("__lambda".to_string()),
                    arguments: Vec::new(),
                })
            }
            Token::Hash | Token::HashHash => {
                self.advance();
                if self.peek_is(Token::Identifier) {
                    self.advance();
                    Ok(Expr::String("".to_string()))
                } else {
                    self.parse_primary()
                }
            }
            _ => Err(self.error_expected("expression")),
        }
    }

    /// Parses a lambda after `[...]`: `(params)` (C++14 may omit the
    /// parens for zero params), optional `mutable` / trailing return,
    /// and a real `{body}`. Returns `None` (cursor reset past `]`) on any
    /// confusion — the caller falls back to the opaque `__lambda`
    /// placeholder, exactly today's behavior.
    fn try_parse_lambda(&mut self, captures: Vec<String>) -> Option<Expr> {
        let entry = self.index;
        // Parameter names: last identifier per comma item (`ulong v` ->
        // `v`, `const float3& x` -> `x`). `(void)` and missing parens
        // both mean zero params.
        let params = if self.peek_is(Token::LParen) {
            match self.scan_lambda_params() {
                Some(p) => p,
                None => {
                    self.index = entry;
                    return None;
                }
            }
        } else {
            Vec::new()
        };
        // Optional `mutable` specifier and trailing return type (both
        // irrelevant post-inline; skipped best-effort).
        if self.peek_is(Token::Identifier)
            && self.current_lexeme().eq_ignore_ascii_case("mutable")
        {
            self.advance();
        }
        if self.peek_is(Token::Arrow) {
            self.advance(); // '->'
            self.speculation_depth += 1;
            let ty = self.parse_type();
            self.speculation_depth -= 1;
            if ty.is_err() {
                self.index = entry;
                return None;
            }
            loop {
                if self.peek_is(Token::Const) {
                    self.advance();
                    continue;
                }
                if self.peek_is(Token::Identifier) {
                    let l = self.current_lexeme().to_ascii_lowercase();
                    if l == "const" || l == "constexpr" {
                        self.advance();
                        continue;
                    }
                }
                if self.peek_is(Token::Star) || self.peek_is(Token::Ampersand) {
                    self.advance();
                    continue;
                }
                break;
            }
        }
        if !self.peek_is(Token::LBrace) {
            self.index = entry;
            return None;
        }
        // Real body parse under the speculation guard (token-stream
        // surgery is forbidden while speculating; only the index rolls
        // back, mirroring the C-style cast probe).
        self.speculation_depth += 1;
        let body = self.parse_block();
        self.speculation_depth -= 1;
        match body {
            Ok(body) => Some(Expr::Lambda { captures, params, body }),
            Err(_) => {
                self.index = entry;
                None
            }
        }
    }

    /// Scans `(params)` at the cursor (which must be `(`), returning the
    /// parameter names in order. Pure index walk (no token surgery), so
    /// rollback is always sound. `None` on anything unexpected (default
    /// args, function pointers, `...`, unbalanced brackets).
    fn scan_lambda_params(&mut self) -> Option<Vec<String>> {
        self.advance(); // '('
        let mut params = Vec::new();
        if self.peek_is(Token::RParen) {
            self.advance();
            return Some(params);
        }
        loop {
            let mut last_ident: Option<String> = None;
                        let mut depth_angle = 0usize;
            let mut depth_paren = 0usize;
            let mut depth_brack = 0usize;
            loop {
                if self.eof() {
                    return None;
                }
                if depth_angle == 0 && depth_paren == 0 && depth_brack == 0
                    && (self.peek_is(Token::Comma) || self.peek_is(Token::RParen))
                {
                    break;
                }
                let tok = self.tokens.get(self.index).map(|t| t.token.clone());
                match tok {
                    Some(Token::Identifier) => {
                        last_ident = Some(self.current_lexeme());
                    }
                    // `>>` lexes jointly inside `A<B<C>>`: closes two levels.
                    Some(Token::ShiftRight) if depth_angle >= 2 => {
                        depth_angle -= 2;
                    }
                    Some(Token::Less) => {
                        depth_angle += 1;
                    }
                    Some(Token::Greater) => {
                        if depth_angle == 0 {
                            return None;
                        }
                        depth_angle -= 1;
                    }
                    Some(Token::LParen) => {
                        depth_paren += 1;
                    }
                    Some(Token::RParen) => {
                        if depth_paren == 0 {
                            return None;
                        }
                        depth_paren -= 1;
                    }
                    Some(Token::LBracket) => {
                        depth_brack += 1;
                    }
                    Some(Token::RBracket) => {
                        if depth_brack == 0 {
                            return None;
                        }
                        depth_brack -= 1;
                    }
                    // Default args, `...`, stray punctuation: give up
                    // (opaque fallback).
                    Some(Token::Equal)
                    | Some(Token::Dot)
                    | Some(Token::Semi)
                    | Some(Token::LBrace)
                    | Some(Token::RBrace) => return None,
                    None => return None,
                    _ => {
                    }
                }
                self.advance();
            }
            match last_ident {
                Some(name) => params.push(name),
                // Nameless items (`(void)`, `...` already excluded) fall
                // back to the opaque placeholder (today's behavior).
                // Nameless items fall back to the opaque placeholder.
                None => return None,
            }
            if self.peek_is(Token::Comma) {
                self.advance();
                // Trailing comma (`(uint a,)`).
                if self.peek_is(Token::RParen) {
                    self.advance();
                    return Some(params);
                }
                continue;
            }
            if self.peek_is(Token::RParen) {
                self.advance();
                return Some(params);
            }
            return None;
        }
    }
}
