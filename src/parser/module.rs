//! Module-level item parsing: resources, structs, functions, parameters.

use std::collections::HashMap;

use super::{Parser, TRACE_ENTER};
use crate::ast::*;
use crate::error::{Phase, Ps5ShaderError, Result, Span};
use crate::token::Token;

use super::types::{is_resource_kind, parse_register_str};

impl Parser {
    pub(crate) fn peek_resource_kind(&self) -> Option<String> {
        let lexeme = self.current_lexeme();
        if is_resource_kind(&lexeme) {
            Some(lexeme)
        } else {
            None
        }
    }

    pub(crate) fn parse_resource(&mut self, _attributes: Vec<Attribute>) -> Result<ResourceDecl> {
        let start = self.current_span().start;
        let kind = self.expect_identifier()?;
        tracing::trace!(
            target: TRACE_ENTER,
            line = start,
            kind = %kind,
            "parse_resource",
        );
                // Parse optional template element for resource type
        let element_ty = if self.peek_is(Token::Less) {
            Some(self.parse_resource_template_element()?)
        } else {
            None
        };
        // Skip optional const/constexpr/static before the resource name
        while self.peek_is(Token::Const)
            || self.peek_is(Token::Static)
            || (self.peek_is(Token::Identifier) && {
                let lex = self.current_lexeme();
                lex.eq_ignore_ascii_case("constexpr") || lex.eq_ignore_ascii_case("const")
            }) {
            self.advance();
        }
        let name = self.expect_identifier()?;
        // Allow comma‑separated resource names (ignore extra names)
        while self.peek_is(Token::Comma) {
            self.advance(); // consume ','
            // optional const/constexpr/static before the next identifier
            while self.peek_is(Token::Const)
                || self.peek_is(Token::Static)
                || (self.peek_is(Token::Identifier) && {
                    let lex = self.current_lexeme();
                    lex.eq_ignore_ascii_case("constexpr") || lex.eq_ignore_ascii_case("const")
                }) {
                self.advance();
            }
            // consume the extra identifier (ignored)
            let _ = self.expect_identifier();
        }
        // Skip optional qualifiers after the name (e.g., `const` before `;`)
        while self.peek_is(Token::Const)
            || self.peek_is(Token::Static)
            || (self.peek_is(Token::Identifier) && {
                let lex = self.current_lexeme();
                lex.eq_ignore_ascii_case("constexpr") || lex.eq_ignore_ascii_case("const")
            }) {
            self.advance();
        }
        let mut register = None;
        let register_space = None;
        let mut semantic = None;
        if self.peek_is(Token::Colon) {
            self.advance();
            let reg_word = self.expect_identifier()?;
            if reg_word.eq_ignore_ascii_case("register") {
                self.expect(Token::LParen)?;
                let reg_str = self.expect_identifier()?;
                if let Some((c, n)) = parse_register_str(&reg_str) {
                    register = Some((c, n));
                } else {
                    return Err(Ps5ShaderError::at(
                        format!("invalid register '{reg_str}'"),
                        Phase::Parse,
                        self.previous_span(),
                    ));
                }
                if self.peek_is(Token::Comma) {
                    self.advance();
                    // Optional space specifier like `space1` — ignore for now.
                    let _ = self.expect_identifier();
                    if self.peek_is(Token::LParen) {
                        // Handle space(1) form? Not needed for corpus.
                    }
                }
                self.expect(Token::RParen)?;
            } else {
                // `: SEMANTIC` binding instead of `register(...)` (e.g.
                // SRT-bound `TextureCube<float3> env : S_SRT_DATA;`).
                semantic = Some(Semantic::Name(reg_word));
            }
        }
        // Cbuffer members share the struct member grammar (nested
        // structs, methods, typedefs, comma declarators, ...): parse the
        // body through a synthetic anonymous struct and keep its fields.
        // (Nested declarations inside still reach `pending_structs` for
        // the module parser to drain.)
        let mut fields = Vec::new();
        if self.peek_is(Token::LBrace) {
            let synth = format!("{name}__cbuffer_members");
            let span = self.current_span();
            self.tokens.insert(
                self.index,
                crate::token::SpannedToken {
                    token: Token::Struct,
                    lexeme: "struct".to_string(),
                    span,
                },
            );
            self.tokens.insert(
                self.index + 1,
                crate::token::SpannedToken {
                    token: Token::Identifier,
                    lexeme: synth,
                    span,
                },
            );
            let member_struct = self.parse_struct()?;
            fields = member_struct.fields;
        } else {
            // Optional initializer, e.g.
            // `static const SamplerState tx_sampler = make_bilinear_sampler();`
            // (psfx compute shaders). Consumed and discarded — resources are
            // bound, not constructed, in the emitted HLSL.
            if self.peek_is(Token::Equal) {
                self.advance();
                if self.peek_is(Token::LBrace) {
                    let mut depth = 0usize;
                    while !self.eof() {
                        if self.peek_is(Token::LBrace) {
                            depth += 1;
                        } else if self.peek_is(Token::RBrace) {
                            depth = depth.saturating_sub(1);
                        }
                        self.advance();
                        if depth == 0 {
                            break;
                        }
                    }
                } else {
                    let _ = self.parse_assignment()?;
                }
            }
            self.expect_semi()?;
        }
        let end = self.previous_span().end;
        Ok(ResourceDecl {
            name,
            kind,
            element_ty,
            register,
            register_space,
            semantic,
            fields,
            span: Span::new(start, end),
        })
    }

    pub(crate) fn parse_struct(&mut self) -> Result<StructDecl> {
        // A `template<...>` prefix on a struct (e.g. `template<typename
        // T> struct V2P`) belongs to the struct (struct templates carry
        // their params for monomorphization); anything else leaves it
        // empty for the next function drain.
        let template_params = std::mem::take(&mut self.pending_template_params);
        // Value-param defaults drain with the params (scoped by struct
        // name once known, below).
        let template_defaults = std::mem::take(&mut self.pending_template_defaults);
        // Field spelling frame for template-id member types (drained at
        // each struct end). Skipped under speculation (index-only
        // rollback would leak frames and misattribute fields).
        let field_frame = self.speculation_depth == 0;
        if field_frame {
            self.pending_field_spellings.push(Vec::new());
        }
        let struct_kw = self.current_span();
        self.expect(Token::Struct)?;
        // Alignment/attribute macro between `struct` and its name
        // (`struct RT_ALIGNAS(64) S64Byte {...}`).
        if self.peek_is(Token::Identifier)
            && self.peek_next_is(Token::LParen)
            && Self::is_macro_like_name(&self.current_lexeme())
        {
            self.advance(); // macro
            self.skip_paren_block()?;
        }
        let mut name = self.expect_identifier()?;
        // Template specialization / generic params `struct Foo<T>` / `struct enable_if<true,T>`
        if self.peek_is(Token::Less) {
            // Best-effort skip of `<...>` including nested `>>`
            let args_start = self.index;
            let _ = self.skip_angle_block();
            // Explicit specialization (`template <> struct Indices<2>`
            // with no pending template params): mangle the definition
            // name to the sanitized spelling (`Indices_2`) so it no
            // longer collides with the primary template and use sites
            // match it exactly. Primary definitions (pending params
            // present) and plain structs keep their names; anything
            // unparseable keeps today's skip behavior.
            if template_params.is_empty() {
                let args_text: String = self.tokens[args_start..self.index]
                    .iter()
                    .map(|t| t.lexeme.as_str())
                    .collect::<Vec<_>>()
                    .join("");
                let args_text: String = args_text
                    .split_whitespace()
                    .collect::<Vec<_>>()
                    .join("");
                // Strip the enclosing `<...>`; require non-empty flat args.
                if let Some(inner) = args_text
                    .strip_prefix('<')
                    .and_then(|s| s.strip_suffix('>'))
                {
                    if !inner.is_empty()
                        && !inner.contains('<')
                        && !inner.contains('>')
                    {
                        // Const-fold pure-integer identifiers via known
                        // constants (`Indices<kTwo>` == `Indices<2>`)
                        // so spellings match canonically; anything else
                        // keeps its spelling.
                        let raw_parts: Vec<String> = inner
                            .split(',')
                            .map(|a| a.trim().to_string())
                            .collect();
                        let folded: Vec<String> = raw_parts
                            .iter()
                            .map(|a| {
                                self.int_consts
                                    .get(a)
                                    .cloned()
                                    .unwrap_or_else(|| a.clone())
                            })
                            .collect();
                        let mut candidate =
                            sanitize_template_name(&format!("{}<{}>", name, folded.join(",")));
                        // Never collide with a declared struct name.
                        let mut n = 0u32;
                        while self.struct_names.contains(&candidate) {
                            n += 1;
                            candidate = format!(
                                "{}_{}",
                                sanitize_template_name(&format!(
                                    "{}<{}>",
                                    name,
                                    folded.join(",")
                                )),
                                n
                            );
                        }
                        // Record raw and folded spellings (whitespace
                        // stripped) so use sites match regardless of
                        // const-foldability at their position.
                        let raw_key = format!("{}<{}>", name, raw_parts.join(","));
                        let spec_map = self
                            .template_site_spellings
                            .entry("@spec".to_string())
                            .or_default();
                        spec_map.insert(raw_key, candidate.clone());
                        let folded_key =
                            format!("{}<{}>", name, folded.join(","));
                        spec_map.insert(folded_key, candidate.clone());
                        name = candidate;
                    }
                }
            }
        }
        self.struct_names.insert(name.clone());
        // Allow forward declaration `struct Foo;`
        if self.peek_is(Token::Semi) {
            // consume the semicolon and return an empty struct declaration
            self.advance();
            let end_span = self.previous_span();
            if field_frame {
                self.drain_field_spellings(&name);
            }
            return Ok(StructDecl {
                name,
                fields: Vec::new(),
                bases: Vec::new(),
                span: struct_kw.merge(end_span),
                template_params: template_params.clone(),
            });
        }
        // Inheritance list `struct Derived : public Base, ...` (e.g.
        // `struct VcoStatsSRT : BasicDrawSRT`). Base names are recorded
        // for post-parse flattening (base fields prepend, derived wins
        // on name clash, matching C++ unqualified lookup).
        let mut bases: Vec<String> = Vec::new();
        if self.peek_is(Token::Colon)
            && !self
                .tokens
                .get(self.index + 1)
                .map(|t| t.token == Token::Colon)
                .unwrap_or(false)
        {
            self.advance(); // ':'
            while !self.peek_is(Token::LBrace)
                && !self.peek_is(Token::Semi)
                && !self.eof()
            {
                if self.peek_is(Token::Less) {
                    let _ = self.skip_angle_block();
                } else if self.peek_is(Token::LParen) {
                    self.skip_paren_block()?;
                } else if self.peek_is(Token::Identifier) {
                    let lex = self.current_lexeme();
                    let lower = lex.to_ascii_lowercase();
                    if lower != "public" && lower != "private" && lower != "protected" && lower != "virtual"
                    {
                        // Strip any `Ns::` qualifier (flattened namespaces
                        // register bare names).
                        let bare = lex.rsplit("::").next().unwrap_or(&lex).to_string();
                        if !bases.contains(&bare) {
                            bases.push(bare);
                        }
                    }
                    self.advance();
                } else {
                    self.advance();
                }
            }
        }
        self.expect(Token::LBrace)?;
        tracing::trace!(
            target: TRACE_ENTER,
            line = struct_kw.start,
            name = %name,
            "parse_struct",
        );

        let mut fields = Vec::new();
        // `static const` member values seen so far in this struct, for
        // array sizes like `float4 planes[NumPlanes]` (C++ member scope:
        // members first, then globals).
        let mut member_consts: HashMap<String, String> = HashMap::new();
        while !self.peek_is(Token::RBrace) {
            // Stray `;` (empty member declaration) — skip it.
            if self.peek_is(Token::Semi) {
                self.advance();
                continue;
            }
            // Leading `[...]` attributes on members (`[isolate] void
            // push(...)` in psr runtime structs): skipped balanced, never
            // interpreted (shader-irrelevant hints; struct fields carry
            // no attribute table). Only at member-head position, where
            // `[` cannot start any other construct.
            while self.peek_is(Token::LBracket) {
                let mut depth = 0usize;
                let mut closed = false;
                while !self.eof() {
                    if self.peek_is(Token::LBracket) {
                        depth += 1;
                    } else if self.peek_is(Token::RBracket) {
                        depth = depth.saturating_sub(1);
                        if depth == 0 {
                            self.advance();
                            closed = true;
                            break;
                        }
                    }
                    self.advance();
                }
                if !closed {
                    break;
                }
            }
            // `template<...>` prefix on member declarations (e.g.
            // `template <typename T> static T getInvalidValue();`).
            // Consumed (recording `typename`/`class` params); the member
            // itself is parsed by the next iteration.
            if self.peek_is(Token::Identifier)
                && self.current_lexeme().eq_ignore_ascii_case("template")
                && self.peek_next_is(Token::Less)
            {
                self.advance(); // 'template'
                let arg_start = self.index; // at '<'
                let _ = self.skip_angle_block();
                for idx in arg_start..self.index.saturating_sub(1) {
                    if let Some(tok) = self.tokens.get(idx) {
                        let lower = tok.lexeme.to_ascii_lowercase();
                        if (lower == "typename" || lower == "class")
                            && tok.token == Token::Identifier
                        {
                            if let Some(next) = self.tokens.get(idx + 1) {
                                if next.token == Token::Identifier {
                                    self.template_type_params.insert(next.lexeme.clone());
                                }
                            }
                        }
                    }
                }
                continue;
            }
            // Optional qualifiers: consumed only when a declaration tail
            // follows (`sample float4 x`); otherwise a type/use/call
            // head (`Sample get_sample(`, `Shared::SRT`, `Sample s;`).
            loop {
                if self.peek_is(Token::Const) || self.peek_is(Token::Static) {
                    self.advance();
                    continue;
                }
                if self.peek_is(Token::Identifier)
                    && Self::is_decl_modifier_word(&self.current_lexeme())
                    && self.modifier_followed_by_decl()
                {
                    self.advance();
                    continue;
                }
                break;
            }
            // Template-id member types (`Indices<2> indices;`): record
            // the spelling window for value-template instantiation
            // (hooked per declarator below; the window covers type and
            // name but not dims/semantics/initializers).
            let member_ty_start = self.index;
            // Nested `enum [class|struct] [Name] [: type] { ... } [instance];`
            // inside a struct — enumerators carry no runtime value; an
            // instance (e.g. `enum { ... } state;`) becomes an Int field
            // (the C++ unscoped-enum lowering).
            if self.peek_is(Token::Identifier)
                && self.current_lexeme().eq_ignore_ascii_case("enum")
            {
                self.advance(); // 'enum'
                if self.peek_is(Token::Identifier)
                    && matches!(
                        self.current_lexeme().to_ascii_lowercase().as_str(),
                        "class" | "struct"
                    )
                {
                    self.advance(); // `enum class` / `enum struct`
                }
                if self.peek_is(Token::Identifier) {
                    self.advance(); // type name
                }
                // Fixed underlying type (`enum E : uint {...}`) — dropped.
                if self.peek_is(Token::Colon)
                    && !self
                        .tokens
                        .get(self.index + 1)
                        .map(|t| t.token == Token::Colon)
                        .unwrap_or(false)
                {
                    self.advance(); // ':'
                    let _ = self.parse_type();
                }
                if self.peek_is(Token::LBrace) {
                    // Nested enumerators register as int globals (same as
                    // top-level enums): qualified uses (`S::E`) fold to
                    // the bare name in lowering. Lenient here — an
                    // unbalanced body fails the enclosing struct parse
                    // downstream exactly as before. One inner vec per
                    // enum body (auto-increment restarts per enum).
                    let names = self.parse_enumerator_names().unwrap_or_default();
                    if !names.is_empty() {
                        self.pending_enum_names.push(names);
                    }
                    // Optional instance `} var [dims] [: sem] [= init];`.
                    // (parse_enumerator_names eats through `}` only.)
                    if self.peek_is(Token::Identifier)
                        && matches!(
                            self.tokens.get(self.index + 1).map(|t| &t.token),
                            Some(
                                Token::Semi
                                    | Token::LBracket
                                    | Token::Colon
                                    | Token::Equal
                                    | Token::Comma
                            )
                        )
                    {
                        loop {
                            let var_name = self.expect_identifier()?;
                            let mut array_size = None;
                            let mut first_dim = true;
                            while self.peek_is(Token::LBracket) {
                                self.advance();
                                if !self.peek_is(Token::RBracket) {
                                    let size_expr = self.parse_assignment()?;
                                    if first_dim {
                                        if let Some(n) = self.resolve_array_size(&size_expr) {
                                            array_size = Some(n);
                                        }
                                    }
                                }
                                self.expect(Token::RBracket)?;
                                first_dim = false;
                            }
                            let semantic = self.parse_optional_semantic()?;
                            if self.peek_is(Token::Equal) {
                                self.advance();
                                let _ = self.parse_assignment()?;
                            }
                            let span = self.previous_span().merge(struct_kw);
                            fields.push(FieldDecl {
                                ty: Type::Int,
                                name: var_name,
                                semantic,
                                array_size,
                                span,
                            });
                            if !self.peek_is(Token::Comma) {
                                break;
                            }
                            self.advance(); // ','
                        }
                        self.expect_semi()?;
                    }
                } else {
                    self.expect_semi()?;
                }
                continue;
            }
            // Nested `struct Name { ... } [instance];`, anonymous
            // `struct { ... } [*]instance;`, or an elaborated type use
            // (`struct MeshData* mesh;`) inside a struct.
            let mut elaborated_ty: Option<Type> = None;
            if self.peek_is(Token::Struct) {
                let t1 = self.tokens.get(self.index + 1).map(|t| t.token.clone());
                let t2 = self.tokens.get(self.index + 2).map(|t| t.token.clone());
                let is_decl = match (&t1, &t2) {
                    (Some(Token::LBrace), _) => true,
                    (Some(Token::Identifier), Some(Token::LBrace) | Some(Token::Semi)) => true,
                    (Some(Token::Identifier), Some(Token::Less)) => {
                        // `struct Foo<...>`: declaration iff the template
                        // list is followed by `{` or `;`.
                        let mut k = self.index + 2;
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
                            } else if matches!(
                                self.tokens[k].token,
                                Token::Semi | Token::LBrace | Token::RBrace
                            ) {
                                break;
                            }
                            k += 1;
                            if k - self.index > 60 {
                                break;
                            }
                        }
                        matches!(
                            closed.and_then(|c| self.tokens.get(c + 1)).map(|t| &t.token),
                            Some(Token::LBrace) | Some(Token::Semi)
                        )
                    }
                    _ => false,
                };
                if !is_decl {
                    self.advance();
                    let mut full = self.expect_identifier()?;
                    while self.peek_is(Token::Colon)
                        && self.tokens.get(self.index + 1).map(|t| t.token == Token::Colon).unwrap_or(false)
                    {
                        full.push_str("::");
                        self.advance();
                        self.advance();
                        full.push_str(&self.expect_identifier()?);
                    }
                    if self.peek_is(Token::Less) {
                        let _ = self.skip_angle_block();
                    }
                    elaborated_ty = Some(Type::Named(full));
                } else {
                    let has_name = matches!(t1, Some(Token::Identifier));
                if !has_name {
                    // Anonymous: inject a synthetic name token so the shared
                    // struct parser consumes the body normally.
                    let synth = format!("{name}__anon{}", self.anon_counter);
                    self.anon_counter += 1;
                    let span = self.current_span();
                    self.tokens.insert(
                        self.index + 1,
                        crate::token::SpannedToken {
                            token: Token::Identifier,
                            lexeme: synth,
                            span,
                        },
                    );
                }
                let decl = self.parse_struct()?;
                let nested_name = decl.name.clone();
                let nested_fields = decl.fields.clone();
                self.pending_structs.push(decl);
                // Instance tail present?
                let has_instance = if self.peek_is(Token::Star)
                    || self.peek_is(Token::Ampersand)
                {
                    true
                } else if self.peek_is(Token::Identifier) {
                    matches!(
                        self.tokens.get(self.index + 1).map(|t| &t.token),
                        Some(
                            Token::Semi
                                | Token::LBracket
                                | Token::Colon
                                | Token::Equal
                                | Token::Comma
                        )
                    )
                } else {
                    false
                };
                if !has_instance {
                    if !has_name {
                        // Anonymous member `struct { ... };` — C promotes
                        // the members into the enclosing struct.
                        fields.extend(nested_fields);
                    }
                    // Named + bare: declaration already stashed.
                } else {
                    // `[*&] name [dims] [: semantic] [= init] [, ...] ;`
                    loop {
                        if self.peek_is(Token::Star)
                            || self.peek_is(Token::Ampersand)
                        {
                            self.advance();
                        }
                        let var_name = self.expect_identifier()?;
                        let mut array_size = None;
                        let mut first_dim = true;
                        while self.peek_is(Token::LBracket) {
                            self.advance();
                            if !self.peek_is(Token::RBracket) {
                                let size_expr = self.parse_assignment()?;
                                if first_dim {
                                    if let Some(n) = self.resolve_array_size(&size_expr) {
                                        array_size = Some(n);
                                    }
                                }
                            }
                            self.expect(Token::RBracket)?;
                            first_dim = false;
                        }
                        let semantic = self.parse_optional_semantic()?;
                        if self.peek_is(Token::Equal) {
                            self.advance();
                            let _ = self.parse_assignment()?;
                        }
                        let span = self.previous_span().merge(struct_kw);
                        fields.push(FieldDecl {
                            ty: Type::Named(nested_name.clone()),
                            name: var_name,
                            semantic,
                            array_size,
                            span,
                        });
                        if !self.peek_is(Token::Comma) {
                            break;
                        }
                        self.advance(); // ','
                    }
                    self.expect_semi()?;
                }
                    // Declaration paths are complete — skip the shared
                    // declarator tail below (the elaborated-use path above
                    // falls through to it instead).
                    continue;
                }
            }
            // `static_assert(...)` / `assert(...)` inside a struct body — skip.
            if self.peek_is(Token::Identifier)
                && matches!(
                    self.current_lexeme().to_ascii_lowercase().as_str(),
                    "static_assert" | "assert"
                )
            {
                self.advance();
                self.skip_paren_block()?;
                self.expect_semi()?;
                continue;
            }
            // `typedef` inside a struct (e.g. `typedef Empty type;` in enable_if)
            if self.peek_is(Token::Identifier)
                && self.current_lexeme().eq_ignore_ascii_case("typedef")
            {
                self.advance(); // 'typedef'
                // Skip storage modifiers (`row_major`, `const`, ...) and
                // the C++ `typename` prefix on dependent types.
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
                let base_ty = self.parse_type()?;
                // Trailing `::type` after a template
                // (`ShuffleType<...>::type`).
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
                    // The alias may re-spell a keyword scalar
                    // (`typedef unsigned int uint;`).
                    let alias = self.expect_alias_name()?;
                    while self.peek_is(Token::LBracket) {
                        self.advance();
                        while !self.peek_is(Token::RBracket) && !self.eof() {
                            self.advance();
                        }
                        self.expect(Token::RBracket)?;
                    }
                    self.typedefs.insert(alias, base_ty.clone());
                    if self.peek_is(Token::Comma) {
                        self.advance();
                    } else {
                        break;
                    }
                }
                self.expect_semi()?;
                continue;
            }
            // `union { ... }` inside a struct – PSSL uses unions for type-punning pointers (addressToPointer)
            if self.peek_is(Token::Identifier) && self.current_lexeme().eq_ignore_ascii_case("union") {
                self.advance();
                if self.peek_is(Token::LBrace) {
                    self.skip_brace_block()?;
                } else if self.peek_is(Token::Identifier) {
                    self.advance();
                    // Template arguments (`union X<A, B> {...}`).
                    if self.peek_is(Token::Less) {
                        let _ = self.skip_angle_block();
                    }
                    if self.peek_is(Token::LBrace) {
                        self.skip_brace_block()?;
                    } else {
                        self.expect_semi()?;
                    }
                } else {
                    self.expect_semi()?;
                }
                continue;
            }
            // Return-less operator declarations inside a struct
            // (`operator float() const;`, `operator[](int i) {...}`):
            // consume `operator` + target, then params/body like a method.
            if self.peek_is(Token::Identifier) && self.current_lexeme() == "operator" {
                self.advance(); // 'operator'
                if self.peek_is(Token::LParen) {
                    // `operator()` target only when empty with params
                    // following; otherwise these ARE the params.
                    let empty = matches!(
                        self.tokens.get(self.index + 1).map(|t| &t.token),
                        Some(Token::RParen)
                    );
                    let params_follow = matches!(
                        self.tokens.get(self.index + 2).map(|t| &t.token),
                        Some(Token::LParen)
                    );
                    if empty && params_follow {
                        self.skip_paren_block()?; // `operator()` target
                    }
                } else {
                    self.skip_operator_target()?;
                }
                if self.peek_is(Token::LParen) {
                    self.skip_paren_block()?;
                }
                if self.peek_is(Token::Const) {
                    self.advance();
                } else if self.peek_is(Token::Identifier)
                    && self.current_lexeme().eq_ignore_ascii_case("const")
                {
                    self.advance();
                }
                if self.peek_is(Token::LBrace) {
                    self.skip_brace_block()?;
                } else {
                    self.expect_semi()?;
                }
                continue;
            }
            // Constructor / method-like declaration inside a struct: `Name(params) [: init] [{ body } | ;]`
            // This covers `Camera(float3 org, ...)` and `bool isHit() const { ... }` patterns
            // that PSSL/C++ headers use for helper structs. Peek ahead for Identifier '('.
            // SRT buffer macros (`REGULAR_BUFFER(T) name;`) also match `Ident(`
            // but declare resources — they are desugared below, not skipped here.
            if self.peek_is(Token::Identifier) && !is_resource_kind(&self.current_lexeme()) {
                let next_is_lparen = self
                    .tokens
                    .get(self.index + 1)
                    .map(|t| t.token == Token::LParen)
                    .unwrap_or(false);
                if next_is_lparen {
                    // Could be a method/constructor without a parsed return type yet;
                    // consume the name and its parameter list.
                    self.advance(); // name
                    self.skip_paren_block()?;
                    // Trailing `const` qualifier for methods: `isHit() const`
                    if self.peek_is(Token::Const) {
                        self.advance();
                    } else if self.peek_is(Token::Identifier)
                        && self.current_lexeme().eq_ignore_ascii_case("const")
                    {
                        self.advance();
                    }
                    // Initializer list: `: m_x(a), m_y{b}, ...`
                    if self.peek_is(Token::Colon) {
                        self.skip_ctor_init_list()?;
                    }
                    if self.peek_is(Token::LBrace) {
                        self.skip_brace_block()?;
                    } else {
                        self.expect_semi()?;
                    }
                    continue;
                }
            }
            // `typename` prefix (C++ dependent type) before `enable_if<...>::type`
            // (skipped for elaborated `struct Name` uses — the type is set).
            if elaborated_ty.is_none()
                && self.peek_is(Token::Identifier)
                && self.current_lexeme().eq_ignore_ascii_case("typename")
            {
                self.advance();
            }
            // SRT buffer macros `REGULAR_BUFFER(T) name;` — desugar to a
            // resource type (downstream lowering models `regularbuffer`).
            // `elaborated_ty` carries `struct Name` uses past this point.
            let ty = match elaborated_ty {
                Some(t) => t,
                None => match self.try_parse_macro_resource_type()? {
                    Some(t) => t,
                    None => self.parse_type()?,
                },
            };
            // Trailing `::type` / `::type` after template (e.g. `enable_if<...>::type`)
            while self.peek_is(Token::Colon)
                && self.tokens.get(self.index + 1).map(|t| t.token == Token::Colon).unwrap_or(false)
            {
                self.advance(); // ':'
                self.advance(); // ':'
                if self.peek_is(Token::Identifier) {
                    // consume `type` part of `::type` — keep as part of same decl
                    self.advance();
                } else {
                    break;
                }
            }
            // Pointer/reference declarators `*`/`&` like `PerFrame *perFrameData` or `float4 &color`
            // PSSL may also spell a pointer-to-const as `Type const* name` — consume
            // the trailing `const`/`constexpr` qualifier (dropped; no HLSL spelling here).
            // This also covers `static unsigned constexpr k = ...` where
            // `parse_type` consumes `unsigned` and leaves `constexpr`.
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
                break;
            }
            if self.peek_is(Token::Star) || self.peek_is(Token::Ampersand) {
                self.advance();
            }
            // Constructor-like: `Camera(float3 org, ...) : m_org(org) { body }`
            // where `ty` is actually the constructor name and the next token is '('.
            if self.peek_is(Token::LParen) {
                self.skip_paren_block()?;
                if self.peek_is(Token::Const) {
                    self.advance();
                } else if self.peek_is(Token::Identifier)
                    && self.current_lexeme().eq_ignore_ascii_case("const")
                {
                    self.advance();
                }
                // Initializer list: `: m_x(a), m_y{b}, ...`
                if self.peek_is(Token::Colon) {
                    self.skip_ctor_init_list()?;
                }
                if self.peek_is(Token::LBrace) {
                    self.skip_brace_block()?;
                } else {
                    self.expect_semi()?;
                }
                continue;
            }
            let mut field_name = self.expect_identifier()?;
            // In-struct operator overloads (`T &operator[](int i) {...}`,
            // `operator floatInVec() const;`): consume the operator target,
            // then parse the remainder as a method (body skipped).
            if field_name == "operator" {
                if self.peek_is(Token::LParen) {
                    // `operator()` target only when empty and followed by
                    // another `(` (the real params); otherwise these ARE
                    // the params of a degenerately-named method.
                    let empty = matches!(
                        self.tokens.get(self.index + 1).map(|t| &t.token),
                        Some(Token::RParen)
                    );
                    let params_follow = matches!(
                        self.tokens.get(self.index + 2).map(|t| &t.token),
                        Some(Token::LParen)
                    );
                    if empty && params_follow {
                        self.skip_paren_block()?; // `operator()` target
                    }
                } else {
                    self.skip_operator_target()?;
                }
            }
            // Method-like: `isHit() const { body }` or `getVertex(...) const`,
            // parsed as a scoped function (scope = this struct) so member
            // calls (`srt.method()`) resolve; emitted as an HLSL struct
            // method. `static`/`const` qualifiers were already consumed
            // above and are dropped (HLSL has no const methods).
            if self.peek_is(Token::LParen) {
                let saved = self.index;
                match self.parse_function(Vec::new(), ty.clone(), field_name.clone()) {
                    Ok(mut method) => {
                        method.scope = Some(name.clone());
                        self.pending_methods.push(method);
                    }
                    Err(_) => {
                        // Unparseable method shape: keep the old skip
                        // behavior so Parse coverage never regresses (the
                        // dropped method stays loud downstream instead).
                        self.index = saved;
                        self.skip_paren_block()?;
                        if self.peek_is(Token::Const) {
                            self.advance();
                        } else if self.peek_is(Token::Identifier)
                            && self.current_lexeme().eq_ignore_ascii_case("const")
                        {
                            self.advance();
                        }
                        if self.peek_is(Token::LBrace) {
                            self.skip_brace_block()?;
                        } else {
                            self.expect_semi()?;
                        }
                    }
                }
                continue;
            }
            // One or more comma-separated declarators sharing the type:
            // `uint32_t index0, index1, index2;`, `float3 u, v, w;`.
            loop {
                // Template-id spelling for this declarator's type
                // (shares the member window above).
                self.note_field_spelling(member_ty_start, self.index, &field_name.clone());
                let mut array_size = None;
                // Multi-dimensional arrays `name[64][2]` — record the first
                // dimension only (matches function-body handling), EXCEPT
                // 2D all-constant shapes which flatten to the row-major
                // product with dims recorded for use linearization
                // (`Partition3::mask3bit[3][2]` → `[6]`, uses
                // `f[i][j]` → `f[i*2+j]` in lowering; mirrors the
                // global/local `multidim` design, qualified to avoid
                // cross-struct collisions). 3D+ and non-constant dims keep
                // legacy first-dimension handling (stay loud if indexed
                // multi-dimensionally).
                let mut first_dim = true;
                let mut field_dims: Vec<u32> = Vec::new();
                let mut dims_ok = true;
                while self.peek_is(Token::LBracket) {
                    self.advance();
                    if !self.peek_is(Token::RBracket) {
                        let size_expr = self.parse_assignment()?;
                        if let Some(n) = self
                            .resolve_array_size_with(&size_expr, Some(&member_consts))
                        {
                            if first_dim {
                                array_size = Some(n);
                            }
                            field_dims.push(n);
                        } else {
                            dims_ok = false;
                        }
                    } else {
                        dims_ok = false;
                    }
                    self.expect(Token::RBracket)?;
                    first_dim = false;
                }
                if dims_ok && field_dims.len() == 2 {
                    let product = field_dims[0].saturating_mul(field_dims[1]);
                    array_size = Some(product);
                    self.field_multidim.insert(
                        format!("{name}::{field_name}"),
                        field_dims,
                    );
                }
                let mut semantic = self.parse_optional_semantic()?;
                // Framework semantic macros (`PSSL_SEMANTIC(...)`,
                // `SHADER_SEMANTIC(...)`) as field semantics, e.g.
                // `float3 positionInModel PSSL_SEMANTIC(POSITION_IN_MODEL);`
                // The inner name is the real semantic, so extract it
                // exactly (mirrors the parameter site below).
                if semantic.is_none()
                    && self.peek_is(Token::Identifier)
                    && Self::is_semantic_macro_name(&self.current_lexeme())
                {
                    self.advance(); // macro name
                    self.expect(Token::LParen)?;
                    if self.peek_is(Token::Identifier) {
                        let inner = self.expect_identifier()?;
                        semantic = Some(crate::ast::Semantic::Name(inner));
                    }
                    // Skip any extra macro args, then `)`.
                    while !self.peek_is(Token::RParen) && !self.eof() {
                        if self.peek_is(Token::LParen) {
                            self.skip_paren_block()?;
                        } else {
                            self.advance();
                        }
                    }
                    self.expect(Token::RParen)?;
                    if semantic.is_none() {
                        // Empty macro args: keep a marker so validation
                        // can still see a semantic was present.
                        semantic = Some(crate::ast::Semantic::Name(
                            "PSSL_SEMANTIC".to_string(),
                        ));
                    }
                }
                // Static const members may have an in-struct initializer (`= 0xffff`)
                // or a bare brace-init (`float3 k{...};`) — values discarded,
                // except const-foldable ones recorded for member array sizes
                // (`static const uint NumPlanes = 4;` then `float4 planes[NumPlanes]`).
                if self.peek_is(Token::Equal) {
                    self.advance();
                    let init_expr = self.parse_assignment()?;
                    if let Some(v) =
                        self.resolve_array_size_with(&init_expr, Some(&member_consts))
                    {
                        member_consts.insert(field_name.clone(), v.to_string());
                        // Qualified `Struct::member` for attribute evaluation
                        // (`[MAX_VERTEX_COUNT((S::kMax - 1) * 4)]`). Sound
                        // for valid PSSL: `S::m` qualification is only legal
                        // on static members, so a recorded literal is the
                        // static's value; instance defaults never legally
                        // appear under `S::`.
                        self.struct_static_consts.insert(
                            format!("{name}::{field_name}"),
                            v.to_string(),
                        );
                    }
                    // Raw initializer for `T::kX` constexpr evaluation
                    // (unfolded; may reference template value-params).
                    // Stored regardless of foldability; evaluation gates
                    // on strict literal/param/pure-op shapes downstream.
                    // Same soundness as above (valid `S::` queries are
                    // statics).
                    self.template_static_inits.insert(
                        format!("{name}::{field_name}"),
                        init_expr.clone(),
                    );
                } else if self.peek_is(Token::LBrace) {
                    self.skip_brace_init()?;
                }
                let span = self.previous_span().merge(struct_kw);
                fields.push(FieldDecl {
                    ty: ty.clone(),
                    name: field_name,
                    semantic,
                    array_size,
                    span,
                });
                if !self.peek_is(Token::Comma) {
                    break;
                }
                self.advance(); // ','
                // Per-declarator `*`/`&` (`int *a, b`).
                if self.peek_is(Token::Star) || self.peek_is(Token::Ampersand) {
                    self.advance();
                }
                field_name = self.expect_identifier()?;
            }
            self.expect_semi()?;
        }

        let end_span = self.current_span();
        self.expect(Token::RBrace)?;
        if self.peek_is(Token::Semi) {
            self.advance();
        }
        if field_frame {
            self.drain_field_spellings(&name);
        }
        // Scope value-param defaults by struct name for partial-arg
        // binding (`PackVertex::BoneCount` → `"0"`).
        for (p, d) in template_defaults {
            self.template_param_defaults
                .insert(format!("{name}::{p}"), d);
        }

        Ok(StructDecl {
            name,
            fields,
            bases,
            span: struct_kw.merge(end_span),
            template_params,
        })
    }

    /// Drains the innermost field-spelling frame into
    /// `template_site_spellings[struct_name]` (no-op when empty).
    fn drain_field_spellings(&mut self, struct_name: &str) {
        if let Some(frame) = self.pending_field_spellings.pop() {
            if !frame.is_empty() {
                self.template_site_spellings
                    .entry(struct_name.to_string())
                    .or_default()
                    .extend(frame);
            }
        }
    }

    pub(crate) fn parse_function(
        &mut self,
        attributes: Vec<Attribute>,
        return_type: Type,
        name: String,
    ) -> Result<Function> {
        // Fresh function scope: local-struct renames from any previous
        // function must not leak in (locals never escape their function).
        // Multi-dim entries likewise tag this function (globals persist
        // with a module-scope tag); restored on every return below.
        // A preceding `template<...>` prefix drains here (ordered params
        // for monomorphization); anything else leaves it empty.
        let template_params = std::mem::take(&mut self.pending_template_params);
        // Value-param defaults drain with scope (function name) for
        // partial-arg binding.
        for (p, d) in std::mem::take(&mut self.pending_template_defaults) {
            self.template_param_defaults
                .insert(format!("{name}::{p}"), d);
        }
        self.local_struct_renames.clear();
        self.function_seq += 1;
        let prev_function = self.current_function.replace(self.function_seq);
        let start_span = self.previous_span();
        // A bare `CXX_SYMBOL("...")` marker preceding this function carries
        // its stage hint; fold it in as a regular `CxxSymbol` attribute.
        let mut attributes = attributes;
        if let Some(symbol) = self.pending_cxx_symbol.take() {
            attributes.insert(
                0,
                Attribute {
                    name: "CxxSymbol".to_string(),
                    arguments: vec![crate::ast::Expr::String(symbol)],
                },
            );
        }
        self.expect(Token::LParen)?;
        let mut parameters = Vec::new();
        tracing::trace!(
            target: TRACE_ENTER,
            line = start_span.start,
            name = %name,
            "parse_function",
        );

        if !self.peek_is(Token::RParen) {
            // C-style `f(void)` — a lone `void` means no parameters.
            if self.peek_is(Token::Void)
                && matches!(
                    self.tokens.get(self.index + 1).map(|t| &t.token),
                    Some(Token::RParen)
                )
            {
                self.advance();
            } else {
                loop {
                    // C varargs (`...`) — consume the dots and stop; no
                    // parameter is recorded for them.
                    if self.peek_is(Token::Dot) {
                        while self.peek_is(Token::Dot) {
                            self.advance();
                        }
                        break;
                    }
                    parameters.push(self.parse_parameter()?);
                    if !self.peek_is(Token::Comma) {
                        break;
                    }
                    self.advance();
                }
            }
        }

        self.expect(Token::RParen)?;
        // Trailing qualifiers on methods (`bool C::get() const {...}`).
        self.skip_trailing_method_qualifiers()?;
        // C++ constructor initializer list `: m_x(a), m_y(b) ...` before
        // the body (also on out-of-class definitions like
        // `C::C(...) : mData(...) { }`). Distinguished from a return
        // semantic (`: SEMANTIC {`) by the `(` after the first identifier;
        // a bare `::` is never an init list.
        let is_ctor_init = self.peek_is(Token::Colon)
            && !self
                .tokens
                .get(self.index + 1)
                .map(|t| t.token == Token::Colon)
                .unwrap_or(false)
            && matches!(
                self.tokens.get(self.index + 1).map(|t| &t.token),
                Some(Token::Identifier)
            )
            && matches!(
                self.tokens.get(self.index + 2).map(|t| &t.token),
                Some(Token::LParen)
            );
        if is_ctor_init {
            self.skip_ctor_init_list()?;
        }
        let mut return_semantic = self.parse_optional_semantic()?;
        // Framework semantic macros (`SHADER_SEMANTIC(...)`, ...) as return
        // semantics: `Varying vs_main(...) SHADER_SEMANTIC(S_POSITION)`.
        // The inner name is the real semantic; a `: SEM` parsed above wins.
        if return_semantic.is_none()
            && self.peek_is(Token::Identifier)
            && self.peek_next_is(Token::LParen)
            && Self::is_semantic_macro_name(&self.current_lexeme())
        {
            self.advance(); // macro name
            self.expect(Token::LParen)?;
            if self.peek_is(Token::Identifier) {
                return_semantic = Some(crate::ast::Semantic::Name(self.expect_identifier()?));
            }
            while !self.peek_is(Token::RParen) && !self.eof() {
                if self.peek_is(Token::LParen) {
                    self.skip_paren_block()?;
                } else {
                    self.advance();
                }
            }
            self.expect(Token::RParen)?;
        }
        // Forward declaration `uint Foo(...);` has no body, just `;`.
        // Template value-params recorded for the prefix (e.g. `fmt` from
        // `template <BvhFormat fmt>` on a prototype) have no body to
        // inject synthetic locals into — drain them here so they never
        // leak into the NEXT parsed body (which would gain a bogus
        // `uint fmt = 0;` per undrained prototype).
        if self.peek_is(Token::Semi) {
            self.advance();
            self.current_function = prev_function;
            self.pending_template_values.clear();
            let end_span = self.previous_span();
            if !self.pending_param_spellings.is_empty() {
                self.template_spellings.insert(
                    name.clone(),
                    self.pending_param_spellings.drain(..).collect(),
                );
            }
            if !self.pending_site_spellings.is_empty() {
                self.template_site_spellings.insert(
                    name.clone(),
                    self.pending_site_spellings.drain(..).collect(),
                );
            }
            return Ok(Function {
                attributes,
                return_type,
                name,
                scope: None,
                parameters,
                return_semantic,
                body: Vec::new(),
                span: start_span.merge(end_span),
                template_params: template_params.clone(),
            });
        }
        let mut body = self.parse_block()?;
        // Inject template non-type params (e.g. `flags` from `template<uint64_t flags>`) as synthetic locals
        if !self.pending_template_values.is_empty() {
            let mut synthetic = Vec::new();
            for pname in self.pending_template_values.drain(..) {
                synthetic.push(crate::ast::Stmt::VarDecl {
                    storage: None,
                    ty: crate::ast::Type::UInt,
                    name: pname,
                    array_size: None,
                    init: Some(crate::ast::Expr::Number("0".to_string())),
                });
            }
            // prepend so they are visible throughout body
            synthetic.extend(body);
            body = synthetic;
        }
        // PSSL often has a trailing `;` after a function definition (`};`)
        if self.peek_is(Token::Semi) {
            self.advance();
        }
        let end_span = self.previous_span();
        self.current_function = prev_function;
        if !self.pending_param_spellings.is_empty() {
            self.template_spellings
                .insert(name.clone(), self.pending_param_spellings.drain(..).collect());
        }
        if !self.pending_site_spellings.is_empty() {
            self.template_site_spellings.insert(
                name.clone(),
                self.pending_site_spellings.drain(..).collect(),
            );
        }

        Ok(Function {
            attributes,
            return_type,
            name,
            scope: None,
            parameters,
            return_semantic,
            body,
            span: start_span.merge(end_span),
            template_params,
        })
    }

    /// Pure lookahead: is the parameter at the cursor (on `(`) a function
    /// pointer `(*name)(args)`? Requires a `*` inside or an argument list
    /// after the close paren (a bare `(x)` is not one).
    pub(crate) fn looks_like_fnptr_param(&self) -> bool {
        let mut k = self.index + 1;
        let mut saw_star = false;
        while matches!(
            self.tokens.get(k).map(|t| &t.token),
            Some(Token::Star | Token::Ampersand)
        ) {
            saw_star = true;
            k += 1;
        }
        if matches!(
            self.tokens.get(k).map(|t| &t.token),
            Some(Token::Identifier)
        ) {
            k += 1;
        }
        // Find the matching close paren from the cursor.
        let mut depth = 0usize;
        let mut j = self.index;
        let mut closed = None;
        while j < self.tokens.len() && j - self.index < 64 {
            if self.tokens[j].token == Token::LParen {
                depth += 1;
            } else if self.tokens[j].token == Token::RParen {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    closed = Some(j);
                    break;
                }
            }
            j += 1;
        }
        let Some(c) = closed else {
            return false;
        };
        saw_star
            || matches!(
                self.tokens.get(c + 1).map(|t| &t.token),
                Some(Token::LParen)
            )
    }

    /// Records a template-id spelling for a parameter when its type
    /// region holds a top-level angle list (`BcnEncodeSrt<uint2> g_srt`
    /// between `ty_start` and `end`): stores `Name<args>` normalized
    /// (single spaces only between word tokens) under the variable name
    /// in the pending drain. No-op without angles.
    pub(crate) fn note_param_spelling(&mut self, ty_start: usize, end: usize, varname: &str) {
        if let Some(spelling) = Self::template_spelling_in(&self.tokens, ty_start, end) {
            self.pending_param_spellings
                .push((varname.to_string(), spelling));
        }
    }

    /// Records a template-id spelling for a local variable or struct
    /// field (`Indices<3> x;`): same capture as parameters, but drained
    /// per function/struct into `template_site_spellings` (kept separate
    /// so existing param consumers are untouched).
    pub(crate) fn note_site_spelling(&mut self, ty_start: usize, end: usize, varname: &str) {
        if let Some(spelling) = Self::template_spelling_in(&self.tokens, ty_start, end) {
            self.pending_site_spellings
                .push((varname.to_string(), spelling));
        }
    }

    /// Records a template-id spelling for a struct field into the
    /// current (innermost) field frame; drained per struct.
    pub(crate) fn note_field_spelling(&mut self, ty_start: usize, end: usize, varname: &str) {
        if let Some(spelling) = Self::template_spelling_in(&self.tokens, ty_start, end) {
            if let Some(frame) = self.pending_field_spellings.last_mut() {
                frame.push((varname.to_string(), spelling));
            }
        }
    }

    /// Normalized `Name<args>` spelling for the type tokens in
    /// `[ty_start, end)`, or `None` when no top-level angle list opens
    /// there. (Shared core of the note_*_spelling recorders.)
    fn template_spelling_in(
        tokens: &[crate::token::SpannedToken],
        ty_start: usize,
        end: usize,
    ) -> Option<String> {
        // Find `<` at zero depth (tracking parens/brackets/angles, so
        // macro types like `REGULAR_BUFFER(T)` don't false-positive).
        let mut angle = 0usize;
        let mut paren = 0usize;
        let mut brack = 0usize;
        let mut lt_at: Option<usize> = None;
        let mut i = ty_start;
        while i < end && i < tokens.len() {
            let tok = &tokens[i].token;
            match tok {
                Token::Less => {
                    if angle == 0 && paren == 0 && brack == 0 && lt_at.is_none() {
                        lt_at = Some(i);
                    }
                    angle += 1;
                }
                Token::Greater => angle = angle.saturating_sub(1),
                Token::ShiftRight => angle = angle.saturating_sub(2),
                Token::LParen => paren += 1,
                Token::RParen => paren = paren.saturating_sub(1),
                Token::LBracket => brack += 1,
                Token::RBracket => brack = brack.saturating_sub(1),
                _ => {}
            }
            i += 1;
        }
        let Some(lt) = lt_at else {
            return None;
        };
        // Walk back over `Name` / `::` segments for the base start.
        let mut base = lt;
        let mut k = lt;
        while k > ty_start {
            let prev = &tokens[k - 1];
            if prev.token == Token::Identifier {
                base = k - 1;
                k -= 1;
            } else if prev.token == Token::Colon
                && k - ty_start >= 2
                && tokens[k - 2].token == Token::Colon
            {
                base = k - 2;
                k -= 2;
            } else {
                break;
            }
        }
        if base >= lt {
            return None;
        }
        // Match the closing `>` from `lt`.
        let mut depth = 0usize;
        let mut gt_end: Option<usize> = None;
        let mut j = lt;
        while j < end && j < tokens.len() {
            match &tokens[j].token {
                Token::Less => depth += 1,
                Token::Greater => {
                    depth = depth.saturating_sub(1);
                    if depth == 0 {
                        gt_end = Some(j + 1);
                        break;
                    }
                }
                Token::ShiftRight => {
                    if depth >= 2 {
                        depth -= 2;
                    } else {
                        depth = 0;
                    }
                    if depth == 0 {
                        gt_end = Some(j + 1);
                        break;
                    }
                }
                _ => {}
            }
            j += 1;
        }
        let Some(gt_end) = gt_end else {
            return None;
        };
        // Normalize: single space only between two word tokens.
        fn is_word(tok: &Token) -> bool {
            match tok {
                Token::Identifier
                | Token::Number
                | Token::Int
                | Token::UInt
                | Token::Float
                | Token::Half
                | Token::Double
                | Token::Bool
                | Token::Void
                | Token::Const => true,
                _ => false,
            }
        }
        let mut out = String::new();
        let mut prev_word = false;
        for t in &tokens[base..gt_end] {
            let word = is_word(&t.token);
            if word && prev_word {
                out.push(' ');
            }
            out.push_str(&t.lexeme);
            prev_word = word;
        }
        Some(out)
    }

    pub(crate) fn parse_parameter(&mut self) -> Result<Parameter> {        let start = self.current().map(|t| t.span.start).unwrap_or(0);
        // Token index at type start, for template-id spelling capture
        // (`BcnEncodeSrt<uint2> g_srt` preserves its args for
        // monomorphization; types erase them).
        let ty_start = self.index;

        // PSSL/C++ const-qualified parameters like `const float2 plane_eq` or
        // `const float z` (used in dof helpers). Consume leading const/constexpr.
        // An explicit `uniform` qualifier (`uniform float4x4 wvp` on entries)
        // marks host-provided constants (lowered to cbuffer members).
        let mut is_uniform = false;
        loop {
            if self.peek_is(Token::Const) {
                self.advance();
                continue;
            }
            if self.peek_is(Token::Identifier) {
                let lex = self.current_lexeme();
                if lex.eq_ignore_ascii_case("constexpr") || lex.eq_ignore_ascii_case("const") {
                    self.advance();
                    continue;
                }
                if lex.eq_ignore_ascii_case("uniform") {
                    is_uniform = true;
                    self.advance();
                    continue;
                }
            }
            break;
        }

        let direction = if self.peek_is(Token::In) {
            self.advance();
            Some(Direction::In)
        } else if self.peek_is(Token::Out) {
            self.advance();
            Some(Direction::Out)
        } else if self.peek_is(Token::Inout) {
            self.advance();
            Some(Direction::InOut)
        } else {
            None
        };

        // Const may also appear after direction (`in const float`).
        loop {
            if self.peek_is(Token::Const) {
                self.advance();
                continue;
            }
            if self.peek_is(Token::Identifier) {
                let lex = self.current_lexeme();
                if lex.eq_ignore_ascii_case("constexpr") {
                    self.advance();
                    continue;
                }
            }
            break;
        }

        // HLSL interpolation modifiers on parameters
        // (`linear`, `centroid`, `sample`, `nointerpolation`, ...).
        self.skip_interp_modifiers();

        // Geometry-shader primitive topology (`point`, `line`,
        // `triangle`, `lineadj`, `triangleadj`) preceding an input
        // parameter. Runs after direction parsing so both `Point T x`
        // and `in Point T x` work. Only a primitive when two more
        // identifiers follow (`Type name`); a lone `point x` keeps
        // `point` as the type (user struct, no regression).
        // The spelling is captured (lowercase) for HLSL emission.
        let mut gs_primitive: Option<String> = None;
        if self.peek_is(Token::Identifier)
            && matches!(
                self.current_lexeme().to_ascii_lowercase().as_str(),
                "point" | "line" | "triangle" | "lineadj" | "triangleadj"
            )
            && matches!(
                self.tokens.get(self.index + 1).map(|t| &t.token),
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
            )
            && matches!(
                self.tokens.get(self.index + 2).map(|t| &t.token),
                Some(Token::Identifier)
            )
        {
            gs_primitive = Some(self.current_lexeme().to_ascii_lowercase());
            self.advance();
        }

        // SRT buffer macros `REGULAR_BUFFER(T) name` in parameter position.
        let mut ty = match self.try_parse_macro_resource_type()? {
            Some(t) => t,
            None => {
                if self.peek_is(Token::Struct) {
                    // Inline definition (`f(struct UserData { float4 x; } p)`,
                    // clear shaders): parse for real and register through
                    // the pending drain so the type resolves downstream.
                    // Anything else (`struct Name`, `struct Name<...>`
                    // references, anonymous bodies) keeps today's path.
                    let is_def = matches!(
                        self.tokens.get(self.index + 1).map(|t| &t.token),
                        Some(Token::Identifier)
                    ) && matches!(
                        self.tokens.get(self.index + 2).map(|t| &t.token),
                        Some(Token::LBrace)
                    );
                    if is_def {
                        let decl = self.parse_struct()?;
                        let struct_name = decl.name.clone();
                        self.pending_structs.push(decl);
                        Type::Named(struct_name)
                    } else {
                        self.advance(); // `struct`
                        let struct_name = if self.peek_is(Token::Identifier) {
                            let n = self.current_lexeme();
                            self.advance();
                            if self.peek_is(Token::LBrace) {
                                self.skip_brace_block()?;
                            }
                            n
                        } else if self.peek_is(Token::LBrace) {
                            // Anonymous inline struct with a body
                            // (`struct { uint mip; uint slice; } srt`):
                            // register it under a synthetic name so the
                            // parameter type and its member accesses
                            // (`srt.mip`) resolve, instead of the opaque
                            // `anonymous_struct` placeholder (DXC "unknown
                            // type name"). Re-splice `struct <synth>` at
                            // the `{` and parse via `parse_struct`.
                            let span = self.current_span();
                            let synth = format!("__anon_srt{}", self.anon_counter);
                            self.anon_counter += 1;
                            self.tokens.insert(
                                self.index,
                                crate::token::SpannedToken {
                                    token: Token::Struct,
                                    lexeme: "struct".to_string(),
                                    span,
                                },
                            );
                            self.tokens.insert(
                                self.index + 1,
                                crate::token::SpannedToken {
                                    token: Token::Identifier,
                                    lexeme: synth.clone(),
                                    span,
                                },
                            );
                            let decl = self.parse_struct()?;
                            self.pending_structs.push(decl);
                            synth
                        } else {
                            self.expect_identifier()?
                        };
                        Type::Named(struct_name)
                    }
                } else {
                    self.parse_type()?
                }
            }
        };
        // Trailing `const`/`constexpr` qualifiers and pointer/reference
        // declarators, in any order and multiplicity (`AnyHitSrt const *srt`,
        // `T * __restrict p`, `T * const *p`) — all dropped.
        loop {
            if self.peek_is(Token::Const) {
                self.advance();
                continue;
            }
            if self.peek_is(Token::Identifier) {
                let l = self.current_lexeme().to_ascii_lowercase();
                // `restrict` is lowercase-only (C99 keyword); a capitalized
                // `Restrict` could be a legitimate name.
                if l == "const"
                    || l == "constexpr"
                    || self.current_lexeme() == "restrict"
                {
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
        // Calling-convention macros between type and name
        // (`const MaterialGeom SCE_GEOMETRY_ARG target`).
        self.skip_interposed_macro();
        // Direction qualifier after the type (`T out x`, PSR convention —
        // HLSL normally spells `out T x`). Only a direction when followed
        // by the parameter name.
        let mut direction = if (self.peek_is(Token::In)
            || self.peek_is(Token::Out)
            || self.peek_is(Token::Inout))
            && matches!(
                self.tokens.get(self.index + 1).map(|t| &t.token),
                Some(Token::Identifier)
            ) {
            let d = if self.peek_is(Token::In) {
                Direction::In
            } else if self.peek_is(Token::Out) {
                Direction::Out
            } else {
                Direction::InOut
            };
            self.advance();
            Some(d)
        } else {
            direction
        };
        // Framework argument macros `PSSL_ARG_OUT(type, name)` (PSR entries):
        // `ty` parsed as a bare macro name followed by `(` — reinterpret
        // as `MACRO(type, name)` and parse the inside.
        let mut arg_macro_name: Option<String> = None;
        if self.peek_is(Token::LParen)
            && matches!(&ty, Type::Named(m) if Self::is_macro_like_name(m) && !self.is_known_type_name(m))
        {
            let macro_name = match &ty {
                Type::Named(m) => m.clone(),
                _ => unreachable!(),
            };
            self.expect(Token::LParen)?;
            ty = self.parse_type()?;
            // Trailing const/* after the inner type.
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
            // Direction from the macro suffix when none parsed
            // (`PSSL_ARG_OUT` → out, `PSSL_ARG_IN` → in).
            if direction.is_none() {
                let upper = macro_name.to_ascii_uppercase();
                if upper.ends_with("_INOUT") {
                    direction = Some(Direction::InOut);
                } else if upper.ends_with("_IN") {
                    direction = Some(Direction::In);
                } else if upper.ends_with("_OUT") {
                    direction = Some(Direction::Out);
                }
            }
            self.expect(Token::Comma)?;
            arg_macro_name = Some(self.expect_identifier()?);
            // Skip any further macro args, then `)`.
            while !self.peek_is(Token::RParen) && !self.eof() {
                if self.peek_is(Token::LParen) {
                    self.skip_paren_block()?;
                } else if self.peek_is(Token::Comma) {
                    // Extra top-level args are not modeled; drop the rest.
                    while !self.peek_is(Token::RParen) && !self.eof() {
                        if self.peek_is(Token::LParen) {
                            self.skip_paren_block()?;
                        } else {
                            self.advance();
                        }
                    }
                    break;
                } else {
                    self.advance();
                }
            }
            self.expect(Token::RParen)?;
        }
        let name = if let Some(mn) = arg_macro_name {
            mn
        } else if self.peek_is(Token::LParen) && self.looks_like_fnptr_param() {
            // Function-pointer parameter (`void *(*entry)(void *)`, signal
            // handlers): `(*name)` then the argument list. Model as an
            // `auto` callable; the return/argument types are irrelevant.
            self.advance(); // '('
            while self.peek_is(Token::Star) || self.peek_is(Token::Ampersand) {
                self.advance();
            }
            let name = if self.peek_is(Token::Identifier) {
                let n = self.current_lexeme();
                self.advance();
                n
            } else {
                format!("__unused_param_{}", self.index)
            };
            // Skip to the matching `)`.
            let mut depth = 1usize;
            while depth > 0 && !self.eof() {
                if self.peek_is(Token::LParen) {
                    depth += 1;
                } else if self.peek_is(Token::RParen) {
                    depth -= 1;
                }
                self.advance();
                if depth == 0 {
                    break;
                }
            }
            // Skip the trailing argument list `(...)`, if present.
            if self.peek_is(Token::LParen) {
                self.skip_paren_block()?;
            }
            return Ok(Parameter {
                ty: Type::Named("auto".to_string()),
                name,
                direction,
                semantic: None,
                array_size: None,
                gs_primitive: None,
                default: None,
                is_uniform,
                span: Span::new(start, self.previous_span().end),
            });
        } else if self.peek_is(Token::Identifier) {
            self.expect_identifier()?
        } else if matches!(
            self.peek(),
            Some(
                Token::In | Token::Out | Token::Inout | Token::Const | Token::Static
            )
        ) {
            // Reserved words used as parameter names (`PrimaryPayload
            // inout` in anyhit shaders — the name is irrelevant).
            let name = self.current_lexeme();
            self.advance();
            name
        } else {
            // Unnamed parameter (`void f(string, T0)`, `void g(int)`) —
            // legal C/C++; synthesize a placeholder (never referenced).
            format!("__unused_param_{}", self.index)
        };
        // Array sizes `name[4]`, multi-dimensional `name[3][2]`
        // (sizes may be constants like `4` or `k_lds_size`). Record the
        // first dimension (matches global/field array handling), but
        // collect every dimension so exactly-2D all-constant shapes can
        // flatten to product size (uses linearize inline).
        let mut array_size = None;
        let mut first_dim = true;
        let mut all_dims: Vec<Option<u32>> = Vec::new();
        while self.peek_is(Token::LBracket) {
            self.advance();
            if !self.peek_is(Token::RBracket) {
                let size_expr = self.parse_assignment()?;
                let resolved = self.resolve_array_size(&size_expr);
                if first_dim {
                    array_size = Some(resolved.unwrap_or(0));
                }
                all_dims.push(resolved);
            } else {
                if first_dim {
                    array_size = Some(0);
                }
                all_dims.push(None);
            }
            self.expect(Token::RBracket)?;
            first_dim = false;
        }
        if first_dim == false && array_size.is_none() {
            array_size = Some(0);
        }
        // Exactly-2D all-constant parameter shapes flatten to product
        // size (uses linearize inline); anything else keeps legacy.
        if let Some(product) = self.note_multidim(&name, &all_dims, false) {
            array_size = Some(product);
        }
        let mut semantic = self.parse_optional_semantic()?;
        // Framework semantic macros (`PSSL_SEMANTIC(...)`,
        // `SHADER_SEMANTIC(...)`) as parameter semantics, e.g.
        // `Shared::SRT srt PSSL_SEMANTIC(S_SRT_DATA)`. The inner name is
        // the real semantic (`: S_SRT_DATA`), so extract it exactly.
        if semantic.is_none()
            && self.peek_is(Token::Identifier)
            && Self::is_semantic_macro_name(&self.current_lexeme())
        {
            self.advance(); // macro name
            self.expect(Token::LParen)?;
            if self.peek_is(Token::Identifier) {
                let inner = self.expect_identifier()?;
                semantic = Some(crate::ast::Semantic::Name(inner));
            }
            // Skip any extra macro args, then `)`.
            while !self.peek_is(Token::RParen) && !self.eof() {
                if self.peek_is(Token::LParen) {
                    self.skip_paren_block()?;
                } else {
                    self.advance();
                }
            }
            self.expect(Token::RParen)?;
        }
        // C++/HLSL default argument, e.g. `uint ootf_adj = 0` or
        // `float gamma = 2.2` — kept for call-site filling during
        // lowering (HLSL has no default args).
        let default = if self.peek_is(Token::Equal) {
            self.advance();
            Some(self.parse_assignment()?)
        } else {
            None
        };
        // Template-id spelling capture (`BcnEncodeSrt<uint2> g_srt`):
        // angle lists erase during type parsing, so preserve the args
        // for monomorphization (pending drain in parse_function).
        self.note_param_spelling(ty_start, self.index, &name);
        let end = self.previous_span().end;

        Ok(Parameter {
            ty,
            name,
            direction,
            semantic,
            array_size,
            gs_primitive,
            default,
            is_uniform,
            span: Span::new(start, end),
        })
    }
}

/// Sanitizes a template-id spelling to an HLSL-safe struct name
/// (`Indices<2>` -> `Indices_2`): non-alphanumeric runs become a
/// single underscore, trimmed. Deterministic both sides (definition
/// and use matching share it).
pub(crate) fn sanitize_template_name(spelling: &str) -> String {
    let mut out = String::new();
    let mut sep = false;
    for c in spelling.chars() {
        if c.is_ascii_alphanumeric() || c == '_' {
            out.push(c);
            sep = false;
        } else if !sep {
            out.push('_');
            sep = true;
        }
    }
    while out.ends_with('_') {
        out.pop();
    }
    out
}
