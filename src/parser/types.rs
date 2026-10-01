//! Type parsing and type-name helpers.

use super::{Parser, TRACE_ENTER, TRACE_TYPES};
use crate::ast::{ScalarType, Semantic, Type};
use crate::error::Result;
use crate::token::Token;

// ------------------------------------------------------------------
// `Parser` methods
// ------------------------------------------------------------------

impl Parser {
    /// Typedef lookup with flattened-namespace fallback: `Ns::...::Alias`
    /// resolves through the bare `Alias` when the qualified spelling is
    /// not itself a known alias. Namespaces flatten at parse (enum
    /// `TraversalFlags` registers bare), but uses keep their scopes
    /// (`sce::Psr::TraversalFlags const flags`), so without the fallback
    /// the qualified spelling leaks to semantic as an unknown type and
    /// the declaration is dropped. The full spelling wins when known,
    /// so real qualified aliases are never hijacked. A bare name that is
    /// a REAL struct (declared struct, not an enum) vetoes the fallback:
    /// a colliding `typedef uint32_t Hit` elsewhere must never hijack
    /// `sce::Psr::Hit` (the struct); the qualified spelling then flows
    /// to scope resolution, which folds it to the struct.
    pub(crate) fn resolve_typedef(&self, name: &str) -> Option<Type> {
        if let Some(ty) = self.typedefs.get(name) {
            return Some(ty.clone());
        }
        if name.contains("::") {
            if let Some(bare) = name.rsplit("::").next() {
                if bare != name {
                    let is_real_struct =
                        self.struct_names.contains(bare) && !self.enum_names.contains(bare);
                    if !is_real_struct {
                        if let Some(ty) = self.typedefs.get(bare) {
                            return Some(ty.clone());
                        }
                    }
                }
            }
        }
        None
    }

    pub(crate) fn parse_type(&mut self) -> Result<Type> {
        let token = self.current().cloned().ok_or_else(|| self.eof_error("type"))?;

        let mut text = self.current_lexeme();
        tracing::trace!(
            target: TRACE_ENTER,
            line = token.span.start,
            "parse_type",
        );

        // Handle scoped names `sce::Psr::Type` — collect `::`-separated identifiers.
        // The lexer produces `:` `:` for `::`, so we look ahead for two colons.
        if token.token == Token::Identifier {
            let mut scoped = text.clone();
            let mut look = self.index + 1;
            while look + 1 < self.tokens.len()
                && self.tokens[look].token == Token::Colon
                && self.tokens[look + 1].token == Token::Colon
            {
                if let Some(id_tok) = self.tokens.get(look + 2) {
                    if id_tok.token == Token::Identifier {
                        scoped.push_str("::");
                        scoped.push_str(&id_tok.lexeme);
                        look += 3;
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            }
            if scoped != text {
                // Advance past the consumed `::Ident` sequences; the final
                // `advance()` below is for the initial token, so set index
                // to the first token after the scoped name minus one.
                self.index = look - 1;
                text = scoped;
            }
        }

        // Handle bare `matrix` (PSSL generic, no dimensions) as `float4x4`.
        // `Matrix4Unaligned` is `typedef column_major matrix` → `float4x4`.
        if text.eq_ignore_ascii_case("matrix") {
            self.advance();
            return Ok(Type::Matrix {
                base: ScalarType::Float,
                rows: 4,
                columns: 4,
            });
        }

        // Elaborated type references `struct Name` (C forward declarations
        // like `typedef struct kevent SceKernelKevent;`).
        if token.token == Token::Struct {
            self.advance(); // 'struct'
            if self.peek_is(Token::Identifier) {
                let name = self.current_lexeme();
                self.advance();
                return Ok(Type::Named(name));
            }
            return Err(self.error_expected("type"));
        }

        // Elaborated `union Name` / `enum Name` / `class Name` references —
        // exact lowercase keywords only (a type named `Union` is itself).
        if token.token == Token::Identifier
            && (text == "union" || text == "enum" || text == "class")
            && matches!(
                self.tokens.get(self.index + 1).map(|t| &t.token),
                Some(Token::Identifier)
            )
        {
            self.advance(); // keyword
            let name = self.current_lexeme();
            self.advance();
            return Ok(Type::Named(name));
        }

        // Handle C `unsigned`/`signed`/`long` type specifiers (e.g. `unsigned long`
        // for host-struct time fields like `m_startTime`). PSSL
        // uses these in host-struct definitions that are included verbatim.
        if text.eq_ignore_ascii_case("unsigned") || text.eq_ignore_ascii_case("signed") {
            let is_unsigned = text.eq_ignore_ascii_case("unsigned");
            let next_lex = self
                .tokens
                .get(self.index + 1)
                .map(|t| t.lexeme.clone())
                .unwrap_or_default();
            let nl = next_lex.to_ascii_lowercase();
            if matches!(nl.as_str(), "long" | "int" | "short" | "char") {
                self.advance(); // unsigned/signed
                self.advance(); // long/int/short/char
                // `long long`
                if nl == "long" && self.current_lexeme().eq_ignore_ascii_case("long") {
                    self.advance();
                }
                return Ok(if is_unsigned { Type::UInt } else { Type::Int });
            }
            // bare `unsigned` / `signed`
            self.advance();
            return Ok(if is_unsigned { Type::UInt } else { Type::Int });
        }
        if text.eq_ignore_ascii_case("long") {
            self.advance();
            if self.current_lexeme().eq_ignore_ascii_case("long") {
                self.advance();
            }
            // `long` is 32-bit on the target platform, map to Int
            return Ok(Type::Int);
        }

        let is_resource_template = matches!(&token.token, Token::Identifier)
            && is_resource_kind(&text)
            && self.peek_next_is(Token::Less);
        // Geometry-shader output streams (`TriangleBuffer<T>`,
        // `TriangleStream<T>`): HLSL `TriangleStream<T>`. Narrow gate
        // (template form only — a bare `TriangleBuffer x` keeps legacy
        // behavior); other spellings (`LineBuffer`, …) are unobserved
        // and stay loud.
        let is_stream_template = matches!(&token.token, Token::Identifier)
            && matches!(
                text.to_ascii_lowercase().as_str(),
                "trianglebuffer" | "trianglestream" | "pointstream" | "linestream"
            )
            && self.peek_next_is(Token::Less);
        let is_generic_template = matches!(&token.token, Token::Identifier)
            && !is_resource_template
            && self.peek_next_is(Token::Less);
        // Tessellation patch types (`InputPatch<T, N>` /
        // `OutputPatch<T, N>`): HLSL-native generics whose arguments
        // must survive (DXC: `use of class template requires template
        // arguments`). Baked into the name; monomorphization ignores
        // them (no struct definition to instantiate).
        // Tessellation patch types (`InputPatch<T, N>` /
        // `OutputPatch<T, N>`) are HLSL-native generics: bake the element
        // and count into the name (DXC rejects bare uses). Anything else
        // (nested templates, complex counts) falls through to the legacy
        // erase path below, so coverage never regresses.
        let is_patch_template = matches!(&token.token, Token::Identifier)
            && matches!(
                text.to_ascii_lowercase().as_str(),
                "inputpatch" | "outputpatch"
            )
            && self.peek_next_is(Token::Less);
        if is_patch_template {
            if let Some(ty) = self.try_parse_patch_type(&text)? {
                return Ok(ty);
            }
        }

        let ty = match &token.token {
            Token::Void => Type::Void,
            Token::Bool => Type::Bool,
            Token::Int => Type::Int,
            Token::UInt => Type::UInt,
            Token::Float => Type::Float,
            Token::Half => Type::Half,
            Token::Double => Type::Double,
            Token::Identifier => {
                if is_resource_template {
                    self.advance(); // resource kind name
                    let element = self.parse_resource_template_element()?;
                    let kind = text.to_ascii_lowercase().replace('_', "");
                    let access = resolve_resource_access(&text);
                    Type::Resource {
                        kind,
                        access,
                        element: Box::new(element),
                    }
                } else if is_stream_template {
                    self.advance(); // `TriangleBuffer`
                    let element = self.parse_resource_template_element()?;
                    Type::Resource {
                        kind: "trianglestream".to_string(),
                        access: crate::ast::ResourceAccess::ReadWrite,
                        element: Box::new(element),
                    }
                } else if self.peek_next_is(Token::Less) {
                    // C++ `enable_if<Cond, T>` (framework conditional members like
                    // `typename enable_if<kHasTangent, float4>::type
                    // m_tangent;`): model as `T`. The member exists in the
                    // preprocessed source, and `if constexpr` guards are
                    // checked as ordinary branches, so `T` is the only
                    // reading under which member uses typecheck.
                    if text.eq_ignore_ascii_case("enable_if")
                        && let Some(ty) = self.try_parse_enable_if_type()?
                    {
                        return Ok(ty);
                    }
                    // Generic template like `V2P<PackVertex<false, ...>>` or
                    // `V2P<PackVertex>` — skip `<...>` and treat as base type.
                    // Must handle `(kUvCount > 0)` inside `enable_if<(kUvCount > 0), float2>`
                    // where `>` inside parentheses is not a template closer.
                    self.advance(); // base name
                    self.advance(); // '<'
                    let args_start = self.index;
                    let mut depth = 1usize;
                    let mut paren_depth = 0usize;
                    while depth > 0 && !self.eof() {
                        if self.peek_is(Token::LParen) {
                            paren_depth += 1;
                        } else if self.peek_is(Token::RParen) {
                            paren_depth = paren_depth.saturating_sub(1);
                        } else if paren_depth == 0 {
                            if self.peek_is(Token::Less) {
                                depth += 1;
                            } else if self.peek_is(Token::ShiftRight) {
                                if self.speculation_depth > 0 {
                                    // Speculative parse (rolls back the
                                    // index only): never mutate tokens.
                                    // Leave the whole `>>` for the caller.
                                    break;
                                }
                                // Split `>>` into `> >` and consume one
                                // closer now; the other remains for the
                                // next iteration (or the caller). Needed
                                // when the `>>` spans this list and an
                                // enclosing one (`Outer<Inner<X>>`).
                                let idx = self.index;
                                let span = self.tokens[idx].span;
                                self.tokens[idx].token = Token::Greater;
                                self.tokens[idx].lexeme = ">".to_string();
                                self.tokens.insert(
                                    idx + 1,
                                    crate::token::SpannedToken {
                                        token: Token::Greater,
                                        lexeme: ">".to_string(),
                                        span,
                                    },
                                );
                                depth -= 1;
                            } else if self.peek_is(Token::Greater) {
                                depth -= 1;
                            }
                        }
                        self.advance();
                        if depth == 0 {
                            break;
                        }
                    }
                    // Record the template-id for monomorphization
                    // (`BcnEncodeSrt<uint2>`): the base erases below, so
                    // preserve (base, args) with whitespace stripped.
                    // Skipped under speculation (index-only rollback would
                    // keep polluting observations) and on unbalanced
                    // skip (deep `>>` surgery edge).
                    if depth == 0 && self.speculation_depth == 0 && args_start + 1 < self.index {
                        let args: String = self.tokens[args_start..self.index - 1]
                            .iter()
                            .map(|t| t.lexeme.as_str())
                            .collect::<Vec<_>>()
                            .join("")
                            .split_whitespace()
                            .collect::<Vec<_>>()
                            .join("");
                        // Skip resource-ish bases (their kinds carry the
                        // element separately) — struct template-ids only.
                        // Scoped framework names record too (filtered at
                        // use by the sce::/CullData gates).
                        self.observed_template_ids
                            .insert((text.clone(), args));
                    }
                    let ty = parse_named_type(&text);
                    match &ty {
                        Type::Named(name) => {
                            if let Some(resolved) = self.resolve_typedef(name) {
                                resolved.clone()
                            } else {
                                ty
                            }
                        }
                        _ => ty,
                    }
                } else if is_bare_resource_kind(&text) {
                    // Bare byte-address buffer with no `<element>` (e.g.
                    // `ByteBuffer buf;`, `RW_ByteBuffer bb;`). PSSL allows
                    // element-less byte buffers; model as a Resource so
                    // lowering/hoisting treat it as a resource (the emitter
                    // drops the element — `ByteAddressBuffer` is untyped).
                    // The element is `UInt` (a byte-address `Load` yields a
                    // dword), never `T`: a `T` placeholder both misfires the
                    // emitter's generic-`T` overload stamping
                    // (`is_generic_t_function`, emitting 8 identical
                    // definitions of every function taking a `ByteBuffer` —
                    // DXC `redefinition of 'findBatch'`) and poisons
                    // semantic operator checking once corrected (`Void`
                    // breaks `buf.Load(i) >> k`, `T` only passes by opaque
                    // passthrough).
                    let kind = text.to_ascii_lowercase().replace('_', "");
                    let access = resolve_resource_access(&text);
                    Type::Resource {
                        kind,
                        access,
                        element: Box::new(Type::UInt),
                    }
                } else if is_bare_texture_kind(&text) {
                    // Bare texture with no `<element>` (PSSL underscore
                    // spellings like `TextureCube_Array probes`). Sampled
                    // textures default to float4 (HLSL's own default, and
                    // the use sites consume float4, e.g.
                    // `probes.SampleLevel(...)` returned as float4).
                    // Underscore-free bare forms (`Texture2D`) keep the
                    // legacy Named path below, which already validates.
                    let kind = text.to_ascii_lowercase().replace('_', "");
                    let access = resolve_resource_access(&text);
                    Type::Resource {
                        kind,
                        access,
                        element: Box::new(Type::Vector {
                            base: crate::ast::ScalarType::Float,
                            width: 4,
                        }),
                    }
                } else {
                    let ty = parse_named_type(&text);
                    match &ty {
                        Type::Named(name) => {
                            if let Some(resolved) = self.resolve_typedef(name) {
                                resolved.clone()
                            } else {
                                ty
                            }
                        }
                        _ => ty,
                    }
                }
            }
            _ => {
                return Err(self.error_expected("type"));
            }
        };

        if !is_resource_template && !is_generic_template && !is_stream_template {
            self.advance();
        }
        tracing::trace!(
            target: TRACE_TYPES,
            line = self.previous_span().start,
            type = ?ty,
        );
        Ok(ty)
    }

    /// Parses `<Element, Count>` after an `InputPatch`/`OutputPatch` base
    /// name, baking both into the returned name (`OutputPatch<HS_OUTPUT,
    /// 1>`); HLSL spells these natively. Returns `Ok(None)` (consuming
    /// nothing) on any unusual shape — nested templates, complex counts
    /// — so the caller falls back to the legacy erase path. Never fails
    /// for I/O reasons (only inner type errors propagate).
    pub(crate) fn try_parse_patch_type(&mut self, text: &str) -> Result<Option<Type>> {
        let saved = self.index;
        let done = |s: &mut Self| -> Result<Option<Type>> {
            s.advance(); // base name
            if !s.peek_is(Token::Less) {
                return Ok(None);
            }
            s.advance();
            let elem = s.parse_type()?;
            if !s.peek_is(Token::Comma) {
                return Ok(None);
            }
            s.advance();
            let count = match s.tokens.get(s.index).map(|t| &t.token) {
                Some(Token::Number) | Some(Token::Identifier) => {
                    let lex = s.current_lexeme();
                    s.advance();
                    lex
                }
                _ => return Ok(None),
            };
            if !s.peek_is(Token::Greater) {
                return Ok(None);
            }
            s.advance();
            Ok(Some(Type::Named(format!(
                "{}<{}, {}>",
                text,
                elem.display(),
                count
            ))))
        };
        match done(self) {
            Ok(Some(ty)) => Ok(Some(ty)),
            Ok(None) => {
                self.index = saved;
                Ok(None)
            }
            Err(_) => {
                self.index = saved;
                Ok(None)
            }
        }
    }

    /// Parses `<Element, Extra...>` after a resource kind name, returning
    /// the element type. Extra template arguments (e.g. cache flags in
    /// `RW_RegularBuffer<uint, CacheFlags::kGL2Only>`) are dropped — only
    /// the element type is modelled.
    pub(crate) fn parse_resource_template_element(&mut self) -> Result<Type> {
        self.expect(Token::Less)?;
        let element = if self.peek_is(Token::Greater) || self.peek_is(Token::ShiftRight) {
            Type::Void
        } else {
            self.parse_type()?
        };
        // Trailing pointer/reference/const on the element
        // (`RegularBuffer<sce::Agc::MipStatsBuffer*>`) — dropped.
        loop {
            if self.peek_is(Token::Star) || self.peek_is(Token::Ampersand) {
                self.advance();
                continue;
            }
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
        while self.peek_is(Token::Comma) {
            self.advance(); // ','
            let mut angle = 0usize;
            while !self.eof() {
                if self.peek_is(Token::Less) {
                    angle += 1;
                    self.advance();
                } else if self.peek_is(Token::ShiftRight) {
                    if self.speculation_depth > 0 {
                        // Speculative parse: consume whole, never mutate.
                        self.advance();
                    } else {
                        // Split `>>` into `> >` so each closer is
                        // counted individually (mirrors expect_greater).
                        let idx = self.index;
                        let span = self.tokens[idx].span;
                        self.tokens[idx].token = Token::Greater;
                        self.tokens[idx].lexeme = ">".to_string();
                        self.tokens.insert(
                            idx + 1,
                            crate::token::SpannedToken {
                                token: Token::Greater,
                                lexeme: ">".to_string(),
                                span,
                            },
                        );
                    }
                } else if self.peek_is(Token::Greater) {
                    if angle == 0 {
                        break;
                    }
                    angle -= 1;
                    self.advance();
                } else if self.peek_is(Token::LParen) {
                    self.skip_paren_block()?;
                } else if self.peek_is(Token::LBracket) {
                    self.advance();
                    while !self.peek_is(Token::RBracket) && !self.eof() {
                        self.advance();
                    }
                    self.expect(Token::RBracket)?;
                } else if self.peek_is(Token::Comma) && angle == 0 {
                    break;
                } else {
                    self.advance();
                }
            }
        }
        self.expect_greater()?;
        Ok(element)
    }

    /// If the cursor is on `MACRO(` where `MACRO` normalizes to a resource
    /// kind (e.g. `REGULAR_BUFFER`), consumes `MACRO(ElementType)` and
    /// returns the desugared [`Type::Resource`]. Otherwise returns `None`
    /// without consuming any tokens.
    ///
    /// Framework headers declare SRT members as `REGULAR_BUFFER(T) name;`.
    /// Downstream lowering/emission already model `regularbuffer` etc., so
    /// desugaring here is exact, not a lossy skip.
    pub(crate) fn try_parse_macro_resource_type(&mut self) -> Result<Option<Type>> {
        if !(self.peek_is(Token::Identifier)
            && self.peek_next_is(Token::LParen)
            && is_resource_kind(&self.current_lexeme()))
        {
            return Ok(None);
        }
        let macro_name = self.current_lexeme();
        self.advance(); // macro name
        self.expect(Token::LParen)?;
        let element = self.parse_type()?;
        // Skip any extra macro arguments (access flags, counts, ...) up to ')'.
        while !self.peek_is(Token::RParen) && !self.eof() {
            if self.peek_is(Token::LParen) {
                self.skip_paren_block()?;
            } else if self.peek_is(Token::LBracket) {
                self.advance();
                while !self.peek_is(Token::RBracket) && !self.eof() {
                    self.advance();
                }
                self.expect(Token::RBracket)?;
            } else {
                self.advance();
            }
        }
        self.expect(Token::RParen)?;
        let kind = macro_name.to_ascii_lowercase().replace('_', "");
        let access = resolve_resource_access(&macro_name);
        Ok(Some(Type::Resource {
            kind,
            access,
            element: Box::new(element),
        }))
    }

    pub(crate) fn parse_optional_semantic(&mut self) -> Result<Option<Semantic>> {
        if !self.peek_is(Token::Colon) {
            return Ok(None);
        }
        self.advance();
        // A semantic name, e.g. `: SV_Position`. A bare number is a C++
        // bitfield width (`field :16;`) — consume and drop it (not modelled).
        if self.peek_is(Token::Identifier) {
            let name = self.expect_identifier()?;
            Ok(Some(Semantic::Name(name)))
        } else if self.peek_is(Token::Number) {
            self.advance();
            Ok(None)
        } else {
            Err(self.error_expected("identifier"))
        }
    }
}

// ------------------------------------------------------------------
// Type-name helpers (free functions)
// ------------------------------------------------------------------

/// Parses an identifier-derived type name, distinguishing vectors and
/// matrices (e.g. `float2`, `float2x3`). Scalar spellings and unrecognised
/// names fall back to [`Type::Named`] (the caller disambiguates).
pub(crate) fn parse_named_type(name: &str) -> Type {
    let lower = name.to_ascii_lowercase();
    // FXAA / GLSL aliases `vec2`/`vec3`/`vec4` → `float2`/`float3`/`float4`
    if matches!(lower.as_str(), "vec2" | "vec3" | "vec4") {
        let width = lower[3..].parse::<u8>().unwrap();
        return Type::Vector {
            base: ScalarType::Float,
            width,
        };
    }
    if lower == "sampler2d" || lower == "sampler2D" {
        // FXAA uses `sampler2D` for `Texture2D`/`SamplerState` pair — treat as resource
        return Type::Named(name.to_string());
    }
    // `ulong` / `dword` aliases: `ulong` is `uint`/`uint32_t` on the target (32-bit), `dword` similarly.
    // Bare `long` is a 32-bit `int` in PSSL.
    if lower == "ulong" || lower == "dword" {
        return Type::UInt;
    }
    if lower == "long" {
        return Type::Int;
    }
    if lower.starts_with("ulong") {
        let rest = &lower["ulong".len()..];
        if rest.is_empty() {
            return Type::UInt;
        }
        if let Some(x) = rest.find('x') {
            let rows = match rest[..x].parse::<u8>() {
                Ok(v) if (1..=4).contains(&v) => v,
                _ => return Type::Named(name.to_string()),
            };
            let columns = match rest[x + 1..].parse::<u8>() {
                Ok(v) if (1..=4).contains(&v) => v,
                _ => return Type::Named(name.to_string()),
            };
            return Type::Matrix {
                base: ScalarType::UInt,
                rows,
                columns,
            };
        }
        if let Ok(width) = rest.parse::<u8>() {
            if (1..=4).contains(&width) {
                return Type::Vector {
                    base: ScalarType::UInt,
                    width,
                };
            }
        }
        return Type::Named(name.to_string());
    }
    if lower.starts_with("dword") {
        let rest = &lower["dword".len()..];
        if rest.is_empty() {
            return Type::UInt;
        }
        if let Ok(width) = rest.parse::<u8>() {
            if (1..=4).contains(&width) {
                return Type::Vector {
                    base: ScalarType::UInt,
                    width,
                };
            }
        }
        return Type::Named(name.to_string());
    }
    // Standard aliases like `uint32_t` — treat as the underlying scalar.
    if matches!(
        lower.as_str(),
        "uint32_t" | "uint16_t" | "uint8_t" | "uint64_t" | "ushort" | "uintptr_t" | "size_t"
    ) {
        return Type::UInt;
    }
    if matches!(
        lower.as_str(),
        "int32_t" | "int16_t" | "int8_t" | "int64_t" | "short" | "intptr_t" | "ptrdiff_t"
    ) {
        return Type::Int;
    }
    if lower == "float16_t" {
        return Type::Half;
    }
    if lower == "float32_t" || lower == "float64_t" {
        return Type::Float;
    }
    // Bare `matrix` is PSSL for `float4x4` (also usable as a constructor).
    if lower == "matrix" {
        return Type::Matrix {
            base: ScalarType::Float,
            rows: 4,
            columns: 4,
        };
    }
    // `ushort2` / `short2` etc. — PSSL 16-bit variants map to UInt/Int vectors.
    if lower.starts_with("ushort") {
        let rest = &lower["ushort".len()..];
        if rest.is_empty() {
            return Type::UInt;
        }
        if let Some(x) = rest.find('x') {
            let rows = match rest[..x].parse::<u8>() {
                Ok(v) if (1..=4).contains(&v) => v,
                _ => return Type::Named(name.to_string()),
            };
            let columns = match rest[x + 1..].parse::<u8>() {
                Ok(v) if (1..=4).contains(&v) => v,
                _ => return Type::Named(name.to_string()),
            };
            return Type::Matrix {
                base: ScalarType::UInt,
                rows,
                columns,
            };
        }
        if let Ok(width) = rest.parse::<u8>() {
            if (1..=4).contains(&width) {
                return Type::Vector {
                    base: ScalarType::UInt,
                    width,
                };
            }
        }
        return Type::Named(name.to_string());
    }
    if lower.starts_with("short") {
        let rest = &lower["short".len()..];
        if rest.is_empty() {
            return Type::Int;
        }
        if let Some(x) = rest.find('x') {
            let rows = match rest[..x].parse::<u8>() {
                Ok(v) if (1..=4).contains(&v) => v,
                _ => return Type::Named(name.to_string()),
            };
            let columns = match rest[x + 1..].parse::<u8>() {
                Ok(v) if (1..=4).contains(&v) => v,
                _ => return Type::Named(name.to_string()),
            };
            return Type::Matrix {
                base: ScalarType::Int,
                rows,
                columns,
            };
        }
        if let Ok(width) = rest.parse::<u8>() {
            if (1..=4).contains(&width) {
                return Type::Vector {
                    base: ScalarType::Int,
                    width,
                };
            }
        }
        return Type::Named(name.to_string());
    }
    let Some(base) = scalar_name(&lower) else {
        return Type::Named(name.to_string());
    };
    let rest = &lower[base.name().len()..];

    if let Some(x) = rest.find('x') {
        let rows = match rest[..x].parse::<u8>() {
            Ok(v) if (1..=4).contains(&v) => v,
            _ => return Type::Named(name.to_string()),
        };
        let columns = match rest[x + 1..].parse::<u8>() {
            Ok(v) if (1..=4).contains(&v) => v,
            _ => return Type::Named(name.to_string()),
        };
        return Type::Matrix {
            base,
            rows,
            columns,
        };
    }

    if !rest.is_empty() {
        match rest.parse::<u8>() {
            Ok(width) if (1..=4).contains(&width) => {
                return Type::Vector { base, width };
            }
            _ => return Type::Named(name.to_string()),
        }
    }

    // Pure scalar spelling (e.g. "float", "half"); resolved by `builtin_type`.
    Type::Named(name.to_string())
}

pub(crate) fn scalar_name(lower: &str) -> Option<ScalarType> {
    // Handle `uint32_t` / `int32_t` style aliases used in framework headers
    if lower == "uint32_t" || lower == "uint16_t" || lower == "uint8_t" {
        return Some(ScalarType::UInt);
    }
    if lower == "int32_t" || lower == "int16_t" || lower == "int8_t" {
        return Some(ScalarType::Int);
    }
    if lower.starts_with("ushort") {
        Some(ScalarType::UInt)
    } else if lower.starts_with("short") {
        Some(ScalarType::Int)
    } else if lower.starts_with("float") {
        Some(ScalarType::Float)
    } else if lower.starts_with("half") {
        Some(ScalarType::Half)
    } else if lower.starts_with("int") {
        Some(ScalarType::Int)
    } else if lower.starts_with("uint") {
        Some(ScalarType::UInt)
    } else if lower.starts_with("bool") {
        Some(ScalarType::Bool)
    } else {
        None
    }
}

/// Detects a builtin constructor type name, e.g. `float`, `float2`, `float4`,
/// `float3x3`. Returns `None` for user-defined or unknown names.
pub(crate) fn builtin_type(name: &str) -> Option<Type> {
    let lower = name.to_ascii_lowercase();
    match lower.as_str() {
        "float" => return Some(Type::Float),
        "half" => return Some(Type::Half),
        "int" => return Some(Type::Int),
        "uint" => return Some(Type::UInt),
        "bool" => return Some(Type::Bool),
        "double" => return Some(Type::Double),
        _ => {}
    }
    match parse_named_type(&lower) {
        Type::Named(_) => None,
        ty => Some(ty),
    }
}

pub(crate) fn unquote(value: &str) -> String {
    match value.strip_prefix('"').and_then(|v| v.strip_suffix('"')) {
        Some(inner) => inner.to_string(),
        None => value.to_string(),
    }
}

impl Parser {
    /// Parses `enable_if<Cond, T>` (cursor on `enable_if`, next token `<`)
    /// as `T`, consuming through the closing `>`. Returns `Ok(None)` (cursor
    /// restored) when there is no top-level comma — the caller falls back to
    /// the generic template skip.
    pub(crate) fn try_parse_enable_if_type(&mut self) -> Result<Option<Type>> {
        if self.speculation_depth > 0 {
            return Ok(None);
        }
        let saved = self.index;
        // Cursor: `enable_if` then `<`.
        self.advance(); // base name
        self.advance(); // '<'
        // Scan the condition to its top-level comma.
        let mut angle = 0usize;
        let mut paren = 0usize;
        let mut found_comma = false;
        while !self.eof() {
            if self.peek_is(Token::LParen) {
                paren += 1;
            } else if self.peek_is(Token::RParen) {
                paren = paren.saturating_sub(1);
            } else if paren == 0 {
                if self.peek_is(Token::Less) {
                    angle += 1;
                } else if self.peek_is(Token::Greater) {
                    if angle == 0 {
                        break;
                    }
                    angle -= 1;
                } else if self.peek_is(Token::ShiftRight) {
                    if angle == 0 {
                        break;
                    }
                    angle = angle.saturating_sub(2);
                } else if self.peek_is(Token::Comma) && angle == 0 {
                    found_comma = true;
                    self.advance(); // ','
                    break;
                } else if self.peek_is(Token::Semi)
                    || self.peek_is(Token::LBrace)
                    || self.peek_is(Token::RBrace)
                {
                    break;
                }
            }
            self.advance();
        }
        if !found_comma {
            self.index = saved;
            return Ok(None);
        }
        let ty = self.parse_type()?;
        // Consume the closing `>` (splitting `>>` like the generic path).
        if self.peek_is(Token::Greater) {
            self.advance();
        } else if self.peek_is(Token::ShiftRight) {
            if self.speculation_depth > 0 {
                self.index = saved;
                return Ok(None);
            }
            let idx = self.index;
            let span = self.tokens[idx].span;
            self.tokens[idx].token = Token::Greater;
            self.tokens[idx].lexeme = ">".to_string();
            self.tokens.insert(
                idx + 1,
                crate::token::SpannedToken {
                    token: Token::Greater,
                    lexeme: ">".to_string(),
                    span,
                },
            );
            self.advance();
        } else {
            self.index = saved;
            return Ok(None);
        }
        let ty = match &ty {
            Type::Named(name) => self.resolve_typedef(name).unwrap_or(ty),
            _ => ty,
        };
        Ok(Some(ty))
    }
}

/// True for resource kinds that may appear without a `<element>` suffix and
/// still be element-less in HLSL (byte-address buffers). Bare textures and
/// structured buffers are NOT included: those always carry an element.
pub(crate) fn is_bare_resource_kind(name: &str) -> bool {
    let normalized = name.to_ascii_lowercase().replace('_', "");
    matches!(
        normalized.as_str(),
        "bytebuffer"
            | "byteaddressbuffer"
            | "rwbytebuffer"
            | "rwbyteaddressbuffer"
    )
}

/// True for texture kinds in PSSL underscore spelling without an
/// `<element>` suffix (e.g. `TextureCube_Array`). Only underscore forms
/// qualify: plain bare `Texture2D` keeps the legacy Named path, which
/// already validates, so it must not change shape.
pub(crate) fn is_bare_texture_kind(name: &str) -> bool {
    if !name.contains('_') {
        return false;
    }
    let normalized = name.to_ascii_lowercase().replace('_', "");
    matches!(
        normalized.as_str(),
        "texture1d"
            | "texture2d"
            | "texture2darray"
            | "texture3d"
            | "texturecube"
            | "texturecubearray"
            | "rwtexture1d"
            | "rwtexture2d"
            | "rwtexture2darray"
            | "rwtexture3d"
            | "rwtexturecube"
            | "rwtexturecubearray"
            | "mstexture2d"
            | "texture2dms"
            | "rwmstexture2d"
    )
}

pub(crate) fn is_resource_kind(name: &str) -> bool {    let normalized = name.to_ascii_lowercase().replace('_', "");
    matches!(
        normalized.as_str(),
        "texture1d"
            | "texture2d"
            | "texture2darray"
            | "texturecube"
            | "texturecubearray"
            | "texture3d"
            | "rwtexture2d"
            | "rwtexture2darray"
            | "rwtexture3d"
            | "rwtexture1d"
            | "rwtexturecube"
            | "rwtexturecubearray"
            | "mstexture2d"
            | "mstexturecubearray"
            | "databuffer"
            | "rwdatabuffer"
            | "regularbuffer"
            | "rwregularbuffer"
            | "structuredbuffer"
            | "rwstructuredbuffer"
            | "buffer"
            | "rwbuffer"
            | "bytebuffer"
            | "rwbytebuffer"
            | "byteaddressbuffer"
            | "rwbyteaddressbuffer"
            | "appendregularbuffer"
            | "appendstructuredbuffer"
            | "consumeregularbuffer"
            | "consumestructuredbuffer"
            | "rwmstexture2d"
            | "constantbuffer"
            | "cbuffer"
            | "sampler"
            | "samplerstate"
            | "samplercomparisonstate"
    )
}

pub(crate) fn resolve_resource_access(name: &str) -> crate::ast::ResourceAccess {
    let normalized = name.to_ascii_lowercase().replace('_', "");
    if normalized.starts_with("rw") {
        crate::ast::ResourceAccess::ReadWrite
    } else {
        crate::ast::ResourceAccess::ReadOnly
    }
}

pub(crate) fn parse_register_str(s: &str) -> Option<(char, u32)> {
    let mut chars = s.chars();
    let c = chars.next()?.to_ascii_lowercase();
    // `t`/`s`/`b`/`u` are SM5+ classes; `c` is the legacy constant
    // register (`register(c0)` on ConstantBuffers in older samples).
    // Downstream only uses the number (the class is re-derived from the
    // resource kind), so accepting `c` is exact for our purposes.
    if !matches!(c, 't' | 's' | 'b' | 'u' | 'c') {
        return None;
    }
    let num_str: String = chars.collect();
    let n = num_str.parse::<u32>().ok()?;
    Some((c, n))
}
