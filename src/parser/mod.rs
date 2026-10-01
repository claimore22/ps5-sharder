//! PSSL parser: token stream -> AST [`Module`].
//!
//! Sub-modules split the work by grammar area:
//! * [`module`] — module-level items (resources, structs, functions)
//! * [`stmt`] — statements
//! * [`expr`] — expressions (precedence climbing)
//! * [`types`] — type names and helpers

pub(crate) mod expr;
pub(crate) mod module;
pub(crate) mod scoped;
pub(crate) mod stmt;
pub(crate) mod types;
#[cfg(test)]
pub(crate) mod tests;

use crate::ast::*;
use crate::error::{Phase, Ps5ShaderError, Result, Span};
use crate::lexer;
use crate::token::{SpannedToken, Token};
use std::collections::HashMap;

pub(crate) const TRACE_TOKENS: &str = "pssl_shader::parse::tokens";
pub(crate) const TRACE_TYPES: &str = "pssl_shader::parse::types";
pub(crate) const TRACE_ENTER: &str = "pssl_shader::parse::enter";

/// Parses PSSL source text into an AST [`Module`].
pub fn parse_shader(source: &str) -> Result<Module> {
    let mut cleaned = strip_sce_sampler_helpers(source);
    cleaned = strip_vertex_layout_helpers(cleaned);
    let tokens = lexer::lex(&cleaned)?;
    Parser::new(tokens).parse_module()
}

/// Elides `static constexpr VertexLayout vertexLayout = VertexLayout::Create(...)`
/// which contains `sce::Agc::Core::Vertex*` C++ constructs not parseable as PSSL.
/// The layout is shader-irrelevant for HLSL (vertex input is inferred), so we
/// remove the entire declaration. This unblocks `shadow_vv` etc.
fn strip_vertex_layout_helpers(source: String) -> String {
    let mut out = source;
    // Remove any `VertexLayout vertexLayout ...;` declaration, whether it has
    // an initializer (`= VertexLayout::Create(...)`) or is already elided to
    // `VertexLayout vertexLayout;`. This handles both the original and the
    // previously-elided form.
    loop {
        let Some(vl_pos) = out.find("VertexLayout") else { break };
        let Some(rel) = out[vl_pos..].find("vertexLayout") else { break };
        let pos = vl_pos + rel;
        let line_start = out[..pos].rfind('\n').map(|p| p + 1).unwrap_or(0);
        let Some(semi_rel) = out[pos..].find(';') else { break };
        let semi_abs = pos + semi_rel;
        let slice = &out[line_start..=semi_abs];
        // Must be a `VertexLayout vertexLayout` declaration (global), not the
        // attribute `[vertexlayout(vertexLayout)]` which also contains the name
        // but not `VertexLayout` with capital V in the same line.
        if slice.contains("VertexLayout") && slice.contains("vertexLayout") {
            let end = std::cmp::min(semi_abs + 1, out.len());
            let end = if out[end..].starts_with('\n') { end + 1 } else { end };
            // IMPORTANT: the replacement must NOT contain "VertexLayout" /
            // "vertexLayout", or the loop below would re-match it, grab the
            // next `;`-terminated chunk, and cascade until the whole file
            // (including `main`) is replaced with comments.
            out.replace_range(line_start..end, "/* [vertex layout declaration elided] */\n");
        } else {
            break;
        }
    }
    out
}

/// Framework helper: `constexpr SamplerState make_*_sampler()` functions that
/// use `sce::Agc::Core::Sampler` (lexed as `Identifier Colon Colon ...`) are
/// not parseable as PSSL and are shader-irrelevant (we emit
/// `make_bilinear_sampler_shim` instead). Elide them to a trivial stub so the
/// rest of the shader parses. This unblocks `psfx_dof_blur_*` and similar.
fn strip_sce_sampler_helpers(source: &str) -> String {
    let mut out = source.to_string();
    for name in ["make_point_sampler", "make_trilinear_sampler"] {
        // Find `constexpr SamplerState make_X_sampler() { ... }` and replace
        // the braced body with `{ return SamplerState(); }`.
        let needle = format!("constexpr SamplerState {name}()");
        // Only one definition per helper exists per shader; elide it once.
        // (Using `while` would re-match the stub itself and loop forever.)
        if let Some(start) = out.find(&needle) {
            let brace_off = out[start..].find('{');
            if let Some(rel) = brace_off {
                let brace_start = start + rel;
                let mut depth = 0usize;
                let mut end_opt = None;
                for (i, ch) in out[brace_start..].char_indices() {
                    if ch == '{' {
                        depth += 1;
                    } else if ch == '}' {
                        depth -= 1;
                        if depth == 0 {
                            end_opt = Some(brace_start + i + ch.len_utf8());
                            break;
                        }
                    }
                }
                if let Some(end) = end_opt {
                    let stub = format!(
                        "constexpr SamplerState {name}() {{ SamplerState s; return s; }}"
                    );
                    out.replace_range(start..end, &stub);
                }
            }
        }
    }
    // Elide `make_bilinear_sampler` entirely — the HLSL emitter provides
    // `make_bilinear_sampler_shim` + `make_bilinear_sampler()` unconditionally,
    // so a stub would be a redefinition (DXC error redefinition of 'make_bilinear_sampler').
    let bil_needle = "constexpr SamplerState make_bilinear_sampler()";
    if let Some(start) = out.find(bil_needle) {
        if let Some(rel) = out[start..].find('{') {
            let brace_start = start + rel;
            let mut depth = 0usize;
            let mut end_opt = None;
            for (i, ch) in out[brace_start..].char_indices() {
                if ch == '{' {
                    depth += 1;
                } else if ch == '}' {
                    depth -= 1;
                    if depth == 0 {
                        end_opt = Some(brace_start + i + ch.len_utf8());
                        break;
                    }
                }
            }
            if let Some(end) = end_opt {
                out.replace_range(
                    start..end,
                    "/* make_bilinear_sampler elided; shim provides it */",
                );
            }
        }
    }
    // Sampler factories (`MakeClampLinearSampler`,
    // `MakeWrapLinearSampler`, `MakeWrapTrilinearAniso8Sampler`, ...):
    // `static constexpr SamplerState
    // Name()` whose body builds `sce::Agc::Core::Sampler().init().set*()`
    // chains. The names vary per shader, but the shape is uniform and
    // shader-irrelevant (samplers are externally bound — same rationale
    // as the named factories above). Stub any such factory to the same
    // trivial body by shape: `constexpr SamplerState <ident>()` whose
    // braced body contains a `Sampler()` construction plus `.init(`.
    // Bodies without both markers (already-stubbed helpers, real logic)
    // are untouched, as are the psfx-owned `make_*_sampler` names.
    {
        let mut search_from = 0usize;
        while let Some(rel) = out[search_from..].find("constexpr SamplerState ") {
            let name_start = search_from + rel + "constexpr SamplerState ".len();
            let name_end = out[name_start..]
                .find(|c: char| !c.is_ascii_alphanumeric() && c != '_')
                .map(|i| name_start + i)
                .unwrap_or(out.len());
            let name: String = out[name_start..name_end].to_string();
            let mut rest = out[name_end..].chars();
            let paren_ok = rest.next() == Some('(')
                && rest.next() == Some(')')
                && {
                    // Allow whitespace between `()` and `{`.
                    let tail: String = rest.collect();
                    tail.trim_start().starts_with('{')
                };
            let skip = name.is_empty()
                || [
                    "make_point_sampler",
                    "make_bilinear_sampler",
                    "make_trilinear_sampler",
                ]
                .contains(&name.as_str())
                || !paren_ok;
            if skip {
                search_from = name_end;
                continue;
            }
            let brace_off = out[name_end..].find('{');
            let Some(relb) = brace_off else {
                search_from = name_end;
                continue;
            };
            let brace_start = name_end + relb;
            let mut depth = 0usize;
            let mut end_opt = None;
            for (i, ch) in out[brace_start..].char_indices() {
                if ch == '{' {
                    depth += 1;
                } else if ch == '}' {
                    depth -= 1;
                    if depth == 0 {
                        end_opt = Some(brace_start + i + ch.len_utf8());
                        break;
                    }
                }
            }
            let Some(end) = end_opt else {
                search_from = name_end;
                continue;
            };
            let body_has_builder =
                out[brace_start..end].contains("Sampler()") && out[brace_start..end].contains(".init(");
            if !body_has_builder {
                search_from = end;
                continue;
            }
            let stub = format!("constexpr SamplerState {name}() {{ SamplerState s; return s; }}");
            out.replace_range(name_start - "constexpr SamplerState ".len()..end, &stub);
            search_from = name_start - "constexpr SamplerState ".len() + stub.len();
        }
    }
    // Target wave intrinsics: `__get_wavemode()` / `__kWaveModeWave64` are used
    // only in `get_lanes_per_wave()` which is not essential for DOF logic.
    // Stub that helper to avoid needing to model wave64 intrinsics.
    let wav_needle = "uint get_lanes_per_wave()";
    if let Some(start) = out.find(wav_needle) {
        if let Some(rel) = out[start..].find('{') {
            let brace_start = start + rel;
            let mut depth = 0usize;
            let mut end_opt = None;
            for (i, ch) in out[brace_start..].char_indices() {
                if ch == '{' {
                    depth += 1;
                } else if ch == '}' {
                    depth -= 1;
                    if depth == 0 {
                        end_opt = Some(brace_start + i + ch.len_utf8());
                        break;
                    }
                }
            }
            if let Some(end) = end_opt {
                tracing::trace!(target: "pssl_shader::psfx::intrinsics", "parser fallback: get_lanes_per_wave → WaveGetLaneCount()");
                out.replace_range(start..end, "uint get_lanes_per_wave() { return WaveGetLaneCount(); }");
            }
        }
    }
    out
}

pub(crate) struct Parser {
    pub(crate) tokens: Vec<SpannedToken>,
    pub(crate) index: usize,
    /// Type aliases introduced by C++ `typedef` declarations (e.g.
    /// `typedef float2 Vector2Unaligned;`). Keyed by alias name.
    pub(crate) typedefs: HashMap<String, Type>,
    /// Integer constants declared with a literal initializer (e.g.
    /// `const int k_sample_count = 4;`). Keyed by name; used to resolve
    /// non-literal array sizes like `samples[k_sample_count]`.
    pub(crate) int_consts: HashMap<String, String>,
    /// Folded `static const` / `static constexpr` struct members, keyed by
    /// qualified `Struct::member` (e.g.
    /// `ParticleInstance::kMaxHistoryBufferGenerations` → `"10"`).
    /// Recorded during struct parsing for attribute evaluation
    /// (`[MAX_VERTEX_COUNT((S::kMax - 1) * 4)]`); flushed into
    /// `Module::constants` at module end.
    pub(crate) struct_static_consts: HashMap<String, String>,
    /// Static member initializers by qualified `Struct::member`, as parsed
    /// (unfolded expressions, may reference template value-params).
    /// Recorded during struct parsing for `T::kX` constexpr evaluation;
    /// flushed into `Module::template_static_inits` at module end.
    pub(crate) template_static_inits: HashMap<String, crate::ast::Expr>,
    /// Template value-param defaults by qualified `Scope::Param` (e.g.
    /// `PackVertex::BoneCount` → `"0"`), as token strings. Recorded
    /// during template-prefix parsing for binding partial specialization
    /// args (`PackVertex<false,false,0>` omits defaulted trailing params);
    /// flushed into `Module::template_param_defaults` at module end.
    /// Only single-literal defaults capture here.
    pub(crate) template_param_defaults: HashMap<String, String>,
    /// Multi-dimensional struct-field dims by qualified `Struct::field`
    /// (e.g. `Partition3::mask3bit` → `[3, 2]`). Recorded during struct
    /// parsing for row-major use linearization; flushed into
    /// `Module::field_multidim` at module end. 2D all-constant only
    /// (anything else keeps legacy first-dimension handling).
    pub(crate) field_multidim: HashMap<String, Vec<u32>>,
    /// Struct names encountered so far (e.g. `PackVertex`, `PsUserData`).
    /// Used to distinguish `Type var` from `var[index].member` assignments.
    /// NOTE: top-level `enum` names are also recorded here (they appear
    /// in type position); `enum_names` below distinguishes them from
    /// real structs for qualified-name resolution.
    pub(crate) struct_names: std::collections::HashSet<String>,
    /// Top-level `enum` type names (a subset of `struct_names`). A
    /// qualified `Ns::X` resolves through the bare `X` typedef only when
    /// `X` is NOT a real struct; without this set a colliding
    /// `typedef uint32_t Hit` would hijack `sce::Psr::Hit` (the struct).
    pub(crate) enum_names: std::collections::HashSet<String>,
    /// Nested struct declarations collected while parsing a struct body,
    /// drained by the module parser after each top-level struct.
    pub(crate) pending_structs: Vec<crate::ast::StructDecl>,
    /// Struct member-function definitions collected while parsing a struct
    /// body (`float3 S::...` in spirit: stored with `scope` set), drained
    /// by the module parser after each top-level struct. Emitted as HLSL
    /// struct methods so `obj.method()` calls resolve natively.
    pub(crate) pending_methods: Vec<crate::ast::Function>,
    /// Counter for synthesized anonymous-struct names (`Parent__anon0`).
    pub(crate) anon_counter: usize,
    /// Nesting depth of speculative parses (C-style cast probes) that
    /// roll back the token index on failure. Token-stream surgery
    /// (`>>` splits) is forbidden while > 0, since only the index is
    /// restored — mutations would corrupt later parsing.
    pub(crate) speculation_depth: u32,
    /// Type params from `template<typename T, class U, ...>` prefixes.
    /// Consulted by declaration lookahead (`Type<...> name` vs `<`
    /// comparison) so template-dependent types count as known types.
    pub(crate) template_type_params: std::collections::HashSet<String>,
    /// Ordered template params from the most recent `template<...>`
    /// prefix: (name, is_value). `typename T`/`class T` are type params;
    /// `bool B`/`int N`/etc. are value params. Drained into the next
    /// parsed function (`Function::template_params`); cleared when a
    /// struct or global follows instead (no misattribution).
    pub(crate) pending_template_params: Vec<(String, bool)>,
    /// Defaults for pending value params (`(param, default-string)`),
    /// drained alongside `pending_template_params` with the decl scope.
    /// Only single-literal defaults capture here.
    pub(crate) pending_template_defaults: Vec<(String, String)>,
    /// Template-id spellings on parameters, pending the enclosing
    /// function: (variable, normalized `Name<args>`). Drained into
    /// `template_spellings[fn_name]` at each function parse.
    pub(crate) pending_param_spellings: Vec<(String, String)>,
    /// Template-id spellings on locals, pending the enclosing function:
    /// (variable, normalized `Name<args>`). Drained into
    /// `template_site_spellings[fn_name]` at each function parse
    /// (separate from params so existing consumers are untouched).
    pub(crate) pending_site_spellings: Vec<(String, String)>,
    /// Template-id spellings on struct fields: one frame per open
    /// struct body. Drained into `template_site_spellings[struct]`
    /// at each struct parse (a stack so nested structs attribute
    /// correctly).
    pub(crate) pending_field_spellings: Vec<Vec<(String, String)>>,
    /// Template-id spellings by function ("" for globals): variable ->
    /// normalized `Name<args>`. Moved into the module at parse end.
    pub(crate) template_spellings: HashMap<String, HashMap<String, String>>,
    /// Template-id spellings on locals (by function) and struct fields
    /// (by struct): variable/field -> normalized `Name<args>`. Drained
    /// from the pending buffers at each function/struct parse; moved
    /// into the module at parse end. Separate from `template_spellings`
    /// (params) so existing consumers are untouched.
    pub(crate) template_site_spellings: HashMap<String, HashMap<String, String>>,
    /// Observed template-ids module-wide: (base, whitespace-stripped
    /// args), recorded at every non-speculative type parse. Drives
    /// struct monomorphization (single-spelling rule). Moved into the
    /// module at parse end.
    pub(crate) observed_template_ids: std::collections::HashSet<(String, String)>,
    /// Pending template non-type param names from a preceding `template<...>` (e.g. `flags` in `template<uint64_t flags>`).
    /// Drained into the next function as synthetic locals so uses like `flags|` type-check.
    pub(crate) pending_template_values: Vec<String>,
    /// Function-local struct declarations (`struct output {...};` inside a
    /// body): original name -> counter-mangled module name. Reset per
    /// function; consulted when a body declaration references the type.
    /// Locals shadow globals while live (C++ scoping); the mangled
    /// declaration is drained via `pending_structs`.
    pub(crate) local_struct_renames: HashMap<String, String>,
    /// Counter for mangled local-struct names (`output__local3`).
    pub(crate) local_struct_counter: usize,
    /// Enumerator names from nested struct enums (one inner vec per
    /// enum body, each entry with its raw value text), drained into int
    /// globals at the same point as `pending_structs` (top-level enums
    /// register via `parse_enum_body` directly).
    pub(crate) pending_enum_names: Vec<Vec<(String, Option<String>)>>,
    /// A bare `CXX_SYMBOL("...")` / `CxxSymbol("...")` embedding marker seen
    /// at module scope. It applies to the next parsed function, so it is
    /// drained into that function's attributes as `CxxSymbol("...")`.
    pub(crate) pending_cxx_symbol: Option<String>,
    /// Multi-dimensional array declarations (`float3 a[Y][X]` globals,
    /// locals, params): name -> (dims, function sequence; `None` is module
    /// scope). The parser flattens exactly-2D all-constant arrays to
    /// product size and linearizes `a[i][j]` uses inline (row-major).
    /// Function-scoped entries shadow globals with the same name.
    pub(crate) multidim: Vec<(String, Vec<u32>, Option<u64>)>,
    /// Sequence counter tagging `multidim` entries with their function.
    pub(crate) function_seq: u64,
    /// Function currently being parsed (for `multidim` scope tags).
    pub(crate) current_function: Option<u64>,
}

impl Parser {
    pub(crate) fn new(tokens: Vec<SpannedToken>) -> Self {
        Self {
            tokens,
            index: 0,
            typedefs: HashMap::new(),
            int_consts: HashMap::new(),
            struct_static_consts: HashMap::new(),
            template_static_inits: HashMap::new(),
            template_param_defaults: HashMap::new(),
            field_multidim: HashMap::new(),
            struct_names: std::collections::HashSet::new(),
            enum_names: std::collections::HashSet::new(),
            pending_structs: Vec::new(),
            pending_methods: Vec::new(),
            anon_counter: 0,
            speculation_depth: 0,
            template_type_params: std::collections::HashSet::new(),
            pending_template_params: Vec::new(),
            pending_template_defaults: Vec::new(),
            pending_param_spellings: Vec::new(),
            pending_site_spellings: Vec::new(),
            pending_field_spellings: Vec::new(),
            template_spellings: HashMap::new(),
            template_site_spellings: HashMap::new(),
            observed_template_ids: std::collections::HashSet::new(),
            pending_template_values: Vec::new(),
            local_struct_renames: HashMap::new(),
            local_struct_counter: 0,
            pending_cxx_symbol: None,
            pending_enum_names: Vec::new(),
            multidim: Vec::new(),
            function_seq: 0,
            current_function: None,
        }
    }

    /// Records a multi-dimensional declaration for index linearization,
    /// returning the flattened array size, or `None` to keep legacy
    /// first-dimension handling. 2D+ all-constant dims qualify (nested
    /// braced inits belong to the vector rule, struct fields/resources
    /// stay loud). The size is the row-major product; 2D uses linearize
    /// as `a[i*X + j]` at index-parse time via [`Self::multidim_stride`],
    /// while 3D+ chains fold at postfix exit via
    /// [`Self::fold_multidim_chain`].
    pub(crate) fn note_multidim(
        &mut self,
        name: &str,
        dims: &[Option<u32>],
        has_nested_init: bool,
    ) -> Option<u32> {
        if dims.len() < 2 || has_nested_init {
            return None;
        }
        let mut all = Vec::with_capacity(dims.len());
        for d in dims {
            all.push((*d)?);
        }
        let product = all.iter().fold(1u32, |a, b| a.saturating_mul(*b));
        self.multidim
            .push((name.to_string(), all, self.current_function));
        Some(product)
    }

    /// Row stride (second dimension) for a flattened 2D array name:
    /// current-function entries shadow module-scope ones; latest match
    /// wins within a scope.
    pub(crate) fn multidim_stride(&self, name: &str) -> Option<u32> {
        let mut global = None;
        for (n, dims, seq) in self.multidim.iter().rev() {
            if n != name || dims.len() != 2 {
                continue;
            }
            if *seq == self.current_function {
                return Some(dims[1]);
            }
            if seq.is_none() && global.is_none() {
                global = Some(dims[1]);
            }
        }
        global
    }

    /// Full dims for a flattened N-D array name (same scope rule as
    /// [`Self::multidim_stride`]).
    pub(crate) fn multidim_dims(&self, name: &str) -> Option<Vec<u32>> {
        let mut global = None;
        for (n, dims, seq) in self.multidim.iter().rev() {
            if n != name {
                continue;
            }
            if *seq == self.current_function {
                return Some(dims.clone());
            }
            if seq.is_none() && global.is_none() {
                global = Some(dims.clone());
            }
        }
        global
    }

    /// Folds a complete N-index chain on a flattened 3D+ array
    /// (`a[i][j][k]` with dims `[D0,D1,D2]` becomes `a[(i*D1+j)*D2+k]`,
    /// row-major, matching PSSL layout). Only fires when the chain depth
    /// exactly matches the dims (partial indexing keeps today's path);
    /// 2D names already fold mid-chain via [`Self::multidim_stride`].
    /// Anything else returns the expression untouched.
    pub(crate) fn fold_multidim_chain(&self, expr: Expr) -> Expr {
        let mut idxs: Vec<Expr> = Vec::new();
        let mut cur = &expr;
        while let Expr::Index { object, index } = cur {
            idxs.push(index.as_ref().clone());
            cur = object.as_ref();
        }
        let Expr::Variable(name) = cur else {
            return expr;
        };
        let Some(dims) = self.multidim_dims(name) else {
            return expr;
        };
        if dims.len() < 3 || idxs.len() != dims.len() {
            return expr;
        }
        // Outside-in collection above; innermost first for the nest.
        idxs.reverse();
        let mut flat = idxs[0].clone();
        for (k, idx) in idxs.iter().enumerate().skip(1) {
            flat = Expr::Binary {
                left: Box::new(Expr::Binary {
                    left: Box::new(flat),
                    op: crate::ast::BinaryOp::Mul,
                    right: Box::new(Expr::Number(dims[k].to_string())),
                }),
                op: crate::ast::BinaryOp::Add,
                right: Box::new(idx.clone()),
            };
        }
        Expr::Index {
            object: Box::new(Expr::Variable(name.clone())),
            index: Box::new(flat),
        }
    }

    /// Resolves a function-local struct name to its mangled module name
    /// (`None` when not a live local). Plain names only; `::`-qualified
    /// names are never locals.
    pub(crate) fn resolve_local_struct(&self, name: &str) -> Option<String> {
        if name.contains("::") {
            return None;
        }
        self.local_struct_renames.get(name).cloned()
    }

    /// Resolves an array-size expression to a literal `u32` when it is a
    /// numeric literal or a named integer constant (e.g.
    /// `samples[k_sample_count]` where `k_sample_count = 4`).
    fn resolve_array_size(&self, size_expr: &Expr) -> Option<u32> {
        self.resolve_array_size_with(size_expr, None)
    }

    /// `resolve_array_size` with an additional scope of named constants
    /// (struct member `static const`s while parsing that struct's body).
    /// The extra scope wins over globals, matching C++ member lookup.
    fn resolve_array_size_with(
        &self,
        size_expr: &Expr,
        extra: Option<&std::collections::HashMap<String, String>>,
    ) -> Option<u32> {
        match size_expr {
            Expr::Number(s) => {
                // Hex first (`0xffff`): `f`/`F` are hex digits, not
                // suffixes, so only integer suffixes (`u/U/l/L`) strip.
                // Decimal strips float suffixes too (legacy).
                if let Some(hex) = s
                    .strip_prefix("0x")
                    .or_else(|| s.strip_prefix("0X"))
                {
                    let hex = hex.trim_end_matches(|c| {
                        c == 'u' || c == 'U' || c == 'l' || c == 'L'
                    });
                    u32::from_str_radix(hex, 16).ok()
                } else {
                    let num_str = s.trim_end_matches(|c| {
                        c == 'f' || c == 'F' || c == 'u' || c == 'U' || c == 'l' || c == 'L'
                    });
                    num_str.parse::<u32>().ok()
                }
            }
            Expr::Variable(name) => extra
                .and_then(|m| m.get(name))
                .or_else(|| self.int_consts.get(name))
                .and_then(|v| v.parse::<u32>().ok()),
            // Const-arithmetic array sizes (e.g. `ao_store[k_lds_size]` where
            // `k_lds_size = k_wavefront_size * k_lds_line_size`). Fold when
            // both operands resolve to known integer constants.
            Expr::Binary { left, op, right } => {
                let a = self.resolve_array_size_with(left, extra)?;
                let b = self.resolve_array_size_with(right, extra)?;
                use crate::ast::BinaryOp::*;
                let v = match op {
                    Add => a.checked_add(b)?,
                    Sub => a.checked_sub(b)?,
                    Mul => a.checked_mul(b)?,
                    Div => b.checked_div(a)?.try_into().ok()?,
                    Mod => (a % b).try_into().ok()?,
                    ShiftLeft => a.checked_shl(b)?,
                    ShiftRight => a.checked_shr(b)?,
                    BitAnd => a & b,
                    BitOr => a | b,
                    BitXor => a ^ b,
                    _ => return None,
                };
                Some(v)
            }
            Expr::Unary { op: crate::ast::UnaryOp::Minus, expr } => {
                let v = self.resolve_array_size_with(expr, extra)?;
                v.checked_neg().and_then(|n| i64::try_from(n).ok()).map(|n| n as u32)
            }
            _ => None,
        }
    }

    // ------------------------------------------------------------------
    // Class-body member parsing (field extraction with safe fallback)
    // ------------------------------------------------------------------

    /// Parses one `class` body member for field extraction. Returns `false`
    /// on anything unrecognized (the caller rewinds and skips the body).
    /// Methods, labels, nested types and macros are skipped; data members
    /// are appended to `fields`.
    pub(crate) fn parse_class_member(
        &mut self,
        fields: &mut Vec<crate::ast::FieldDecl>,
    ) -> bool {
        use crate::token::Token;
        // Access labels (`public:`) and lone `;`.
        if self.peek_is(Token::Semi) {
            self.advance();
            return true;
        }
        if self.peek_is(Token::Identifier)
            && matches!(
                self.current_lexeme().to_ascii_lowercase().as_str(),
                "public" | "private" | "protected"
            )
            && matches!(
                self.tokens.get(self.index + 1).map(|t| &t.token),
                Some(Token::Colon)
            )
            && !matches!(
                self.tokens.get(self.index + 2).map(|t| &t.token),
                Some(Token::Colon)
            )
        {
            self.advance();
            self.advance();
            return true;
        }
        // `template<...>` prefix — skip the angle list, re-examine.
        if self.peek_is(Token::Identifier) && self.current_lexeme() == "template" {
            self.advance();
            if self.peek_is(Token::Less) && self.skip_angle_block().is_err() {
                return false;
            }
            return true;
        }
        // Destructor / operator / typedef / using / asserts — skip whole.
        if self.peek_is(Token::Tilde) {
            self.advance();
            return self.skip_to_semi_or_body();
        }
        if self.peek_is(Token::Identifier)
            && matches!(
                self.current_lexeme().to_ascii_lowercase().as_str(),
                "operator" | "typedef" | "using" | "static_assert" | "assert"
            )
        {
            self.advance();
            if self.peek_is(Token::Semi) {
                self.advance();
                return true;
            }
            return self.skip_to_semi_or_body();
        }
        // Nested types — skip whole.
        if self.peek_is(Token::Identifier)
            && matches!(
                self.current_lexeme().as_str(),
                "struct" | "class" | "union" | "enum"
            )
        {
            self.advance();
            if self.peek_is(Token::Identifier) {
                self.advance();
            }
            if self.peek_is(Token::Less) && self.skip_angle_block().is_err() {
                return false;
            }
            if self.peek_is(Token::LBrace) {
                if self.skip_brace_block().is_err() {
                    return false;
                }
                // Trailing `;` / instance / macro — consume through `;`.
                while !self.peek_is(Token::Semi)
                    && !self.peek_is(Token::RBrace)
                    && !self.eof()
                {
                    self.advance();
                }
                if self.peek_is(Token::Semi) {
                    self.advance();
                }
                return true;
            }
            return self.skip_to_semi();
        }
        // Leading member qualifiers and interposed ALL-CAPS macros, in any
        // order (`SCE_MACRO const float getX();`, `SCE_MACRO sce::T x;`).
        // A macro is skipped only when a type and a declarator follow
        // (`MACRO Type name...`); a `TYPE name;` pair never skips (the token
        // after the name is `;`, not a declarator), so all-caps type
        // spellings (`UINT x;`) are safe.
        for _ in 0..6 {
            let mut progressed = false;
            loop {
                if self.peek_is(Token::Const) || self.peek_is(Token::Static) {
                    self.advance();
                    progressed = true;
                    continue;
                }
                if self.peek_is(Token::Identifier)
                    && matches!(
                        self.current_lexeme().to_ascii_lowercase().as_str(),
                        "const"
                            | "constexpr"
                            | "static"
                            | "inline"
                            | "mutable"
                            | "virtual"
                            | "explicit"
                            | "friend"
                            | "typename"
                    )
                {
                    self.advance();
                    progressed = true;
                    continue;
                }
                break;
            }
            if self.peek_is(Token::Identifier)
                && Self::is_macro_like_name(&self.current_lexeme())
            {
                let t1 = self.tokens.get(self.index + 1).map(|t| &t.token);
                let t2 = self.tokens.get(self.index + 2).map(|t| &t.token);
                let t1_ok = matches!(
                    t1,
                    Some(
                        Token::Identifier
                            | Token::Void
                            | Token::Bool
                            | Token::Int
                            | Token::UInt
                            | Token::Float
                            | Token::Half
                            | Token::Double
                            | Token::Const
                            | Token::Static
                    )
                );
                let t2_is_scope = matches!(t2, Some(Token::Colon))
                    && matches!(
                        self.tokens.get(self.index + 3).map(|t| &t.token),
                        Some(Token::Colon)
                    );
                let t2_ok = matches!(
                    t2,
                    Some(Token::Identifier)
                        | Some(Token::Star)
                        | Some(Token::Ampersand)
                        | Some(Token::Less)
                        | Some(Token::LParen)
                ) || t2_is_scope;
                if t1_ok && t2_ok {
                    self.advance();
                    progressed = true;
                    continue;
                }
            }
            if !progressed {
                break;
            }
        }
        if self.eof() || self.peek_is(Token::RBrace) {
            return true;
        }
        // `~dtor` / `operator` after qualifiers.
        if self.peek_is(Token::Tilde) {
            self.advance();
            return self.skip_to_semi_or_body();
        }
        if self.peek_is(Token::Identifier) && self.current_lexeme() == "operator" {
            self.advance();
            return self.skip_to_semi_or_body();
        }
        // A type must follow. `(` here means something unshaped — bail.
        let ty = match self.parse_type() {
            Ok(t) => t,
            Err(_) => return false,
        };
        // Trailing `::type`.
        while self.peek_is(Token::Colon)
            && matches!(
                self.tokens.get(self.index + 1).map(|t| &t.token),
                Some(Token::Colon)
            )
        {
            self.advance();
            self.advance();
            if self.peek_is(Token::Identifier) {
                self.advance();
            } else {
                break;
            }
        }
        // `(` directly after the type: constructor/method without a name.
        if self.peek_is(Token::LParen) {
            return self.skip_method_rest();
        }
        // Pointer/reference between type and name (`float &foo();`).
        if self.peek_is(Token::Star) || self.peek_is(Token::Ampersand) {
            while self.peek_is(Token::Star) || self.peek_is(Token::Ampersand) {
                self.advance();
            }
            // `(*name)` / `(&name)` parenthesized declarators (function
            // pointers, grouping): skip to matching `)` then method/field
            // handling continues below via the name check.
            if self.peek_is(Token::LParen) {
                if self.skip_paren_block().is_err() {
                    return false;
                }
            }
        }
        // One or more comma-separated declarators (`float xres, yres;`).
        // A `name(` declarator is actually a method (or conversion-style
        // `operator` target) — skip it as a method and finish.
        let mut first = true;
        loop {
            if !self.peek_is(Token::Identifier) {
                // No declarator at all (e.g. a bare type) bails; otherwise
                // the list is done (the `;` check below confirms).
                if first {
                    return false;
                }
                break;
            }
            first = false;
            let name = self.current_lexeme();
            self.advance();
            if name == "operator" {
                let _ = self.skip_operator_target();
                return self.skip_method_rest();
            }
            if self.peek_is(Token::LParen) {
                return self.skip_method_rest();
            }
            // Array dims.
            let mut array_size = None;
            while self.peek_is(Token::LBracket) {
                self.advance();
                if !self.peek_is(Token::RBracket) {
                    let mut depth = 0usize;
                    while !self.eof() {
                        if self.peek_is(Token::LBracket) {
                            depth += 1;
                        } else if self.peek_is(Token::RBracket) {
                            if depth == 0 {
                                break;
                            }
                            depth -= 1;
                        }
                        self.advance();
                    }
                }
                if !self.peek_is(Token::RBracket) {
                    return false;
                }
                self.advance();
                array_size = Some(0);
            }
            // Brace init (`int x{0};`) or `= init` (skipped, balanced).
            if self.peek_is(Token::LBrace) {
                if self.skip_brace_block().is_err() {
                    return false;
                }
            } else if self.peek_is(Token::Equal) {
                self.advance();
                let mut paren = 0usize;
                while !self.eof()
                    && !(self.peek_is(Token::Semi) && paren == 0)
                    && !(self.peek_is(Token::Comma) && paren == 0)
                    && !self.peek_is(Token::RBrace)
                {
                    if self.peek_is(Token::LParen) {
                        paren += 1;
                    } else if self.peek_is(Token::RParen) {
                        if paren == 0 {
                            break;
                        }
                        paren -= 1;
                    } else if self.peek_is(Token::LBrace) {
                        if self.skip_brace_block().is_err() {
                            return false;
                        }
                        continue;
                    }
                    self.advance();
                }
            }
            let span = crate::error::Span::default();
            fields.push(crate::ast::FieldDecl {
                ty: ty.clone(),
                name,
                semantic: None,
                array_size,
                span,
            });
            // Trailing marker macros (`float d SCE_VECTORMATH_UNUSED;`).
            while self.peek_is(Token::Identifier)
                && Self::is_macro_like_name(&self.current_lexeme())
                && matches!(
                    self.tokens.get(self.index + 1).map(|t| &t.token),
                    Some(Token::Semi) | Some(Token::Comma)
                )
            {
                self.advance();
            }
            if self.peek_is(Token::Comma) {
                self.advance();
                continue;
            }
            break;
        }
        if !self.peek_is(Token::Semi) {
            return false;
        }
        self.advance();
        true
    }

    /// Skips a method signature remainder: optional ctor-init list, then a
    /// body `{...}` (plus `;`) or a plain `;`. Always succeeds unless EOF.
    pub(crate) fn skip_method_rest(&mut self) -> bool {
        use crate::token::Token;
        while self.peek_is(Token::LParen) {
            if self.skip_paren_block().is_err() {
                return false;
            }
        }
        // Trailing qualifiers / macros / ctor-init `: m(x)` until `{`/`;`.
        loop {
            if self.eof() || self.peek_is(Token::Semi) || self.peek_is(Token::LBrace) {
                break;
            }
            if self.peek_is(Token::LParen) {
                if self.skip_paren_block().is_err() {
                    return false;
                }
                continue;
            }
            if self.peek_is(Token::RBrace) {
                break;
            }
            self.advance();
        }
        if self.peek_is(Token::LBrace) {
            if self.skip_brace_block().is_err() {
                return false;
            }
            if self.peek_is(Token::Semi) {
                self.advance();
            }
        } else if self.peek_is(Token::Semi) {
            self.advance();
        } else {
            return false;
        }
        true
    }

    /// Skips to `;` (or a body, which is then skipped). Used for member
    /// shapes with no extractable info.
    pub(crate) fn skip_to_semi_or_body(&mut self) -> bool {
        use crate::token::Token;
        loop {
            if self.eof() {
                return false;
            }
            if self.peek_is(Token::Semi) {
                self.advance();
                return true;
            }
            if self.peek_is(Token::LBrace) {
                if self.skip_brace_block().is_err() {
                    return false;
                }
                if self.peek_is(Token::Semi) {
                    self.advance();
                }
                return true;
            }
            if self.peek_is(Token::RBrace) {
                return true;
            }
            if self.peek_is(Token::LParen) {
                if self.skip_paren_block().is_err() {
                    return false;
                }
                continue;
            }
            self.advance();
        }
    }

    /// Skips to `;`. Returns `false` on EOF/RBrace (caller rewinds).
    pub(crate) fn skip_to_semi(&mut self) -> bool {
        use crate::token::Token;
        while !self.eof() && !self.peek_is(Token::Semi) && !self.peek_is(Token::RBrace) {
            self.advance();
        }
        if self.peek_is(Token::Semi) {
            self.advance();
            return true;
        }
        false
    }

    // ------------------------------------------------------------------
    // Module-level items
    // ------------------------------------------------------------------

    /// Parses an enum body (`{ A [= v], ... }`), returning enumerator
    /// names with their raw value text (`None` when omitted). Expects
    /// the cursor on `{`, consumes through `}`. Values nest parens
    /// (e.g. `(1 << 2)`); the text is captured verbatim for constant
    /// evaluation by the caller.
    pub(crate) fn parse_enumerator_names(&mut self) -> Result<Vec<(String, Option<String>)>> {
        self.expect(Token::LBrace)?;
        let mut names = Vec::new();
        loop {
            if self.peek_is(Token::Identifier) {
                let name = self.current_lexeme();
                self.advance();
                // Optional `= value`: capture the top-level `,`/`}`-span
                // verbatim (values nest parens, e.g. `(1 << 2)`).
                let mut value: Option<String> = None;
                if self.peek_is(Token::Equal) {
                    self.advance();
                    let mut text = String::new();
                    let mut paren = 0usize;
                    let mut angle = 0usize;
                    while !self.eof() {
                        if self.peek_is(Token::LParen) {
                            paren += 1;
                        } else if self.peek_is(Token::RParen) {
                            if paren == 0 {
                                break;
                            }
                            paren -= 1;
                        } else if paren == 0 {
                            if self.peek_is(Token::Less) {
                                angle += 1;
                            } else if self.peek_is(Token::Greater) {
                                if angle == 0 {
                                    break;
                                }
                                angle -= 1;
                            } else if self.peek_is(Token::Comma)
                                || self.peek_is(Token::RBrace)
                                || self.peek_is(Token::Semi)
                            {
                                break;
                            }
                        }
                        text.push_str(&self.current_lexeme());
                        text.push(' ');
                        self.advance();
                    }
                    let text = text.trim().to_string();
                    if !text.is_empty() {
                        value = Some(text);
                    }
                }
                names.push((name, value));
            }
            if self.peek_is(Token::Comma) {
                self.advance();
                continue;
            }
            break;
        }
        if self.peek_is(Token::RBrace) {
            self.advance();
        }
        Ok(names)
    }

    /// Evaluates an enumerator value to an integer: literals (decimal,
    /// hex, suffixed), unary minus, `| & ^ + - * / << >>` over values,
    /// parenthesized subexpressions, and identifiers from the same enum
    /// (earlier enumerators) or `int_consts`. `None` when omitted
    /// (caller applies auto-increment) or unevaluatable (caller keeps
    /// the value-less loud path).
    pub(crate) fn eval_enumerator_value(
        &self,
        text: Option<&str>,
        table: &std::collections::HashMap<String, i64>,
    ) -> Option<i64> {
        let text = text?;
        Self::eval_const_int_expr(text, table, &self.int_consts)
    }

    /// Constant-integer expression evaluator (subset): `| ^ & + - * /
    /// << >>` with C precedence, unary minus/plus/tilde/bang, parens,
    /// integer literals, and identifiers from `table`/`int_consts`.
    fn eval_const_int_expr(
        text: &str,
        table: &std::collections::HashMap<String, i64>,
        int_consts: &std::collections::HashMap<String, String>,
    ) -> Option<i64> {
        // Tokenize to numbers, identifiers, operators, parens.
        let mut toks: Vec<String> = Vec::new();
        let bytes = text.as_bytes();
        let mut i = 0;
        while i < text.len() {
            let c = bytes[i] as char;
            if c.is_whitespace() {
                i += 1;
                continue;
            }
            if c.is_ascii_alphanumeric() || c == '_' || c == '\'' {
                let mut j = i;
                while j < text.len() {
                    let d = bytes[j] as char;
                    if !(d.is_ascii_alphanumeric() || d == '_' || d == '\'') {
                        break;
                    }
                    j += 1;
                }
                toks.push(text[i..j].to_string());
                i = j;
                continue;
            }
            if c == '<' && bytes.get(i + 1) == Some(&b'<') {
                toks.push("<<".to_string());
                i += 2;
                continue;
            }
            if c == '>' && bytes.get(i + 1) == Some(&b'>') {
                toks.push(">>".to_string());
                i += 2;
                continue;
            }
            if "()|&^+ -*/~!".contains(c) {
                toks.push(c.to_string());
                i += 1;
                continue;
            }
            return None;
        }
        if toks.is_empty() {
            return None;
        }
        fn parse_int_literal(s: &str) -> Option<i64> {
            let t = s.trim_end_matches(['u', 'U', 'l', 'L']);
            if let Some(hex) = t.strip_prefix("0x").or_else(|| t.strip_prefix("0X")) {
                i64::from_str_radix(hex, 16).ok()
            } else {
                t.parse::<i64>().ok()
            }
        }
        struct P<'a> {
            toks: &'a [String],
            pos: usize,
            table: &'a std::collections::HashMap<String, i64>,
            int_consts: &'a std::collections::HashMap<String, String>,
        }
        impl P<'_> {
            fn peek(&self) -> Option<&str> {
                self.toks.get(self.pos).map(|s| s.as_str())
            }
            fn eat(&mut self, op: &str) -> bool {
                if self.peek() == Some(op) {
                    self.pos += 1;
                    true
                } else {
                    false
                }
            }
            fn expr(&mut self) -> Option<i64> {
                self.bit_or()
            }
            fn bit_or(&mut self) -> Option<i64> {
                let mut v = self.bit_xor()?;
                while self.eat("|") {
                    v |= self.bit_xor()?;
                }
                Some(v)
            }
            fn bit_xor(&mut self) -> Option<i64> {
                let mut v = self.bit_and()?;
                while self.eat("^") {
                    v ^= self.bit_and()?;
                }
                Some(v)
            }
            fn bit_and(&mut self) -> Option<i64> {
                let mut v = self.shift()?;
                while self.eat("&") {
                    v &= self.shift()?;
                }
                Some(v)
            }
            fn shift(&mut self) -> Option<i64> {
                let mut v = self.add()?;
                loop {
                    if self.eat("<<") {
                        v <<= self.add()?;
                    } else if self.eat(">>") {
                        v >>= self.add()?;
                    } else {
                        break;
                    }
                }
                Some(v)
            }
            fn add(&mut self) -> Option<i64> {
                let mut v = self.mul()?;
                loop {
                    if self.eat("+") {
                        v += self.mul()?;
                    } else if self.eat("-") {
                        v -= self.mul()?;
                    } else {
                        break;
                    }
                }
                Some(v)
            }
            fn mul(&mut self) -> Option<i64> {
                let mut v = self.unary()?;
                loop {
                    if self.eat("*") {
                        v *= self.unary()?;
                    } else if self.eat("/") {
                        let d = self.unary()?;
                        if d == 0 {
                            return None;
                        }
                        v /= d;
                    } else {
                        break;
                    }
                }
                Some(v)
            }
            fn unary(&mut self) -> Option<i64> {
                if self.eat("-") {
                    return Some(-self.unary()?);
                }
                if self.eat("+") {
                    return self.unary();
                }
                if self.eat("~") {
                    return Some(!self.unary()?);
                }
                if self.eat("!") {
                    return Some((self.unary()? == 0) as i64);
                }
                self.primary()
            }
            fn primary(&mut self) -> Option<i64> {
                let t = self.peek()?.to_string();
                if self.eat("(") {
                    let v = self.expr()?;
                    if !self.eat(")") {
                        return None;
                    }
                    return Some(v);
                }
                self.pos += 1;
                if let Some(v) = parse_int_literal(&t) {
                    return Some(v);
                }
                if let Some(v) = self.table.get(&t) {
                    return Some(*v);
                }
                if let Some(s) = self.int_consts.get(&t) {
                    return parse_int_literal(s);
                }
                None
            }
        }
        let mut p = P {
            toks: &toks,
            pos: 0,
            table,
            int_consts,
        };
        let v = p.expr()?;
        if p.pos != toks.len() {
            return None;
        }
        Some(v)
    }

    /// Parses an enum body (`{ A [= v], ... }`), recording each enumerator
    /// as an `int` global so uses (`opts & kOptX`) resolve. Evaluatable
    /// values (literals, auto-increment, same-enum references, constant
    /// arithmetic) additionally register in `module.constants`, so the
    /// global emits `static const int N = v;` — usable in `case` labels
    /// and other constant positions. Unevaluatable values keep today's
    /// value-less loud path (a guessed value would miscompile silently).
    /// Expects the cursor on `{`, consumes through `}`.
    pub(crate) fn parse_enum_body(&mut self, module: &mut Module) -> Result<()> {
        let mut seen = std::collections::HashSet::new();
        let mut table: std::collections::HashMap<String, i64> = std::collections::HashMap::new();
        let mut prev: i64 = -1;
        for (name, value_text) in self.parse_enumerator_names()? {
            if !seen.insert(name.clone()) {
                continue;
            }
            let evaluated = match value_text.as_deref() {
                Some(t) => Self::eval_const_int_expr(t, &table, &self.int_consts),
                None => Some(prev + 1),
            };
            if let Some(v) = evaluated {
                prev = v;
                table.insert(name.clone(), v);
            }
            let known = self.struct_names.contains(&name)
                || self.typedefs.contains_key(&name)
                || module.globals.iter().any(|g| g.name == name);
            if !known {
                // Range-typed backing: HLSL `static const int x = V`
                // miscompiles (with only a warning) when V exceeds int32
                // — DXC truncates the value to 0. `ulong`-backed framework
                // enums (`TraversalFlags`, `InvalidIntersection::kValue`
                // = 0xffffffff) routinely exceed it, so pick the
                // narrowest type that holds the value. Beyond uint32
                // there is no IR scalar (values that large are header
                // hints like `kScalarize*`, unreferenced by shader
                // logic); keep `Int` there — it warns but still emits.
                let ty = match evaluated {
                    Some(v) if v < 0 || v <= i32::MAX as i64 => {
                        crate::ast::Type::Int
                    }
                    Some(v) if v <= u32::MAX as i64 => crate::ast::Type::UInt,
                    _ => crate::ast::Type::Int,
                };
                module.globals.push(crate::ast::GlobalDecl {
                    ty,
                    name: name.clone(),
                    semantic: None,
                    array_size: None,
                    shared: false,
                    init: None,
                    span: crate::error::Span::default(),
                });
            }
            // Real constexprs win: never overwrite an existing constant.
            if evaluated.is_some() && !module.constants.contains_key(&name) {
                module.constants.insert(
                    name.clone(),
                    crate::ast::Expr::Number(evaluated.unwrap().to_string()),
                );
            }
            // Evaluated enumerators also feed integer-constant array sizes
            // (`RW_RegularBuffer<uint> bufs[k_tile_type_count]`): without
            // this the dimension is lost and flattened bindings decay to
            // scalars. Existing entries win (same rule as above).
            if evaluated.is_some() && !self.int_consts.contains_key(&name) {
                self.int_consts
                    .insert(name, evaluated.unwrap().to_string());
            }
        }
        Ok(())
    }

    pub(crate) fn parse_module(&mut self) -> Result<Module> {
        let mut module = Module::default();

        while !self.eof() {
            // Stray `;` (empty declaration, e.g. `};;`) — legal in C++;
            // skip it.
            if self.peek_is(Token::Semi) {
                self.advance();
                continue;
            }
            // Handle C++ `namespace foo { ... }` blocks.
            // All namespaces flatten (declarations land in the
            // module scope; uses keep their `Ns::` spelling and fold
            // at parse scope-resolution time). This includes the
            // `sce::Psr` ray-tracing framework: its enums (`TraversalFlags`,
            // `RayFlags`, ...), plain structs (`Ray`, `Hit`, ...), and
            // traversal prototypes (`Closest`, `Occluded`, ...) are real
            // PSSL-relevant declarations — skipping the namespace drops
            // them and orphans every use site (`sce`, `run`,
            // `FunctionShader` DXC failures). Unparseable C++ members
            // inside still fall back to the per-declaration skip paths
            // below, so headers degrade member-wise, not wholesale.
            if self.peek_is(Token::Identifier)
                && self.current_lexeme().eq_ignore_ascii_case("namespace")
            {
                self.advance(); // 'namespace'
                while !self.eof() && !self.peek_is(Token::LBrace) {
                    if self.peek_is(Token::Semi) {
                        break;
                    }
                    self.advance();
                }
                if self.peek_is(Token::LBrace) {
                    // Enter namespace – just consume `{` and parse inner
                    self.advance();
                }
                continue;
            }
            // Closing brace of a `SampleUtil` namespace – just consume and continue
            if self.peek_is(Token::RBrace) {
                // This handles `}` closing a previously entered SampleUtil namespace
                // where we didn't skip the block. Check if next tokens look like namespace close
                // (allow `}}}` for nested). Just consume one and continue.
                self.advance();
                // Also consume following `;` if present (some headers have `}};`)
                if self.peek_is(Token::Semi) {
                    self.advance();
                }
                continue;
            }

            // Skip C++ `template<class T>` prefix for generic helper functions
            // like `template<class T> T tex2D_bicubic(Texture2D<T> tex, ...)`
            // Consume the `template<...>` and then parse the following function
            // with `T` as a type variable (`Type::Named("T")`).
            if self.peek_is(Token::Identifier)
                && self.current_lexeme().eq_ignore_ascii_case("template")
            {
                tracing::trace!(target: "pssl_shader::parse::template", "skipping template<...> prefix");
                self.advance(); // 'template'
                if self.peek_is(Token::Less) {
                    let start = self.index + 1; // after '<'
                    let mut depth = 1;
                    self.advance(); // '<'
                    while !self.eof() && depth > 0 {
                        if self.peek_is(Token::Less) {
                            depth += 1;
                        } else if self.peek_is(Token::Greater) {
                            depth -= 1;
                        }
                        self.advance();
                    }
                    let end = self.index; // one past '>'
                    self.record_template_params(start, end);
                }
                continue;
            }

            // Skip C++ top-level `enum` / `enum class Name : type { ... }`
            // declarations (shader-irrelevant; PSSL enums carry no runtime value).
            // Exact lowercase match: a type named `Enum` must not take this path.
            if self.peek_is(Token::Identifier)
                && self.current_lexeme() == "enum"
            {
                self.advance(); // 'enum'
                if self.peek_is(Token::Identifier) {
                    let lex = self.current_lexeme().to_ascii_lowercase();
                    if lex == "class" || lex == "struct" {
                        self.advance();
                    }
                }
                if self.peek_is(Token::Identifier) {
                    // Record the enum name as a known type (enums appear in
                    // declarations, casts, and template params like
                    // `template <StackType stackType>`), and as an int
                    // typedef: PSSL enums are int-backed, so value and
                    // parameter positions (`OutputEncoding e`, casts)
                    // lower directly instead of leaking the name to HLSL.
                    // Also record in `enum_names`: a colliding `typedef`
                    // must never hijack a qualified `Ns::X` when `X` is a
                    // real struct (only enums take the typedef path).
                    self.struct_names.insert(self.current_lexeme());
                    self.enum_names.insert(self.current_lexeme());
                    self.typedefs.insert(
                        self.current_lexeme(),
                        crate::ast::Type::Int,
                    );
                    self.advance(); // enum name
                }
                if self.peek_is(Token::Colon) {
                    self.advance();
                    let _ = self.parse_type();
                }
                if self.peek_is(Token::LBrace) {
                    self.parse_enum_body(&mut module)?;
                    if self.peek_is(Token::Semi) {
                        self.advance();
                    }
                } else {
                    self.expect_semi()?;
                }
                continue;
            }

            // C++ `union [MACRO(args)] [Name] { ... };` at module scope —
            // members are only used for C++ type-punning; skip the body
            // like the `class` handler below does.
            // Exact lowercase match: a type named `Union` must not take this path.
            if self.peek_is(Token::Identifier)
                && self.current_lexeme() == "union"
            {
                self.advance(); // 'union'
                // Alignment/attribute macro between `union` and its name
                // (`union RT_ALIGNAS(64) TriangleFanNode {...}`).
                if self.peek_is(Token::Identifier)
                    && self.peek_next_is(Token::LParen)
                    && Self::is_macro_like_name(&self.current_lexeme())
                {
                    self.advance(); // macro
                    let _ = self.skip_paren_block();
                }
                if self.peek_is(Token::Identifier) {
                    // Record the union name as a known type so later uses
                    // (`Union u;`) parse as declarations; members stay
                    // opaque (type-punning) via the unknown-struct rules.
                    self.struct_names.insert(self.current_lexeme());
                    self.advance(); // name
                }
                // Template arguments after the name (`union X<A, B> {...}`,
                // e.g. partial specializations).
                if self.peek_is(Token::Less) {
                    let _ = self.skip_angle_block();
                }
                if self.peek_is(Token::LBrace) {
                    self.skip_brace_block()?;
                } else {
                    self.expect_semi()?;
                }
                continue;
            }

            // C++ `class Name { ... };` — shader headers use it for meshlet helpers.
            // Treat like struct but skip access specifiers (`public:` etc.) inside.
            // Exact lowercase match: a type named `Class` must not take this path.
            if self.peek_is(Token::Identifier)
                && self.current_lexeme() == "class"
            {
                self.advance(); // 'class'
                // Alignment/attribute macro between `class` and its name
                // (`class RT_ALIGNAS(64) Name {...}`).
                if self.peek_is(Token::Identifier)
                    && self.peek_next_is(Token::LParen)
                    && Self::is_macro_like_name(&self.current_lexeme())
                {
                    self.advance(); // macro
                    let _ = self.skip_paren_block();
                }
                // Optional template params `<...>` before name (unlikely but handle)
                if self.peek_is(Token::Less) {
                    let _ = self.skip_angle_block();
                }
                let mut class_name: Option<String> = None;
                if self.peek_is(Token::Identifier) {
                    class_name = Some(self.current_lexeme());
                    self.advance(); // name
                }
                // Template specialization after name: `Class<T>`
                if self.peek_is(Token::Less) {
                    let _ = self.skip_angle_block();
                }
                // Inheritance `: public Base, ...` until `{` or `;`
                if self.peek_is(Token::Colon) {
                    self.advance();
                    while !self.peek_is(Token::LBrace)
                        && !self.peek_is(Token::Semi)
                        && !self.eof()
                    {
                        if self.peek_is(Token::Less) {
                            let _ = self.skip_angle_block();
                        } else if self.peek_is(Token::LParen) {
                            let _ = self.skip_paren_block();
                        } else {
                            self.advance();
                        }
                    }
                }
                if self.peek_is(Token::LBrace) {
                    // Best-effort field extraction so out-of-class method
                    // bodies (`T Class::method() { ... mData ... }`) resolve
                    // bare members via scope injection. Methods and exotic
                    // members are skipped; on any confusion the whole body
                    // is skipped exactly as before.
                    let body_start = self.index;
                    let mut fields = Vec::new();
                    let mut ok = true;
                    self.advance(); // '{'
                    while !self.peek_is(Token::RBrace) && !self.eof() {
                        if !self.parse_class_member(&mut fields) {
                            ok = false;
                            break;
                        }
                    }
                    if !ok || !self.peek_is(Token::RBrace) {
                        // Fall back to skipping the body.
                        eprintln!("CLASSFALLBACK {:?} at {:?}", class_name, self.current_lexeme());
                        self.index = body_start;
                        self.skip_brace_block()?;
                    } else {
                        self.advance(); // '}'
                        if let Some(name) = class_name.clone() {
                            self.struct_names.insert(name.clone());
                            let start = body_start;
                            let end = self.previous_span().end;
                            module.structs.push(crate::ast::StructDecl {
                                name,
                                fields,
                                bases: Vec::new(),
                                span: crate::error::Span::new(start, end),
                                template_params: Vec::new(),
                            });
                        }
                    }
                } else {
                    self.expect_semi()?;
                }
                // Alignment macro after a dropped `class` body
                // (`} SCE_VECTORMATH_POSTALIGN_16;`): consume `MACRO ;`.
                if self.peek_is(Token::Identifier)
                    && matches!(
                        self.tokens.get(self.index + 1).map(|t| &t.token),
                        Some(Token::Semi)
                    )
                    && Self::is_macro_like_name(&self.current_lexeme())
                {
                    self.advance();
                    self.advance();
                }
                continue;
            }

            if self.peek_is(Token::Struct) {
                let decl = self.parse_struct()?;
                let struct_name = decl.name.clone();
                // Nested declarations collected while parsing (post-order:
                // innermost first) precede their parent.
                let mut nested = std::mem::take(&mut self.pending_structs);
                module.structs.append(&mut nested);
                module.structs.push(decl);
                // Member functions collected alongside (same drain point).
                let mut methods = std::mem::take(&mut self.pending_methods);
                module.functions.append(&mut methods);
                // Nested-enum enumerators register as int globals (same as
                // top-level enums): qualified uses (`S::E`) fold to bare
                // names in lowering; evaluatable values become constants
                // (usable in `case` labels). Each nested enum evaluates
                // with a fresh table (C++ auto-increment restarts per
                // enum); duplicates across enums keep the first.
                {
                    let mut seen = std::collections::HashSet::new();
                    for body in std::mem::take(&mut self.pending_enum_names) {
                        let mut table: std::collections::HashMap<String, i64> =
                            std::collections::HashMap::new();
                        let mut prev: i64 = -1;
                        for (name, value_text) in body {
                            if !seen.insert(name.clone()) {
                                continue;
                            }
                            let evaluated = match value_text.as_deref() {
                                Some(t) => Self::eval_const_int_expr(t, &table, &self.int_consts),
                                None => Some(prev + 1),
                            };
                            if let Some(v) = evaluated {
                                prev = v;
                                table.insert(name.clone(), v);
                            }
                            let known = self.struct_names.contains(&name)
                                || self.typedefs.contains_key(&name)
                                || module.globals.iter().any(|g| g.name == name);
                            if !known {
                                module.globals.push(crate::ast::GlobalDecl {
                                    ty: crate::ast::Type::Int,
                                    name: name.clone(),
                                    semantic: None,
                                    array_size: None,
                                    shared: false,
                                    init: None,
                                    span: crate::error::Span::default(),
                                });
                            }
                            if evaluated.is_some() && !module.constants.contains_key(&name) {
                                module.constants.insert(
                                    name,
                                    crate::ast::Expr::Number(evaluated.unwrap().to_string()),
                                );
                            }
                        }
                    }
                }
                // Alignment macro after the body (`} SCE_ALIGN_16;`):
                // consume `MACRO ;` instead of declaring an instance.
                if self.peek_is(Token::Identifier)
                    && matches!(
                        self.tokens.get(self.index + 1).map(|t| &t.token),
                        Some(Token::Semi)
                    )
                    && Self::is_macro_like_name(&self.current_lexeme())
                    && !self.is_known_type_name(&self.current_lexeme().clone())
                {
                    self.advance();
                    self.advance();
                    continue;
                }
                // Handle `struct Foo { ... } var : semantic;` immediate variable instance
                if self.peek_is(Token::Identifier) {
                    let is_instance = {
                        let after = self.tokens.get(self.index + 1).map(|t| t.token.clone());
                        // `::` after the name means a qualified definition
                        // (`S::S()`, `Ret S::m()`) — not an instance with a
                        // `: semantic` suffix.
                        let is_scope = matches!(after, Some(Token::Colon))
                            && self.tokens.get(self.index + 2).map(|t| t.token == Token::Colon).unwrap_or(false);
                        !is_scope && (matches!(
                            after,
                            Some(Token::Colon) | Some(Token::Semi) | Some(Token::LBracket) | Some(Token::Equal) | Some(Token::Comma)
                        ) || matches!(self.tokens.get(self.index + 1).map(|t| &t.token), Some(Token::Identifier) if self.tokens.get(self.index + 2).map(|t| t.token == Token::Comma).unwrap_or(false)))
                    };
                    if is_instance {
                        let var_name = self.expect_identifier()?;
                        let mut array_size = None;
                        while self.peek_is(Token::LBracket) {
                            self.advance();
                            if !self.peek_is(Token::RBracket) {
                                let _ = self.parse_assignment();
                            }
                            self.expect(Token::RBracket)?;
                        }
                        let semantic = self.parse_optional_semantic()?;
                        self.expect_semi()?;
                        let span = Span::new(0, 0);
                        module.globals.push(GlobalDecl {
                            ty: Type::Named(struct_name),
                            name: var_name,
                            semantic,
                            array_size,
                            shared: false,
                            init: None,
                            span,
                        });
                        continue;
                    }
                }
                continue;
            }

            // C linkage markers (`__BEGIN_DECLS` / `__END_DECLS` from
            // sys/cdefs.h, usually `extern "C" {` / `}`) — not PSSL; skip.
            if self.peek_is(Token::Identifier)
                && (self.current_lexeme() == "__BEGIN_DECLS"
                    || self.current_lexeme() == "__END_DECLS")
            {
                self.advance();
                if self.peek_is(Token::Semi) {
                    self.advance();
                }
                continue;
            }

            // C++ `typedef` type aliases (e.g. `typedef float2 Vector2Unaligned;`)
            // from inlined framework headers. Record the alias and consume the decl.
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
                // C-style `typedef enum Name { ... } Alias;` (host C headers):
                // parse the enumerators, then alias to `int`. Only when an
                // enum body (or `Name {`) actually follows — a type merely
                // named `Enum` falls through to the normal typedef path.
                if self.peek_is(Token::Identifier) && self.current_lexeme() == "enum" {
                    let mut j = self.index + 1;
                    if matches!(
                        self.tokens.get(j).map(|t| &t.token),
                        Some(Token::Identifier)
                    ) {
                        j += 1;
                    }
                    let is_enum_decl = matches!(
                        self.tokens.get(j).map(|t| &t.token),
                        Some(Token::LBrace)
                    );
                    if is_enum_decl {
                        self.advance(); // 'enum'
                        if self.peek_is(Token::Identifier) {
                            self.struct_names.insert(self.current_lexeme());
                            self.advance(); // enum name
                        }
                        if self.peek_is(Token::LBrace) {
                            self.parse_enum_body(&mut module)?;
                        }
                        // Trailing alias(es) (`} SceDbgLogLevel;`).
                        loop {
                            while self.peek_is(Token::Star) || self.peek_is(Token::Ampersand) {
                                self.advance();
                            }
                            if !self.peek_is(Token::Identifier) {
                                break;
                            }
                            let alias = self.current_lexeme();
                            self.advance();
                            self.typedefs.insert(
                                alias,
                                crate::ast::Type::Int,
                            );
                            if self.peek_is(Token::Comma) {
                                self.advance();
                                continue;
                            }
                            break;
                        }
                        self.expect_semi()?;
                        continue;
                    }
                }
                // C-style `typedef struct Name { ... } Alias;` (host C
                // headers): skip the body (host types), alias to an opaque
                // struct name so later uses parse and flow opaquely.
                // Same for `typedef union ...` (type-punning helpers).
                if self.peek_is(Token::Struct)
                    || (self.peek_is(Token::Identifier)
                        && self.current_lexeme() == "union")
                {
                    let mut j = self.index + 1;
                    let mut j = self.index + 1;
                    let mut struct_name: Option<String> = None;
                    if matches!(
                        self.tokens.get(j).map(|t| &t.token),
                        Some(Token::Identifier)
                    ) {
                        struct_name = self.tokens.get(j).map(|t| t.lexeme.clone());
                        j += 1;
                    }
                    if matches!(
                        self.tokens.get(j).map(|t| &t.token),
                        Some(Token::LBrace)
                    ) {
                        self.advance(); // 'struct'
                        if let Some(name) = struct_name.clone() {
                            self.struct_names.insert(name.clone());
                            self.advance(); // struct name
                        }
                        self.skip_brace_block()?;
                        loop {
                            while self.peek_is(Token::Star) || self.peek_is(Token::Ampersand) {
                                self.advance();
                            }
                            if !self.peek_is(Token::Identifier) {
                                break;
                            }
                            let alias = self.current_lexeme();
                            self.advance();
                            self.typedefs.insert(
                                alias.clone(),
                                crate::ast::Type::Named(
                                    struct_name.clone().unwrap_or(alias),
                                ),
                            );
                            if self.peek_is(Token::Comma) {
                                self.advance();
                                continue;
                            }
                            break;
                        }
                        self.expect_semi()?;
                        continue;
                    }
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
                // Calling-convention macros between type and alias
                // (`typedef const floatInVec SCE_..._ARG floatInVec_arg;`).
                self.skip_interposed_macro();
                loop {
                    // Pointer/reference declarators (`typedef T* P;`,
                    // `typedef T& R;`) — dropped; the alias keeps the base.
                    while self.peek_is(Token::Star) || self.peek_is(Token::Ampersand) {
                        self.advance();
                    }
                    // The alias may re-spell a keyword scalar
                    // (`typedef unsigned int uint;`).
                    let name = self.expect_alias_name()?;
                    // Array declarators (`typedef float x[4];`) — dropped.
                    while self.peek_is(Token::LBracket) {
                        self.advance();
                        while !self.peek_is(Token::RBracket) && !self.eof() {
                            self.advance();
                        }
                        self.expect(Token::RBracket)?;
                    }
                    self.typedefs.insert(name, base_ty.clone());
                    if self.peek_is(Token::Comma) {
                        self.advance();
                    } else {
                        break;
                    }
                }
                self.expect_semi()?;
                continue;
            }

            // C++ `CXX_SYMBOL("...")` / `CxxSymbol("...")` embedding marker — not PSSL;
            // capture the symbol for the next function (it carries the stage
            // hint, e.g. `CXX_SYMBOL("...shader_vv")`), then consume the
            // parenthesised argument and optional trailing `;`.
            if self.peek_is(Token::Identifier)
                && self.current_lexeme().to_ascii_lowercase() == "cxx_symbol"
            {
                self.advance();
                if self.peek_is(Token::LParen) {
                    self.advance();
                    if self.peek_is(Token::String) {
                        self.pending_cxx_symbol =
                            Some(types::unquote(&self.current_lexeme()));
                        self.advance();
                    }
                    while !self.peek_is(Token::RParen) && !self.eof() {
                        if self.peek_is(Token::LParen) {
                            if self.skip_paren_block().is_err() {
                                break;
                            }
                        } else {
                            self.advance();
                        }
                    }
                    if self.peek_is(Token::RParen) {
                        self.advance();
                    }
                }
                if self.peek_is(Token::Semi) {
                    self.advance();
                }
                continue;
            }

            // C++ `using namespace X;` / `using X = Y;` directives from inlined
            // headers — not PSSL; consume through the terminating `;`.
            if self.peek_is(Token::Identifier)
                && self.current_lexeme().eq_ignore_ascii_case("using")
            {
                while !self.eof() && !self.peek_is(Token::Semi) {
                    self.advance();
                }
                self.expect_semi()?;
                continue;
            }

            // Generic module-scope macro invocation like
            // `SCE_PSR_WITH_TRINITY_SUPPORT([argument(target=trinity)])`.
            // At module scope a function is `Type Name(` (two tokens before '(');
            // a bare `IDENT(` is almost always a macro, not a declaration.
            // Exception: SRT buffer macros `REGULAR_BUFFER(T) name;` declare
            // a resource — desugar them instead of skipping.
            if self.peek_is(Token::Identifier) && self.peek_next_is(Token::LParen) {
                if types::is_resource_kind(&self.current_lexeme()) {
                    // Scan to the matching ')' : a declaration has an
                    // identifier (the resource name) right after it, while a
                    // plain call has `;` (or end) after it.
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
                    let is_decl = matches!(
                        rparen.and_then(|k| self.tokens.get(k + 1)),
                        Some(t) if t.token == Token::Identifier
                    );
                    if is_decl {
                        let start = self.current_span().start;
                        if let Some(Type::Resource { kind, element, .. }) =
                            self.try_parse_macro_resource_type()?
                        {
                            let name = self.expect_identifier()?;
                            let mut register = None;
                            if self.peek_is(Token::Colon) {
                                self.advance();
                                let reg_word = self.expect_identifier()?;
                                if reg_word.eq_ignore_ascii_case("register") {
                                    self.expect(Token::LParen)?;
                                    let reg_str = self.expect_identifier()?;
                                    if let Some((c, n)) =
                                        types::parse_register_str(&reg_str)
                                    {
                                        register = Some((c, n));
                                    }
                                    self.expect(Token::RParen)?;
                                }
                            }
                            let semi =
                                self.current().map(|t| t.span.end).unwrap_or(start);
                            self.expect_semi()?;
                            module.resources.push(ResourceDecl {
                                name,
                                kind,
                                element_ty: Some(*element),
                                register,
                                register_space: None,
                                semantic: None,
                                fields: Vec::new(),
                                span: Span::new(start, semi),
                            });
                            continue;
                        }
                    }
                }
                self.advance(); // macro name
                self.skip_module_macro_rest()?;
                continue;
            }

            // Stray parenthesized expression at module scope, e.g. an
            // expanded config/assertion macro fragment whose head was
            // consumed elsewhere (`((lanes == 64 && ...) || ...)` in
            // blendshape_decoder). No legitimate module item starts with
            // `(`, so skip it with no effect.
            if self.peek_is(Token::LParen) {
                self.skip_paren_block()?;
                if self.peek_is(Token::Semi) {
                    self.advance();
                }
                continue;
            }

            // Unknown ALL-CAPS prefix macro on a declaration
            // (`SCE_VECTORMATH_ALWAYS_INLINE vec_ullong1 f(...)`,
            // `SCE_... boolInVec::boolInVec(...)`): a macro-like unknown
            // identifier followed by a type means `MACRO Type ...` — drop
            // the macro and re-process the declaration. Without this the
            // macro parses as the return type and everything downstream
            // misaligns.
            if self.peek_is(Token::Identifier)
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
                            | Token::Const
                            | Token::Static
                    )
                )
                && matches!(
                    self.tokens.get(self.index + 2).map(|t| &t.token),
                    Some(
                        Token::Identifier
                            | Token::Star
                            | Token::Ampersand
                            | Token::Colon
                    )
                )
            {
                let first = self.current_lexeme();
                if Self::is_macro_like_name(&first) && !self.is_known_type_name(&first) {
                    tracing::trace!(
                        target: "pssl_shader::parse::template",
                        macro_name = %first,
                        "dropping unknown prefix macro",
                    );
                    self.advance(); // drop the macro qualifier
                    continue;
                }
            }

            let mut attributes = self.parse_attributes()?;
            // An attribute block can precede a C++ aggregate declaration
            // (`[ThisAsPointer] class PCG32RNG { ... }`, `[x] union U {...}`,
            // `[x] enum E {...}`, `[x] namespace ns {...}`,
            // `[ThisAsPointer] template <> struct iBVH<false> {...}`). The
            // loop-top handlers re-dispatch these only when the keyword
            // starts a loop iteration; re-enter the loop so the current
            // keyword is handled there (attributes were already consumed
            // above and are shader-irrelevant hints). Note `struct` lexes
            // as its own keyword token, unlike class/union/enum.
            if self.peek_is(Token::Struct) {
                continue;
            }
            if self.peek_is(Token::Identifier)
                && (self.current_lexeme() == "class"
                    || self.current_lexeme() == "union"
                    || self.current_lexeme() == "enum"
                    || self.current_lexeme().eq_ignore_ascii_case("namespace"))
            {
                continue;
            }
            // Bare macro invocation after attributes
            // (`[profile(..)] SCE_MACRO(..)`): same skip as below.
            if self.peek_is(Token::Identifier) && self.peek_next_is(Token::LParen) {
                self.advance(); // macro name
                self.skip_module_macro_rest()?;
                continue;
            }
            // Handle `template <...>` that may appear after stray attribute blocks from macro defines (e.g. ATTR)
            if self.peek_is(Token::Identifier) && self.current_lexeme().eq_ignore_ascii_case("template") {
                self.advance();
                if self.peek_is(Token::Less) {
                    let start = self.index + 1; // after '<'
                    let mut depth = 1;
                    self.advance();
                    while !self.eof() && depth > 0 {
                        if self.peek_is(Token::Less) {
                            depth += 1;
                        } else if self.peek_is(Token::Greater) {
                            depth -= 1;
                        }
                        self.advance();
                    }
                    let end = self.index;
                    // Same param recording as the loop-top handler: without
                    // it `template`s after attributes (e.g. following
                    // `[call_site_debugloc]`) lose their type/value params.
                    self.record_template_params(start, end);
                }
                // An aggregate keyword after the template head (`template <>`
                // `struct iBVH<false> {...}` explicit specializations) is a
                // type declaration, not a function return type: re-enter
                // the loop so the struct/class/union/enum handlers own it.
                // Anything else (e.g. `template<typename T> T foo()`)
                // falls through to return_type parsing below as before.
                if self.peek_is(Token::Struct)
                    || (self.peek_is(Token::Identifier)
                        && (self.current_lexeme() == "class"
                            || self.current_lexeme() == "union"
                            || self.current_lexeme() == "enum"))
                {
                    continue;
                }
                // Merge with any already-parsed attributes and continue to next item – the `template` itself is not a function
                // The next token should be return_type (`void` etc.), so we need to re-enter loop without consuming it as attributes for a different function.
                // Instead of `continue`, we want to parse the following function in same iteration.
                // So we keep `attributes` as is and fall through to return_type parsing, but we have already consumed `template<...>` so the next token is return_type.
                // No need to `continue` – just proceed.
            }
            // Handle storage qualifiers that can precede resources/globals/
            // functions: `static`, `const`, and `constexpr` (lexed as an
            // identifier since there is no dedicated keyword), plus
            // linkage/storage spellings (`extern`, `uniform`, `volatile`,
            // `mutable`, `shared`, `precise`). These must be consumed here
            // (before the type) so e.g. `extern const uint32_t k[...];`
            // does not strand `const` for `parse_type`.
            let mut _global_storage = None;
            let mut _storage_count = 0;
            // Per-workgroup LDS (`thread_group_memory` / `groupshared` /
            // `shared`): recorded on `GlobalDecl::shared` so HLSL emission
            // can later use `groupshared` instead of a read-only global.
            let mut is_shared = false;
            while _storage_count < 5 {
                if self.peek_is(Token::Static) || self.peek_is(Token::Const) {
                    _global_storage = Some(StorageClass::Const);
                    self.advance();
                } else if self.peek_is(Token::Identifier) {
                    // A `::` scope qualifier (`Shared::SRT`) is a type
                    // name, not a storage modifier — never consume it.
                    let is_scope = self.peek_next_is(Token::Colon)
                        && self
                            .tokens
                            .get(self.index + 2)
                            .map(|t| t.token == Token::Colon)
                            .unwrap_or(false);
                    if is_scope {
                        break;
                    }
                    let lex = self.current_lexeme();
                    if lex.eq_ignore_ascii_case("constexpr")
                        || lex.eq_ignore_ascii_case("const")
                    || lex.eq_ignore_ascii_case("inline")
                        || lex.eq_ignore_ascii_case("thread_group_memory")
                        || lex.eq_ignore_ascii_case("groupshared")
                        || lex.eq_ignore_ascii_case("groupShared")
                        || lex.eq_ignore_ascii_case("extern")
                        || lex.eq_ignore_ascii_case("uniform")
                        || lex.eq_ignore_ascii_case("volatile")
                        || lex.eq_ignore_ascii_case("mutable")
                        || lex.eq_ignore_ascii_case("shared")
                        || lex.eq_ignore_ascii_case("precise")
                    {
                        if lex.eq_ignore_ascii_case("thread_group_memory")
                            || lex.eq_ignore_ascii_case("groupshared")
                            || lex.eq_ignore_ascii_case("shared")
                        {
                            is_shared = true;
                        }
                        _global_storage = Some(StorageClass::Const);
                        self.advance();
                    } else if Self::is_macro_like_name(&lex)
                        && !self.is_known_type_name(&lex)
                        && !matches!(
                            self.tokens.get(self.index + 1).map(|t| &t.token),
                            Some(Token::LParen) | Some(Token::Semi) | None
                        )
                        && !matches!(
                            self.tokens.get(self.index + 2).map(|t| &t.token),
                            Some(Token::Semi)
                                | Some(Token::Comma)
                                | Some(Token::RParen)
                                | Some(Token::RBrace)
                                | Some(Token::Equal)
                                | None
                        )
                    {
                        // Calling-convention / annotation macros before the
                        // type (`SCE_VECTORMATH_INLINE const floatInVec
                        // distSqr(...)`): not invocations (`(`) or statements
                        // (`;`), and not a lone `TYPE name;` pair (the token
                        // after the name is a terminator there), so they
                        // qualify the declaration.
                        eprintln!("MACROSKIP {:?}", lex);
                        self.advance();
                    } else {
                        break;
                    }
                } else {
                    break;
                }
                _storage_count += 1;
            }
            // Out-of-class operator definition without a return type:
            // `Scope::operator <op-or-type> (params) [body|;]`
            // (e.g. `boolInVec::operator bool() const`). Skipped like all
            // operators (never entry points).
            if self.peek_is(Token::Identifier)
                && self.tokens.get(self.index + 1).map(|t| t.token == Token::Colon).unwrap_or(false)
                && self.tokens.get(self.index + 2).map(|t| t.token == Token::Colon).unwrap_or(false)
                && matches!(self.tokens.get(self.index + 3), Some(t) if t.token == Token::Identifier && t.lexeme == "operator")
            {
                // Consume `Scope :: operator`.
                self.advance();
                self.advance();
                self.advance();
                self.advance();
                self.skip_operator_target()?;
                self.skip_operator_signature()?;
                continue;
            }
            if let Some(_kind) = self.peek_resource_kind() {
                // A function returning a resource type (e.g.
                // `static constexpr SamplerState MakeClampLinearSampler() {...}`)
                // matches `Kind Name (` — resources never have `(` directly
                // after the name, so route those to function parsing below.
                // The kind may carry a template list (`Kind<A, B> Name (`),
                // e.g. `RW_RegularBuffer<T, cacheFlags> makeRwRegularBuffer(...)`.
                let mut j = self.index + 1;
                if matches!(self.tokens.get(j).map(|t| &t.token), Some(Token::Less)) {
                    let mut depth = 0usize;
                    let mut k = j;
                    while k < self.tokens.len() {
                        if self.tokens[k].token == Token::Less {
                            depth += 1;
                        } else if self.tokens[k].token == Token::Greater {
                            depth = depth.saturating_sub(1);
                            if depth == 0 {
                                j = k + 1;
                                break;
                            }
                        } else if self.tokens[k].token == Token::ShiftRight {
                            if depth <= 2 {
                                j = k + 1;
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
                        }
                        k += 1;
                        if k - (self.index + 1) > 200 {
                            break; // give up: treat as a resource below
                        }
                    }
                }
                let is_function = matches!(
                    self.tokens.get(j).map(|t| &t.token),
                    Some(Token::Identifier)
                ) && matches!(
                    self.tokens.get(j + 1).map(|t| &t.token),
                    Some(Token::LParen)
                );
                if !is_function {
                    module.resources.push(self.parse_resource(attributes)?);
                    continue;
                }
                // else fall through to return_type/name/function parsing
            }
            // `static_assert(...)` / `assert(...)` at module scope — skip.
            if self.peek_is(Token::Identifier)
                && matches!(
                    self.current_lexeme().to_ascii_lowercase().as_str(),
                    "static_assert" | "assert"
                )
            {
                self.advance(); // assert name
                self.skip_paren_block()?;
                self.expect_semi()?;
                continue;
            }
            // Framework SRT signature blocks: `SrtSignature Name { DescriptorTable(...) }`
            if self.peek_is(Token::Identifier)
                && matches!(
                    self.current_lexeme().to_ascii_lowercase().as_str(),
                    "srtsignature" | "globalsrtsignature"
                )
            {
                self.advance();
                let _name = self.expect_identifier()?;
                if self.peek_is(Token::LBrace) {
                    self.skip_brace_block()?;
                } else {
                    self.expect_semi()?;
                }
                continue;
            }
            self.skip_interp_modifiers();
            let return_type = self.parse_type()?;
            // GNU/MSVC attributes between return type and name
            // (`int __attribute__((...)) module_start(...)`).
            loop {
                if self.peek_is(Token::Identifier)
                    && (self.current_lexeme() == "__attribute__"
                        || self.current_lexeme() == "__declspec")
                    && self.peek_next_is(Token::LParen)
                {
                    self.advance();
                    self.skip_paren_block()?;
                    continue;
                }
                break;
            }
            // Handle `*`/`&`/`const`/`constexpr` suffix after type for `T const *` etc.
            loop {
                if self.peek_is(Token::Const) {
                    self.advance();
                    continue;
                }
                if self.peek_is(Token::Identifier) && {
                    let l = self.current_lexeme().to_ascii_lowercase();
                    l == "const" || l == "constexpr"
                } {
                    self.advance();
                    continue;
                }
                if self.peek_is(Token::Star) || self.peek_is(Token::Ampersand) {
                    self.advance();
                    continue;
                }
                break;
            }
            // Constructor / destructor definition without a return type:
            // `Scope::Scope(...) { ... }` was consumed as a scoped
            // `return_type` and the cursor is on `(`. Model constructors as
            // void functions; skip destructors.
            if self.peek_is(Token::LParen) {
                if let Type::Named(scoped) = &return_type {
                    if scoped.contains("::") {
                        let mut parts: Vec<&str> = scoped.split("::").collect();
                        let method = parts.pop().unwrap_or("").to_string();
                        let scope_last = parts.pop().unwrap_or("").to_string();
                        let is_dtor = method.starts_with('~')
                            || scope_last.starts_with('~')
                            || method == format!("~{scope_last}");
                        if is_dtor {
                            self.skip_paren_block()?;
                            if self.peek_is(Token::LBrace) {
                                self.skip_brace_block()?;
                            } else {
                                self.expect_semi()?;
                            }
                            continue;
                        }
                        let ret = if method == scope_last {
                            Type::Void
                        } else {
                            Type::Void
                        };
                        let mut func = self.parse_function(
                            attributes,
                            ret,
                            method.trim_start_matches('~').to_string(),
                        )?;
                        // Scope is everything but the method name.
                        let mut scope_segs: Vec<&str> = parts.clone();
                        scope_segs.push(&scope_last);
                        if !scope_segs.is_empty() {
                            func.scope = Some(scope_segs.join("::"));
                        }
                        module.functions.push(func);
                        continue;
                    }
                }
            }
            let mut name = self.expect_identifier()?;
            // C++ out-of-class definitions `Ret Scope::method(...) { ... }`
            // (e.g. `uint4 MeshletSkinData::uncompressBoneIndices(...)` in
            // clustered-geometry shaders). Treat as a free function `method`.
            if self.peek_is(Token::Colon)
                && self.tokens.get(self.index + 1).map(|t| t.token == Token::Colon).unwrap_or(false)
            {
                let mut func_name = name.clone();
                let mut scope_parts = vec![name.clone()];
                let mut is_special = false;
                while self.peek_is(Token::Colon)
                    && self.tokens.get(self.index + 1).map(|t| t.token == Token::Colon).unwrap_or(false)
                {
                    self.advance(); // ':'
                    self.advance(); // ':'
                    // Destructor `Scope::~Class(` or `operator==` overloads:
                    // skip the body; they carry no entry-point relevance.
                    if self.peek_is(Token::Tilde) {
                        is_special = true;
                        self.advance();
                    }
                    if self.peek_is(Token::Identifier)
                        && self.current_lexeme().eq_ignore_ascii_case("operator")
                    {
                        is_special = true;
                        self.advance(); // 'operator'
                        // Consume the operator symbol(s): `==`, `[]`, `()`,
                        // or a conversion target type (`operator bool`).
                        if self.peek_is(Token::LParen) {
                            // `operator()` — consume the parens as part of the name.
                            self.advance();
                            if self.peek_is(Token::RParen) {
                                self.advance();
                            }
                        } else if self.peek_is(Token::LBracket) {
                            self.advance();
                            self.expect(Token::RBracket)?;
                        } else if self.peek_is(Token::Identifier)
                            || matches!(
                                self.peek(),
                                Some(
                                    Token::Void
                                        | Token::Bool
                                        | Token::Int
                                        | Token::UInt
                                        | Token::Float
                                        | Token::Half
                                        | Token::Double
                                )
                            )
                        {
                            // Conversion operator (`::operator bool(`) — the
                            // type is irrelevant without a name; discard it.
                            let _ = self.parse_type();
                        } else if !self.eof() {
                            self.advance();
                        }
                        func_name = "operator".to_string();
                        continue;
                    }
                    if self.peek_is(Token::Identifier) {
                        func_name = self.current_lexeme();
                        scope_parts.push(func_name.clone());
                        self.advance();
                    } else {
                        break;
                    }
                }
                // Optional template args after the method name: `method<T>(...)`.
                if self.peek_is(Token::Less) {
                    let _ = self.skip_angle_block();
                }
                if self.peek_is(Token::LParen) && !is_special {
                    let mut func = self.parse_function(
                        attributes,
                        return_type,
                        func_name,
                    )?;
                    // Scope is everything but the method name.
                    if scope_parts.len() > 1 {
                        func.scope = Some(
                            scope_parts[..scope_parts.len() - 1].join("::"),
                        );
                    }
                    module.functions.push(func);
                    continue;
                } else if self.peek_is(Token::LParen) {
                    // Destructor / operator: consume signature + body/semicolon.
                    self.skip_paren_block()?;
                    // Trailing `const` etc. (`C::operator bool() const`).
                    self.skip_trailing_method_qualifiers()?;
                    if self.peek_is(Token::LBrace) {
                        self.skip_brace_block()?;
                    } else {
                        self.expect_semi()?;
                    }
                    continue;
                }
                // Otherwise it is a qualified static data member
                // (`Type Scope::var ...`); model as a global `var`.
                name = func_name;
            }

            // Template argument list after a function name
            // (`template<> Ret Name<0>(...)` explicit specializations).
            // A `<` here can only open template args (comparisons never
            // appear in this position).
            if self.peek_is(Token::Less) {
                let _ = self.skip_angle_block();
            }

            // C++ operator overloads (`Ret operator * (...);`,
            // `operator==`, `operator[]`, conversion `operator T(...)`):
            // operators are never entry points; uses (`a * b`) parse as
            // ordinary binary expressions. Skip the whole declaration.
            // (`operator` arrives as `name` since it lexes as an
            // identifier — or right after it when a macro-prefixed return
            // like `SCE_... const floatInVec` precedes it.)
            let sees_operator = name == "operator"
                || (self.peek_is(Token::Identifier) && self.current_lexeme() == "operator");
            if sees_operator {
                if name != "operator" {
                    self.advance(); // consume 'operator' after the name
                }
                if self.peek_is(Token::LParen) {
                    self.skip_paren_block()?; // `operator()`
                } else if self.peek_is(Token::LBracket) {
                    self.advance(); // `operator[`
                    self.expect(Token::RBracket)?; // `]`
                } else if self.peek_is(Token::Identifier)
                    && matches!(
                        self.current_lexeme().to_ascii_lowercase().as_str(),
                        "new" | "delete"
                    )
                {
                    self.advance(); // `operator new` / `operator delete`
                    if self.peek_is(Token::LBracket) {
                        self.advance();
                        self.expect(Token::RBracket)?; // `new[]`
                    }
                } else if self.peek_is(Token::Identifier)
                    || matches!(
                        self.peek(),
                        Some(
                            Token::Void
                                | Token::Bool
                                | Token::Int
                                | Token::UInt
                                | Token::Float
                                | Token::Half
                                | Token::Double
                        )
                    )
                {
                    // Conversion operator (`operator float4(...)`) — the
                    // type is irrelevant without a name; discard it.
                    let _ = self.parse_type();
                } else if !self.peek_is(Token::LParen) {
                    // Any symbolic operator (`*`, `==`, `<<`, ...): one token
                    // (multi-char ops lex as single tokens).
                    self.advance();
                }
                if self.peek_is(Token::LParen) {
                    self.skip_paren_block()?;
                    if self.peek_is(Token::LBrace) {
                        self.skip_brace_block()?;
                    } else {
                        self.expect_semi()?;
                    }
                } else {
                    // No parameter list — skip to `;` defensively.
                    while !self.peek_is(Token::Semi) && !self.eof() {
                        if self.peek_is(Token::LBrace) {
                            self.skip_brace_block()?;
                        } else {
                            self.advance();
                        }
                    }
                    self.expect_semi()?;
                }
                continue;
            }

            // `Type name(args);` is a paren-initialized variable (e.g.
            // `static constexpr uint2 k(8, 8);`), not a function, when the
            // parenthesised list cannot be a parameter list.
            let is_paren_init =
                self.peek_is(Token::LParen) && self.looks_like_paren_init();
            if self.peek_is(Token::LParen) && !is_paren_init {
                module
                    .functions
                    .push(self.parse_function(attributes, return_type, name)?);
            } else {
                // Comma-separated declarators sharing one type/storage
                // (`uint a, b = 1, c[4];`, `thread_group_memory uint x, y;`).
                // The first name is already parsed; later ones follow commas.
                let mut pending_name = Some(name);
                let mut pending_paren_init = is_paren_init;
                // Paren-init values survive here (`static constexpr uint2
                // G(8, 8)` keeps `uint2(8, 8)` exactly like `= uint2(8, 8)`
                // would), so constant uses (`G.x` in NUM_THREADS, array
                // sizes) resolve downstream. Reset per declarator.
                let mut pending_const_init: Option<Expr> = None;
                loop {
                    let name = pending_name.take().expect("declarator name");
                    if pending_paren_init {
                        match self.try_parse_paren_init(&return_type, true) {
                            Some(ctor) => pending_const_init = Some(ctor),
                            None => self.skip_paren_block()?,
                        }
                    }
                    let start = self.previous_span().start;
                let mut array_size = None;
                let mut is_array = false;
                // Multi-dimensional arrays `name[64][2]` — record the first
                // dimension only (matches function-body handling), but
                // collect every dimension so exactly-2D all-constant
                // shapes can flatten to product size below.
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
                        if first_dim {
                            array_size = Some(0);
                        }
                        all_dims.push(None);
                    }
                    self.expect(Token::RBracket)?;
                    first_dim = false;
                }
                if is_array && array_size.is_none() {
                    array_size = Some(0);
                }
                // Exactly-2D all-constant shapes flatten to product size
                // (uses linearize inline at index-parse time).
                if let Some(product) = self.note_multidim(&name, &all_dims, false) {
                    array_size = Some(product);
                }
                let mut const_value: Option<Expr> = pending_const_init.take();
                // Retain any `= init` expression for lowering (alias-style
                // globals resolve from it; other inits keep today's
                // ignored-at-lowering behavior). Starts as any paren-init.
                let mut global_init: Option<Expr> = const_value.clone();
                if self.peek_is(Token::Equal) {
                    self.advance();
                    if self.peek_is(Token::LBrace) {
                        // Braced initializer (`static const float3 k[] = { ... }`
                        // or sized `float h[4] = { ... }`). Skip the list but
                        // count top-level elements so an unsized array global
                        // gains an explicit dimension (`distinctColors[]` with
                        // 267 entries → `distinctColors[267]`), which DXC
                        // requires (`array type needs an explicit size`).
                        let mut depth = 0usize;
                        let mut paren = 0usize;
                        let mut elems = 0usize;
                        while !self.eof() {
                            if self.peek_is(Token::LBrace) {
                                depth += 1;
                                if depth == 1 {
                                    elems += 1;
                                }
                                self.advance();
                            } else if self.peek_is(Token::RBrace) {
                                depth -= 1;
                                self.advance();
                                if depth == 0 {
                                    break;
                                }
                            } else if self.peek_is(Token::LParen) {
                                paren += 1;
                                self.advance();
                            } else if self.peek_is(Token::RParen) {
                                if paren > 0 {
                                    paren -= 1;
                                }
                                self.advance();
                            } else if self.peek_is(Token::Comma)
                                && depth == 1
                                && paren == 0
                            {
                                // A trailing comma before `}` is not a
                                // separator (`{ a, b, }` has 2 elements).
                                let next_is_rbrace = matches!(
                                    self.tokens.get(self.index + 1).map(|t| &t.token),
                                    Some(Token::RBrace)
                                );
                                if !next_is_rbrace {
                                    elems += 1;
                                }
                                self.advance();
                            } else {
                                self.advance();
                            }
                        }
                        if is_array && (array_size.is_none() || array_size == Some(0)) && elems > 0 {
                            array_size = Some(elems as u32);
                        }
                    } else {                        let init = self.parse_assignment()?;
                        global_init = Some(init.clone());
                        // Record constant initializers so attributes like
                        // `[NUM_THREADS(k_wavefront_size.x, ...)]` and globals like
                        // `static const short2 k_wavefront_size = ushort2(4,16)` preserve their initializer.
                        let is_const_init = matches!(
                            &init,
                            Expr::Number(_)
                                | Expr::Bool(_)
                                | Expr::Variable(_)
                                | Expr::Constructor { .. }
                                | Expr::Binary { .. }
                                | Expr::Unary { .. }
                        );
                        if is_const_init {
                            if let Some(v) = self.resolve_array_size(&init) {
                                self.int_consts.insert(name.clone(), v.to_string());
                            }
                            const_value = Some(init.clone());
                            // Keep constructor constants (e.g. ushort2(4,16)) for global emission as well.
                            // For bare numeric constants the above insert already handles NUM_THREADS.
                            // For constructors we keep the full init so `static const short2 k = ushort2(4,16)` emits.
                            if let Expr::Constructor { .. } = &init {
                                // Also register in int_consts via eval_uint_resolved fallback (handled in lowering),
                                // but keep const_value so the global retains its initializer.
                            }
                        }
                    }
                } else if self.peek_is(Token::LBrace) {
                    // Bare braced initializer without `=`
                    // (`static const float3 k{...};`) — value discarded.
                    self.skip_brace_init()?;
                }
                let semantic = self.parse_optional_semantic()?;
                let semi = self.current().map(|t| t.span.end).unwrap_or(start);
                let const_name = name.clone();
                module.globals.push(GlobalDecl {
                    ty: return_type.clone(),
                    name,
                    semantic,
                    array_size,
                    shared: is_shared,
                    init: global_init,
                    span: Span::new(start, semi),
                });
                if let Some(v) = const_value {
                    module.constants.insert(const_name, v);
                }
                    if self.peek_is(Token::Comma) {
                        self.advance();
                        pending_name = Some(self.expect_identifier()?);
                        pending_paren_init = self.peek_is(Token::LParen)
                            && self.looks_like_paren_init();
                        continue;
                    }
                    break;
                }
                self.expect_semi()?;
            }
        }

        // Drain any leftover nested declarations (notably function-local
        // structs and their methods, which have no top-level drain point).
        let mut leftover_structs = std::mem::take(&mut self.pending_structs);
        module.structs.append(&mut leftover_structs);
        let mut leftover_methods = std::mem::take(&mut self.pending_methods);
        module.functions.append(&mut leftover_methods);

        // Struct inheritance flattens before name resolution so base
        // members (`srt.position` through `VcoStatsSRT : BasicDrawSRT`)
        // resolve like own fields. Then namespaces are flattened on
        // declaration but uses keep scopes (`sce::Got::BasicDrawSRT` vs
        // declared `BasicDrawSRT`): resolve scoped references to declared
        // structs now that the full struct table is known. Unknown scopes
        // stay untouched (loud DXC error).
        scoped::flatten_struct_bases(&mut module);
        scoped::resolve_scoped_struct_types(&mut module);
        // Template-id declaration spellings preserved for
        // monomorphization (angle lists erase in types).
        module.template_spellings = std::mem::take(&mut self.template_spellings);
        module.template_site_spellings =
            std::mem::take(&mut self.template_site_spellings);
        module.observed_template_ids = std::mem::take(&mut self.observed_template_ids);
        // Qualified struct static consts (`S::kMax` → literal) for
        // attribute evaluation. Bare constants win on collision (there
        // is none: qualified keys contain `::`), so only absent keys
        // insert.
        for (k, v) in std::mem::take(&mut self.struct_static_consts) {
            module
                .constants
                .entry(k)
                .or_insert(crate::ast::Expr::Number(v));
        }
        // Static member initializers (unfolded) for `T::kX` constexpr
        // evaluation. Later definitions win (same rule as constants).
        for (k, v) in std::mem::take(&mut self.template_static_inits) {
            module.template_static_inits.insert(k, v);
        }
        // Template value-param defaults (partial-arg binding). Later
        // definitions win.
        for (k, v) in std::mem::take(&mut self.template_param_defaults) {
            module.template_param_defaults.insert(k, v);
        }
        // Struct-field multi-dim dims (use linearization). Later
        // definitions win.
        for (k, v) in std::mem::take(&mut self.field_multidim) {
            module.field_multidim.insert(k, v);
        }

        Ok(module)
    }

    /// Drops trailing method qualifiers after a parameter list:
    /// `const`/`constexpr`/`volatile`, `noexcept[(...)]`,
    /// `override`/`final`, and `&`/`&&` ref-qualifiers. None affect
    /// validation.
    pub(crate) fn skip_trailing_method_qualifiers(&mut self) -> Result<()> {
        loop {
            if self.peek_is(Token::Const) {
                self.advance();
                continue;
            }
            if self.peek_is(Token::Ampersand) {
                self.advance();
                continue;
            }
            if self.peek_is(Token::Identifier) {
                let l = self.current_lexeme().to_ascii_lowercase();
                if l == "const"
                    || l == "constexpr"
                    || l == "volatile"
                    || l == "override"
                    || l == "final"
                {
                    self.advance();
                    continue;
                }
                if l == "noexcept" {
                    self.advance();
                    if self.peek_is(Token::LParen) {
                        self.skip_paren_block()?;
                    }
                    continue;
                }
                // GNU/MSVC function attributes (`__attribute__((...))`,
                // `__declspec(...)`) after the parameter list.
                if (self.current_lexeme() == "__attribute__"
                    || self.current_lexeme() == "__declspec")
                    && self.peek_next_is(Token::LParen)
                {
                    self.advance(); // attribute keyword
                    self.skip_paren_block()?;
                    continue;
                }
                // Bare `__`-prefixed markers (`__dead2`): compiler-reserved
                // attributes with no argument list.
                if self.current_lexeme().starts_with("__") {
                    self.advance();
                    continue;
                }
                // Trailing attribute macros on declarations
                // (`... _SCE_DBG_ATTRIBUTE_PRINTF(5);`, GNU attributes):
                // ALL-CAPS unknown `MACRO(...)` after the parameter list.
                if Self::is_macro_like_name(&self.current_lexeme())
                    && !self.is_known_type_name(&self.current_lexeme())
                    && self.peek_next_is(Token::LParen)
                {
                    self.advance(); // macro name
                    self.skip_paren_block()?;
                    continue;
                }
            }
            break;
        }
        Ok(())
    }

    /// Consumes an operator target after the `operator` keyword: `()`,
    /// `[]`, `new`/`delete`, a conversion target type, or one symbolic
    /// token (`*`, `==`, `<<`, ...).
    pub(crate) fn skip_operator_target(&mut self) -> Result<()> {
        if self.peek_is(Token::LParen) {
            self.skip_paren_block()
        } else if self.peek_is(Token::LBracket) {
            self.advance(); // `operator[`
            self.expect(Token::RBracket) // `]`
        } else if self.peek_is(Token::Identifier)
            && matches!(
                self.current_lexeme().to_ascii_lowercase().as_str(),
                "new" | "delete"
            )
        {
            self.advance(); // `operator new` / `operator delete`
            if self.peek_is(Token::LBracket) {
                self.advance();
                self.expect(Token::RBracket)?; // `new[]`
            }
            Ok(())
        } else if self.peek_is(Token::Identifier)
            || matches!(
                self.peek(),
                Some(
                    Token::Void
                        | Token::Bool
                        | Token::Int
                        | Token::UInt
                        | Token::Float
                        | Token::Half
                        | Token::Double
                )
            )
        {
            // Conversion operator (`operator bool`, `operator float4`) —
            // the type is irrelevant without a name; discard it.
            let _ = self.parse_type();
            Ok(())
        } else if !self.eof() {
            // Any symbolic operator (`*`, `==`, `<<`, ...): one token
            // (multi-char ops lex as single tokens).
            self.advance();
            Ok(())
        } else {
            Err(self.eof_error("operator"))
        }
    }

    /// Skips an operator overload's parameter list + body (or `;`),
    /// including trailing method qualifiers (`...() const { ... }`).
    pub(crate) fn skip_operator_signature(&mut self) -> Result<()> {
        self.skip_trailing_method_qualifiers()?;
        if self.peek_is(Token::LParen) {
            self.skip_paren_block()?;
            self.skip_trailing_method_qualifiers()?;
            if self.peek_is(Token::LBrace) {
                self.skip_brace_block()?;
            } else {
                self.expect_semi()?;
            }
        } else {
            // No parameter list — skip to `;` defensively.
            while !self.peek_is(Token::Semi) && !self.eof() {
                if self.peek_is(Token::LBrace) {
                    self.skip_brace_block()?;
                } else {
                    self.advance();
                }
            }
            self.expect_semi()?;
        }
        Ok(())
    }

    /// Records template parameter names from an already-consumed `<...>`
    /// token range `[start, end)`: `typename`/`class` params become known
    /// type names; scalar-typed value params (`uint64_t flags`) become
    /// pending template values + integer constants.
    /// Classifies one top-level `template<...>` item (`[start, end)`) as
    /// (name, is_value), or `None` when unrecognized. `typename T` /
    /// `class T` are type params; `<type> <name>` (defaults stripped)
    /// are value params. Mirrors the `pending_template_values` rules
    /// below so both mechanisms agree.
    fn classify_template_item(
        &self,
        start: usize,
        end: usize,
        type_names: &[&str],
    ) -> Option<(String, bool, Option<String>)> {
        // Collect significant tokens (skip a trailing `= default` for
        // classification, but capture single-literal defaults for
        // partial-arg binding: `int BoneCount = 0` → default `"0"`).
        // Complex defaults decline (None) and stay loud downstream.
        let toks: Vec<(Token, String)> = self.tokens[start..end.min(self.tokens.len())]
            .iter()
            .map(|t| (t.token.clone(), t.lexeme.clone()))
            .collect();
        let (toks, default) = match toks.iter().position(|(t, _)| *t == Token::Equal) {
            Some(eq) => {
                let def_toks = &toks[eq + 1..];
                // Single Number/Bool/String literal only (no side
                // effects, no parsing); anything else declines.
                let def = if def_toks.len() == 1 {
                    match &def_toks[0] {
                        (Token::Number, s) | (Token::Bool, s) | (Token::String, s) => {
                            Some(s.clone())
                        }
                        (Token::True, _) => Some("true".to_string()),
                        (Token::False, _) => Some("false".to_string()),
                        (Token::Identifier, s)
                            if s.eq_ignore_ascii_case("true")
                                || s.eq_ignore_ascii_case("false") =>
                        {
                            Some(s.clone())
                        }
                        _ => None,
                    }
                } else {
                    None
                };
                (toks[..eq].to_vec(), def)
            }
            None => (toks, None),
        };
        if toks.len() == 2 {
            let first = toks[0].1.to_ascii_lowercase();
            if (first == "typename" || first == "class") && toks[1].0 == Token::Identifier {
                return Some((toks[1].1.clone(), false, None));
            }
        }
        if toks.len() >= 2 {
            if let Some((last_tok, last_lex)) = toks.last() {
                if *last_tok == Token::Identifier {
                    let first_lower = toks[0].1.to_ascii_lowercase();
                    let is_type_name = type_names.contains(&first_lower.as_str())
                        || matches!(
                            toks[0].0,
                            Token::Int
                                | Token::UInt
                                | Token::Float
                                | Token::Half
                                | Token::Double
                                | Token::Bool
                        )
                        || (toks[0].0 == Token::Identifier
                            && self.is_known_type_name(&toks[0].1)
                            && first_lower != "typename"
                            && first_lower != "class");
                    let n_lower = last_lex.to_ascii_lowercase();
                    if is_type_name
                        && !type_names.contains(&n_lower.as_str())
                        && n_lower != "class"
                        && n_lower != "typename"
                    {
                        return Some((last_lex.clone(), true, default));
                    }
                }
            }
        }
        None
    }

    pub(crate) fn record_template_params(&mut self, start: usize, end: usize) {
        let type_names = [
            "uint64_t", "uint64", "uint32_t", "uint16_t", "uint8_t", "int64_t",
            "uint", "int", "bool", "float", "half", "double", "ulong", "long",
            "dword", "ushort", "short", "unsigned", "signed",
        ];
        // Ordered (name, is_value) capture for the pending drain is done
        // after the legacy recording below (same token range).
        // Ordered (name, is_value) params for the pending drain
        // (`Function::template_params`): split top-level commas, then
        // classify each item. `typename T`/`class T` are type params;
        // `<type> <name>` (with optional `= default`) are value params.
        {
            let mut ordered: Vec<(String, bool)> = Vec::new();
            let mut item_start = start;
            let mut angle = 0usize;
            let mut paren = 0usize;
            // end is one past '>'; scan [start, end-1).
            let mut i = start;
            while i < end {
                let is_comma = self.tokens.get(i).map(|t| t.token == Token::Comma).unwrap_or(false)
                    && angle == 0
                    && paren == 0;
                let is_end = i + 1 >= end;
                if self.tokens.get(i).map(|t| t.token == Token::Less).unwrap_or(false) {
                    angle += 1;
                } else if self.tokens.get(i).map(|t| matches!(t.token, Token::Greater | Token::ShiftRight)).unwrap_or(false) {
                    angle = angle.saturating_sub(1);
                } else if self.tokens.get(i).map(|t| t.token == Token::LParen).unwrap_or(false) {
                    paren += 1;
                } else if self.tokens.get(i).map(|t| t.token == Token::RParen).unwrap_or(false) {
                    paren = paren.saturating_sub(1);
                }
                if is_comma || is_end {
                    let item_end = if is_comma { i } else { end - 1 };
                    if let Some((pname, is_value, default)) =
                        self.classify_template_item(item_start, item_end, &type_names)
                    {
                        ordered.push((pname.clone(), is_value));
                        if let Some(d) = default {
                            self.pending_template_defaults.push((pname, d));
                        }
                    }
                    item_start = i + 1;
                }
                i += 1;
            }
            self.pending_template_params = ordered;
        }
        for idx in start..end.saturating_sub(1) {
            if let Some(tok) = self.tokens.get(idx) {
                let lower = tok.lexeme.to_ascii_lowercase();
                // Type params `typename T` / `class T` as known type names
                // (declaration lookahead).
                if (lower == "typename" || lower == "class")
                    && tok.token == Token::Identifier
                {
                    if let Some(next) = self.tokens.get(idx + 1) {
                        if next.token == Token::Identifier {
                            self.template_type_params.insert(next.lexeme.clone());
                        }
                    }
                }
                let is_type_name = type_names.contains(&lower.as_str())
                    || matches!(tok.token, Token::Int | Token::UInt | Token::Float | Token::Half | Token::Double | Token::Bool)
                    || lower == "uint64_t"
                    || (tok.token == Token::Identifier
                        && self.is_known_type_name(&tok.lexeme)
                        && lower != "typename"
                        && lower != "class");
                if is_type_name {
                    if let Some(next) = self.tokens.get(idx + 1) {
                        if next.token == Token::Identifier {
                            let n_lower = next.lexeme.to_ascii_lowercase();
                            if !type_names.contains(&n_lower.as_str())
                                && n_lower != "class"
                                && n_lower != "typename"
                            {
                                self.pending_template_values.push(next.lexeme.clone());
                                self.int_consts.insert(next.lexeme.clone(), "0".to_string());
                            }
                        }
                    }
                }
            }
        }
    }

    /// Skips a C++ constructor initializer list after `:`:
    /// `m_x(a), m_y{b}, Base<T>(c) ...` — stops before `{` or `;`.
    /// Unlike flat skipping, brace- and paren-initializers are consumed
    /// as units so their braces never read as the function body.
    pub(crate) fn skip_ctor_init_list(&mut self) -> Result<()> {
        self.expect(Token::Colon)?;
        loop {
            // Optional leading `::` (global scope) — rare but harmless.
            while self.peek_is(Token::Colon)
                && self
                    .tokens
                    .get(self.index + 1)
                    .map(|t| t.token == Token::Colon)
                    .unwrap_or(false)
            {
                self.advance();
                self.advance();
            }
            if self.peek_is(Token::Identifier) {
                self.advance(); // member or base name
            }
            if self.peek_is(Token::Less) {
                let _ = self.skip_angle_block();
            }
            if self.peek_is(Token::LParen) {
                self.skip_paren_block()?;
            } else if self.peek_is(Token::LBrace) {
                self.skip_brace_init()?;
            }
            if self.peek_is(Token::Comma) {
                self.advance();
                continue;
            }
            break;
        }
        Ok(())
    }

    // ------------------------------------------------------------------
    // Attributes
    // ------------------------------------------------------------------

    fn parse_attributes(&mut self) -> Result<Vec<Attribute>> {
        let mut attributes = Vec::new();
        while self.peek_is(Token::LBracket) {
            self.advance();
            // Support `[A, B]` or `[A(...), B(...)]` inside a single bracket pair
            loop {
                let name = self.expect_identifier()?;
                let mut arguments = Vec::new();

                if self.peek_is(Token::LParen) {
                    self.advance();
                    // Skip stray `#`/`##` from macro stringification before arguments
                    while self.peek_is(Token::Hash) || self.peek_is(Token::HashHash) {
                        self.advance();
                    }
                    if !self.peek_is(Token::RParen) {
                        loop {
                            while self.peek_is(Token::Hash) || self.peek_is(Token::HashHash) {
                                self.advance();
                            }
                            arguments.push(self.parse_assignment()?);
                            // Merge C macro stringification/adjacency into
                            // the previous argument: `"a" #M "b"` and
                            // `"a" "b"` concatenate into one string.
                            loop {
                                while self.peek_is(Token::Hash)
                                    || self.peek_is(Token::HashHash)
                                {
                                    self.advance();
                                    if self.peek_is(Token::Identifier) {
                                        self.advance();
                                    }
                                }
                                if self.peek_is(Token::String)
                                    && matches!(arguments.last(), Some(crate::ast::Expr::String(_)))
                                {
                                    let more = types::unquote(&self.current_lexeme());
                                    self.advance();
                                    if let Some(crate::ast::Expr::String(prev)) =
                                        arguments.last_mut()
                                    {
                                        prev.push_str(&more);
                                    }
                                    continue;
                                }
                                break;
                            }
                            if !self.peek_is(Token::Comma) {
                                break;
                            }
                            self.advance();
                            while self.peek_is(Token::Hash) || self.peek_is(Token::HashHash) {
                                self.advance();
                            }
                        }
                    }
                    while self.peek_is(Token::Hash) || self.peek_is(Token::HashHash) {
                        self.advance();
                    }
                    self.expect(Token::RParen)?;
                }

                attributes.push(Attribute { name, arguments });

                if self.peek_is(Token::Comma) {
                    self.advance();
                    continue;
                }
                break;
            }

            self.expect(Token::RBracket)?;
        }
        Ok(attributes)
    }

    // ------------------------------------------------------------------
    // Token helpers
    // ------------------------------------------------------------------

    pub(crate) fn current(&self) -> Option<&SpannedToken> {
        self.tokens.get(self.index)
    }

    pub(crate) fn peek(&self) -> Option<Token> {
        self.current().map(|t| t.token.clone())
    }

    pub(crate) fn peek_is(&self, token: Token) -> bool {
        matches!(self.peek(), Some(t) if t == token)
    }

    /// Whether the token at `index + 1` equals `token` (peek past the current).
    pub(crate) fn peek_next_is(&self, token: Token) -> bool {
        matches!(self.tokens.get(self.index + 1), Some(t) if t.token == token)
    }

    /// The lexeme of the token at `index + offset` (peek past the current),
    /// or empty when out of range.
    pub(crate) fn lexeme_after(&self, offset: usize) -> String {
        self.tokens
            .get(self.index + offset)
            .map(|t| t.lexeme.clone())
            .unwrap_or_default()
    }

    /// Consumes a `{ ... }` braced block, leaving the cursor after the closing
    /// `}` (and any optional trailing `;`). Errors if the current token is not
    /// a `{`.
    pub(crate) fn expect_greater(&mut self) -> Result<()> {
        if self.peek_is(Token::Greater) {
            self.advance();
            Ok(())
        } else if self.peek_is(Token::ShiftRight) {
            // `>>` lexed as `>>` inside `A<B<C>>` should be treated as `> >`
            if self.speculation_depth > 0 {
                // Speculative parse: consume whole, never mutate.
                self.advance();
                return Ok(());
            }
            let idx = self.index;
            let span = self.tokens[idx].span;
            self.tokens[idx].token = Token::Greater;
            self.tokens[idx].lexeme = ">".to_string();
            let new_tok = crate::token::SpannedToken {
                token: Token::Greater,
                lexeme: ">".to_string(),
                span,
            };
            self.tokens.insert(idx + 1, new_tok);
            self.advance();
            Ok(())
        } else {
            Err(self.error_expected("`>`"))
        }
    }

    pub(crate) fn skip_brace_block(&mut self) -> Result<()> {
        let start = self.index;
        self.expect(Token::LBrace)?;
        let mut depth = 1usize;
        while depth > 0 {
            if self.eof() {
                return Err(self.eof_error("'}'"));
            }
            if self.peek_is(Token::LBrace) {
                depth += 1;
            } else if self.peek_is(Token::RBrace) {
                depth -= 1;
            }
            self.advance();
        }
        if self.index - start > 5000 {
            eprintln!("BIGSKIP braces {}..{}", start, self.index);
        }
        if self.peek_is(Token::Semi) {
            self.advance();
        }
        Ok(())
    }

    // Skip angle‑bracket template argument block `< ... >`
    pub(crate) fn skip_angle_block(&mut self) -> Result<()> {
        self.expect(Token::Less)?;
        let mut depth = 1usize;
        while depth > 0 {
            if self.eof() {
                return Err(self.eof_error(">"));
            }
            if self.peek_is(Token::Less) {
                depth += 1;
            } else if self.peek_is(Token::Greater) {
                depth -= 1;
            } else if self.peek_is(Token::ShiftRight) {
                // `>>` closes (up to) two levels.
                if depth <= 2 {
                    depth = 0;
                } else {
                    depth -= 2;
                }
            }
            self.advance();
        }
        Ok(())
    }

    /// Skips the `(...)` (and optional `;`) of a bare module-scope
    /// macro invocation, assuming the macro name was consumed.
    pub(crate) fn skip_module_macro_rest(&mut self) -> Result<()> {
        self.skip_paren_block()?;
        if self.peek_is(Token::Semi) {
            self.advance();
        }
        Ok(())
    }

    pub(crate) fn skip_paren_block(&mut self) -> Result<()> {
        let start = self.index;
        self.expect(Token::LParen)?;
        let mut depth = 1usize;
        while depth > 0 {
            if self.eof() {
                return Err(self.eof_error("')'"));
            }
            if self.peek_is(Token::LParen) {
                depth += 1;
            } else if self.peek_is(Token::RParen) {
                depth -= 1;
            }
            self.advance();
        }
        if self.index - start > 5000 {
            eprintln!("BIGSKIP parens {}..{}", start, self.index);
        }
        Ok(())
    }

    /// Parses a paren-initializer `(args)` as a constructor of type `ty`
    /// (`static constexpr uint2 G(8, 8)` → `uint2(8, 8)`), for globals
    /// whose values must survive to lowering (`G.x` in `NUM_THREADS`,
    /// array sizes). Returns `None` (caller keeps today's skip) on any
    /// non-constant shape when `const_only` (mirroring the `=`-init gate
    /// at the global site); locals pass `false` (any value shape
    /// initializes). Never fails: on parse error the cursor rewinds for
    /// `skip_paren_block`.
    pub(crate) fn try_parse_paren_init(
        &mut self,
        ty: &crate::ast::Type,
        const_only: bool,
    ) -> Option<Expr> {
        let saved = self.index;
        let done = |s: &mut Self| {
            if s.eof() {
                return None;
            }
            if !s.peek_is(Token::LParen) {
                return None;
            }
            s.advance();
            let mut args = Vec::new();
            if s.peek_is(Token::RParen) {
                s.advance();
                return None;
            }
            loop {
                match s.parse_assignment() {
                    Ok(e) => args.push(e),
                    Err(_) => return None,
                }
                if s.peek_is(Token::Comma) {
                    s.advance();
                    continue;
                }
                if s.peek_is(Token::RParen) {
                    s.advance();
                    break;
                }
                return None;
            }
            if args.is_empty() {
                return None;
            }
            let const_arg = |e: &Expr| {
                !const_only
                    || matches!(
                        e,
                        Expr::Number(_)
                            | Expr::Bool(_)
                            | Expr::Variable(_)
                            | Expr::Constructor { .. }
                            | Expr::Binary { .. }
                            | Expr::Unary { .. }
                    )
            };
            if !args.iter().all(const_arg) {
                return None;
            }
            Some(Expr::Constructor { ty: ty.clone(), arguments: args })
        };
        let out = done(self);
        if out.is_none() {
            self.index = saved;
        }
        out
    }

    /// Skips a braced initializer `{ ... }` WITHOUT consuming a trailing
    /// `;` (unlike [`skip_brace_block`](Self::skip_brace_block), which is
    /// for declarations). Used for bare brace-inits (`Type name{...};`)
    /// whose value is discarded at parse level.
    pub(crate) fn skip_brace_init(&mut self) -> Result<()> {
        self.expect(Token::LBrace)?;
        let mut depth = 1usize;
        while depth > 0 {
            if self.eof() {
                return Err(self.eof_error("'}'"));
            }
            if self.peek_is(Token::LBrace) {
                depth += 1;
            } else if self.peek_is(Token::RBrace) {
                depth -= 1;
            }
            self.advance();
        }
        Ok(())
    }

    /// Non-consuming lookahead: with the cursor on `(` in `Type Name (` at
    /// module scope, is this a paren-initialized variable (e.g.
    /// `static constexpr uint2 k(8, 8);`) rather than a function?
    /// True when the parenthesised list cannot be a parameter list: any
    /// top-level segment starting with a literal, a lone identifier, or
    /// otherwise failing the `Type name` shape.
    pub(crate) fn looks_like_paren_init(&self) -> bool {
        // Top-level comma segments as (start, end) token ranges.
        // Tracks `()` depth and `<>` angle so commas/relations inside
        // nested lists don't split or confuse the scan. A stray top-level
        // `>`/`>>` (e.g. `k(a > b)`) can only be an expression.
        let mut segments: Vec<(usize, usize)> = Vec::new();
        let mut seg_start = self.index + 1;
        let mut depth = 0usize;
        let mut angle = 0usize;
        let mut k = self.index + 1;
        let mut closed = false;
        while k < self.tokens.len() {
            match self.tokens[k].token {
                Token::LParen => depth += 1,
                Token::RParen => {
                    if depth == 0 {
                        segments.push((seg_start, k));
                        closed = true;
                        break;
                    }
                    depth -= 1;
                }
                Token::Less if depth == 0 => angle += 1,
                Token::Greater if depth == 0 => {
                    if angle == 0 {
                        return true;
                    }
                    angle -= 1;
                }
                Token::ShiftRight if depth == 0 => {
                    if angle >= 2 {
                        angle -= 2;
                    } else {
                        return true;
                    }
                }
                Token::Comma if depth == 0 && angle == 0 => {
                    segments.push((seg_start, k));
                    seg_start = k + 1;
                }
                Token::Semi | Token::LBrace | Token::RBrace if depth == 0 => {
                    return false // give up: assume function
                }
                _ => {}
            }
            k += 1;
            if k - self.index > 150 {
                return false; // give up: assume function
            }
        }
        if !closed {
            return false;
        }
        if segments.len() == 1 && segments[0].0 == segments[0].1 {
            return false; // `()` → function (most-vexing-parse convention)
        }
        segments
            .iter()
            .any(|&(s, e)| self.paren_seg_is_init(s, e))
    }

    /// Whether one top-level `(...)` segment looks like an initializer
    /// expression rather than a `Type name [= default]` parameter.
    fn paren_seg_is_init(&self, s: usize, e: usize) -> bool {
        if s >= e {
            return true;
        }
        // Framework macro parameters (`PSSL_ARG_OUT(T, N) SEM(M)`,
        // `REGULAR_BUFFER(T) x`) shape parameters, not initializers.
        if self.segment_has_macro_param(s, e) {
            return false;
        }
        // Trailing `*`/`&` with no declarator name (`void *`, `T &`) is an
        // unnamed pointer/reference parameter, not an initializer (a
        // dangling dereference is not a valid expression).
        if e > s
            && matches!(
                self.tokens.get(e - 1).map(|t| &t.token),
                Some(Token::Star | Token::Ampersand)
            )
        {
            return false;
        }
        let tok = |i: usize| self.tokens.get(i).map(|t| &t.token);
        // C-style `void` / variadic `...` segments are parameters.
        if e - s == 1 && matches!(tok(s), Some(Token::Void)) {
            return false;
        }
        if (s..e).all(|i| matches!(tok(i), Some(Token::Dot))) {
            return false;
        }
        // Leading literal → expression, not a parameter.
        if matches!(
            tok(s),
            Some(Token::Number | Token::String | Token::True | Token::False)
        ) {
            return true;
        }
        // Lone keyword scalar (`(int)`) → unnamed parameter.
        if e - s == 1
            && matches!(
                tok(s),
                Some(
                    Token::Void
                        | Token::Bool
                        | Token::Int
                        | Token::UInt
                        | Token::Float
                        | Token::Half
                        | Token::Double
                )
            )
        {
            return false;
        }
        // Lone identifier: an unnamed parameter (`(string, T0)`) when it
        // names a known type (incl. HLSL `string` and template params);
        // otherwise an expression (`(kSize)`).
        if e - s == 1 && matches!(tok(s), Some(Token::Identifier)) {
            return !self.is_known_type_name(&self.tokens[s].lexeme.clone());
        }
        // Strip leading qualifiers and C type words (`unsigned`, ...).
        // `unsigned name` still finds `name` below, so keep scanning.
        let mut i = s;
        loop {
            if i >= e {
                return true;
            }
            let t = &self.tokens[i];
            if t.token == Token::Const
                || t.token == Token::Static
                || t.token == Token::In
                || t.token == Token::Out
                || t.token == Token::Inout
            {
                i += 1;
                continue;
            }
            if t.token == Token::Identifier {
                match t.lexeme.to_ascii_lowercase().as_str() {
                    "const" | "constexpr" | "static" | "inline" | "unsigned" | "signed"
                    | "long" | "short" => {
                        i += 1;
                        continue;
                    }
                    _ => {}
                }
            }
            break;
        }
        if i >= e {
            return true;
        }
        // A parameter needs a NAME after the type prefix: an Identifier
        // (angle-0; `<>`-nested commas were hidden by the splitter)
        // preceded by type-plausible tokens and followed by a declarator
        // continuation. Adjacent `Ident Ident` never occurs in expressions,
        // so `V2P input` classifies correctly even when `V2P` is
        // forward-declared. The name cannot be the first token (the type
        // must precede it); after qualifier stripping the scan may start
        // at the first type token.
        let mut j = if i == s { s + 1 } else { i };
        let mut angle = 0usize;
        while j < e {
            let t = &self.tokens[j];
            if t.token == Token::Less {
                angle += 1;
            } else if t.token == Token::Greater {
                angle = angle.saturating_sub(1);
            } else if t.token == Token::ShiftRight {
                // `>>` closes up to two levels (stray `>>` was already
                // filtered by the splitter, so this is template syntax).
                angle = angle.saturating_sub(2);
            } else if angle == 0
                && (t.token == Token::Identifier
                    || matches!(
                        t.token,
                        Token::In | Token::Out | Token::Inout | Token::Const | Token::Static
                    ))
            {
                // Reserved words double as parameter names
                // (`PrimaryPayload inout`); they can never start an
                // expression, so they are always name-shaped here.
                let before_ok = matches!(
                    self.tokens.get(j - 1).map(|p| &p.token),
                    Some(
                        Token::Identifier
                            | Token::Void
                            | Token::Bool
                            | Token::Int
                            | Token::UInt
                            | Token::Float
                            | Token::Half
                            | Token::Double
                            | Token::Star
                            | Token::Ampersand
                            | Token::Greater
                            | Token::ShiftRight
                            | Token::Const
                            | Token::Static
                            | Token::In
                            | Token::Out
                            | Token::Inout
                    )
                ) || matches!(
                    self.tokens
                        .get(j - 1)
                        .map(|p| p.lexeme.to_ascii_lowercase())
                        .as_deref(),
                    Some(
                        "const"
                            | "constexpr"
                            | "static"
                            | "inline"
                            | "unsigned"
                            | "signed"
                            | "long"
                            | "short"
                    )
                );
                // A trailing `::` means this ident is a scope, not the name
                // (`sce::Agc x` — keep scanning for `x`).
                let is_scope = matches!(
                    self.tokens.get(j + 1).map(|n| &n.token),
                    Some(Token::Colon)
                ) && matches!(
                    self.tokens.get(j + 2).map(|n| &n.token),
                    Some(Token::Colon)
                );
                let after_ok = j + 1 >= e
                    || matches!(
                        self.tokens.get(j + 1).map(|n| &n.token),
                        Some(
                            Token::LBracket
                                | Token::Equal
                                | Token::Colon
                                | Token::LBrace
                                | Token::Comma
                                | Token::RParen
                        )
                    )
                    // `name FRAMEWORK_SEMANTIC(...)` — framework semantic macros after
                    // the parameter name are still parameters.
                    || (matches!(
                        self.tokens.get(j + 1),
                        Some(n) if Self::is_semantic_macro_name(&n.lexeme)
                    ) && matches!(
                        self.tokens.get(j + 2).map(|n| &n.token),
                        Some(Token::LParen)
                    ));
                if before_ok && after_ok && !is_scope {
                    return false; // `Type ... name ...` → parameter-shaped
                }
            }
            j += 1;
        }
        true
    }

    pub(crate) fn current_span(&self) -> Span {
        self.current()
            .map(|t| t.span)
            .unwrap_or_else(|| self.end_span())
    }

    pub(crate) fn previous_span(&self) -> Span {
        self.tokens
            .get(self.index.saturating_sub(1))
            .map(|t| t.span)
            .unwrap_or_else(|| self.end_span())
    }

    pub(crate) fn current_lexeme(&self) -> String {
        self.current().map(|t| t.lexeme.clone()).unwrap_or_default()
    }

    pub(crate) fn end_span(&self) -> Span {
        let offset = self.tokens.last().map(|t| t.span.end).unwrap_or(0);
        Span::new(offset, offset)
    }

    pub(crate) fn advance(&mut self) {
        let span = self.current_span();
        let desc = self
            .current()
            .map(|t| format!("{:?} `{}`", t.token, t.lexeme))
            .unwrap_or_else(|| "eof".to_string());
        tracing::trace!(
            target: TRACE_TOKENS,
            index = self.index,
            line = span.start,
            token = %desc,
        );
        self.index = self.index.saturating_add(1);
    }

    pub(crate) fn eof(&self) -> bool {
        self.index >= self.tokens.len()
    }

    pub(crate) fn expect_identifier(&mut self) -> Result<String> {
        if !self.peek_is(Token::Identifier) {
            return Err(self.error_expected("identifier"));
        }
        let value = self.current_lexeme();
        self.advance();
        Ok(value)
    }

    /// Skips HLSL interpolation / misc prefix modifiers that may appear
    /// before a declaration's type (struct/cbuffer fields, parameters,
    /// globals, locals). Consumed and ignored for validation.
    ///
    /// Covers `linear`, `centroid`, `nointerpolation`, `noperspective`,
    /// `sample` (e.g. `sample float4 pos : S_POSITION;` in AGC shading
    /// samples), plus `precise`, `unorm`/`snorm`, `uniform`, `extern`,
    /// `volatile`, `shared`, `groupshared`, `globallycoherent`.
    /// Storage keywords (`static`/`const`/`constexpr`/`inline`,
    /// `column_major`/`row_major`, `thread_group_memory`) are handled by
    /// callers and intentionally NOT consumed here.
    pub(crate) fn skip_interp_modifiers(&mut self) {
        loop {
            if self.peek_is(Token::Identifier) {
                if !Self::is_decl_modifier_word(&self.current_lexeme()) {
                    break;
                }
                // Only a modifier when a declaration tail follows
                // (`sample float4 x`); otherwise a type/use/call head
                // (`Sample get_sample(`, `sample(x)`, `Shared::SRT`,
                // `Sample s;`).
                if !self.modifier_followed_by_decl() {
                    break;
                }
                self.advance();
                continue;
            }
            break;
        }
    }

    /// Whether an identifier in `is_decl_modifier_word` is in fact used
    /// as a modifier: skip any further chained modifiers (stopping at
    /// scope heads like `Shared::`), then require a full type shape
    /// (scoped/template/multiword aware) followed by a declarator
    /// continuation (a name, `*`, or `&`). Distinguishes
    /// `sample float4 x` from `Sample get_sample(`, `sample(x)`,
    /// `Shared::SRT x`, and `Sample s;`.
    pub(crate) fn modifier_followed_by_decl(&self) -> bool {
        // A `::` scope head (`Shared::SRT`) is a type, never a modifier.
        if self.peek_next_is(Token::Colon)
            && self
                .tokens
                .get(self.index + 2)
                .map(|t| t.token == Token::Colon)
                .unwrap_or(false)
        {
            return false;
        }
        // Skip chained modifiers, stopping at scope heads. `const` and
        // `static` lex as dedicated tokens (not identifiers), so chains
        // like `constexpr const bool` must count them too — otherwise the
        // lookahead vetoes a real declaration (`static constexpr const
        // bool value` in psr headers) and parsing desyncs downstream.
        let mut j = self.index;
        loop {
            let Some(tok) = self.tokens.get(j) else {
                return false;
            };
            if matches!(tok.token, Token::Const | Token::Static) {
                let scope_after = self.tokens.get(j + 1).map(|t| t.token == Token::Colon).unwrap_or(false)
                    && self.tokens.get(j + 2).map(|t| t.token == Token::Colon).unwrap_or(false);
                if scope_after {
                    break;
                }
                j += 1;
                continue;
            }
            if tok.token != Token::Identifier || !Self::is_decl_modifier_word(&tok.lexeme) {
                break;
            }
            let scope_after = self.tokens.get(j + 1).map(|t| t.token == Token::Colon).unwrap_or(false)
                && self.tokens.get(j + 2).map(|t| t.token == Token::Colon).unwrap_or(false);
            if scope_after {
                break;
            }
            j += 1;
        }
        // Type shape at j: keyword scalar, or identifier with optional
        // `::` scope chains, `<...>` template args, and C multiword
        // tails (`unsigned int`, `long long`).
        let mut k = j;
        match self.tokens.get(k).map(|t| &t.token) {
            Some(
                Token::Void
                | Token::Bool
                | Token::Int
                | Token::UInt
                | Token::Float
                | Token::Half
                | Token::Double,
            ) => {
                k += 1;
            }
            Some(Token::Identifier) => {
                k += 1;
                // `::Ident` scope chains.
                while self.tokens.get(k).map(|t| t.token == Token::Colon).unwrap_or(false)
                    && self.tokens.get(k + 1).map(|t| t.token == Token::Colon).unwrap_or(false)
                {
                    k += 2;
                    if matches!(
                        self.tokens.get(k).map(|t| &t.token),
                        Some(Token::Identifier)
                    ) {
                        k += 1;
                    } else {
                        break;
                    }
                }
                // `<...>` template arguments (nesting-aware).
                if matches!(
                    self.tokens.get(k).map(|t| &t.token),
                    Some(Token::Less)
                ) {
                    let mut angle = 0usize;
                    let mut closed = false;
                    while k < self.tokens.len() {
                        if self.tokens[k].token == Token::Less {
                            angle += 1;
                        } else if self.tokens[k].token == Token::Greater {
                            angle = angle.saturating_sub(1);
                            if angle == 0 {
                                closed = true;
                                k += 1;
                                break;
                            }
                        } else if self.tokens[k].token == Token::ShiftRight {
                            if angle <= 2 {
                                closed = true;
                                k += 1;
                                break;
                            }
                            angle -= 2;
                        } else if matches!(
                            self.tokens[k].token,
                            Token::Semi | Token::LBrace | Token::RBrace | Token::RParen
                        ) {
                            break;
                        }
                        k += 1;
                        if k - j > 60 {
                            break;
                        }
                    }
                    if !closed {
                        return false;
                    }
                }
            }
            _ => return false,
        }
        // Declarator continuation: a name, `*`, or `&`.
        matches!(
            self.tokens.get(k).map(|t| &t.token),
            Some(Token::Identifier | Token::Star | Token::Ampersand)
        )
    }

    /// Identifier spellings droppable as declaration modifiers (storage
    /// class, interpolation, and misc HLSL/C++ qualifiers).
    pub(crate) fn is_decl_modifier_word(lexeme: &str) -> bool {
        matches!(
            lexeme.to_ascii_lowercase().as_str(),
            "column_major"
                | "row_major"
                | "constexpr"
                | "static"
                | "inline"
                | "const"
                | "linear"
                | "centroid"
                | "nointerpolation"
                | "nointerp"
                | "noperspective"
                | "custominterp"
                | "sample"
                | "precise"
                | "unorm"
                | "snorm"
                | "uniform"
                | "extern"
                | "volatile"
                | "shared"
                | "groupshared"
                | "globallycoherent"
                | "thread_group_memory"
        )
    }

    /// Whether a `(...)` segment contains a framework macro parameter:
    /// leading `MACRO(T, N)` with a top-level comma inside (the macro
    /// stands where a type would), or `REGULAR_BUFFER(T) name`. Such
    /// segments declare parameters; plain calls (`foo(a, b)`) and inits
    /// (`k(make(1))`) do not match.
    fn segment_has_macro_param(&self, s: usize, e: usize) -> bool {
        let mut k = s;
        while k < e {
            let is_macro_paren = matches!(
                self.tokens.get(k).map(|t| &t.token),
                Some(Token::Identifier)
            ) && matches!(
                self.tokens.get(k + 1).map(|t| &t.token),
                Some(Token::LParen)
            );
            if !is_macro_paren {
                k += 1;
                continue;
            }
            let name = self.tokens[k].lexeme.clone();
            let known_resource = types::is_resource_kind(&name);
            let unknown_macro =
                Self::is_macro_like_name(&name) && !self.is_known_type_name(&name);
            if !known_resource && !unknown_macro {
                k += 1;
                continue;
            }
            // Find the matching `)` (paren-depth aware), noting a
            // top-level comma inside.
            let mut depth = 0usize;
            let mut has_comma = false;
            let mut j = k + 1;
            let mut closed = None;
            while j < e {
                if self.tokens[j].token == Token::LParen {
                    depth += 1;
                } else if self.tokens[j].token == Token::RParen {
                    depth = depth.saturating_sub(1);
                    if depth == 0 {
                        closed = Some(j);
                        break;
                    }
                } else if self.tokens[j].token == Token::Comma && depth == 1 {
                    has_comma = true;
                }
                j += 1;
            }
            let Some(c) = closed else {
                return false;
            };
            if unknown_macro {
                // Leading `MACRO(T, N)` (the macro stands where a type
                // would) with a top-level comma inside.
                if has_comma && k == s {
                    return true;
                }
            } else if matches!(
                self.tokens.get(c + 1).map(|t| &t.token),
                Some(Token::Identifier)
            ) {
                // `REGULAR_BUFFER(T) name`.
                return true;
            }
            k = c + 1;
        }
        false
    }

    /// Whether `lexeme` looks like a framework macro name (all caps): unknown
    /// ALL-CAPS identifiers in declaration positions are qualifiers or
    /// calling-convention markers, never types or variables.
    pub(crate) fn is_macro_like_name(lexeme: &str) -> bool {
        !lexeme.is_empty()
            && lexeme
                .chars()
                .all(|c| c.is_ascii_uppercase() || c == '_' || c.is_ascii_digit())
    }

    /// Whether `lexeme` names a framework semantic macro (`PSSL_SEMANTIC`,
    /// `SHADER_SEMANTIC`, ...): ALL-CAPS ending in `SEMANTIC`.
    pub(crate) fn is_semantic_macro_name(lexeme: &str) -> bool {
        Self::is_macro_like_name(lexeme) && lexeme.ends_with("SEMANTIC")
    }

    /// Skips calling-convention / attribute macros interposed in
    /// declarations (`typedef T MACRO alias;`): an ALL-CAPS unknown
    /// identifier followed by another identifier. Stops otherwise.
    pub(crate) fn skip_interposed_macro(&mut self) {
        while self.peek_is(Token::Identifier)
            && matches!(
                self.tokens.get(self.index + 1).map(|t| &t.token),
                Some(Token::Identifier)
            )
        {
            let lexeme = self.current_lexeme();
            if !Self::is_macro_like_name(&lexeme) || self.is_known_type_name(&lexeme) {
                break;
            }
            self.advance();
        }
    }

    /// Consumes a typedef alias name, which may be a plain identifier or
    /// a keyword scalar spelling being (re)defined
    /// (`typedef unsigned int uint;`).
    pub(crate) fn expect_alias_name(&mut self) -> Result<String> {
        if self.peek_is(Token::Identifier) {
            return self.expect_identifier();
        }
        if matches!(
            self.peek(),
            Some(
                Token::Void
                    | Token::Bool
                    | Token::Int
                    | Token::UInt
                    | Token::Float
                    | Token::Half
                    | Token::Double
            )
        ) {
            let name = self.current_lexeme();
            self.advance();
            return Ok(name);
        }
        Err(self.error_expected("identifier"))
    }

    pub(crate) fn expect(&mut self, expected: Token) -> Result<()> {
        if !self.peek_is(expected.clone()) {
            return Err(self.error_expected(expected.describe()));
        }
        self.advance();
        Ok(())
    }

    pub(crate) fn expect_semi(&mut self) -> Result<()> {
        if self.peek_is(Token::Semi) {
            self.advance();
            Ok(())
        } else if self.is_type_start()
            || self.peek_is(Token::If)
            || self.peek_is(Token::For)
            || self.peek_is(Token::While)
            || self.peek_is(Token::Return)
            || self.peek_is(Token::LBrace)
            || self.peek_is(Token::RBrace)
            || self.peek_is(Token::Identifier)
            || self.peek_is(Token::Number)
            || self.peek_is(Token::String)
            || self.peek_is(Token::True)
            || self.peek_is(Token::False)
        {
            tracing::warn!(target: TRACE_TOKENS, "missing semicolon, recovering");
            Ok(())
        } else {
            Err(self.error_expected("`;`"))
        }
    }

    pub(crate) fn eof_error(&self, expected: &str) -> Ps5ShaderError {
        let offset = self.end_span().start;
        Ps5ShaderError::at(
            format!("expected {expected}, found end of file"),
            Phase::Parse,
            Span::new(offset, offset),
        )
    }

    pub(crate) fn error_expected(&self, expected: &str) -> Ps5ShaderError {
        let span = self.current_span();
        let found = self
            .current()
            .map(|t| t.token.describe().to_string())
            .unwrap_or_else(|| "end of file".to_string());
        let lexeme = self.current_lexeme();
        tracing::error!(
            target: TRACE_TOKENS,
            line = span.start,
            expected = %expected,
            found = %found,
            lexeme = %lexeme,
            "parse failure",
        );
        Ps5ShaderError::at(
            format!("expected {expected}, found {found}"),
            Phase::Parse,
            span,
        )
    }
}
