//! Preprocessor directive helpers: `#include` / `#define` parsing and basic
//! object-like macro expansion.

use std::collections::HashMap;

/// Whether the trimmed line begins a controlled `#include` directive.
pub(crate) fn is_include_directive(trimmed: &str) -> bool {
    if trimmed.starts_with("#include") {
        if let Some(rest) = trimmed.get(8..) {
            let rest = rest.trim_start();
            return rest.starts_with('"') || rest.starts_with('<');
        }
    }
    if trimmed.starts_with("# include") {
        if let Some(rest) = trimmed.get(9..) {
            let rest = rest.trim_start();
            return rest.starts_with('"') || rest.starts_with('<');
        }
    }
    false
}

/// Extracts the include target name (without quotes/angle brackets).
pub(crate) fn parse_include_name(trimmed: &str) -> Option<String> {
    let hash_pos = trimmed.find('#')?;
    let after_hash = trimmed[hash_pos + 1..].trim_start();
    if !after_hash.starts_with("include") {
        return None;
    }
    let after_include = after_hash["include".len()..].trim_start();
    if after_include.starts_with('"') {
        let end = after_include[1..].find('"')?;
        Some(after_include[1..1 + end].to_string())
    } else if after_include.starts_with('<') {
        let end = after_include[1..].find('>')?;
        Some(after_include[1..1 + end].to_string())
    } else {
        None
    }
}

/// Whether the trimmed line begins a `#define` directive.
pub(crate) fn is_define_directive(trimmed: &str) -> bool {
    trimmed.starts_with("#define") || trimmed.starts_with("# define")
}

/// Parses a function-like `#define NAME(p1, p2, ...) body` into
/// `(name, params, body)`.
///
/// The `(` must immediately follow the name (C rule: `#define FOO (x)`
/// is an object-like `FOO` with value `(x)`). Only fixed-arity,
/// single-line forms qualify, plus two variadic patterns:
/// - `#define NAME(...)` with an empty body (the `IF_PSR_CPU` pattern, i.e.
///   "drop the invocation entirely", any arity);
/// - `#define NAME(...) __VA_ARGS__` (the `IF_PSR_GPU` pattern, i.e. "splice
///   the arguments through").
///
/// Named `...` tails (`#define F(x, ...)`) and any other `__VA_ARGS__`
/// use return `None` and keep today's loud path.
pub(crate) fn parse_func_define(trimmed: &str) -> Option<(String, Vec<String>, String)> {
    let hash_pos = trimmed.find('#')?;
    let after_hash = trimmed[hash_pos + 1..].trim_start();
    if !after_hash.starts_with("define") {
        return None;
    }
    let after_define = after_hash["define".len()..].trim_start();
    // Name runs to `(` or whitespace; `(` must be immediate.
    let mut name_end = after_define.len();
    for (i, ch) in after_define.char_indices() {
        if ch == '(' || ch.is_whitespace() {
            name_end = i;
            break;
        }
    }
    let name = after_define[..name_end].trim().to_string();
    if name.is_empty() || !is_ident(&name) {
        return None;
    }
    let rest = &after_define[name_end..];
    if !rest.starts_with('(') {
        return None;
    }
    // Balanced parameter list (no nesting possible in params, but be
    // strict: stop at the first `)`).
    let close = rest.find(')')?;
    let params_str = &rest[1..close];
    // Pure-variadic (`...`) with an empty or `__VA_ARGS__`-only body:
    // record with a `...` marker param (expansion special-cases it).
    if params_str.trim() == "..." {
        let body = rest[close + 1..].trim().to_string();
        let body = strip_comments(&body).trim().to_string();
        if body.is_empty() || body == "__VA_ARGS__" {
            return Some((name, vec!["...".to_string()], body));
        }
        return None;
    }
    let mut params = Vec::new();
    if !params_str.trim().is_empty() {
        for p in params_str.split(',') {
            let p = p.trim().to_string();
            if p.is_empty() || !is_ident(&p) || p == "..." || p.contains("...") {
                return None;
            }
            params.push(p);
        }
    }
    let body = rest[close + 1..].trim().to_string();
    if body.is_empty() {
        return None;
    }
    Some((name, params, strip_comments(&body).trim().to_string()))
}

/// Parses `#undef NAME` into the name (`None` for anything else,
/// notably `#ifdef` / `#ifndef` which merely share a prefix).
pub(crate) fn parse_undef(trimmed: &str) -> Option<String> {
    let hash_pos = trimmed.find('#')?;
    let after_hash = trimmed[hash_pos + 1..].trim_start();
    if !after_hash.starts_with("undef") {
        return None;
    }
    let after_undef = after_hash["undef".len()..].trim_start();
    let mut name_end = after_undef.len();
    for (i, ch) in after_undef.char_indices() {
        if ch.is_whitespace() {
            name_end = i;
            break;
        }
    }
    let name = after_undef[..name_end].trim().to_string();
    if name.is_empty() || !is_ident(&name) {
        return None;
    }
    Some(name)
}

/// Expands function-like invocations `NAME(args...)` across a line.
///
/// Each invocation parses balanced, top-level-comma-split arguments;
/// arity must match; parameters substitute whole-word (raw, C-like);
/// `##` operands paste (surrounding whitespace dropped). One pass —
/// the driver rescans to a fixpoint so nested factories
/// (`CONCAT(S_TARGET_OUTPUT, slot)` → `CONCAT_(…)` → `S_TARGET_OUTPUT1`)
/// fully unfold. Unparseable invocations and arity mismatches stay
/// verbatim (loud, never mis-expanded).
pub(crate) fn expand_func_defines(
    line: &str,
    funcs: &HashMap<String, (Vec<String>, String)>,
) -> String {
    if funcs.is_empty() {
        return line.to_string();
    }
    let mut out = String::with_capacity(line.len());
    let bytes = line.as_bytes();
    // Longest-name-first so `CONCAT_` wins over `CONCAT` at the same
    // position (word-boundary + `(` checks make this defensive only).
    let mut names: Vec<&String> = funcs.keys().collect();
    names.sort_by(|a, b| b.len().cmp(&a.len()));
    let mut i = 0;
    while i < line.len() {
        let mut replacement: Option<String> = None;
        let mut end = i;
        for name in &names {
            if line[i..].starts_with(name.as_str()) && is_word_boundary_before(bytes, i) {
                let mut j = i + name.len();
                while j < line.len() && bytes[j].is_ascii_whitespace() {
                    j += 1;
                }
                if j < line.len() && bytes[j] == b'(' {
                    if let Some((args, close)) = parse_call_args(line, j) {
                        let (params, body) = &funcs[name.as_str()];
                        // Pure-variadic marker (`...`): any arity; empty
                        // body drops the invocation, `__VA_ARGS__` splices
                        // the arguments through verbatim.
                        if params.len() == 1 && params[0] == "..." {
                            if body.is_empty() {
                                replacement = Some(String::new());
                                end = close;
                                break;
                            }
                            if body == "__VA_ARGS__" {
                                replacement = Some(args.join(", "));
                                end = close;
                                break;
                            }
                        } else if args.len() == params.len() {
                            let mut expanded = body.clone();
                            for (p, a) in params.iter().zip(args.iter()) {
                                expanded = replace_word(&expanded, p, a);
                            }
                            expanded = paste_tokens(&expanded);
                            replacement = Some(expanded);
                            end = close;
                            break;
                        }
                    }
                }
            }
        }
        if let Some(expanded) = replacement {
            out.push_str(&expanded);
            i = end;
        } else {
            let ch = line[i..].chars().next().unwrap();
            out.push(ch);
            i += ch.len_utf8();
        }
    }
    out
}

/// Parses `(args...)` at byte offset `open` (which must be `(`),
/// returning top-level-comma-split arguments and the offset just past
/// the closing `)`. `None` when unbalanced.
fn parse_call_args(line: &str, open: usize) -> Option<(Vec<String>, usize)> {
    let bytes = line.as_bytes();
    if bytes.get(open) != Some(&b'(') {
        return None;
    }
    let mut args = Vec::new();
    let mut depth = 0usize;
    let mut start = open + 1;
    let mut i = open;
    while i < line.len() {
        match bytes[i] {
            b'(' => depth += 1,
            b')' => {
                depth -= 1;
                if depth == 0 {
                    let last = line[start..i].trim();
                    if !last.is_empty() {
                        args.push(last.to_string());
                    }
                    return Some((args, i + 1));
                }
            }
            b',' if depth == 1 => {
                args.push(line[start..i].trim().to_string());
                start = i + 1;
            }
            _ => {}
        }
        i += 1;
    }
    None
}

/// Resolves `##` token pastes (`x ## y` → `xy`, whitespace dropped).
/// Operands substitute before pasting (the caller substitutes params
/// first), matching the corpus idiom `CONCAT(S_TARGET_OUTPUT, slot)`.
fn paste_tokens(s: &str) -> String {
    let mut out = s.to_string();
    loop {
        let Some(pos) = out.find("##") else {
            break;
        };
        // Left operand end: skip spaces back over ident chars.
        let mut l = pos;
        while l > 0 && out.as_bytes()[l - 1].is_ascii_whitespace() {
            l -= 1;
        }
        let mut lstart = l;
        while lstart > 0 {
            let c = out.as_bytes()[lstart - 1] as char;
            if c.is_ascii_alphanumeric() || c == '_' {
                lstart -= 1;
            } else {
                break;
            }
        }
        // Right operand start: skip spaces forward over ident chars.
        let mut r = pos + 2;
        while r < out.len() && out.as_bytes()[r].is_ascii_whitespace() {
            r += 1;
        }
        let mut rend = r;
        while rend < out.len() {
            let c = out.as_bytes()[rend] as char;
            if c.is_ascii_alphanumeric() || c == '_' {
                rend += 1;
            } else {
                break;
            }
        }
        if lstart == l || r == rend {
            // Malformed paste (missing operand): stop rather than mangle.
            break;
        }
        let joined = format!("{}{}", &out[lstart..l], &out[r..rend]);
        out.replace_range(lstart..rend, &joined);
    }
    out
}

/// Parses an object-like `#define NAME value` into `(name, value)`.
pub(crate) fn parse_define(trimmed: &str) -> Option<(String, String)> {
    let hash_pos = trimmed.find('#')?;
    let after_hash = trimmed[hash_pos + 1..].trim_start();
    if !after_hash.starts_with("define") {
        return None;
    }
    let after_define = after_hash["define".len()..].trim_start();
    if after_define.is_empty() {
        return None;
    }
    let mut parts = after_define.splitn(2, |c: char| c.is_whitespace());
    let name = parts.next()?.trim().to_string();
    if name.is_empty() || !is_ident(&name) {
        return None;
    }
    let value = parts.next().unwrap_or("").trim().to_string();
    Some((name, strip_comments(&value).trim().to_string()))
}

/// Removes `//` line comments and `/* ... */` block comments from replacement
/// text. A `#define` value must not carry the `#define` line's trailing
/// comment — doing so would leak comment text (including any `;` inside it)
/// into every expansion site.
fn strip_comments(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if s[i..].starts_with("//") {
            break;
        }
        if s[i..].starts_with("/*") {
            match s[i + 2..].find("*/") {
                Some(end) => {
                    i += 2 + end + 2;
                    continue;
                }
                None => break,
            }
        }
        let ch = s[i..].chars().next().unwrap();
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}

fn is_ident(s: &str) -> bool {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) if c.is_ascii_alphabetic() || c == '_' => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// Expands object-like defines across a line, replacing each defined name with
/// its value wherever it appears as a whole identifier. Iterates to a
/// fixpoint (bounded) so nested defines (`LDS_X` → `(THREADS_X + 2)` →
/// `(16 + 2)`) fully expand regardless of hash-map order.
pub(crate) fn expand_defines(line: &str, defines: &HashMap<String, String>) -> String {
    if defines.is_empty() {
        return line.to_string();
    }
    let mut result = line.to_string();
    for _ in 0..8 {
        let mut changed = false;
        for (name, value) in defines {
            let next = replace_word(&result, name, value);
            if next != result {
                result = next;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    result
}

fn replace_word(haystack: &str, needle: &str, replacement: &str) -> String {
    let mut out = String::with_capacity(haystack.len());
    let mut i = 0;
    let bytes = haystack.as_bytes();
    let needle_bytes = needle.as_bytes();
    let nlen = needle_bytes.len();
    while i < haystack.len() {
        if haystack[i..].starts_with(needle)
            && is_word_boundary_before(bytes, i)
            && is_word_boundary_after(bytes, i + nlen)
        {
            out.push_str(replacement);
            i += nlen;
        } else {
            // Advance by one char
            let ch = haystack[i..].chars().next().unwrap();
            out.push(ch);
            i += ch.len_utf8();
        }
    }
    out
}

fn is_word_boundary_before(bytes: &[u8], pos: usize) -> bool {
    if pos == 0 {
        return true;
    }
    let prev = bytes[pos - 1] as char;
    !(prev.is_ascii_alphanumeric() || prev == '_')
}

fn is_word_boundary_after(bytes: &[u8], pos: usize) -> bool {
    if pos >= bytes.len() {
        return true;
    }
    let next = bytes[pos] as char;
    !(next.is_ascii_alphanumeric() || next == '_')
}
