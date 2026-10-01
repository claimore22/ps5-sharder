//! Minimal conditional-expression evaluation for `#if` / `#elif`.

use std::collections::HashMap;

/// Minimal conditional-expression evaluator.
///
/// Supports the guard shapes that actually appear in the PSSL corpus:
/// * `0` / `1`
/// * `defined(X)` and `!defined(X)`
/// * a single logical `&&` / `||` of the above
///
/// Anything unrecognized evaluates as true (kept for forward compatibility with
/// unknown feature guards) — matching the pre-existing behavior.
pub(crate) fn eval_cond(expr: &str, defines: &HashMap<String, String>) -> bool {
    let expr = expr.trim();
    if expr.is_empty() {
        return false;
    }

    // Single `defined(X)` / `!defined(X)` — only when the whole expression is
    // exactly one such term (no `&&`/`||` operators). Parsing `defined(` and a
    // trailing `)` blindly would otherwise misparse `!defined(A) || defined(B)`
    // as `!defined("A) || defined(B")`, wrongly turning it into `!0` → true.
    if !expr.contains("&&") && !expr.contains("||") {
        let single = eval_defined_expr(expr, defines);
        if let Some(v) = single {
            return v;
        }
    }

    // `!(A && B)` / `!(A || B)`: the outer `!` negates the whole parenthesized
    // group. Naively splitting on `&&`/`||` below would produce `["!(A", "B)"]`
    // and evaluate the wrong thing (`!defined(A) && defined(B)` instead of
    // `!(defined(A) && defined(B))`). Evaluate and negate the group first.
    if let Some(rest) = expr.strip_prefix('!') {
        if let Some(inner) = rest.trim().strip_prefix('(').and_then(|x| x.strip_suffix(')')) {
            return !eval_cond(inner, defines);
        }
    }

    // Tokenize top-level `&&` / `||`.
    let or_parts: Vec<&str> = expr.split("||").collect();
    if or_parts.len() > 1 {
        return or_parts.iter().any(|p| eval_cond(p.trim(), defines));
    }
    let and_parts: Vec<&str> = expr.split("&&").collect();
    if and_parts.len() > 1 {
        return and_parts.iter().all(|p| eval_cond(p.trim(), defines));
    }

    // Numeric comparisons (`#if (FXAA_QUALITY__PS > 12)`, `#if X == 29`):
    // expand defines, then compare numbers. Abstains (falls through) unless
    // both sides resolve to numbers, so unknown feature guards keep the
    // pre-existing keep-unknown behavior.
    if let Some(v) = eval_comparison(expr, defines) {
        return v;
    }

    // Bare define or numeric literal:
    // `#if RAD_FULL` where `RAD_FULL` is `1` should be true; undefined
    // identifier should be false (C preprocessor: undefined → 0).
    if let Some(val) = defines.get(expr) {
        let v = val.trim();
        // Empty define (`#define X`) → true (defined as 1)
        if v.is_empty() {
            return true;
        }
        return !(v == "0" || v == "0x0" || v == "0X0");
    }
    if expr.chars().all(|c| c.is_alphanumeric() || c == '_') {
        // Single bare identifier not in `defines` → undefined → 0 → false.
        // This makes `#elif RAD_HALF` correctly false when only `RAD_FULL` is defined.
        return false;
    }
    // Parenthesized bare define ` (RAD_FULL) `
    if let Some(inner) = expr.strip_prefix('(').and_then(|x| x.strip_suffix(')')) {
        let inner = inner.trim();
        if let Some(val) = defines.get(inner) {
            let v = val.trim();
            if v.is_empty() {
                return true;
            }
            return !(v == "0" || v == "0x0");
        }
        if inner.chars().all(|c| c.is_alphanumeric() || c == '_') {
            return false;
        }
    }
    // Numeric literal / unknown symbol → non-zero means true.
    !(expr == "0")
}

/// Evaluates a top-level numeric comparison (`==`, `!=`, `>=`, `<=`, `>`,
/// `<`) after define expansion. Returns `None` unless both sides resolve to
/// numbers (unknown guards abstain to preserve keep-unknown behavior).
/// A single leading `!` negates the result.
fn eval_comparison(expr: &str, defines: &HashMap<String, String>) -> Option<bool> {
    let mut expr = expr.trim();
    let mut negate = false;
    if let Some(rest) = expr.strip_prefix('!') {
        // `!=` is handled as an operator below, not negation.
        if !rest.trim_start().starts_with('=') {
            negate = true;
            expr = rest.trim();
        }
    }
    // Strip outer paren layers wrapping the whole expression
    // (`#if (X > 12)`).
    loop {
        let t = expr.trim();
        if !t.starts_with('(') {
            break;
        }
        let bytes = t.as_bytes();
        let mut depth = 0usize;
        let mut wraps = false;
        let mut i = 0;
        while i < bytes.len() {
            match bytes[i] {
                b'(' => depth += 1,
                b')' => {
                    depth = depth.saturating_sub(1);
                    if depth == 0 {
                        wraps = i == bytes.len() - 1;
                        break;
                    }
                }
                _ => {}
            }
            i += 1;
        }
        if !wraps {
            break;
        }
        expr = t[1..t.len() - 1].trim();
    }
    // Find a top-level comparison operator (two-character forms first).
    let bytes = expr.as_bytes();
    let mut depth = 0usize;
    let mut op_at: Option<(usize, usize)> = None;
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'(' => depth += 1,
            b')' => depth = depth.saturating_sub(1),
            b'=' if depth == 0 && i + 1 < bytes.len() && bytes[i + 1] == b'=' => {
                op_at = Some((i, 2));
                break;
            }
            b'!' if depth == 0 && i + 1 < bytes.len() && bytes[i + 1] == b'=' => {
                op_at = Some((i, 2));
                break;
            }
            b'>' if depth == 0 => {
                if i + 1 < bytes.len() && bytes[i + 1] == b'=' {
                    op_at = Some((i, 2));
                } else {
                    op_at = Some((i, 1));
                }
                break;
            }
            b'<' if depth == 0 => {
                if i + 1 < bytes.len() && bytes[i + 1] == b'=' {
                    op_at = Some((i, 2));
                } else {
                    op_at = Some((i, 1));
                }
                break;
            }
            _ => {}
        }
        i += 1;
    }
    let (pos, len) = op_at?;
    let op = &expr[pos..pos + len];
    let left = eval_number(&super::directive::expand_defines(&expr[..pos], defines))?;
    let right = eval_number(&super::directive::expand_defines(&expr[pos + len..], defines))?;
    let result = match op {
        "==" => left == right,
        "!=" => left != right,
        ">=" => left >= right,
        "<=" => left <= right,
        ">" => left > right,
        "<" => left < right,
        _ => return None,
    };
    Some(if negate { !result } else { result })
}

/// Resolves `s` to a number: optional surrounding parens, decimal/hex
/// integer or float literal. Returns `None` for anything else (identifiers
/// that survived expansion, expressions, ...).
fn eval_number(s: &str) -> Option<f64> {
    let mut s = s.trim();
    while s.starts_with('(') && s.ends_with(')') && s.len() >= 2 {
        s = s[1..s.len() - 1].trim();
    }
    if s.is_empty() {
        return None;
    }
    // Hex integer.
    let lower = s.to_ascii_lowercase();
    if let Some(hex) = lower.strip_prefix("0x") {
        if !hex.is_empty() && hex.chars().all(|c| c.is_ascii_hexdigit()) {
            return u64::from_str_radix(hex, 16).ok().map(|v| v as f64);
        }
        return None;
    }
    // Suffixes occasionally seen in conditions (`8u`, `1.0f`).
    let s = s.trim_end_matches(|c| c == 'u' || c == 'U' || c == 'l' || c == 'L');
    let s = if s.ends_with('f') || s.ends_with('F') {
        // A trailing `f` only marks a float literal when the body parses.
        let body = &s[..s.len() - 1];
        if body.contains('.') || body.contains('e') || body.contains('E') {
            body
        } else {
            return None;
        }
    } else {
        s
    };
    // Must be a bare numeric literal (reject identifiers/expressions).
    if s.is_empty()
        || !s
            .chars()
            .all(|c| c.is_ascii_digit() || c == '.' || c == '+' || c == '-' || c == 'e' || c == 'E')
    {
        return None;
    }
    // A lone sign or dot is not a number; `+`/`-` may only lead.
    let body = s.trim_start_matches(['+', '-']);
    if body.is_empty() || body == "." {
        return None;
    }
    s.parse::<f64>().ok()
}

/// Normalizes `defined (X)` / `!defined (X)` (optional whitespace between
/// `defined` and the paren, as written throughout the corpus) to the canonical
/// `defined(X)` / `!defined(X)` form. Returns the input unchanged when the
/// shape does not match.
fn normalize_defined_spacing(expr: &str) -> String {
    let mut out = expr.to_string();
    // Repeat for nested `!defined (` wrappers (e.g. `!defined (A)`).
    for _ in 0..4 {
        let next = normalize_defined_once(&out);
        if next == out {
            return next;
        }
        out = next;
    }
    out
}

fn normalize_defined_once(expr: &str) -> String {
    // Find a `defined` keyword occurrence and check for `<space>* (` after it.
    let bytes = expr.as_bytes();
    let mut i = 0;
    while i + 7 <= bytes.len() {
        if &expr[i..i + 7] == "defined"
            && (i == 0 || !(bytes[i - 1] as char).is_ascii_alphanumeric() && bytes[i - 1] != b'_')
        {
            let mut j = i + 7;
            while j < bytes.len() && (bytes[j] == b' ' || bytes[j] == b'\t') {
                j += 1;
            }
            if j < bytes.len() && bytes[j] == b'(' && j > i + 7 {
                // Whitespace present between `defined` and `(`: remove it.
                let mut s = String::with_capacity(expr.len());
                s.push_str(&expr[..i + 7]);
                s.push_str(&expr[j..]);
                return s;
            }
            i = j.max(i + 7);
        } else {
            i += 1;
        }
    }
    expr.to_string()
}

/// Evaluates a single `defined(X)` / `!defined(X)` / `0` / `1` term if it has no
/// surrounding operators; returns `None` when it is not a bare term.
fn eval_defined_expr(expr: &str, defines: &HashMap<String, String>) -> Option<bool> {
    let e = expr.trim();
    // `defined` with optional whitespace before the paren (`defined (X)`,
    // as written throughout the corpus, e.g. `#if defined ( __PROSPERO__ )`):
    // normalize to `defined(X)` / `!defined(X)` first.
    let e_norm: String;
    let e: &str = {
        e_norm = normalize_defined_spacing(e);
        &e_norm
    };
    if let Some(rest) = e.strip_prefix('!') {
        // `!defined(X)` or `!(...)` or `!RAD_FULL`
        let inner = rest
            .strip_prefix("defined(")
            .and_then(|x| x.strip_suffix(')'));
        if let Some(name) = inner {
            return Some(!defines.contains_key(name.trim()));
        }
        if let Some(inner2) = rest.strip_prefix('(').and_then(|x| x.strip_suffix(')')) {
            return eval_defined_expr(inner2, defines).map(|v| !v);
        }
        // `!RAD_FULL` where `RAD_FULL` is a plain define
        let bare = rest.trim();
        if let Some(val) = defines.get(bare) {
            let v = val.trim();
            let is_zero = v == "0" || v == "0x0" || v.is_empty() && false;
            // `!0` → true, `!1` → false, `!` + non-zero → false
            return Some(v == "0" || v == "0x0");
        }
        if bare.chars().all(|c| c.is_alphanumeric() || c == '_') {
            // `!UNDEFINED` → !0 → true
            return Some(true);
        }
        // `!0` / `!1`
        return match bare {
            "0" => Some(true),
            "1" => Some(false),
            _ => None,
        };
    }
    if let Some(inner) = e.strip_prefix("defined(").and_then(|x| x.strip_suffix(')')) {
        let name = inner.trim();
        return Some(defines.contains_key(name));
    }
    if let Some(rest) = e.strip_prefix("defined ") {
        let name = rest.split_whitespace().next()?;
        return Some(defines.contains_key(name));
    }
    match e {
        "0" => Some(false),
        "1" => Some(true),
        _ => None,
    }
}
