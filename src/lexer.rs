use logos::Logos;

use crate::error::{Phase, Ps5ShaderError, Result, Span};
use crate::token::{SpannedToken, Token};

/// Raw, regex-attributed token used by logos. The lexer maps each variant to
/// our logical [`Token`] while preserving source spans and lexemes.
#[derive(Debug, Clone, PartialEq, Logos)]
enum RawToken {
    // Keywords
    #[token("struct")]
    Struct,
    #[token("return")]
    Return,
    #[token("if")]
    If,
    #[token("else")]
    Else,
    #[token("for")]
    For,
    #[token("while")]
    While,
    #[token("const")]
    Const,
    // C restrict qualifiers behave like `const` for parsing purposes.
    #[token("__restrict")]
    #[token("__restrict__")]
    RestrictConst,
    #[token("static")]
    Static,
    #[token("in")]
    In,
    #[token("out")]
    Out,
    #[token("inout")]
    Inout,
    #[token("true")]
    True,
    #[token("false")]
    False,
    #[token("break")]
    Break,
    #[token("continue")]
    Continue,
    #[token("discard")]
    Discard,

    // Void and scalar types
    #[token("void")]
    Void,
    #[token("bool")]
    Bool,
    #[token("int")]
    Int,
    #[token("uint")]
    UInt,
    #[token("float")]
    Float,
    #[token("half")]
    Half,
    #[token("double")]
    Double,

    // Literals
    // Numbers: supports decimal/float literals and C++ `0x` hex / `0b`
    // binary literals with `'` digit separators (e.g. `0x1'0000'0000`,
    // `1'000.5`, `0b100000`). The separator is
    // stripped from the lexeme in `lex()` so the downstream numeric value is clean.
    // Also supports C++ integer suffixes `u`, `l`, `ul`, `ull`, `lu` etc.,
    // PSSL `s`/`us` (short/ushort, e.g. `128us`), and `f`/`h` for
    // half/float, so `~0ul` lexes as `~` `0ul` not `0` `ul`.
    #[regex(r"(?:0[xX][0-9A-Fa-f]+(?:'[0-9A-Fa-f]+)*[uUlLsS]*|0[bB][01]+(?:'[01]+)*[uUlLsS]*|[0-9]+(?:'[0-9]+)*(?:\.[0-9]*(?:'[0-9]+)*)?(?:[eE][+-]?[0-9]+(?:'[0-9]+)*)?[fFhHuUlLsS]*|\.[0-9]+(?:'[0-9]+)*(?:[eE][+-]?[0-9]+(?:'[0-9]+)*)?[fFhHuUlLsS]*)")]
    Number,
    #[regex(r#""([^"\\]|\\.)*""#)]
    String,

    // Identifiers (also covers compound types like float2, float3x3, and
    // semantic/type names, which the parser resolves).
    #[regex(r"[A-Za-z_][A-Za-z0-9_]*")]
    Identifier,

    // Operators / punctuation
    #[token(">>=")]
    ShiftRightEqual,
    #[token("<<=")]
    ShiftLeftEqual,
    #[token("++")]
    PlusPlus,
    #[token("--")]
    MinusMinus,
    #[token("+=")]
    PlusEqual,
    #[token("-=")]
    MinusEqual,
    #[token("*=")]
    StarEqual,
    #[token("/=")]
    SlashEqual,
    #[token("%=")]
    PercentEqual,
    #[token("^=")]
    CaretEqual,
    #[token("|=")]
    PipeEqual,
    #[token("&=")]
    AmpersandEqual,
    #[token("+")]
    Plus,
    #[token("-")]
    Minus,
    #[token("*")]
    Star,
    #[token("/")]
    Slash,
    #[token("%")]
    Percent,
    #[token("=")]
    Equal,
    #[token("==")]
    EqualEqual,
    #[token("!=")]
    NotEqual,
    #[token("<")]
    Less,
    #[token("<=")]
    LessEqual,
    #[token(">")]
    Greater,
    #[token(">=")]
    GreaterEqual,
    #[token("&&")]
    AndAnd,
    #[token("&")]
    Ampersand,
    #[token("|")]
    Pipe,
    #[token("^")]
    Caret,
    #[token("~")]
    Tilde,
    #[token("<<")]
    ShiftLeft,
    #[token(">>")]
    ShiftRight,
    #[token("||")]
    OrOr,
    #[token("->")]
    Arrow,
    #[token("!")]
    Bang,
    #[token("?")]
    Question,
    #[token(":")]
    Colon,
    #[token(",")]
    Comma,
    #[token(".")]
    Dot,
    #[token(";")]
    Semi,
    #[token("(")]
    LParen,
    #[token(")")]
    RParen,
    #[token("{")]
    LBrace,
    #[token("}")]
    RBrace,
    #[token("[")]
    LBracket,
    #[token("]")]
    RBracket,
    #[token("##")]
    HashHash,
    #[token("#")]
    Hash,

    // Skipped trivia — note: backslash line continuation `\\\n` must be handled
    // before Number/Identifier so `~0ul` etc. are not split.
    #[regex(r"\\[ \t]*\r?\n", logos::skip)]
    Continuation,
    // Stray backslashes (e.g. `\ + 0` inside a static_assert from a
    // half-expanded size macro, or lone `\` lines): no PSSL construct
    // uses a bare `\`, so skip them instead of erroring. The longer
    // `Continuation` match above still wins for real continuations.
    #[regex(r"\\", logos::skip)]
    StrayBackslash,
    #[regex(r"[ \t\r\n]+", logos::skip)]
    Whitespace,
    #[regex(r"//[^\r\n]*?", logos::skip)]
    LineComment,
    #[regex(r"/\*[^*]*\*+(?:[^/*][^*]*\*+)*/", logos::skip)]
    BlockComment,
}

/// Maps a raw lexed token variant to the logical [`Token`].
///
/// Trivia variants (whitespace/comments) are skipped by logos and never reach
/// this function, but they are still handled defensively (mapped to an
/// identifier) so that no panic path exists in the lexer.
fn map_token(raw: &RawToken) -> Token {
    use RawToken::*;
    match raw {
        Struct => Token::Struct,
        Return => Token::Return,
        If => Token::If,
        Else => Token::Else,
        For => Token::For,
        While => Token::While,
        Const => Token::Const,
        RestrictConst => Token::Const,
        Static => Token::Static,
        In => Token::In,
        Out => Token::Out,
        Inout => Token::Inout,
        True => Token::True,
        False => Token::False,
        Break => Token::Break,
        Continue => Token::Continue,
        Discard => Token::Discard,
        Void => Token::Void,
        Bool => Token::Bool,
        Int => Token::Int,
        UInt => Token::UInt,
        Float => Token::Float,
        Half => Token::Half,
        Double => Token::Double,
        Number => Token::Number,
        String => Token::String,
        Identifier => Token::Identifier,
        Plus => Token::Plus,
        Minus => Token::Minus,
        Star => Token::Star,
        Slash => Token::Slash,
        Percent => Token::Percent,
        Equal => Token::Equal,
        ShiftRightEqual => Token::ShiftRightEqual,
        ShiftLeftEqual => Token::ShiftLeftEqual,
        PlusEqual => Token::PlusEqual,
        MinusEqual => Token::MinusEqual,
        StarEqual => Token::StarEqual,
        SlashEqual => Token::SlashEqual,
        PercentEqual => Token::PercentEqual,
        CaretEqual => Token::CaretEqual,
        PipeEqual => Token::PipeEqual,
        AmpersandEqual => Token::AmpersandEqual,
        PlusPlus => Token::PlusPlus,
        MinusMinus => Token::MinusMinus,
        EqualEqual => Token::EqualEqual,
        NotEqual => Token::NotEqual,
        Less => Token::Less,
        LessEqual => Token::LessEqual,
        Greater => Token::Greater,
        GreaterEqual => Token::GreaterEqual,
        AndAnd => Token::AndAnd,
        Ampersand => Token::Ampersand,
        Pipe => Token::Pipe,
        Caret => Token::Caret,
        Tilde => Token::Tilde,
        ShiftLeft => Token::ShiftLeft,
        ShiftRight => Token::ShiftRight,
        OrOr => Token::OrOr,
        Arrow => Token::Arrow,
        Bang => Token::Bang,
        Question => Token::Question,
        Colon => Token::Colon,
        Comma => Token::Comma,
        Dot => Token::Dot,
        Semi => Token::Semi,
        LParen => Token::LParen,
        RParen => Token::RParen,
        LBrace => Token::LBrace,
        RBrace => Token::RBrace,
        LBracket => Token::LBracket,
        RBracket => Token::RBracket,
        Hash => Token::Hash,
        HashHash => Token::HashHash,
        // Defensive fallback; never reached because logos skips trivia.
        Continuation | Whitespace | LineComment | BlockComment | StrayBackslash => {
            Token::Identifier
        }
    }
}

/// Lexes `source` into a vector of [`SpannedToken`]s, skipping whitespace and
/// comments. Every produced token carries its source span and lexeme.
pub fn lex(source: &str) -> Result<Vec<SpannedToken>> {
    let mut lexer = RawToken::lexer(source);
    let mut output = Vec::new();

    while let Some(item) = lexer.next() {
        let span = lexer.span();
        let start = span.start;
        let end = span.end;

        match item {
            Ok(raw) => {
                let token = map_token(&raw);
                let mut lexeme = source
                    .get(span.clone())
                    .map(str::to_owned)
                    .unwrap_or_default();
                // Strip C++ `'` digit separators from numeric literals so the
                // downstream parser/emitter sees a clean value.
                if token == Token::Number {
                    lexeme.retain(|c| c != '\'');
                }
                output.push(SpannedToken::new(token, Span::new(start, end), lexeme));
            }
            Err(()) => {
                let lexeme = source
                    .get(span.clone())
                    .map(str::to_owned)
                    .unwrap_or_default();
                let message = if lexeme.is_empty() {
                    "unrecognized token".to_string()
                } else {
                    format!("unrecognized token {lexeme:?}")
                };
                return Err(Ps5ShaderError::at(
                    message,
                    Phase::Lex,
                    Span::new(start, end),
                ));
            }
        }
    }

    // Merge `##` token pasting (`A##B##C` → single identifier) from
    // half-expanded function-generation macros (e.g.
    // `closest_hit_caller_##VGPR_COUNT##_vgprs`). Only `Ident ## Ident`
    // chains merge; anything else keeps its tokens.
    let mut merged: Vec<SpannedToken> = Vec::with_capacity(output.len());
    let mut i = 0usize;
    while i < output.len() {
        if output[i].token == Token::Identifier {
            let mut lexeme = output[i].lexeme.clone();
            let mut end = output[i].span.end;
            let mut j = i;
            loop {
                let hash = matches!(output.get(j + 1), Some(t) if t.token == Token::HashHash);
                let ident = matches!(output.get(j + 2), Some(t) if t.token == Token::Identifier);
                if !(hash && ident) {
                    break;
                }
                lexeme.push_str(&output[j + 2].lexeme);
                end = output[j + 2].span.end;
                j += 2;
            }
            if j != i {
                let start = output[i].span.start;
                merged.push(SpannedToken::new(
                    Token::Identifier,
                    Span::new(start, end),
                    lexeme,
                ));
                i = j + 1;
                continue;
            }
        }
        merged.push(output[i].clone());
        i += 1;
    }

    Ok(merged)
}
