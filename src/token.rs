use crate::error::Span;

/// The tokenized vocabulary of the PSSL language.
///
/// This is the shared, logical token type consumed by the parser. The lexer
/// (see [`crate::lexer`]) is responsible for turning raw source text into
/// these tokens, attaching source spans.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Token {
    // Keywords
    Struct,
    Return,
    If,
    Else,
    For,
    While,
    Const,
    Static,
    In,
    Out,
    Inout,
    True,
    False,
    Break,
    Continue,
    Discard,

    // Void and scalar types
    Void,
    Bool,
    Int,
    UInt,
    Float,
    Half,
    Double,

    // Compound (vector/matrix) types are lexed as Identifier-like names and
    // interpreted by the parser via `TypeName`? No: we keep vector/matrix
    // spellings as `Identifier` and let the parser treat them specially via
    // the `TypeName`/builtin table. This keeps the token stream small and
    // consistent with how "float4(...)" works as a constructor.

    // Literals
    Number,
    String,

    // Identifiers
    Identifier,

    // Operators / punctuation
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    Equal,
    PlusEqual,
    MinusEqual,
    StarEqual,
    SlashEqual,
    PercentEqual,
    ShiftLeftEqual,
    ShiftRightEqual,
    CaretEqual,
    PipeEqual,
    AmpersandEqual,
    PlusPlus,
    MinusMinus,
    EqualEqual,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    AndAnd,
    Ampersand,
    Pipe,
    Caret,
    Tilde,
    ShiftLeft,
    ShiftRight,
    OrOr,
    Bang,
    Question,
    Colon,
    Comma,
    Dot,
    Arrow,
    Semi,
    LParen,
    RParen,
    LBrace,
    RBrace,
    LBracket,
    RBracket,
    Hash,
    HashHash,
}

/// A token together with the source span it was produced from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpannedToken {
    pub token: Token,
    pub span: Span,
    pub lexeme: String,
}

impl SpannedToken {
    pub fn new(token: Token, span: Span, lexeme: String) -> Self {
        Self {
            token,
            span,
            lexeme,
        }
    }
}

impl Token {
    /// A short human-readable name used in diagnostics, e.g. expected vs found.
    pub fn describe(&self) -> &'static str {
        match self {
            Token::Struct => "`struct`",
            Token::Return => "`return`",
            Token::If => "`if`",
            Token::Else => "`else`",
            Token::For => "`for`",
            Token::While => "`while`",
            Token::Const => "`const`",
            Token::Static => "`static`",
            Token::In => "`in`",
            Token::Out => "`out`",
            Token::Inout => "`inout`",
            Token::True => "`true`",
            Token::False => "`false`",
            Token::Break => "`break`",
            Token::Continue => "`continue`",
            Token::Discard => "`discard`",
            Token::Void => "`void`",
            Token::Bool => "`bool`",
            Token::Int => "`int`",
            Token::UInt => "`uint`",
            Token::Float => "`float`",
            Token::Half => "`half`",
            Token::Double => "`double`",
            Token::Number => "number literal",
            Token::String => "string literal",
            Token::Identifier => "identifier",
            Token::Plus => "`+`",
            Token::Minus => "`-`",
            Token::Star => "`*`",
            Token::Slash => "`/`",
            Token::Percent => "`%`",
            Token::Equal => "`=`",
            Token::PlusEqual => "`+=`",
            Token::MinusEqual => "`-=`",
            Token::StarEqual => "`*=`",
            Token::SlashEqual => "`/=`",
            Token::PercentEqual => "`%=`",
            Token::ShiftLeftEqual => "`<<=`",
            Token::ShiftRightEqual => "`>>=`",
            Token::CaretEqual => "`^=`",
            Token::PipeEqual => "`|=`",
            Token::AmpersandEqual => "`&=`",
            Token::PlusPlus => "`++`",
            Token::MinusMinus => "`--`",
            Token::Arrow => "`->`",
            Token::EqualEqual => "`==`",
            Token::NotEqual => "`!=`",
            Token::Less => "`<`",
            Token::LessEqual => "`<=`",
            Token::Greater => "`>`",
            Token::GreaterEqual => "`>=`",
            Token::AndAnd => "`&&`",
            Token::Ampersand => "`&`",
            Token::Pipe => "`|`",
            Token::Caret => "`^`",
            Token::Tilde => "`~`",
            Token::ShiftLeft => "`<<`",
            Token::ShiftRight => "`>>`",
            Token::OrOr => "`||`",
            Token::Bang => "`!`",
            Token::Question => "`?`",
            Token::Colon => "`:`",
            Token::Comma => "`,`",
            Token::Dot => "`.`",
            Token::Semi => "`;`",
            Token::LParen => "`(`",
            Token::RParen => "`)`",
            Token::LBrace => "`{`",
            Token::RBrace => "`}`",
            Token::LBracket => "`[`",
            Token::RBracket => "`]`",
            Token::Hash => "`#`",
            Token::HashHash => "`##`",
        }
    }
}
