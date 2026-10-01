use std::path::PathBuf;

use thiserror::Error;
use serde::{Deserialize, Serialize};

/// The compiler phase in which an error occurred.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Phase {
    Preprocess,
    Lex,
    Parse,
    Semantic,
    Lower,
    Emit,
    DxcCompile,
    SpirvValidate,
    /// SPIR-V inspection (rspirv debugging/analysis only; never validation).
    Inspect,
    Io,
}

impl Phase {
    pub fn as_str(self) -> &'static str {
        match self {
            Phase::Preprocess => "preprocessing",
            Phase::Lex => "lexing",
            Phase::Parse => "parsing",
            Phase::Semantic => "semantic analysis",
            Phase::Lower => "lowering",
            Phase::Emit => "HLSL emission",
            Phase::DxcCompile => "DXC compilation",
            Phase::SpirvValidate => "SPIR-V validation",
            Phase::Inspect => "SPIR-V inspection",
            Phase::Io => "I/O",
        }
    }
}

/// A source location spanning a byte range within a single source file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

impl Span {
    pub fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }

    pub fn merge(self, other: Span) -> Span {
        Span {
            start: self.start.min(other.start),
            end: self.end.max(other.end),
        }
    }
}

/// The unified error type for the entire compiler pipeline.
///
/// Every phase reports through this type so that a single error retains the
/// source file, span, line, column, phase, and a human-readable message.
#[derive(Debug, Error)]
#[error("{message}")]
pub struct Ps5ShaderError {
    pub message: String,
    pub phase: Phase,
    pub span: Option<Span>,
    pub file: Option<PathBuf>,
    /// 1-based line, when the span has been resolved against a source map.
    pub line: Option<usize>,
    /// 1-based column, when the span has been resolved against a source map.
    pub column: Option<usize>,
    #[source]
    source: Option<Box<dyn std::error::Error + Send + Sync>>,
}

impl Ps5ShaderError {
    pub fn new(message: impl Into<String>, phase: Phase) -> Self {
        Self {
            message: message.into(),
            phase,
            span: None,
            file: None,
            line: None,
            column: None,
            source: None,
        }
    }

    pub fn at(message: impl Into<String>, phase: Phase, span: Span) -> Self {
        Self {
            message: message.into(),
            phase,
            span: Some(span),
            file: None,
            line: None,
            column: None,
            source: None,
        }
    }

    pub fn with_source(
        message: impl Into<String>,
        phase: Phase,
        source: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        Self {
            message: message.into(),
            phase,
            span: None,
            file: None,
            line: None,
            column: None,
            source: Some(Box::new(source)),
        }
    }

    /// Attaches the originating source file path.
    pub fn with_file(mut self, file: impl Into<PathBuf>) -> Self {
        self.file = Some(file.into());
        self
    }

    /// Attaches resolved line/column information.
    pub fn with_position(mut self, line: usize, column: usize) -> Self {
        self.line = Some(line);
        self.column = Some(column);
        self
    }

    pub fn source_message(&self) -> Option<&dyn std::error::Error> {
        self.source.as_deref().map(|b| b as &dyn std::error::Error)
    }
}

pub type Result<T> = std::result::Result<T, Ps5ShaderError>;
