use crate::error::{Ps5ShaderError, Span};

/// Maps byte offsets within a single source buffer to (line, column) pairs.
///
/// Line and column are both 1-based. Columns are counted in bytes (code
/// points beyond ASCII are treated as single bytes for now).
#[derive(Debug, Clone)]
pub struct SourceMap {
    /// For each line, the byte offset at which it begins.
    line_starts: Vec<usize>,
}

impl SourceMap {
    /// Builds a source map for `source`. The line-start table is computed
    /// once and reused for every diagnostic, so this is cheap to keep around.
    pub fn new(source: &str) -> Self {
        let mut line_starts = vec![0];
        for (index, byte) in source.bytes().enumerate() {
            if byte == b'\n' {
                line_starts.push(index + 1);
            }
        }
        Self { line_starts }
    }

    /// Resolves a byte offset to a (line, column). `offset` is clamped into
    /// `0..=len`.
    pub fn position(&self, offset: usize) -> (usize, usize) {
        let line_index = match self.line_starts.binary_search(&offset) {
            Ok(index) => index,
            Err(index) => index.saturating_sub(1),
        };

        let line_start = self.line_starts[line_index];
        let column = offset.saturating_sub(line_start) + 1;
        (line_index + 1, column)
    }

    /// Renders the error as a formatted, source-aware diagnostic string.
    ///
    /// Format: `{file}:{line}:{col}: error[{phase}]: {message}`
    pub fn render(&self, error: &Ps5ShaderError) -> String {
        let mut output = String::new();

        if let Some(file) = &error.file {
            let name = file.display();
            match (error.line, error.column) {
                (Some(line), Some(column)) => {
                    let _ = std::fmt::write(&mut output, format_args!("{name}:{line}:{column}: "));
                }
                _ => {
                    let _ = std::fmt::write(&mut output, format_args!("{name}: "));
                }
            }
        }

        let _ = std::fmt::write(
            &mut output,
            format_args!("error[{}]: {}", error.phase.as_str(), error.message),
        );

        output
    }
}

/// Resolves a span to a (line, column) pair within `source`.
pub fn span_position(source: &str, span: Span) -> (usize, usize) {
    SourceMap::new(source).position(span.start)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map(source: &str) -> SourceMap {
        SourceMap::new(source)
    }

    #[test]
    fn first_line_is_one_one() {
        let source = "abc";
        assert_eq!(map(source).position(0), (1, 1));
        assert_eq!(map(source).position(2), (1, 3));
    }

    #[test]
    fn newline_starts_new_line() {
        let source = "abc\ndef\n";
        assert_eq!(map(source).position(4), (2, 1));
        assert_eq!(map(source).position(6), (2, 3));
    }
}
