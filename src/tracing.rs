//! Filterable tracing initialization shared across the CLI binaries.
//!
//! Reads the `RUST_LOG` environment variable to select which compiler
//! subsystems are traced, e.g. `RUST_LOG=pssl_shader::lower::call=trace`.

/// Initializes the `tracing` / `tracing-subscriber` pipeline.
///
/// The default filter is `warn` when `RUST_LOG` is unset, so enabling this in
/// a binary is inert unless the user asks for more verbose logging.
pub fn init_tracing() {
    use tracing_subscriber::EnvFilter;

    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn"));

    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(true)
        .with_line_number(true)
        .init();
}
