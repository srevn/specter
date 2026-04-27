//! Initialize the global tracing subscriber from a [`LogLevel`].
//!
//! ANSI escapes are auto-detected from `stderr().is_terminal()` so logs
//! piped to journald / syslog stay clean. Source
//! file/line is suppressed; module path is kept (`with_target(true)`)
//! for operator-friendly attribution.
//!
//! `try_init` is tolerant — a second call (e.g., from a re-entrant
//! integration test) returns `Err` from `tracing_subscriber` without
//! panicking. Production `main` calls this once before any thread
//! spawn; tests that need a different subscriber install theirs first
//! and let our `try_init` no-op.

use specter_config::LogLevel;
use std::io::IsTerminal;
use tracing_subscriber::EnvFilter;

/// Install a stderr `fmt` subscriber filtered by `level`.
///
/// Idempotent on re-entry: the second call's `try_init` returns `Err`
/// silently.
pub fn init_tracing(level: LogLevel) {
    let filter = EnvFilter::new(log_level_directive(level));
    let ansi = std::io::stderr().is_terminal();
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(true)
        .with_thread_names(true)
        .with_file(false)
        .with_line_number(false)
        .with_ansi(ansi)
        .try_init();
}

/// Map a [`LogLevel`] enum variant to the `EnvFilter` directive
/// `tracing-subscriber` expects. Returns a `'static` string so call
/// sites can pass directly without intermediate allocation.
#[must_use]
pub const fn log_level_directive(level: LogLevel) -> &'static str {
    match level {
        LogLevel::Trace => "trace",
        LogLevel::Debug => "debug",
        LogLevel::Info => "info",
        LogLevel::Warn => "warn",
        LogLevel::Error => "error",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn directive_for_each_level() {
        assert_eq!(log_level_directive(LogLevel::Trace), "trace");
        assert_eq!(log_level_directive(LogLevel::Debug), "debug");
        assert_eq!(log_level_directive(LogLevel::Info), "info");
        assert_eq!(log_level_directive(LogLevel::Warn), "warn");
        assert_eq!(log_level_directive(LogLevel::Error), "error");
    }

    #[test]
    fn init_tracing_is_idempotent() {
        // First call may or may not succeed depending on test ordering
        // (cargo test shares process; another test or a global init may
        // have set the subscriber). Both calls below must not panic.
        init_tracing(LogLevel::Warn);
        init_tracing(LogLevel::Debug);
    }
}
