//! `specter-bin` — process orchestration for the specter daemon.
//!
//! Wires the actor crates into a runnable binary: signal pipeline,
//! channel topology, engine driver loop, hot-reload pipeline, and
//! shutdown sequence. The library entry point [`run`] dispatches the
//! parsed top-level `Cli` to the daemon ([`app::run`]) or to one of
//! the operator-client stubs; integration tests drive the lifecycle
//! in-process via this entry point, and production `main` is a thin
//! wrapper over `clap::Parser::parse` + [`run`].

// Bin is wiring: channels, signals, the driver loop. Any FFI need
// lives in the actor crates it composes (`sensor`, `actuator`), never
// here; `forbid` is the strictest level (cannot be locally overridden
// by `#[allow]`), matching the discipline of `core` / `engine` /
// `config`.
#![forbid(unsafe_code)]

mod app;
mod channels;
mod driver;
mod loader;
mod observability;
mod signals;

use specter_config::{Cli, Command};
use std::process::ExitCode;

/// Top-level dispatcher.
///
/// Routes the parsed top-level subcommand to its implementation. Only
/// the `Run` arm has a live implementation today; every operator-
/// client verb surfaces as a stub that prints an "unimplemented" line
/// on stderr and exits `2`.
///
/// `Cli` is taken by value because `app::run` consumes `DaemonArgs`
/// (config moves into the driver; concurrency knobs are extracted
/// then dropped). Client variants are matched-but-not-read here;
/// their arg payloads drop on this stack frame.
#[must_use]
pub fn run(cli: Cli) -> ExitCode {
    match cli.command {
        Command::Run(args) => app::run(args),
        cmd => stub_client(&cmd),
    }
}

/// Stub handler for every client subcommand.
///
/// Exits `2` (the conventional "unimplemented" code) so callers can
/// distinguish from `0` (success) and `1` (startup / runtime failure).
fn stub_client(cmd: &Command) -> ExitCode {
    let verb = match cmd {
        Command::Run(_) => unreachable!("Run is dispatched in `run`"),
        Command::Status(_) => "status",
        Command::List(_) => "list",
        Command::Show(_) => "show",
        Command::Disable(_) => "disable",
        Command::Enable(_) => "enable",
        Command::Reload(_) => "reload",
        Command::Tail(_) => "tail",
        Command::Wait(_) => "wait",
    };
    eprintln!("specter {verb}: unimplemented");
    ExitCode::from(2)
}
