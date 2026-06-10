//! `specter-bin` — process orchestration for the specter daemon.
//!
//! Wires the actor crates into a runnable binary: signal pipeline, channel topology, engine driver
//! loop, hot-reload pipeline, and shutdown sequence. The library entry point [`run`] dispatches the
//! parsed top-level `Cli` to the daemon (`app::run`) or to one of the operator-client handlers;
//! integration tests drive the lifecycle in-process via this entry point, and production `main` is
//! a thin wrapper over `clap::Parser::parse` + [`run`].

// Bin is wiring: channels, signal pipeline, the driver loop. Any FFI need lives in the actor crates
// it composes (`sensor`, `actuator`), never here; `forbid` is the strictest level (cannot be
// locally overridden by `#[allow]`), matching the discipline of `core` / `engine` / `config`.
#![forbid(unsafe_code)]

mod actuator;
mod app;
mod driver;
mod ipc;
mod loader;
mod observability;
mod signals;

use specter_config::{Cli, Command};
use std::process::ExitCode;

/// Top-level dispatcher.
///
/// Routes every parsed top-level subcommand to its implementation: `Run` drives the daemon; the
/// remaining variants are operator- client one-shots that connect to the running daemon over the
/// IPC socket.
///
/// `Cli` is taken by value because `app::run` consumes `DaemonArgs` (config moves into the driver;
/// concurrency knobs are extracted then dropped). The client arms borrow their args.
///
/// The match is exhaustive without a catch-all — adding a new [`Command`] variant is a compile
/// error here, surfacing the missing handler at compile time rather than at runtime.
#[must_use]
pub fn run(cli: Cli) -> ExitCode {
    match cli.command {
        Command::Run(args) => app::run(args),
        Command::Status(args) => ipc::client::status::run(&args),
        Command::List(args) => ipc::client::list::run(&args),
        Command::Show(args) => ipc::client::show::run(&args),
        Command::Disable(args) => ipc::client::disable::run(&args),
        Command::Enable(args) => ipc::client::enable::run(&args),
        Command::Absorb(args) => ipc::client::absorb::run(&args),
        Command::Reload(args) => ipc::client::reload::run(&args),
        Command::Tail(args) => ipc::client::tail::run(&args),
        Command::Wait(args) => ipc::client::wait::run(&args),
    }
}
