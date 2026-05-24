//! `specter-bin` — process orchestration for the specter daemon.
//!
//! Wires the actor crates into a runnable binary: signal pipeline,
//! channel topology, engine driver loop, hot-reload pipeline, and
//! shutdown sequence. The library entry point [`run`] is exposed
//! so integration tests can drive the lifecycle in-process; production
//! `main` is a thin wrapper over `clap::Parser::parse` + [`run`].

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

pub use app::run;
