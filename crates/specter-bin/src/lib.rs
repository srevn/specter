//! `specter-bin` — process orchestration for the specter daemon.
//!
//! Wires the actor crates into a runnable binary: signal pipeline,
//! channel topology, engine driver loop, hot-reload pipeline, and
//! shutdown sequence. The library entry point [`run`] is exposed
//! so integration tests can drive the lifecycle in-process; production
//! `main` is a thin wrapper over `clap::Parser::parse` + [`run`].

mod app;
mod channels;
mod driver;
mod loader;
mod signals;
mod tracing_init;

pub use app::run;
