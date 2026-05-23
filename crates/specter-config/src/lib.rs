//! `specter-config` — TOML parsing, validation, hot-reload diff.
//!
//! Stateless translation layer. Inputs: TOML strings, file paths, CLI argv.
//! Outputs: [`Config`], [`SubRegistryDiff`](specter_core::SubRegistryDiff),
//! [`Cli`]. No engine or actor deps.

// Config is pure data — TOML parse + clap argv + a SubRegistryDiff. No
// FFI need exists or is foreseeable; `forbid` is the strictest level
// (cannot be locally overridden by `#[allow]`), matching the
// discipline of `core` / `engine`.
#![forbid(unsafe_code)]

mod action;
mod cli;
mod config;
mod diff;
mod error;
mod file_meta;
mod path;
mod raw;
mod template;

pub use cli::{
    Cli, ClientArgs, Command, DaemonArgs, ListArgs, NameTargetArgs, OutputFormat, ShowArgs,
    StatusArgs, TailArgs, WaitArgs, WaitKind,
};
pub use config::{Config, LogConfig, LogDestination, LogLevel, PromoterSpec, SubSpec};
pub use diff::diff;
pub use error::{ConfigError, IssueKind, ValidationIssue};
pub use file_meta::FileMeta;
pub use template::{TemplateError, parse_arg};
