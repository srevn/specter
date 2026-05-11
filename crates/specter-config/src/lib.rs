//! `specter-config` — TOML parsing, validation, hot-reload diff.
//!
//! Stateless translation layer. Inputs: TOML strings, file paths, CLI argv.
//! Outputs: [`Config`], [`SubRegistryDiff`](specter_core::SubRegistryDiff),
//! [`Cli`]. No engine or actor deps.

mod action;
mod cli;
mod config;
mod diff;
mod error;
mod file_meta;
mod path;
mod raw;
mod template;

pub use cli::Cli;
pub use config::{Config, LogConfig, LogDestination, LogLevel, PromoterSpec, SubSpec};
pub use diff::diff;
pub use error::{ConfigError, IssueKind, ValidationIssue};
pub use file_meta::FileMeta;
pub use path::{PathError, canonicalize_lenient};
pub use template::{TemplateError, parse_arg};
