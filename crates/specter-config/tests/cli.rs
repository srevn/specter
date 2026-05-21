//! Integration tests: clap-derived `Cli` parsing.

use clap::{CommandFactory, Parser};
use specter_config::Cli;
use std::num::NonZeroUsize;

#[test]
fn parse_full_set_of_flags() {
    let cli = Cli::try_parse_from([
        "specter",
        "--config",
        "/etc/specter.toml",
        "--log-level",
        "warn",
        "--concurrency",
        "8",
        "--probe-concurrency",
        "16",
    ])
    .unwrap();
    assert_eq!(cli.config, std::path::Path::new("/etc/specter.toml"));
    assert_eq!(cli.concurrency, NonZeroUsize::new(8));
    assert_eq!(cli.probe_concurrency, NonZeroUsize::new(16));
}

#[test]
fn parse_help_then_version_distinct_kinds() {
    let h = Cli::try_parse_from(["specter", "--help"]).unwrap_err();
    let v = Cli::try_parse_from(["specter", "--version"]).unwrap_err();
    assert_eq!(h.kind(), clap::error::ErrorKind::DisplayHelp);
    assert_eq!(v.kind(), clap::error::ErrorKind::DisplayVersion);
}

#[test]
fn long_help_contains_all_flag_descriptions() {
    let mut cmd = Cli::command();
    let mut buf = Vec::new();
    cmd.write_long_help(&mut buf).unwrap();
    let help = String::from_utf8(buf).unwrap();
    assert!(help.contains("--config"));
    assert!(help.contains("--log-level"));
    assert!(help.contains("--concurrency"));
    assert!(help.contains("--probe-concurrency"));
}
