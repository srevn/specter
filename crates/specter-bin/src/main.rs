//! `specter` binary entry point — clap parse + delegate to [`specter_bin::run`].
//!
//! All lifecycle logic lives in the library (`src/lib.rs` + sibling
//! modules) so integration tests can exercise it in-process without
//! shelling out to the binary.

use clap::Parser;
use specter_config::Cli;
use std::process::ExitCode;

fn main() -> ExitCode {
    let cli = Cli::parse();
    specter_bin::run(cli)
}
