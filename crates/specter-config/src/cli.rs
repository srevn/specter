use crate::config::{LogDestination, LogLevel};
use clap::Parser;
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[command(
    name = "specter",
    version,
    about = "Prove the absence of change",
    long_about = None,
)]
pub struct Cli {
    /// Path to TOML config (required).
    #[arg(long, short = 'c')]
    pub config: PathBuf,

    /// Override `[log] level` from config (cli wins).
    #[arg(long, value_enum)]
    pub log_level: Option<LogLevel>,

    /// Override `[log] destination` from config (cli wins). When `file`,
    /// the resolved path must come from either `--log-path` or
    /// `[log] path` in the config.
    #[arg(long, value_enum)]
    pub log_destination: Option<LogDestination>,

    /// Override `[log] path` (must be absolute). Only meaningful when the
    /// resolved destination is `file`.
    #[arg(long)]
    pub log_path: Option<PathBuf>,

    /// Global cap on concurrent Effect spawns. Omit for default (`2 × num_cpus`).
    #[arg(long, value_parser = clap::value_parser!(u32).range(1..))]
    pub concurrency: Option<u32>,

    /// Worker count for the Prober pool. Omit for default (4).
    #[arg(long, value_parser = clap::value_parser!(u32).range(1..))]
    pub probe_concurrency: Option<u32>,

    /// Disable the config-file auto-reload watcher; SIGHUP remains the
    /// only reload trigger.
    ///
    /// Default-on auto-reload covers the common-case operator workflow
    /// (edit + save the running daemon's config and have it pick up
    /// the change). Disable when the config lives on a filesystem
    /// where the watcher's preconditions don't hold:
    ///
    /// - **Network filesystems** (NFS, SMB, CIFS, FUSE-over-network) —
    ///   fanotify / inotify / kqueue do not deliver kernel events for
    ///   server-side mutations; the watcher would init successfully
    ///   but never fire.
    /// - **Symlink-leaf retargeted post-startup** — `canonicalize` runs
    ///   once at watcher init; a later retarget at the leaf leaves the
    ///   watcher pinned to the original inode.
    /// - **Parent dir replaced underneath the watch** — the parent fd
    ///   pins the original parent inode but observes nothing further
    ///   on the new dir.
    ///
    /// Also useful for ops scripts that want strict SIGHUP-only
    /// reload semantics.
    #[arg(long, env = "SPECTER_NO_CONFIG_WATCH")]
    pub no_config_watch: bool,
}

#[cfg(test)]
mod tests {
    use super::Cli;
    use crate::config::LogLevel;
    use clap::{CommandFactory, Parser};

    fn parse(argv: &[&str]) -> Result<Cli, clap::Error> {
        Cli::try_parse_from(argv.iter().copied())
    }

    #[test]
    fn config_only_succeeds() {
        let cli = parse(&["specter", "--config", "/foo"]).unwrap();
        assert_eq!(cli.config, std::path::Path::new("/foo"));
        assert!(cli.log_level.is_none());
        assert!(cli.concurrency.is_none());
        assert!(cli.probe_concurrency.is_none());
        assert!(!cli.no_config_watch, "default-on auto-reload");
    }

    #[test]
    fn missing_config_fails() {
        let err = parse(&["specter"]).unwrap_err();
        assert_eq!(err.kind(), clap::error::ErrorKind::MissingRequiredArgument);
    }

    #[test]
    fn log_level_parses_each_variant() {
        for s in ["trace", "debug", "info", "warn", "error"] {
            let cli = parse(&["specter", "--config", "/foo", "--log-level", s]).unwrap();
            assert!(cli.log_level.is_some(), "log_level `{s}` parses");
        }
    }

    #[test]
    fn log_level_debug_is_debug() {
        let cli = parse(&["specter", "--config", "/foo", "--log-level", "debug"]).unwrap();
        assert_eq!(cli.log_level, Some(LogLevel::Debug));
    }

    #[test]
    fn unknown_log_level_rejected() {
        let err = parse(&["specter", "--config", "/foo", "--log-level", "verbose"]).unwrap_err();
        assert_eq!(err.kind(), clap::error::ErrorKind::InvalidValue);
    }

    #[test]
    fn concurrency_zero_rejected() {
        let err = parse(&["specter", "--config", "/foo", "--concurrency", "0"]).unwrap_err();
        assert_eq!(err.kind(), clap::error::ErrorKind::ValueValidation);
    }

    #[test]
    fn concurrency_positive_accepted() {
        let cli = parse(&["specter", "--config", "/foo", "--concurrency", "16"]).unwrap();
        assert_eq!(cli.concurrency, Some(16));
    }

    #[test]
    fn probe_concurrency_zero_rejected() {
        let err = parse(&["specter", "--config", "/foo", "--probe-concurrency", "0"]).unwrap_err();
        assert_eq!(err.kind(), clap::error::ErrorKind::ValueValidation);
    }

    #[test]
    fn probe_concurrency_positive_accepted() {
        let cli = parse(&["specter", "--config", "/foo", "--probe-concurrency", "8"]).unwrap();
        assert_eq!(cli.probe_concurrency, Some(8));
    }

    #[test]
    fn short_config_flag_works() {
        let cli = parse(&["specter", "-c", "/bar"]).unwrap();
        assert_eq!(cli.config, std::path::Path::new("/bar"));
    }

    #[test]
    fn help_prints_all_flags() {
        let mut cmd = Cli::command();
        let mut buf: Vec<u8> = Vec::new();
        cmd.write_help(&mut buf).unwrap();
        let help = String::from_utf8(buf).unwrap();
        assert!(help.contains("--config"));
        assert!(help.contains("--log-level"));
        assert!(help.contains("--concurrency"));
        assert!(help.contains("--probe-concurrency"));
        assert!(help.contains("--no-config-watch"));
    }

    #[test]
    fn no_config_watch_flag_sets_field() {
        let cli = parse(&["specter", "--config", "/foo", "--no-config-watch"]).unwrap();
        assert!(cli.no_config_watch);
    }

    #[test]
    fn no_config_watch_unset_defaults_to_false() {
        let cli = parse(&["specter", "--config", "/foo"]).unwrap();
        assert!(!cli.no_config_watch);
    }

    #[test]
    fn version_matches_pkg_version() {
        let cmd = Cli::command();
        assert_eq!(cmd.get_version(), Some(env!("CARGO_PKG_VERSION")));
    }

    #[test]
    fn help_flag_is_recognised() {
        let err = parse(&["specter", "--help"]).unwrap_err();
        assert_eq!(err.kind(), clap::error::ErrorKind::DisplayHelp);
    }

    #[test]
    fn version_flag_is_recognised() {
        let err = parse(&["specter", "--version"]).unwrap_err();
        assert_eq!(err.kind(), clap::error::ErrorKind::DisplayVersion);
    }
}
