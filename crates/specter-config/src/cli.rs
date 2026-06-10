use crate::config::{LogDestination, LogLevel};
use clap::builder::{
    Styles,
    styling::{AnsiColor, Style},
};
use clap::{Args, Parser, Subcommand, ValueEnum};
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::time::Duration;

const HEADING: Style = AnsiColor::Yellow.on_default().bold();
const LITERAL: Style = AnsiColor::Cyan.on_default().bold();
const PLACEHOLDER: Style = AnsiColor::Green.on_default();

const STYLES: Styles = Styles::styled()
    .header(HEADING)
    .usage(HEADING)
    .literal(LITERAL)
    .placeholder(PLACEHOLDER);

fn banner() -> String {
    format!(
        "{}specter{} — prove the absence of change",
        LITERAL.render(),
        LITERAL.render_reset(),
    )
}

/// Top-level CLI parser — subcommand dispatcher.
///
/// `specter run` is the daemon (the historical flat invocation, now under a subcommand). The other
/// verbs are operator clients that connect to the running daemon over a UNIX socket. The client
/// surface is declared here so `--help` exposes it; verbs without a live handler exit `2`.
#[derive(Debug, Parser)]
#[command(
    name = "specter",
    version,
    about = banner(),
    styles = STYLES,
    subcommand_required = true,
    arg_required_else_help = true,
)]
#[must_use]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

/// One-of-N top-level subcommand.
///
/// `Run` carries the daemon arguments; every other variant carries a client-side argument struct
/// (always including [`ClientArgs`] via `#[command(flatten)]` for `--socket`).
#[derive(Debug, Subcommand)]
pub enum Command {
    /// Run the specter daemon.
    Run(DaemonArgs),
    /// Print process / lifecycle status from the running daemon.
    Status(StatusArgs),
    /// List every attached watch and its state.
    List(ListArgs),
    /// Show one watch in detail by name.
    Show(ShowArgs),
    /// Disable one watch by name (runtime override).
    Disable(NameTargetArgs),
    /// Enable a watch previously disabled via `specter disable`.
    Enable(NameTargetArgs),
    /// Absorb the next change on a watch instead of firing.
    Absorb(AbsorbArgs),
    /// Request a config reload (equivalent to SIGHUP).
    Reload(ClientArgs),
    /// Stream diagnostics from the daemon.
    Tail(TailArgs),
    /// Block until a watch fires or detaches.
    Wait(WaitArgs),
}

/// Daemon arguments — the historical flat `Cli` fields, preserved under `Command::Run`.
#[derive(Debug, Args)]
#[must_use]
pub struct DaemonArgs {
    /// Path to TOML config (required).
    #[arg(long, short = 'c')]
    pub config: PathBuf,

    /// IPC socket path to bind.
    ///
    /// Omitted ⇒ the per-platform convention; `$SPECTER_SOCK` overrides that and `--socket`
    /// overrides both. Must be absolute.
    #[arg(long)]
    pub socket: Option<PathBuf>,

    /// Override log level from config.
    #[arg(long, value_enum)]
    pub log_level: Option<LogLevel>,

    /// Override log destination from config.
    ///
    /// When `file`, the resolved path must come from either `--log-path` or `[log] path` in the
    /// config.
    #[arg(long, value_enum)]
    pub log_destination: Option<LogDestination>,

    /// Override log file path (must be absolute).
    ///
    /// Only meaningful when the resolved destination is `file`.
    #[arg(long)]
    pub log_path: Option<PathBuf>,

    /// Global cap on concurrent Effect spawns.
    ///
    /// Omit for default (`2 × num_cpus`).
    #[arg(long, value_parser = clap::value_parser!(NonZeroUsize))]
    pub concurrency: Option<NonZeroUsize>,

    /// Worker count for the Prober pool.
    ///
    /// Omit for default (4).
    #[arg(long, value_parser = clap::value_parser!(NonZeroUsize))]
    pub probe_concurrency: Option<NonZeroUsize>,

    /// Disable the config-file auto-reload watcher.
    ///
    /// SIGHUP remains the only reload trigger.
    ///
    /// Default-on auto-reload covers the common-case operator workflow (edit + save the running
    /// daemon's config and have it pick up the change). Disable when the config lives on a
    /// filesystem where the watcher's preconditions don't hold:
    ///
    /// - Network filesystems (NFS, SMB, CIFS, FUSE-over-network) — fanotify / inotify / kqueue do
    ///   not deliver kernel events for server-side mutations; the watcher would init successfully
    ///   but never fire.
    ///
    /// - Symlink-leaf retargeted post-startup — `canonicalize` runs once at watcher init; a later
    ///   retarget at the leaf leaves the watcher pinned to the original inode.
    ///
    /// - Parent dir replaced underneath the watch — the parent fd pins the original parent inode
    ///   but observes nothing further on the new dir.
    ///
    /// Also useful for ops scripts that want strict SIGHUP-only reload semantics.
    #[arg(long, env = "SPECTER_NO_CONFIG_WATCH")]
    pub no_config_watch: bool,
}

/// Arguments common to every client subcommand. Flattened via `#[command(flatten)]` so `--socket`
/// reads naturally on each verb.
#[derive(Debug, Args)]
#[must_use]
pub struct ClientArgs {
    /// Daemon IPC socket path to connect to.
    ///
    /// Omitted ⇒ probe the per-platform convention; `$SPECTER_SOCK` overrides that and `--socket`
    /// overrides both. Must be absolute.
    #[arg(long)]
    pub socket: Option<PathBuf>,

    /// ANSI color policy for client output.
    ///
    /// `auto` styles only when the target stream is a terminal and the environment allows it
    /// (`NO_COLOR` / `CLICOLOR` / `CLICOLOR_FORCE`); `always` / `never` override that gate. Stdout
    /// and stderr resolve independently; `-o json` is never styled.
    #[arg(long, value_enum, default_value_t = ColorWhen::Auto)]
    pub color: ColorWhen,
}

/// `specter status` arguments.
#[derive(Debug, Args)]
#[must_use]
pub struct StatusArgs {
    #[command(flatten)]
    pub client: ClientArgs,

    /// Output format.
    #[arg(long, short = 'o', value_enum, default_value_t = OutputFormat::Human)]
    pub output: OutputFormat,

    /// Include rarely-needed fields (counters, ids, full paths). Only affects `-o human`; `-o json`
    /// is always lossless.
    #[arg(long)]
    pub wide: bool,
}

/// `specter list` arguments.
#[derive(Debug, Args)]
#[must_use]
pub struct ListArgs {
    #[command(flatten)]
    pub client: ClientArgs,

    /// Output format.
    #[arg(long, short = 'o', value_enum, default_value_t = OutputFormat::Human)]
    pub output: OutputFormat,

    /// Include rarely-needed columns (profile/sub ids, dedup count, settle ms). Only affects `-o
    /// human`; `-o json` is always lossless.
    #[arg(long)]
    pub wide: bool,
}

/// `specter show <name>` arguments.
#[derive(Debug, Args)]
#[must_use]
pub struct ShowArgs {
    /// Name of the watch to show.
    pub name: String,

    #[command(flatten)]
    pub client: ClientArgs,

    /// Output format.
    #[arg(long, short = 'o', value_enum, default_value_t = OutputFormat::Human)]
    pub output: OutputFormat,
}

/// `specter disable <name>` / `specter enable <name>` arguments — identical shape; the verb itself
/// selects the operation.
#[derive(Debug, Args)]
#[must_use]
pub struct NameTargetArgs {
    /// Name of the watch to act on.
    pub name: String,

    #[command(flatten)]
    pub client: ClientArgs,
}

/// `specter absorb <name> [--for <dur>]` arguments — arm a fold-without-fire window on the named
/// watch's Profile.
#[derive(Debug, Args)]
#[must_use]
pub struct AbsorbArgs {
    /// Name of the watch to absorb on.
    pub name: String,

    #[command(flatten)]
    pub client: ClientArgs,

    /// Window length. Omitted ⇒ a one-shot window covering the next change; `--for <dur>` holds it
    /// open to absorb a run of changes. humantime format (`500ms`, `30s`, `1m30s`).
    #[arg(long = "for", value_parser = parse_duration)]
    pub for_: Option<Duration>,
}

/// `specter tail` arguments.
#[derive(Debug, Args)]
#[must_use]
pub struct TailArgs {
    #[command(flatten)]
    pub client: ClientArgs,

    /// Restrict the stream to one or more `WireDiagnostic` variant names (e.g. `SubFired`,
    /// `SubDetached`). Repeatable; case- sensitive. Empty (the default) streams every variant.
    #[arg(long)]
    pub filter: Vec<String>,

    /// Output format. `human` pretty-prints one event per line; `json` emits the lossless wire
    /// shape (one JSON object per line, symmetric with the daemon's emission).
    #[arg(long, short = 'o', value_enum, default_value_t = OutputFormat::Human)]
    pub output: OutputFormat,
}

/// `specter wait <name>` arguments.
#[derive(Debug, Args)]
#[must_use]
pub struct WaitArgs {
    /// Name of the watch to wait on.
    pub name: String,

    #[command(flatten)]
    pub client: ClientArgs,

    /// Event class to wait for. `fire` (default) matches `SubFired`; `detach` matches `SubDetached`.
    #[arg(long, value_enum, default_value_t = WaitKind::Fire)]
    pub kind: WaitKind,

    /// Time budget. Omitted ⇒ wait indefinitely. humantime format (`500ms`, `30s`, `1m30s`).
    #[arg(long, value_parser = parse_duration)]
    pub timeout: Option<Duration>,
}

/// Output format shared by `status` / `list` / `show`.
#[derive(Debug, Copy, Clone, Eq, PartialEq, ValueEnum)]
pub enum OutputFormat {
    /// Human-readable rendering (default).
    Human,
    /// Lossless JSON, one object per response.
    Json,
}

/// When to colorize client output — shared by every verb via [`ClientArgs`].
///
/// The renderer-side resolution (env precedence, TTY detection, the `Styler` it produces) lives in
/// `specter-bin`'s `ipc::render::style`; this enum is only the operator's stated preference.
#[derive(Debug, Copy, Clone, Eq, PartialEq, ValueEnum)]
pub enum ColorWhen {
    /// Style only when the target stream is a terminal and the environment permits it.
    Auto,
    /// Always style, regardless of stream or environment.
    Always,
    /// Never style.
    Never,
}

/// Event class for `specter wait`.
#[derive(Debug, Copy, Clone, Eq, PartialEq, ValueEnum)]
pub enum WaitKind {
    /// Match `SubFired` — the Sub emitted at least one Effect.
    Fire,
    /// Match `SubDetached` — the Sub left the engine (IPC `disable`, config-removal, or
    /// `modified_identity` rebind).
    Detach,
}

/// clap value-parser bridge for [`Duration`] in humantime form. The `String` error path is what
/// clap's `value_parser` expects — it surfaces in the user's CLI error verbatim.
fn parse_duration(s: &str) -> Result<Duration, String> {
    humantime::parse_duration(s).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::{CommandFactory, Parser};

    fn parse(argv: &[&str]) -> Result<Cli, clap::Error> {
        Cli::try_parse_from(argv.iter().copied())
    }

    fn run_args(cli: Cli) -> DaemonArgs {
        match cli.command {
            Command::Run(args) => args,
            other => panic!("expected Run subcommand, got {other:?}"),
        }
    }

    #[test]
    fn run_parses_full_daemon_flag_set() {
        let cli = parse(&[
            "specter",
            "run",
            "--config",
            "/etc/specter.toml",
            "--socket",
            "/run/specter/specter.sock",
            "--log-level",
            "debug",
            "--log-destination",
            "stderr",
            "--log-path",
            "/var/log/specter.log",
            "--concurrency",
            "8",
            "--probe-concurrency",
            "16",
            "--no-config-watch",
        ])
        .unwrap();
        let args = run_args(cli);
        assert_eq!(args.config, std::path::Path::new("/etc/specter.toml"));
        assert_eq!(
            args.socket.as_deref(),
            Some(std::path::Path::new("/run/specter/specter.sock"))
        );
        assert_eq!(args.log_level, Some(LogLevel::Debug));
        assert_eq!(args.log_destination, Some(LogDestination::Stderr));
        assert_eq!(
            args.log_path.as_deref(),
            Some(std::path::Path::new("/var/log/specter.log"))
        );
        assert_eq!(args.concurrency, NonZeroUsize::new(8));
        assert_eq!(args.probe_concurrency, NonZeroUsize::new(16));
        assert!(args.no_config_watch);
    }

    #[test]
    fn run_defaults_no_config_watch_to_false() {
        let cli = parse(&["specter", "run", "--config", "/foo"]).unwrap();
        let args = run_args(cli);
        assert!(!args.no_config_watch);
        assert!(args.log_level.is_none());
    }

    #[test]
    fn run_short_config_flag_works() {
        let cli = parse(&["specter", "run", "-c", "/bar"]).unwrap();
        assert_eq!(run_args(cli).config, std::path::Path::new("/bar"));
    }

    #[test]
    fn run_missing_config_fails() {
        let err = parse(&["specter", "run"]).unwrap_err();
        assert_eq!(err.kind(), clap::error::ErrorKind::MissingRequiredArgument);
    }

    #[test]
    fn run_concurrency_zero_rejected() {
        let err = parse(&["specter", "run", "--config", "/foo", "--concurrency", "0"]).unwrap_err();
        assert_eq!(err.kind(), clap::error::ErrorKind::ValueValidation);
    }

    #[test]
    fn no_subcommand_shows_help() {
        let err = parse(&["specter"]).unwrap_err();
        assert_eq!(
            err.kind(),
            clap::error::ErrorKind::DisplayHelpOnMissingArgumentOrSubcommand,
        );
    }

    #[test]
    fn status_parses_with_wide_and_output() {
        let cli = parse(&[
            "specter",
            "status",
            "--socket",
            "/tmp/s.sock",
            "-o",
            "json",
            "--wide",
        ])
        .unwrap();
        match cli.command {
            Command::Status(args) => {
                assert_eq!(
                    args.client.socket.as_deref(),
                    Some(std::path::Path::new("/tmp/s.sock"))
                );
                assert_eq!(args.output, OutputFormat::Json);
                assert!(args.wide);
            }
            other => panic!("expected Status, got {other:?}"),
        }
    }

    #[test]
    fn list_parses_with_wide_and_output() {
        let cli = parse(&[
            "specter",
            "list",
            "--socket",
            "/tmp/s.sock",
            "-o",
            "json",
            "--wide",
        ])
        .unwrap();
        match cli.command {
            Command::List(args) => {
                assert_eq!(
                    args.client.socket.as_deref(),
                    Some(std::path::Path::new("/tmp/s.sock"))
                );
                assert_eq!(args.output, OutputFormat::Json);
                assert!(args.wide);
            }
            other => panic!("expected List, got {other:?}"),
        }
    }

    #[test]
    fn show_requires_name() {
        let err = parse(&["specter", "show"]).unwrap_err();
        assert_eq!(err.kind(), clap::error::ErrorKind::MissingRequiredArgument);
    }

    #[test]
    fn wait_parses_kind_and_timeout() {
        let cli = parse(&[
            "specter",
            "wait",
            "my-watch",
            "--kind",
            "detach",
            "--timeout",
            "1500ms",
        ])
        .unwrap();
        match cli.command {
            Command::Wait(args) => {
                assert_eq!(args.name, "my-watch");
                assert_eq!(args.kind, WaitKind::Detach);
                assert_eq!(args.timeout, Some(Duration::from_millis(1500)));
            }
            other => panic!("expected Wait, got {other:?}"),
        }
    }

    #[test]
    fn wait_rejects_invalid_timeout() {
        let err =
            parse(&["specter", "wait", "my-watch", "--timeout", "not-a-duration"]).unwrap_err();
        assert_eq!(err.kind(), clap::error::ErrorKind::ValueValidation);
    }

    #[test]
    fn absorb_without_for_defaults_to_none() {
        let cli = parse(&["specter", "absorb", "my-watch"]).unwrap();
        match cli.command {
            Command::Absorb(args) => {
                assert_eq!(args.name, "my-watch");
                assert_eq!(args.for_, None, "omitted --for ⇒ None (engine default)");
            }
            other => panic!("expected Absorb, got {other:?}"),
        }
    }

    #[test]
    fn absorb_parses_for_duration() {
        let cli = parse(&["specter", "absorb", "my-watch", "--for", "5s"]).unwrap();
        match cli.command {
            Command::Absorb(args) => {
                assert_eq!(args.name, "my-watch");
                assert_eq!(args.for_, Some(Duration::from_secs(5)));
            }
            other => panic!("expected Absorb, got {other:?}"),
        }
    }

    #[test]
    fn absorb_rejects_invalid_for_duration() {
        let err = parse(&["specter", "absorb", "my-watch", "--for", "not-a-duration"]).unwrap_err();
        assert_eq!(err.kind(), clap::error::ErrorKind::ValueValidation);
    }

    #[test]
    fn tail_collects_repeated_filter() {
        let cli = parse(&[
            "specter",
            "tail",
            "--filter",
            "SubFired",
            "--filter",
            "SubDetached",
        ])
        .unwrap();
        match cli.command {
            Command::Tail(args) => {
                assert_eq!(args.filter, vec!["SubFired", "SubDetached"]);
            }
            other => panic!("expected Tail, got {other:?}"),
        }
    }

    #[test]
    fn tail_output_defaults_to_human() {
        let cli = parse(&["specter", "tail"]).unwrap();
        match cli.command {
            Command::Tail(args) => {
                assert_eq!(args.output, OutputFormat::Human);
                assert!(args.filter.is_empty(), "default filter is empty");
            }
            other => panic!("expected Tail, got {other:?}"),
        }
    }

    #[test]
    fn tail_parses_output_format() {
        let cli = parse(&["specter", "tail", "-o", "json"]).unwrap();
        match cli.command {
            Command::Tail(args) => assert_eq!(args.output, OutputFormat::Json),
            other => panic!("expected Tail, got {other:?}"),
        }
    }

    #[test]
    fn top_level_help_lists_every_subcommand() {
        let mut cmd = Cli::command();
        let mut buf: Vec<u8> = Vec::new();
        cmd.write_long_help(&mut buf).unwrap();
        let help = String::from_utf8(buf).unwrap();
        for verb in [
            "run", "status", "list", "show", "disable", "enable", "absorb", "reload", "tail",
            "wait",
        ] {
            assert!(help.contains(verb), "top-level help missing `{verb}`");
        }
    }

    #[test]
    fn version_flag_recognised_at_top_level() {
        let err = parse(&["specter", "--version"]).unwrap_err();
        assert_eq!(err.kind(), clap::error::ErrorKind::DisplayVersion);
    }

    /// `--color` rides on every client verb via the flattened [`ClientArgs`], defaults to
    /// [`ColorWhen::Auto`], and parses the `always` / `never` overrides. Pins the flatten + field
    /// wiring the renderer-side `style::resolve` reads.
    #[test]
    fn color_flag_defaults_auto_and_parses_overrides() {
        let cli = parse(&["specter", "status"]).unwrap();
        match cli.command {
            Command::Status(args) => assert_eq!(args.client.color, ColorWhen::Auto),
            other => panic!("expected Status, got {other:?}"),
        }

        let cli = parse(&["specter", "list", "--color", "always"]).unwrap();
        match cli.command {
            Command::List(args) => assert_eq!(args.client.color, ColorWhen::Always),
            other => panic!("expected List, got {other:?}"),
        }

        let cli = parse(&["specter", "tail", "--color", "never"]).unwrap();
        match cli.command {
            Command::Tail(args) => assert_eq!(args.client.color, ColorWhen::Never),
            other => panic!("expected Tail, got {other:?}"),
        }
    }
}
