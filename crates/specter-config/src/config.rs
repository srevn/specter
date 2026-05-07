use crate::error::{ConfigError, IssueKind, ValidationIssue};
use crate::path::canonicalize_lenient;
use crate::raw::{RawConfig, RawLogConfig, RawWatch};
use crate::template;
use compact_str::CompactString;
use specter_core::{
    self as core, ArgTemplate, ClassSet, CommandTemplate, EffectScope, GlobPattern, ScanConfig,
    SubAttachRequest,
};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

const DEFAULT_SETTLE_MS: u64 = 200;
const SETTLE_FACTOR: u64 = 60;
const MAX_SETTLE_FLOOR_FACTOR: u64 = 4;
const MAX_SETTLE_CEIL_MS: u64 = 3_600_000;

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Config {
    pub log: LogConfig,
    pub watches: Vec<SubSpec>,
}

/// Engine-telemetry configuration — the operator-facing diagnostic
/// stream's level, sink, and (for [`LogDestination::File`]) target path.
///
/// This block is *only* about engine logs. Subprocess output is a
/// separate concern controlled per-watch by [`SubSpec::log_output`].
#[derive(Debug, Default, Clone, Eq, PartialEq)]
pub struct LogConfig {
    pub level: LogLevel,
    pub destination: LogDestination,
    /// Required iff `destination == LogDestination::File`. Validated at
    /// load time: must be absolute. For `LogDestination::Stderr`,
    /// callers should ignore this field.
    pub path: Option<PathBuf>,
}

impl LogConfig {
    /// Merge CLI overrides onto a config-loaded [`LogConfig`].
    ///
    /// Precedence is symmetric for every field: `CLI > config > default`.
    /// When destination resolves to [`LogDestination::File`] but no path
    /// was supplied (neither CLI nor config), returns
    /// [`ConfigError::Validate`] with [`IssueKind::Empty`] on `log.path`.
    /// CLI-supplied paths must be absolute (matching the config-time
    /// rule), or the same error surfaces with [`IssueKind::NonAbsolute`].
    pub fn merge_cli(
        mut self,
        level: Option<LogLevel>,
        destination: Option<LogDestination>,
        path: Option<PathBuf>,
    ) -> Result<Self, ConfigError> {
        if let Some(l) = level {
            self.level = l;
        }
        if let Some(d) = destination {
            self.destination = d;
        }
        if let Some(p) = path {
            self.path = Some(p);
        }
        let mut errors: Vec<ValidationIssue> = Vec::new();
        match (self.destination, self.path.as_deref()) {
            (LogDestination::Stderr, _) => {
                self.path = None;
            }
            (LogDestination::File, None) => errors.push(ValidationIssue::new(
                None,
                "log.path",
                IssueKind::Empty,
                "log.path is required when destination = \"file\" \
                 (provide --log-path or `[log] path` in the config)"
                    .to_owned(),
            )),
            (LogDestination::File, Some(p)) => {
                if !p.is_absolute() {
                    errors.push(ValidationIssue::new(
                        None,
                        "log.path",
                        IssueKind::NonAbsolute,
                        format!("log.path `{}` must be absolute", p.display()),
                    ));
                }
            }
        }
        if errors.is_empty() {
            Ok(self)
        } else {
            Err(ConfigError::Validate { path: None, errors })
        }
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Default, clap::ValueEnum)]
pub enum LogDestination {
    /// Engine telemetry to stderr. Supervisor (systemd / launchd /
    /// FreeBSD `daemon -o`) captures it.
    #[default]
    Stderr,
    /// Engine telemetry to a regular file via `tracing-appender`'s
    /// non-blocking writer. Reopened on SIGHUP for logrotate
    /// `copytruncate`-style rotation.
    File,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct SubSpec {
    pub name: CompactString,
    pub path: PathBuf,
    pub command: CommandTemplate,
    pub scope: EffectScope,
    pub settle: Duration,
    pub max_settle: Duration,
    pub scan: ScanConfig,
    /// User-declared event-class mask. Materialized by `validate_watch`
    /// — explicit when the TOML carries an `events` array, otherwise the
    /// scope-conditional default ([`ClassSet::DEFAULT_SUBTREE_ROOT`] for
    /// `subtree-root`, [`ClassSet::DEFAULT_PER_FILE`] for
    /// `per-stable-file`). Folded into the Profile's `config_hash` by the
    /// engine — `PartialEq`-derived diffs ensure a hot-reload flip on
    /// this field reaps the old Profile and attaches a fresh one.
    pub events: ClassSet,
    /// Forward subprocess stdout/stderr to Specter's own stdio. False by
    /// default — children run with `Stdio::null()`. When true, the
    /// actuator uses `Stdio::inherit()` and the supervisor's log facility
    /// (systemd journal, launchd `StandardOutPath`, FreeBSD `daemon -o`)
    /// captures the bytes. Engine threads this through `SubAttachRequest`
    /// → `Sub.log_output` → `Effect.capture_output`.
    pub log_output: bool,
}

impl SubSpec {
    #[must_use]
    pub fn to_attach_request(&self) -> SubAttachRequest {
        SubAttachRequest::for_path(
            self.name.to_string(),
            self.path.clone(),
            self.scan.clone(),
            self.max_settle,
            self.settle,
            self.command.clone(),
            self.scope,
            self.events,
            self.log_output,
        )
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Default, clap::ValueEnum)]
pub enum LogLevel {
    Trace,
    Debug,
    #[default]
    Info,
    Warn,
    Error,
}

impl LogLevel {
    fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "trace" => Some(Self::Trace),
            "debug" => Some(Self::Debug),
            "info" => Some(Self::Info),
            "warn" | "warning" => Some(Self::Warn),
            "error" => Some(Self::Error),
            _ => None,
        }
    }
}

impl Config {
    /// Parse a TOML string into a validated `Config`.
    ///
    /// Inherent name shadows `std::str::FromStr::from_str` (which is
    /// also implemented for ergonomic `"...".parse::<Config>()` use); both
    /// resolve to the same logic.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Result<Self, ConfigError> {
        Self::from_str_inner(s, None)
    }

    pub fn from_path(path: &Path) -> Result<Self, ConfigError> {
        let s = std::fs::read_to_string(path).map_err(|e| ConfigError::Io {
            path: path.to_owned(),
            source: e,
        })?;
        let cfg = Self::from_str_inner(&s, Some(path))?;
        tracing::info!(
            path = %path.display(),
            watches = cfg.watches.len(),
            "config loaded",
        );
        Ok(cfg)
    }

    fn from_str_inner(s: &str, path: Option<&Path>) -> Result<Self, ConfigError> {
        let raw: RawConfig = toml::from_str(s).map_err(|e| ConfigError::parse(path, &e))?;
        match validate(&raw, path) {
            Ok(cfg) => Ok(cfg),
            Err(e) => {
                if let ConfigError::Validate { errors, .. } = &e {
                    for issue in errors {
                        tracing::warn!(
                            path = ?path.map(Path::display),
                            "{issue}",
                        );
                    }
                }
                Err(e)
            }
        }
    }
}

impl std::str::FromStr for Config {
    type Err = ConfigError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::from_str_inner(s, None)
    }
}

fn validate(raw: &RawConfig, path: Option<&Path>) -> Result<Config, ConfigError> {
    let mut errors: Vec<ValidationIssue> = Vec::new();

    let log = validate_log(raw.log.as_ref(), &mut errors);

    let mut watches: Vec<SubSpec> = Vec::with_capacity(raw.watches.len());
    let mut seen_names: BTreeMap<&str, usize> = BTreeMap::new();
    for (i, raw_w) in raw.watches.iter().enumerate() {
        if let Some(prev) = seen_names.get(raw_w.name.as_str()) {
            errors.push(ValidationIssue::new(
                Some(i),
                "name",
                IssueKind::DuplicateName,
                format!("name `{}` already used by watch[{prev}]", raw_w.name),
            ));
        } else {
            seen_names.insert(raw_w.name.as_str(), i);
        }
        match validate_watch(i, raw_w) {
            Ok(spec) => watches.push(spec),
            Err(mut watch_errors) => errors.append(&mut watch_errors),
        }
    }

    if errors.is_empty() {
        Ok(Config { log, watches })
    } else {
        Err(ConfigError::Validate {
            path: path.map(Path::to_owned),
            errors,
        })
    }
}

/// Resolve the `[log]` block. Field-level errors push into `errors`; the
/// returned [`LogConfig`] uses defaults for any field that failed
/// validation so the rest of the config can keep parsing.
fn validate_log(raw: Option<&RawLogConfig>, errors: &mut Vec<ValidationIssue>) -> LogConfig {
    let Some(raw) = raw else {
        return LogConfig::default();
    };

    let level = match raw.level.as_deref() {
        None => LogLevel::Info,
        Some(s) => match LogLevel::parse(s) {
            Some(lvl) => lvl,
            None => {
                errors.push(ValidationIssue::new(
                    None,
                    "log.level",
                    IssueKind::InvalidEnum,
                    format!("unknown log level `{s}`"),
                ));
                LogLevel::Info
            }
        },
    };

    let destination = match raw.destination.as_deref() {
        None => LogDestination::Stderr,
        Some("stderr") => LogDestination::Stderr,
        Some("file") => LogDestination::File,
        Some(other) => {
            errors.push(ValidationIssue::new(
                None,
                "log.destination",
                IssueKind::InvalidEnum,
                format!("unknown log destination `{other}` (expected `stderr` or `file`)"),
            ));
            LogDestination::Stderr
        }
    };

    let path = match (destination, raw.path.as_deref()) {
        (LogDestination::Stderr, _) => None,
        (LogDestination::File, None) => {
            errors.push(ValidationIssue::new(
                None,
                "log.path",
                IssueKind::Empty,
                "log.path is required when destination = \"file\"".to_owned(),
            ));
            None
        }
        (LogDestination::File, Some(p)) => {
            let pb = PathBuf::from(p);
            if pb.is_absolute() {
                Some(pb)
            } else {
                errors.push(ValidationIssue::new(
                    None,
                    "log.path",
                    IssueKind::NonAbsolute,
                    format!("log.path `{p}` must be absolute"),
                ));
                None
            }
        }
    };

    LogConfig {
        level,
        destination,
        path,
    }
}

fn validate_watch(idx: usize, raw: &RawWatch) -> Result<SubSpec, Vec<ValidationIssue>> {
    let mut errors: Vec<ValidationIssue> = Vec::new();
    let issue = |field: &'static str, kind: IssueKind, detail: String| {
        ValidationIssue::new(Some(idx), field, kind, detail)
    };

    if raw.name.is_empty() {
        errors.push(issue(
            "name",
            IssueKind::Empty,
            "name must not be empty".to_owned(),
        ));
    }

    let path: Option<PathBuf> = if Path::new(&raw.path).is_absolute() {
        match canonicalize_lenient(Path::new(&raw.path)) {
            Ok(p) => Some(p),
            Err(e) => {
                errors.push(issue(
                    "path",
                    IssueKind::NotCanonical,
                    format!("`{}`: {e}", raw.path),
                ));
                None
            }
        }
    } else {
        errors.push(issue(
            "path",
            IssueKind::NonAbsolute,
            format!("path `{}` must be absolute", raw.path),
        ));
        None
    };

    let mut command_failed = false;
    let command: Option<CommandTemplate> = if raw.command.is_empty() {
        errors.push(issue(
            "command",
            IssueKind::EmptyCommand,
            "command must have at least one argv slot".to_owned(),
        ));
        None
    } else {
        let mut argv: Vec<ArgTemplate> = Vec::with_capacity(raw.command.len());
        for (j, slot) in raw.command.iter().enumerate() {
            if slot.is_empty() {
                errors.push(issue(
                    "command",
                    IssueKind::EmptyArgv,
                    format!("argv[{j}] is empty"),
                ));
                command_failed = true;
                continue;
            }
            match template::parse_arg(slot) {
                Ok(arg) => argv.push(arg),
                Err(e) => {
                    errors.push(issue(
                        "command",
                        IssueKind::UnknownPlaceholder,
                        format!("argv[{j}]: {e}"),
                    ));
                    command_failed = true;
                }
            }
        }
        if command_failed {
            None
        } else {
            Some(CommandTemplate::new(argv))
        }
    };

    let settle_ms = raw.settle_ms.unwrap_or(DEFAULT_SETTLE_MS);
    if settle_ms == 0 {
        errors.push(issue(
            "settle_ms",
            IssueKind::SettleTooSmall,
            "settle_ms must be ≥ 1".to_owned(),
        ));
    }
    let max_settle_ms = match raw.max_settle_ms {
        Some(v) => {
            let floor = MAX_SETTLE_FLOOR_FACTOR.saturating_mul(settle_ms);
            if v < floor {
                errors.push(issue(
                    "max_settle_ms",
                    IssueKind::MaxSettleTooSmall,
                    format!("max_settle_ms ({v}) must be ≥ 4 × settle_ms ({floor})"),
                ));
            }
            if v > MAX_SETTLE_CEIL_MS {
                errors.push(issue(
                    "max_settle_ms",
                    IssueKind::MaxSettleTooLarge,
                    format!("max_settle_ms ({v}) must be ≤ {MAX_SETTLE_CEIL_MS} (1 hour)"),
                ));
            }
            v
        }
        None => settle_ms
            .saturating_mul(SETTLE_FACTOR)
            .min(MAX_SETTLE_CEIL_MS),
    };

    let scope = match raw.scope.as_deref().unwrap_or("subtree-root") {
        "subtree-root" => Some(EffectScope::SubtreeRoot),
        "per-stable-file" => Some(EffectScope::PerStableFile),
        other => {
            errors.push(issue(
                "scope",
                IssueKind::InvalidEnum,
                format!("unknown scope `{other}` (expected `subtree-root` or `per-stable-file`)"),
            ));
            None
        }
    };

    // Parse `events` after scope so the default resolver can read scope.
    // If scope itself failed validation, fall back to the default scope
    // (SubtreeRoot) for the events default — this avoids a cascade of
    // phantom errors; the scope error is already collected above.
    let events = parse_events_field(
        raw.events.as_deref(),
        scope.unwrap_or_default(),
        idx,
        &mut errors,
    );

    if raw.max_depth == Some(0) {
        errors.push(issue(
            "max_depth",
            IssueKind::MaxDepthZero,
            "max_depth must be ≥ 1 or omitted (None = unbounded)".to_owned(),
        ));
    }

    let mut sb = ScanConfig::builder()
        .recursive(raw.recursive.unwrap_or(true))
        .hidden(raw.hidden.unwrap_or(false))
        .max_depth(raw.max_depth);

    if let Some(p) = raw.pattern.as_deref() {
        match GlobPattern::compile(p) {
            Ok(g) => sb = sb.pattern(g),
            Err(core::ConfigError::InvalidGlob { message, .. }) => {
                errors.push(issue(
                    "pattern",
                    IssueKind::InvalidGlob,
                    format!("`{p}`: {message}"),
                ));
            }
        }
    }
    if let Some(excs) = raw.exclude.as_deref() {
        let mut compiled = Vec::with_capacity(excs.len());
        for ex in excs {
            match GlobPattern::compile(ex) {
                Ok(g) => compiled.push(g),
                Err(core::ConfigError::InvalidGlob { message, .. }) => {
                    errors.push(issue(
                        "exclude",
                        IssueKind::InvalidGlob,
                        format!("`{ex}`: {message}"),
                    ));
                }
            }
        }
        sb = sb.excludes(compiled);
    }

    if !errors.is_empty() {
        return Err(errors);
    }

    Ok(SubSpec {
        name: CompactString::new(&raw.name),
        path: path.expect("path validated"),
        command: command.expect("command validated"),
        scope: scope.expect("scope validated"),
        settle: Duration::from_millis(settle_ms),
        max_settle: Duration::from_millis(max_settle_ms),
        scan: sb.build(),
        events,
        log_output: raw.log_output.unwrap_or(false),
    })
}

/// Parse the optional TOML `events = [...]` array into a [`ClassSet`].
///
/// - Field omitted → scope-conditional default
///   ([`ClassSet::DEFAULT_SUBTREE_ROOT`] for `subtree-root`,
///   [`ClassSet::DEFAULT_PER_FILE`] for `per-stable-file`).
/// - Empty array → [`IssueKind::EventsEmpty`]. "I want zero classes" can
///   only be a typo; toggling a watch off is removal-by-name.
/// - Unknown value → [`IssueKind::InvalidEnum`].
/// - Repeated value → [`IssueKind::DuplicateEventClass`].
///
/// Issues accumulate into `errors`; the partial [`ClassSet`] returned on
/// the error path is discarded by the caller.
fn parse_events_field(
    raw: Option<&[String]>,
    scope: EffectScope,
    idx: usize,
    errors: &mut Vec<ValidationIssue>,
) -> ClassSet {
    let Some(values) = raw else {
        return match scope {
            EffectScope::SubtreeRoot => ClassSet::DEFAULT_SUBTREE_ROOT,
            EffectScope::PerStableFile => ClassSet::DEFAULT_PER_FILE,
        };
    };

    if values.is_empty() {
        errors.push(ValidationIssue::new(
            Some(idx),
            "events",
            IssueKind::EventsEmpty,
            "events array must not be empty (omit the field to take the \
             scope-conditional default)"
                .to_owned(),
        ));
        return ClassSet::EMPTY;
    }

    let mut out = ClassSet::EMPTY;
    for v in values {
        let bit = match v.as_str() {
            "structure" => ClassSet::STRUCTURE,
            "content" => ClassSet::CONTENT,
            "metadata" => ClassSet::METADATA,
            other => {
                // Surface the whitespace-is-significant case explicitly:
                // serde-toml preserves quoted whitespace, so a raw entry
                // like `events = [" structure "]` reaches us with the
                // padding intact and silently fails the literal match.
                // The hint catches the typo at first glance instead of
                // forcing the operator to inspect quoting rules.
                let trimmed = other.trim();
                let hint = if trimmed != other && !trimmed.is_empty() {
                    format!(
                        "leading/trailing whitespace is significant — \
                         did you mean `{trimmed}`?"
                    )
                } else {
                    "expected `structure`, `content`, or `metadata`".to_owned()
                };
                errors.push(ValidationIssue::new(
                    Some(idx),
                    "events",
                    IssueKind::InvalidEnum,
                    format!("unknown event class `{other}` ({hint})"),
                ));
                continue;
            }
        };
        if out.intersects(bit) {
            errors.push(ValidationIssue::new(
                Some(idx),
                "events",
                IssueKind::DuplicateEventClass,
                format!("event class `{v}` appears more than once"),
            ));
            continue;
        }
        out |= bit;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{Config, LogDestination, LogLevel, SubSpec};
    use crate::error::{ConfigError, IssueKind};
    use specter_core::{ArgPart, ClassSet, EffectScope, Placeholder};
    use std::time::Duration;

    const ROOT: &str = "/";

    fn minimal_toml(extra: &str) -> String {
        format!("[[watch]]\nname = \"build\"\npath = \"{ROOT}\"\ncommand = [\"echo\"]\n{extra}")
    }

    fn validation_errors(err: ConfigError) -> Vec<crate::error::ValidationIssue> {
        match err {
            ConfigError::Validate { errors, .. } => errors,
            other => panic!("expected Validate, got {other:?}"),
        }
    }

    fn assert_only_kind(toml: &str, kind: IssueKind) {
        let err = Config::from_str(toml).unwrap_err();
        let errors = validation_errors(err);
        assert_eq!(
            errors.len(),
            1,
            "expected exactly one issue, got {errors:?}"
        );
        assert_eq!(errors[0].kind, kind);
    }

    #[test]
    fn empty_input_yields_default_log_config_and_no_watches() {
        let cfg = Config::from_str("").unwrap();
        assert_eq!(cfg.log.level, LogLevel::Info);
        assert_eq!(cfg.log.destination, LogDestination::Stderr);
        assert!(cfg.log.path.is_none());
        assert!(cfg.watches.is_empty());
    }

    #[test]
    fn log_level_block_parses_each_variant() {
        for (s, expected) in [
            ("trace", LogLevel::Trace),
            ("debug", LogLevel::Debug),
            ("info", LogLevel::Info),
            ("warn", LogLevel::Warn),
            ("warning", LogLevel::Warn),
            ("error", LogLevel::Error),
        ] {
            let cfg = Config::from_str(&format!("[log]\nlevel = \"{s}\"")).unwrap();
            assert_eq!(cfg.log.level, expected, "input `{s}`");
        }
    }

    #[test]
    fn unknown_log_level_rejected() {
        let err = Config::from_str("[log]\nlevel = \"verbose\"").unwrap_err();
        let errors = validation_errors(err);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, IssueKind::InvalidEnum);
        assert_eq!(errors[0].field, "log.level");
        assert!(errors[0].watch_index.is_none());
    }

    #[test]
    fn legacy_top_level_log_level_is_rejected_as_unknown_field() {
        // Clean alpha break — the old top-level `log_level` field is gone.
        // RawConfig has `deny_unknown_fields`, so the parse fails fast
        // rather than silently dropping the value.
        let err = Config::from_str("log_level = \"debug\"").unwrap_err();
        assert!(matches!(err, ConfigError::Parse { .. }));
    }

    #[test]
    fn log_destination_file_requires_path() {
        let err = Config::from_str("[log]\ndestination = \"file\"").unwrap_err();
        let errors = validation_errors(err);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, IssueKind::Empty);
        assert_eq!(errors[0].field, "log.path");
    }

    #[test]
    fn log_destination_file_with_relative_path_rejected() {
        let err =
            Config::from_str("[log]\ndestination = \"file\"\npath = \"specter.log\"").unwrap_err();
        let errors = validation_errors(err);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, IssueKind::NonAbsolute);
        assert_eq!(errors[0].field, "log.path");
    }

    #[test]
    fn log_destination_file_with_absolute_path_round_trips() {
        let cfg =
            Config::from_str("[log]\ndestination = \"file\"\npath = \"/var/log/specter.log\"")
                .unwrap();
        assert_eq!(cfg.log.destination, LogDestination::File);
        assert_eq!(
            cfg.log.path.as_deref(),
            Some(std::path::Path::new("/var/log/specter.log"))
        );
    }

    #[test]
    fn log_destination_unknown_value_rejected() {
        let err = Config::from_str("[log]\ndestination = \"syslog\"").unwrap_err();
        let errors = validation_errors(err);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, IssueKind::InvalidEnum);
        assert_eq!(errors[0].field, "log.destination");
    }

    #[test]
    fn log_path_ignored_for_stderr_destination() {
        // Stderr path is dropped (set to None) — the operator may have
        // legacy config with `path = ...`; we don't fail validation,
        // because the field carries no meaning when destination = stderr.
        let cfg =
            Config::from_str("[log]\ndestination = \"stderr\"\npath = \"/var/log/ignored.log\"")
                .unwrap();
        assert_eq!(cfg.log.destination, LogDestination::Stderr);
        assert!(
            cfg.log.path.is_none(),
            "path is dropped for stderr destination"
        );
    }

    #[test]
    fn minimal_valid_watch_round_trips_with_defaults() {
        let cfg = Config::from_str(&minimal_toml("")).unwrap();
        assert_eq!(cfg.watches.len(), 1);
        let w: &SubSpec = &cfg.watches[0];
        assert_eq!(w.name, "build");
        assert_eq!(w.scope, EffectScope::SubtreeRoot);
        assert_eq!(w.settle, Duration::from_millis(200));
        assert_eq!(w.max_settle, Duration::from_secs(12));
        assert!(w.scan.recursive);
        assert!(!w.scan.hidden);
        assert!(w.scan.exclude.is_empty());
        assert!(w.scan.pattern.is_none());
        assert_eq!(w.scan.max_depth, None);
        assert_eq!(w.command.argv.len(), 1);
        assert!(!w.log_output, "log_output defaults to false");
    }

    #[test]
    fn log_output_explicit_true_round_trips() {
        let cfg = Config::from_str(&minimal_toml("log_output = true\n")).unwrap();
        assert!(cfg.watches[0].log_output);
    }

    #[test]
    fn log_output_explicit_false_round_trips() {
        let cfg = Config::from_str(&minimal_toml("log_output = false\n")).unwrap();
        assert!(!cfg.watches[0].log_output);
    }

    #[test]
    fn log_output_threads_into_attach_request() {
        let cfg = Config::from_str(&minimal_toml("log_output = true\n")).unwrap();
        let req = cfg.watches[0].to_attach_request();
        assert!(
            req.log_output,
            "SubSpec.log_output reaches SubAttachRequest.log_output via to_attach_request",
        );
    }

    #[test]
    fn empty_name_rejected() {
        let toml = format!("[[watch]]\nname = \"\"\npath = \"{ROOT}\"\ncommand = [\"echo\"]");
        assert_only_kind(&toml, IssueKind::Empty);
    }

    #[test]
    fn relative_path_rejected() {
        let toml = "[[watch]]\nname = \"a\"\npath = \"src\"\ncommand = [\"echo\"]";
        assert_only_kind(toml, IssueKind::NonAbsolute);
    }

    #[test]
    fn empty_command_array_rejected() {
        let toml = format!("[[watch]]\nname = \"a\"\npath = \"{ROOT}\"\ncommand = []");
        assert_only_kind(&toml, IssueKind::EmptyCommand);
    }

    #[test]
    fn empty_argv_slot_rejected() {
        let toml = format!("[[watch]]\nname = \"a\"\npath = \"{ROOT}\"\ncommand = [\"\"]");
        assert_only_kind(&toml, IssueKind::EmptyArgv);
    }

    #[test]
    fn lowercase_typo_placeholder_still_rejected_as_unknown() {
        // Lowercase non-catalog names remain typo errors; the catalog is
        // exclusively lowercase, so a lowercase miss is almost always a typo.
        let toml =
            format!("[[watch]]\nname = \"a\"\npath = \"{ROOT}\"\ncommand = [\"fmt\", \"$paht\"]");
        assert_only_kind(&toml, IssueKind::UnknownPlaceholder);
    }

    #[test]
    fn uppercase_env_var_passes_through_for_shell_expansion() {
        // Env vars (`$SPECTER_PATH`) and conventional shell vars
        // (`$HOME`, `$USER`) must reach the spawned shell unchanged.
        for cmd in [
            "[\"sh\", \"-c\", \"echo $SPECTER_PATH\"]",
            "[\"sh\", \"-c\", \"cd $HOME\"]",
            "[\"sh\", \"-c\", \"echo hi $User\"]",
        ] {
            let toml = format!("[[watch]]\nname = \"a\"\npath = \"{ROOT}\"\ncommand = {cmd}");
            let cfg = Config::from_str(&toml).expect("config should accept uppercase shell vars");
            assert_eq!(cfg.watches.len(), 1);
        }
    }

    #[test]
    fn invalid_pattern_glob_rejected() {
        let toml = minimal_toml("pattern = \"[bad\"\n");
        assert_only_kind(&toml, IssueKind::InvalidGlob);
    }

    #[test]
    fn invalid_exclude_glob_rejected_one_per_bad_entry() {
        let toml = minimal_toml("exclude = [\"[bad\", \"good/**\"]\n");
        let err = Config::from_str(&toml).unwrap_err();
        let errors = validation_errors(err);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, IssueKind::InvalidGlob);
        assert!(errors[0].detail.contains("[bad"));
    }

    #[test]
    fn settle_zero_rejected() {
        let toml = minimal_toml("settle_ms = 0\n");
        assert_only_kind(&toml, IssueKind::SettleTooSmall);
    }

    #[test]
    fn max_settle_below_floor_rejected() {
        let toml = minimal_toml("settle_ms = 100\nmax_settle_ms = 200\n");
        assert_only_kind(&toml, IssueKind::MaxSettleTooSmall);
    }

    #[test]
    fn max_settle_above_ceiling_rejected() {
        let toml = minimal_toml("settle_ms = 100\nmax_settle_ms = 4000000\n");
        assert_only_kind(&toml, IssueKind::MaxSettleTooLarge);
    }

    #[test]
    fn max_settle_default_clamps_to_one_hour() {
        let toml = minimal_toml("settle_ms = 70000\n");
        let cfg = Config::from_str(&toml).unwrap();
        assert_eq!(
            cfg.watches[0].max_settle,
            Duration::from_hours(1),
            "default formula caps at 1 hour even for large settle_ms",
        );
    }

    #[test]
    fn max_depth_zero_rejected() {
        let toml = minimal_toml("max_depth = 0\n");
        assert_only_kind(&toml, IssueKind::MaxDepthZero);
    }

    #[test]
    fn unknown_scope_rejected_as_invalid_enum() {
        let toml = minimal_toml("scope = \"weekly\"\n");
        assert_only_kind(&toml, IssueKind::InvalidEnum);
    }

    #[test]
    fn duplicate_name_rejected_for_each_extra_occurrence() {
        let toml = format!(
            "[[watch]]\nname = \"a\"\npath = \"{ROOT}\"\ncommand = [\"echo\"]\n\
             [[watch]]\nname = \"a\"\npath = \"{ROOT}\"\ncommand = [\"echo\"]\n",
        );
        let err = Config::from_str(&toml).unwrap_err();
        let errors = validation_errors(err);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, IssueKind::DuplicateName);
        assert_eq!(errors[0].watch_index, Some(1));
    }

    #[test]
    fn duplicate_name_three_blocks_yields_two_issues() {
        let toml = format!(
            "[[watch]]\nname = \"a\"\npath = \"{ROOT}\"\ncommand = [\"echo\"]\n\
             [[watch]]\nname = \"a\"\npath = \"{ROOT}\"\ncommand = [\"echo\"]\n\
             [[watch]]\nname = \"a\"\npath = \"{ROOT}\"\ncommand = [\"echo\"]\n",
        );
        let err = Config::from_str(&toml).unwrap_err();
        let errors = validation_errors(err);
        assert_eq!(errors.len(), 2);
        assert!(errors.iter().all(|e| e.kind == IssueKind::DuplicateName));
        assert_eq!(errors[0].watch_index, Some(1));
        assert_eq!(errors[1].watch_index, Some(2));
    }

    #[test]
    fn unknown_top_level_field_yields_parse_error() {
        let err = Config::from_str("foo = \"bar\"").unwrap_err();
        assert!(matches!(err, ConfigError::Parse { .. }));
    }

    #[test]
    fn unknown_watch_field_yields_parse_error() {
        let toml = format!(
            "[[watch]]\nname = \"a\"\npath = \"{ROOT}\"\ncommand = [\"echo\"]\nfoo = \"bar\""
        );
        let err = Config::from_str(&toml).unwrap_err();
        assert!(matches!(err, ConfigError::Parse { .. }));
    }

    #[test]
    fn brace_expansion_pattern_compiles() {
        let toml = minimal_toml("pattern = \"**/*.{c,h,rs}\"\n");
        let cfg = Config::from_str(&toml).unwrap();
        assert!(cfg.watches[0].scan.pattern.is_some());
    }

    #[test]
    fn excludes_sorted_by_source_after_validate() {
        let toml = minimal_toml("exclude = [\"z/**\", \"a/**\", \"m/**\"]\n");
        let cfg = Config::from_str(&toml).unwrap();
        let sources: Vec<&str> = cfg.watches[0]
            .scan
            .exclude
            .iter()
            .map(specter_core::GlobPattern::source)
            .collect();
        assert_eq!(sources, vec!["a/**", "m/**", "z/**"]);
    }

    #[test]
    fn command_template_carries_lexed_argv() {
        let toml = format!(
            "[[watch]]\nname = \"a\"\npath = \"{ROOT}\"\n\
             command = [\"fmt\", \"--input=$path\", \"$created\"]"
        );
        let cfg = Config::from_str(&toml).unwrap();
        let argv = &cfg.watches[0].command.argv;
        assert_eq!(argv.len(), 3);
        assert_eq!(argv[0].parts[0], ArgPart::literal("fmt"));
        assert_eq!(argv[1].parts[0], ArgPart::literal("--input="));
        assert_eq!(argv[1].parts[1], ArgPart::Placeholder(Placeholder::Path));
        assert_eq!(argv[2].parts[0], ArgPart::Placeholder(Placeholder::Created));
    }

    #[test]
    fn multiple_errors_in_one_watch_collected() {
        let toml =
            "[[watch]]\nname = \"\"\npath = \"src\"\ncommand = []\nsettle_ms = 0\nmax_depth = 0";
        let err = Config::from_str(toml).unwrap_err();
        let errors = validation_errors(err);
        let kinds: Vec<IssueKind> = errors.iter().map(|e| e.kind).collect();
        assert!(kinds.contains(&IssueKind::Empty));
        assert!(kinds.contains(&IssueKind::NonAbsolute));
        assert!(kinds.contains(&IssueKind::EmptyCommand));
        assert!(kinds.contains(&IssueKind::SettleTooSmall));
        assert!(kinds.contains(&IssueKind::MaxDepthZero));
        assert_eq!(errors.len(), 5);
    }

    #[test]
    fn errors_across_multiple_watches_preserve_source_order() {
        let toml = "[[watch]]\nname = \"a\"\npath = \"src1\"\ncommand = [\"echo\"]\n\
                    [[watch]]\nname = \"b\"\npath = \"src2\"\ncommand = [\"echo\"]\n";
        let err = Config::from_str(toml).unwrap_err();
        let errors = validation_errors(err);
        assert_eq!(errors.len(), 2);
        assert_eq!(errors[0].watch_index, Some(0));
        assert_eq!(errors[1].watch_index, Some(1));
    }

    #[test]
    fn to_attach_request_uses_for_path_with_canonicalized_path() {
        let cfg = Config::from_str(&minimal_toml("")).unwrap();
        let req = cfg.watches[0].to_attach_request();
        assert_eq!(req.name, "build");
        assert!(req.path.is_some());
        assert_eq!(
            req.path.as_ref().unwrap(),
            &cfg.watches[0].path,
            "request carries the same path stored in SubSpec"
        );
        assert_eq!(
            req.events,
            ClassSet::DEFAULT_SUBTREE_ROOT,
            "to_attach_request threads the parsed events ClassSet through \
             into the engine surface",
        );
    }

    #[test]
    fn events_default_for_subtree_root_scope_is_structure_plus_content() {
        let cfg = Config::from_str(&minimal_toml("scope = \"subtree-root\"\n")).unwrap();
        assert_eq!(cfg.watches[0].events, ClassSet::DEFAULT_SUBTREE_ROOT);
    }

    #[test]
    fn events_default_for_subtree_root_when_scope_omitted() {
        let cfg = Config::from_str(&minimal_toml("")).unwrap();
        assert_eq!(cfg.watches[0].scope, EffectScope::SubtreeRoot);
        assert_eq!(cfg.watches[0].events, ClassSet::DEFAULT_SUBTREE_ROOT);
    }

    #[test]
    fn events_default_for_per_stable_file_scope_is_content_plus_metadata() {
        let cfg = Config::from_str(&minimal_toml("scope = \"per-stable-file\"\n")).unwrap();
        assert_eq!(cfg.watches[0].events, ClassSet::DEFAULT_PER_FILE);
    }

    #[test]
    fn explicit_events_overrides_default() {
        let cfg = Config::from_str(&minimal_toml("events = [\"structure\"]\n")).unwrap();
        assert_eq!(cfg.watches[0].events, ClassSet::STRUCTURE);
    }

    #[test]
    fn explicit_events_accepts_all_three_classes() {
        let cfg = Config::from_str(&minimal_toml(
            "events = [\"structure\", \"content\", \"metadata\"]\n",
        ))
        .unwrap();
        assert_eq!(
            cfg.watches[0].events,
            ClassSet::STRUCTURE | ClassSet::CONTENT | ClassSet::METADATA,
        );
    }

    #[test]
    fn events_explicit_overrides_per_stable_file_default() {
        let cfg = Config::from_str(&minimal_toml(
            "scope = \"per-stable-file\"\nevents = [\"metadata\"]\n",
        ))
        .unwrap();
        assert_eq!(cfg.watches[0].events, ClassSet::METADATA);
    }

    #[test]
    fn unknown_event_class_rejected() {
        let toml = minimal_toml("events = [\"strucutre\"]\n");
        assert_only_kind(&toml, IssueKind::InvalidEnum);
    }

    #[test]
    fn duplicate_event_class_rejected() {
        let toml = minimal_toml("events = [\"structure\", \"structure\"]\n");
        assert_only_kind(&toml, IssueKind::DuplicateEventClass);
    }

    #[test]
    fn empty_events_array_rejected() {
        // Distinct from "field omitted" (which takes the scope default);
        // an explicit empty array is always a typo and earns its own
        // IssueKind.
        let toml = minimal_toml("events = []\n");
        assert_only_kind(&toml, IssueKind::EventsEmpty);
    }

    #[test]
    fn events_unknown_value_does_not_short_circuit_remaining_values() {
        // Unknown values report individually — they don't poison the
        // rest of the array. The watch still fails validation overall,
        // but each issue is collected.
        let toml = minimal_toml("events = [\"strucutre\", \"content\"]\n");
        let err = Config::from_str(&toml).unwrap_err();
        let errors = validation_errors(err);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, IssueKind::InvalidEnum);
        assert!(errors[0].detail.contains("strucutre"));
    }

    #[test]
    fn duplicate_event_class_emits_one_issue_per_extra_occurrence() {
        let toml = minimal_toml("events = [\"content\", \"content\", \"content\"]\n");
        let err = Config::from_str(&toml).unwrap_err();
        let errors = validation_errors(err);
        assert_eq!(errors.len(), 2);
        assert!(
            errors
                .iter()
                .all(|e| e.kind == IssueKind::DuplicateEventClass)
        );
    }

    #[test]
    fn invalid_scope_does_not_cascade_into_events_error() {
        // When scope fails, events falls back to the SubtreeRoot default
        // so we don't double-report a phantom events failure caused by
        // the scope failure.
        let toml = minimal_toml("scope = \"weekly\"\n");
        let err = Config::from_str(&toml).unwrap_err();
        let errors = validation_errors(err);
        assert_eq!(errors.len(), 1, "got {errors:?}");
        assert_eq!(errors[0].kind, IssueKind::InvalidEnum);
        assert_eq!(errors[0].field, "scope");
    }

    #[test]
    fn events_field_value_is_case_sensitive() {
        // TOML enum values are kebab-case throughout — uppercase or
        // mixed case is rejected, matching the existing `scope` parser.
        for bad in ["Structure", "STRUCTURE", "Content", "Meta-Data"] {
            let toml = minimal_toml(&format!("events = [\"{bad}\"]\n"));
            assert_only_kind(&toml, IssueKind::InvalidEnum);
        }
    }

    #[test]
    fn events_field_whitespace_emits_did_you_mean_hint() {
        // serde-toml preserves whitespace inside quoted strings, so
        // `events = [" structure "]` reaches the parser with padding
        // intact. The emitted message must surface the trim hint so
        // operators don't re-read the TOML spec to find the bug.
        for padded in [" structure", "structure ", " structure ", "\tcontent"] {
            let toml = minimal_toml(&format!("events = [\"{padded}\"]\n"));
            let err = Config::from_str(&toml).unwrap_err();
            let errors = validation_errors(err);
            assert_eq!(errors.len(), 1, "got {errors:?}");
            assert_eq!(errors[0].kind, IssueKind::InvalidEnum);
            assert!(
                errors[0].detail.contains("whitespace"),
                "padded `{padded}` should mention whitespace; got `{}`",
                errors[0].detail,
            );
            assert!(
                errors[0].detail.contains(padded.trim()),
                "padded `{padded}` should suggest the trimmed value; got `{}`",
                errors[0].detail,
            );
        }
    }

    #[test]
    fn events_field_non_whitespace_typo_omits_whitespace_hint() {
        // Non-whitespace typos (`strucutre`) get the standard error,
        // not the whitespace-specific hint — keeps the message tight
        // for the common typo case.
        let toml = minimal_toml("events = [\"strucutre\"]\n");
        let err = Config::from_str(&toml).unwrap_err();
        let errors = validation_errors(err);
        assert_eq!(errors.len(), 1, "got {errors:?}");
        assert_eq!(errors[0].kind, IssueKind::InvalidEnum);
        assert!(
            !errors[0].detail.contains("whitespace"),
            "non-whitespace typo should NOT mention whitespace; got `{}`",
            errors[0].detail,
        );
    }

    #[test]
    fn explicit_events_does_not_alter_other_defaults() {
        let cfg = Config::from_str(&minimal_toml("events = [\"structure\"]\n")).unwrap();
        let w = &cfg.watches[0];
        assert_eq!(w.scope, EffectScope::SubtreeRoot);
        assert_eq!(w.settle, Duration::from_millis(200));
        assert_eq!(w.max_settle, Duration::from_secs(12));
        assert!(w.scan.recursive);
    }

    #[test]
    fn pending_path_validates_via_lenient_canonicalize() {
        let td = tempfile::tempdir().unwrap();
        let pending = td.path().join("does-not-exist").join("leaf");
        let toml = format!(
            "[[watch]]\nname = \"p\"\npath = \"{}\"\ncommand = [\"echo\"]",
            pending.display(),
        );
        let cfg = Config::from_str(&toml).unwrap();
        assert!(
            cfg.watches[0]
                .path
                .ends_with(std::path::Path::new("does-not-exist/leaf")),
            "got {}",
            cfg.watches[0].path.display(),
        );
    }
}
