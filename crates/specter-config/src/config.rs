use crate::action::{Action, lower_to_program};
use crate::error::{ConfigError, IssueKind, ValidationIssue};
use crate::file_meta::FileMeta;
use crate::path::canonicalize_lenient;
use crate::raw::{RawAction, RawConfig, RawExec, RawLogConfig, RawWatch};
use crate::template;
use compact_str::CompactString;
use specter_core::{
    self as core, ActionProgram, ArgTemplate, ClassSet, EffectScope, ExecAction, GlobPattern,
    PatternSpec, PromoterAttachRequest, ScanConfig, SubAttachRequest,
};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

/// Default debounce window when `[[watch]] settle` is omitted.
pub(crate) const DEFAULT_SETTLE: Duration = Duration::from_millis(200);
/// Default forced-fire deadline when `[[watch]] max_settle` is omitted.
/// Flat 1 hour, independent of `settle` — if the tree stays active for an
/// hour, the user's workflow is outside Specter's scope; manual triggering
/// is the better answer.
pub(crate) const DEFAULT_MAX_SETTLE: Duration = Duration::from_hours(1);
/// Lower bound on `max_settle` relative to `settle`. Catches the swap
/// typo (`settle = "1h"`, `max_settle = "200ms"`) and the semantic
/// nonsense of `max_settle ≤ settle` (a single settle round would
/// already exceed it).
const MAX_SETTLE_FLOOR_FACTOR: u32 = 4;

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Config {
    pub log: LogConfig,
    /// Static `[[watch]]` blocks — paths without glob discriminator
    /// characters (`*?[{`). Each entry maps to one [`SubSpec`] and is
    /// attached as a Sub by the bin's initial-attach pass.
    pub watches: Vec<SubSpec>,
    /// Dynamic `[[watch]]` blocks — paths with glob discriminator
    /// characters routed via [`PatternSpec::is_dynamic`]. Each entry
    /// maps to one [`PromoterSpec`] which the engine treats as a
    /// pattern source: matched paths become synthesized dynamic Subs
    /// via the Promoter lifecycle. Schema is unified — there is no
    /// `[[promoter]]` table; the dispatch happens on `path`.
    pub promoters: Vec<PromoterSpec>,
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
    /// Lowered bytecode IR. Built once at config validation; cloned by
    /// Arc into each [`SubAttachRequest`] (and from there into every
    /// emitted `Effect`). Equality is structural over the instruction
    /// sequence — two TOML configs that lower to the same program
    /// compare equal, so the hot-reload diff suppresses no-op churn on
    /// cosmetic edits (whitespace, comment, key ordering).
    pub program: Arc<ActionProgram>,
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
    /// Operator-controlled suppression flag. `true` (TOML default) ⇒
    /// the entry is effective; `false` ⇒ structurally equivalent to
    /// "absent from the config." Disabled entries flow through parsing
    /// and validation unchanged (so typos surface at config load, not
    /// silently at re-enable time) but are filtered out of every
    /// runtime view by [`Config::active_watches`]. The engine never
    /// learns about disabled entries — every transition (initial
    /// attach, hot-reload diff, drain-window derivation) consumes the
    /// filtered iterator.
    ///
    /// Included in [`PartialEq`] so two specs differing only on this
    /// field compare unequal. The diff layer's filter strips disabled
    /// entries *before* the equality check, so this matters only for
    /// future consumers that compare unfiltered specs.
    pub enabled: bool,
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
            Arc::clone(&self.program),
            self.scope,
            self.events,
            self.log_output,
        )
    }
}

/// Validated dynamic-watch entry — the config-layer mirror of the
/// engine's [`specter_core::Promoter`].
///
/// Materialised by [`validate_dynamic_watch`] when the dispatcher
/// observes a glob discriminator character (`*?[{`) in `path`. Each
/// `PromoterSpec` translates to one [`PromoterAttachRequest`] via
/// [`Self::to_attach_request`]; the engine assigns a `PromoterId` at
/// attach time.
///
/// Field shape mirrors [`SubSpec`]: per-attachment knobs (settle,
/// max_settle, scope, events, log_output, scan) are independent of the
/// pattern itself, so two Promoters that share a pattern but differ on
/// e.g. `settle` are correctly distinct. Equality is structural across
/// every field; the diff path uses it to drive the wholesale-replace
/// (reap + reattach) on modify.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct PromoterSpec {
    pub name: CompactString,
    pub pattern: PatternSpec,
    /// Lowered bytecode IR. See [`SubSpec::program`]; the Promoter
    /// holds the same Arc and clones it into every synthesised dynamic
    /// Sub via [`Self::to_attach_request`].
    pub program: Arc<ActionProgram>,
    pub scope: EffectScope,
    pub settle: Duration,
    pub max_settle: Duration,
    pub scan: ScanConfig,
    /// Threaded into each synthesized dynamic Sub. Same scope-conditional
    /// default as [`SubSpec::events`].
    pub events: ClassSet,
    /// Threaded into each synthesized dynamic Sub. See
    /// [`SubSpec::log_output`].
    pub log_output: bool,
    /// Operator-controlled suppression flag — see [`SubSpec::enabled`].
    /// Disabling a Promoter is structurally equivalent to removing it:
    /// no descent runs, no dynamic Subs are spawned, no
    /// `watch_demand` is contributed. Re-enabling triggers a fresh
    /// `attach_promoter_inner` (no zombie revival path exists for
    /// Promoters in v1; dynamic Subs spawned across a disable/enable
    /// cycle get freshly-minted `SubId`s).
    pub enabled: bool,
}

impl PromoterSpec {
    #[must_use]
    pub fn to_attach_request(&self) -> PromoterAttachRequest {
        PromoterAttachRequest {
            name: self.name.to_string(),
            pattern_spec: self.pattern.clone(),
            config: self.scan.clone(),
            max_settle: self.max_settle,
            settle: self.settle,
            program: Arc::clone(&self.program),
            scope: self.scope,
            events: self.events,
            log_output: self.log_output,
        }
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
    /// Iterator over enabled static watches in source order.
    ///
    /// Sole authority for "what's effective right now": every runtime
    /// consumer ([`crate::diff`], the bin's initial-attach pass,
    /// [`crate::Config`] drain-window derivation, the startup /
    /// reload load logs) goes through this helper. Iterating the raw
    /// [`Self::watches`] field directly bypasses the per-entry
    /// `enabled` filter and is almost always wrong outside config
    /// introspection / round-trip serialization.
    ///
    /// Discipline: `enabled = false ⇔ entry absent from the effective
    /// config`. Every Add/Remove transition the engine handles flows
    /// from a flip in this iterator's output, so disabled entries
    /// never reach the engine — they remain in `self.watches` for
    /// introspection but are otherwise inert.
    pub fn active_watches(&self) -> impl Iterator<Item = &SubSpec> + '_ {
        self.watches.iter().filter(|s| s.enabled)
    }

    /// Iterator over enabled dynamic watches in source order — the
    /// Promoter analogue of [`Self::active_watches`]. Same discipline.
    pub fn active_promoters(&self) -> impl Iterator<Item = &PromoterSpec> + '_ {
        self.promoters.iter().filter(|p| p.enabled)
    }

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
        log_config_loaded(&cfg, path);
        Ok(cfg)
    }

    /// Atomic content + filesystem-identity capture: opens `path`,
    /// captures [`FileMeta`] from the bound inode, then reads the
    /// content from the same handle. The inode is pinned by `f`, so a
    /// concurrent `rename(2)` over `path` (atomic-save) cannot rotate
    /// the meta out from under the bytes — the next `FileMeta::from_path`
    /// observes the path-level rotation as a meta delta.
    ///
    /// `f.metadata()` is called **before** the `read_to_string` so that
    /// any in-place mutation of the still-bound inode during the read
    /// surfaces on the next path-level lstat as `stored != current`.
    /// Reversing the order would absorb the mutation into the stored
    /// meta and silently pin the loader to stale content.
    pub fn from_path_with_meta(path: &Path) -> Result<(Self, FileMeta), ConfigError> {
        use std::io::Read as _;
        let mut f = std::fs::File::open(path).map_err(|e| ConfigError::Io {
            path: path.to_owned(),
            source: e,
        })?;
        let meta = FileMeta::from_metadata(&f.metadata().map_err(|e| ConfigError::Io {
            path: path.to_owned(),
            source: e,
        })?);
        let mut s = String::new();
        f.read_to_string(&mut s).map_err(|e| ConfigError::Io {
            path: path.to_owned(),
            source: e,
        })?;
        let cfg = Self::from_str_inner(&s, Some(path))?;
        log_config_loaded(&cfg, path);
        Ok((cfg, meta))
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

/// Emit the `"config loaded"` info-level event with shape shared by
/// [`Config::from_path`] and [`Config::from_path_with_meta`].
///
/// `disabled_watches` / `disabled_promoters` carry the names of entries
/// the operator suppressed via `enabled = false`. The macro renders
/// empty `Vec`s as `[]` — accept the noise for the all-enabled case
/// rather than branching the format string. Operators triaging "why
/// isn't watch X firing?" can grep the log for the watch's name in
/// the disabled lists rather than re-reading the TOML.
fn log_config_loaded(cfg: &Config, path: &Path) {
    let disabled_watches: Vec<&str> = cfg
        .watches
        .iter()
        .filter(|s| !s.enabled)
        .map(|s| s.name.as_str())
        .collect();
    let disabled_promoters: Vec<&str> = cfg
        .promoters
        .iter()
        .filter(|p| !p.enabled)
        .map(|p| p.name.as_str())
        .collect();
    tracing::info!(
        path = %path.display(),
        watches = cfg.watches.len(),
        promoters = cfg.promoters.len(),
        ?disabled_watches,
        ?disabled_promoters,
        "config loaded",
    );
}

fn validate(raw: &RawConfig, path: Option<&Path>) -> Result<Config, ConfigError> {
    let mut errors: Vec<ValidationIssue> = Vec::new();

    let log = validate_log(raw.log.as_ref(), &mut errors);

    let mut watches: Vec<SubSpec> = Vec::new();
    let mut promoters: Vec<PromoterSpec> = Vec::new();
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

        // Auto-detect: any of `*?[{` in `path` routes the entry to the
        // dynamic validator. The dispatcher is the contract — neither
        // validator second-guesses it on the well-trodden path.
        if PatternSpec::is_dynamic(&raw_w.path) {
            match validate_dynamic_watch(i, raw_w) {
                Ok(spec) => promoters.push(spec),
                Err(mut errs) => errors.append(&mut errs),
            }
        } else {
            match validate_static_watch(i, raw_w) {
                Ok(spec) => watches.push(spec),
                Err(mut errs) => errors.append(&mut errs),
            }
        }
    }

    if errors.is_empty() {
        Ok(Config {
            log,
            watches,
            promoters,
        })
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

/// Validate the `name` field. Two failures are mutually exclusive:
/// empty (rejected as [`IssueKind::Empty`]) and `@`-bearing
/// (rejected as [`IssueKind::InvalidName`] — `@` is reserved for the
/// engine's synthesized `<promoter_name>@<resolved_path>` shape).
///
/// Both static and dynamic validators call this so the rule lives in
/// one place. Duplicate-name detection is handled at the outer
/// dispatch loop (it spans both kinds and so cannot be a per-watch
/// helper concern).
fn validate_name(idx: usize, raw_name: &str, errors: &mut Vec<ValidationIssue>) {
    if raw_name.is_empty() {
        errors.push(ValidationIssue::new(
            Some(idx),
            "name",
            IssueKind::Empty,
            "name must not be empty".to_owned(),
        ));
        return;
    }
    if raw_name.contains('@') {
        errors.push(ValidationIssue::new(
            Some(idx),
            "name",
            IssueKind::InvalidName,
            format!(
                "name `{raw_name}` must not contain `@` (reserved for \
                 synthesized dynamic Sub names of the form \
                 `<promoter_name>@<resolved_path>`)",
            ),
        ));
    }
}

/// Validate the `actions` array. Returns `Some(Arc<ActionProgram>)`
/// when every action lexes cleanly.
///
/// Errors accumulate into `errors`; one issue per offending action /
/// argv slot. The function returns `None` when *any* part of the
/// program failed validation — partial programs are not handed back,
/// since a half-built program in the engine would be observably worse
/// than none at all.
///
/// Returns `Ok(Arc<ActionProgram>)` on success, `Err(Vec<ValidationIssue>)`
/// on any failure — the [`Result`] shape ties the validated program to
/// the absence of errors at the type level, so callers cannot reach for
/// the `Arc` without first resolving the failure case. This rules out
/// the historical "validator returned `None` without pushing an issue
/// ⇒ caller `.expect()` panics" foot-gun.
///
/// The returned Arc is the same allocation the engine's `Sub.program`
/// and every emitted `Effect.program` references — one shared
/// bytecode IR per validated Sub.
fn validate_actions(
    idx: usize,
    raw_actions: &[RawAction],
) -> Result<Arc<ActionProgram>, Vec<ValidationIssue>> {
    if raw_actions.is_empty() {
        return Err(vec![ValidationIssue::new(
            Some(idx),
            "actions",
            IssueKind::EmptyActions,
            "actions must have at least one entry".to_owned(),
        )]);
    }

    let mut errors: Vec<ValidationIssue> = Vec::new();
    let Some(tree) = validate_action_list(idx, "actions", raw_actions, &mut errors) else {
        // `validate_action_list` populated `errors` for every failed
        // element; the empty case is the caller-side bug we're
        // type-protecting against — assert to surface a regression
        // loudly in debug, then return whatever was collected.
        debug_assert!(
            !errors.is_empty(),
            "validate_action_list returned None without pushing an issue",
        );
        return Err(errors);
    };
    match lower_to_program(&tree) {
        Ok(program) => Ok(program),
        Err(e) => {
            errors.push(ValidationIssue::from_program_error(
                &e,
                Some(idx),
                "actions",
            ));
            Err(errors)
        }
    }
}

/// Recursive validation of a `[RawAction]` slice. `path` is the
/// breadcrumb-style label of the slice within the watch — `"actions"`
/// at the top, `"actions[0].then"` inside a then-branch, etc. The
/// returned `Vec<Action>` is `Some` iff every element validated; on
/// any failure the per-element errors are recorded and the function
/// returns `None`.
///
/// Empty input is the *caller's* responsibility to reject (only the
/// top-level `actions = []` carries [`IssueKind::EmptyActions`];
/// nested empty arrays are rejected via [`IssueKind::EmptyConditional`]
/// against the enclosing conditional). This function is silent on
/// emptiness — it returns `Some(Vec::new())` in that case so the
/// caller can fold the empty branch into the AST as `None` (no else)
/// or apply the conditional-level check.
fn validate_action_list(
    watch_idx: usize,
    path: &str,
    raw_actions: &[RawAction],
    errors: &mut Vec<ValidationIssue>,
) -> Option<Vec<Action>> {
    let mut tree: Vec<Action> = Vec::with_capacity(raw_actions.len());
    let mut any_failed = false;
    for (j, raw) in raw_actions.iter().enumerate() {
        let child_path = format!("{path}[{j}]");
        match validate_one_action(watch_idx, &child_path, raw, errors) {
            Some(action) => tree.push(action),
            None => any_failed = true,
        }
    }
    if any_failed { None } else { Some(tree) }
}

/// Validate a single action entry. The "exactly one variant set" rule
/// is the single source of truth across `exec`, the conditional
/// triple (`when` + `then` + optional `else`), and (future) `pipe` —
/// it stays the same shape as new variants land.
///
/// `path` is the action's breadcrumb-style label
/// (`"actions[0]"`, `"actions[0].then[1]"`, etc). Error messages
/// quote it so operators can locate the offending entry without
/// re-counting nested arrays.
fn validate_one_action(
    watch_idx: usize,
    path: &str,
    raw: &RawAction,
    errors: &mut Vec<ValidationIssue>,
) -> Option<Action> {
    let is_exec = raw.exec.is_some();
    let is_pipe = raw.pipe.is_some();
    let is_conditional = raw.when.is_some() || raw.then.is_some() || raw.otherwise.is_some();
    let variants_set = usize::from(is_exec) + usize::from(is_pipe) + usize::from(is_conditional);
    if variants_set == 0 {
        errors.push(ValidationIssue::new(
            Some(watch_idx),
            "actions",
            IssueKind::ActionMissingVariant,
            format!("{path}: must specify a variant (`exec`, `pipe`, or `when`/`then`)"),
        ));
        return None;
    }
    if variants_set > 1 {
        errors.push(ValidationIssue::new(
            Some(watch_idx),
            "actions",
            IssueKind::ActionAmbiguousVariant,
            format!(
                "{path}: must specify exactly one variant \
                 (`exec`, `pipe`, and `when`/`then` are mutually exclusive)",
            ),
        ));
        return None;
    }
    // `timeout` at the action level applies only to `exec`. Predicates
    // (`when`) carry their own per-step `timeout` inside the nested
    // `RawExec`; `pipe` stages each carry their own on the nested
    // `RawExec` of each stage.
    if raw.timeout.is_some() && !is_exec {
        errors.push(ValidationIssue::new(
            Some(watch_idx),
            "actions",
            IssueKind::TimeoutNotApplicable,
            format!(
                "{path}: top-level `timeout` requires `exec` \
                 (predicates set `when.timeout`; pipe stages each set their own `timeout`)",
            ),
        ));
        return None;
    }
    if let Some(argv) = raw.exec.as_deref() {
        return validate_exec_argv(watch_idx, path, "exec", argv, raw.timeout, errors)
            .map(Action::Exec);
    }
    if let Some(stages) = raw.pipe.as_deref() {
        return validate_pipe(watch_idx, path, stages, errors);
    }
    if is_conditional {
        return validate_conditional(watch_idx, path, raw, errors);
    }
    unreachable!("variants_set ∈ {{1}} and exec/pipe/conditional are exhaustive in v1")
}

/// Validate the `pipe = [{ exec = [...], timeout = "..." }, ...]`
/// variant of a [`RawAction`].
///
/// Structural rules:
/// - `pipe = []` ⇒ [`IssueKind::EmptyPipe`] (no stages to wire).
/// - `pipe = [solo]` ⇒ [`IssueKind::SingleStagePipe`] (degenerate;
///   use top-level `exec` directly).
/// - Each stage's argv goes through [`validate_exec_argv`] under the
///   path label `"<path>.pipe[i].exec"` so per-stage errors are
///   unambiguous; stage-local errors don't short-circuit later stages.
///
/// Stages are stored as `Arc<[ExecAction]>` so lowering can
/// `Arc::clone` into [`specter_core::program::SpawnBody::Pipe`] without
/// re-allocating.
fn validate_pipe(
    watch_idx: usize,
    path: &str,
    stages: &[RawExec],
    errors: &mut Vec<ValidationIssue>,
) -> Option<Action> {
    if stages.is_empty() {
        errors.push(ValidationIssue::new(
            Some(watch_idx),
            "actions",
            IssueKind::EmptyPipe,
            format!("{path}.pipe must have at least two stages (got 0)"),
        ));
        return None;
    }
    if stages.len() < 2 {
        errors.push(ValidationIssue::new(
            Some(watch_idx),
            "actions",
            IssueKind::SingleStagePipe,
            format!(
                "{path}.pipe must have at least two stages \
                 (single stages should use top-level `exec` directly)",
            ),
        ));
        return None;
    }
    let mut validated: Vec<ExecAction> = Vec::with_capacity(stages.len());
    let mut any_failed = false;
    for (idx, stage) in stages.iter().enumerate() {
        let stage_path = format!("{path}.pipe[{idx}]");
        match validate_raw_exec(watch_idx, &stage_path, stage, errors) {
            Some(exec) => validated.push(exec),
            None => any_failed = true,
        }
    }
    if any_failed {
        return None;
    }
    Some(Action::Pipe {
        stages: Arc::from(validated),
    })
}

/// Validate the `when` / `then` / `else` triple on a [`RawAction`].
///
/// Structural rules:
/// - Both `when` and `then` must be present (one without the other is
///   [`IssueKind::ConditionalIncomplete`]). `else` is optional.
/// - `then = []` AND (`else` absent OR `else = []`) is
///   [`IssueKind::EmptyConditional`] — the predicate would fire with
///   no observable effect.
/// - `then = []` with non-empty `else` is permitted (operationally
///   equivalent to a negated predicate).
///
/// Per-branch errors (argv shape, nested action variants, etc.) are
/// collected alongside the structural errors so a single config-load
/// surfaces every issue.
fn validate_conditional(
    watch_idx: usize,
    path: &str,
    raw: &RawAction,
    errors: &mut Vec<ValidationIssue>,
) -> Option<Action> {
    let when_raw = raw.when.as_ref();
    let then_raw = raw.then.as_deref();
    let otherwise_raw = raw.otherwise.as_deref();

    if when_raw.is_none() || then_raw.is_none() {
        errors.push(ValidationIssue::new(
            Some(watch_idx),
            "actions",
            IssueKind::ConditionalIncomplete,
            format!(
                "{path}: conditional requires both `when` and `then` \
                 (got when={}, then={}{})",
                when_raw.is_some(),
                then_raw.is_some(),
                if otherwise_raw.is_some() {
                    ", else=true"
                } else {
                    ""
                },
            ),
        ));
        return None;
    }
    let when_raw = when_raw.expect("checked Some directly above");
    let then_raw = then_raw.expect("checked Some directly above");

    // Empty-conditional check fires *before* per-branch validation so
    // operators see one structural error rather than a flood of
    // per-empty-branch issues (which is exactly nothing in that case).
    let else_empty = otherwise_raw.is_none_or(<[RawAction]>::is_empty);
    if then_raw.is_empty() && else_empty {
        errors.push(ValidationIssue::new(
            Some(watch_idx),
            "actions",
            IssueKind::EmptyConditional,
            format!(
                "{path}: conditional has empty `then` and no `else` body \
                 (predicate would have no observable effect)",
            ),
        ));
        return None;
    }

    let when_path = format!("{path}.when");
    let when = validate_raw_exec(watch_idx, &when_path, when_raw, errors);

    let then_path = format!("{path}.then");
    let then = validate_action_list(watch_idx, &then_path, then_raw, errors);

    // `Some(empty)` normalises to `None` so the AST shape matches the
    // lowering precondition (`Some(_)` ⇒ non-empty body). Lowering also
    // handles empty defensively, but normalising here keeps the AST
    // the canonical form.
    let otherwise = match otherwise_raw {
        Some(body) if !body.is_empty() => {
            let path = format!("{path}.else");
            validate_action_list(watch_idx, &path, body, errors).map(Some)
        }
        Some(_) | None => Some(None),
    };

    Some(Action::Conditional {
        when: when?,
        then: then?.into_boxed_slice(),
        otherwise: otherwise?.map(Vec::into_boxed_slice),
    })
}

/// Validate the nested-exec table used inside a conditional predicate
/// (`when = { exec = [...], timeout = "..." }`). Delegates to
/// [`validate_exec_argv`] under the path label `"<path>.exec"` so
/// argv-slot errors quote the surrounding action's location.
fn validate_raw_exec(
    watch_idx: usize,
    path: &str,
    raw: &RawExec,
    errors: &mut Vec<ValidationIssue>,
) -> Option<ExecAction> {
    validate_exec_argv(watch_idx, path, "exec", &raw.exec, raw.timeout, errors)
}

/// Validate one `exec = [...]` argv plus its optional `timeout`. Each
/// empty slot or parse failure yields one [`ValidationIssue`]; the
/// function returns `None` on any failure so the partial argv can't
/// reach the engine.
///
/// `timeout` is the operator-supplied per-step deadline (humantime
/// at the TOML layer; [`Duration`] here). When `Some`, the validated
/// duration is also checked for `> 0` — a zero-duration timeout is
/// rejected as a configuration error: the SIGTERM would fire before
/// the child has any chance to make progress, which is almost
/// certainly a typo (`"0s"`, `"0ms"`). Operators wanting "no timeout"
/// omit the field entirely.
///
/// `path` is the surrounding action's breadcrumb (e.g.,
/// `"actions[0]"` or `"actions[0].then[1]"`); `argv_field` is the
/// field name of the argv slot within that action — `"exec"` for both
/// top-level exec and predicate-exec; future `pipe` stages will pass
/// `"pipe[i].exec"`. The error detail joins them with `.` so the
/// path is unambiguous.
fn validate_exec_argv(
    watch_idx: usize,
    path: &str,
    argv_field: &str,
    raw_argv: &[String],
    timeout: Option<Duration>,
    errors: &mut Vec<ValidationIssue>,
) -> Option<ExecAction> {
    if raw_argv.is_empty() {
        errors.push(ValidationIssue::new(
            Some(watch_idx),
            "actions",
            IssueKind::EmptyArgv,
            format!("{path}.{argv_field} must have at least one slot"),
        ));
        return None;
    }
    let mut argv: Vec<ArgTemplate> = Vec::with_capacity(raw_argv.len());
    let mut any_failed = false;
    for (k, slot) in raw_argv.iter().enumerate() {
        if slot.is_empty() {
            errors.push(ValidationIssue::new(
                Some(watch_idx),
                "actions",
                IssueKind::EmptyArgv,
                format!("{path}.{argv_field}[{k}] is empty"),
            ));
            any_failed = true;
            continue;
        }
        match template::parse_arg(slot) {
            Ok(arg) => argv.push(arg),
            Err(e) => {
                errors.push(ValidationIssue::new(
                    Some(watch_idx),
                    "actions",
                    IssueKind::UnknownPlaceholder,
                    format!("{path}.{argv_field}[{k}]: {e}"),
                ));
                any_failed = true;
            }
        }
    }
    if let Some(d) = timeout
        && d.is_zero()
    {
        errors.push(ValidationIssue::new(
            Some(watch_idx),
            "actions",
            IssueKind::TimeoutZero,
            format!(
                "{path}.timeout must be > 0 \
                 (omit the field for `no deadline`)",
            ),
        ));
        any_failed = true;
    }
    if any_failed {
        None
    } else {
        let exec = ExecAction::new(argv);
        Some(match timeout {
            Some(d) => exec.with_timeout(d),
            None => exec,
        })
    }
}

/// Validate `settle` / `max_settle`. Returns `(settle, max_settle)`
/// always — invalid values flow through with their raw value so the
/// caller can carry on collecting field-level errors; the issues are
/// surfaced to the operator on the error path.
///
/// - `settle` defaults to [`DEFAULT_SETTLE`] when omitted; rejected
///   with [`IssueKind::SettleTooSmall`] when zero.
/// - `max_settle` defaults to [`DEFAULT_MAX_SETTLE`] (a flat 1h) when
///   omitted; rejected with [`IssueKind::MaxSettleTooSmall`] when below
///   `4 × settle`. There is no upper bound — Instant arithmetic at the
///   engine layer handles values up to many years; humantime input
///   format makes magnitude typos obvious at the source.
fn validate_settle(
    idx: usize,
    raw_settle: Option<Duration>,
    raw_max_settle: Option<Duration>,
    errors: &mut Vec<ValidationIssue>,
) -> (Duration, Duration) {
    let settle = raw_settle.unwrap_or(DEFAULT_SETTLE);
    if settle.is_zero() {
        errors.push(ValidationIssue::new(
            Some(idx),
            "settle",
            IssueKind::SettleTooSmall,
            "settle must be > 0".to_owned(),
        ));
    }
    let max_settle = match raw_max_settle {
        Some(v) => {
            let floor = settle.saturating_mul(MAX_SETTLE_FLOOR_FACTOR);
            if v < floor {
                errors.push(ValidationIssue::new(
                    Some(idx),
                    "max_settle",
                    IssueKind::MaxSettleTooSmall,
                    format!(
                        "max_settle ({}) must be ≥ 4 × settle ({})",
                        humantime::format_duration(v),
                        humantime::format_duration(floor),
                    ),
                ));
            }
            v
        }
        None => DEFAULT_MAX_SETTLE,
    };
    (settle, max_settle)
}

/// Validate `scope`. Returns `Ok(EffectScope)` on success; `Err` with
/// the parse failure issue. Scope failures are single-issue by
/// construction (one input string, one valid set), so the error path
/// is a single [`ValidationIssue`] rather than a `Vec`.
///
/// The caller forwards a copied [`EffectScope`] to the events parser
/// even when validation fails — the default ([`EffectScope::default`])
/// is used solely to keep `parse_events_field` from cascading a
/// phantom error against an unresolved scope.
fn validate_scope(idx: usize, raw_scope: Option<&str>) -> Result<EffectScope, ValidationIssue> {
    match raw_scope.unwrap_or("subtree-root") {
        "subtree-root" => Ok(EffectScope::SubtreeRoot),
        "per-stable-file" => Ok(EffectScope::PerStableFile),
        other => Err(ValidationIssue::new(
            Some(idx),
            "scope",
            IssueKind::InvalidEnum,
            format!("unknown scope `{other}` (expected `subtree-root` or `per-stable-file`)"),
        )),
    }
}

/// Validate a static watch's `path`: absolute, lenient-canonicalisable.
/// Returns `Ok(PathBuf)` on success; `Err(ValidationIssue)` on either
/// "not absolute" or "canonicalisation failed" — single-issue by
/// construction.
fn validate_static_path(idx: usize, raw_path: &str) -> Result<PathBuf, ValidationIssue> {
    if !Path::new(raw_path).is_absolute() {
        return Err(ValidationIssue::new(
            Some(idx),
            "path",
            IssueKind::NonAbsolute,
            format!("path `{raw_path}` must be absolute"),
        ));
    }
    canonicalize_lenient(Path::new(raw_path)).map_err(|e| {
        ValidationIssue::new(
            Some(idx),
            "path",
            IssueKind::NotCanonical,
            format!("`{raw_path}`: {e}"),
        )
    })
}

/// Validate a dynamic watch's `path` as a [`PatternSpec`]. The parser
/// itself enforces structural invariants (absolute, no `**`, no
/// `.`/`..`, no empty segments, no Windows prefix); a parse failure
/// surfaces as [`IssueKind::InvalidPattern`].
fn validate_dynamic_pattern(idx: usize, raw_path: &str) -> Result<PatternSpec, ValidationIssue> {
    PatternSpec::parse(raw_path).map_err(|e| {
        ValidationIssue::new(
            Some(idx),
            "path",
            IssueKind::InvalidPattern,
            format!("`{raw_path}`: {e}"),
        )
    })
}

/// Validate the `[[watch]]` block's scan-config knobs (`recursive`,
/// `hidden`, `max_depth`, `pattern`, `exclude`) — orthogonal to the
/// path-vs-pattern dispatch. Always returns a [`ScanConfig`]; invalid
/// values are recorded in `errors` and the offending field falls back
/// to its default so downstream consumers never see a half-built
/// builder.
fn validate_scan(idx: usize, raw: &RawWatch, errors: &mut Vec<ValidationIssue>) -> ScanConfig {
    if raw.max_depth == Some(0) {
        errors.push(ValidationIssue::new(
            Some(idx),
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
                errors.push(ValidationIssue::new(
                    Some(idx),
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
                    errors.push(ValidationIssue::new(
                        Some(idx),
                        "exclude",
                        IssueKind::InvalidGlob,
                        format!("`{ex}`: {message}"),
                    ));
                }
            }
        }
        sb = sb.excludes(compiled);
    }

    sb.build()
}

/// Validator for `[[watch]]` blocks whose `path` carries no glob
/// discriminator characters (`*?[{`) — pure-literal anchors. Caller
/// (the dispatcher in [`validate`]) gates on
/// [`PatternSpec::is_dynamic`] before invoking this; the inner
/// `is_dynamic` check is defense-in-depth for direct test callers
/// that bypass the dispatcher.
///
/// Validation runs every sub-validator unconditionally and collects
/// every issue before returning. The success path destructures the
/// per-field [`Result`]s via a tuple match — there is no `.expect`
/// on a validated field, so a future regression that produces a
/// `None`/`Err` without a corresponding issue surfaces as an `Err`,
/// not a panic.
fn validate_static_watch(idx: usize, raw: &RawWatch) -> Result<SubSpec, Vec<ValidationIssue>> {
    let mut errors: Vec<ValidationIssue> = Vec::new();
    let issue = |field: &'static str, kind: IssueKind, detail: String| {
        ValidationIssue::new(Some(idx), field, kind, detail)
    };

    // Defense-in-depth: the dispatcher decides static vs. dynamic on
    // `is_dynamic(path)`. Bypassing the dispatcher (test surface) and
    // landing here with a glob-bearing path is a contract violation;
    // surface it as a dedicated kind so the breach is observable in
    // error output rather than masquerading as `NonAbsolute` /
    // `NotCanonical` further down.
    if PatternSpec::is_dynamic(&raw.path) {
        errors.push(issue(
            "path",
            IssueKind::PathContainsGlobChars,
            format!(
                "path `{}` contains a glob discriminator character \
                 (`*?[{{`); this entry should have been routed to the \
                 dynamic validator",
                raw.path,
            ),
        ));
    }

    validate_name(idx, &raw.name, &mut errors);

    let path_r = validate_static_path(idx, &raw.path);
    let program_r = validate_actions(idx, &raw.actions);
    let scope_r = validate_scope(idx, raw.scope.as_deref());
    let (settle, max_settle) = validate_settle(idx, raw.settle, raw.max_settle, &mut errors);
    // Parse `events` after scope so the default resolver can read
    // scope. If scope itself failed validation, fall back to the
    // default scope for the events default — this avoids a cascade
    // of phantom errors; the scope error is already pending in
    // `scope_r`.
    let events = parse_events_field(
        raw.events.as_deref(),
        scope_r.as_ref().copied().unwrap_or_default(),
        idx,
        &mut errors,
    );
    let scan = validate_scan(idx, raw, &mut errors);

    // Applicative collection: the Ok arm destructures all three
    // per-field Results into their values directly — no .expect()
    // and no panic site. The guard catches the case where the Result-
    // returning validators all succeeded but a side-effect validator
    // (validate_name, validate_settle, validate_scan, parse_events_field,
    // the glob-char guard above) pushed an issue.
    match (path_r, program_r, scope_r) {
        (Ok(path), Ok(program), Ok(scope)) if errors.is_empty() => Ok(SubSpec {
            name: CompactString::new(&raw.name),
            path,
            program,
            scope,
            settle,
            max_settle,
            scan,
            events,
            log_output: raw.log_output.unwrap_or(false),
            enabled: raw.enabled.unwrap_or(true),
        }),
        (path_r, program_r, scope_r) => {
            if let Err(e) = path_r {
                errors.push(e);
            }
            if let Err(es) = program_r {
                errors.extend(es);
            }
            if let Err(e) = scope_r {
                errors.push(e);
            }
            Err(errors)
        }
    }
}

/// Validator for `[[watch]]` blocks whose `path` carries at least one
/// glob discriminator character (`*?[{`). Caller gates on
/// [`PatternSpec::is_dynamic`]; the parser itself enforces the
/// pattern's structural invariants (absolute, no `**`, no `.`/`..`,
/// no empty segments, no Windows prefix).
///
/// Validation shape mirrors [`validate_static_watch`]: every sub-
/// validator runs, results are collected via tuple match, and the
/// success path destructures without `.expect()`.
fn validate_dynamic_watch(
    idx: usize,
    raw: &RawWatch,
) -> Result<PromoterSpec, Vec<ValidationIssue>> {
    let mut errors: Vec<ValidationIssue> = Vec::new();

    validate_name(idx, &raw.name, &mut errors);

    let pattern_r = validate_dynamic_pattern(idx, &raw.path);
    let program_r = validate_actions(idx, &raw.actions);
    let scope_r = validate_scope(idx, raw.scope.as_deref());
    let (settle, max_settle) = validate_settle(idx, raw.settle, raw.max_settle, &mut errors);
    let events = parse_events_field(
        raw.events.as_deref(),
        scope_r.as_ref().copied().unwrap_or_default(),
        idx,
        &mut errors,
    );
    let scan = validate_scan(idx, raw, &mut errors);

    match (pattern_r, program_r, scope_r) {
        (Ok(pattern), Ok(program), Ok(scope)) if errors.is_empty() => Ok(PromoterSpec {
            name: CompactString::new(&raw.name),
            pattern,
            program,
            scope,
            settle,
            max_settle,
            scan,
            events,
            log_output: raw.log_output.unwrap_or(false),
            enabled: raw.enabled.unwrap_or(true),
        }),
        (pattern_r, program_r, scope_r) => {
            if let Err(e) = pattern_r {
                errors.push(e);
            }
            if let Err(es) = program_r {
                errors.extend(es);
            }
            if let Err(e) = scope_r {
                errors.push(e);
            }
            Err(errors)
        }
    }
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
    use specter_core::program::SpawnBody;
    use specter_core::{ArgPart, ClassSet, EffectScope, Placeholder};
    use std::time::Duration;

    const ROOT: &str = "/";

    fn minimal_toml(extra: &str) -> String {
        format!(
            "[[watch]]\nname = \"build\"\npath = \"{ROOT}\"\nactions = [{{ exec = [\"echo\"] }}]\n{extra}"
        )
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
        assert_eq!(w.max_settle, Duration::from_hours(1));
        assert!(w.scan.recursive);
        assert!(!w.scan.hidden);
        assert!(w.scan.exclude.is_empty());
        assert!(w.scan.pattern.is_none());
        assert_eq!(w.scan.max_depth, None);
        let SpawnBody::Exec(exec) = &w.program.ops()[0].body else {
            panic!("expected SpawnBody::Exec");
        };
        assert_eq!(exec.argv.len(), 1);
        assert!(!w.log_output, "log_output defaults to false");
        assert!(w.enabled, "enabled defaults to true (field omitted)");
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
    fn enabled_false_round_trips() {
        // Disabled entries still land in `Config.watches` — the filter
        // is applied at the runtime view (`active_watches`), not at
        // parse time.
        let cfg = Config::from_str(&minimal_toml("enabled = false\n")).unwrap();
        assert!(!cfg.watches[0].enabled);
        assert_eq!(cfg.watches.len(), 1, "disabled entry kept in raw Vec");
    }

    #[test]
    fn enabled_false_round_trips_for_dynamic_watch() {
        // Mirror the static-side round-trip on the dynamic dispatch
        // path (path containing `*?[{` routes to the Promoter
        // validator).
        let toml = "[[watch]]\nname = \"logs\"\npath = \"/var/log/*\"\n\
                    actions = [{ exec = [\"echo\"] }]\nenabled = false\n";
        let cfg = Config::from_str(toml).unwrap();
        assert_eq!(cfg.promoters.len(), 1);
        assert!(!cfg.promoters[0].enabled);
    }

    #[test]
    fn active_watches_and_promoters_filter_disabled_preserving_order() {
        // Mixed config exercises both helpers: disabled `b` and `d`
        // are stripped from the static side, disabled `e` from the
        // dynamic side. Source order is preserved among the entries
        // each helper yields.
        let toml = format!(
            "[[watch]]\nname = \"a\"\npath = \"{ROOT}\"\nactions = [{{ exec = [\"echo\"] }}]\n\
             [[watch]]\nname = \"b\"\npath = \"{ROOT}\"\nactions = [{{ exec = [\"echo\"] }}]\nenabled = false\n\
             [[watch]]\nname = \"c\"\npath = \"{ROOT}\"\nactions = [{{ exec = [\"echo\"] }}]\n\
             [[watch]]\nname = \"d\"\npath = \"{ROOT}\"\nactions = [{{ exec = [\"echo\"] }}]\nenabled = false\n\
             [[watch]]\nname = \"e\"\npath = \"/srv/*\"\nactions = [{{ exec = [\"echo\"] }}]\nenabled = false\n\
             [[watch]]\nname = \"f\"\npath = \"/srv/*\"\nactions = [{{ exec = [\"echo\"] }}]\n",
        );
        let cfg = Config::from_str(&toml).unwrap();
        let watches: Vec<&str> = cfg.active_watches().map(|s| s.name.as_str()).collect();
        let promoters: Vec<&str> = cfg.active_promoters().map(|p| p.name.as_str()).collect();
        assert_eq!(watches, vec!["a", "c"]);
        assert_eq!(promoters, vec!["f"]);
    }

    #[test]
    fn empty_name_rejected() {
        let toml = format!(
            "[[watch]]\nname = \"\"\npath = \"{ROOT}\"\nactions = [{{ exec = [\"echo\"] }}]"
        );
        assert_only_kind(&toml, IssueKind::Empty);
    }

    #[test]
    fn relative_path_rejected() {
        let toml = "[[watch]]\nname = \"a\"\npath = \"src\"\nactions = [{ exec = [\"echo\"] }]";
        assert_only_kind(toml, IssueKind::NonAbsolute);
    }

    #[test]
    fn empty_command_array_rejected() {
        let toml =
            format!("[[watch]]\nname = \"a\"\npath = \"{ROOT}\"\nactions = [{{ exec = [] }}]");
        assert_only_kind(&toml, IssueKind::EmptyArgv);
    }

    #[test]
    fn empty_argv_slot_rejected() {
        let toml =
            format!("[[watch]]\nname = \"a\"\npath = \"{ROOT}\"\nactions = [{{ exec = [\"\"] }}]");
        assert_only_kind(&toml, IssueKind::EmptyArgv);
    }

    #[test]
    fn lowercase_typo_placeholder_still_rejected_as_unknown() {
        // *Inside the `${specter.…}` namespace*, lowercase non-catalog
        // names remain typo errors; the catalog is exclusively lowercase,
        // so a lowercase miss inside the namespace is almost always a
        // typo. Bare `$paht` (outside the namespace) is literal
        // pass-through under the new grammar — exercised by
        // `template::tests::bare_dollar_name_is_literal`.
        let toml = format!(
            "[[watch]]\nname = \"a\"\npath = \"{ROOT}\"\nactions = [{{ exec = [\"fmt\", \"${{specter.paht}}\"] }}]"
        );
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
            let toml = format!(
                "[[watch]]\nname = \"a\"\npath = \"{ROOT}\"\nactions = [{{ exec = {cmd} }}]"
            );
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
        let toml = minimal_toml("settle = \"0ms\"\n");
        assert_only_kind(&toml, IssueKind::SettleTooSmall);
    }

    #[test]
    fn max_settle_below_floor_rejected() {
        let toml = minimal_toml("settle = \"100ms\"\nmax_settle = \"200ms\"\n");
        assert_only_kind(&toml, IssueKind::MaxSettleTooSmall);
    }

    #[test]
    fn max_settle_at_floor_accepted() {
        // Boundary: exactly 4 × settle passes. Catches off-by-one in
        // the floor comparison.
        let toml = minimal_toml("settle = \"100ms\"\nmax_settle = \"400ms\"\n");
        let cfg = Config::from_str(&toml).unwrap();
        assert_eq!(cfg.watches[0].max_settle, Duration::from_millis(400));
    }

    #[test]
    fn default_max_settle_is_one_hour_independent_of_settle() {
        // The 60× factor is gone — `max_settle` defaults to a flat 1h
        // regardless of `settle`. A few representative `settle` values
        // all observe the same default.
        for settle in ["1ms", "200ms", "5s", "30s", "1m"] {
            let toml = minimal_toml(&format!("settle = \"{settle}\"\n"));
            let cfg = Config::from_str(&toml).unwrap();
            assert_eq!(
                cfg.watches[0].max_settle,
                Duration::from_hours(1),
                "settle = {settle:?}",
            );
        }
    }

    #[test]
    fn max_settle_above_one_hour_accepted() {
        // No upper bound — operator may opt into multi-hour windows.
        let toml = minimal_toml("settle = \"200ms\"\nmax_settle = \"6h\"\n");
        let cfg = Config::from_str(&toml).unwrap();
        assert_eq!(cfg.watches[0].max_settle, Duration::from_hours(6),);
    }

    #[test]
    fn humantime_compound_settle_accepted() {
        // humantime accepts compound forms (`"1m 30s"`); pin the
        // semantics so a parser swap doesn't regress silently.
        let toml = minimal_toml("settle = \"1m 30s\"\nmax_settle = \"10m\"\n");
        let cfg = Config::from_str(&toml).unwrap();
        assert_eq!(cfg.watches[0].settle, Duration::from_secs(90));
        assert_eq!(cfg.watches[0].max_settle, Duration::from_mins(10));
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
            "[[watch]]\nname = \"a\"\npath = \"{ROOT}\"\nactions = [{{ exec = [\"echo\"] }}]\n\
             [[watch]]\nname = \"a\"\npath = \"{ROOT}\"\nactions = [{{ exec = [\"echo\"] }}]\n",
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
            "[[watch]]\nname = \"a\"\npath = \"{ROOT}\"\nactions = [{{ exec = [\"echo\"] }}]\n\
             [[watch]]\nname = \"a\"\npath = \"{ROOT}\"\nactions = [{{ exec = [\"echo\"] }}]\n\
             [[watch]]\nname = \"a\"\npath = \"{ROOT}\"\nactions = [{{ exec = [\"echo\"] }}]\n",
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
            "[[watch]]\nname = \"a\"\npath = \"{ROOT}\"\nactions = [{{ exec = [\"echo\"] }}]\nfoo = \"bar\""
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
             actions = [{{ exec = [\"fmt\", \"--input=${{specter.path}}\", \"${{specter.created}}\"] }}]"
        );
        let cfg = Config::from_str(&toml).unwrap();
        let SpawnBody::Exec(exec) = &cfg.watches[0].program.ops()[0].body else {
            panic!("expected SpawnBody::Exec");
        };
        let argv = &exec.argv;
        assert_eq!(argv.len(), 3);
        assert_eq!(argv[0].parts[0], ArgPart::literal("fmt"));
        assert_eq!(argv[1].parts[0], ArgPart::literal("--input="));
        assert_eq!(argv[1].parts[1], ArgPart::Placeholder(Placeholder::Path));
        assert_eq!(argv[2].parts[0], ArgPart::Placeholder(Placeholder::Created));
    }

    #[test]
    fn multiple_errors_in_one_watch_collected() {
        let toml = "[[watch]]\nname = \"\"\npath = \"src\"\nactions = [{ exec = [] }]\nsettle = \"0ms\"\nmax_depth = 0";
        let err = Config::from_str(toml).unwrap_err();
        let errors = validation_errors(err);
        let kinds: Vec<IssueKind> = errors.iter().map(|e| e.kind).collect();
        assert!(kinds.contains(&IssueKind::Empty));
        assert!(kinds.contains(&IssueKind::NonAbsolute));
        assert!(kinds.contains(&IssueKind::EmptyArgv));
        assert!(kinds.contains(&IssueKind::SettleTooSmall));
        assert!(kinds.contains(&IssueKind::MaxDepthZero));
        assert_eq!(errors.len(), 5);
    }

    #[test]
    fn errors_across_multiple_watches_preserve_source_order() {
        let toml = "[[watch]]\nname = \"a\"\npath = \"src1\"\nactions = [{ exec = [\"echo\"] }]\n\
                    [[watch]]\nname = \"b\"\npath = \"src2\"\nactions = [{ exec = [\"echo\"] }]\n";
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
        assert_eq!(w.max_settle, Duration::from_hours(1));
        assert!(w.scan.recursive);
    }

    #[test]
    fn pending_path_validates_via_lenient_canonicalize() {
        let td = tempfile::tempdir().unwrap();
        let pending = td.path().join("does-not-exist").join("leaf");
        let toml = format!(
            "[[watch]]\nname = \"p\"\npath = \"{}\"\nactions = [{{ exec = [\"echo\"] }}]",
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

    // ---- @-in-name rejection ----

    /// `@` is reserved for the synthesized `<promoter_name>@<resolved_path>`
    /// shape of dynamic Subs. A static [[watch]] with `@` in its name
    /// would collide with that scheme on a Promoter sharing the
    /// substring; reject at config-load.
    #[test]
    fn at_sign_in_static_name_rejected() {
        let toml = format!(
            "[[watch]]\nname = \"foo@bar\"\npath = \"{ROOT}\"\nactions = [{{ exec = [\"echo\"] }}]"
        );
        assert_only_kind(&toml, IssueKind::InvalidName);
    }

    /// Same rule for dynamic [[watch]] entries — operators get a
    /// consistent name-grammar regardless of which validator their
    /// path routes to.
    #[test]
    fn at_sign_in_dynamic_name_rejected() {
        let toml = "[[watch]]\nname = \"foo@bar\"\npath = \"/var/log/*\"\nactions = [{ exec = [\"echo\"] }]";
        assert_only_kind(toml, IssueKind::InvalidName);
    }

    /// Empty static name still surfaces as `Empty` (not `InvalidName`)
    /// — the helper short-circuits empty before checking `@`.
    #[test]
    fn empty_static_name_emits_empty_kind_not_invalid_name() {
        let toml = format!(
            "[[watch]]\nname = \"\"\npath = \"{ROOT}\"\nactions = [{{ exec = [\"echo\"] }}]"
        );
        assert_only_kind(&toml, IssueKind::Empty);
    }

    /// Empty dynamic name surfaces as `Empty` for the same reason.
    #[test]
    fn empty_dynamic_name_emits_empty_kind_not_invalid_name() {
        let toml =
            "[[watch]]\nname = \"\"\npath = \"/var/log/*\"\nactions = [{ exec = [\"echo\"] }]";
        assert_only_kind(toml, IssueKind::Empty);
    }

    // ---- Auto-detect dispatch ----

    /// Pure-literal absolute path → static dispatch → `Config.watches`.
    #[test]
    fn pure_literal_path_dispatches_to_static() {
        let toml = "[[watch]]\nname = \"static\"\npath = \"/var/log/myapp\"\nactions = [{ exec = [\"echo\"] }]";
        let cfg = Config::from_str(toml).unwrap();
        assert_eq!(cfg.watches.len(), 1);
        assert!(cfg.promoters.is_empty());
        assert_eq!(cfg.watches[0].name, "static");
    }

    /// Path with `*` discriminator → dynamic dispatch → `Config.promoters`.
    #[test]
    fn glob_star_path_dispatches_to_dynamic() {
        let toml =
            "[[watch]]\nname = \"dyn\"\npath = \"/var/log/*\"\nactions = [{ exec = [\"echo\"] }]";
        let cfg = Config::from_str(toml).unwrap();
        assert!(cfg.watches.is_empty());
        assert_eq!(cfg.promoters.len(), 1);
        assert_eq!(cfg.promoters[0].name, "dyn");
        assert_eq!(cfg.promoters[0].pattern.source(), "/var/log/*");
    }

    /// Path with `?` → dynamic.
    #[test]
    fn question_mark_path_dispatches_to_dynamic() {
        let toml =
            "[[watch]]\nname = \"dyn\"\npath = \"/srv/?/data\"\nactions = [{ exec = [\"echo\"] }]";
        let cfg = Config::from_str(toml).unwrap();
        assert_eq!(cfg.promoters.len(), 1);
    }

    /// Path with `[…]` → dynamic.
    #[test]
    fn bracket_path_dispatches_to_dynamic() {
        let toml = "[[watch]]\nname = \"dyn\"\npath = \"/srv/[a-z]/data\"\nactions = [{ exec = [\"echo\"] }]";
        let cfg = Config::from_str(toml).unwrap();
        assert_eq!(cfg.promoters.len(), 1);
    }

    /// Path with `{a,b}` (brace expansion) → dynamic [H-1].
    #[test]
    fn brace_path_dispatches_to_dynamic() {
        let toml = "[[watch]]\nname = \"dyn\"\npath = \"/var/log/{app,system}/access.log\"\n\
                    actions = [{ exec = [\"echo\"] }]";
        let cfg = Config::from_str(toml).unwrap();
        assert_eq!(cfg.promoters.len(), 1);
        // brace stays as a single Glob component; literal_prefix_len = 3.
        assert_eq!(cfg.promoters[0].pattern.literal_prefix_len(), 3);
    }

    /// Mixed config — both kinds in source order, each routed to the
    /// right slot. Source-order is preserved within each list, but
    /// across kinds the two lists are independent.
    #[test]
    fn mixed_static_and_dynamic_routes_each_correctly() {
        let toml = "\
            [[watch]]\nname = \"a\"\npath = \"/foo\"\nactions = [{ exec = [\"echo\"] }]\n\
            [[watch]]\nname = \"b\"\npath = \"/bar/*\"\nactions = [{ exec = [\"echo\"] }]\n\
            [[watch]]\nname = \"c\"\npath = \"/baz\"\nactions = [{ exec = [\"echo\"] }]\n\
            [[watch]]\nname = \"d\"\npath = \"/qux/{a,b}\"\nactions = [{ exec = [\"echo\"] }]\n\
        ";
        let cfg = Config::from_str(toml).unwrap();
        assert_eq!(cfg.watches.len(), 2);
        assert_eq!(cfg.promoters.len(), 2);
        assert_eq!(cfg.watches[0].name, "a");
        assert_eq!(cfg.watches[1].name, "c");
        assert_eq!(cfg.promoters[0].name, "b");
        assert_eq!(cfg.promoters[1].name, "d");
    }

    /// Cross-kind duplicate name still rejected — the duplicate-name
    /// check runs at the dispatch loop so both lists are scanned.
    #[test]
    fn duplicate_name_across_static_and_dynamic_rejected() {
        let toml = "\
            [[watch]]\nname = \"foo\"\npath = \"/foo\"\nactions = [{ exec = [\"echo\"] }]\n\
            [[watch]]\nname = \"foo\"\npath = \"/foo/*\"\nactions = [{ exec = [\"echo\"] }]\n\
        ";
        let err = Config::from_str(toml).unwrap_err();
        let errors = validation_errors(err);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, IssueKind::DuplicateName);
        assert_eq!(errors[0].watch_index, Some(1));
    }

    // ---- Dynamic pattern parse failures ----

    /// Globstar (`**`) is unsupported in v1 — surfaced as
    /// `IssueKind::InvalidPattern`.
    #[test]
    fn globstar_pattern_rejected_as_invalid_pattern() {
        let toml =
            "[[watch]]\nname = \"d\"\npath = \"/var/log/**/x\"\nactions = [{ exec = [\"echo\"] }]";
        assert_only_kind(toml, IssueKind::InvalidPattern);
    }

    /// Dynamic-detected non-absolute path (e.g., `var/log/*`) routes
    /// to the dynamic validator and the parser surfaces `NonAbsolute`
    /// as `IssueKind::InvalidPattern` with the source rendered.
    #[test]
    fn relative_dynamic_path_rejected_as_invalid_pattern() {
        let toml =
            "[[watch]]\nname = \"d\"\npath = \"var/log/*\"\nactions = [{ exec = [\"echo\"] }]";
        let err = Config::from_str(toml).unwrap_err();
        let errors = validation_errors(err);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, IssueKind::InvalidPattern);
        assert!(
            errors[0].detail.contains("var/log/*"),
            "got {}",
            errors[0].detail
        );
        assert!(
            errors[0].detail.contains("absolute"),
            "Display message must mention `absolute`; got {}",
            errors[0].detail,
        );
    }

    /// Double-slash → empty segment via PatternSpec parser → InvalidPattern.
    #[test]
    fn double_slash_dynamic_path_rejected_as_invalid_pattern() {
        let toml =
            "[[watch]]\nname = \"d\"\npath = \"//var/log/*\"\nactions = [{ exec = [\"echo\"] }]";
        assert_only_kind(toml, IssueKind::InvalidPattern);
    }

    /// Malformed glob segment — unbalanced `[` — surfaces via the
    /// PatternSpec parser as `InvalidGlob`, which we re-cast to
    /// `IssueKind::InvalidPattern`.
    #[test]
    fn malformed_glob_segment_rejected_as_invalid_pattern() {
        let toml = "[[watch]]\nname = \"d\"\npath = \"/var/log/[unbalanced\"\nactions = [{ exec = [\"echo\"] }]";
        assert_only_kind(toml, IssueKind::InvalidPattern);
    }

    // ---- PromoterSpec materialization ----

    /// Minimal dynamic watch round-trips defaults the same way as the
    /// static validator (settle = 200ms, max_settle = 12s, etc.).
    #[test]
    fn minimal_dynamic_watch_round_trips_with_defaults() {
        let toml =
            "[[watch]]\nname = \"logs\"\npath = \"/var/log/*\"\nactions = [{ exec = [\"echo\"] }]";
        let cfg = Config::from_str(toml).unwrap();
        let p = &cfg.promoters[0];
        assert_eq!(p.name, "logs");
        assert_eq!(p.scope, EffectScope::SubtreeRoot);
        assert_eq!(p.settle, Duration::from_millis(200));
        assert_eq!(p.max_settle, Duration::from_hours(1));
        assert!(p.scan.recursive);
        assert_eq!(p.events, ClassSet::DEFAULT_SUBTREE_ROOT);
        assert!(!p.log_output);
    }

    /// `to_attach_request` threads every field into a
    /// `PromoterAttachRequest` byte-equal to the spec.
    #[test]
    fn promoter_to_attach_request_threads_fields() {
        let toml = "[[watch]]\nname = \"logs\"\npath = \"/var/log/*\"\n\
                    actions = [{ exec = [\"fmt\", \"${specter.path}\"] }]\n\
                    settle = \"300ms\"\nmax_settle = \"1200ms\"\n\
                    scope = \"per-stable-file\"\n\
                    events = [\"content\"]\n\
                    log_output = true\n\
                    pattern = \"*.log\"\n\
                    recursive = false\nhidden = true\n";
        let cfg = Config::from_str(toml).unwrap();
        let req = cfg.promoters[0].to_attach_request();
        assert_eq!(req.name, "logs");
        assert_eq!(req.pattern_spec.source(), "/var/log/*");
        assert_eq!(req.scope, EffectScope::PerStableFile);
        assert_eq!(req.settle, Duration::from_millis(300));
        assert_eq!(req.max_settle, Duration::from_millis(1200));
        assert_eq!(req.events, ClassSet::CONTENT);
        assert!(req.log_output);
        assert!(!req.config.recursive);
        assert!(req.config.hidden);
        assert!(req.config.pattern.is_some());
    }

    /// Dynamic watches accept `pattern` (per-Sub include filter) and
    /// `exclude` (per-Sub exclude list) the same way static watches
    /// do — they're orthogonal to the path-pattern dispatch.
    #[test]
    fn dynamic_watch_carries_scan_pattern_and_excludes() {
        let toml = "[[watch]]\nname = \"logs\"\npath = \"/var/log/*\"\n\
                    actions = [{ exec = [\"echo\"] }]\n\
                    pattern = \"*.log\"\n\
                    exclude = [\"*.gz\"]\n";
        let cfg = Config::from_str(toml).unwrap();
        let p = &cfg.promoters[0];
        assert!(p.scan.pattern.is_some());
        assert_eq!(p.scan.exclude.len(), 1);
    }

    /// Multiple errors in one dynamic watch accumulate — pattern parse
    /// failure does NOT short-circuit settle / scope / events
    /// validation. The operator gets the full list at once.
    #[test]
    fn multiple_errors_in_one_dynamic_watch_accumulate() {
        let toml = "[[watch]]\nname = \"\"\npath = \"/foo/**/x\"\nactions = [{ exec = [] }]\n\
                    settle = \"0ms\"\nmax_depth = 0";
        let err = Config::from_str(toml).unwrap_err();
        let errors = validation_errors(err);
        let kinds: Vec<IssueKind> = errors.iter().map(|e| e.kind).collect();
        assert!(kinds.contains(&IssueKind::Empty));
        assert!(kinds.contains(&IssueKind::InvalidPattern));
        assert!(kinds.contains(&IssueKind::EmptyArgv));
        assert!(kinds.contains(&IssueKind::SettleTooSmall));
        assert!(kinds.contains(&IssueKind::MaxDepthZero));
        assert_eq!(errors.len(), 5, "got {errors:?}");
    }

    /// FS-root pattern `/*` parses to a one-segment glob; spec carries
    /// `literal_prefix_len = 1`.
    #[test]
    fn fs_root_glob_pattern_accepted() {
        let toml = "[[watch]]\nname = \"root\"\npath = \"/*\"\nactions = [{ exec = [\"echo\"] }]";
        let cfg = Config::from_str(toml).unwrap();
        assert_eq!(cfg.promoters.len(), 1);
        assert_eq!(cfg.promoters[0].pattern.literal_prefix_len(), 1);
    }

    /// The static validator's defensive `is_dynamic` re-check fires
    /// only for direct internal callers (tests bypass the dispatcher).
    /// The production dispatch path never lands here. Tested via the
    /// validator function directly, mirroring the defense-in-depth
    /// contract.
    #[test]
    fn static_validator_rejects_glob_path_via_defensive_check() {
        // Construct a `RawWatch` by hand so we bypass the dispatcher.
        let raw = crate::raw::RawWatch::for_test(
            "name".to_owned(),
            "/var/log/*".to_owned(),
            vec!["echo".to_owned()],
        );
        let errors = super::validate_static_watch(0, &raw).unwrap_err();
        let kinds: Vec<IssueKind> = errors.iter().map(|e| e.kind).collect();
        assert!(
            kinds.contains(&IssueKind::PathContainsGlobChars),
            "got {kinds:?}",
        );
    }
}

#[cfg(all(test, unix))]
mod from_path_with_meta_tests {
    use super::Config;
    use crate::FileMeta;
    use std::path::Path;
    use tempfile::TempDir;

    const MINIMAL: &str =
        "[[watch]]\nname = \"x\"\npath = \"/\"\nactions = [{ exec = [\"echo\"] }]\n";

    fn write(path: &Path, bytes: &[u8]) {
        std::fs::write(path, bytes).expect("write tempfile");
    }

    #[test]
    fn from_path_with_meta_returns_consistent_config_and_meta() {
        let dir = TempDir::new().unwrap();
        let p = dir.path().join("specter.toml");
        write(&p, MINIMAL.as_bytes());

        let (cfg, meta) = Config::from_path_with_meta(&p).unwrap();

        assert_eq!(cfg.watches.len(), 1);
        assert_eq!(cfg.watches[0].name, "x");

        // Without intervening mutation, the lstat-equivalent
        // re-capture compares bit-equal to the atomically-captured
        // meta — this is the steady-state invariant the driver's
        // settle-expiry filter relies on for "no change ⇒ no
        // reload".
        let lstat_meta = FileMeta::from_path(&p).unwrap();
        assert_eq!(meta, lstat_meta);

        assert_eq!(meta.size, MINIMAL.len() as u64);
    }

    #[test]
    fn from_path_with_meta_matches_from_path_on_config_payload() {
        let dir = TempDir::new().unwrap();
        let p = dir.path().join("specter.toml");
        write(&p, MINIMAL.as_bytes());

        let cfg_only = Config::from_path(&p).unwrap();
        let (cfg_with_meta, _meta) = Config::from_path_with_meta(&p).unwrap();

        // The two entry points must produce identical Config values —
        // `from_path` is the portable path; `from_path_with_meta`
        // additionally returns the inode meta. Divergence here would
        // mean reloads (which take the meta path) parse differently
        // from initial loads (which historically took `from_path`).
        assert_eq!(cfg_only, cfg_with_meta);
    }

    #[test]
    fn from_path_with_meta_propagates_io_errors() {
        let dir = TempDir::new().unwrap();
        let missing = dir.path().join("never-existed.toml");

        let err = Config::from_path_with_meta(&missing).unwrap_err();
        assert!(matches!(err, super::ConfigError::Io { .. }));
    }

    #[test]
    fn from_path_with_meta_propagates_parse_errors() {
        let dir = TempDir::new().unwrap();
        let p = dir.path().join("broken.toml");
        write(&p, b"this-is-not = = valid toml\n");

        let err = Config::from_path_with_meta(&p).unwrap_err();
        assert!(matches!(err, super::ConfigError::Parse { .. }));
    }

    #[test]
    fn captured_meta_inode_pinned_against_atomic_save_during_read() {
        // The full atomic-capture invariant: `from_path_with_meta`'s
        // returned meta belongs to the inode opened, even when the
        // path is renamed out from under us between `File::open` and
        // any subsequent path-level stat. Simulates the atomic-save
        // race by performing the rename after the call returns and
        // confirming `meta` still reflects the original (now orphan)
        // inode while a fresh path-level `from_path` reflects the
        // replacement.
        let dir = TempDir::new().unwrap();
        let p = dir.path().join("specter.toml");
        write(&p, MINIMAL.as_bytes());

        let (cfg, meta) = Config::from_path_with_meta(&p).unwrap();
        assert_eq!(cfg.watches.len(), 1);

        // Atomic-save shape: replace path with a new inode.
        let staging = dir.path().join("specter.toml.new");
        write(&staging, MINIMAL.as_bytes());
        std::fs::rename(&staging, &p).unwrap();

        let lstat_after_save = FileMeta::from_path(&p).unwrap();
        assert_ne!(
            meta.inode, lstat_after_save.inode,
            "rename must produce a fresh inode at the path",
        );
        // The driver's lstat filter would detect this as `stored !=
        // current` and fire a reload; this test asserts the precondition
        // (meta deltas are observable across atomic-save).
    }
}
