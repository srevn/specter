use crate::action::{Action, lower_to_program};
use crate::error::{ConfigError, IssueKind, ValidationIssue};
use crate::file_meta::FileMeta;
use crate::path::{PathError, canonicalize_lenient};
use crate::raw::{RawAction, RawConfig, RawExec, RawLogConfig, RawWatch};
use crate::template;
use compact_str::CompactString;
use specter_core::{
    self as core, ActionProgram, ArgTemplate, ClassSet, EffectScope, ExecAction, GlobPattern,
    MintTemplate, PatternSpec, ProfileIdentity, ReactionSpec, ScanConfig, SpawnSpec,
    SubAttachAnchor, SubAttachRequest, SubParams,
};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

/// Default debounce window when `[[watch]] settle` is omitted.
pub(crate) const DEFAULT_SETTLE: Duration = Duration::from_millis(200);
/// Default forced-fire deadline when `[[watch]] max_settle` is omitted. Flat 1 hour, independent of
/// `settle` — if the tree stays active for an hour, the user's workflow is outside Specter's scope;
/// manual triggering is the better answer.
pub(crate) const DEFAULT_MAX_SETTLE: Duration = Duration::from_hours(1);
/// Lower bound on `max_settle` relative to `settle`. Catches the swap typo (`settle = "1h"`,
/// `max_settle = "200ms"`) and the semantic nonsense of `max_settle ≤ settle` (a single settle
/// round would already exceed it).
const MAX_SETTLE_FLOOR_FACTOR: u32 = 4;
/// Debounce window of a discovery Sub's own Profile — the walk that observes chain membership, not
/// the user's reaction. Pinned to a constant (never the user's `settle`) so a `settle = "30s"`
/// reaction debounce cannot become 30 s of mint latency, and so one pattern always maps to one
/// discovery Profile (`Profile.settle = min over attached Subs` is constant-stable when every
/// template carries the same pair). 150 ms coalesces an untar-style membership burst into one
/// reconcile.
const DISCOVERY_SETTLE: Duration = Duration::from_millis(150);
/// Forced-fire ceiling of a discovery Profile — bounds mint latency under sustained chain churn.
/// See [`DISCOVERY_SETTLE`] for why the pair is constant.
const DISCOVERY_MAX_SETTLE: Duration = Duration::from_secs(2);
// Compile-time pin of the `validate_settle` floor the constants bypass (they never pass through the
// raw-field validator): a drift below `4 × settle` would otherwise only surface as `Profile::new`'s
// debug assertion at attach time.
const _: () = assert!(
    DISCOVERY_MAX_SETTLE.as_millis()
        >= MAX_SETTLE_FLOOR_FACTOR as u128 * DISCOVERY_SETTLE.as_millis()
);
/// Hard cap on `[[watch.actions]]` conditional nesting depth. Each `when` / `then` / `else` triple
/// descends one level of validator recursion; [`validate_action_list`] short-circuits with
/// [`IssueKind::ConditionalNestedTooDeep`] when the depth bound is exceeded, keeping the
/// validator's stack consumption finite under adversarial input. Sensible operator configs are ≤5
/// levels deep, so `32` is generous defense-in-depth, not a real workflow constraint. The bound is
/// parser-independent — the underlying TOML crate's own recursion limit is a separate concern.
const MAX_CONDITIONAL_DEPTH: u8 = 32;

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Config {
    pub log: LogConfig,
    /// Every `[[watch]]` block, in source order. Each entry maps to one [`SubSpec`] and is attached
    /// as a Sub by the bin's initial-attach pass. The static/dynamic dispatch happens on `path`
    /// ([`PatternSpec::is_dynamic`]): a glob-bearing path lowers to a discovery Sub — a
    /// template-bearing [`SubSpec`] whose Profile walks the pattern's match chain and mints one
    /// dynamic Sub per match — while a literal path lowers to a plain static [`SubSpec`]. One list,
    /// one attach pipeline; the kind difference is carried by [`SubSpec::template`].
    pub watches: Vec<SubSpec>,
}

/// Engine-telemetry configuration — the operator-facing diagnostic stream's level, sink, and (for
/// [`LogDestination::File`]) target path.
///
/// This block is *only* about engine logs. Subprocess output is a separate concern controlled
/// per-watch by [`SubSpec::log_output`].
#[derive(Debug, Default, Clone, Eq, PartialEq)]
pub struct LogConfig {
    pub level: LogLevel,
    pub destination: LogDestination,
    /// Required iff `destination == LogDestination::File`. Validated at load time: must be
    /// absolute. For `LogDestination::Stderr`, callers should ignore this field.
    pub path: Option<PathBuf>,
}

impl LogConfig {
    /// Merge CLI overrides onto a config-loaded [`LogConfig`].
    ///
    /// Precedence is symmetric for every field: `CLI > config > default`. When destination resolves
    /// to [`LogDestination::File`] but no path was supplied (neither CLI nor config), returns a
    /// [`ValidationIssue`] with [`IssueKind::EmptyLogPath`] on `log.path`. CLI-supplied paths must
    /// be absolute (matching the config-time rule), or the same error surfaces with
    /// [`IssueKind::NonAbsolute`].
    ///
    /// Returns the bare [`ValidationIssue`] rather than wrapping it in [`ConfigError::Validate`]:
    /// this entry point is the CLI-merge flow, not a TOML parse; wrapping would mislabel the
    /// operator- visible source as `<inline>` (the [`ConfigError::Validate`] fallback when no file
    /// path is associated). The bin caller owns the CLI-source context in the format string instead
    /// — see `specter-bin`'s `App::run` and `EngineDriver::apply_log_reload`.
    pub fn merge_cli(
        mut self,
        level: Option<LogLevel>,
        destination: Option<LogDestination>,
        path: Option<&Path>,
    ) -> Result<Self, ValidationIssue> {
        if let Some(l) = level {
            self.level = l;
        }
        if let Some(d) = destination {
            self.destination = d;
        }
        if let Some(p) = path {
            self.path = Some(p.to_path_buf());
        }
        self.path = validate_log_path(
            self.destination,
            self.path.as_deref(),
            " (provide --log-path or `[log] path` in the config)",
        )?;
        Ok(self)
    }
}

/// Resolve the `(destination, path)` pair against the
/// [`LogDestination::File`]-implies-absolute-path rule. Returns `Ok(Some(p))` when File is paired
/// with an absolute path, `Ok(None)` when [`LogDestination::Stderr`] (path is dropped — File is the
/// only destination that carries a path) and `Err(issue)` otherwise. Single-issue by construction:
/// one input pair, one possible failure mode.
///
/// Shared by [`LogConfig::merge_cli`] (the CLI-overrides flow, applied post-config-load by
/// `specter-bin`) and [`validate_log`] (the config-load flow). The structural rule lives here once;
/// callers don't recheck it.
///
/// `empty_hint` is appended to the [`IssueKind::EmptyLogPath`] detail when File is paired with no
/// path. The CLI flow passes `" (provide --log-path or `\[log\] path` in the config)"` so the
/// operator sees both override sites; the config-load flow passes the empty string for the bare
/// structural rule.
fn validate_log_path(
    destination: LogDestination,
    path: Option<&Path>,
    empty_hint: &str,
) -> Result<Option<PathBuf>, ValidationIssue> {
    match (destination, path) {
        (LogDestination::Stderr, _) => Ok(None),
        (LogDestination::File, None) => Err(ValidationIssue::new(
            None,
            "log.path",
            IssueKind::EmptyLogPath,
            format!("log.path is required when destination = \"file\"{empty_hint}"),
        )),
        (LogDestination::File, Some(p)) if p.is_absolute() => Ok(Some(p.to_path_buf())),
        (LogDestination::File, Some(p)) => Err(ValidationIssue::new(
            None,
            "log.path",
            IssueKind::NonAbsolute,
            format!("log.path `{}` must be absolute", p.display()),
        )),
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Default, clap::ValueEnum)]
pub enum LogDestination {
    /// Engine telemetry to stderr. Supervisor (systemd / launchd / FreeBSD `daemon -o`) captures it.
    #[default]
    Stderr,
    /// Engine telemetry to a regular file via `tracing-appender`'s non-blocking writer. Reopened on
    /// SIGHUP for logrotate `copytruncate`-style rotation.
    File,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct SubSpec {
    pub name: CompactString,
    pub path: PathBuf,
    /// Lowered bytecode IR. Built once at config validation; cloned by Arc into each
    /// [`SubAttachRequest`] (and from there into every emitted `Effect`). Equality is structural
    /// over the instruction sequence — two TOML configs that lower to the same program compare
    /// equal, so the hot-reload diff suppresses no-op churn on cosmetic edits (whitespace, comment,
    /// key ordering).
    pub program: Arc<ActionProgram>,
    pub scope: EffectScope,
    pub settle: Duration,
    pub max_settle: Duration,
    pub scan: ScanConfig,
    /// User-declared event-class mask. Materialized by `validate_watch` — explicit when the TOML
    /// carries an `events` array, otherwise the scope-conditional default
    /// ([`ClassSet::DEFAULT_SUBTREE_ROOT`] for `subtree-root`, [`ClassSet::DEFAULT_PER_FILE`] for
    /// `per-stable-file`). Folded into the Profile's `config_hash` by the engine —
    /// `PartialEq`-derived diffs ensure a hot-reload flip on this field reaps the old Profile and
    /// attaches a fresh one.
    pub events: ClassSet,
    /// Forward subprocess stdout/stderr to Specter's own stdio. False by default — children run with
    /// `Stdio::null()`. When true, the actuator uses `Stdio::inherit()` and the supervisor's log
    /// facility (systemd journal, launchd `StandardOutPath`, FreeBSD `daemon -o`) captures the bytes.
    /// Engine threads this through `SubAttachRequest` → `Sub.log_output` → `Effect.capture_output`.
    pub log_output: bool,
    /// Operator-controlled suppression flag. `true` (TOML default) ⇒ the entry is effective;
    /// `false` ⇒ structurally equivalent to "absent from the config." Disabled entries flow through
    /// parsing and validation unchanged (so typos surface at config load, not silently at re-enable
    /// time) but are filtered out of every runtime view by [`Config::active_watches`]. The engine
    /// never learns about disabled entries — every transition (initial attach, hot-reload diff,
    /// drain-window derivation) consumes the filtered iterator.
    ///
    /// Included in [`PartialEq`] so two specs differing only on this field compare unequal. The
    /// diff layer's filter strips disabled entries *before* the equality check, so this matters
    /// only for future consumers that compare unfiltered specs.
    ///
    /// **Cost of a disable → re-enable cycle.** Because the diff surfaces a `false → true` flip as
    /// `subs.added` (and the reverse as `subs.removed`), every disable/re-enable cycle reaps the
    /// Sub's Profile (its last live Sub left) and mints a fresh Profile on re-enable. The fresh
    /// Profile has no baseline, so changes that landed on the tree *during* the disabled window are
    /// folded into the first post-re-enable Seed rather than surfacing as a fire. Operators relying
    /// on baseline continuity across reconfiguration should avoid the disable/re-enable pattern for
    /// transient toggles; a future "suspend, don't reap" path is the deeper fix. For a discovery
    /// Sub the same cycle additionally reaps the minted set (the detach cascade) and re-mints fresh
    /// `SubId`s on the first post-re-enable reconcile.
    pub enabled: bool,
    /// `Some` ⇒ this spec is a discovery Sub lowered from a dynamic `[[watch]]` block: `scan` is
    /// `MatchChain`, `path` is the pattern's canonicalised literal prefix, `settle`/`max_settle`/
    /// `events` are the discovery constants, and every user knob lives here instead — the identity
    /// the minted Subs' Profiles run under. `None` for static watches.
    pub template: Option<TemplateSpec>,
}

/// The user's knobs of a dynamic `[[watch]]` block.
///
/// Everything here is what the minted Subs (not the discovery Sub itself) run under, lowered
/// verbatim into a [`specter_core::MintTemplate`] by [`SubSpec::to_attach_request`].
///
/// Config-side mirror rather than `MintTemplate` directly: `MintTemplate`/`ProfileIdentity`
/// deliberately carry no `Eq` (the hash is the engine's only identity comparison), while the diff
/// layer compares *specs* structurally — its existing discipline. The mirror keeps that discipline
/// without weakening the core types, and keeps the sealed `SpawnSpec` / eager identity hash out of
/// the `Eq` derive (the spec's flat `program`/`scope`/`log_output` already carry the same
/// information for comparison).
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct TemplateSpec {
    /// Minted Profiles' scan — the `Subtree` built from the block's `recursive` / `hidden` /
    /// `max_depth` / `pattern` / `exclude` knobs.
    pub scan: ScanConfig,
    /// Minted Profiles' event-class mask (the block's `events`, or its scope-conditional default).
    pub events: ClassSet,
    /// Minted Subs' debounce (the block's `settle`).
    pub settle: Duration,
    /// Minted Profiles' forced-fire ceiling (the block's `max_settle`). Validated against `settle`
    /// by `validate_settle` on the raw fields — the user pair keeps today's meaning exactly.
    pub max_settle: Duration,
}

impl SubSpec {
    #[must_use]
    pub fn to_attach_request(&self) -> SubAttachRequest {
        // The spec's flat program/scope/log_output seal into one SpawnSpec either way; the template
        // fork decides whose reaction it is — the Sub's own (static) or the minted Subs' (the
        // discovery Sub itself spawns nothing).
        let spawn = SpawnSpec::new(Arc::clone(&self.program), self.scope, self.log_output);
        SubAttachRequest::from_parts(
            SubAttachAnchor::Path(self.path.clone()),
            ProfileIdentity::new(self.scan.clone(), self.max_settle, self.events),
            SubParams {
                name: self.name.clone(),
                settle: self.settle,
                reaction: match &self.template {
                    // The projection adds nothing: the template's knobs land in the MintTemplate
                    // verbatim, so the minted Profiles' identity hash equals one hand-built over
                    // the same user fields.
                    Some(t) => ReactionSpec::Mint(Arc::new(MintTemplate {
                        identity: ProfileIdentity::new(t.scan.clone(), t.max_settle, t.events),
                        settle: t.settle,
                        spawn,
                    })),
                    None => ReactionSpec::Spawn {
                        spec: spawn,
                        minted_by: None,
                    },
                },
            },
        )
    }

    /// True iff changing the live attachment from `self` to `other` would require a different
    /// Profile partition — the discriminator behind hot-reload's `modified_identity` vs
    /// `modified_params` split.
    ///
    /// The engine partitions Profiles on `(anchor_resource, ProfileIdentity::config_hash())`. Four
    /// `SubSpec` fields fold into that key: `path` resolves into the anchor resource; `scan`,
    /// `max_settle`, and `events` fold into the hash via `ProfileIdentity::config_hash`. Any
    /// difference on these forces the Sub onto a different Profile and routes the entry into
    /// `modified_identity`.
    ///
    /// The remaining `SubSpec` fields — `program`, `scope`, `settle`, `log_output` — live on
    /// `SubParams` and rebind in place; they do not partition the Profile. The `name` field is the
    /// diff key (two specs differing on `name` are distinct entries, not a modification) and
    /// `enabled` is filtered out before the diff runs (an enabled-flip surfaces as added/removed,
    /// not modified).
    ///
    /// Field-derived, not hash-derived: comparing fields directly is allocation-free and avoids
    /// constructing a [`ProfileIdentity`] only to compare and discard.
    ///
    /// **Template-bearing pairs always classify identity.** The diff only consults this on unequal
    /// specs, so the head guard reads: *any* field change on a discovery spec — including
    /// `program`/`scope`/`log_output`, the params-class fields — is a wholesale replace, never an
    /// in-place rebind. Minted Subs hold `Arc`s of the template Sub's program; a rebind would
    /// strand them on the old program (the registry's both-`None` rebind assertion is the
    /// engine-side backstop). Static↔dynamic path edits fall out of the same guard: template
    /// presence differs across the pair.
    #[must_use]
    pub(crate) fn requires_new_profile(&self, other: &Self) -> bool {
        if self.template.is_some() || other.template.is_some() {
            return true;
        }
        self.path != other.path
            || self.scan != other.scan
            || self.max_settle != other.max_settle
            || self.events != other.events
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
    /// Sole authority for "what's effective right now": every runtime consumer ([`crate::diff()`],
    /// the bin's initial-attach pass, [`crate::Config`] drain-window derivation, the startup /
    /// reload load logs) goes through this helper. Iterating the raw [`Self::watches`] field
    /// directly bypasses the per-entry `enabled` filter and is almost always wrong outside config
    /// introspection / round-trip serialization.
    ///
    /// Discipline: `enabled = false ⇔ entry absent from the effective config`. Every Add/Remove
    /// transition the engine handles flows from a flip in this iterator's output, so disabled
    /// entries never reach the engine — they remain in `self.watches` for introspection but are
    /// otherwise inert.
    pub fn active_watches(&self) -> impl Iterator<Item = &SubSpec> + '_ {
        self.watches.iter().filter(|s| s.enabled)
    }

    /// Resolve an operator-facing watch name to its [`SubSpec`] when the entry is enabled — the
    /// name-keyed inverse of [`Self::active_watches`].
    ///
    /// Returns `None` when the name is absent OR when its entry carries `enabled = false`; callers
    /// needing to distinguish the two cases inspect [`Self::watches`] directly.
    ///
    /// O(N) linear scan over [`Self::watches`]; static-name uniqueness (enforced upstream by
    /// `validate`) guarantees at most one match.
    #[must_use]
    pub fn find_active_watch(&self, name: &str) -> Option<&SubSpec> {
        self.active_watches().find(|s| s.name == name)
    }

    /// Names of every operator-suppressed entry — the complement of [`Self::active_watches`]
    /// flattened to the names the runtime needs for tracing, in source order. Sole consumers are
    /// the startup-info log and the per-load `"config loaded"` summary — both want the same
    /// `?disabled_watches` payload, so routing them through one helper keeps the two surfaces from
    /// drifting apart when the underlying spec shape evolves.
    #[must_use]
    pub fn disabled_names(&self) -> Vec<&str> {
        self.watches
            .iter()
            .filter(|s| !s.enabled)
            .map(|s| s.name.as_str())
            .collect()
    }

    /// Advisory findings on a validated config — hazards that load and run but probably don't mean
    /// what the operator wrote. Pull-computed rather than carried on the value: `Config` stays
    /// exactly the shape the reload diff compares, and the startup load parses before tracing init,
    /// so the bin pulls and logs these once a subscriber exists (startup and every reload pulse).
    /// Pure: no filesystem access — dynamic prefixes canonicalise at lowering (like static paths),
    /// so anchor divergence is structurally impossible and there is nothing here to resolve.
    ///
    /// One kind today, [`IssueKind::EventsIncompleteMask`]: the watch's effective event mask
    /// (static `events`, or the dynamic template's) cannot witness its scan shape's quiescence
    /// classes, so every fire is proven by the hash channel — at least two consecutive agreeing
    /// full subtree walks at the anchor, mtime-skip disabled. A supported, deliberately expensive
    /// configuration (the safety net for writers the kernel may not surface as events); the warning
    /// makes the cost visible.
    ///
    /// Disabled entries warn too, mirroring the validator's discipline: hazards surface at config
    /// load, not at re-enable time.
    #[must_use]
    pub fn warnings(&self) -> Vec<ValidationIssue> {
        let mut found = Vec::new();
        for (i, spec) in self.watches.iter().enumerate() {
            // The identity the watch's firing Profiles actually run under: a static entry's own
            // (scan, events); a dynamic entry's template pair (the discovery Sub itself carries the
            // STRUCTURE constant, which always witnesses its MatchChain shape).
            let (scan, events) = match &spec.template {
                Some(t) => (&t.scan, t.events),
                None => (&spec.scan, spec.events),
            };
            if !events.contains(scan.quiescence_witness_classes()) {
                found.push(ValidationIssue::new(
                    Some(i),
                    "events",
                    IssueKind::EventsIncompleteMask,
                    "events mask cannot witness this scan shape's quiescence classes \
                     (for a subtree watch: content), so settle-window silence proves nothing \
                     and every fire requires two consecutive agreeing full subtree walks at \
                     the anchor with mtime-skip disabled; intended for mmap/async-I/O/splice \
                     writers the kernel may not surface as events — add \"content\" to events \
                     if no such writers touch this tree"
                        .to_owned(),
                ));
            }
        }
        found
    }

    /// Parse a TOML string into a validated `Config`.
    ///
    /// Inherent name shadows `std::str::FromStr::from_str` (which is also implemented for ergonomic
    /// `"...".parse::<Config>()` use); both resolve to the same logic.
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

    /// Atomic content + filesystem-identity capture: opens `path`, captures [`FileMeta`] from the
    /// bound inode, then reads the content from the same handle. The inode is pinned by `f`, so a
    /// concurrent `rename(2)` over `path` (atomic-save) cannot rotate the meta out from under the
    /// bytes — the next `FileMeta::from_path` observes the path-level rotation as a meta delta.
    ///
    /// `f.metadata()` is called **before** the `read_to_string` so that any in-place mutation of
    /// the still-bound inode during the read surfaces on the next path-level lstat as `stored !=
    /// current`. Reversing the order would absorb the mutation into the stored meta and silently
    /// pin the loader to stale content.
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
        validate(&raw, path)
    }
}

impl std::str::FromStr for Config {
    type Err = ConfigError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::from_str_inner(s, None)
    }
}

/// Emit the `"config loaded"` info-level event with shape shared by [`Config::from_path`] and
/// [`Config::from_path_with_meta`].
///
/// `disabled_watches` carries the names of entries the operator suppressed via `enabled = false`.
/// The macro renders an empty `Vec` as `[]` — accept the noise for the all-enabled case rather than
/// branching the format string. Operators triaging "why isn't watch X firing?" can grep the log for
/// the watch's name in the disabled list rather than re-reading the TOML. `discovery` is the
/// template-bearing subset of `watches` — the operator's "how many patterns" view.
fn log_config_loaded(cfg: &Config, path: &Path) {
    let disabled_watches = cfg.disabled_names();
    tracing::info!(
        path = %path.display(),
        watches = cfg.watches.len(),
        discovery = cfg.watches.iter().filter(|s| s.template.is_some()).count(),
        ?disabled_watches,
        "config loaded",
    );
}

fn validate(raw: &RawConfig, path: Option<&Path>) -> Result<Config, ConfigError> {
    let mut errors: Vec<ValidationIssue> = Vec::new();

    // Log errors land first in the output (operator-facing ordering). On the failure path the
    // default `LogConfig` is irrelevant — `errors.is_empty()` below short-circuits to `Err`.
    let log = match validate_log(&raw.log) {
        Ok(log) => log,
        Err(mut errs) => {
            errors.append(&mut errs);
            LogConfig::default()
        }
    };

    let mut watches: Vec<SubSpec> = Vec::new();
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

        // Auto-detect: any of `*?[{` in `path` routes the entry to the dynamic validator. The
        // dispatcher is the contract — neither validator second-guesses it on the well-trodden
        // path. Both kinds lower to a SubSpec in the one source-ordered list; the dynamic one is
        // template-bearing.
        let validated = if PatternSpec::is_dynamic(&raw_w.path) {
            validate_dynamic_watch(i, raw_w)
        } else {
            validate_static_watch(i, raw_w)
        };
        match validated {
            Ok(spec) => watches.push(spec),
            Err(mut errs) => errors.append(&mut errs),
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

/// Resolve the `[log]` block. Returns `Ok(LogConfig)` when every field validates;
/// `Err(Vec<ValidationIssue>)` collects every field-level failure (level / destination / path can
/// each independently fail, so multi-issue is the right shape).
///
/// `raw` is materialised by serde from either a present `[log]` table or the implicit `default()`
/// for an absent one (see [`RawConfig::log`](crate::raw::RawConfig::log)); both shapes unfold into
/// the documented defaults here (`LogLevel::Info`, `LogDestination::Stderr`, `path = None`).
fn validate_log(raw: &RawLogConfig) -> Result<LogConfig, Vec<ValidationIssue>> {
    let mut errors: Vec<ValidationIssue> = Vec::new();

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

    let path = match validate_log_path(destination, raw.path.as_deref().map(Path::new), "") {
        Ok(p) => p,
        Err(issue) => {
            errors.push(issue);
            None
        }
    };

    if errors.is_empty() {
        Ok(LogConfig {
            level,
            destination,
            path,
        })
    } else {
        Err(errors)
    }
}

/// Validate the `name` field. Two failures are mutually exclusive: empty (rejected as
/// [`IssueKind::EmptyName`]) and `@`-bearing (rejected as [`IssueKind::InvalidName`] — `@` is
/// reserved for the engine's minted `<template_name>@<matched_path>` shape). Single-issue by
/// construction — at most one failure mode per call.
///
/// Both static and dynamic validators call this so the rule lives in one place. Duplicate-name
/// detection is handled at the outer dispatch loop (it spans both kinds and so cannot be a
/// per-watch helper concern).
fn validate_name(idx: usize, raw_name: &str) -> Result<(), ValidationIssue> {
    if raw_name.is_empty() {
        return Err(ValidationIssue::new(
            Some(idx),
            "name",
            IssueKind::EmptyName,
            "name must not be empty".to_owned(),
        ));
    }
    if raw_name.contains('@') {
        return Err(ValidationIssue::new(
            Some(idx),
            "name",
            IssueKind::InvalidName,
            format!(
                "name `{raw_name}` must not contain `@` (reserved for \
                 minted dynamic Sub names of the form \
                 `<template_name>@<matched_path>`)",
            ),
        ));
    }
    Ok(())
}

/// Validate the `actions` array and lower the resulting AST into the engine's bytecode IR. The
/// [`Result`] shape ties the validated program to the absence of errors at the type level — callers
/// cannot reach for the `Arc` without first resolving the failure case.
///
/// The empty-input case (`actions = []`) is rejected as [`IssueKind::EmptyActions`]; partial
/// programs are never handed back (a half-built program in the engine would be observably worse
/// than none at all).
///
/// The returned Arc is the same allocation the engine's `Sub.program` and every emitted
/// `Effect.program` references — one shared bytecode IR per validated Sub.
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

    let tree = validate_action_list(idx, "actions", raw_actions, 0)?;
    lower_to_program(&tree).map_err(|e| {
        vec![ValidationIssue::from_program_error(
            &e,
            Some(idx),
            "actions",
        )]
    })
}

/// Recursive validation of a `[RawAction]` slice. `path` is the breadcrumb-style label of the slice
/// within the watch — `"actions"` at the top, `"actions[0].then"` inside a then-branch, etc.
/// Returns `Ok(Vec<Action>)` iff every element validated; on any failure the per-element errors are
/// collected and the function returns `Err`.
///
/// Empty input is the *caller's* responsibility to reject (only the top-level `actions = []`
/// carries [`IssueKind::EmptyActions`]; nested empty arrays are rejected via
/// [`IssueKind::EmptyConditional`] against the enclosing conditional). This function is silent on
/// emptiness — it returns `Ok(Vec::new())` in that case so the caller can fold the empty branch
/// into the AST as `None` (no else) or apply the conditional-level check.
///
/// `depth` is the conditional nesting level of *this* slice: `0` at the [`validate_actions`] entry,
/// incremented by `1` each time [`validate_conditional`] recurses into a `then` / `else` body. When
/// `depth` exceeds [`MAX_CONDITIONAL_DEPTH`] the function short-circuits with
/// [`IssueKind::ConditionalNestedTooDeep`] *before* iterating — adversarial inputs cannot drive the
/// validator past the bound, and the lowering pass downstream can rely on the guarantee without a
/// mirror check.
fn validate_action_list(
    watch_idx: usize,
    path: &str,
    raw_actions: &[RawAction],
    depth: u8,
) -> Result<Vec<Action>, Vec<ValidationIssue>> {
    if depth > MAX_CONDITIONAL_DEPTH {
        return Err(vec![ValidationIssue::new(
            Some(watch_idx),
            "actions",
            IssueKind::ConditionalNestedTooDeep,
            format!(
                "{path}: conditional nesting exceeds the maximum depth of {MAX_CONDITIONAL_DEPTH}",
            ),
        )]);
    }
    let mut tree: Vec<Action> = Vec::with_capacity(raw_actions.len());
    let mut errors: Vec<ValidationIssue> = Vec::new();
    for (j, raw) in raw_actions.iter().enumerate() {
        let child_path = format!("{path}[{j}]");
        match validate_one_action(watch_idx, &child_path, raw, depth) {
            Ok(action) => tree.push(action),
            Err(mut es) => errors.append(&mut es),
        }
    }
    if errors.is_empty() {
        Ok(tree)
    } else {
        Err(errors)
    }
}

/// Validate a single action entry. The "exactly one variant set" rule is the single source of truth
/// across `exec`, the conditional triple (`when` + `then` + optional `else`), and (future) `pipe` —
/// it stays the same shape as new variants land.
///
/// `path` is the action's breadcrumb-style label (`"actions[0]"`, `"actions[0].then[1]"`, etc).
/// Error messages quote it so operators can locate the offending entry without re-counting nested
/// arrays. `depth` is the conditional nesting level of the enclosing slice — passed unchanged to
/// [`validate_conditional`], which bumps it by `1` before descending into `then` / `else`.
fn validate_one_action(
    watch_idx: usize,
    path: &str,
    raw: &RawAction,
    depth: u8,
) -> Result<Action, Vec<ValidationIssue>> {
    let is_exec = raw.exec.is_some();
    let is_pipe = raw.pipe.is_some();
    let is_conditional = raw.when.is_some() || raw.then.is_some() || raw.otherwise.is_some();
    let variants_set = usize::from(is_exec) + usize::from(is_pipe) + usize::from(is_conditional);
    if variants_set == 0 {
        return Err(vec![ValidationIssue::new(
            Some(watch_idx),
            "actions",
            IssueKind::ActionMissingVariant,
            format!("{path}: must specify a variant (`exec`, `pipe`, or `when`/`then`)"),
        )]);
    }
    if variants_set > 1 {
        return Err(vec![ValidationIssue::new(
            Some(watch_idx),
            "actions",
            IssueKind::ActionAmbiguousVariant,
            format!(
                "{path}: must specify exactly one variant \
                 (`exec`, `pipe`, and `when`/`then` are mutually exclusive)",
            ),
        )]);
    }
    // `timeout` at the action level applies only to `exec`. Predicates (`when`) carry their own
    // per-step `timeout` inside the nested `RawExec`; `pipe` stages each carry their own on the
    // nested `RawExec` of each stage.
    if raw.timeout.is_some() && !is_exec {
        return Err(vec![ValidationIssue::new(
            Some(watch_idx),
            "actions",
            IssueKind::TimeoutNotApplicable,
            format!(
                "{path}: top-level `timeout` requires `exec` \
                 (predicates set `when.timeout`; pipe stages each set their own `timeout`)",
            ),
        )]);
    }
    if let Some(argv) = raw.exec.as_deref() {
        return validate_exec_argv(watch_idx, path, "exec", argv, raw.timeout).map(Action::Exec);
    }
    if let Some(stages) = raw.pipe.as_deref() {
        return validate_pipe(watch_idx, path, stages);
    }
    if is_conditional {
        return validate_conditional(watch_idx, path, raw, depth);
    }
    unreachable!("variants_set ∈ {{1}} and exec/pipe/conditional are exhaustive in v1")
}

/// Validate the `pipe = [{ exec = [...], timeout = "..." }, ...]` variant of a [`RawAction`].
///
/// Structural rules:
/// - `pipe = []` ⇒ [`IssueKind::EmptyPipe`] (no stages to wire).
/// - `pipe = [solo]` ⇒ [`IssueKind::SingleStagePipe`] (degenerate; use top-level `exec` directly).
/// - Each stage's argv goes through [`validate_exec_argv`] under the path label
///   `"<path>.pipe[i].exec"` so per-stage errors are unambiguous; stage-local errors don't
///   short-circuit later stages.
///
/// Stages are stored as `Arc<[ExecAction]>` so lowering can `Arc::clone` them into
/// [`specter_core::program::MultiStage::new`], which reifies the validated `>= 2` guarantee into
/// [`specter_core::program::SpawnBody::Pipe`] without re-allocating (the newtype is zero-cost over
/// the shared `Arc`).
fn validate_pipe(
    watch_idx: usize,
    path: &str,
    stages: &[RawExec],
) -> Result<Action, Vec<ValidationIssue>> {
    if stages.is_empty() {
        return Err(vec![ValidationIssue::new(
            Some(watch_idx),
            "actions",
            IssueKind::EmptyPipe,
            format!("{path}.pipe must have at least two stages (got 0)"),
        )]);
    }
    if stages.len() < 2 {
        return Err(vec![ValidationIssue::new(
            Some(watch_idx),
            "actions",
            IssueKind::SingleStagePipe,
            format!(
                "{path}.pipe must have at least two stages \
                 (single stages should use top-level `exec` directly)",
            ),
        )]);
    }
    let mut validated: Vec<ExecAction> = Vec::with_capacity(stages.len());
    let mut errors: Vec<ValidationIssue> = Vec::new();
    for (idx, stage) in stages.iter().enumerate() {
        let stage_path = format!("{path}.pipe[{idx}]");
        match validate_raw_exec(watch_idx, &stage_path, stage) {
            Ok(exec) => validated.push(exec),
            Err(mut es) => errors.append(&mut es),
        }
    }
    if errors.is_empty() {
        Ok(Action::Pipe {
            stages: Arc::from(validated),
        })
    } else {
        Err(errors)
    }
}

/// Validate the `when` / `then` / `else` triple on a [`RawAction`].
///
/// Structural rules:
/// - Both `when` and `then` must be present (one without the other is
///   [`IssueKind::ConditionalIncomplete`]). `else` is optional.
/// - `then = []` AND (`else` absent OR `else = []`) is [`IssueKind::EmptyConditional`] — the
///   predicate would fire with no observable effect.
/// - `then = []` with non-empty `else` is permitted (operationally equivalent to a negated
///   predicate).
///
/// Per-branch errors (argv shape, nested action variants, etc.) are collected alongside the
/// structural errors so a single config-load surfaces every issue. `depth` is the nesting level of
/// the slice this conditional lives in; recursion into `then` / `else` advances it by `1` so
/// [`validate_action_list`]'s [`MAX_CONDITIONAL_DEPTH`] gate sees the descending count.
fn validate_conditional(
    watch_idx: usize,
    path: &str,
    raw: &RawAction,
    depth: u8,
) -> Result<Action, Vec<ValidationIssue>> {
    let when_raw = raw.when.as_ref();
    let then_raw = raw.then.as_deref();
    let otherwise_raw = raw.otherwise.as_deref();

    if when_raw.is_none() || then_raw.is_none() {
        return Err(vec![ValidationIssue::new(
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
        )]);
    }
    let when_raw = when_raw.expect("checked Some directly above");
    let then_raw = then_raw.expect("checked Some directly above");

    // Empty-conditional check fires *before* per-branch validation so operators see one structural
    // error rather than a flood of per-empty-branch issues (which is exactly nothing in that case).
    let else_empty = otherwise_raw.is_none_or(<[RawAction]>::is_empty);
    if then_raw.is_empty() && else_empty {
        return Err(vec![ValidationIssue::new(
            Some(watch_idx),
            "actions",
            IssueKind::EmptyConditional,
            format!(
                "{path}: conditional has empty `then` and no `else` body \
                 (predicate would have no observable effect)",
            ),
        )]);
    }

    let when_path = format!("{path}.when");
    let when_r = validate_raw_exec(watch_idx, &when_path, when_raw);

    let then_path = format!("{path}.then");
    let then_r = validate_action_list(watch_idx, &then_path, then_raw, depth + 1);

    // `Some(empty)` normalises to `None` so the AST shape matches the lowering precondition
    // (`Some(_)` ⇒ non-empty body). Lowering also handles empty defensively, but normalising here
    // keeps the AST the canonical form.
    let otherwise_r: Result<Option<Vec<Action>>, Vec<ValidationIssue>> = match otherwise_raw {
        Some(body) if !body.is_empty() => {
            let p = format!("{path}.else");
            validate_action_list(watch_idx, &p, body, depth + 1).map(Some)
        }
        Some(_) | None => Ok(None),
    };

    match (when_r, then_r, otherwise_r) {
        (Ok(when), Ok(then), Ok(otherwise)) => Ok(Action::Conditional {
            when,
            then: then.into_boxed_slice(),
            otherwise: otherwise.map(Vec::into_boxed_slice),
        }),
        (when_r, then_r, otherwise_r) => {
            let mut errors: Vec<ValidationIssue> = Vec::new();
            if let Err(es) = when_r {
                errors.extend(es);
            }
            if let Err(es) = then_r {
                errors.extend(es);
            }
            if let Err(es) = otherwise_r {
                errors.extend(es);
            }
            Err(errors)
        }
    }
}

/// Validate the nested-exec table used inside a conditional predicate (`when = { exec = [...],
/// timeout = "..." }`) and pipe stages. Delegates to [`validate_exec_argv`] under the path label
/// `"<path>.exec"` so argv-slot errors quote the surrounding action's location.
fn validate_raw_exec(
    watch_idx: usize,
    path: &str,
    raw: &RawExec,
) -> Result<ExecAction, Vec<ValidationIssue>> {
    validate_exec_argv(watch_idx, path, "exec", &raw.exec, raw.timeout)
}

/// Validate one `exec = [...]` argv plus its optional `timeout`. Each empty slot or parse failure
/// yields one [`ValidationIssue`]; the function returns `Err` on any failure so the partial argv
/// can't reach the engine.
///
/// `timeout` is the operator-supplied per-step deadline (humantime at the TOML layer; [`Duration`]
/// here). When `Some`, the validated duration is also checked for `> 0` — a zero-duration timeout
/// is rejected as a configuration error: the SIGTERM would fire before the child has any chance to
/// make progress, which is almost certainly a typo (`"0s"`, `"0ms"`). Operators wanting "no
/// timeout" omit the field entirely.
///
/// `path` is the surrounding action's breadcrumb (e.g., `"actions[0]"` or `"actions[0].then[1]"`);
/// `argv_field` is the field name of the argv slot within that action — `"exec"` for both top-level
/// exec and predicate-exec, `"pipe[i].exec"` for pipe stages. The error detail joins them with `.`
/// so the path is unambiguous.
fn validate_exec_argv(
    watch_idx: usize,
    path: &str,
    argv_field: &str,
    raw_argv: &[String],
    timeout: Option<Duration>,
) -> Result<ExecAction, Vec<ValidationIssue>> {
    if raw_argv.is_empty() {
        return Err(vec![ValidationIssue::new(
            Some(watch_idx),
            "actions",
            IssueKind::EmptyArgv,
            format!("{path}.{argv_field} must have at least one slot"),
        )]);
    }
    let mut argv: Vec<ArgTemplate> = Vec::with_capacity(raw_argv.len());
    let mut errors: Vec<ValidationIssue> = Vec::new();
    for (k, slot) in raw_argv.iter().enumerate() {
        if slot.is_empty() {
            errors.push(ValidationIssue::new(
                Some(watch_idx),
                "actions",
                IssueKind::EmptyArgv,
                format!("{path}.{argv_field}[{k}] is empty"),
            ));
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
    }
    if errors.is_empty() {
        Ok(ExecAction::new(argv, timeout))
    } else {
        Err(errors)
    }
}

/// Validate `settle` / `max_settle`. Returns `Ok((settle, max_settle))` when both pass;
/// `Err(Vec<ValidationIssue>)` accumulates both failures when they fire concurrently.
///
/// - `settle` defaults to [`DEFAULT_SETTLE`] when omitted; rejected with
///   [`IssueKind::SettleTooSmall`] when zero.
/// - `max_settle` defaults to [`DEFAULT_MAX_SETTLE`] (a flat 1h) when omitted; rejected with
///   [`IssueKind::MaxSettleTooSmall`] when below `4 × settle`. There is no upper bound — Instant
///   arithmetic at the engine layer handles values up to many years; humantime input format makes
///   magnitude typos obvious at the source.
fn validate_settle(
    idx: usize,
    raw_settle: Option<Duration>,
    raw_max_settle: Option<Duration>,
) -> Result<(Duration, Duration), Vec<ValidationIssue>> {
    let mut errors: Vec<ValidationIssue> = Vec::new();
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
    if errors.is_empty() {
        Ok((settle, max_settle))
    } else {
        Err(errors)
    }
}

/// Validate `scope`. Returns `Ok(EffectScope)` on success; `Err` with the parse failure issue.
/// Scope failures are single-issue by construction (one input string, one valid set), so the error
/// path is a single [`ValidationIssue`] rather than a `Vec`.
///
/// The caller forwards a copied [`EffectScope`] to the events parser even when validation fails —
/// the default ([`EffectScope::default`]) is used solely to keep `parse_events_field` from
/// cascading a phantom error against an unresolved scope.
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

/// Validate a static watch's `path`: lenient-canonicalisable, with one typed [`IssueKind`] per
/// [`PathError`] variant. Returns `Ok(PathBuf)` on success; `Err(ValidationIssue)` carrying the
/// operator-actionable category — single-issue by construction (one input, one possible failure
/// mode per call).
///
/// Routes [`PathError`] → [`IssueKind`] symmetrically: structural and I/O failures each land in
/// their own arm so operators can triage "permissions" from "typo in `..`" from "no such file"
/// without inspecting the error detail string. No `is_absolute()` prefilter is needed —
/// `canonicalize_lenient` enforces the rule itself and routes the failure through
/// `PathError::NotAbsolute` here.
fn validate_static_path(idx: usize, raw_path: &str) -> Result<PathBuf, ValidationIssue> {
    canonicalize_lenient(Path::new(raw_path)).map_err(|e| {
        let (kind, detail) = match &e {
            PathError::NotAbsolute => (
                IssueKind::NonAbsolute,
                format!("path `{raw_path}` must be absolute"),
            ),
            PathError::Empty => (IssueKind::EmptyPath, "path must not be empty".to_owned()),
            PathError::ContainsParentDir => (
                IssueKind::PathContainsParentDir,
                format!(
                    "path `{raw_path}` must not contain `..` components — \
                     provide a literal absolute path",
                ),
            ),
            PathError::Inaccessible { at, source } => {
                let at_display = at.display().to_string();
                let detail = if at_display == raw_path {
                    format!("path `{raw_path}` is inaccessible: {source}")
                } else {
                    format!("path `{raw_path}` is inaccessible at `{at_display}`: {source}")
                };
                (IssueKind::PathInaccessible, detail)
            }
            PathError::NonUtf8 { resolved } => (
                IssueKind::NonUtf8Path,
                format!(
                    "path `{raw_path}` resolves to a non-UTF-8 buffer `{}` — \
                     engine requires UTF-8",
                    resolved.display(),
                ),
            ),
        };
        ValidationIssue::new(Some(idx), "path", kind, detail)
    })
}

/// Validate a dynamic watch's `path` as a [`PatternSpec`], resolving its literal-prefix anchor
/// through the same `canonicalize_lenient` every static path gets. Two stages mirror the static
/// path's parse-then-resolve shape:
///
/// 1. [`PatternSpec::parse`] (pure) enforces the structural invariants (absolute, no `**`, no
///    `.`/`..`, no empty segments, no Windows prefix); a parse failure is
///    [`IssueKind::InvalidPattern`].
/// 2. `canonicalize_lenient` resolves the literal prefix (I/O), and [`PatternSpec::reanchor`]
///    splices the resolved path back onto the pattern. A resolution fault is **fatal** — exactly
///    like a static path that cannot canonicalise.
///
/// The result's anchor and identity-bearing `source` are both symlink-free, so a dynamic and a
/// static watch over one tree anchor the same Tree branch and the kernel's unconditional
/// `O_NOFOLLOW` watch opens (anchor / watch-root parent / descent prefixes) land on real
/// directories rather than ELOOP-ing on a symlinked prefix component.
fn validate_dynamic_pattern(idx: usize, raw_path: &str) -> Result<PatternSpec, ValidationIssue> {
    let pattern = PatternSpec::parse(raw_path).map_err(|e| {
        ValidationIssue::new(
            Some(idx),
            "path",
            IssueKind::InvalidPattern,
            format!("`{raw_path}`: {e}"),
        )
    })?;
    let prefix = pattern.literal_prefix_path();
    let canonical = canonicalize_lenient(&prefix)
        .map_err(|e| dynamic_prefix_issue(idx, raw_path, &prefix, &e))?;
    Ok(pattern.reanchor(&canonical))
}

/// Map a literal-prefix `canonicalize_lenient` fault to a fatal [`ValidationIssue`].
///
/// Post-parse only `Inaccessible` and `NonUtf8` are reachable — [`PatternSpec::parse`] already
/// guaranteed the prefix is absolute, non-empty, and `..`-free — so those two carry pattern-aware
/// detail and reuse the static path's kinds ([`IssueKind::PathInaccessible`] /
/// [`IssueKind::NonUtf8Path`]): a dynamic prefix that cannot be canonicalised fails to load,
/// symmetrically with a static path that cannot. The structurally-unreachable trio is debug-loud
/// and collapses to `PathInaccessible` carrying the rendered fault, rather than inventing a kind
/// for a state the parser already excluded.
fn dynamic_prefix_issue(
    idx: usize,
    raw_path: &str,
    prefix: &Path,
    err: &PathError,
) -> ValidationIssue {
    let shown = prefix.display();
    let (kind, detail) = match err {
        PathError::Inaccessible { at, source } => {
            let detail = if at.as_path() == prefix {
                format!(
                    "pattern `{raw_path}` cannot anchor: its literal prefix `{shown}` is \
                     inaccessible: {source}"
                )
            } else {
                format!(
                    "pattern `{raw_path}` cannot anchor: its literal prefix `{shown}` is \
                     inaccessible at `{}`: {source}",
                    at.display(),
                )
            };
            (IssueKind::PathInaccessible, detail)
        }
        PathError::NonUtf8 { resolved } => (
            IssueKind::NonUtf8Path,
            format!(
                "pattern `{raw_path}` cannot anchor: its literal prefix `{shown}` resolves to a \
                 non-UTF-8 buffer `{}` — engine requires UTF-8",
                resolved.display(),
            ),
        ),
        // Unreachable after `PatternSpec::parse` (which rejected non-absolute / empty / `..`
        // prefixes); surface the real fault rather than fabricate a kind, loud in debug.
        PathError::NotAbsolute | PathError::Empty | PathError::ContainsParentDir => {
            debug_assert!(false, "canonicalize fault unreachable after parse: {err:?}");
            (
                IssueKind::PathInaccessible,
                format!(
                    "pattern `{raw_path}` cannot anchor: literal prefix `{shown}` is invalid: {err}"
                ),
            )
        }
    };
    ValidationIssue::new(Some(idx), "path", kind, detail)
}

/// Validate the `[[watch]]` block's scan-config knobs (`recursive`, `hidden`, `max_depth`,
/// `pattern`, `exclude`) — orthogonal to the path-vs-pattern dispatch. Returns `Ok(ScanConfig)`
/// when every field validates; `Err(Vec<ValidationIssue>)` accumulates per-field failures
/// (max_depth, pattern, and each exclude glob can fire independently).
fn validate_scan(idx: usize, raw: &RawWatch) -> Result<ScanConfig, Vec<ValidationIssue>> {
    let mut errors: Vec<ValidationIssue> = Vec::new();
    if raw.max_depth == Some(0) {
        errors.push(ValidationIssue::new(
            Some(idx),
            "max_depth",
            IssueKind::MaxDepthZero,
            "max_depth must be ≥ 1 or omitted (None = unbounded)".to_owned(),
        ));
    }

    let mut sb = ScanConfig::builder()
        .recursive(raw.recursive)
        .hidden(raw.hidden)
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
            Err(core::ConfigError::UnreachableGlob { reason, .. }) => {
                errors.push(ValidationIssue::new(
                    Some(idx),
                    "pattern",
                    IssueKind::UnreachableGlob,
                    format!("`{p}`: {reason}"),
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
                Err(core::ConfigError::UnreachableGlob { reason, .. }) => {
                    errors.push(ValidationIssue::new(
                        Some(idx),
                        "exclude",
                        IssueKind::UnreachableGlob,
                        format!("`{ex}`: {reason}"),
                    ));
                }
            }
        }
        sb = sb.excludes(compiled);
    }

    if errors.is_empty() {
        Ok(sb.build())
    } else {
        Err(errors)
    }
}

/// Per-attachment fields shared between static and dynamic `[[watch]]` blocks. Everything in this
/// struct is independent of the path-vs-pattern dispatch: the same operator-supplied knobs (`name`,
/// `actions`, `scope`, `settle`, etc.) carry the same meaning for both kinds of watch and run through
/// the same validators. Materialised once by [`validate_watch_attachment`]; the two thin wrappers
/// ([`validate_static_watch`] / [`validate_dynamic_watch`]) consume it via the `into_*_spec`
/// projections after resolving their kind-specific anchor — the dynamic projection re-homes the user
/// knobs onto the [`TemplateSpec`] and pins the Sub's own identity to the discovery constants.
struct WatchAttachmentFields {
    name: CompactString,
    program: Arc<ActionProgram>,
    scope: EffectScope,
    settle: Duration,
    max_settle: Duration,
    scan: ScanConfig,
    events: ClassSet,
    log_output: bool,
    enabled: bool,
}

impl WatchAttachmentFields {
    /// Move the validated common fields plus a resolved static anchor into a [`SubSpec`]. Cannot
    /// fail — both inputs are already validated by construction.
    fn into_sub_spec(self, path: PathBuf) -> SubSpec {
        SubSpec {
            name: self.name,
            path,
            program: self.program,
            scope: self.scope,
            settle: self.settle,
            max_settle: self.max_settle,
            scan: self.scan,
            events: self.events,
            log_output: self.log_output,
            enabled: self.enabled,
            template: None,
        }
    }

    /// Move the validated common fields plus a parsed dynamic pattern into a discovery [`SubSpec`].
    /// Mirror of [`Self::into_sub_spec`] for the dynamic dispatch.
    ///
    /// The discovery Sub's *own* identity is pinned to constants — `MatchChain` scan, `STRUCTURE`
    /// events (membership changes are the chain proof object's only witness classes, and dir-only
    /// chain FDs follow from the mask), [`DISCOVERY_SETTLE`] / [`DISCOVERY_MAX_SETTLE`] — so one
    /// pattern always maps to one discovery Profile. Every user knob moves into the
    /// [`TemplateSpec`]: the `[[watch]]` surface keeps its meaning (`settle` debounces the
    /// *reaction*, i.e. the minted Subs). `program` / `scope` / `log_output` stay on the spec's
    /// flat fields — `to_attach_request` seals them into the template's `SpawnSpec`, the minted
    /// Subs' reaction (the discovery Sub itself spawns nothing).
    ///
    /// The anchor is the pattern's **canonicalised** literal prefix — symlink-resolved at lowering
    /// by [`validate_dynamic_pattern`], like every other watched path. Matching is positional
    /// ([`PatternSpec::matches_at`] reads only the glob tail, never the prefix), so
    /// canonicalisation sites the anchor where the kernel's `O_NOFOLLOW` watch opens succeed
    /// without changing which termini match; the identity hash folds the canonical `source`, so
    /// anchor and identity stay consistent. `pattern` arrives already re-anchored —
    /// `pattern.literal_prefix_path()` is the canonical path and the sole source of the anchor.
    fn into_discovery_spec(self, pattern: PatternSpec) -> SubSpec {
        SubSpec {
            name: self.name,
            path: pattern.literal_prefix_path(),
            program: self.program,
            scope: self.scope,
            settle: DISCOVERY_SETTLE,
            max_settle: DISCOVERY_MAX_SETTLE,
            scan: ScanConfig::MatchChain(Arc::new(pattern)),
            events: ClassSet::STRUCTURE,
            log_output: self.log_output,
            enabled: self.enabled,
            template: Some(TemplateSpec {
                scan: self.scan,
                events: self.events,
                settle: self.settle,
                max_settle: self.max_settle,
            }),
        }
    }
}

/// Validate the [`RawWatch`] fields that don't depend on the static/dynamic dispatch. Returns
/// `Ok(WatchAttachmentFields)` when every sub-validator succeeds, `Err(Vec<ValidationIssue>)`
/// accumulating every per-field failure.
///
/// **Scope → events ordering** is load-bearing: [`parse_events_field`] reads scope to pick the
/// scope-conditional default ([`ClassSet`]), so `validate_scope` must run first. On a scope failure
/// the events parser falls back to the default scope ([`EffectScope::default`]) to keep a phantom
/// events error from cascading — the scope error already surfaces from `scope_r`.
///
/// Single source of validation for the common fields: the two thin wrappers
/// ([`validate_static_watch`] / [`validate_dynamic_watch`]) call this and then prepend their
/// kind-specific anchor (path / pattern). The only differences across the two dispatch paths (anchor
/// resolution and output type) live in the wrappers; this helper is path-agnostic by construction.
fn validate_watch_attachment(
    idx: usize,
    raw: &RawWatch,
) -> Result<WatchAttachmentFields, Vec<ValidationIssue>> {
    let name_r = validate_name(idx, &raw.name);
    let program_r = validate_actions(idx, &raw.actions);
    let scope_r = validate_scope(idx, raw.scope.as_deref());
    let settle_r = validate_settle(idx, raw.settle, raw.max_settle);
    let scan_r = validate_scan(idx, raw);
    let events_r = parse_events_field(
        raw.events.as_deref(),
        scope_r.as_ref().copied().unwrap_or_default(),
        idx,
    );

    match (name_r, program_r, scope_r, settle_r, scan_r, events_r) {
        (Ok(()), Ok(program), Ok(scope), Ok((settle, max_settle)), Ok(scan), Ok(events)) => {
            Ok(WatchAttachmentFields {
                name: CompactString::new(&raw.name),
                program,
                scope,
                settle,
                max_settle,
                scan,
                events,
                log_output: raw.log_output,
                enabled: raw.enabled,
            })
        }
        (name_r, program_r, scope_r, settle_r, scan_r, events_r) => {
            let mut errors: Vec<ValidationIssue> = Vec::new();
            if let Err(e) = name_r {
                errors.push(e);
            }
            if let Err(es) = program_r {
                errors.extend(es);
            }
            if let Err(e) = scope_r {
                errors.push(e);
            }
            if let Err(es) = settle_r {
                errors.extend(es);
            }
            if let Err(es) = scan_r {
                errors.extend(es);
            }
            if let Err(es) = events_r {
                errors.extend(es);
            }
            Err(errors)
        }
    }
}

/// Validator for `[[watch]]` blocks whose `path` carries no glob discriminator characters (`*?[{`)
/// — pure-literal anchors. The dispatcher in [`validate`] gates on [`PatternSpec::is_dynamic`]
/// before invoking this, so a glob-bearing path cannot reach here.
///
/// Thin composition: resolve the static anchor and the common attachment fields independently, then
/// either project a [`SubSpec`] from both Oks or accumulate every failure across both halves.
/// Cross-half error fan-in is preserved — a path failure does not short-circuit the attachment-side
/// issues, and vice versa.
fn validate_static_watch(idx: usize, raw: &RawWatch) -> Result<SubSpec, Vec<ValidationIssue>> {
    let path_r = validate_static_path(idx, &raw.path);
    let fields_r = validate_watch_attachment(idx, raw);
    match (path_r, fields_r) {
        (Ok(path), Ok(fields)) => Ok(fields.into_sub_spec(path)),
        (path_r, fields_r) => {
            let mut errors: Vec<ValidationIssue> = Vec::new();
            if let Err(e) = path_r {
                errors.push(e);
            }
            if let Err(es) = fields_r {
                errors.extend(es);
            }
            Err(errors)
        }
    }
}

/// Validator for `[[watch]]` blocks whose `path` carries at least one glob discriminator character
/// (`*?[{`). Caller gates on [`PatternSpec::is_dynamic`]; the parser itself enforces the pattern's
/// structural invariants (absolute, no `**`, no `.`/`..`, no empty segments, no Windows prefix).
///
/// Mirror of [`validate_static_watch`] for the dynamic dispatch: same composition, same error
/// fan-in, different anchor resolution ([`PatternSpec`] vs [`PathBuf`]) and projection
/// ([`WatchAttachmentFields::into_discovery_spec`] vs `into_sub_spec`).
fn validate_dynamic_watch(idx: usize, raw: &RawWatch) -> Result<SubSpec, Vec<ValidationIssue>> {
    let pattern_r = validate_dynamic_pattern(idx, &raw.path);
    let fields_r = validate_watch_attachment(idx, raw);
    match (pattern_r, fields_r) {
        (Ok(pattern), Ok(fields)) => Ok(fields.into_discovery_spec(pattern)),
        (pattern_r, fields_r) => {
            let mut errors: Vec<ValidationIssue> = Vec::new();
            if let Err(e) = pattern_r {
                errors.push(e);
            }
            if let Err(es) = fields_r {
                errors.extend(es);
            }
            Err(errors)
        }
    }
}

/// Parse the optional TOML `events = [...]` array into a [`ClassSet`].
///
/// - Field omitted → `Ok` with the scope-conditional default ([`ClassSet::DEFAULT_SUBTREE_ROOT`]
///   for `subtree-root`, [`ClassSet::DEFAULT_PER_FILE`] for `per-stable-file`).
/// - Empty array → `Err` with [`IssueKind::EventsEmpty`]. "I want zero classes" can only be a typo;
///   toggling a watch off is removal-by-name.
/// - Unknown value → `Err` with [`IssueKind::InvalidEnum`] (one per).
/// - Repeated value → `Err` with [`IssueKind::DuplicateEventClass`] (one per extra occurrence).
///
/// Per-value errors accumulate so a single load surfaces every issue.
fn parse_events_field(
    raw: Option<&[String]>,
    scope: EffectScope,
    idx: usize,
) -> Result<ClassSet, Vec<ValidationIssue>> {
    let Some(values) = raw else {
        return Ok(match scope {
            EffectScope::SubtreeRoot => ClassSet::DEFAULT_SUBTREE_ROOT,
            EffectScope::PerStableFile => ClassSet::DEFAULT_PER_FILE,
        });
    };

    if values.is_empty() {
        return Err(vec![ValidationIssue::new(
            Some(idx),
            "events",
            IssueKind::EventsEmpty,
            "events array must not be empty (omit the field to take the \
             scope-conditional default)"
                .to_owned(),
        )]);
    }

    let mut errors: Vec<ValidationIssue> = Vec::new();
    let mut out = ClassSet::EMPTY;
    for v in values {
        let bit = match v.as_str() {
            "structure" => ClassSet::STRUCTURE,
            "content" => ClassSet::CONTENT,
            "metadata" => ClassSet::METADATA,
            other => {
                // Surface the whitespace-is-significant case explicitly: serde-toml preserves
                // quoted whitespace, so a raw entry like `events = [" structure "]` reaches us with
                // the padding intact and silently fails the literal match. The hint catches the
                // typo at first glance instead of forcing the operator to inspect quoting rules.
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
    if errors.is_empty() {
        Ok(out)
    } else {
        Err(errors)
    }
}

#[cfg(test)]
mod tests {
    use super::{Config, LogConfig, LogDestination, LogLevel, SubAttachAnchor, SubSpec};
    use crate::error::{ConfigError, IssueKind};
    use specter_core::program::SpawnBody;
    use specter_core::{
        ArgPart, ClassSet, EffectScope, Placeholder, ProfileIdentity, ReactionSpec, ScanConfig,
    };
    use std::path::Path;
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
        // A top-level `log_level` field is not part of the schema. RawConfig has
        // `deny_unknown_fields`, so the parse fails fast rather than silently dropping the value.
        let err = Config::from_str("log_level = \"debug\"").unwrap_err();
        assert!(matches!(err, ConfigError::Parse { .. }));
    }

    #[test]
    fn log_destination_file_requires_path() {
        let err = Config::from_str("[log]\ndestination = \"file\"").unwrap_err();
        let errors = validation_errors(err);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, IssueKind::EmptyLogPath);
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
        // Stderr path is dropped (set to None) — an operator may set `path = ...` alongside a
        // stderr destination; we drop it rather than fail validation, because the field carries no
        // meaning when destination = stderr.
        let cfg =
            Config::from_str("[log]\ndestination = \"stderr\"\npath = \"/var/log/ignored.log\"")
                .unwrap();
        assert_eq!(cfg.log.destination, LogDestination::Stderr);
        assert!(
            cfg.log.path.is_none(),
            "path is dropped for stderr destination"
        );
    }

    /// A relative `--log-path` overrides into a `LogDestination::File` fixture; `merge_cli` returns
    /// the bare typed [`crate::error::ValidationIssue`] (no `Vec`, no [`ConfigError::Validate`]
    /// envelope) so the bin caller can format the CLI-merge context in its own `eprintln!` without
    /// the `<inline>:` prefix that the config-layer error type would impose. Pinning the field / kind
    /// here documents the post-Bundle-B contract: CLI-merge failures bypass [`ConfigError`] entirely.
    #[test]
    fn merge_cli_relative_log_path_yields_non_absolute_issue() {
        let cfg = LogConfig {
            level: LogLevel::Info,
            destination: LogDestination::File,
            path: None,
        };
        let issue = cfg
            .merge_cli(None, None, Some(Path::new("relative-log.txt")))
            .unwrap_err();
        assert!(issue.watch_index.is_none());
        assert_eq!(issue.field, "log.path");
        assert_eq!(issue.kind, IssueKind::NonAbsolute);
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
        let ScanConfig::Subtree {
            recursive,
            hidden,
            exclude,
            pattern,
            max_depth,
        } = &w.scan
        else {
            panic!("static watch lowers to Subtree, got {:?}", w.scan);
        };
        assert!(*recursive);
        assert!(!*hidden);
        assert!(exclude.is_empty());
        assert!(pattern.is_none());
        assert_eq!(*max_depth, None);
        let SpawnBody::Exec(exec) = w.program.ops()[0].body() else {
            panic!("expected SpawnBody::Exec");
        };
        assert_eq!(exec.argv().len(), 1);
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
        let ReactionSpec::Spawn { spec, .. } = &req.params.reaction else {
            panic!("static watch lowers to a Spawn reaction");
        };
        assert!(
            spec.log_output(),
            "SubSpec.log_output reaches the request's SpawnSpec via to_attach_request",
        );
    }

    #[test]
    fn enabled_false_round_trips() {
        // Disabled entries still land in `Config.watches` — the filter is applied at the runtime
        // view (`active_watches`), not at parse time.
        let cfg = Config::from_str(&minimal_toml("enabled = false\n")).unwrap();
        assert!(!cfg.watches[0].enabled);
        assert_eq!(cfg.watches.len(), 1, "disabled entry kept in raw Vec");
    }

    #[test]
    fn enabled_false_round_trips_for_dynamic_watch() {
        // Mirror the static-side round-trip on the dynamic dispatch path (path containing `*?[{`
        // routes to the discovery validator).
        let toml = "[[watch]]\nname = \"logs\"\npath = \"/srv/log/*\"\n\
                    actions = [{ exec = [\"echo\"] }]\nenabled = false\n";
        let cfg = Config::from_str(toml).unwrap();
        assert_eq!(cfg.watches.len(), 1);
        assert!(!cfg.watches[0].enabled);
        assert!(cfg.watches[0].template.is_some());
    }

    /// `disabled_names` is the structural complement of `active_watches` — each name appears in
    /// exactly one of the two views, static and discovery entries alike. Asserting both in a single
    /// fixture pins the partition so a future refactor of either filter cannot drift the two
    /// summaries apart.
    #[test]
    fn disabled_names_partitions_complement_of_active_in_source_order() {
        let toml = format!(
            "[[watch]]\nname = \"a\"\npath = \"{ROOT}\"\nactions = [{{ exec = [\"echo\"] }}]\n\
             [[watch]]\nname = \"b\"\npath = \"{ROOT}\"\nactions = [{{ exec = [\"echo\"] }}]\nenabled = false\n\
             [[watch]]\nname = \"c\"\npath = \"{ROOT}\"\nactions = [{{ exec = [\"echo\"] }}]\n\
             [[watch]]\nname = \"d\"\npath = \"{ROOT}\"\nactions = [{{ exec = [\"echo\"] }}]\nenabled = false\n\
             [[watch]]\nname = \"e\"\npath = \"/srv/*\"\nactions = [{{ exec = [\"echo\"] }}]\nenabled = false\n\
             [[watch]]\nname = \"f\"\npath = \"/srv/*\"\nactions = [{{ exec = [\"echo\"] }}]\n",
        );
        let cfg = Config::from_str(&toml).unwrap();
        assert_eq!(cfg.disabled_names(), vec!["b", "d", "e"]);
        let active: Vec<&str> = cfg.active_watches().map(|s| s.name.as_str()).collect();
        assert_eq!(active, vec!["a", "c", "f"]);
    }

    /// `find_active_watch` returns the SubSpec for an enabled name, `None` for a `enabled = false`
    /// entry, and `None` for an absent name. The disabled / absent collapse is by design: the
    /// operator IPC layer treats both as "not active right now".
    #[test]
    fn find_active_watch_resolves_enabled_only() {
        let toml = format!(
            "[[watch]]\nname = \"on\"\npath = \"{ROOT}\"\nactions = [{{ exec = [\"echo\"] }}]\n\
             [[watch]]\nname = \"off\"\npath = \"{ROOT}\"\nactions = [{{ exec = [\"echo\"] }}]\nenabled = false\n",
        );
        let cfg = Config::from_str(&toml).unwrap();
        assert_eq!(
            cfg.find_active_watch("on").map(|s| s.name.as_str()),
            Some("on"),
        );
        assert!(
            cfg.find_active_watch("off").is_none(),
            "disabled entry hidden"
        );
        assert!(
            cfg.find_active_watch("ghost").is_none(),
            "absent name hidden"
        );
    }

    #[test]
    fn empty_name_rejected() {
        let toml = format!(
            "[[watch]]\nname = \"\"\npath = \"{ROOT}\"\nactions = [{{ exec = [\"echo\"] }}]"
        );
        assert_only_kind(&toml, IssueKind::EmptyName);
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
        // *Inside the `${specter.…}` namespace*, lowercase non-catalog names remain typo errors;
        // the catalog is exclusively lowercase, so a lowercase miss inside the namespace is almost
        // always a typo. Bare `$paht` (outside the namespace) is literal pass-through under the new
        // grammar — exercised by `template::tests::bare_dollar_name_is_literal`.
        let toml = format!(
            "[[watch]]\nname = \"a\"\npath = \"{ROOT}\"\nactions = [{{ exec = [\"fmt\", \"${{specter.paht}}\"] }}]"
        );
        assert_only_kind(&toml, IssueKind::UnknownPlaceholder);
    }

    #[test]
    fn uppercase_env_var_passes_through_for_shell_expansion() {
        // Env vars (`$SPECTER_PATH`) and conventional shell vars (`$HOME`, `$USER`) must reach the
        // spawned shell unchanged.
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

    /// `GlobPattern::compile` rejects four shapes the walker would silently treat as no-ops; this
    /// test pins that they surface as distinct `IssueKind::UnreachableGlob` errors per offending
    /// entry, on both the `pattern` and `exclude` fields. The gitignore-style `target/` typo (the
    /// worst footgun — silently excludes nothing) is the headline case.
    #[test]
    fn unreachable_globs_rejected_per_offending_entry() {
        let toml = minimal_toml(
            "pattern = \"/foo\"\n\
             exclude = [\"target/\", \".\", \"\", \"good/**\", \"*.rs\", \"**\"]\n",
        );
        let err = Config::from_str(&toml).unwrap_err();
        let errors = validation_errors(err);
        // pattern + 3 exclude (target/, ., empty) = 4 issues.
        assert_eq!(errors.len(), 4, "got {errors:#?}");
        assert!(
            errors.iter().all(|e| e.kind == IssueKind::UnreachableGlob),
            "every issue should be UnreachableGlob: {errors:#?}",
        );
        // Field assignment is preserved (pattern issue lists pattern, exclude issues list exclude)
        // — operator triage needs the distinction.
        let pattern_issue = errors
            .iter()
            .find(|e| e.field == "pattern")
            .expect("pattern issue present");
        assert!(pattern_issue.detail.contains("/foo"));
        assert_eq!(
            errors.iter().filter(|e| e.field == "exclude").count(),
            3,
            "exclude must carry one issue per offending entry",
        );
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
        // Boundary: exactly 4 × settle passes. Catches off-by-one in the floor comparison.
        let toml = minimal_toml("settle = \"100ms\"\nmax_settle = \"400ms\"\n");
        let cfg = Config::from_str(&toml).unwrap();
        assert_eq!(cfg.watches[0].max_settle, Duration::from_millis(400));
    }

    #[test]
    fn default_max_settle_is_one_hour_independent_of_settle() {
        // `max_settle` defaults to a flat 1h regardless of `settle` — it is not derived from `settle`
        // by any factor. A few representative `settle` values all observe the same default.
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
        // humantime accepts compound forms (`"1m 30s"`); pin the semantics so a parser swap doesn't
        // regress silently.
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
        let ScanConfig::Subtree { pattern, .. } = &cfg.watches[0].scan else {
            panic!(
                "static watch lowers to Subtree, got {:?}",
                cfg.watches[0].scan
            );
        };
        assert!(pattern.is_some());
    }

    #[test]
    fn excludes_sorted_by_source_after_validate() {
        let toml = minimal_toml("exclude = [\"z/**\", \"a/**\", \"m/**\"]\n");
        let cfg = Config::from_str(&toml).unwrap();
        let ScanConfig::Subtree { exclude, .. } = &cfg.watches[0].scan else {
            panic!(
                "static watch lowers to Subtree, got {:?}",
                cfg.watches[0].scan
            );
        };
        let sources: Vec<&str> = exclude
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
        let SpawnBody::Exec(exec) = cfg.watches[0].program.ops()[0].body() else {
            panic!("expected SpawnBody::Exec");
        };
        let argv = exec.argv();
        assert_eq!(argv.len(), 3);
        assert_eq!(argv[0].parts()[0], ArgPart::literal("fmt"));
        assert_eq!(argv[1].parts()[0], ArgPart::literal("--input="));
        assert_eq!(argv[1].parts()[1], ArgPart::Placeholder(Placeholder::Path));
        assert_eq!(
            argv[2].parts()[0],
            ArgPart::Placeholder(Placeholder::Created)
        );
    }

    #[test]
    fn multiple_errors_in_one_watch_collected() {
        let toml = "[[watch]]\nname = \"\"\npath = \"src\"\nactions = [{ exec = [] }]\nsettle = \"0ms\"\nmax_depth = 0";
        let err = Config::from_str(toml).unwrap_err();
        let errors = validation_errors(err);
        let kinds: Vec<IssueKind> = errors.iter().map(|e| e.kind).collect();
        assert!(kinds.contains(&IssueKind::EmptyName));
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
    fn to_attach_request_uses_path_anchor_with_canonicalized_path() {
        let cfg = Config::from_str(&minimal_toml("")).unwrap();
        let req = cfg.watches[0].to_attach_request();
        assert_eq!(req.params.name, "build");
        assert!(matches!(req.anchor, SubAttachAnchor::Path(_)));
        let SubAttachAnchor::Path(p) = &req.anchor else {
            panic!("expected Path anchor")
        };
        assert_eq!(
            p, &cfg.watches[0].path,
            "request carries the same path stored in SubSpec"
        );
        assert_eq!(
            req.identity.events(),
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
        // Distinct from "field omitted" (which takes the scope default); an explicit empty array is
        // always a typo and earns its own IssueKind.
        let toml = minimal_toml("events = []\n");
        assert_only_kind(&toml, IssueKind::EventsEmpty);
    }

    #[test]
    fn events_unknown_value_does_not_short_circuit_remaining_values() {
        // Unknown values report individually — they don't poison the rest of the array. The watch
        // still fails validation overall, but each issue is collected.
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
        // When scope fails, events falls back to the SubtreeRoot default so we don't double-report
        // a phantom events failure caused by the scope failure.
        let toml = minimal_toml("scope = \"weekly\"\n");
        let err = Config::from_str(&toml).unwrap_err();
        let errors = validation_errors(err);
        assert_eq!(errors.len(), 1, "got {errors:?}");
        assert_eq!(errors[0].kind, IssueKind::InvalidEnum);
        assert_eq!(errors[0].field, "scope");
    }

    #[test]
    fn events_field_value_is_case_sensitive() {
        // TOML enum values are kebab-case throughout — uppercase or mixed case is rejected,
        // matching the existing `scope` parser.
        for bad in ["Structure", "STRUCTURE", "Content", "Meta-Data"] {
            let toml = minimal_toml(&format!("events = [\"{bad}\"]\n"));
            assert_only_kind(&toml, IssueKind::InvalidEnum);
        }
    }

    #[test]
    fn events_field_whitespace_emits_did_you_mean_hint() {
        // serde-toml preserves whitespace inside quoted strings, so `events = [" structure "]`
        // reaches the parser with padding intact. The emitted message must surface the trim hint so
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
        // Non-whitespace typos (`strucutre`) get the standard error, not the whitespace-specific
        // hint — keeps the message tight for the common typo case.
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
        let ScanConfig::Subtree { recursive, .. } = &w.scan else {
            panic!("static watch lowers to Subtree, got {:?}", w.scan);
        };
        assert!(*recursive);
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

    /// The glob-in-canonical-segment wrinkle, end to end: a real directory literally named
    /// `weird[x]` reached through a symlink lowers its anchor onto `.../weird[x]` with the
    /// metacharacter segment spliced as a `Literal` — never re-parsed into a `Glob` (which would
    /// shrink `literal_prefix_len` and mis-site the anchor). Pins that the full canonicalise →
    /// `reanchor` pipeline never feeds `source` back through `parse`.
    #[cfg(unix)]
    #[test]
    fn dynamic_prefix_canonical_glob_metachar_segment_stays_literal() {
        let td = tempfile::tempdir().unwrap();
        let canon = td.path().canonicalize().unwrap();
        let weird = canon.join("weird[x]");
        std::fs::create_dir(&weird).unwrap();
        let link = canon.join("link");
        std::os::unix::fs::symlink(&weird, &link).unwrap();

        let toml = format!(
            "[[watch]]\nname = \"d\"\npath = \"{}/*.log\"\nactions = [{{ exec = [\"echo\"] }}]",
            link.display(),
        );
        let cfg = Config::from_str(&toml).unwrap();
        let w = &cfg.watches[0];

        // Anchor resolved onto the real `weird[x]` directory — the `[x]` did not split the prefix.
        assert_eq!(w.path, weird);
        let ScanConfig::MatchChain(spec) = &w.scan else {
            panic!("dynamic watch lowers to MatchChain, got {:?}", w.scan);
        };
        // The `weird[x]` segment is part of the literal-prefix anchor; only `*.log` is below it.
        assert_eq!(spec.literal_prefix_path(), weird);
        assert_eq!(spec.terminus_depth(), 1);
        assert!(spec.matches_at(1, "access.log"));
        assert!(!spec.matches_at(1, "access.txt"));
    }

    // ---- @-in-name rejection ----

    /// `@` is reserved for the synthesized `<template_name>@<matched_path>` shape of minted Subs. A
    /// static [[watch]] with `@` in its name would collide with that scheme on a discovery template
    /// sharing the substring; reject at config-load.
    #[test]
    fn at_sign_in_static_name_rejected() {
        let toml = format!(
            "[[watch]]\nname = \"foo@bar\"\npath = \"{ROOT}\"\nactions = [{{ exec = [\"echo\"] }}]"
        );
        assert_only_kind(&toml, IssueKind::InvalidName);
    }

    /// Same rule for dynamic [[watch]] entries — operators get a consistent name-grammar regardless
    /// of which validator their path routes to.
    #[test]
    fn at_sign_in_dynamic_name_rejected() {
        let toml = "[[watch]]\nname = \"foo@bar\"\npath = \"/srv/log/*\"\nactions = [{ exec = [\"echo\"] }]";
        assert_only_kind(toml, IssueKind::InvalidName);
    }

    /// Empty static name surfaces as `EmptyName` (not `InvalidName`) — the helper short-circuits
    /// empty before checking `@`.
    #[test]
    fn empty_static_name_emits_empty_name_kind_not_invalid_name() {
        let toml = format!(
            "[[watch]]\nname = \"\"\npath = \"{ROOT}\"\nactions = [{{ exec = [\"echo\"] }}]"
        );
        assert_only_kind(&toml, IssueKind::EmptyName);
    }

    /// Empty dynamic name surfaces as `EmptyName` for the same reason.
    #[test]
    fn empty_dynamic_name_emits_empty_name_kind_not_invalid_name() {
        let toml =
            "[[watch]]\nname = \"\"\npath = \"/srv/log/*\"\nactions = [{ exec = [\"echo\"] }]";
        assert_only_kind(toml, IssueKind::EmptyName);
    }

    // ---- Auto-detect dispatch ----

    /// Pure-literal absolute path → static dispatch → template-free [`SubSpec`].
    #[test]
    fn pure_literal_path_dispatches_to_static() {
        let toml = "[[watch]]\nname = \"static\"\npath = \"/var/log/myapp\"\nactions = [{ exec = [\"echo\"] }]";
        let cfg = Config::from_str(toml).unwrap();
        assert_eq!(cfg.watches.len(), 1);
        assert_eq!(cfg.watches[0].name, "static");
        assert!(cfg.watches[0].template.is_none());
    }

    /// Path with `*` discriminator → dynamic dispatch → template-bearing discovery [`SubSpec`]
    /// whose scan carries the pattern.
    #[test]
    fn glob_star_path_dispatches_to_dynamic() {
        let toml =
            "[[watch]]\nname = \"dyn\"\npath = \"/srv/log/*\"\nactions = [{ exec = [\"echo\"] }]";
        let cfg = Config::from_str(toml).unwrap();
        assert_eq!(cfg.watches.len(), 1);
        let w = &cfg.watches[0];
        assert_eq!(w.name, "dyn");
        assert!(w.template.is_some());
        let ScanConfig::MatchChain(spec) = &w.scan else {
            panic!("dynamic watch lowers to MatchChain, got {:?}", w.scan);
        };
        assert_eq!(spec.source(), "/srv/log/*");
    }

    /// Path with `?` → dynamic.
    #[test]
    fn question_mark_path_dispatches_to_dynamic() {
        let toml =
            "[[watch]]\nname = \"dyn\"\npath = \"/srv/?/data\"\nactions = [{ exec = [\"echo\"] }]";
        let cfg = Config::from_str(toml).unwrap();
        assert!(cfg.watches[0].template.is_some());
    }

    /// Path with `[…]` → dynamic.
    #[test]
    fn bracket_path_dispatches_to_dynamic() {
        let toml = "[[watch]]\nname = \"dyn\"\npath = \"/srv/[a-z]/data\"\nactions = [{ exec = [\"echo\"] }]";
        let cfg = Config::from_str(toml).unwrap();
        assert!(cfg.watches[0].template.is_some());
    }

    /// Path with `{a,b}` (brace expansion) → dynamic; the brace stays one glob component, so the
    /// anchor is the literal prefix in front of it.
    #[test]
    fn brace_path_dispatches_to_dynamic() {
        let toml = "[[watch]]\nname = \"dyn\"\npath = \"/srv/log/{app,system}/access.log\"\n\
                    actions = [{ exec = [\"echo\"] }]";
        let cfg = Config::from_str(toml).unwrap();
        assert!(cfg.watches[0].template.is_some());
        assert_eq!(cfg.watches[0].path, Path::new("/srv/log"));
    }

    /// Mixed config — both kinds land in the one `watches` list in source order; the kind
    /// difference is template presence.
    #[test]
    fn mixed_static_and_dynamic_routes_each_correctly() {
        let toml = "\
            [[watch]]\nname = \"a\"\npath = \"/foo\"\nactions = [{ exec = [\"echo\"] }]\n\
            [[watch]]\nname = \"b\"\npath = \"/bar/*\"\nactions = [{ exec = [\"echo\"] }]\n\
            [[watch]]\nname = \"c\"\npath = \"/baz\"\nactions = [{ exec = [\"echo\"] }]\n\
            [[watch]]\nname = \"d\"\npath = \"/qux/{a,b}\"\nactions = [{ exec = [\"echo\"] }]\n\
        ";
        let cfg = Config::from_str(toml).unwrap();
        let kinds: Vec<(&str, bool)> = cfg
            .watches
            .iter()
            .map(|w| (w.name.as_str(), w.template.is_some()))
            .collect();
        assert_eq!(
            kinds,
            vec![("a", false), ("b", true), ("c", false), ("d", true)],
        );
    }

    /// Cross-kind duplicate name still rejected — the duplicate-name check runs at the dispatch
    /// loop so both lists are scanned.
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

    /// Globstar (`**`) is unsupported in v1 — surfaced as `IssueKind::InvalidPattern`.
    #[test]
    fn globstar_pattern_rejected_as_invalid_pattern() {
        let toml =
            "[[watch]]\nname = \"d\"\npath = \"/var/log/**/x\"\nactions = [{ exec = [\"echo\"] }]";
        assert_only_kind(toml, IssueKind::InvalidPattern);
    }

    /// Dynamic-detected non-absolute path (e.g., `var/log/*`) routes to the dynamic validator and
    /// the parser surfaces `NonAbsolute` as `IssueKind::InvalidPattern` with the source rendered.
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

    /// Malformed glob segment — unbalanced `[` — surfaces via the PatternSpec parser as
    /// `InvalidGlob`, which we re-cast to `IssueKind::InvalidPattern`.
    #[test]
    fn malformed_glob_segment_rejected_as_invalid_pattern() {
        let toml = "[[watch]]\nname = \"d\"\npath = \"/var/log/[unbalanced\"\nactions = [{ exec = [\"echo\"] }]";
        assert_only_kind(toml, IssueKind::InvalidPattern);
    }

    // ---- Discovery lowering ----

    /// Minimal dynamic watch: the discovery Sub's own identity is the constant pair plus `STRUCTURE`
    /// and `MatchChain`; the template carries the same defaults the static validator would produce
    /// (settle = 200ms, max_settle = 1h, recursive `Subtree`, scope-conditional events).
    #[test]
    fn minimal_dynamic_watch_round_trips_with_defaults() {
        let toml =
            "[[watch]]\nname = \"logs\"\npath = \"/srv/log/*\"\nactions = [{ exec = [\"echo\"] }]";
        let cfg = Config::from_str(toml).unwrap();
        let w = &cfg.watches[0];
        assert_eq!(w.name, "logs");
        assert_eq!(w.scope, EffectScope::SubtreeRoot);
        assert_eq!(w.settle, Duration::from_millis(150));
        assert_eq!(w.max_settle, Duration::from_secs(2));
        assert_eq!(w.events, ClassSet::STRUCTURE);
        assert!(!w.log_output);
        let t = w.template.as_ref().expect("dynamic watch carries template");
        assert_eq!(t.settle, Duration::from_millis(200));
        assert_eq!(t.max_settle, Duration::from_hours(1));
        assert_eq!(t.events, ClassSet::DEFAULT_SUBTREE_ROOT);
        let ScanConfig::Subtree { recursive, .. } = &t.scan else {
            panic!("template scan is the user Subtree, got {:?}", t.scan);
        };
        assert!(*recursive);
    }

    /// The full lowering grid: a dynamic block's user knobs land on the template *verbatim* (`scan` /
    /// `events` / `settle` / `max_settle`), `program`/`scope`/`log_output` stay on the Sub (doubling
    /// as the minted reaction spec), and the Sub's own identity is pinned to the discovery constants
    /// with the literal-prefix anchor. The projected `MintTemplate`'s identity hash equals a
    /// hand-built [`ProfileIdentity`] over the same knobs — the projection adds nothing.
    #[test]
    fn dynamic_block_lowers_to_discovery_sub_spec() {
        let toml = "[[watch]]\nname = \"logs\"\npath = \"/srv/log/*\"\n\
                    actions = [{ exec = [\"fmt\", \"${specter.path}\"] }]\n\
                    settle = \"300ms\"\nmax_settle = \"1200ms\"\n\
                    scope = \"per-stable-file\"\n\
                    events = [\"content\"]\n\
                    log_output = true\n\
                    pattern = \"*.log\"\n\
                    recursive = false\nhidden = true\n";
        let cfg = Config::from_str(toml).unwrap();
        let w = &cfg.watches[0];

        // Discovery Sub's own identity: constants + MatchChain + literal-prefix anchor.
        assert_eq!(w.path, Path::new("/srv/log"));
        assert_eq!(w.settle, Duration::from_millis(150));
        assert_eq!(w.max_settle, Duration::from_secs(2));
        assert_eq!(w.events, ClassSet::STRUCTURE);
        let ScanConfig::MatchChain(spec) = &w.scan else {
            panic!("dynamic watch lowers to MatchChain, got {:?}", w.scan);
        };
        assert_eq!(spec.source(), "/srv/log/*");

        // Reaction spec stays on the Sub.
        assert_eq!(w.scope, EffectScope::PerStableFile);
        assert!(w.log_output);

        // User knobs land on the template verbatim.
        let t = w.template.as_ref().expect("dynamic watch carries template");
        assert_eq!(t.settle, Duration::from_millis(300));
        assert_eq!(t.max_settle, Duration::from_millis(1200));
        assert_eq!(t.events, ClassSet::CONTENT);
        let ScanConfig::Subtree {
            recursive,
            hidden,
            pattern,
            ..
        } = &t.scan
        else {
            panic!("template scan is the user Subtree, got {:?}", t.scan);
        };
        assert!(!*recursive);
        assert!(*hidden);
        assert!(pattern.is_some());

        // The attach-request projection: minted identity hash equals the hand-built one.
        let req = cfg.watches[0].to_attach_request();
        let ReactionSpec::Mint(minted) = &req.params.reaction else {
            panic!("dynamic watch lowers to a Mint reaction");
        };
        assert_eq!(minted.settle, Duration::from_millis(300));
        let hand_built = ProfileIdentity::new(t.scan.clone(), t.max_settle, t.events);
        assert_eq!(minted.identity.config_hash(), hand_built.config_hash());
        // The discovery Sub's own request identity is the constant shape.
        assert_eq!(req.identity.max_settle(), Duration::from_secs(2));
        assert_eq!(req.identity.events(), ClassSet::STRUCTURE);
    }

    /// Dynamic watches accept `pattern` (per-Sub include filter) and `exclude` (per-Sub exclude
    /// list) the same way static watches do — they're orthogonal to the path-pattern dispatch and
    /// scope the *minted* Profiles via the template scan.
    #[test]
    fn dynamic_watch_carries_scan_pattern_and_excludes() {
        let toml = "[[watch]]\nname = \"logs\"\npath = \"/srv/log/*\"\n\
                    actions = [{ exec = [\"echo\"] }]\n\
                    pattern = \"*.log\"\n\
                    exclude = [\"*.gz\"]\n";
        let cfg = Config::from_str(toml).unwrap();
        let t = cfg.watches[0]
            .template
            .as_ref()
            .expect("dynamic watch carries template");
        let ScanConfig::Subtree {
            pattern, exclude, ..
        } = &t.scan
        else {
            panic!("template scan is the user Subtree, got {:?}", t.scan);
        };
        assert!(pattern.is_some());
        assert_eq!(exclude.len(), 1);
    }

    /// Multiple errors in one dynamic watch accumulate — pattern parse failure does NOT
    /// short-circuit settle / scope / events validation. The operator gets the full list at once.
    #[test]
    fn multiple_errors_in_one_dynamic_watch_accumulate() {
        let toml = "[[watch]]\nname = \"\"\npath = \"/foo/**/x\"\nactions = [{ exec = [] }]\n\
                    settle = \"0ms\"\nmax_depth = 0";
        let err = Config::from_str(toml).unwrap_err();
        let errors = validation_errors(err);
        let kinds: Vec<IssueKind> = errors.iter().map(|e| e.kind).collect();
        assert!(kinds.contains(&IssueKind::EmptyName));
        assert!(kinds.contains(&IssueKind::InvalidPattern));
        assert!(kinds.contains(&IssueKind::EmptyArgv));
        assert!(kinds.contains(&IssueKind::SettleTooSmall));
        assert!(kinds.contains(&IssueKind::MaxDepthZero));
        assert_eq!(errors.len(), 5, "got {errors:?}");
    }

    /// FS-root pattern `/*` lowers to a discovery Sub anchored at `/` — the bare-root anchor with
    /// no parent edge.
    #[test]
    fn fs_root_glob_pattern_accepted() {
        let toml = "[[watch]]\nname = \"root\"\npath = \"/*\"\nactions = [{ exec = [\"echo\"] }]";
        let cfg = Config::from_str(toml).unwrap();
        assert!(cfg.watches[0].template.is_some());
        assert_eq!(cfg.watches[0].path, Path::new("/"));
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

        // Without intervening mutation, the lstat-equivalent re-capture compares bit-equal to the
        // atomically-captured meta — this is the steady-state invariant the driver's settle-expiry
        // filter relies on for "no change ⇒ no reload".
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

        // The two entry points must produce identical Config values — `from_path` is the meta-free
        // path; `from_path_with_meta` additionally returns the inode meta. Divergence here would
        // mean the two entry points parse the same bytes differently.
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
        // The full atomic-capture invariant: `from_path_with_meta`'s returned meta belongs to the
        // inode opened, even when the path is renamed out from under us between `File::open` and
        // any subsequent path-level stat. Simulates the atomic-save race by performing the rename
        // after the call returns and confirming `meta` still reflects the original (now orphan)
        // inode while a fresh path-level `from_path` reflects the replacement.
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
        // The driver's lstat filter would detect this as `stored != current` and fire a reload;
        // this test asserts the precondition (meta deltas are observable across atomic-save).
    }
}
