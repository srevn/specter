use std::fmt;
use std::path::{Path, PathBuf};

#[non_exhaustive]
#[derive(Debug)]
pub enum ConfigError {
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    Parse {
        path: Option<PathBuf>,
        message: String,
    },
    Validate {
        path: Option<PathBuf>,
        errors: Vec<ValidationIssue>,
    },
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ValidationIssue {
    pub watch_index: Option<usize>,
    pub field: &'static str,
    pub kind: IssueKind,
    pub detail: String,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum IssueKind {
    Empty,
    /// `actions = []` — at least one entry required.
    EmptyActions,
    /// `actions[i]` carries no variant (e.g., `actions = [{}]`) — none
    /// of `exec`, `pipe`, or the conditional triple (`when` / `then` /
    /// `else`) is set. Exactly one variant must be supplied.
    ActionMissingVariant,
    /// `actions[i]` carries multiple variants set simultaneously
    /// (e.g., both `exec` and `pipe`, or `exec` together with the
    /// conditional triple). The three variants are mutually exclusive.
    ActionAmbiguousVariant,
    EmptyArgv,
    NonAbsolute,
    NotCanonical,
    InvalidGlob,
    UnknownPlaceholder,
    SettleTooSmall,
    MaxSettleTooSmall,
    MaxDepthZero,
    DuplicateName,
    InvalidEnum,
    EventsEmpty,
    DuplicateEventClass,
    /// Reserved-character violation in the user-supplied `name` field.
    /// Currently emitted when `name` contains `@`, which the engine
    /// reserves for the synthesized `<promoter_name>@<resolved_path>`
    /// shape of dynamic Subs. Distinct from
    /// [`Self::Empty`] (empty name) and [`Self::DuplicateName`].
    InvalidName,
    /// `path` of a dynamic `[[watch]]` failed `PatternSpec::parse` —
    /// any of `**`, `.`/`..`, empty segment, non-absolute, Windows
    /// prefix, or a malformed glob segment. Detail carries the
    /// rendered [`specter_core::PatternError`] message.
    InvalidPattern,
    /// `actions[i].timeout` is set on an action variant that doesn't
    /// support a top-level timeout. v1: only `exec` accepts it. Future
    /// variants (`pipe`, `conditional`) set timeouts on their stages /
    /// predicate, not on the action itself; this kind catches the
    /// "operator misread the schema" case at config-load time.
    TimeoutNotApplicable,
    /// `actions[i].timeout` is `Some(Duration::ZERO)`. A zero-duration
    /// timeout would SIGTERM the child before it makes any progress,
    /// which is almost certainly a typo. Operators wanting "no
    /// deadline" omit the field entirely.
    TimeoutZero,
    /// `actions[i]` partially sets the conditional triple: `when`
    /// without `then`, or `then` / `else` without `when`. The grammar
    /// requires both `when` and `then` together; `else` is optional.
    /// Fires before the conditional body is validated so per-branch
    /// errors don't pile on a structurally-broken entry.
    ConditionalIncomplete,
    /// `actions[i]` is a fully-formed conditional with both `then = []`
    /// and `else = []` (or `else` absent). The predicate would run for
    /// no observable effect; almost certainly an operator mistake.
    /// Empty `then` with a non-empty `else` is allowed (equivalent to
    /// a negated predicate).
    EmptyConditional,
    /// `actions[i].pipe = []` — an empty pipe has no stages to wire,
    /// so the actuator has nothing to spawn. Almost certainly an
    /// operator-side typo; the validator surfaces it as a distinct
    /// kind from [`Self::EmptyArgv`] (which refers to an empty `exec`
    /// argv inside one slot).
    EmptyPipe,
    /// `actions[i].pipe = [{ exec = [...] }]` — a single-stage pipe
    /// degenerates to a plain `exec` with extra TOML structure. The
    /// validator rejects it so the operator's intent is unambiguous;
    /// `exec = [...]` at the action's top level is the right shape.
    SingleStagePipe,
    /// `actions[i]` nests `when` / `then` / `else` past the validator's
    /// recursion bound (see `MAX_CONDITIONAL_DEPTH` in `config.rs`).
    /// Surfaces before any further descent at the offending level —
    /// keeps adversarial inputs from blowing the validator's stack
    /// without constraining sensible operator workflows (real configs
    /// rarely exceed five levels). Independent of the underlying TOML
    /// parser's own recursion limit (a separate, parser-version-
    /// dependent concern).
    ConditionalNestedTooDeep,
    /// Internal lowering invariant violation — an unpatched edge, a
    /// backward branch, or an out-of-bounds target leaked from the
    /// lowering pass. Unreachable from a correct lowering, surfaced as
    /// a validation issue (rather than a panic) so the operator gets a
    /// loadable error message instead of a crashed process.
    ///
    /// The "program too large" case (> `u32::MAX` ops) is not
    /// representable: the builder panics on emit as a precondition
    /// failure (~128 GiB of in-memory state, physically impossible).
    LoweringInternal,
}

impl ConfigError {
    pub(crate) fn parse(path: Option<&Path>, e: &toml::de::Error) -> Self {
        Self::Parse {
            path: path.map(Path::to_owned),
            message: e.to_string(),
        }
    }
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, source } => {
                write!(f, "io error reading {}: {source}", path.display())
            }
            Self::Parse { path, message } => match path {
                Some(p) => write!(f, "parse error in {}: {message}", p.display()),
                None => write!(f, "parse error: {message}"),
            },
            Self::Validate { path, errors } => {
                let prefix = path
                    .as_ref()
                    .map_or_else(|| "<inline>".to_owned(), |p| p.display().to_string());
                writeln!(f, "{prefix}: {} validation error(s):", errors.len())?;
                for e in errors {
                    writeln!(f, "  {e}")?;
                }
                Ok(())
            }
        }
    }
}

impl std::error::Error for ConfigError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        if let Self::Io { source, .. } = self {
            Some(source)
        } else {
            None
        }
    }
}

impl ValidationIssue {
    pub(crate) const fn new(
        watch_index: Option<usize>,
        field: &'static str,
        kind: IssueKind,
        detail: String,
    ) -> Self {
        Self {
            watch_index,
            field,
            kind,
            detail,
        }
    }
}

impl fmt::Display for ValidationIssue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.watch_index {
            Some(i) => write!(
                f,
                "watch[{i}].{}: {} ({})",
                self.field,
                self.detail,
                kind_label(self.kind)
            ),
            None => write!(
                f,
                "{}: {} ({})",
                self.field,
                self.detail,
                kind_label(self.kind)
            ),
        }
    }
}

const fn kind_label(k: IssueKind) -> &'static str {
    match k {
        IssueKind::Empty => "empty",
        IssueKind::EmptyActions => "empty-actions",
        IssueKind::ActionMissingVariant => "action-missing-variant",
        IssueKind::ActionAmbiguousVariant => "action-ambiguous-variant",
        IssueKind::EmptyArgv => "empty-argv",
        IssueKind::NonAbsolute => "non-absolute",
        IssueKind::NotCanonical => "not-canonical",
        IssueKind::InvalidGlob => "invalid-glob",
        IssueKind::UnknownPlaceholder => "unknown-placeholder",
        IssueKind::SettleTooSmall => "settle-too-small",
        IssueKind::MaxSettleTooSmall => "max-settle-too-small",
        IssueKind::MaxDepthZero => "max-depth-zero",
        IssueKind::DuplicateName => "duplicate-name",
        IssueKind::InvalidEnum => "invalid-enum",
        IssueKind::EventsEmpty => "events-empty",
        IssueKind::DuplicateEventClass => "duplicate-event-class",
        IssueKind::InvalidName => "invalid-name",
        IssueKind::InvalidPattern => "invalid-pattern",
        IssueKind::TimeoutNotApplicable => "timeout-not-applicable",
        IssueKind::TimeoutZero => "timeout-zero",
        IssueKind::ConditionalIncomplete => "conditional-incomplete",
        IssueKind::EmptyConditional => "empty-conditional",
        IssueKind::EmptyPipe => "empty-pipe",
        IssueKind::SingleStagePipe => "single-stage-pipe",
        IssueKind::ConditionalNestedTooDeep => "conditional-nested-too-deep",
        IssueKind::LoweringInternal => "lowering-internal",
    }
}

impl ValidationIssue {
    /// Map a [`specter_core::program::ProgramError`] into the
    /// validation-issue surface. Every variant collapses to
    /// [`IssueKind::LoweringInternal`] — none are reachable from a
    /// correct lowering pass, but the validator captures them as
    /// issues rather than panicking. The "program too large" case is
    /// not representable on the source error (the builder panics on
    /// emit as a precondition failure — physically impossible to
    /// load).
    pub(crate) fn from_program_error(
        e: &specter_core::program::ProgramError,
        watch_index: Option<usize>,
        field: &'static str,
    ) -> Self {
        Self::new(
            watch_index,
            field,
            IssueKind::LoweringInternal,
            format!("lowering invariant violated: {e}"),
        )
    }
}
