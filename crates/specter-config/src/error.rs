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
    /// `actions[i]` carries no variant (e.g., `actions = [{}]`). Once
    /// new variants land alongside `exec`, this kind also fires when
    /// every variant is `None` — exactly one must be set.
    ActionMissingVariant,
    /// `actions[i]` carries multiple variants set simultaneously. v1's
    /// single `exec` variant means this is unreachable today; the kind
    /// is reserved so the validator can keep the "exactly one variant"
    /// rule as a single check across both v1 and v2.
    ActionAmbiguousVariant,
    /// PR 1 guard: `actions.len() > 1` is rejected pending PR 2's
    /// multi-step actuator support. Removed once PR 2 lands.
    MultiStepNotYetSupported,
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
    /// prefix, or a malformed glob segment. Detail carries the rendered
    /// [`specter_core::PatternError`] message.
    InvalidPattern,
    /// Defense-in-depth: `validate_static_watch` was reached with a
    /// path containing one of the four glob discriminator characters
    /// (`*?[{`). Production paths gate on `PatternSpec::is_dynamic`
    /// upstream, so this kind is unreachable through the dispatcher;
    /// it surfaces only when an internal caller bypasses the dispatch
    /// (e.g., a future test that invokes the static validator
    /// directly). Distinct from [`Self::InvalidPattern`] so the
    /// dispatcher's contract is observable in error output.
    PathContainsGlobChars,
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
        IssueKind::MultiStepNotYetSupported => "multi-step-not-yet-supported",
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
        IssueKind::PathContainsGlobChars => "path-contains-glob-chars",
    }
}
