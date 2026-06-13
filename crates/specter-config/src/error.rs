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
    /// User-supplied `[[watch]] name` is empty. Distinct from [`Self::EmptyLogPath`] (the
    /// file-destination path) and [`Self::EmptyPath`] (the watch path) so operator-triage
    /// categories stay one-to-one with their fields.
    EmptyName,
    /// `[log] destination = "file"` with no `[log] path` (and no `--log-path` CLI override). The
    /// `path` field is required iff the destination resolves to `File`. Distinct from
    /// [`Self::EmptyName`] and [`Self::EmptyPath`].
    EmptyLogPath,
    /// `[[watch]] path` is empty. Maps from `PathError::Empty`. Distinct from [`Self::EmptyName`]
    /// and [`Self::EmptyLogPath`] so operator-triage categories stay one-to-one with their fields.
    EmptyPath,
    /// `[[watch]] path` contains a `..` component (anywhere). Maps from
    /// `PathError::ContainsParentDir`. The operator must supply a literal absolute path without
    /// parent-dir traversal — `..` segments would silently change filesystem semantics by
    /// collapsing one symlink boundary.
    PathContainsParentDir,
    /// `[[watch]] path` canonicalisation hit a non-`NotFound` `io::Error` — `PermissionDenied`
    /// (EACCES), symlink loop (ELOOP), non-directory in path (ENOTDIR), EIO, etc. Maps from
    /// `PathError::Inaccessible`. The detail line carries the cursor at fault and the underlying
    /// error so operators can distinguish the failure class without reaching for `strace` / `dtruss`.
    PathInaccessible,
    /// `[[watch]] path` canonicalised to a buffer carrying non-UTF-8 segments — typically via
    /// symlink resolution onto a non-UTF-8 byte path. Maps from `PathError::NonUtf8`. The engine's
    /// [`specter_core::Tree::parse_attach_path`] gate would reject the path; surface up-front so
    /// the validator's contract ("ok ⇒ engine accepts the path") holds.
    NonUtf8Path,
    /// `actions = []` — at least one entry required.
    EmptyActions,
    /// `actions[i]` carries no variant (e.g., `actions = [{}]`) — none of `exec`, `pipe`, or the
    /// conditional triple (`when` / `then` / `else`) is set. Exactly one variant must be supplied.
    ActionMissingVariant,
    /// `actions[i]` carries multiple variants set simultaneously (e.g., both `exec` and `pipe`, or
    /// `exec` together with the conditional triple). The three variants are mutually exclusive.
    ActionAmbiguousVariant,
    EmptyArgv,
    NonAbsolute,
    InvalidGlob,
    UnreachableGlob,
    UnknownPlaceholder,
    SettleTooSmall,
    MaxSettleTooSmall,
    MaxDepthZero,
    DuplicateName,
    InvalidEnum,
    EventsEmpty,
    DuplicateEventClass,
    /// Reserved-character violation in the user-supplied `name` field. Currently emitted when `name`
    /// contains `@`, which the engine reserves for the minted `<template_name>@<matched_path>` shape
    /// of dynamic Subs. Distinct from [`Self::EmptyName`] (empty name) and [`Self::DuplicateName`].
    InvalidName,
    /// `path` of a dynamic `[[watch]]` failed `PatternSpec::parse` — any of `**`, `.`/`..`, empty
    /// segment, non-absolute, Windows prefix, or a malformed glob segment. Detail carries the
    /// rendered [`specter_core::PatternError`] message.
    InvalidPattern,
    /// The watch's effective event mask (the static entry's `events`, or a dynamic entry's template
    /// `events`) does not cover the classes its scan shape needs to witness quiescence over a
    /// settle window — for a subtree watch, CONTENT. Advisory, never fatal: emitted by
    /// [`crate::Config::warnings`], never by `validate`. The configuration is documented and
    /// intended (the hash-channel safety net for `mmap` / async-I/O / `splice(2)` writers whose
    /// in-place writes the kernel may not surface as events), but its price is structural: every
    /// fire must prove quiescence through two consecutive agreeing full subtree walks at the anchor
    /// with mtime-skip disabled, instead of one event-scoped walk. The warning makes that cost
    /// visible at config load; the detail carries the trade-off and the opt-out (add `"content"` to
    /// `events` when no such writers exist).
    EventsIncompleteMask,
    /// `actions[i].timeout` is set on an action variant that doesn't support a top-level timeout.
    /// v1: only `exec` accepts it. Future variants (`pipe`, `conditional`) set timeouts on their
    /// stages / predicate, not on the action itself; this kind catches the "operator misread the
    /// schema" case at config-load time.
    TimeoutNotApplicable,
    /// `actions[i].timeout` is `Some(Duration::ZERO)`. A zero-duration timeout would SIGTERM the
    /// child before it makes any progress, which is almost certainly a typo. Operators wanting "no
    /// deadline" omit the field entirely.
    TimeoutZero,
    /// `actions[i]` partially sets the conditional triple: `when` without `then`, or `then` /
    /// `else` without `when`. The grammar requires both `when` and `then` together; `else` is
    /// optional. Fires before the conditional body is validated so per-branch errors don't pile on
    /// a structurally-broken entry.
    ConditionalIncomplete,
    /// `actions[i]` is a fully-formed conditional with both `then = []` and `else = []` (or `else`
    /// absent). The predicate would run for no observable effect; almost certainly an operator
    /// mistake. Empty `then` with a non-empty `else` is allowed (equivalent to a negated predicate).
    EmptyConditional,
    /// `actions[i].pipe = []` — an empty pipe has no stages to wire, so the actuator has nothing to
    /// spawn. Almost certainly an operator-side typo; the validator surfaces it as a distinct kind
    /// from [`Self::EmptyArgv`] (which refers to an empty `exec` argv inside one slot).
    EmptyPipe,
    /// `actions[i].pipe = [{ exec = [...] }]` — a single-stage pipe degenerates to a plain `exec`
    /// with extra TOML structure. The validator rejects it so the operator's intent is unambiguous;
    /// `exec = [...]` at the action's top level is the right shape.
    SingleStagePipe,
    /// `actions[i]` nests `when` / `then` / `else` past the validator's recursion bound (see
    /// `MAX_CONDITIONAL_DEPTH` in `config.rs`). Surfaces before any further descent at the offending
    /// level — keeps adversarial inputs from blowing the validator's stack without constraining
    /// sensible operator workflows (real configs rarely exceed five levels). Independent of the
    /// underlying TOML parser's own recursion limit (a separate, parser-version- dependent concern).
    ConditionalNestedTooDeep,
    /// Internal lowering invariant violation — an unpatched edge, a backward branch, or an
    /// out-of-bounds target leaked from the lowering pass. Unreachable from a correct lowering,
    /// surfaced as a validation issue (rather than a panic) so the operator gets a loadable error
    /// message instead of a crashed process.
    ///
    /// The "program too large" case (> `u32::MAX` ops) is not representable: the builder panics on
    /// emit as a precondition failure (~128 GiB of in-memory state, physically impossible).
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
                // `Path::display()` implements `Display`, so write the header through the formatter
                // directly instead of routing through a transient `String` allocation. The
                // `<inline>` literal stands in for the file-less (string-source) case where
                // `Config::from_str` is the entry point.
                let n = errors.len();
                match path {
                    Some(p) => writeln!(f, "{}: {n} validation error(s):", p.display())?,
                    None => writeln!(f, "<inline>: {n} validation error(s):")?,
                }
                // Trailing-newline hygiene: `writeln!` every issue except the last, which uses
                // `write!`. The outer printer adds the final newline (println / eprintln / tracing).
                // `split_last` returns `None` only on an empty slice — a shape the constructor never
                // produces, but we degrade gracefully (header alone) rather than panic.
                if let Some((last, rest)) = errors.split_last() {
                    for e in rest {
                        writeln!(f, "  {e}")?;
                    }
                    write!(f, "  {last}")?;
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
        IssueKind::EmptyName => "empty-name",
        IssueKind::EmptyLogPath => "empty-log-path",
        IssueKind::EmptyPath => "empty-path",
        IssueKind::PathContainsParentDir => "path-contains-parent-dir",
        IssueKind::PathInaccessible => "path-inaccessible",
        IssueKind::NonUtf8Path => "non-utf8-path",
        IssueKind::EmptyActions => "empty-actions",
        IssueKind::ActionMissingVariant => "action-missing-variant",
        IssueKind::ActionAmbiguousVariant => "action-ambiguous-variant",
        IssueKind::EmptyArgv => "empty-argv",
        IssueKind::NonAbsolute => "non-absolute",
        IssueKind::InvalidGlob => "invalid-glob",
        IssueKind::UnreachableGlob => "unreachable-glob",
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
        IssueKind::EventsIncompleteMask => "events-incomplete-mask",
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
    /// Map a [`specter_core::program::ProgramError`] into the validation-issue surface. Every
    /// variant collapses to [`IssueKind::LoweringInternal`] — none are reachable from a correct
    /// lowering pass, but the validator captures them as issues rather than panicking. The "program
    /// too large" case is not representable on the source error (the builder panics on emit as a
    /// precondition failure — physically impossible to load).
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
