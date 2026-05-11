//! Leaf types for the action IR — argv specs and placeholders.
//!
//! These types describe *what one process looks like*: a frozen
//! `Box<[ArgTemplate]>` of argv parts, plus a per-step timeout. They are
//! shared between every op variant in [`super::SpawnBody`] (single
//! `Exec` and N-stage `Pipe`) and never carry control-flow state — that
//! lives one layer up in [`super::ProgramOp`].

use compact_str::CompactString;
use smallvec::SmallVec;
use std::time::Duration;

/// One leaf-process specification.
///
/// `argv` is frozen `Box<[ArgTemplate]>` — once a config is validated,
/// the argv shape is fixed and `Vec`'s capacity slot is dead weight;
/// `Box` saves the two extra words per leaf and prevents accidental
/// push paths.
///
/// `timeout` is the deadline applied to the spawned process. `None` ⇒
/// no timeout. `Some(d)` ⇒ SIGTERM at `now + d`; if still alive after
/// the actuator's shutdown grace, SIGKILL. Reaped as
/// `EffectOutcome::Failed { exit_code: None, signal: Some(15 | 9) }`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExecAction {
    pub argv: Box<[ArgTemplate]>,
    pub timeout: Option<Duration>,
}

impl ExecAction {
    #[must_use]
    pub fn new(argv: impl IntoIterator<Item = ArgTemplate>) -> Self {
        Self {
            argv: argv.into_iter().collect::<Vec<_>>().into_boxed_slice(),
            timeout: None,
        }
    }

    /// Builder-style setter for the per-step timeout.
    #[must_use]
    pub const fn with_timeout(mut self, d: Duration) -> Self {
        self.timeout = Some(d);
        self
    }

    /// `true` iff any argv part references a diff-derived placeholder.
    #[must_use]
    pub fn references_diff_derived(&self) -> bool {
        self.argv
            .iter()
            .any(|arg| arg.parts.iter().any(ArgPart::is_diff_derived))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ArgTemplate {
    pub parts: SmallVec<[ArgPart; 2]>,
}

impl ArgTemplate {
    #[must_use]
    pub fn new(parts: impl IntoIterator<Item = ArgPart>) -> Self {
        Self {
            parts: parts.into_iter().collect(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ArgPart {
    Literal(CompactString),
    Placeholder(Placeholder),
    /// `${env.<NAME>}` or `${env.<NAME>:-default}`. The default is a
    /// frozen literal — nested placeholders are rejected at the lexer.
    /// Strict resolution: `default = None` AND env unset ⇒
    /// `EffectOutcome::Failed`.
    EnvVar {
        name: CompactString,
        default: Option<CompactString>,
    },
}

impl ArgPart {
    #[must_use]
    pub fn literal(s: impl Into<CompactString>) -> Self {
        Self::Literal(s.into())
    }

    /// True iff this part is a multi-value [`Placeholder`]. Thin
    /// delegator over [`Placeholder::is_multivalue`] for ergonomic
    /// `iter().any(ArgPart::is_multivalue)` use at call sites that need
    /// to inspect mixed `Literal` / `Placeholder` parts. `EnvVar` is
    /// single-value by construction.
    #[must_use]
    pub const fn is_multivalue(&self) -> bool {
        match self {
            Self::Placeholder(p) => p.is_multivalue(),
            Self::Literal(_) | Self::EnvVar { .. } => false,
        }
    }

    /// True iff this part is a diff-derived [`Placeholder`]. See
    /// [`Placeholder::is_diff_derived`] for the precise predicate.
    /// `EnvVar` reads the actuator's captured environment snapshot,
    /// never the burst's `Diff`, so it never flips this predicate.
    #[must_use]
    pub const fn is_diff_derived(&self) -> bool {
        match self {
            Self::Placeholder(p) => p.is_diff_derived(),
            Self::Literal(_) | Self::EnvVar { .. } => false,
        }
    }
}

/// Argv-template substitution token. The catalog spans two predicates:
///
/// - **[`Self::is_multivalue`]** — true for any placeholder that can
///   expand to >1 argv slot: `Created`, `Deleted`, `Modified`,
///   `RenamedFrom`, `RenamedTo`, `Excluded`. Drives the resolver's
///   prefix-accumulator branching.
/// - **[`Self::is_diff_derived`]** — true for the multi-value
///   placeholders sourced from the burst's `Diff`: the original five.
///   `Excluded` is multi-value but reads from `Profile.exclude_strings`,
///   not from a `Diff` — keeping it OUT of `is_diff_derived` is what
///   prevents `Sub.needs_diff` from falsely ratcheting on `Excluded`.
///
/// Single-value variants (`Path`, `Relative`, `Anchor`, `Watch`,
/// `Parent`, `Time`) render to one argv slot; multi-value variants
/// drop the surrounding argv slot when their source list is empty.
///
/// `Parent` semantics for the corner cases:
///
/// | Scope    | Anchor    | Segment    | `target_path`     | `Parent`          |
/// |----------|-----------|------------|-------------------|-------------------|
/// | PerFile  | `/anchor` | `foo.rs`   | `/anchor/foo.rs`  | `/anchor`         |
/// | PerFile  | `/anchor` | `src/lib`  | `/anchor/src/lib` | `/anchor/src`     |
/// | PerFile  | `/`       | `foo.rs`   | `/foo.rs`         | `/` (NOT empty)   |
/// | Subtree  | `/anchor` | (n/a)      | `/anchor`         | `/`               |
/// | Subtree  | `/`       | (n/a)      | `/`               | `""` (only case)  |
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum Placeholder {
    Path,
    Relative,
    Anchor,
    Watch,
    Parent,
    /// RFC 3339 UTC second-precision (`2026-05-10T12:34:56Z`). Sampled
    /// at spawn-time, not at engine emit time — operators reading
    /// `$SPECTER_TIME` see the wall-clock instant immediately before
    /// the kernel runs the user's command.
    Time,
    Created,
    Deleted,
    Modified,
    RenamedFrom,
    RenamedTo,
    /// One argv slot per pattern in `Profile.exclude_strings`. NOT
    /// diff-derived: `Sub.needs_diff` does not ratchet on this.
    Excluded,
}

impl Placeholder {
    /// True for any placeholder that can expand to >1 argv slot:
    /// `Created`, `Deleted`, `Modified`, `RenamedFrom`, `RenamedTo`,
    /// `Excluded`. Drives the resolver's prefix-accumulator branching.
    #[must_use]
    pub const fn is_multivalue(self) -> bool {
        matches!(
            self,
            Self::Created
                | Self::Deleted
                | Self::Modified
                | Self::RenamedFrom
                | Self::RenamedTo
                | Self::Excluded
        )
    }

    /// True for multi-value placeholders sourced from the burst's
    /// `Diff` (the original five). `Excluded` is multi-value but reads
    /// from `Profile.exclude_strings`, NOT from a `Diff` — it is
    /// excluded from this predicate so the `Sub.needs_diff` derivation
    /// doesn't falsely ratchet on the `Excluded` variant.
    ///
    /// Invariant: `is_diff_derived ⇒ is_multivalue`. The converse does
    /// not hold (`Excluded` breaks it).
    #[must_use]
    pub const fn is_diff_derived(self) -> bool {
        matches!(
            self,
            Self::Created | Self::Deleted | Self::Modified | Self::RenamedFrom | Self::RenamedTo
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{ArgPart, ArgTemplate, ExecAction, Placeholder};
    use std::time::Duration;

    /// `ArgPart::EnvVar` never flips `is_diff_derived` — the resolver
    /// reads the actuator's captured snapshot, not the burst's diff.
    /// Pinning this prevents future refactors from silently ratcheting
    /// `Sub.needs_diff` on env-only argv.
    #[test]
    fn env_var_arg_part_is_not_diff_derived() {
        let part = ArgPart::EnvVar {
            name: "HOME".into(),
            default: None,
        };
        assert!(!part.is_diff_derived());
        assert!(!part.is_multivalue());
    }

    /// `ExecAction::with_timeout` sets the per-step deadline; default
    /// (no setter call) leaves `timeout = None`.
    #[test]
    fn exec_action_with_timeout_records_duration() {
        let exec = ExecAction::new([ArgTemplate::new([ArgPart::literal("/bin/true")])])
            .with_timeout(Duration::from_secs(2));
        assert_eq!(exec.timeout, Some(Duration::from_secs(2)));

        let exec_default = ExecAction::new([ArgTemplate::new([ArgPart::literal("/bin/true")])]);
        assert_eq!(exec_default.timeout, None);
    }

    /// `ExecAction::references_diff_derived` is `true` iff any argv
    /// part is a diff-derived placeholder. Anchor-only argv ⇒ `false`.
    #[test]
    fn exec_action_references_diff_derived_matches_argv() {
        let anchor_only = ExecAction::new([ArgTemplate::new([
            ArgPart::literal("/bin/build"),
            ArgPart::Placeholder(Placeholder::Path),
        ])]);
        assert!(!anchor_only.references_diff_derived());

        for p in [
            Placeholder::Created,
            Placeholder::Deleted,
            Placeholder::Modified,
            Placeholder::RenamedFrom,
            Placeholder::RenamedTo,
        ] {
            let exec = ExecAction::new([ArgTemplate::new([ArgPart::Placeholder(p)])]);
            assert!(
                exec.references_diff_derived(),
                "references_diff_derived must be true for argv containing {p:?}"
            );
        }
    }

    /// `Placeholder::is_multivalue` covers the five diff entries plus
    /// `Excluded`. Single-value variants stay outside the set.
    #[test]
    fn placeholder_is_multivalue_includes_excluded() {
        for p in [
            Placeholder::Created,
            Placeholder::Deleted,
            Placeholder::Modified,
            Placeholder::RenamedFrom,
            Placeholder::RenamedTo,
            Placeholder::Excluded,
        ] {
            assert!(p.is_multivalue(), "{p:?}: must be multi-value");
        }
        for p in [
            Placeholder::Path,
            Placeholder::Relative,
            Placeholder::Anchor,
            Placeholder::Watch,
            Placeholder::Parent,
            Placeholder::Time,
        ] {
            assert!(!p.is_multivalue(), "{p:?}: must not be multi-value");
        }
    }

    /// `Placeholder::is_diff_derived` covers only the five diff entries.
    /// `Excluded` is multi-value but sourced from `Profile.exclude_strings`,
    /// not from a `Diff` — keeping it out of the predicate prevents the
    /// `Sub.needs_diff` derivation from falsely ratcheting.
    #[test]
    fn placeholder_is_diff_derived_excludes_excluded() {
        for p in [
            Placeholder::Created,
            Placeholder::Deleted,
            Placeholder::Modified,
            Placeholder::RenamedFrom,
            Placeholder::RenamedTo,
        ] {
            assert!(p.is_diff_derived(), "{p:?}: must be diff-derived");
        }
        for p in [
            Placeholder::Path,
            Placeholder::Relative,
            Placeholder::Anchor,
            Placeholder::Watch,
            Placeholder::Parent,
            Placeholder::Time,
            Placeholder::Excluded,
        ] {
            assert!(!p.is_diff_derived(), "{p:?}: must not be diff-derived");
        }
    }
}
