//! `Effect` and friends.
//!
//! No `baseline_snapshot` / `captured_current` on `Effect`: the
//! Engine re-probes after `EffectComplete::Ok` rather than trust a
//! snapshot taken at emission time. The `diff` field is populated only
//! when the Sub's plan references diff-derived placeholders or the
//! Sub's scope is `PerStableFile`; otherwise `None`.

use crate::diff::Diff;
use crate::ids::{ProfileId, ResourceId, SubId};
use crate::resource::ResourceKind;
use crate::sub::ActionPlan;
use compact_str::CompactString;
use std::path::Path;
use std::sync::Arc;

/// Resolved command (substitution output). Constructed by the actuator's
/// resolver immediately before spawn, from the substitution-domain
/// projection carried on [`Effect`].
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct CommandResolved {
    pub argv: Vec<String>,
}

/// Effect — a command-to-be plus engine bookkeeping.
///
/// **Coalescing identity + bookkeeping.**
/// `key` drives `DedupKey`-based coalescing; `forced` mirrors
/// `Burst.forced` at emission time (every Standard burst Effect carries
/// the deadline-crossed flag, regardless of whether the eventual probe
/// verdict was stable). `diff` is `Some` iff `sub.needs_diff` AND the
/// diff source (a `baseline` snapshot) was present.
///
/// `target` is the Resource this Effect addresses — the anchor directory
/// for `DedupKey::Subtree`, or the file resource for `DedupKey::PerFile`
/// (where it duplicates `key.resource` by construction). Captured at
/// emission time; the pair `(self.key.sub(), self.target)` is the
/// total-ordered sort key for [`crate::output::StepOutput::effects`]
/// applied by [`crate::StepOutput::sort_for_emission`]. Carried on the
/// Effect rather than derived from a Profile lookup at sort time: a
/// frozen value survives any state churn between `emit_effects` and
/// `sort_for_emission`.
///
/// `capture_output` mirrors the Sub's `log_output` at emission time. The
/// actuator reads it to choose between `Stdio::null()` (the default —
/// child output is discarded) and `Stdio::inherit()` (child output is
/// forwarded to Specter's own stdout/stderr, where the supervisor's
/// log facility — systemd journal, launchd `StandardOutPath`, FreeBSD
/// `daemon -o` — captures it).
///
/// **Substitution-domain projection of `(Sub, Profile, Tree)`.**
/// The remaining fields are everything the actuator-side resolver reads
/// to render argv + env + cwd at spawn time. Frozen at emit time;
/// consumed at spawn time. Flat (not nested in an `EffectContext`)
/// because there is no second consumer for the group, and a name for a
/// group of fields that has no second consumer is overhead.
///
/// - `sub_name` — `${specter.watch}` substitute and `SPECTER_WATCH` env
///   value. Owned `CompactString` rather than `Arc<str>` so the resolver
///   reaches it via `Deref<Target = str>` without naming the type.
/// - `plan` — the parsed action plan, Arc-cloned from `Sub.plan` at emit
///   time so coalesced Effects share one allocation. Validation
///   guarantees at least one `Action::Exec` step for v1; the actuator
///   walks the steps in order, stopping on the first non-`Ok` outcome.
/// - `anchor_path`, `anchor_kind` — the anchor's filesystem path and
///   classification. `anchor_path` is `Arc<Path>` so the engine builds
///   it once per `emit_effects` call and every Effect emitted from that
///   call (one per Sub × Diff entry) Arc-clones the same allocation.
///   The actuator computes `cwd` from these via
///   `compute_cwd(anchor_path, anchor_kind)`. `anchor_kind` is one byte;
///   carrying it lets the actuator pick the correct cwd shape (parent
///   dir for File anchors, the path itself for Dir / Unknown) without a
///   round-trip to the engine.
/// - `target_relative` — `${specter.relative}` substitute and the
///   per-entry segment used by the resolver to derive `target_path`
///   (`${specter.path}` / `SPECTER_PATH`) at spawn time. Empty for `DedupKey::Subtree`
///   (target_path == anchor_path); the file segment for
///   `DedupKey::PerFile` (target_path == anchor_path.join(segment)).
///   Carrying only the relative — not the joined path — defers the
///   `PathBuf` allocation to the spawn boundary, where Latest-coalesce
///   has already filtered Effects that won't reach a syscall.
/// - `exclude` — Arc-clone of `Profile.exclude_strings`. Carried so
///   the resolver can render the `${specter.excluded}` placeholder and
///   `SPECTER_EXCLUDED` env value without a back-channel to the engine.
///
/// `SPECTER_EVENT_KIND` (`dir-subtree` vs `file`) and the resolver-side
/// dispatch on scope are derived from `key`'s variant — no separate
/// scope field is carried. `key` already partitions Effects into
/// `Subtree` and `PerFile` arms by construction; storing the scope a
/// second time invites drift.
#[derive(Clone, Debug)]
pub struct Effect {
    pub key: DedupKey,
    pub target: ResourceId,
    pub forced: bool,
    pub correlation: CorrelationId,
    pub diff: Option<Arc<Diff>>,
    pub capture_output: bool,

    pub sub_name: CompactString,
    pub plan: Arc<ActionPlan>,
    pub anchor_path: Arc<Path>,
    pub anchor_kind: ResourceKind,
    pub target_relative: CompactString,
    pub exclude: Arc<[CompactString]>,
}

/// Per-Effect correlation token. Engine-monotonic in v1.
#[derive(Debug, Default, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct CorrelationId(pub u64);

/// Coalescing identity.
///
/// Both variants carry the owning Profile. The `profile` field on
/// `PerFile` adds no partitioning power (the `sub` already determines
/// the Profile), but it makes the `key → profile` lookup constant-time
/// symmetrically across both arms — the engine credits the per-Profile
/// `BurstPhase::Awaiting` counter on every `EffectComplete`, so this
/// lookup is hot.
///
/// `Ord` drives the actuator's `BTreeMap<DedupKey, Slot>` and the
/// engine's `BTreeSet<DedupKey>` (`Profile::fired_subs`).
/// `Hash` is intentionally not derived — no `HashMap`/`HashSet` keys on
/// this type and `core` bans `hashbrown` outright.
#[derive(Debug, Clone, Eq, PartialEq, Ord, PartialOrd)]
pub enum DedupKey {
    PerFile {
        sub: SubId,
        profile: ProfileId,
        resource: ResourceId,
    },
    Subtree {
        sub: SubId,
        profile: ProfileId,
    },
}

impl DedupKey {
    /// The Profile that owns this key's emission record. Both variants
    /// carry the field; the match is exhaustive and `const`.
    #[must_use]
    pub const fn profile(&self) -> ProfileId {
        match *self {
            Self::PerFile { profile, .. } | Self::Subtree { profile, .. } => profile,
        }
    }

    /// The Sub that emitted this key. Both variants carry the field;
    /// callers needing the `(sub, target)` sort key for
    /// [`crate::StepOutput::sort_for_emission`] reach through here
    /// rather than re-implementing the match.
    #[must_use]
    pub const fn sub(&self) -> SubId {
        match *self {
            Self::PerFile { sub, .. } | Self::Subtree { sub, .. } => sub,
        }
    }
}

impl Default for DedupKey {
    /// Null `DedupKey` for test fodder — used by tests that synthesize
    /// `Input::EffectComplete` with a Sub that is not in the registry
    /// (engine emits `EffectCompleteForUnknownSub` and drops). Not a
    /// `SmallVec` sentinel: `DedupKey` is stored inside `Effect`, not
    /// directly in `StepOutput.effects`.
    fn default() -> Self {
        Self::Subtree {
            sub: SubId::default(),
            profile: ProfileId::default(),
        }
    }
}

#[derive(Debug, Default, Clone, Eq, PartialEq, Hash)]
pub enum EffectOutcome {
    #[default]
    Ok,
    Failed {
        exit_code: Option<i32>,
        signal: Option<i32>,
    },
}

#[cfg(test)]
mod dedup_key_tests {
    use super::DedupKey;
    use crate::ids::{ProfileId, ResourceId, SubId};
    use slotmap::KeyData;

    #[test]
    fn profile_returns_owning_profile_for_both_variants() {
        let p = ProfileId::from(KeyData::from_ffi(7));
        let s = SubId::from(KeyData::from_ffi(11));
        let r = ResourceId::from(KeyData::from_ffi(13));
        let perfile = DedupKey::PerFile {
            sub: s,
            profile: p,
            resource: r,
        };
        let subtree = DedupKey::Subtree { sub: s, profile: p };
        assert_eq!(
            perfile.profile(),
            p,
            "PerFile.profile() returns the owning Profile",
        );
        assert_eq!(
            subtree.profile(),
            p,
            "Subtree.profile() returns the owning Profile",
        );
    }
}
