//! `Effect` and friends.
//!
//! No `baseline_snapshot` / `captured_current` on `Effect`: the
//! Engine re-probes after `EffectComplete::Ok` rather than trust a
//! snapshot taken at emission time. The `diff` field is populated only when
//! the Sub's command template references diff entries (or scope is
//! `PerStableFile`); otherwise `None`.

pub mod resolve;

#[cfg(test)]
mod tests;

use crate::diff::Diff;
use crate::ids::{ProfileId, ResourceId, SubId};
use std::path::PathBuf;
use std::sync::Arc;

/// Resolved command (substitution output). The data shape ships here;
/// the resolver that turns `(CommandTemplate, Sub, Profile, Tree, Diff)`
/// into argv strings lives next to the Actuator.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct CommandResolved {
    pub argv: Vec<String>,
}

/// Effect — a command ready for the Actuator.
///
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
#[derive(Clone, Debug)]
pub struct Effect {
    pub key: DedupKey,
    pub target: ResourceId,
    pub command: CommandResolved,
    pub env: Vec<(String, String)>,
    pub cwd: PathBuf,
    pub forced: bool,
    pub correlation: CorrelationId,
    pub diff: Option<Arc<Diff>>,
    pub capture_output: bool,
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
/// engine's `BTreeMap<DedupKey, u128>` (`Profile::last_emitted_dir_hash`).
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
