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
#[derive(Clone, Debug)]
pub struct Effect {
    pub key: DedupKey,
    pub command: CommandResolved,
    pub env: Vec<(String, String)>,
    pub cwd: PathBuf,
    pub forced: bool,
    pub correlation: CorrelationId,
    pub diff: Option<Arc<Diff>>,
}

impl Default for Effect {
    /// Sentinel for `tinyvec::Array`'s `T: Default` bound on
    /// `StepOutput.effects`. Inline slots are overwritten before they're
    /// ever read.
    fn default() -> Self {
        Self {
            key: DedupKey::default(),
            command: CommandResolved::default(),
            env: Vec::new(),
            cwd: PathBuf::new(),
            forced: false,
            correlation: CorrelationId::default(),
            diff: None,
        }
    }
}

/// Per-Effect correlation token. Engine-monotonic in v1.
#[derive(Debug, Default, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct CorrelationId(pub u64);

/// Coalescing identity.
///
/// Both variants carry the owning Profile. `PerFile` was originally keyed by
/// `(sub, resource)` alone — `sub` already determines `profile`, so the field
/// adds no partitioning power, but makes the `key → profile` lookup constant-
/// time symmetrically across both arms. The Phase 09 fire-cycle work needs
/// that lookup at every `EffectComplete` to credit the per-Profile counter
/// in `BurstPhase::Awaiting`.
#[derive(Debug, Clone, Eq, PartialEq, Hash, Ord, PartialOrd)]
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
}

impl Default for DedupKey {
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
        let subtree = DedupKey::Subtree {
            sub: s,
            profile: p,
        };
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
