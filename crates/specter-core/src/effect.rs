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
#[derive(Debug, Clone, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub enum DedupKey {
    PerFile { sub: SubId, resource: ResourceId },
    Subtree { sub: SubId, profile: ProfileId },
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
