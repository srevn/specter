//! `Effect` and friends.
//!
//! No baseline/current snapshot on `Effect`: the engine re-probes after
//! `EffectComplete::Ok` rather than trust a snapshot taken at emission
//! time. A diff is carried only when the Sub's program references
//! diff-derived placeholders (Subtree, optional) or the fire is
//! per-stable-file (mandatory).

use crate::diff::Diff;
use crate::ids::{CorrelationId, ProfileId, ResourceId, SubId};
use crate::program::ActionProgram;
use crate::resource::ResourceKind;
use compact_str::CompactString;
use std::borrow::Cow;
use std::path::Path;
use std::sync::Arc;

/// Effect — a command-to-be plus the bookkeeping needed to spawn and
/// coalesce it.
///
/// The flat fields are irreducible identity scalars (frozen at emit
/// time; `(sub, profile, anchor)` survives any post-emit state churn)
/// plus the substitution payload the actuator-side resolver reads to
/// render argv/env/cwd. The fire *shape* — whole-subtree vs
/// per-stable-file — is the [`EffectTarget`] sum. Every cross-cutting
/// concern ([`key`](Effect::key), [`sort_key`](Effect::sort_key),
/// [`target_path`](Effect::target_path), [`diff`](Effect::diff),
/// [`relative`](Effect::relative)) is a derived method, never a stored
/// field, so a stored projection cannot drift from the shape.
///
/// `capture_output` mirrors the Sub's `log_output` at emit time: the
/// actuator picks `Stdio::null()` (discard) vs `Stdio::inherit()`
/// (forward to Specter's stdout/stderr, where the supervisor's log
/// facility captures it). `program` / `anchor_path` / `exclude` are
/// `Arc`-shared so coalesced Effects from one emit call reuse one
/// allocation each.
#[derive(Clone, Debug)]
pub struct Effect {
    pub sub: SubId,
    pub profile: ProfileId,
    /// The Profile's anchor resource (frozen `Profile.resource`). Not
    /// derivable from [`Effect::key`] — `DedupKey::Subtree` does not
    /// carry it — so it is an irreducible identity scalar.
    pub anchor: ResourceId,
    /// Operator-narration only — not part of the engine↔actuator
    /// completion-routing contract. The engine resolves a
    /// [`crate::Input::EffectComplete`] back to its Profile via
    /// `DedupKey::profile()` (in `on_effect_complete`'s pass-1 route)
    /// and the actuator coalesces by [`DedupKey`] alone; this id is
    /// consumed only for the `SPECTER_CORRELATION` env, the diff tmp
    /// filename, and tracing keys. The engine mints it because the
    /// monotone floor already lives at engine scope
    /// (`Engine.effect_correlations`); moving the mint actuator-side
    /// would buy nothing — the same operator-narration id with one
    /// extra cross-actor hop.
    pub correlation: CorrelationId,
    pub forced: bool,
    pub capture_output: bool,
    pub sub_name: CompactString,
    pub program: Arc<ActionProgram>,
    pub anchor_path: Arc<Path>,
    pub anchor_kind: ResourceKind,
    pub exclude: Arc<[CompactString]>,
    pub target: EffectTarget,
}

/// The fire shape of an [`Effect`].
///
/// Named `EffectTarget` (not `EffectScope` — that is `sub.rs`'s
/// user-intent axis) so the two Subtree/PerFile vocabularies stay
/// distinct. Carries the only fields whose meaning differs per shape;
/// shared identity/payload stays flat on [`Effect`].
#[derive(Clone, Debug)]
pub enum EffectTarget {
    /// Whole-subtree fire. `target_path == anchor_path`. `diff` is
    /// `Some` iff the Sub needs a diff-derived placeholder and a
    /// baseline existed.
    Subtree { diff: Option<Arc<Diff>> },
    /// Per-stable-file fire. `target_path == anchor_path.join(segment)`.
    /// `diff` is mandatory: the type guarantees what
    /// `PerStableFile ⇒ needs_diff` previously enforced by convention.
    PerFile {
        resource: ResourceId,
        segment: CompactString,
        diff: Arc<Diff>,
    },
}

/// Constructor input for [`Effect`].
///
/// Destructured into the flat identity/payload fields, never stored.
/// Shared by both engine emit arms and every test fixture; the multiple
/// consumers earn it a name (a stored field group with no second
/// consumer would not).
#[derive(Debug)]
pub struct EffectCommon {
    pub sub: SubId,
    pub profile: ProfileId,
    pub anchor: ResourceId,
    pub correlation: CorrelationId,
    pub forced: bool,
    pub capture_output: bool,
    pub sub_name: CompactString,
    pub program: Arc<ActionProgram>,
    pub anchor_path: Arc<Path>,
    pub anchor_kind: ResourceKind,
    pub exclude: Arc<[CompactString]>,
}

impl Effect {
    /// Whole-subtree Effect. `diff` is `Some` iff the Sub needs a
    /// diff-derived placeholder and a baseline existed.
    #[must_use]
    pub fn subtree(common: EffectCommon, diff: Option<Arc<Diff>>) -> Self {
        Self::from_common(common, EffectTarget::Subtree { diff })
    }

    /// Per-stable-file Effect. `diff` is mandatory.
    #[must_use]
    pub fn per_file(
        common: EffectCommon,
        resource: ResourceId,
        segment: CompactString,
        diff: Arc<Diff>,
    ) -> Self {
        Self::from_common(
            common,
            EffectTarget::PerFile {
                resource,
                segment,
                diff,
            },
        )
    }

    /// Single construction choke: destructure the parameter struct into
    /// the flat fields and attach the shape.
    fn from_common(common: EffectCommon, target: EffectTarget) -> Self {
        Self {
            sub: common.sub,
            profile: common.profile,
            anchor: common.anchor,
            correlation: common.correlation,
            forced: common.forced,
            capture_output: common.capture_output,
            sub_name: common.sub_name,
            program: common.program,
            anchor_path: common.anchor_path,
            anchor_kind: common.anchor_kind,
            exclude: common.exclude,
            target,
        }
    }

    /// Coalescing identity — the actuator's `BTreeMap<DedupKey, Slot>`.
    /// The engine's fire-history is per-Sub ([`crate::Sub::has_fired`]),
    /// not a projection of this key. Slotmap keys are `Copy`, so this
    /// is cheap; callers need it owned anyway.
    #[must_use]
    pub const fn key(&self) -> DedupKey {
        match &self.target {
            EffectTarget::Subtree { .. } => DedupKey::Subtree {
                sub: self.sub,
                profile: self.profile,
            },
            EffectTarget::PerFile { resource, .. } => DedupKey::PerFile {
                sub: self.sub,
                profile: self.profile,
                resource: *resource,
            },
        }
    }

    /// Total order for [`crate::output::StepOutput`] effects: Subtree
    /// keys on the anchor resource, PerFile on the file resource.
    /// Replay determinism depends on this being stable.
    ///
    /// No `ProfileId` in the tuple. `SubId` already determines the
    /// Profile (`Sub.profile` is functional), so adding it cannot
    /// refine the partition. [`DedupKey::PerFile`] carries it for an
    /// unrelated reason — that key doubles as the actuator's
    /// per-Profile completion-credit lookup.
    #[must_use]
    pub const fn sort_key(&self) -> (SubId, ResourceId) {
        let resource = match &self.target {
            EffectTarget::Subtree { .. } => self.anchor,
            EffectTarget::PerFile { resource, .. } => *resource,
        };
        (self.sub, resource)
    }

    /// Spawn `target_path`. Subtree borrows `anchor_path` (no alloc);
    /// PerFile joins the segment at call time, keeping the `PathBuf`
    /// allocation at the resolve boundary (post-coalesce).
    #[must_use]
    pub fn target_path(&self) -> Cow<'_, Path> {
        match &self.target {
            EffectTarget::Subtree { .. } => Cow::Borrowed(&*self.anchor_path),
            EffectTarget::PerFile { segment, .. } => {
                Cow::Owned(self.anchor_path.join(segment.as_str()))
            }
        }
    }

    /// `${specter.relative}` / `SPECTER_RELATIVE_PATH` source: empty for
    /// Subtree, the file segment for PerFile.
    #[must_use]
    pub fn relative(&self) -> &str {
        match &self.target {
            EffectTarget::Subtree { .. } => "",
            EffectTarget::PerFile { segment, .. } => segment.as_str(),
        }
    }

    /// Uniform diff access: PerFile always `Some`, Subtree conditional.
    #[must_use]
    pub const fn diff(&self) -> Option<&Arc<Diff>> {
        match &self.target {
            EffectTarget::Subtree { diff } => diff.as_ref(),
            EffectTarget::PerFile { diff, .. } => Some(diff),
        }
    }
}

/// Coalescing identity.
///
/// Both variants carry the owning Profile. The `profile` field on
/// `PerFile` adds no partitioning power (the `sub` already determines
/// the Profile), but it makes the `key → profile` lookup constant-time
/// symmetrically across both arms — the engine credits the per-Profile
/// `PostFirePhase::Awaiting` counter on every `EffectComplete`, so this
/// lookup is hot.
///
/// `Ord` drives the actuator's `BTreeMap<DedupKey, Slot>`. The engine's
/// fire-history is per-Sub ([`crate::Sub::has_fired`]) — not this type
/// nor any projection of it. `Hash` is intentionally not derived — the
/// total order above is the load-bearing key shape (sorted iteration is
/// the contract for replay), so a `HashMap`-shaped lookup would be a
/// strictly weaker substitute that nothing in the engine asks for.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Ord, PartialOrd)]
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

/// Terminal outcome of an Effect's plan at the engine boundary.
///
/// The engine's v1 policy is Ignore — it discriminates only `Ok` vs
/// `Failed`. The [`Termination`] payload is diagnostic (logging) and
/// drives the actuator's internal pipe re-aggregation; it is not a
/// routing input.
///
/// `Hash` is intentionally not derived — outcomes are consumed by name
/// (`Ok` vs `Failed`) at the engine's effect-completion dispatcher; no
/// caller keys a map by an outcome, so a `Hash` impl would be dead
/// surface.
#[derive(Debug, Default, Clone, Eq, PartialEq)]
pub enum EffectOutcome {
    #[default]
    Ok,
    Failed(Termination),
}

/// Why a plan terminated unsuccessfully. The four variants are exactly
/// the four reachable `(exit_code, signal)` shapes — a total, named
/// encoding, not a state-space change.
///
/// `Hash` is intentionally not derived — variants are consumed by name
/// for diagnostic formatting; no caller keys a map by a termination
/// payload, so a `Hash` impl would be dead surface.
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum Termination {
    /// Resolver/spawn failure, waiter panic, or a synthesised plan
    /// outcome — no exit code and no signal.
    Internal,
    /// Clean non-zero exit: a single process, or a pipe with no
    /// signalled stage.
    Exit(i32),
    /// Killed by signal: a single process, or a pipe with no non-zero
    /// exit.
    Signal(i32),
    /// A pipe where one stage exited non-zero and another was
    /// signalled (last non-zero exit, first observed signal).
    PipeMixed { last_exit: i32, first_signal: i32 },
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
