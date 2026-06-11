//! `StepOutput` — the engine's per-step buffer.
//!
//! Five streams ride one value; shapes and seal mechanisms differ because the consumers and roles
//! differ:
//!
//! - **`watch_ops`** — `pub SmallVec<[WatchOp; 2]>`. Sorted by [`WatchOp::resource`]; resealed at
//!   the boundary by [`StepOutput::sort_for_emission`].
//! - **`effects`** — private [`SortedEffects`] (`SmallVec`). Sorted by [`Effect::sort_key`];
//!   resealed at the boundary by [`StepOutput::sort_for_emission`].
//! - **`probe_ops`** — private [`ProbeOps`] (`BTreeMap`). Order is intrinsic to the [`ProfileId`]
//!   key; sealed structurally — a per-owner last-writer-wins upsert ([`StepOutput::push_probe_op`])
//!   makes "at most one op per owner, in owner order" unrepresentable to violate.
//! - **`cancel_effects`** — private [`CancelEffects`] (`BTreeSet`). Per-profile dedup set; order
//!   intrinsic to the [`ProfileId`] key. Engine emits at most one Cancel per profile per step
//!   ([`StepOutput::push_cancel_effect`]); set semantics make the structural invariant
//!   unrepresentable to violate.
//! - **`diagnostics`** — `pub SmallVec<[Diagnostic; 2]>`. Insertion order, intentionally unsealed:
//!   operator-readable, not part of the determinism contract.
//!
//! Probe **correctness** is the engine's response gate, not this stream's. A stale or superseded
//! response folds to `StaleProbeResponse` regardless of op order; the per-owner shape mirrors the
//! sensor's `expected` map only to spare the missed- syscall stall an unstructured stream could
//! induce.
//!
//! Residual discipline: every `StepOutput`-returning entry point calls
//! [`StepOutput::sort_for_emission`] before returning. Pinned at the engine boundary by
//! `tests/integration.rs` — `step_output_is_sorted` and
//! `cancel_all_in_flight_probes_returns_sealed_output`.

use crate::diag::Diagnostic;
use crate::effect::Effect;
use crate::ids::ProfileId;
use crate::op::{ProbeOp, WatchOp};
use smallvec::SmallVec;
use std::collections::{BTreeMap, BTreeSet};

/// The `effects` stream in emission order.
///
/// Append-only from outside the module ([`StepOutput::push_effect`]); the order-establishing
/// `reseal` and the unsorted `push_unsorted` are module-private, so the only public view
/// ([`std::ops::Deref`] to `&[Effect]`) is always the sorted one.
#[derive(Clone, Debug, Default)]
pub struct SortedEffects(SmallVec<[Effect; 2]>);

impl SortedEffects {
    /// Append without re-establishing order. The companion [`Self::reseal`] restores the invariant
    /// before the value escapes.
    fn push_unsorted(&mut self, e: Effect) {
        self.0.push(e);
    }

    /// Restore the emission order ([`Effect::sort_key`]).
    fn reseal(&mut self) {
        self.0.sort_by_key(Effect::sort_key);
    }
}

impl std::ops::Deref for SortedEffects {
    type Target = [Effect];
    fn deref(&self) -> &[Effect] {
        &self.0
    }
}

/// The `probe_ops` stream as a sealed per-owner map.
///
/// Isomorphic to the sensor's `WorkerProber.expected: BTreeMap<ProfileId, ProbeCorrelation>`
/// (`submit` inserts, `cancel` removes, both owner-keyed). Producer and consumer share one shape;
/// `BTreeMap<ProfileId, ProbeOp>` is type-honest, not a dedup tactic.
///
/// Append-only from outside the module ([`StepOutput::push_probe_op`], a per-owner upsert); read
/// via [`ProbeOps::iter`] or, at the terminal drain, [`StepOutput::into_parts`]. An empty map does
/// not allocate (the common case), so the sealed shape is lighter on the hot `StepOutput` than
/// always-resident inline slots would be.
#[derive(Clone, Debug, Default)]
pub struct ProbeOps(BTreeMap<ProfileId, ProbeOp>);

impl ProbeOps {
    /// Record `op`, replacing any prior op for the same owner (last-writer-wins — see type
    /// rustdoc). Module-private: external writes go through [`StepOutput::push_probe_op`].
    fn upsert(&mut self, op: ProbeOp) {
        self.0.insert(op.owner(), op);
    }

    /// The ops in [`ProfileId`] order.
    #[must_use]
    pub fn iter(&self) -> impl DoubleEndedIterator<Item = &ProbeOp> {
        self.0.values()
    }

    /// `true` iff no op was recorded this step (the common case).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Number of distinct owners with a recorded op — equivalently, the op count, since the map
    /// holds at most one per owner.
    #[must_use]
    pub fn len(&self) -> usize {
        self.0.len()
    }
}

/// The `cancel_effects` stream as a sealed per-profile dedup set.
///
/// Engine→actuator cancel emissions are profile-scoped — the actuator fans the cancel across every
/// slot whose [`crate::DedupKey::profile`] matches — so the wire identity is the [`ProfileId`]
/// alone. Set semantics are structurally sufficient: the only emission site is
/// `handle_gate_deadline`, a once-per-burst edge, so two cancels for one profile in one step would
/// mean two gate-deadline timers fired in one tick — impossible by construction.
///
/// `BTreeSet<ProfileId>` matches the discipline of [`ProbeOps`] (owner-keyed map): order is
/// intrinsic to the key, no [`StepOutput::sort_for_emission`] work. Append-only from outside the
/// module ([`StepOutput::push_cancel_effect`]); read via [`Self::iter`] or, at the terminal drain,
/// [`StepOutput::into_parts`]. An empty set does not allocate (the common case — most steps emit no
/// cancel), so the sealed shape is lighter on the hot `StepOutput` than an always-resident inline
/// slot would be.
#[derive(Clone, Debug, Default)]
pub struct CancelEffects(BTreeSet<ProfileId>);

impl CancelEffects {
    /// Insert `profile` into the dedup set. Module-private: external writes go through
    /// [`StepOutput::push_cancel_effect`].
    fn insert(&mut self, profile: ProfileId) {
        self.0.insert(profile);
    }

    /// The profiles in [`ProfileId`] order. Iterates by-value (the id is `Copy`), mirroring
    /// [`BTreeSet::iter`]'s `.copied()` shape without forcing the caller to chain it.
    #[must_use]
    pub fn iter(&self) -> impl DoubleEndedIterator<Item = ProfileId> + '_ {
        self.0.iter().copied()
    }

    /// `true` iff no cancel was emitted this step (the common case).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Number of distinct profiles to cancel — equivalently, the cancel count, since the set holds
    /// at most one per profile.
    #[must_use]
    pub fn len(&self) -> usize {
        self.0.len()
    }
}

/// The five owned streams yielded by [`StepOutput::into_parts`], in dispatch order: watch ops,
/// probe ops (owner-keyed), effects (emission-sorted), cancel-effects (profile-keyed), diagnostics.
///
/// Cancel-effects sit between effects and diagnostics so the bin's `forward` can dispatch cancels
/// first then submits in one pass over the tuple — "kill stale before spawn new" is the right
/// ordering (defense in depth: a same-step cancel + submit for one profile cannot construct in
/// production, but if it ever did, this ordering is the correctness contract).
pub type StepOutputParts = (
    SmallVec<[WatchOp; 2]>,
    BTreeMap<ProfileId, ProbeOp>,
    SmallVec<[Effect; 2]>,
    BTreeSet<ProfileId>,
    SmallVec<[Diagnostic; 2]>,
);

#[derive(Debug, Default, Clone)]
pub struct StepOutput {
    pub watch_ops: SmallVec<[WatchOp; 2]>,
    probe_ops: ProbeOps,
    effects: SortedEffects,
    cancel_effects: CancelEffects,
    pub diagnostics: SmallVec<[Diagnostic; 2]>,
}

impl StepOutput {
    /// Append an [`Effect`] in unsorted (insertion) order. [`Self::sort_for_emission`]
    /// re-establishes the contract before the value is handed downstream.
    pub fn push_effect(&mut self, e: Effect) {
        self.effects.push_unsorted(e);
    }

    /// The emitted effects, always in sort order.
    #[must_use]
    pub const fn effects(&self) -> &SortedEffects {
        &self.effects
    }

    /// Record a [`ProbeOp`], replacing any prior op for the same owner (last-writer-wins — see
    /// [`ProbeOps`]). The `probe_ops` analogue of [`Self::push_effect`].
    ///
    /// The upsert silently swallows a `Cancel` when a `Probe` for the same owner lands in the same
    /// step — the anchor-loss wrapper cancels a mid-Verifying walk and emits the descent probe in
    /// one step, and the overflow reseed has the same shape. That is sound because probe submission
    /// *is* displacement: `WorkerProber::submit` (specter-sensor) overwrites the owner's expected
    /// correlation, so the superseded walk's late result drops at the pool exactly as an explicitly
    /// cancelled one would. A prober that queued per-owner instead of displacing would need the
    /// Cancel preserved — revisit this map's semantics before changing that contract.
    pub fn push_probe_op(&mut self, op: ProbeOp) {
        self.probe_ops.upsert(op);
    }

    /// The emitted probe ops, owner-keyed with at most one per owner.
    #[must_use]
    pub const fn probe_ops(&self) -> &ProbeOps {
        &self.probe_ops
    }

    /// Record a cancel-effect for `profile`. Idempotent (set semantics): repeated calls in one step
    /// coalesce to one emission — by construction, only one gate-deadline can fire per profile per
    /// step, so this guard is structural rather than load-bearing.
    pub fn push_cancel_effect(&mut self, profile: ProfileId) {
        self.cancel_effects.insert(profile);
    }

    /// The cancel-effects emitted this step, profile-keyed with at most one per profile. The
    /// `cancel_effects` analogue of [`Self::probe_ops`].
    #[must_use]
    pub const fn cancel_effects(&self) -> &CancelEffects {
        &self.cancel_effects
    }

    /// Consume into the five owned streams, effects in emission order, probe ops owner-keyed,
    /// cancel-effects profile-keyed.
    ///
    /// Terminal-consumer drain: `self` is owned, so this is only reachable on a fully built,
    /// already-resealed value (every `StepOutput`-returning entry point reseals before returning).
    /// Zero-copy — every stream moves out.
    #[must_use]
    pub fn into_parts(self) -> StepOutputParts {
        (
            self.watch_ops,
            self.probe_ops.0,
            self.effects.0,
            self.cancel_effects.0,
            self.diagnostics,
        )
    }

    /// Seal the order-determined streams: `watch_ops` by [`WatchOp::resource`], `effects` by
    /// [`Effect::sort_key`]. `probe_ops` and `cancel_effects` are key-ordered by construction (no
    /// reseal); `diagnostics` follow insertion order (not part of the contract).
    pub fn sort_for_emission(&mut self) {
        self.watch_ops.sort_by_key(WatchOp::resource);
        self.effects.reseal();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diff::Diff;
    use crate::effect::{DedupKey, EffectCommon};
    use crate::ids::{CorrelationId, ProbeCorrelation, ProfileId, ResourceId, SubId};
    use crate::op::ProbeRequest;
    use crate::program::{
        ActionProgram, ArgPart, ArgTemplate, BranchTarget, ExecAction, ProgramBuilder, SpawnBody,
    };
    use crate::resource::ResourceKind;
    use crate::sub::ClassSet;
    use compact_str::CompactString;
    use slotmap::KeyData;
    use std::path::PathBuf;
    use std::sync::Arc;

    fn rid(n: u64) -> ResourceId {
        ResourceId::from(KeyData::from_ffi(n))
    }

    fn pidn(n: u64) -> ProfileId {
        ProfileId::from(KeyData::from_ffi(n))
    }

    fn sidn(n: u64) -> SubId {
        SubId::from(KeyData::from_ffi(n))
    }

    /// Minimal one-op program for fixture construction; the body content doesn't matter for the
    /// sort-emission tests below, only that the field is well-typed and non-empty.
    fn fixture_program() -> Arc<ActionProgram> {
        let mut b = ProgramBuilder::new();
        let h = b.emit(SpawnBody::Exec(ExecAction::new(
            [ArgTemplate::new([ArgPart::literal("/bin/true")])],
            None,
        )));
        b.patch_on_ok(h, BranchTarget::Escape).unwrap();
        b.patch_on_failed(h, BranchTarget::Terminate).unwrap();
        Arc::new(b.build().unwrap())
    }

    /// Test fixture for an [`Effect`] when only `(key, target)` matter (sort tests). Other fields
    /// take their natural empty values. `anchor` doubles as the Subtree sort resource: setting it
    /// to `target` keeps `sort_key().1 == target` for the Subtree arm. The PerFile arm keys on its
    /// own `resource`, which the test always passes equal to `target`.
    fn effect(key: DedupKey, target: ResourceId) -> Effect {
        let common = EffectCommon {
            sub: key.sub(),
            profile: key.profile(),
            anchor: target,
            correlation: CorrelationId::default(),
            forced: false,
            capture_output: false,
            sub_name: CompactString::new(""),
            program: fixture_program(),
            anchor_path: Arc::from(PathBuf::new()),
            anchor_kind: ResourceKind::Dir,
            exclude: Arc::from(Vec::<CompactString>::new()),
        };
        match key {
            DedupKey::Subtree { .. } => Effect::subtree(common, None::<Arc<Diff>>),
            DedupKey::PerFile { resource, .. } => Effect::per_file(
                common,
                resource,
                CompactString::new(""),
                Arc::new(Diff::default()),
            ),
        }
    }

    #[test]
    fn sort_for_emission_orders_watch_ops_by_resource_id() {
        let r1 = rid(1);
        let r2 = rid(2);
        let r3 = rid(3);
        let mut out = StepOutput::default();
        out.watch_ops.push(WatchOp::Unwatch { resource: r3 });
        out.watch_ops.push(WatchOp::Watch {
            resource: r1,
            path: Arc::from(std::path::Path::new("/x")),
            kind: ResourceKind::Unknown,
            events: ClassSet::EMPTY,
        });
        out.watch_ops.push(WatchOp::Unwatch { resource: r2 });

        out.sort_for_emission();

        let resources: Vec<ResourceId> = out.watch_ops.iter().map(WatchOp::resource).collect();
        assert_eq!(resources, vec![r1, r2, r3]);
    }

    /// [`ProbeOps`] iterates in [`ProfileId`] order *intrinsically* — the order is the map key,
    /// established at insert, not by a sort. Push out of owner order and assert iteration order
    /// *before* any `sort_for_emission`; the seal call then leaves it unchanged (it is a no-op for
    /// `probe_ops`).
    #[test]
    fn probe_ops_iterate_in_owner_order_without_a_sort() {
        let p1 = pidn(1);
        let p2 = pidn(2);
        let mut out = StepOutput::default();
        out.push_probe_op(ProbeOp::Cancel { owner: p2 });
        out.push_probe_op(ProbeOp::Probe {
            request: ProbeRequest::AnchorFile {
                owner: p1,
                correlation: ProbeCorrelation::from(7),
                target_path: Arc::from(std::path::Path::new("/y")),
            },
        });

        let owners = |o: &StepOutput| -> Vec<ProfileId> {
            o.probe_ops().iter().map(ProbeOp::owner).collect()
        };
        let expected = vec![p1, p2];
        // Owner-ordered by construction, before any seal.
        assert_eq!(owners(&out), expected);
        out.sort_for_emission();
        // `sort_for_emission` is a no-op for `probe_ops`.
        assert_eq!(owners(&out), expected);
    }

    /// Last-writer-wins per owner: a second op for an owner *replaces* the first, isomorphic to the
    /// sensor's `expected` map. This is what makes "at most one op per owner" structural — the
    /// violation is unrepresentable, not asserted-against.
    #[test]
    fn push_probe_op_replaces_prior_op_for_same_owner() {
        let p = pidn(1);
        let mut out = StepOutput::default();
        out.push_probe_op(ProbeOp::Probe {
            request: ProbeRequest::AnchorFile {
                owner: p,
                correlation: ProbeCorrelation::from(7),
                target_path: Arc::from(std::path::Path::new("/y")),
            },
        });
        out.push_probe_op(ProbeOp::Cancel { owner: p });

        assert_eq!(out.probe_ops().len(), 1);
        let only = out.probe_ops().iter().next().expect("one op for p");
        assert!(
            matches!(only, ProbeOp::Cancel { owner } if *owner == p),
            "the later Cancel must replace the earlier Probe (last-writer-wins)",
        );
    }

    /// Distinct owners are independent keys: both survive, in owner order. The map is owner-keyed,
    /// not a single slot — the per-owner replace must not collapse different owners.
    #[test]
    fn push_probe_op_keeps_distinct_owners() {
        let p1 = pidn(1);
        let p2 = pidn(2);
        let mut out = StepOutput::default();
        out.push_probe_op(ProbeOp::Cancel { owner: p2 });
        out.push_probe_op(ProbeOp::Probe {
            request: ProbeRequest::AnchorFile {
                owner: p1,
                correlation: ProbeCorrelation::from(7),
                target_path: Arc::from(std::path::Path::new("/y")),
            },
        });

        assert_eq!(out.probe_ops().len(), 2);
        let owners: Vec<ProfileId> = out.probe_ops().iter().map(ProbeOp::owner).collect();
        assert_eq!(owners, vec![p1, p2]);
    }

    /// `sort_for_emission` orders mixed-arm effects by `(sub, target)`. Mixing `PerFile` and
    /// `Subtree` in one step is the production case (a multi-Sub Profile firing both kinds
    /// simultaneously); the ordering interleaves them by sub and target rather than segregating by
    /// variant. The captured `Effect.target` survives any post-emission state churn — sort is
    /// independent of `ProfileMap`.
    #[test]
    fn sort_for_emission_orders_effects_by_sub_then_target() {
        let sub_a = sidn(2);
        let sub_b = sidn(5);
        let prof = pidn(7);
        let r_lo = rid(1);
        let r_hi = rid(9);
        // Push out of order: (sub_b, hi), (sub_a, hi), (sub_a, lo).
        let mut out = StepOutput::default();
        out.push_effect(effect(
            DedupKey::Subtree {
                sub: sub_b,
                profile: prof,
            },
            r_hi,
        ));
        out.push_effect(effect(
            DedupKey::PerFile {
                sub: sub_a,
                profile: prof,
                resource: r_hi,
            },
            r_hi,
        ));
        out.push_effect(effect(
            DedupKey::PerFile {
                sub: sub_a,
                profile: prof,
                resource: r_lo,
            },
            r_lo,
        ));

        out.sort_for_emission();

        let keys: Vec<(SubId, ResourceId)> = out.effects().iter().map(Effect::sort_key).collect();
        assert_eq!(keys, vec![(sub_a, r_lo), (sub_a, r_hi), (sub_b, r_hi)]);
    }
}
