//! `StepOutput` and its determinism contract.
//!
//! `effects` is sealed: the field is private, wrapped in
//! [`SortedEffects`] whose mutators are module-private. Callers append
//! via [`StepOutput::push_effect`] and read via [`StepOutput::effects`]
//! / [`StepOutput::into_parts`], so observing effects in any order
//! other than the emission sort is unrepresentable. The one residual:
//! every `StepOutput`-returning entry point must call
//! [`StepOutput::sort_for_emission`] before returning the value.
//!
//! Not every stream's order carries the same weight. `watch_ops` and
//! `effects` ordering is replay determinism — a reordering changes the
//! log, not the outcome. `probe_ops` ordering is *operational
//! correctness*: the sensor drains it into a shared per-owner
//! expectation map whose `submit` / `cancel` do not commute, so the
//! sort being *exact* for that stream — guaranteed by the
//! at-most-one-`ProbeOp`-per-owner invariant asserted in
//! [`StepOutput::sort_for_emission`] — is load-bearing, not cosmetic.

use crate::diag::Diagnostic;
use crate::effect::Effect;
use crate::op::{ProbeOp, WatchOp};
use smallvec::SmallVec;

/// The `effects` stream in emission order.
///
/// Append-only from outside the module ([`StepOutput::push_effect`]);
/// the order-establishing `reseal` and the unsorted `push_unsorted` are
/// module-private, so the only public view ([`std::ops::Deref`] to
/// `&[Effect]`) is always the sorted one.
#[derive(Clone, Debug, Default)]
pub struct SortedEffects(SmallVec<[Effect; 2]>);

impl SortedEffects {
    /// Append without re-establishing order. The companion
    /// [`Self::reseal`] restores the invariant before the value escapes.
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

/// The four owned streams yielded by [`StepOutput::into_parts`], in
/// dispatch order: watch ops, probe ops, effects (emission-sorted),
/// diagnostics.
pub type StepOutputParts = (
    SmallVec<[WatchOp; 2]>,
    SmallVec<[ProbeOp; 4]>,
    SmallVec<[Effect; 2]>,
    SmallVec<[Diagnostic; 2]>,
);

#[derive(Debug, Default, Clone)]
pub struct StepOutput {
    pub watch_ops: SmallVec<[WatchOp; 2]>,
    pub probe_ops: SmallVec<[ProbeOp; 4]>,
    effects: SortedEffects,
    pub diagnostics: SmallVec<[Diagnostic; 2]>,
}

impl StepOutput {
    /// Append an [`Effect`] in unsorted (insertion) order.
    /// [`Self::sort_for_emission`] re-establishes the contract before
    /// the value is handed downstream.
    pub fn push_effect(&mut self, e: Effect) {
        self.effects.push_unsorted(e);
    }

    /// The emitted effects, always in sort order.
    #[must_use]
    pub const fn effects(&self) -> &SortedEffects {
        &self.effects
    }

    /// Consume into the four owned streams, effects in emission order.
    ///
    /// Terminal-consumer drain: `self` is owned, so this is only
    /// reachable on a fully built, already-resealed value (every
    /// `StepOutput`-returning entry point reseals before returning).
    /// Zero-copy — every stream moves out.
    #[must_use]
    pub fn into_parts(self) -> StepOutputParts {
        (
            self.watch_ops,
            self.probe_ops,
            self.effects.0,
            self.diagnostics,
        )
    }

    /// Sort the streams to the engine's determinism contract:
    /// `watch_ops` by [`WatchOp::resource`]; `probe_ops` by
    /// [`ProbeOp::owner`]; `effects` by [`Effect::sort_key`].
    /// `diagnostics` follow insertion order — they are not part of the
    /// user-visible sort guarantee.
    ///
    /// `watch_ops` / `effects` order is **replay determinism** only —
    /// reorder them and the same work still happens, just logged in a
    /// different sequence. `probe_ops` order is **operational
    /// correctness**: the sensor applies `submit` / `cancel` against a
    /// shared per-owner expectation map whose insert and remove do not
    /// commute, so a mis-ordered same-owner pair can drop a live probe.
    /// The injective-key property that makes the sort exact (not merely
    /// stable) for that stream is the debug-asserted invariant below.
    ///
    /// Pure: every key is derived from the op/effect's own fields, so
    /// this method needs no engine state.
    pub fn sort_for_emission(&mut self) {
        self.watch_ops.sort_by_key(WatchOp::resource);
        self.probe_ops.sort_by_key(ProbeOp::owner);
        self.effects.reseal();
        #[cfg(debug_assertions)]
        self.debug_assert_one_probe_op_per_owner();
    }

    /// Debug-only structural witness for the probe-protocol invariant:
    /// **at most one [`ProbeOp`] per owner per `StepOutput`**. This is
    /// what makes [`ProbeOp::owner`] an *injective* key over a step's
    /// `probe_ops`: with it, `sort_by_key` vs `sort_unstable_by_key` is
    /// immaterial, and the engine→sensor `submit` / `cancel` drain —
    /// which is *not* commutative over the sensor's shared per-owner
    /// expectation map — cannot reorder a redundant `Cancel` ahead of a
    /// `Probe` and lose it.
    ///
    /// Holds by construction: a `Probe` is preceded by arming the
    /// owner's `ProbeSlot` (construct-time `ProbeSlot::armed`, or
    /// `ProbeSlot::arm`'s unconditional arm-once assert); a `Cancel` is
    /// emitted only by `cancel_owner_probe`, which disarms iff the slot
    /// was armed, so it fires at most once per owner per step; and no
    /// step emits both for one owner — the lone structural candidate,
    /// the overflow-reseed Active arm, is reseed-XOR-reap (disarm-only
    /// on reseed; `Cancel`-only on reap). The runtime dual of
    /// `finish_burst_to_idle`'s cancel-first entry precondition. Zero
    /// release cost.
    #[cfg(debug_assertions)]
    fn debug_assert_one_probe_op_per_owner(&self) {
        // Post-sort ⇒ equal owners are adjacent ⇒ an O(n) adjacent-pair
        // scan is exhaustive, with no allocation.
        for w in self.probe_ops.windows(2) {
            debug_assert_ne!(
                w[0].owner(),
                w[1].owner(),
                ">1 ProbeOp for one owner in a StepOutput — a redundant \
                 Cancel beside a Probe, or a double emit. The sensor's \
                 per-owner expectation-map drain is order-sensitive; one \
                 ProbeOp per owner per step is the contract (ProbeSlot \
                 linearity + cancel_owner_probe).",
            );
        }
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

    /// Minimal one-op program for fixture construction; the body content
    /// doesn't matter for the sort-emission tests below, only that the
    /// field is well-typed and non-empty.
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

    /// Test fixture for an [`Effect`] when only `(key, target)` matter
    /// (sort tests). Other fields take their natural empty values.
    /// `anchor` doubles as the Subtree sort resource: setting it to
    /// `target` keeps `sort_key().1 == target` for the Subtree arm,
    /// matching the old flat `Effect.target` behaviour. The PerFile arm
    /// keys on its own `resource`, which the test always passes equal to
    /// `target`.
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
        out.watch_ops.push(WatchOp::Suppress { resource: r3 });
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

    #[test]
    fn sort_for_emission_orders_probe_ops_by_owner() {
        use crate::op::ProbeOwner;
        let p1 = pidn(1);
        let p2 = pidn(2);
        let mut out = StepOutput::default();
        out.probe_ops.push(ProbeOp::Cancel {
            owner: ProbeOwner::Profile(p2),
        });
        out.probe_ops.push(ProbeOp::Probe {
            request: ProbeRequest::AnchorFile {
                owner: ProbeOwner::Profile(p1),
                correlation: ProbeCorrelation::from(7),
                target_path: Arc::from(std::path::Path::new("/y")),
            },
        });

        out.sort_for_emission();

        let owners: Vec<ProbeOwner> = out.probe_ops.iter().map(ProbeOp::owner).collect();
        assert_eq!(
            owners,
            vec![ProbeOwner::Profile(p1), ProbeOwner::Profile(p2)]
        );
    }

    /// A `Probe` and a `Cancel` for the *same* owner in one step trips
    /// the debug-only at-most-one-`ProbeOp`-per-owner witness — the
    /// sensor's per-owner expectation-map drain is order-sensitive, so
    /// this pair is the contract violation it guards. `should_panic`
    /// over a `debug_assert` is sound here: nextest's default debug
    /// profile keeps `debug_assertions` on.
    #[test]
    #[should_panic(expected = ">1 ProbeOp for one owner in a StepOutput")]
    fn sort_for_emission_panics_on_two_probe_ops_for_one_owner() {
        use crate::op::ProbeOwner;
        let p = pidn(1);
        let mut out = StepOutput::default();
        out.probe_ops.push(ProbeOp::Probe {
            request: ProbeRequest::AnchorFile {
                owner: ProbeOwner::Profile(p),
                correlation: ProbeCorrelation::from(7),
                target_path: Arc::from(std::path::Path::new("/y")),
            },
        });
        out.probe_ops.push(ProbeOp::Cancel {
            owner: ProbeOwner::Profile(p),
        });

        out.sort_for_emission();
    }

    /// The per-owner witness is owner-keyed, not count-keyed: a `Probe`
    /// and a `Cancel` for *distinct* owners are the ordinary two-probe
    /// step and must sort normally (Profile order) without tripping the
    /// assertion. Guards against the witness being over-eager.
    #[test]
    fn sort_for_emission_allows_two_probe_ops_for_distinct_owners() {
        use crate::op::ProbeOwner;
        let p1 = pidn(1);
        let p2 = pidn(2);
        let mut out = StepOutput::default();
        out.probe_ops.push(ProbeOp::Cancel {
            owner: ProbeOwner::Profile(p2),
        });
        out.probe_ops.push(ProbeOp::Probe {
            request: ProbeRequest::AnchorFile {
                owner: ProbeOwner::Profile(p1),
                correlation: ProbeCorrelation::from(7),
                target_path: Arc::from(std::path::Path::new("/y")),
            },
        });

        out.sort_for_emission();

        let owners: Vec<ProbeOwner> = out.probe_ops.iter().map(ProbeOp::owner).collect();
        assert_eq!(
            owners,
            vec![ProbeOwner::Profile(p1), ProbeOwner::Profile(p2)]
        );
    }

    /// `sort_for_emission` orders mixed-arm effects by `(sub, target)`.
    /// Mixing `PerFile` and `Subtree` in one step is the production
    /// case (a multi-Sub Profile firing both kinds simultaneously); the
    /// ordering interleaves them by sub and target rather than
    /// segregating by variant. The captured `Effect.target` survives
    /// any post-emission state churn — sort is independent of
    /// `ProfileMap`.
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
