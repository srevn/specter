//! `StepOutput` and its determinism contract.
//!
//! `effects` is sealed: the field is private, wrapped in
//! [`SortedEffects`] whose mutators are module-private. Callers append
//! via [`StepOutput::push_effect`] and read via [`StepOutput::effects`]
//! / [`StepOutput::into_parts`], so observing effects in any order
//! other than the emission sort is unrepresentable. The one residual:
//! every `StepOutput`-returning entry point must call
//! [`StepOutput::sort_for_emission`] before returning the value.

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
    /// Pure: every key is derived from the op/effect's own fields, so
    /// this method needs no engine state.
    pub fn sort_for_emission(&mut self) {
        self.watch_ops.sort_by_key(WatchOp::resource);
        self.probe_ops.sort_by_key(ProbeOp::owner);
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
            path: PathBuf::from("/x"),
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
                target_path: PathBuf::from("/y"),
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
