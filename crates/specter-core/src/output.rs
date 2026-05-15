//! `StepOutput` and its determinism contract.
//!
//! The sort guarantee is owned by the type itself: every consumer that
//! returns `StepOutput` calls [`StepOutput::sort_for_emission`] before
//! handing it back, so callers always see sorted slices. `SmallVec`
//! tracks inline-slot validity via `MaybeUninit` — uninitialized slots
//! are unreachable rather than depending on a `Default` sentinel.

use crate::diag::Diagnostic;
use crate::effect::Effect;
use crate::op::{ProbeOp, WatchOp};
use smallvec::SmallVec;

#[derive(Debug, Default, Clone)]
pub struct StepOutput {
    pub watch_ops: SmallVec<[WatchOp; 2]>,
    pub probe_ops: SmallVec<[ProbeOp; 4]>,
    pub effects: SmallVec<[Effect; 2]>,
    pub diagnostics: SmallVec<[Diagnostic; 2]>,
}

impl StepOutput {
    /// Sort the three slices to the engine's determinism contract:
    /// `watch_ops` by [`WatchOp::resource`]; `probe_ops` by
    /// [`ProbeOp::owner`]; `effects` by `(key.sub(), target)`.
    /// `diagnostics` follow insertion order — they are not part of the
    /// user-visible sort guarantee.
    ///
    /// Pure: every key is captured on the op/effect at construction
    /// time, so this method needs no engine state.
    pub fn sort_for_emission(&mut self) {
        self.watch_ops.sort_by_key(WatchOp::resource);
        self.probe_ops.sort_by_key(ProbeOp::owner);
        self.effects.sort_by_key(|e| (e.key.sub(), e.target));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diff::Diff;
    use crate::effect::DedupKey;
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
    fn effect(key: DedupKey, target: ResourceId) -> Effect {
        Effect {
            key,
            target,
            forced: false,
            correlation: CorrelationId::default(),
            diff: None::<Arc<Diff>>,
            capture_output: false,
            sub_name: CompactString::new(""),
            program: fixture_program(),
            anchor_path: Arc::from(PathBuf::new()),
            anchor_kind: ResourceKind::Dir,
            target_relative: CompactString::new(""),
            exclude: Arc::from(Vec::<CompactString>::new()),
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
        out.effects.push(effect(
            DedupKey::Subtree {
                sub: sub_b,
                profile: prof,
            },
            r_hi,
        ));
        out.effects.push(effect(
            DedupKey::PerFile {
                sub: sub_a,
                profile: prof,
                resource: r_hi,
            },
            r_hi,
        ));
        out.effects.push(effect(
            DedupKey::PerFile {
                sub: sub_a,
                profile: prof,
                resource: r_lo,
            },
            r_lo,
        ));

        out.sort_for_emission();

        let keys: Vec<(SubId, ResourceId)> = out
            .effects
            .iter()
            .map(|e| (e.key.sub(), e.target))
            .collect();
        assert_eq!(keys, vec![(sub_a, r_lo), (sub_a, r_hi), (sub_b, r_hi)]);
    }
}
