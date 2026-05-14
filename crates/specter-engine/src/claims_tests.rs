//! Unit tests for `Engine::discard_anchor_state` — pins the helper's
//! contract: which Profile fields are cleared, which are preserved,
//! idempotence, post-vacate safety, and invariance of the
//! lifetime-fixed fields (`events_union`, `has_per_file_fds`).
//!
//! Co-located via `#[path]` on `claims.rs`. Goes hand-in-hand with the
//! per-site `dispatch_*_clears_profile_kind` assertions in
//! `transitions_tests.rs`, which exercise the helper through each
//! production call site.

#![allow(
    clippy::items_after_statements,
    clippy::manual_let_else,
    clippy::missing_const_for_fn,
    clippy::needless_pass_by_value,
    clippy::too_many_lines
)]

use crate::Engine;
use compact_str::CompactString;
use specter_core::testkit::single_exec_program;
use specter_core::{
    ActionProgram, AnchorClaim, ArgPart, ArgTemplate, ChildEntry, ClassSet, DedupKey, DirChild,
    DirMeta, DirSnapshot, EffectScope, EntryKind, FsIdentity, Input, LeafEntry, ProbeCorrelation,
    ProbeOp, ProbeOutcome, ProbeOwner, ProbeResponse, ProfileId, ResourceId, ResourceKind,
    ResourceRole, ScanConfig, StepOutput, SubAttachRequest, SubId, WatchOp,
};
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, Instant, UNIX_EPOCH};

const SETTLE: Duration = Duration::from_millis(100);
const MAX_SETTLE: Duration = Duration::from_secs(6);

fn empty_program() -> Arc<ActionProgram> {
    single_exec_program([ArgTemplate::new([ArgPart::literal("/bin/true")])])
}

fn dir_snap(children: Vec<(&str, EntryKind, u64)>) -> Arc<DirSnapshot> {
    let mut map: BTreeMap<CompactString, ChildEntry> = BTreeMap::new();
    for (name, kind, inode) in children {
        let child = match kind {
            EntryKind::Dir => ChildEntry::Dir(DirChild::Uncovered(FsIdentity { inode, device: 0 })),
            _ => ChildEntry::Leaf(LeafEntry::new(
                kind,
                0,
                UNIX_EPOCH,
                FsIdentity { inode, device: 0 },
            )),
        };
        map.insert(CompactString::new(name), child);
    }
    Arc::new(DirSnapshot::new(
        DirMeta {
            mtime: UNIX_EPOCH,
            fs_id: FsIdentity {
                inode: 0,
                device: 0,
            },
        },
        0,
        map,
    ))
}

fn first_probe_corr(out: &StepOutput) -> Option<ProbeCorrelation> {
    out.probe_ops.iter().find_map(|op| match op {
        ProbeOp::Probe { request } => Some(request.correlation()),
        ProbeOp::Cancel { .. } => None,
    })
}

/// Build an Engine + a Profile materialised at `root`. Returns the
/// `(SubId, ProfileId, anchor_id, parent_id)` tuple. The anchor sits
/// under a parent slot so `watch_root_parent` is set; both are Dir;
/// `events = ClassSet::EMPTY` keeps `has_per_file_fds = false`.
fn engine_with_materialised_profile(
    events: ClassSet,
) -> (Engine, SubId, ProfileId, ResourceId, ResourceId) {
    let mut e = Engine::new();
    let parent = e.tree_mut().ensure_root("var", ResourceRole::User);
    e.tree_mut().set_kind(parent, ResourceKind::Dir);
    let anchor = e
        .tree_mut()
        .ensure_child(parent, "log", ResourceRole::User)
        .expect("test live parent");
    e.tree_mut().set_kind(anchor, ResourceKind::Dir);

    let req = SubAttachRequest::for_resource(
        "watch".into(),
        anchor,
        ScanConfig::builder().recursive(true).build(),
        MAX_SETTLE,
        SETTLE,
        empty_program(),
        EffectScope::SubtreeRoot,
        events,
        false,
    );
    let attach_out = e.step(Input::AttachSub(req), Instant::now());
    let sid = specter_core::testkit::first_attached_sub(&attach_out).expect("attach_sub succeeded");
    let pid = e.subs().get(sid).unwrap().profile;

    // Drive Seed-Ok to materialise current + baseline.
    let corr = first_probe_corr(&attach_out).expect("Seed probe at attach");
    e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
            correlation: corr,
            outcome: ProbeOutcome::SubtreeOk(dir_snap(vec![])),
        }),
        Instant::now(),
    );

    (e, sid, pid, anchor, parent)
}

#[test]
fn discard_anchor_state_clears_kind_baseline_current_anchor_claim() {
    let (mut e, _sid, pid, _anchor, _parent) = engine_with_materialised_profile(ClassSet::EMPTY);

    // Pre-condition.
    {
        let p = e.profiles().get(pid).expect("Profile lives");
        assert_eq!(p.kind, Some(ResourceKind::Dir));
        assert!(p.baseline.is_some());
        assert!(p.current.is_some());
        assert_eq!(p.anchor_claim, AnchorClaim::Held);
    }

    let mut out = StepOutput::default();
    e.discard_anchor_state(pid, &mut out);

    let p = e.profiles().get(pid).expect("Profile lives");
    assert!(p.kind.is_none(), "kind cleared");
    assert!(p.baseline.is_none(), "baseline cleared");
    assert!(p.current.is_none(), "current taken by descendant release");
    assert_eq!(p.anchor_claim, AnchorClaim::None, "anchor claim released");
}

#[test]
fn discard_anchor_state_preserves_watch_root_parent() {
    let (mut e, _sid, pid, _anchor, parent) = engine_with_materialised_profile(ClassSet::EMPTY);
    assert_eq!(
        e.profiles().get(pid).unwrap().watch_root_parent,
        Some(parent),
    );

    let mut out = StepOutput::default();
    e.discard_anchor_state(pid, &mut out);

    assert_eq!(
        e.profiles().get(pid).unwrap().watch_root_parent,
        Some(parent),
        "recovery channel preserved across anchor loss",
    );
    // Parent's watch_demand still carries this Profile's STRUCTURE
    // contribution — the recompute walks covering Profiles, finds this
    // one still claims the parent, and keeps the union.
    assert!(
        e.tree().get(parent).is_some_and(|r| r.watch_demand() >= 1),
        "parent watch_demand preserved",
    );
}

#[test]
fn discard_anchor_state_captures_last_settled_hash_at_loss() {
    let (mut e, _sid, pid, _anchor, _parent) = engine_with_materialised_profile(ClassSet::EMPTY);

    let pre_loss_hash = e
        .profiles()
        .get(pid)
        .and_then(|p| p.baseline.as_ref().map(specter_core::TreeSnapshot::hash))
        .expect("fixture must produce baseline");
    assert!(
        e.profiles()
            .get(pid)
            .unwrap()
            .last_settled_hash_at_loss
            .is_none(),
        "active mode: witness must be None pre-loss",
    );

    let mut out = StepOutput::default();
    e.discard_anchor_state(pid, &mut out);

    let p = e.profiles().get(pid).unwrap();
    assert!(p.baseline.is_none(), "discard cleared baseline");
    assert_eq!(
        p.last_settled_hash_at_loss,
        Some(pre_loss_hash),
        "witness captured pre-loss baseline hash",
    );
}

#[test]
fn discard_anchor_state_preserves_fired_subs() {
    // Negative-space contract: the helper does not touch fired_subs.
    // Fire history must survive anchor loss for post-recovery drift to
    // re-fire emitted-once Effects.
    let (mut e, sid, pid, _anchor, _parent) = engine_with_materialised_profile(ClassSet::EMPTY);
    let key = DedupKey::Subtree {
        sub: sid,
        profile: pid,
    };
    if let Some(p) = e.profiles.get_mut(pid) {
        p.fired_subs.insert(key);
    }
    let set_before = e.profiles().get(pid).unwrap().fired_subs.clone();

    let mut out = StepOutput::default();
    e.discard_anchor_state(pid, &mut out);

    let set_after = &e.profiles().get(pid).unwrap().fired_subs;
    assert_eq!(&set_before, set_after, "fired_subs survives anchor loss");
}

#[test]
fn discard_anchor_state_no_witness_when_baseline_already_none() {
    let (mut e, _sid, pid, _anchor, _parent) = engine_with_materialised_profile(ClassSet::EMPTY);

    let mut out = StepOutput::default();
    e.discard_anchor_state(pid, &mut out);
    let witness_after_first = e.profiles().get(pid).unwrap().last_settled_hash_at_loss;
    assert!(
        witness_after_first.is_some(),
        "first discard captures witness",
    );

    let mut out2 = StepOutput::default();
    e.discard_anchor_state(pid, &mut out2);

    assert_eq!(
        e.profiles().get(pid).unwrap().last_settled_hash_at_loss,
        witness_after_first,
        "second discard with baseline = None preserves prior witness",
    );
}

#[test]
fn discard_anchor_state_idempotent() {
    let (mut e, _sid, pid, _anchor, _parent) = engine_with_materialised_profile(ClassSet::EMPTY);
    let mut out = StepOutput::default();
    e.discard_anchor_state(pid, &mut out);

    let snap_after_first = {
        let p = e.profiles().get(pid).expect("Profile lives");
        (
            p.kind,
            p.baseline.is_some(),
            p.current.is_some(),
            p.anchor_claim,
        )
    };

    let mut out2 = StepOutput::default();
    e.discard_anchor_state(pid, &mut out2);

    let snap_after_second = {
        let p = e.profiles().get(pid).expect("Profile lives");
        (
            p.kind,
            p.baseline.is_some(),
            p.current.is_some(),
            p.anchor_claim,
        )
    };

    assert_eq!(
        snap_after_first, snap_after_second,
        "second invocation observes the same Profile state",
    );
    assert!(
        out2.watch_ops.is_empty() && out2.probe_ops.is_empty(),
        "second invocation emits no ops; got watch_ops={:?} probe_ops={:?}",
        out2.watch_ops,
        out2.probe_ops,
    );
}

#[test]
fn discard_anchor_state_safe_after_vacate() {
    // anchor contributions were cleared (e.g., by WatchOpRejected
    // → `Tree::vacate`); `release_anchor_claim`'s `sub_watch` must
    // silently skip the absent `ContribKey::ProfileAnchor(pid)` key
    // and skip emitting a second Unwatch (vacate already emitted one).
    let (mut e, _sid, pid, anchor, _parent) = engine_with_materialised_profile(ClassSet::EMPTY);

    // Capture the pre-vacate counter to make sure vacate actually fires.
    assert!(e.tree().get(anchor).is_some_and(|r| r.watch_demand() > 0));

    let mut vacate_out = StepOutput::default();
    e.tree_mut().vacate(anchor, &mut vacate_out);
    assert_eq!(
        e.tree()
            .get(anchor)
            .map_or(0, specter_core::Resource::watch_demand),
        0,
        "vacate zeroed the counter",
    );

    let mut out = StepOutput::default();
    e.discard_anchor_state(pid, &mut out);
    // Anchor's edge already fired during vacate; the helper must
    // not emit a second Unwatch on the post-vacate counter.
    assert!(
        !out.watch_ops
            .iter()
            .any(|op| matches!(op, WatchOp::Unwatch { resource } if *resource == anchor)),
        "no stray Unwatch on post-vacate anchor; got {:?}",
        out.watch_ops,
    );
    // Profile state still cleared correctly.
    let p = e.profiles().get(pid).expect("Profile lives");
    assert_eq!(p.anchor_claim, AnchorClaim::None);
    assert!(p.kind.is_none());
}

#[test]
fn discard_anchor_state_no_op_on_already_lost_profile() {
    let (mut e, _sid, pid, _anchor, _parent) = engine_with_materialised_profile(ClassSet::EMPTY);
    let mut first_out = StepOutput::default();
    e.discard_anchor_state(pid, &mut first_out);

    // Second call against a fully-cleared Profile — no ops, no
    // diagnostics.
    let mut out = StepOutput::default();
    e.discard_anchor_state(pid, &mut out);

    assert!(
        out.watch_ops.is_empty(),
        "no watch ops; got {:?}",
        out.watch_ops
    );
    assert!(
        out.probe_ops.is_empty(),
        "no probe ops; got {:?}",
        out.probe_ops
    );
    assert!(
        out.diagnostics.is_empty(),
        "no diagnostics; got {:?}",
        out.diagnostics,
    );
    assert!(out.effects.is_empty());
}

#[test]
fn discard_anchor_state_preserves_events_union_and_per_file_fds() {
    // events_union and has_per_file_fds are invariant for the Profile's
    // lifetime under the events-folds-into-config_hash discipline; the
    // helper must not touch them.
    let (mut e, _sid, pid, _anchor, _parent) = engine_with_materialised_profile(ClassSet::CONTENT);
    let (events_before, fds_before) = {
        let p = e.profiles().get(pid).expect("Profile lives");
        (p.events_union, p.has_per_file_fds)
    };
    assert_eq!(events_before, ClassSet::CONTENT);
    assert!(fds_before, "CONTENT events ⇒ per-file FDs enabled");

    let mut out = StepOutput::default();
    e.discard_anchor_state(pid, &mut out);

    let p = e.profiles().get(pid).expect("Profile lives");
    assert_eq!(p.events_union, events_before, "events_union invariant");
    assert_eq!(p.has_per_file_fds, fds_before, "has_per_file_fds invariant");
}

#[test]
fn discard_anchor_state_walks_descendants_and_releases_their_demand() {
    // Materialise a Profile with a Dir child; verify the per-descendant
    // contribution is released by the helper.
    let mut e = Engine::new();
    let anchor = e.tree_mut().ensure_root("src", ResourceRole::User);
    e.tree_mut().set_kind(anchor, ResourceKind::Dir);

    let req = SubAttachRequest::for_resource(
        "watch".into(),
        anchor,
        ScanConfig::builder().recursive(true).build(),
        MAX_SETTLE,
        SETTLE,
        empty_program(),
        EffectScope::SubtreeRoot,
        ClassSet::EMPTY,
        false,
    );
    let attach_out = e.step(Input::AttachSub(req), Instant::now());
    let sid = specter_core::testkit::first_attached_sub(&attach_out).expect("attach_sub succeeded");
    let pid = e.subs().get(sid).unwrap().profile;
    let corr = first_probe_corr(&attach_out).expect("Seed probe at attach");
    let snap = dir_snap(vec![("nested", EntryKind::Dir, 1)]);
    e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
            correlation: corr,
            outcome: ProbeOutcome::SubtreeOk(snap),
        }),
        Instant::now(),
    );

    // Confirm the child slot is materialised + watched.
    let nested_id = e.tree().lookup(Some(anchor), "nested").expect("child slot");
    assert!(
        e.tree()
            .get(nested_id)
            .is_some_and(|r| r.watch_demand() >= 1),
        "child watch_demand bumped by graft",
    );

    let mut out = StepOutput::default();
    e.discard_anchor_state(pid, &mut out);

    // Child's contribution from this Profile released; the slot may
    // even have been reaped if no other claimers remain. Either way,
    // its watch_demand drops to 0.
    let child_demand = e
        .tree()
        .get(nested_id)
        .map_or(0, specter_core::Resource::watch_demand);
    assert_eq!(
        child_demand, 0,
        "descendant contribution released after discard_anchor_state",
    );
    let _ = sid;
}

/// F-CRIT-1 regression: a covered descendant whose `suppress_count` was
/// bumped during the Profile's Active(Batching) window must have its
/// `Unsuppress` emitted symmetrically when `release_descendant_claim`
/// reaches `Tree::vacate` through `delete_child`. Pre-fix this path
/// panicked in dev (the suppress precondition `debug_assert!`) and
/// silently orphaned the sensor's per-Resource suppress bookkeeping in
/// release. Post-fix, vacate is the protocol closer: any outstanding
/// `suppress_count > 0` at the slot terminus emits the closing op
/// before the slot is reaped.
///
/// Lifecycle reproduced:
/// 1. Profile P at `/a` (Dir), STRUCTURE-only, with materialised
///    descendant `/a/b` (Dir) — `b.watch_demand == 1`.
/// 2. `FsEvent` at `/a` ⇒ `start_standard_burst` ⇒
///    `Active(PreFire(PreFireBurst { phase: Batching, ... }))`.
///    Anchor's `suppress_count` rises to 1.
/// 3. `FsEvent` at `/a/b` mid-Batching ⇒ `event_drives_batching`
///    inserts `b` into `PreFireBurst.suppressed_resources` and bumps
///    `b.suppress_count` to 1.
/// 4. `WatchOpRejected` on the anchor ⇒ `on_watch_op_rejected` ⇒
///    clamp + `finalize_anchor_lost(P)` ⇒ `discard_anchor_state(P)`
///    ⇒ `release_descendant_claim(P)` walks the snapshot
///    ⇒ `delete_child(b)` ⇒ `sub_watch(b, ProfileDescendant(P))`
///    empties `b.contributions` ⇒ `tree.vacate(b, out)` emits
///    `WatchOp::Unsuppress { resource: b }` and zeroes
///    `b.suppress_count`.
/// 5. `finish_burst_to_idle(P)`'s defensive drain finds `b`'s slot
///    reaped (or its counter at zero) and short-circuits — no double
///    Unsuppress.
#[test]
fn release_descendant_claim_drains_suppress_via_vacate() {
    // Materialise P at /a with Dir descendant /a/b. STRUCTURE-only ⇒
    // `has_per_file_fds = false`, so the descendant clause's Dir branch
    // is the contribution that the regression exercises.
    let mut e = Engine::new();
    let anchor = e.tree_mut().ensure_root("a", ResourceRole::User);
    e.tree_mut().set_kind(anchor, ResourceKind::Dir);

    let req = SubAttachRequest::for_resource(
        "watch".into(),
        anchor,
        ScanConfig::builder().recursive(true).build(),
        MAX_SETTLE,
        SETTLE,
        empty_program(),
        EffectScope::SubtreeRoot,
        ClassSet::STRUCTURE,
        false,
    );
    let attach_out = e.step(Input::AttachSub(req), Instant::now());
    let sid = specter_core::testkit::first_attached_sub(&attach_out).expect("attach_sub succeeded");
    let pid = e.subs().get(sid).unwrap().profile;

    // Seed-Ok response materialises descendant /a/b as a Dir.
    let corr = first_probe_corr(&attach_out).expect("Seed probe at attach");
    let snap = dir_snap(vec![("b", EntryKind::Dir, 7)]);
    e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
            correlation: corr,
            outcome: ProbeOutcome::SubtreeOk(snap),
        }),
        Instant::now(),
    );
    let b_id = e.tree().lookup(Some(anchor), "b").expect("b materialised");
    assert_eq!(
        e.tree().get(b_id).unwrap().watch_demand(),
        1,
        "descendant b carries P's STRUCTURE contribution",
    );
    assert_eq!(e.tree().get(b_id).unwrap().suppress_count(), 0);

    // FsEvent at the anchor opens a Standard burst (Idle → Active).
    // The anchor's suppress is bracketed by `start_standard_burst` /
    // `finish_burst_to_idle`; non-anchor descendants accumulate via
    // `event_drives_batching` below.
    let _ = e.step(
        Input::FsEvent {
            resource: anchor,
            event: specter_core::FsEvent::StructureChanged,
        },
        Instant::now(),
    );

    // FsEvent at the descendant mid-Batching adds `b` to the burst's
    // `suppressed_resources` and bumps `b.suppress_count` to 1.
    let _ = e.step(
        Input::FsEvent {
            resource: b_id,
            event: specter_core::FsEvent::StructureChanged,
        },
        Instant::now(),
    );
    assert_eq!(
        e.tree().get(b_id).unwrap().suppress_count(),
        1,
        "event_drives_batching bumped b.suppress_count",
    );

    // Synthesise WatchOpRejected on the anchor: triggers the F-CRIT-1
    // path through finalize_anchor_lost → discard_anchor_state →
    // release_descendant_claim → delete_child(b) → vacate(b, out).
    // Pre-fix this panicked at vacate's suppress precondition; post-fix
    // vacate emits the closing Unsuppress and zeroes the counter.
    let purge_out = e.step(
        Input::WatchOpRejected {
            resource: anchor,
            op: WatchOp::Watch {
                resource: anchor,
                path: std::path::PathBuf::from("a"),
                kind: ResourceKind::Dir,
                events: ClassSet::STRUCTURE,
            },
            failure: specter_core::WatchFailure::Pressure { errno: 24 },
        },
        Instant::now(),
    );

    // Descendant's slot is reaped (no back-refs, watch_demand=0,
    // suppress_count=0) — or, in the multi-contributor variant, alive
    // with both counters at zero. This test exercises the
    // single-contributor case so reap is the expected outcome.
    assert!(
        e.tree().get(b_id).is_none(),
        "descendant b reaped after delete_child + try_reap",
    );

    // Sensor-side balance: the prior Suppress(b) (bump from 0→1 in
    // event_drives_batching) must be paired with exactly one
    // Unsuppress(b). Pre-fix the count would have been zero (vacate
    // silently zeroed in release / panicked in dev).
    let unsuppress_b = purge_out
        .watch_ops
        .iter()
        .filter(|op| matches!(op, WatchOp::Unsuppress { resource } if *resource == b_id))
        .count();
    assert_eq!(
        unsuppress_b, 1,
        "exactly one Unsuppress(b) emitted via vacate's protocol-closer; \
         got {:?}",
        purge_out.watch_ops,
    );

    // Profile reverts to anchor-loss state: anchor_claim cleared,
    // baseline / kind cleared, watch_root_parent preserved (the
    // recovery channel — but the anchor is a root in this fixture, so
    // `watch_root_parent` is None throughout).
    let p = e.profiles().get(pid).expect("Profile lives");
    assert_eq!(p.anchor_claim, AnchorClaim::None);
    assert!(p.kind.is_none());
    assert!(p.baseline.is_none());
    assert!(p.current.is_none());

    // Profile is back to Idle (finish_burst_to_idle ran).
    assert!(matches!(p.state, specter_core::ProfileState::Idle,));
}
