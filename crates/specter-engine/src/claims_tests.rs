//! Unit tests for `Engine::discard_anchor_state` — pins the helper's contract: which Profile fields
//! are cleared, which are preserved, idempotence, post-vacate safety, and invariance of the
//! lifetime-fixed fields (`events_union`, `has_per_file_fds`).
//!
//! Co-located via `#[path]` on `claims.rs`. Goes hand-in-hand with the per-site
//! `dispatch_*_clears_profile_kind` assertions in `transitions_tests.rs`, which exercise the helper
//! through each production call site.

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
    ActionProgram, AnchorClaim, ArgPart, ArgTemplate, ChildEntry, ClassSet, DirChild, DirMeta,
    DirSnapshot, EffectScope, EntryKind, FsIdentity, Input, LeafEntry, ProbeOutcome, ProbeResponse,
    ProfileId, ProofAuthority, ResourceId, ResourceKind, ResourceRole, ScanConfig, StepOutput,
    SubAttachAnchor, SubAttachRequest, SubId, WatchOp,
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
            EntryKind::Dir => ChildEntry::Dir(DirChild::Uncovered(FsIdentity::synthetic(inode, 0))),
            _ => ChildEntry::Leaf(LeafEntry::synthetic(
                kind,
                0,
                UNIX_EPOCH,
                FsIdentity::synthetic(inode, 0),
            )),
        };
        map.insert(CompactString::new(name), child);
    }
    Arc::new(DirSnapshot::new(
        DirMeta::synthetic(UNIX_EPOCH, FsIdentity::synthetic(0, 0)),
        0,
        map,
    ))
}

/// Drive a fresh-attach cold Seed burst from `Active(PreFire(Verifying))` through its quiescence
/// verdict to pinned `Idle`, committing `snap` as `current` + `baseline`.
///
/// The cold-arm Seed burst pins on the first `Authoritative` sample: a cold-Seed `SilentPin`
/// consequence does not owe quiescence proof, so the witness is [`QuiescenceWitness::EventsReliable`]
/// and the fold folds to `Stable(StableReason::Natural)`; dispatch reaches `SilentPin` (no fired
/// Subs, no drift) and finishes to Idle. The cold-arm Verifying-first contract puts the probe in
/// flight at burst construction, so this helper answers it directly — no settle expiry step.
fn drive_fresh_seed_to_idle(e: &mut Engine, pid: ProfileId, snap: Arc<DirSnapshot>, t0: Instant) {
    let corr = e
        .pending_probe_for(pid)
        .expect("cold-arm Seed Verifying probe in flight at burst construction");
    e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid,
            correlation: corr,
            outcome: ProbeOutcome::SubtreeProven {
                snapshot: Arc::clone(&snap),
                authority: ProofAuthority::Authoritative,
            },
        }),
        t0 + SETTLE,
    );
    assert!(
        matches!(
            e.profiles().get(pid).unwrap().state(),
            specter_core::ProfileState::Idle
        ),
        "two settle-spaced equal Seed samples pin the baseline → Idle",
    );
}

/// Build an Engine + a Profile materialised at `root`. Returns the `(SubId, ProfileId, anchor_id,
/// parent_id)` tuple. The anchor sits under a parent slot so `watch_root_parent` is set; both are
/// Dir; `events = ClassSet::EMPTY` keeps `has_per_file_fds = false`.
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

    let req = SubAttachRequest::for_anchor(
        "watch".into(),
        SubAttachAnchor::Resource(anchor),
        ScanConfig::builder().recursive(true).build(),
        MAX_SETTLE,
        SETTLE,
        empty_program(),
        EffectScope::SubtreeRoot,
        events,
        false,
    );
    let t0 = Instant::now();
    let attach_out = e.step(Input::AttachSub(req), t0);
    let sid = specter_core::testkit::first_attached_sub(&attach_out).expect("attach_sub succeeded");
    let pid = e.subs().get(sid).unwrap().profile();

    // Drive the cold-arm Seed through its quiescence proof so `current` and `baseline` pin to the
    // empty-dir observation.
    drive_fresh_seed_to_idle(&mut e, pid, dir_snap(vec![]), t0);

    (e, sid, pid, anchor, parent)
}

#[test]
fn discard_anchor_state_clears_kind_baseline_current_anchor_claim() {
    let (mut e, _sid, pid, _anchor, _parent) = engine_with_materialised_profile(ClassSet::EMPTY);

    // Pre-condition.
    {
        let p = e.profiles().get(pid).expect("Profile lives");
        assert_eq!(p.kind(), Some(ResourceKind::Dir));
        assert!(p.baseline().is_some());
        assert!(p.current().is_some());
        assert_eq!(p.anchor_claim(), AnchorClaim::Held);
    }

    let mut out = StepOutput::default();
    e.discard_anchor_state(pid, &mut out);

    let p = e.profiles().get(pid).expect("Profile lives");
    assert!(p.kind().is_none(), "kind cleared");
    assert!(p.baseline().is_none(), "baseline cleared");
    assert!(p.current().is_none(), "current taken by descendant release");
    assert_eq!(p.anchor_claim(), AnchorClaim::None, "anchor claim released");
}

#[test]
fn discard_anchor_state_preserves_watch_root_parent() {
    let (mut e, _sid, pid, _anchor, parent) = engine_with_materialised_profile(ClassSet::EMPTY);
    assert_eq!(
        e.profiles().get(pid).unwrap().watch_root_parent(),
        Some(parent),
    );

    let mut out = StepOutput::default();
    e.discard_anchor_state(pid, &mut out);

    assert_eq!(
        e.profiles().get(pid).unwrap().watch_root_parent(),
        Some(parent),
        "recovery channel preserved across anchor loss",
    );
    // Parent's watch_demand still carries this Profile's STRUCTURE contribution — the recompute
    // walks covering Profiles, finds this one still claims the parent, and keeps the union.
    assert!(
        e.tree().get(parent).is_some_and(|r| r.watch_demand() >= 1),
        "parent watch_demand preserved",
    );
}

#[test]
fn discard_anchor_state_carries_settled_hash_through_loss() {
    let (mut e, _sid, pid, _anchor, _parent) = engine_with_materialised_profile(ClassSet::EMPTY);

    let pre_loss_hash = e
        .profiles()
        .get(pid)
        .and_then(|p| p.baseline().map(|s| s.hash()))
        .expect("fixture must produce baseline");
    // Active mode: the settled reference *is* the live baseline — a separate survival witness
    // alongside a held baseline is not representable in the anchor sum.
    assert_eq!(
        e.profiles().get(pid).unwrap().settled_hash(),
        Some(pre_loss_hash),
        "active mode: settled reference is the live baseline hash",
    );

    let mut out = StepOutput::default();
    e.discard_anchor_state(pid, &mut out);

    let p = e.profiles().get(pid).unwrap();
    assert!(p.baseline().is_none(), "discard cleared the baseline");
    assert_eq!(
        p.settled_hash(),
        Some(pre_loss_hash),
        "the survival witness carries the pre-loss baseline hash through \
         the loss window so post-recovery drift still has a reference",
    );
}

#[test]
fn discard_anchor_state_preserves_fired_subs() {
    // Negative-space contract: anchor loss does not clear fire history. The history is now per-Sub
    // (`Sub.has_fired`) and `discard_anchor_state` operates on the Profile only, so survival across
    // the loss window is structural — but the property still matters: post-recovery drift must
    // re-fire emitted-once Effects.
    let (mut e, sid, pid, _anchor, _parent) = engine_with_materialised_profile(ClassSet::EMPTY);
    e.subs.mark_fired(sid);
    assert!(
        e.subs.get(sid).is_some_and(specter_core::Sub::has_fired),
        "precondition: fire recorded",
    );

    let mut out = StepOutput::default();
    e.discard_anchor_state(pid, &mut out);

    assert!(
        e.subs.get(sid).is_some_and(specter_core::Sub::has_fired),
        "fire history survives anchor loss",
    );
}

#[test]
fn discard_anchor_state_idempotent_preserves_witness() {
    let (mut e, _sid, pid, _anchor, _parent) = engine_with_materialised_profile(ClassSet::EMPTY);

    let mut out = StepOutput::default();
    e.discard_anchor_state(pid, &mut out);
    let witness_after_first = e.profiles().get(pid).unwrap().settled_hash();
    assert!(
        witness_after_first.is_some(),
        "first discard captures the survival witness",
    );

    let mut out2 = StepOutput::default();
    e.discard_anchor_state(pid, &mut out2);

    assert_eq!(
        e.profiles().get(pid).unwrap().settled_hash(),
        witness_after_first,
        "second discard against an already-Unclassified anchor preserves \
         the prior witness rather than overwriting it with None",
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
            p.kind(),
            p.baseline().is_some(),
            p.current().is_some(),
            p.anchor_claim(),
        )
    };

    let mut out2 = StepOutput::default();
    e.discard_anchor_state(pid, &mut out2);

    let snap_after_second = {
        let p = e.profiles().get(pid).expect("Profile lives");
        (
            p.kind(),
            p.baseline().is_some(),
            p.current().is_some(),
            p.anchor_claim(),
        )
    };

    assert_eq!(
        snap_after_first, snap_after_second,
        "second invocation observes the same Profile state",
    );
    assert!(
        out2.watch_ops.is_empty() && out2.probe_ops().is_empty(),
        "second invocation emits no ops; got watch_ops={:?} probe_ops={:?}",
        out2.watch_ops,
        out2.probe_ops(),
    );
}

#[test]
fn discard_anchor_state_safe_after_vacate() {
    // anchor contributions were cleared (e.g., by WatchOpRejected → `Tree::vacate`);
    // `release_anchor_claim`'s `sub_watch` must silently skip the absent
    // `ContribKey::ProfileAnchor(pid)` key and skip emitting a second Unwatch (vacate already
    // emitted one).
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
    // Anchor's edge already fired during vacate; the helper must not emit a second Unwatch on the
    // post-vacate counter.
    assert!(
        !out.watch_ops
            .iter()
            .any(|op| matches!(op, WatchOp::Unwatch { resource } if *resource == anchor)),
        "no stray Unwatch on post-vacate anchor; got {:?}",
        out.watch_ops,
    );
    // Profile state still cleared correctly.
    let p = e.profiles().get(pid).expect("Profile lives");
    assert_eq!(p.anchor_claim(), AnchorClaim::None);
    assert!(p.kind().is_none());
}

#[test]
fn discard_anchor_state_no_op_on_already_lost_profile() {
    let (mut e, _sid, pid, _anchor, _parent) = engine_with_materialised_profile(ClassSet::EMPTY);
    let mut first_out = StepOutput::default();
    e.discard_anchor_state(pid, &mut first_out);

    // Second call against a fully-cleared Profile — no ops, no diagnostics.
    let mut out = StepOutput::default();
    e.discard_anchor_state(pid, &mut out);

    assert!(
        out.watch_ops.is_empty(),
        "no watch ops; got {:?}",
        out.watch_ops
    );
    assert!(
        out.probe_ops().is_empty(),
        "no probe ops; got {:?}",
        out.probe_ops()
    );
    assert!(
        out.diagnostics.is_empty(),
        "no diagnostics; got {:?}",
        out.diagnostics,
    );
    assert!(out.effects().is_empty());
}

#[test]
fn discard_anchor_state_preserves_events_union_and_per_file_fds() {
    // events_union and has_per_file_fds are invariant for the Profile's lifetime under the
    // events-folds-into-config_hash discipline; the helper must not touch them.
    let (mut e, _sid, pid, _anchor, _parent) = engine_with_materialised_profile(ClassSet::CONTENT);
    let (events_before, fds_before) = {
        let p = e.profiles().get(pid).expect("Profile lives");
        (p.events(), p.has_per_file_fds())
    };
    assert_eq!(events_before, ClassSet::CONTENT);
    assert!(fds_before, "CONTENT events ⇒ per-file FDs enabled");

    let mut out = StepOutput::default();
    e.discard_anchor_state(pid, &mut out);

    let p = e.profiles().get(pid).expect("Profile lives");
    assert_eq!(p.events(), events_before, "events_union invariant");
    assert_eq!(
        p.has_per_file_fds(),
        fds_before,
        "has_per_file_fds invariant"
    );
}

#[test]
fn discard_anchor_state_walks_descendants_and_releases_their_demand() {
    // Materialise a Profile with a Dir child; verify the per-descendant contribution is released by
    // the helper.
    let mut e = Engine::new();
    let anchor = e.tree_mut().ensure_root("src", ResourceRole::User);
    e.tree_mut().set_kind(anchor, ResourceKind::Dir);

    let req = SubAttachRequest::for_anchor(
        "watch".into(),
        SubAttachAnchor::Resource(anchor),
        ScanConfig::builder().recursive(true).build(),
        MAX_SETTLE,
        SETTLE,
        empty_program(),
        EffectScope::SubtreeRoot,
        ClassSet::EMPTY,
        false,
    );
    let t0 = Instant::now();
    let attach_out = e.step(Input::AttachSub(req), t0);
    let sid = specter_core::testkit::first_attached_sub(&attach_out).expect("attach_sub succeeded");
    let pid = e.subs().get(sid).unwrap().profile();
    drive_fresh_seed_to_idle(
        &mut e,
        pid,
        dir_snap(vec![("nested", EntryKind::Dir, 1)]),
        t0,
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

    // Child's contribution from this Profile released; the slot may even have been reaped if no
    // other claimers remain. Either way, its watch_demand drops to 0.
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

/// Anchor-loss mid-burst with a dirty descendant: the abnormal-end path through `Tree::vacate`
/// cleanly reaps the descendant slot with the kernel-watch protocol balanced. `vacate` is a
/// single-protocol (`Unwatch`-only) terminus, so no suppress-precondition can be violated. This
/// pins the *positive* invariant — exactly one `Unwatch(b)` closes the descendant's watch via the
/// terminus, the slot reaps, and the Profile reverts to anchor-loss state.
///
/// Lifecycle reproduced:
/// 1. Profile P at `/a` (Dir), STRUCTURE-only, with materialised descendant `/a/b` (Dir) —
///    `b.watch_demand == 1`.
/// 2. `FsEvent` at `/a` ⇒ `start_standard_burst` ⇒ `Active(PreFire(Batching))`.
/// 3. `FsEvent` at `/a/b` mid-Batching ⇒ `event_drives_batching` tracks `b`'s path in the burst's
///    `dirty` provenance.
/// 4. `WatchOpRejected` on the anchor ⇒ `on_watch_op_rejected` ⇒ `finalize_anchor_lost(P)` ⇒
///    `discard_anchor_state(P)` ⇒ `release_descendant_claim(P)` walks the snapshot ⇒
///    `delete_child(b)` ⇒ `sub_watch_then_try_reap(b)`: the last contribution drains (emits
///    `WatchOp::Unwatch { resource: b }`) then `try_reap` removes the slot — `vacate`'s `Unwatch`
///    branch is dormant there (the map is already empty by `has_anchors`' contract).
#[test]
fn release_descendant_claim_clean_reaps_dirty_descendant_via_vacate() {
    // Materialise P at /a with Dir descendant /a/b. STRUCTURE-only ⇒ `has_per_file_fds = false`, so
    // the descendant clause's Dir branch is the contribution this exercises.
    let mut e = Engine::new();
    let anchor = e.tree_mut().ensure_root("a", ResourceRole::User);
    e.tree_mut().set_kind(anchor, ResourceKind::Dir);

    let req = SubAttachRequest::for_anchor(
        "watch".into(),
        SubAttachAnchor::Resource(anchor),
        ScanConfig::builder().recursive(true).build(),
        MAX_SETTLE,
        SETTLE,
        empty_program(),
        EffectScope::SubtreeRoot,
        ClassSet::STRUCTURE,
        false,
    );
    let t0 = Instant::now();
    let attach_out = e.step(Input::AttachSub(req), t0);
    let sid = specter_core::testkit::first_attached_sub(&attach_out).expect("attach_sub succeeded");
    let pid = e.subs().get(sid).unwrap().profile();

    // Seed-Ok response materialises descendant /a/b as a Dir.
    drive_fresh_seed_to_idle(&mut e, pid, dir_snap(vec![("b", EntryKind::Dir, 7)]), t0);
    let b_id = e.tree().lookup(Some(anchor), "b").expect("b materialised");
    assert_eq!(
        e.tree().get(b_id).unwrap().watch_demand(),
        1,
        "descendant b carries P's STRUCTURE contribution",
    );

    // FsEvent at the anchor opens a Standard burst (Idle → Active).
    e.step(
        Input::FsEvent {
            resource: anchor,
            event: specter_core::FsEvent::StructureChanged,
        },
        Instant::now(),
    );

    // FsEvent at the descendant mid-Batching: `event_drives_batching` tracks `b` in the burst's
    // dirty / force-walk accumulator. This is the per-event state the deleted global suppress
    // filter used to poison for a co-resident Profile; assert it concretely.
    e.step(
        Input::FsEvent {
            resource: b_id,
            event: specter_core::FsEvent::StructureChanged,
        },
        Instant::now(),
    );
    {
        let p = e.profiles().get(pid).expect("Profile lives");
        let pre = match p.state() {
            specter_core::ProfileState::Active(specter_core::ActiveBurst::PreFire(pre), _) => pre,
            other => panic!("expected Active(PreFire) mid-burst, got {other:?}"),
        };
        assert!(
            matches!(pre.phase, specter_core::PreFirePhase::Batching { .. }),
            "descendant event keeps the burst Batching",
        );
        let b_path = e.tree().path_of(b_id).expect("b path resolves");
        assert!(
            pre.dirty.chains().contains(&b_path),
            "event_drives_batching tracked b's path in dirty (the obligation basis)",
        );
    }

    // WatchOpRejected on the anchor: the abnormal-end path through finalize_anchor_lost →
    // discard_anchor_state → release_descendant_claim → delete_child(b) → vacate. The
    // single-protocol terminus makes the old suppress-precondition dev-panic unconstructable; the
    // test reaching its asserts is itself the no-panic witness.
    let purge_out = e.step(
        Input::WatchOpRejected {
            resource: anchor,
            failure: specter_core::WatchFailure::Pressure { errno: 24 },
        },
        Instant::now(),
    );

    // Clean reap: the single-contributor descendant slot is gone.
    assert!(
        e.tree().get(b_id).is_none(),
        "descendant b reaped after delete_child + try_reap",
    );

    // The kernel-watch protocol stays balanced through the single-protocol vacate terminus: exactly
    // one Unwatch(b), no other op references the reaped descendant.
    let unwatch_b = purge_out
        .watch_ops
        .iter()
        .filter(|op| matches!(op, WatchOp::Unwatch { resource } if *resource == b_id))
        .count();
    assert_eq!(
        unwatch_b, 1,
        "exactly one Unwatch(b) closes b's watch via the vacate terminus; \
         got {:?}",
        purge_out.watch_ops,
    );

    // Profile reverts to anchor-loss state: anchor_claim cleared, baseline / kind cleared,
    // watch_root_parent preserved (the recovery channel — but the anchor is a root in this fixture,
    // so `watch_root_parent` is None throughout).
    let p = e.profiles().get(pid).expect("Profile lives");
    assert_eq!(p.anchor_claim(), AnchorClaim::None);
    assert!(p.kind().is_none());
    assert!(p.baseline().is_none());
    assert!(p.current().is_none());

    // Profile is back to Idle (finish_burst_to_idle ran).
    assert!(matches!(p.state(), specter_core::ProfileState::Idle));
}

/// `release_anchor_claim` flips a materialised `Held` claim to `None` and is idempotent — a second
/// call is a no-op on both the claim and the Tree (the early-return guard on `anchor_claim`), so it
/// emits no further watch op. Isolates the helper that `discard_anchor_state` composes; pins the
/// materialise ↔ release symmetry directly.
#[test]
fn release_anchor_claim_is_symmetric_and_idempotent() {
    let (mut e, _sid, pid, _anchor, _parent) = engine_with_materialised_profile(ClassSet::EMPTY);

    assert_eq!(
        e.profiles().get(pid).unwrap().anchor_claim(),
        AnchorClaim::Held,
        "fixture materialised the anchor → claim Held",
    );

    let mut out = StepOutput::default();
    e.release_anchor_claim(pid, &mut out);
    assert_eq!(
        e.profiles().get(pid).unwrap().anchor_claim(),
        AnchorClaim::None,
        "release flips Held → None (symmetric with materialise)",
    );

    let mut out2 = StepOutput::default();
    e.release_anchor_claim(pid, &mut out2);
    assert_eq!(
        e.profiles().get(pid).unwrap().anchor_claim(),
        AnchorClaim::None,
        "second release is idempotent — stays None",
    );
    assert!(
        out2.watch_ops.is_empty(),
        "idempotent release emits no further Tree watch op: {:?}",
        out2.watch_ops,
    );
}
