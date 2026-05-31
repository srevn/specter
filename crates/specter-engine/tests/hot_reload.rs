//! Hot-reload via `Input::ConfigDiff`. Atomic apply of
//! `removed → modified → added`; reap-pending mid-burst handling;
//! in-flight Effect race after detach.

use compact_str::CompactString;
use specter_core::testkit::{dir_snap, empty_program};
use specter_core::{
    BurstFinish, DedupKey, Diagnostic, EffectCompletion, EffectOutcome, EffectScope, FsEvent,
    Input, ProbeOp, ResourceKind, ResourceRole, ScanConfig, SubAttachAnchor, SubAttachRequest,
    SubRegistryDiff, WatchOp, WatchRegistryDiff,
};
use specter_engine::Engine;
use specter_engine::testkit::{
    DEFAULT_EVENTS, MAX_SETTLE, NO_EVENTS, SETTLE, drain_due, seed_to_idle, verify,
};
use std::path::PathBuf;
use std::time::Instant;

#[test]
fn config_diff_add_sub_to_existing_profile() {
    // Engine has Sub A; ConfigDiff adds Sub B at the same anchor with the
    // same config — both share one Profile. The Profile's Sub count goes 1 → 2; no
    // new Watch/Probe.
    let mut e = Engine::new();
    let r = e.tree_mut().ensure_root("src", ResourceRole::User);
    e.tree_mut().set_kind(r, ResourceKind::Dir);
    let cfg = ScanConfig::builder().recursive(true).build();
    let now = Instant::now();
    let attach = e.step(
        Input::AttachSub(SubAttachRequest::for_anchor(
            "A".into(),
            SubAttachAnchor::Resource(r),
            cfg.clone(),
            MAX_SETTLE,
            SETTLE,
            empty_program(),
            EffectScope::SubtreeRoot,
            NO_EVENTS,
            false,
        )),
        now,
    );
    let sid_a = specter_core::testkit::first_attached_sub(&attach).expect("attach_sub succeeded");
    let pid = e.subs().get(sid_a).unwrap().profile();
    assert_eq!(e.subs().at(pid).len(), 1);

    // ConfigDiff with one added Sub at the same anchor + same cfg.
    let mut diff = SubRegistryDiff::default();
    diff.added.push(SubAttachRequest::for_anchor(
        "B".into(),
        SubAttachAnchor::Resource(r),
        cfg,
        MAX_SETTLE,
        SETTLE,
        empty_program(),
        EffectScope::SubtreeRoot,
        NO_EVENTS,
        false,
    ));
    let out = e.step(
        Input::ConfigDiff(WatchRegistryDiff {
            subs: diff,
            ..Default::default()
        }),
        now,
    );

    assert_eq!(e.subs().at(pid).len(), 2);
    let new_watches = out
        .watch_ops
        .iter()
        .filter(|op| matches!(op, WatchOp::Watch { .. }))
        .count();
    let new_probes = out
        .probe_ops()
        .iter()
        .filter(|op| matches!(op, ProbeOp::Probe { .. }))
        .count();
    assert_eq!(new_watches, 0, "no fresh Watch on existing Profile");
    assert_eq!(new_probes, 0, "no fresh Probe on existing Profile");
    let _ = e.cancel_all_in_flight_probes();
}

#[test]
fn config_diff_remove_sole_sub_reaps_profile() {
    // Engine has Sub A on its own Profile, post-Seed Idle. ConfigDiff
    // removes A. Profile reaped immediately (no Subs remain, Idle);
    // anchor unwatched.
    let mut e = Engine::new();
    let r = e.tree_mut().ensure_root("src", ResourceRole::User);
    e.tree_mut().set_kind(r, ResourceKind::Dir);

    let req = SubAttachRequest::for_anchor(
        "A".into(),
        SubAttachAnchor::Resource(r),
        ScanConfig::builder().build(),
        MAX_SETTLE,
        SETTLE,
        empty_program(),
        EffectScope::SubtreeRoot,
        NO_EVENTS,
        false,
    );
    let now = Instant::now();
    let attach_out = e.step(Input::AttachSub(req), now);
    let sid_a =
        specter_core::testkit::first_attached_sub(&attach_out).expect("attach_sub succeeded");
    let pid = e.subs().get(sid_a).unwrap().profile();
    let seed_done = seed_to_idle(&mut e, pid, &dir_snap(&[]), now);

    // Profile is Idle. Remove via ConfigDiff (by operator watch name).
    let post_seed = seed_done + SETTLE;
    let mut diff = SubRegistryDiff::default();
    diff.removed.push(CompactString::from("A"));
    let out = e.step(
        Input::ConfigDiff(WatchRegistryDiff {
            subs: diff,
            ..Default::default()
        }),
        post_seed,
    );

    assert!(e.profiles().get(pid).is_none(), "Profile reaped");
    assert!(out.diagnostics.iter().any(|d| matches!(
        d,
        Diagnostic::ProfileReaped {
            via: specter_core::ReapTrigger::Immediate,
            ..
        }
    )));
    let unwatches = out
        .watch_ops
        .iter()
        .filter(|op| matches!(op, WatchOp::Unwatch { .. }))
        .count();
    assert!(unwatches >= 1, "anchor unwatched");
}

#[test]
fn config_diff_mid_burst_remove_defers_reap() {
    // Engine has Sub A; Standard burst in flight; ConfigDiff removes A.
    // reap_pending=true; on burst-end, no Effect; Profile reaped.
    let mut e = Engine::new();
    let r = e.tree_mut().ensure_root("src", ResourceRole::User);
    e.tree_mut().set_kind(r, ResourceKind::Dir);
    let now = Instant::now();
    let attach_out = e.step(
        Input::AttachSub(SubAttachRequest::for_anchor(
            "A".into(),
            SubAttachAnchor::Resource(r),
            ScanConfig::builder().build(),
            MAX_SETTLE,
            SETTLE,
            empty_program(),
            EffectScope::SubtreeRoot,
            DEFAULT_EVENTS,
            false,
        )),
        now,
    );
    let sid_a =
        specter_core::testkit::first_attached_sub(&attach_out).expect("attach_sub succeeded");
    let pid = e.subs().get(sid_a).unwrap().profile();
    let seed_done = seed_to_idle(&mut e, pid, &dir_snap(&[]), now);

    // Drive a Standard burst (after the Seed's two settle windows).
    let t1 = seed_done + SETTLE;
    e.step(
        Input::FsEvent {
            resource: r,
            event: FsEvent::ContentChanged,
        },
        t1,
    );

    // Mid-burst ConfigDiff: remove A (by operator watch name).
    let mut diff = SubRegistryDiff::default();
    diff.removed.push(CompactString::from("A"));
    let _ = e.step(
        Input::ConfigDiff(WatchRegistryDiff {
            subs: diff,
            ..Default::default()
        }),
        t1,
    );
    assert!(
        matches!(
            e.profiles().get(pid).unwrap().state().burst_finish(),
            Some(BurstFinish::Reap)
        ),
        "reap deferred to burst end",
    );

    // Drain settle to enter Probing.
    let t2 = t1 + SETTLE * 2;
    drain_due(&mut e, t2);

    // The single Authoritative verify response folds to `Stable`;
    // reap-pending suppresses the Effect and finishes by reaping.
    let v = verify(&mut e, pid, &dir_snap(&[]), t2);
    let out = v.out;
    assert!(out.effects().is_empty(), "reap_pending suppresses Effect");
    assert!(
        e.profiles().get(pid).is_none(),
        "Profile reaped at burst end"
    );
}

/// Same identity (anchor + scan + max_settle + events), different
/// per-Sub fields ⇒ `modified_params`. Mid-Standard-burst, the engine
/// rebinds the live Sub in place: the burst stays on the same
/// `ProfileId`, the anchor's `watch_demand` is unchanged (no
/// Unwatch/re-Watch), no fresh probe is emitted (the existing settle
/// timer still owns the burst lifecycle), and `SubRebound` narrates
/// the edge. Under `modified_params` the path does not go through
/// detach+attach via the zombie-revival branch, so
/// `ReapPendingCancelled` must **not** appear.
#[test]
fn config_diff_modified_params_mid_burst_rebinds_in_place() {
    let mut e = Engine::new();
    let r = e.tree_mut().ensure_root("src", ResourceRole::User);
    e.tree_mut().set_kind(r, ResourceKind::Dir);
    let now = Instant::now();
    let cfg = ScanConfig::builder().build();
    let attach_out = e.step(
        Input::AttachSub(SubAttachRequest::for_anchor(
            "A".into(),
            SubAttachAnchor::Resource(r),
            cfg.clone(),
            MAX_SETTLE,
            SETTLE,
            empty_program(),
            EffectScope::SubtreeRoot,
            NO_EVENTS,
            false,
        )),
        now,
    );
    let sid_a =
        specter_core::testkit::first_attached_sub(&attach_out).expect("attach_sub succeeded");
    let pid = e.subs().get(sid_a).unwrap().profile();
    let seed_done = seed_to_idle(&mut e, pid, &dir_snap(&[]), now);

    // Drive a Standard burst (after the Seed's two settle windows).
    let t1 = seed_done + SETTLE;
    e.step(
        Input::FsEvent {
            resource: r,
            event: FsEvent::ContentChanged,
        },
        t1,
    );
    let watch_demand_before = e.tree().get(r).unwrap().watch_demand();

    // Mid-burst ConfigDiff: rebind the watch "A" via `modified_params`
    // (same identity; different per-Sub field — same `empty_program`
    // here, but the path is exercised by the bucket choice).
    let mut diff = SubRegistryDiff::default();
    diff.modified_params.push(SubAttachRequest::for_anchor(
        "A".into(),
        SubAttachAnchor::Resource(r),
        cfg,
        MAX_SETTLE,
        SETTLE,
        empty_program(),
        EffectScope::SubtreeRoot,
        NO_EVENTS,
        false,
    ));
    let out = e.step(
        Input::ConfigDiff(WatchRegistryDiff {
            subs: diff,
            ..Default::default()
        }),
        t1,
    );

    // Rebind preserves the SubId — `A` still resolves to `sid_a`.
    let sid_b = e.subs().find_by_name("A").expect("A still live");
    assert_eq!(sid_b, sid_a, "modified_params rebind preserves SubId");
    assert_eq!(
        e.subs().get(sid_a).unwrap().profile(),
        pid,
        "Sub stays on the same Profile",
    );
    assert_eq!(e.subs().at(pid).len(), 1, "exactly one live Sub (A)");
    assert_eq!(
        e.tree().get(r).unwrap().watch_demand(),
        watch_demand_before,
        "anchor watch_demand unchanged (no Unwatch/re-Watch)",
    );
    assert!(
        out.diagnostics
            .iter()
            .any(|d| matches!(d, Diagnostic::SubRebound { sub } if *sub == sid_a)),
        "SubRebound emitted for the rebound SubId; got {:?}",
        out.diagnostics,
    );
    assert!(
        !out.diagnostics
            .iter()
            .any(|d| matches!(d, Diagnostic::ReapPendingCancelled { .. })),
        "modified_params does not go through detach+reap_pending+revival; \
         ReapPendingCancelled must not appear",
    );
    let new_probes = out
        .probe_ops()
        .iter()
        .filter(|op| matches!(op, ProbeOp::Probe { .. }))
        .count();
    assert_eq!(
        new_probes, 0,
        "rebind doesn't emit probes — existing Standard burst's settle timer owns the lifecycle",
    );
}

/// Settle-change rebind on a post-Seed Idle Profile: the SubId, the
/// ProfileId, and the kernel watch all survive; `Profile.settle` is
/// recomputed to the new value; the `SubRebound` diagnostic narrates
/// the edge; no fresh probe or Unwatch is emitted. T2 in the
/// validate-then-act plan. (The orthogonal `has_fired`-preservation
/// claim is pinned by [`specter_core::SubRegistry::rebind`]'s own
/// unit test — driving a real fire here would over-couple the
/// integration test to actuator-side mechanics.)
#[test]
fn config_diff_modified_params_settle_change_recomputes_profile_settle() {
    let mut e = Engine::new();
    let r = e.tree_mut().ensure_root("src", ResourceRole::User);
    e.tree_mut().set_kind(r, ResourceKind::Dir);
    let now = Instant::now();
    let cfg = ScanConfig::builder().build();
    let attach_out = e.step(
        Input::AttachSub(SubAttachRequest::for_anchor(
            "A".into(),
            SubAttachAnchor::Resource(r),
            cfg.clone(),
            MAX_SETTLE,
            SETTLE,
            empty_program(),
            EffectScope::SubtreeRoot,
            NO_EVENTS,
            false,
        )),
        now,
    );
    let sid = specter_core::testkit::first_attached_sub(&attach_out).expect("attach_sub succeeded");
    let pid = e.subs().get(sid).unwrap().profile();
    let _ = seed_to_idle(&mut e, pid, &dir_snap(&[]), now);

    // Rebind to a longer settle. Everything else identical, including
    // the program Arc.
    let new_settle = SETTLE + SETTLE; // doubled
    let mut diff = SubRegistryDiff::default();
    diff.modified_params.push(SubAttachRequest::for_anchor(
        "A".into(),
        SubAttachAnchor::Resource(r),
        cfg,
        MAX_SETTLE,
        new_settle,
        empty_program(),
        EffectScope::SubtreeRoot,
        NO_EVENTS,
        false,
    ));
    let watch_demand_before = e.tree().get(r).unwrap().watch_demand();

    let post_seed = now + SETTLE + SETTLE + SETTLE; // safely past the seed window
    let out = e.step(
        Input::ConfigDiff(WatchRegistryDiff {
            subs: diff,
            ..Default::default()
        }),
        post_seed,
    );

    let sub_after = e.subs().get(sid).expect("Sub preserved across rebind");
    assert_eq!(
        sub_after.profile(),
        pid,
        "rebind preserves the Sub's ProfileId",
    );
    assert_eq!(sub_after.settle, new_settle, "Sub.settle is the new value");
    assert_eq!(
        e.profiles().get(pid).unwrap().settle,
        new_settle,
        "Profile.settle recomputed to the new min over live Subs",
    );
    assert_eq!(
        e.tree().get(r).unwrap().watch_demand(),
        watch_demand_before,
        "rebind doesn't touch kernel watches",
    );
    let new_probes = out
        .probe_ops()
        .iter()
        .filter(|op| matches!(op, ProbeOp::Probe { .. }))
        .count();
    assert_eq!(new_probes, 0, "rebind doesn't probe");
    assert!(
        !out.watch_ops
            .iter()
            .any(|op| matches!(op, WatchOp::Unwatch { .. })),
        "rebind doesn't Unwatch the anchor (silent biggest win over identity arm)",
    );
    assert!(
        out.diagnostics
            .iter()
            .any(|d| matches!(d, Diagnostic::SubRebound { sub } if *sub == sid)),
        "SubRebound emitted; got {:?}",
        out.diagnostics,
    );
}

/// T1 — malformed `modified_identity` request leaves the old Sub in
/// place. The validate-then-act composition pre-checks the new
/// anchor's parse; on failure the engine emits
/// `AttachPathInvalid`, the detach never runs, and the live
/// attachment survives unchanged. Structural rollback at the
/// composition layer.
#[test]
fn config_diff_modified_identity_validate_failure_leaves_old_sub_in_place() {
    let mut e = Engine::new();
    let r = e
        .tree_mut()
        .ensure_path(&["/", "src"], ResourceRole::User)
        .expect("non-empty fixture");
    e.tree_mut().set_kind(r, ResourceKind::Dir);
    let now = Instant::now();
    let attach_out = e.step(
        Input::AttachSub(SubAttachRequest::for_anchor(
            "A".into(),
            SubAttachAnchor::Path(PathBuf::from("/src")),
            ScanConfig::builder().recursive(true).build(),
            MAX_SETTLE,
            SETTLE,
            empty_program(),
            EffectScope::SubtreeRoot,
            NO_EVENTS,
            false,
        )),
        now,
    );
    let sid = specter_core::testkit::first_attached_sub(&attach_out).expect("attach_sub succeeded");
    let pid = e.subs().get(sid).unwrap().profile();
    let _ = seed_to_idle(&mut e, pid, &dir_snap(&[]), now);

    // modified_identity with a malformed path — `Tree::parse_attach_path`
    // rejects relative paths, so this fails validation.
    let mut diff = SubRegistryDiff::default();
    diff.modified_identity.push(SubAttachRequest::for_anchor(
        "A".into(),
        SubAttachAnchor::Path(PathBuf::from("relative/path")),
        ScanConfig::builder().recursive(false).build(),
        MAX_SETTLE,
        SETTLE,
        empty_program(),
        EffectScope::SubtreeRoot,
        NO_EVENTS,
        false,
    ));
    let post_seed = now + SETTLE + SETTLE + SETTLE;
    let out = e.step(
        Input::ConfigDiff(WatchRegistryDiff {
            subs: diff,
            ..Default::default()
        }),
        post_seed,
    );

    assert!(
        out.diagnostics
            .iter()
            .any(|d| matches!(d, Diagnostic::AttachPathInvalid { .. })),
        "validate emits AttachPathInvalid on failure; got {:?}",
        out.diagnostics,
    );
    assert_eq!(
        e.subs().find_by_name("A"),
        Some(sid),
        "old SubId survives validate failure — structural rollback",
    );
    assert_eq!(
        e.subs().get(sid).unwrap().profile(),
        pid,
        "old Profile unchanged",
    );
    assert!(
        !out.watch_ops
            .iter()
            .any(|op| matches!(op, WatchOp::Unwatch { .. })),
        "no Unwatch emitted — the detach didn't run",
    );
    let _ = e.cancel_all_in_flight_probes();
}

#[test]
fn effect_complete_after_detach_drops_silently() {
    // Engine has Sub on Idle Profile; an Effect was previously emitted
    // (we mock the EffectComplete path manually). Detach the Sub; then
    // inject EffectComplete for the now-removed Sub. Engine drops with
    // a Diagnostic — no panic, no reseed.
    let mut e = Engine::new();
    let r = e.tree_mut().ensure_root("src", ResourceRole::User);
    e.tree_mut().set_kind(r, ResourceKind::Dir);
    let now = Instant::now();
    let attach_out = e.step(
        Input::AttachSub(SubAttachRequest::for_anchor(
            "A".into(),
            SubAttachAnchor::Resource(r),
            ScanConfig::builder().build(),
            MAX_SETTLE,
            SETTLE,
            empty_program(),
            EffectScope::SubtreeRoot,
            NO_EVENTS,
            false,
        )),
        now,
    );
    let sid = specter_core::testkit::first_attached_sub(&attach_out).expect("attach_sub succeeded");
    let pid = e.subs().get(sid).unwrap().profile();
    let seed_done = seed_to_idle(&mut e, pid, &dir_snap(&[]), now);

    // Detach via ConfigDiff (by operator watch name).
    let post_seed = seed_done + SETTLE;
    let mut diff = SubRegistryDiff::default();
    diff.removed.push(CompactString::from("A"));
    e.step(
        Input::ConfigDiff(WatchRegistryDiff {
            subs: diff,
            ..Default::default()
        }),
        post_seed,
    );

    // Inject EffectComplete for the now-removed Sub.
    let out = e.step(
        Input::EffectComplete(EffectCompletion {
            sub: sid,
            key: DedupKey::Subtree {
                sub: sid,
                profile: pid,
            },
            outcome: EffectOutcome::Ok,
        }),
        post_seed,
    );

    assert!(
        out.diagnostics
            .iter()
            .any(|d| matches!(d, Diagnostic::EffectCompleteForUnknownSub { .. }))
    );
    // No Probe re-emitted (no reseed).
    let new_probes = out
        .probe_ops()
        .iter()
        .filter(|op| matches!(op, ProbeOp::Probe { .. }))
        .count();
    assert_eq!(new_probes, 0);
}

/// T4 — same-path identity change exercises `modified_identity`
/// against the prepare/commit bug an earlier plan would have
/// introduced. Sub "A" at `/src` with `recursive=true`; reload edits
/// to `recursive=false` — same path, different `config_hash` ⇒ the
/// Sub must move to a different Profile. The composition is
/// validate → detach → attach: the old Profile is reaped (last Sub
/// left) and the new Sub mints on a fresh Profile.
///
/// The historical bug was that a prepare/commit split would observe
/// the old Profile's anchor claim getting released by the detach,
/// reaping the slot mid-operation; the commit then panicked
/// `expect("resource has no live Tree slot")`. Validate-then-act
/// captures no engine state, so the attach re-materializes the slot
/// via `ensure_path` and no panic is possible.
#[test]
fn config_diff_modified_identity_same_path_rebinds_profile_safely() {
    let mut e = Engine::new();
    let r = e
        .tree_mut()
        .ensure_path(&["/", "src"], ResourceRole::User)
        .expect("non-empty fixture");
    e.tree_mut().set_kind(r, ResourceKind::Dir);
    let now = Instant::now();
    let attach_out = e.step(
        Input::AttachSub(SubAttachRequest::for_anchor(
            "A".into(),
            SubAttachAnchor::Resource(r),
            ScanConfig::builder().recursive(true).build(),
            MAX_SETTLE,
            SETTLE,
            empty_program(),
            EffectScope::SubtreeRoot,
            NO_EVENTS,
            false,
        )),
        now,
    );
    let sid_a =
        specter_core::testkit::first_attached_sub(&attach_out).expect("attach_sub succeeded");
    let pid_a = e.subs().get(sid_a).unwrap().profile();
    let seed_done = seed_to_idle(&mut e, pid_a, &dir_snap(&[]), now);

    // Same path, different scan ⇒ `modified_identity`. Path-based
    // anchor to exercise the re-materialise-after-reap path that the
    // prepare/commit shape would have broken.
    let post_seed = seed_done + SETTLE;
    let mut diff = SubRegistryDiff::default();
    diff.modified_identity.push(SubAttachRequest::for_anchor(
        "A".into(),
        SubAttachAnchor::Path(PathBuf::from("/src")),
        ScanConfig::builder().recursive(false).build(),
        MAX_SETTLE,
        SETTLE,
        empty_program(),
        EffectScope::SubtreeRoot,
        NO_EVENTS,
        false,
    ));
    let _out = e.step(
        Input::ConfigDiff(WatchRegistryDiff {
            subs: diff,
            ..Default::default()
        }),
        post_seed,
    );

    // Old Sub gone; new Sub at the same name resolves to a fresh
    // id on a fresh Profile (different `config_hash`).
    assert!(e.subs().get(sid_a).is_none(), "old Sub detached");
    let sid_b = e.subs().find_by_name("A").expect("A re-attached");
    assert_ne!(sid_b, sid_a, "fresh SubId minted on identity change");
    assert_eq!(e.subs().len(), 1, "exactly one Sub remains");
    let _ = e.cancel_all_in_flight_probes();
}
