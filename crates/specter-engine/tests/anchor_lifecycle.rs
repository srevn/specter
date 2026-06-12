//! Anchor-loss lifecycle: terminal-driven recovery end-to-end.
//!
//! An *observed* loss — an anchor terminal event, or probe `Vanished` — re-enters pending descent
//! at the `watch_root_parent` inside the loss step itself, `witnessed = true`. The descent
//! re-classifies the anchor from the parent listing and materialises into a *triggered* Seed
//! (Batching-first), whose stable verdict owes the recovery fire: `RecoveryFire` for a fired Sub
//! (survival-witness drift), `FreshSeedFire` for a never-fired one (the loss-entry latch). The
//! replace family pins the consequences: atomic replaces re-fire with post-replace content, save
//! storms debounce to one fire per settle window, repeated terminals loop finalize → descend → Seed
//! safely, a descendant-LCA `Vanished` resolves a live anchor in one hop, and a delete-then-write
//! save parks the descent (narrated) until the create lands.
//!
//! The probe-shape pins: `discard_anchor_state` clears `Profile.kind` at every loss, so a stale
//! kind cannot misroute the recovery probe (`Some(File)` against a recreated-as-Dir slot would emit
//! a wasted `ProbeRequest::AnchorFile`). The two loss flavors then diverge:
//!
//! - **Observed loss**: the parent's directory listing re-classifies the anchor's kind *before* any
//!   anchor probe is emitted — the recovery Seed probes with the freshly-observed shape in both
//!   flip directions (File→Dir ⇒ Subtree; Dir→File ⇒ AnchorFile), one anchor probe per recovery.
//! - **Probe `Failed`**: the Profile parks anchorless (`ProfileState::Parked`) with `kind = None`.
//!   A recovery event at the anchor slot re-enters descent through the event-scan recovery arm; the
//!   re-classifying descent feeds the kind-agnostic Subtree arm of the materialised Seed.

use specter_core::testkit::{anchor_ok, dir_snap, file_leaf, proven};
use specter_core::{
    AnchorClaim, ClassSet, Diagnostic, EntryKind, FsEvent, Input, ProbeFailure, ProbeOp,
    ProbeOutcome, ProbeRequest, ProbeResponse, ProfileId, ProfileState, ResourceId, ResourceKind,
    ResourceRole, ScanConfig, StepOutput, SubAttachAnchor, SubId,
};
use specter_engine::Engine;
use specter_engine::testkit::{
    assert_seed_verifying, attach_returning, complete_effect_to_rebasing, descent_advance,
    drain_due, fire_standard_once, first_probe_correlation, pre_place_dir, respond_anchor_file,
    seed_to_idle, seed_to_idle_with,
};
use std::time::{Duration, Instant};

const SETTLE: Duration = Duration::from_millis(100);
const MAX_SETTLE: Duration = Duration::from_secs(6);

fn first_probe_request(out: &StepOutput) -> Option<&ProbeRequest> {
    out.probe_ops().iter().find_map(|op| match op {
        ProbeOp::Probe { request } => Some(request),
        ProbeOp::Cancel { .. } => None,
    })
}

fn count_probes(out: &StepOutput) -> usize {
    out.probe_ops()
        .iter()
        .filter(|op| matches!(op, ProbeOp::Probe { .. }))
        .count()
}

/// [`attach_returning`] at a pre-placed anchor with the suite's recursive `Subtree` config.
fn attach_at(
    e: &mut Engine,
    name: &str,
    anchor: ResourceId,
    events: ClassSet,
    max_settle: Duration,
    now: Instant,
) -> (SubId, ProfileId, StepOutput) {
    attach_returning(
        e,
        name,
        SubAttachAnchor::Resource(anchor),
        ScanConfig::builder().recursive(true).build(),
        events,
        max_settle,
        now,
    )
}

/// Pre-place `/watch/app.log` — a File anchor under a Dir parent, the atomic-save fixture shape.
fn place_file_anchor(e: &mut Engine) -> (ResourceId, ResourceId) {
    let parent = pre_place_dir(e, &["watch"]);
    let anchor = e
        .tree_mut()
        .ensure_child(parent, "app.log", ResourceRole::User)
        .expect("test live parent");
    e.tree_mut().set_kind(anchor, ResourceKind::File);
    (parent, anchor)
}

/// One atomic-replace cycle against a single-Profile file watch: the anchor terminal re-enters
/// descent (witnessed) inside the loss step; the rename's parent STRUCTURE notification is absorbed
/// by the in-flight descent (I5); the descent finds the replacement (`inode`) and materialises into
/// a triggered Seed — Batching-first — whose settle expiry surfaces the verify probe. Returns
/// (effects fired by the recovery verdict, instant after).
fn replace_cycle(
    e: &mut Engine,
    pid: ProfileId,
    parent: ResourceId,
    inode: u64,
    now: Instant,
) -> (usize, Instant) {
    let anchor = e.profiles().get(pid).unwrap().resource();
    let t1 = now + Duration::from_millis(10);
    let _ = e.step(
        Input::FsEvent {
            resource: anchor,
            event: FsEvent::Removed,
        },
        t1,
    );
    {
        let p = e.profiles().get(pid).expect("profile survives anchor loss");
        assert!(
            matches!(p.state(), ProfileState::Pending(_)),
            "observed loss re-enters descent in the loss step itself",
        );
        assert!(p.current().is_none());
    }
    // Parent STRUCTURE event (the rename's dir notification): the live descent absorbs it — the
    // latch is already set and the I5 gate drops the re-probe.
    let t2 = t1 + Duration::from_millis(1);
    let _ = e.step(
        Input::FsEvent {
            resource: parent,
            event: FsEvent::StructureChanged,
        },
        t2,
    );
    // Descent finds the replacement file -> materialize -> triggered Seed (Batching-first).
    let out = descent_advance(e, pid, Some(EntryKind::File), t2);
    assert!(out.effects().is_empty(), "descent itself never fires");
    assert!(
        e.pending_probe_for(pid).is_none(),
        "triggered Seed opens Batching-first — no cold walk in flight",
    );
    // Settle expiry -> Verifying; the response folds the recovery verdict.
    let t3 = t2 + SETTLE * 2;
    drain_due(e, t3);
    let out = respond_anchor_file(e, pid, inode, t3);
    let fired = out.effects().len();
    assert!(matches!(
        e.profiles().get(pid).unwrap().state(),
        ProfileState::Idle | ProfileState::Active(_, _)
    ));
    (fired, t3 + SETTLE)
}

#[test]
fn recovery_from_file_to_dir_anchor_uses_subtree_probe() {
    // Multi-Profile sharing a File-classified anchor. Profile P loses its anchor via probe
    // Vanished; `discard_anchor_state` clears `kind`. With Q's anchor claim keeping the watch
    // alive, a subsequent FsEvent at the anchor routes through `drive_burst` into
    // `start_seed_burst` for P (Idle, current=None). Post-fix: kind=None, start_seed_burst routes
    // through the kind-agnostic Subtree arm — recovery in one round-trip via descent regardless of
    // the recreated anchor's shape. Pre-fix the cached `Some(File)` misrouted as a
    // `ProbeRequest::AnchorFile` and wasted a round-trip.
    let mut e = Engine::new();
    let parent = e.tree_mut().ensure_root("var", ResourceRole::User);
    e.tree_mut().set_kind(parent, ResourceKind::Dir);
    let anchor = e
        .tree_mut()
        .ensure_child(parent, "log", ResourceRole::User)
        .expect("test live parent");
    e.tree_mut().set_kind(anchor, ResourceKind::File);

    // Q first — completes its Seed (File anchor → AnchorOk) → Idle, kind=Some(File).
    let t_q = Instant::now();
    let (_sid_q, pid_q, out_q) = attach_at(
        &mut e,
        "Q",
        anchor,
        ClassSet::EMPTY,
        MAX_SETTLE + Duration::from_secs(1),
        t_q,
    );
    assert!(
        first_probe_correlation(&out_q).is_some(),
        "cold-arm Seed: probe emitted at burst construction",
    );
    let _ = seed_to_idle_with(
        &mut e,
        pid_q,
        || anchor_ok(file_leaf(EntryKind::File, 1)),
        t_q,
    );
    assert_eq!(
        e.profiles().get(pid_q).unwrap().kind(),
        Some(ResourceKind::File),
    );

    // P next — Active(PreFire(Seed Batching)) right after attach (no probe yet). Attach strictly
    // after Q's two settle windows.
    let t_p = t_q + SETTLE * 3;
    let (_sid_p, pid_p, out_p) = attach_at(&mut e, "P", anchor, ClassSet::EMPTY, MAX_SETTLE, t_p);
    assert!(
        first_probe_correlation(&out_p).is_some(),
        "cold-arm Seed: probe emitted at burst construction",
    );
    // `Profile.kind` is pinned at construction from the anchor's classified kind, independent of
    // the Batching-first Seed.
    assert_eq!(
        e.profiles().get(pid_p).unwrap().kind(),
        Some(ResourceKind::File),
    );
    // Both Profiles claim the anchor → watch_demand = 2 (the claim is bumped at attach by
    // `bootstrap_immediate`, before the Seed).
    assert_eq!(e.tree().get(anchor).unwrap().watch_demand(), 2);

    // Expire P's first settle window → first Seed probe, then drive it to Vanished.
    // discard_anchor_state clears P.kind, P.current, P.baseline, P.anchor_claim. Vanished
    // terminates the Seed on its first response.
    let (p_corr, p_at) = assert_seed_verifying(&mut e, pid_p, t_p);
    e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid_p,
            correlation: p_corr,
            outcome: ProbeOutcome::Vanished,
        }),
        p_at,
    );
    let p = e.profiles().get(pid_p).expect("P alive");
    assert!(p.kind().is_none(), "P.kind cleared by discard_anchor_state");
    assert!(p.current().is_none());
    assert!(p.baseline().is_none());
    assert_eq!(p.anchor_claim(), AnchorClaim::None);
    assert!(
        matches!(p.state(), ProfileState::Pending(_)),
        "observed loss re-enters descent in the loss step itself",
    );
    // Q's claim keeps the anchor alive.
    assert_eq!(e.tree().get(anchor).unwrap().watch_demand(), 1);

    // Answer the loss step's descent probe: the segment re-classifies the anchor from the live
    // filesystem — `log` is a Dir now. Materialization re-reads kind from the segment, so the stale
    // `Some(File)` cannot leak into the recovery probe's shape; the witnessed descent opens a
    // triggered Seed (Batching-first, no cold walk).
    let mat_at = p_at + SETTLE;
    let mat_out = descent_advance(&mut e, pid_p, Some(EntryKind::Dir), mat_at);
    assert_eq!(
        e.profiles().get(pid_p).unwrap().kind(),
        Some(ResourceKind::Dir),
        "kind re-classified from the parent's directory listing",
    );
    assert_eq!(
        count_probes(&mat_out),
        0,
        "witnessed descent materializes into a triggered Seed — Batching-first",
    );

    // Expire the recovery Seed's settle window → the recovery Seed probe materializes. The
    // freshly-observed Dir routes it through the Subtree arm; a stale `Some(File)` would have
    // emitted a `ProbeRequest::AnchorFile`.
    let probe_at = mat_at + SETTLE;
    let mut p_probe_out = None;
    while let Some(en) = e.pop_expired(probe_at) {
        let o = e.step(
            Input::TimerExpired {
                profile: en.profile,
                kind: en.kind,
                id: en.id,
            },
            probe_at,
        );
        if first_probe_request(&o).is_some() {
            p_probe_out = Some(o);
        }
    }
    let p_probe_out = p_probe_out.expect("recovery Seed probe after settle expiry");
    let p_probe = p_probe_out
        .probe_ops()
        .iter()
        .find_map(|op| match op {
            ProbeOp::Probe { request } if request.owner() == pid_p => Some(request),
            _ => None,
        })
        .expect("P emits a recovery Seed probe");
    assert!(
        matches!(p_probe, ProbeRequest::Subtree { .. }),
        "freshly-observed Dir routes recovery through the Subtree probe; got {p_probe:?}",
    );
    let _ = e.cancel_all_in_flight_probes();
}

#[test]
fn recovery_from_dir_to_file_anchor_bounded_to_one_round_trip() {
    // The Dir→File flip direction: the loss step's descent re-classifies the anchor as a File from
    // the parent listing, so the recovery Seed probes `AnchorFile` — the cheap lstat shape — and
    // the recovery stays bounded at one anchor probe.
    let mut e = Engine::new();
    let parent = e.tree_mut().ensure_root("src", ResourceRole::User);
    e.tree_mut().set_kind(parent, ResourceKind::Dir);
    let anchor = e
        .tree_mut()
        .ensure_child(parent, "build", ResourceRole::User)
        .expect("test live parent");
    e.tree_mut().set_kind(anchor, ResourceKind::Dir);

    // Q's Seed (Dir anchor → SubtreeProven) → Idle.
    let t_q = Instant::now();
    let (_sid_q, pid_q, out_q) = attach_at(
        &mut e,
        "Q",
        anchor,
        ClassSet::EMPTY,
        MAX_SETTLE + Duration::from_secs(1),
        t_q,
    );
    assert!(
        first_probe_correlation(&out_q).is_some(),
        "cold-arm Seed: probe emitted at burst construction",
    );
    let _ = seed_to_idle(&mut e, pid_q, &dir_snap(&[]), t_q);

    // P attaches strictly after Q's two settle windows; Batching-first (no probe at attach).
    // `Profile.kind` is pinned at construction from the anchor's classified kind, independent of
    // the Seed shape.
    let t_p = t_q + SETTLE * 3;
    let (_sid_p, pid_p, out_p) = attach_at(&mut e, "P", anchor, ClassSet::EMPTY, MAX_SETTLE, t_p);
    assert!(
        first_probe_correlation(&out_p).is_some(),
        "cold-arm Seed: probe emitted at burst construction",
    );
    assert_eq!(
        e.profiles().get(pid_p).unwrap().kind(),
        Some(ResourceKind::Dir),
    );

    // Expire P's first settle window → first Seed probe; drive it to Vanished. The loss step clears
    // P.kind and re-enters descent at the parent.
    let (p_corr, p_at) = assert_seed_verifying(&mut e, pid_p, t_p);
    e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid_p,
            correlation: p_corr,
            outcome: ProbeOutcome::Vanished,
        }),
        p_at,
    );
    assert!(e.profiles().get(pid_p).unwrap().kind().is_none());
    assert!(matches!(
        e.profiles().get(pid_p).unwrap().state(),
        ProfileState::Pending(_),
    ));

    // The descent probe re-classifies the anchor: `build` is a File now. Materialization opens the
    // triggered Seed Batching-first.
    let mat_at = p_at + SETTLE;
    let mat_out = descent_advance(&mut e, pid_p, Some(EntryKind::File), mat_at);
    assert_eq!(
        e.profiles().get(pid_p).unwrap().kind(),
        Some(ResourceKind::File),
        "kind re-classified from the descent probe's segment answer",
    );
    assert_eq!(count_probes(&mat_out), 0, "triggered Seed: Batching-first");

    // Expire the recovery Seed's settle window. The bound: P's recovery emits exactly one anchor
    // probe, and the freshly-observed File routes it through the cheap `AnchorFile` lstat shape.
    let probe_at = mat_at + SETTLE;
    let mut settle_out = None;
    while let Some(en) = e.pop_expired(probe_at) {
        let o = e.step(
            Input::TimerExpired {
                profile: en.profile,
                kind: en.kind,
                id: en.id,
            },
            probe_at,
        );
        if first_probe_request(&o).is_some() {
            settle_out = Some(o);
        }
    }
    let settle_out = settle_out.expect("recovery Seed probe after settle expiry");
    let p_probe_count = mat_out
        .probe_ops()
        .iter()
        .chain(settle_out.probe_ops().iter())
        .filter(|op| matches!(op, ProbeOp::Probe { request } if request.owner() == pid_p))
        .count();
    assert_eq!(
        p_probe_count, 1,
        "exactly one anchor probe emitted for P during recovery",
    );
    let p_probe = first_probe_request(&settle_out).expect("recovery probe emitted");
    assert!(
        matches!(p_probe, ProbeRequest::AnchorFile { .. }),
        "Dir→File recovery probes the freshly-observed File shape; got {p_probe:?}",
    );
    let _ = e.cancel_all_in_flight_probes();
}

#[test]
fn anchor_loss_via_probe_failed_clears_kind_and_recovers_via_subtree() {
    // Mirror of `recovery_from_file_to_dir_anchor_uses_subtree_probe` for the Failed dispatch path.
    // dispatch_pre_fire_failed shares the helper; the post-recovery probe must be Subtree.
    let mut e = Engine::new();
    let parent = e.tree_mut().ensure_root("var", ResourceRole::User);
    e.tree_mut().set_kind(parent, ResourceKind::Dir);
    let anchor = e
        .tree_mut()
        .ensure_child(parent, "log", ResourceRole::User)
        .expect("test live parent");
    e.tree_mut().set_kind(anchor, ResourceKind::File);

    // Q's Seed (File anchor → AnchorOk) → Idle.
    let t_q = Instant::now();
    let (_sid_q, pid_q, out_q) = attach_at(
        &mut e,
        "Q",
        anchor,
        ClassSet::EMPTY,
        MAX_SETTLE + Duration::from_secs(1),
        t_q,
    );
    assert!(
        first_probe_correlation(&out_q).is_some(),
        "cold-arm Seed: probe emitted at burst construction",
    );
    let _ = seed_to_idle_with(
        &mut e,
        pid_q,
        || anchor_ok(file_leaf(EntryKind::File, 1)),
        t_q,
    );

    // P attaches strictly after Q's two settle windows; Batching-first.
    let t_p = t_q + SETTLE * 3;
    let (_sid_p, pid_p, out_p) = attach_at(&mut e, "P", anchor, ClassSet::EMPTY, MAX_SETTLE, t_p);
    assert!(
        first_probe_correlation(&out_p).is_some(),
        "cold-arm Seed: probe emitted at burst construction",
    );

    // Expire P's first settle window → first Seed probe; drive it to Failed. dispatch_pre_fire_failed
    // clears P.kind and terminates the Seed on its first response.
    let (p_corr, p_at) = assert_seed_verifying(&mut e, pid_p, t_p);
    e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid_p,
            correlation: p_corr,
            outcome: ProbeOutcome::Failed(ProbeFailure::Anchor { errno: 5 }),
        }),
        p_at,
    );
    assert!(e.profiles().get(pid_p).unwrap().kind().is_none());

    // The park left P `Parked`-anchorless. A recovery FsEvent at the anchor slot routes through the
    // event-scan recovery arm (which selects a `Parked` Profile whose own anchor slot is the event
    // resource) into a Pending descent — the descent probe materializes at the parent prefix.
    let recovery_t0 = p_at + SETTLE;
    let recovery_out = e.step(
        Input::FsEvent {
            resource: anchor,
            event: FsEvent::ContentChanged,
        },
        recovery_t0,
    );
    assert!(
        matches!(
            e.profiles().get(pid_p).unwrap().state(),
            ProfileState::Pending(_)
        ),
        "recovery from a probe-Failed park re-enters descent",
    );
    assert_eq!(
        count_probes(&recovery_out),
        1,
        "the recovery descent probes the parent prefix immediately",
    );

    // The descent re-classifies the anchor as a Dir (File→Dir flip) → cold Seed (Verifying-first,
    // unwitnessed recovery). The Seed's probe materializes inline with the descent's terminal
    // response and must be Subtree-shaped: the Failed-driven discard left kind=None, and the
    // descent observed a Dir.
    let settle_out = descent_advance(&mut e, pid_p, Some(EntryKind::Dir), recovery_t0);
    let p_probe = settle_out
        .probe_ops()
        .iter()
        .find_map(|op| match op {
            ProbeOp::Probe { request } if request.owner() == pid_p => Some(request),
            _ => None,
        })
        .expect("P emits a recovery Seed probe");
    assert!(matches!(p_probe, ProbeRequest::Subtree { .. }));
    let _ = e.cancel_all_in_flight_probes();
}

/// A static file watch that HAS fired: an atomic replace re-fires via `RecoveryFire` — the fired
/// Sub's survival witness drifts against the post-graft replacement.
#[test]
fn replace_of_fired_anchor_recovery_fires() {
    let mut e = Engine::new();
    let (parent, anchor) = place_file_anchor(&mut e);

    let now = Instant::now();
    let (sid, pid, _) = attach_at(
        &mut e,
        "static-fired",
        anchor,
        ClassSet::DEFAULT_SUBTREE_ROOT,
        MAX_SETTLE,
        now,
    );
    let t0 = seed_to_idle_with(
        &mut e,
        pid,
        || anchor_ok(file_leaf(EntryKind::File, 1)),
        now,
    );
    let t1 = fire_standard_once(&mut e, sid, anchor, 2, t0 + SETTLE);
    assert!(e.subs().get(sid).unwrap().has_fired());

    // Replace 2 -> 3: recovery must re-fire.
    let (fired, _t2) = replace_cycle(&mut e, pid, parent, 3, t1 + SETTLE);
    assert_eq!(fired, 1, "fired Sub: replace -> RecoveryFire re-fires");
    let _ = e.cancel_all_in_flight_probes();
}

/// A static file watch that has NEVER fired: the loss-entry latch makes the terminal-driven descent
/// witnessed, so the first replace classifies `FreshSeedFire` and fires once the settle window
/// passes — `has_fired` flips through a replace.
#[test]
fn replace_of_never_fired_anchor_fires_first_fire() {
    let mut e = Engine::new();
    let (parent, anchor) = place_file_anchor(&mut e);

    let now = Instant::now();
    let (sid, pid, _) = attach_at(
        &mut e,
        "static-unfired",
        anchor,
        ClassSet::DEFAULT_SUBTREE_ROOT,
        MAX_SETTLE,
        now,
    );
    let t0 = seed_to_idle_with(
        &mut e,
        pid,
        || anchor_ok(file_leaf(EntryKind::File, 1)),
        now,
    );

    // Replace 1 -> 2: the witnessed loss owes — and fires — the first fire.
    let (fired, t1) = replace_cycle(&mut e, pid, parent, 2, t0 + SETTLE);
    assert_eq!(
        fired, 1,
        "never-fired Sub: the witnessed replace fires (FreshSeedFire)",
    );
    assert!(e.subs().get(sid).unwrap().has_fired());

    // Drain the fire cycle: effect Ok -> rebase -> Idle.
    let key = specter_core::DedupKey::Subtree {
        sub: sid,
        profile: pid,
    };
    let _ = complete_effect_to_rebasing(&mut e, sid, key, t1);
    let _ = respond_anchor_file(&mut e, pid, 2, t1);
    assert!(matches!(
        e.profiles().get(pid).unwrap().state(),
        ProfileState::Idle
    ));
    let _ = e.cancel_all_in_flight_probes();
}

/// Ordering race: the rename's parent STRUCTURE notification often *precedes* the anchor terminal.
/// The early parent event is a no-op against a healthy Profile (anchor present — not a recovery
/// candidate), and the terminal itself then drives the descent, so recovery completes with NO
/// post-terminal parent event at all — the live-daemon stall is structurally dead.
#[test]
fn parent_event_before_terminal_still_recovers_and_fires() {
    let mut e = Engine::new();
    let (parent, anchor) = place_file_anchor(&mut e);

    let now = Instant::now();
    let (sid, pid, _) = attach_at(
        &mut e,
        "ordering-race",
        anchor,
        ClassSet::DEFAULT_SUBTREE_ROOT,
        MAX_SETTLE,
        now,
    );
    let t0 = seed_to_idle_with(
        &mut e,
        pid,
        || anchor_ok(file_leaf(EntryKind::File, 1)),
        now,
    );
    let t1 = fire_standard_once(&mut e, sid, anchor, 2, t0 + SETTLE);

    // The parent notification lands FIRST: the Profile is still healthy, so nothing moves.
    let t2 = t1 + Duration::from_millis(10);
    let _ = e.step(
        Input::FsEvent {
            resource: parent,
            event: FsEvent::StructureChanged,
        },
        t2,
    );
    assert!(matches!(
        e.profiles().get(pid).unwrap().state(),
        ProfileState::Idle
    ));
    assert!(e.pending_probe_for(pid).is_none());

    // The terminal lands second — and is itself the recovery driver. No further parent event is
    // delivered for the rest of the test.
    let t3 = t2 + Duration::from_millis(1);
    let _ = e.step(
        Input::FsEvent {
            resource: anchor,
            event: FsEvent::Removed,
        },
        t3,
    );
    assert!(matches!(
        e.profiles().get(pid).unwrap().state(),
        ProfileState::Pending(_)
    ));
    let out = descent_advance(&mut e, pid, Some(EntryKind::File), t3);
    assert!(out.effects().is_empty());
    let t4 = t3 + SETTLE * 2;
    drain_due(&mut e, t4);
    let out = respond_anchor_file(&mut e, pid, 3, t4);
    assert_eq!(
        out.effects().len(),
        1,
        "recovery fires with no post-terminal parent event",
    );
    let _ = e.cancel_all_in_flight_probes();
}

/// `dispatch_pre_fire_vanished` (Standard route) at a descendant LCA while the anchor survives: an
/// `rm -rf` racing the walk yields `Vanished` at the dirty-LCA descendant. The descent resolves the
/// ambiguity in one hop — the anchor re-materializes from the parent listing and the triggered Seed
/// fires (the `rm` was a change) instead of the watch parking dead.
#[test]
fn standard_vanished_at_descendant_lca_recovers_live_anchor_and_fires() {
    let mut e = Engine::new();
    let dir = pre_place_dir(&mut e, &["watch", "data"]);
    let now = Instant::now();
    let (sid, pid, _) = attach_at(
        &mut e,
        "descendant-lca",
        dir,
        ClassSet::DEFAULT_SUBTREE_ROOT,
        MAX_SETTLE,
        now,
    );
    let t0 = seed_to_idle_with(
        &mut e,
        pid,
        || proven(dir_snap(&[("f.txt", EntryKind::File, 1)])),
        now,
    );
    let child = e
        .tree()
        .lookup(Some(dir), "f.txt")
        .expect("per-file reconcile created the child slot");

    // A change at the child opens a Standard burst whose dirty-LCA — and probe target — is the
    // child itself.
    let t1 = t0 + SETTLE;
    let _ = e.step(
        Input::FsEvent {
            resource: child,
            event: FsEvent::ContentChanged,
        },
        t1,
    );
    let t2 = t1 + SETTLE * 2;
    drain_due(&mut e, t2);
    let corr = e.pending_probe_for(pid).expect("verify probe in flight");
    let _ = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid,
            correlation: corr,
            outcome: ProbeOutcome::Vanished,
        }),
        t2,
    );
    assert!(
        matches!(
            e.profiles().get(pid).unwrap().state(),
            ProfileState::Pending(_)
        ),
        "descendant-LCA Vanished re-enters descent rather than parking a dead watch",
    );

    // The descent probe shows the anchor alive: re-materialize -> triggered Seed -> the rm's
    // change fires once settled.
    let out = descent_advance(&mut e, pid, Some(EntryKind::Dir), t2);
    assert!(out.effects().is_empty());
    let t3 = t2 + SETTLE * 2;
    drain_due(&mut e, t3);
    let corr = e.pending_probe_for(pid).expect("Seed verify in flight");
    let out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid,
            correlation: corr,
            outcome: proven(dir_snap(&[])),
        }),
        t3,
    );
    assert_eq!(
        out.effects().len(),
        1,
        "the interrupted change still fires after the one-hop recovery",
    );
    assert!(e.subs().get(sid).unwrap().has_fired());
    let _ = e.cancel_all_in_flight_probes();
}

/// Save storm: N replace cycles inside one settle window debounce to exactly one fire — each
/// terminal lands during the previous cycle's Seed Batching, loops finalize → descend → Seed, and
/// only the last save's Seed survives its settle window.
#[test]
fn replace_storm_within_settle_window_fires_once() {
    let mut e = Engine::new();
    let (_parent, anchor) = place_file_anchor(&mut e);

    let now = Instant::now();
    let (sid, pid, _) = attach_at(
        &mut e,
        "storm",
        anchor,
        ClassSet::DEFAULT_SUBTREE_ROOT,
        MAX_SETTLE,
        now,
    );
    let t0 = seed_to_idle_with(
        &mut e,
        pid,
        || anchor_ok(file_leaf(EntryKind::File, 1)),
        now,
    );

    // Three replaces 10ms apart — all inside one settle window. The anchor's slot id is stable
    // across replaces ((parent, segment) identity).
    let mut t = t0 + SETTLE;
    let mut fired_during_storm = 0;
    for _ in 0..3 {
        t += Duration::from_millis(10);
        let _ = e.step(
            Input::FsEvent {
                resource: anchor,
                event: FsEvent::Removed,
            },
            t,
        );
        assert!(
            matches!(
                e.profiles().get(pid).unwrap().state(),
                ProfileState::Pending(_)
            ),
            "every terminal in the storm re-enters descent",
        );
        let out = descent_advance(&mut e, pid, Some(EntryKind::File), t);
        fired_during_storm += out.effects().len();
        assert!(
            e.pending_probe_for(pid).is_none(),
            "each cycle re-opens Batching",
        );
    }
    assert_eq!(fired_during_storm, 0, "nothing fires inside the storm");

    // The storm ends; the last save's Seed survives its settle window -> exactly one fire, with
    // the last save's content.
    let t_end = t + SETTLE * 2;
    drain_due(&mut e, t_end);
    let out = respond_anchor_file(&mut e, pid, 4, t_end);
    assert_eq!(out.effects().len(), 1, "the storm debounces to one fire");
    assert!(e.subs().get(sid).unwrap().has_fired());
    let _ = e.cancel_all_in_flight_probes();
}

/// A repeated terminal during the recovery Seed's *Verifying* phase: the loss wrapper cancels the
/// armed verify slot (tripwire-safe), re-enters descent with a fresh correlation, the cancelled
/// walk's late response drops stale without disturbing the descent, and the last save fires exactly
/// once.
#[test]
fn repeated_terminal_mid_verifying_cancels_and_recovers() {
    let mut e = Engine::new();
    let (_parent, anchor) = place_file_anchor(&mut e);

    let now = Instant::now();
    let (sid, pid, _) = attach_at(
        &mut e,
        "mid-verify",
        anchor,
        ClassSet::DEFAULT_SUBTREE_ROOT,
        MAX_SETTLE,
        now,
    );
    let t0 = seed_to_idle_with(
        &mut e,
        pid,
        || anchor_ok(file_leaf(EntryKind::File, 1)),
        now,
    );

    // First replace: descend, materialize (inode 2), drain the settle window -> Verifying with an
    // armed probe slot.
    let t1 = t0 + SETTLE;
    let _ = e.step(
        Input::FsEvent {
            resource: anchor,
            event: FsEvent::Removed,
        },
        t1,
    );
    let _ = descent_advance(&mut e, pid, Some(EntryKind::File), t1);
    let t2 = t1 + SETTLE * 2;
    drain_due(&mut e, t2);
    let stale_corr = e
        .pending_probe_for(pid)
        .expect("recovery Seed verify probe in flight");

    // Second terminal lands mid-Verifying: cancel the armed slot, re-enter descent with a fresh
    // correlation.
    let t3 = t2 + Duration::from_millis(1);
    let _ = e.step(
        Input::FsEvent {
            resource: anchor,
            event: FsEvent::Removed,
        },
        t3,
    );
    assert!(matches!(
        e.profiles().get(pid).unwrap().state(),
        ProfileState::Pending(_)
    ));
    let descent_corr = e
        .pending_probe_for(pid)
        .expect("descent probe re-armed by the second loss");
    assert_ne!(descent_corr, stale_corr, "fresh correlation minted");

    // The cancelled walk's late response drops stale; the descent is undisturbed.
    let out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid,
            correlation: stale_corr,
            outcome: anchor_ok(file_leaf(EntryKind::File, 2)),
        }),
        t3,
    );
    assert!(
        out.diagnostics
            .iter()
            .any(|d| matches!(d, Diagnostic::StaleProbeResponse { .. })),
        "late verify response drops stale",
    );
    assert_eq!(e.pending_probe_for(pid), Some(descent_corr));

    // Recovery completes against the second replacement -> exactly one fire.
    let out = descent_advance(&mut e, pid, Some(EntryKind::File), t3);
    assert!(out.effects().is_empty());
    let t4 = t3 + SETTLE * 2;
    drain_due(&mut e, t4);
    let out = respond_anchor_file(&mut e, pid, 3, t4);
    assert_eq!(out.effects().len(), 1, "the last save fires exactly once");
    assert!(e.subs().get(sid).unwrap().has_fired());
    let _ = e.cancel_all_in_flight_probes();
}

/// A non-atomic delete-then-write save: the witnessed descent finds the prefix but not the awaited
/// segment and parks — narrated via `PendingPathAwaitingSegment` (witnessed descents only;
/// attach-time descents park silently as their steady state) — then the create's parent
/// notification re-probes, the advance does not narrate, and the latched recovery still owes — and
/// fires — the save's fire.
#[test]
fn delete_then_write_parks_narrated_then_recovers() {
    let mut e = Engine::new();
    let (parent, anchor) = place_file_anchor(&mut e);

    let now = Instant::now();
    let (_sid, pid, _) = attach_at(
        &mut e,
        "del-then-write",
        anchor,
        ClassSet::DEFAULT_SUBTREE_ROOT,
        MAX_SETTLE,
        now,
    );
    let t0 = seed_to_idle_with(
        &mut e,
        pid,
        || anchor_ok(file_leaf(EntryKind::File, 1)),
        now,
    );

    // The delete: the loss step re-enters descent witnessed; the write hasn't landed yet, so the
    // parent listing lacks the segment — the descent parks, narrated.
    let t1 = t0 + SETTLE;
    let _ = e.step(
        Input::FsEvent {
            resource: anchor,
            event: FsEvent::Removed,
        },
        t1,
    );
    let parked = descent_advance(&mut e, pid, None, t1);
    assert!(
        parked.diagnostics.iter().any(|d| matches!(
            d,
            Diagnostic::PendingPathAwaitingSegment { profile, prefix, segment }
                if *profile == pid && *prefix == parent && segment == "app.log",
        )),
        "witnessed park narrates the awaited segment; got {:?}",
        parked.diagnostics,
    );
    assert!(matches!(
        e.profiles().get(pid).unwrap().state(),
        ProfileState::Pending(_)
    ));
    assert!(
        e.pending_probe_for(pid).is_none(),
        "parked descent awaits the next event",
    );

    // The write lands: the parent notification re-probes; the advancing response materialises the
    // triggered Seed (Batching-first) without narrating a park.
    let t2 = t1 + Duration::from_millis(5);
    let _ = e.step(
        Input::FsEvent {
            resource: parent,
            event: FsEvent::StructureChanged,
        },
        t2,
    );
    let advanced = descent_advance(&mut e, pid, Some(EntryKind::File), t2);
    assert!(
        !advanced
            .diagnostics
            .iter()
            .any(|d| matches!(d, Diagnostic::PendingPathAwaitingSegment { .. })),
        "an advancing response never narrates a park",
    );
    assert!(
        e.pending_probe_for(pid).is_none(),
        "triggered Seed opens Batching-first",
    );

    // The latch persisted through the park: the recovery owes — and fires — the save's fire.
    let t3 = t2 + SETTLE * 2;
    drain_due(&mut e, t3);
    let out = respond_anchor_file(&mut e, pid, 2, t3);
    assert_eq!(
        out.effects().len(),
        1,
        "the delete-then-write save fires once recovered",
    );
    let _ = e.cancel_all_in_flight_probes();
}
