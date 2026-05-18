//! Pending-path descent end-to-end. Drives `Engine::attach_sub` with a
//! path-based request, walks descent through scaffolds, and confirms
//! anchor materialization triggers a Seed burst.

#![allow(
    clippy::items_after_statements,
    clippy::manual_let_else,
    clippy::match_wildcard_for_single_variants,
    clippy::missing_const_for_fn,
    clippy::needless_pass_by_value,
    clippy::option_if_let_else,
    clippy::single_match_else,
    clippy::too_many_lines
)]

use compact_str::CompactString;
use specter_core::testkit::single_exec_program;
use specter_core::{
    ActionProgram, ActiveBurst, ChildEntry, ClassSet, Diagnostic, DirChild, DirMeta, DirSnapshot,
    EffectScope, EntryKind, FsEvent, FsIdentity, Input, LeafEntry, ProbeCorrelation, ProbeOp,
    ProbeOutcome, ProbeOwner, ProbeResponse, ProfileState, ProofAuthority, ResourceKind,
    ResourceRole, ScanConfig, StepOutput, SubAttachAnchor, SubAttachRequest,
};
use specter_engine::Engine;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant, UNIX_EPOCH};

const SETTLE: Duration = Duration::from_millis(100);
const MAX_SETTLE: Duration = Duration::from_secs(6);
const NO_EVENTS: ClassSet = ClassSet::EMPTY;

fn empty_program() -> Arc<ActionProgram> {
    single_exec_program([specter_core::ArgTemplate::new([
        specter_core::ArgPart::literal("/bin/true"),
    ])])
}

/// V5-native helper: build a `DirSnapshot` with single-component
/// children. The walker speaks paths; engine identity stays engine-side,
/// so the snapshot carries pure content. Tests in this file use
/// leaf-name segments only.
fn dir_snap_with(children: Vec<(&str, EntryKind, u64)>) -> std::sync::Arc<DirSnapshot> {
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

/// Pluck the correlation from the (single) Probe in `out`.
fn first_probe_corr(out: &StepOutput) -> Option<ProbeCorrelation> {
    out.probe_ops().iter().find_map(|op| match op {
        ProbeOp::Probe { request } => Some(request.correlation()),
        ProbeOp::Cancel { .. } => None,
    })
}

/// Drive a Batching-first Seed burst through its full N=2 quiescence
/// proof to `Idle`. A Seed runs the same two-settle-spaced
/// equal-sample proof as a Standard burst: no probe fires at burst
/// start; the first Seed probe materializes only after the initial
/// settle window (`t0 + SETTLE`) expires and `Batching → Verifying`.
///
/// 1. expire settle #1 (`t0 + SETTLE`) → first Seed probe; respond with
///    `seed_snap`. The carrier's prior `certified` is `None`,
///    so the verdict is `Unstable` by construction → graft + re-batch.
/// 2. expire settle #2 (`t0 + SETTLE*2`) → second Seed probe; respond
///    with the hash-equal `seed_snap` → `Stable` → seed pin + rebase →
///    `Idle`.
///
/// `t0` is the instant the Seed burst *starts*: for a `Resource` attach
/// against a live anchor that is the attach instant; for a path attach
/// that materializes through descent it is the instant of the step that
/// materialized the anchor (`Pending → Idle → Active(Seed Batching)`),
/// not the original attach. Both responses are hash-equal and within
/// `MAX_SETTLE`, so the burst reaches a clean `Stable` and is never
/// forced; a fresh Seed emits no Effects.
fn complete_seed_burst(
    e: &mut Engine,
    pid: specter_core::ProfileId,
    seed_snap: Arc<DirSnapshot>,
    t0: Instant,
) {
    for at in [t0 + SETTLE, t0 + SETTLE * 2] {
        while let Some(en) = e.pop_expired(at) {
            e.step(
                Input::TimerExpired {
                    profile: en.profile,
                    kind: en.kind,
                    id: en.id,
                },
                at,
            );
        }
        let c = e
            .pending_probe_for(ProbeOwner::Profile(pid))
            .expect("Seed Verifying probe in flight after settle expiry");
        let out = e.step(
            Input::ProbeResponse(ProbeResponse {
                owner: ProbeOwner::Profile(pid),
                correlation: c,
                outcome: ProbeOutcome::SubtreeProven {
                    snapshot: Arc::clone(&seed_snap),
                    authority: ProofAuthority::Authoritative,
                },
            }),
            at,
        );
        assert!(out.effects().is_empty(), "a fresh Seed never emits Effects");
    }
    assert!(
        matches!(e.profiles().get(pid).unwrap().state(), ProfileState::Idle,),
        "Seed burst completes its N=2 proof and returns to Idle",
    );
}

#[test]
fn attach_sub_path_pending_then_anchor_appears() {
    // Tree has /var only. attach_sub at path /var/log/myapp pending state:
    // prefix=/var, remaining=[log, myapp]. Inject probe responses showing
    // log appears, then myapp appears. Anchor materializes; Seed burst
    // starts.
    let mut e = Engine::new();
    let var = e
        .tree_mut()
        .ensure_path(&["/", "var"], ResourceRole::User)
        .expect("non-empty fixture");
    e.tree_mut().set_kind(var, ResourceKind::Dir);

    let req = SubAttachRequest::for_anchor(
        "watch".into(),
        SubAttachAnchor::Path(PathBuf::from("/var/log/myapp")),
        ScanConfig::builder().recursive(true).build(),
        MAX_SETTLE,
        SETTLE,
        empty_program(),
        EffectScope::SubtreeRoot,
        NO_EVENTS,
        false,
    );
    let now = Instant::now();
    let attach_out = e.step(Input::AttachSub(req), now);
    let sid = specter_core::testkit::first_attached_sub(&attach_out).expect("attach_sub succeeded");
    let pid = e.subs().get(sid).unwrap().profile;

    // Initial pending state: intermediate scaffold in place; anchor
    // already has role=User ("role = User for the
    // anchor, role = DescentScaffold for everything else"). Pending
    // status lives on `Profile.state == ProfileState::Pending(_)`, not
    // on the anchor's role.
    let log = e.tree().lookup(Some(var), "log").expect("log scaffold");
    let myapp = e.tree().lookup(Some(log), "myapp").expect("anchor slot");
    assert!(matches!(
        e.tree().get(log).unwrap().role,
        ResourceRole::DescentScaffold,
    ));
    assert!(
        matches!(e.tree().get(myapp).unwrap().role, ResourceRole::User),
        "anchor's role is User even when pending",
    );
    let var_corr = first_probe_corr(&attach_out).expect("descent probe at /var emitted");

    // Inject probe response showing `log` appears.
    let log_advance = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
            correlation: var_corr,
            outcome: ProbeOutcome::DirEnumerated(dir_snap_with(vec![("log", EntryKind::Dir, 1)])),
        }),
        now,
    );
    let log_corr = first_probe_corr(&log_advance).expect("descent probe at /var/log emitted");

    // Inject probe response showing `myapp` appears under /var/log.
    let myapp_materialize = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
            correlation: log_corr,
            outcome: ProbeOutcome::DirEnumerated(dir_snap_with(vec![("myapp", EntryKind::Dir, 2)])),
        }),
        now,
    );

    // Anchor materialized: kind set from the snapshot's entry; role
    // stays User (set at attach time).
    assert!(matches!(
        e.tree().get(myapp).unwrap().role,
        ResourceRole::User,
    ));
    assert_eq!(e.tree().get(myapp).unwrap().kind(), Some(ResourceKind::Dir));

    // Profile is now in Active(PreFire(Seed)) — the Seed burst was
    // started at materialization. A Seed is Batching-first:
    // the materializing step opens the burst in
    // `Batching { settle_timer }` and emits NO probe; the first Seed
    // probe materializes only after the initial settle window expires.
    match e.profiles().get(pid).unwrap().state() {
        ProfileState::Active(ActiveBurst::PreFire(pre), _) => {
            assert_eq!(pre.intent, specter_core::BurstIntent::Seed);
            assert!(
                matches!(pre.phase, specter_core::PreFirePhase::Batching { .. }),
                "Seed opens Batching-first; got {:?}",
                pre.phase,
            );
        }
        s => panic!("expected Active(PreFire(Seed)), got {s:?}"),
    }
    assert!(
        first_probe_corr(&myapp_materialize).is_none(),
        "Batching-first Seed emits no probe at materialization",
    );

    // Drive the N=2 Seed proof. `t0` is the instant the Seed burst
    // started — the step that materialized the anchor (`now`), not the
    // original attach.
    complete_seed_burst(&mut e, pid, dir_snap_with(vec![]), now);

    // Profile should now be Idle with baseline established.
    let p = e.profiles().get(pid).unwrap();
    assert!(matches!(p.state(), ProfileState::Idle));
    assert!(p.baseline().is_some());
}

#[test]
fn pending_path_failed_probe_retains_state() {
    let mut e = Engine::new();
    let var = e
        .tree_mut()
        .ensure_path(&["/", "var"], ResourceRole::User)
        .expect("non-empty fixture");
    e.tree_mut().set_kind(var, ResourceKind::Dir);

    let req = SubAttachRequest::for_anchor(
        "watch".into(),
        SubAttachAnchor::Path(PathBuf::from("/var/missing")),
        ScanConfig::builder().build(),
        MAX_SETTLE,
        SETTLE,
        empty_program(),
        EffectScope::SubtreeRoot,
        NO_EVENTS,
        false,
    );
    let attach_out = e.step(Input::AttachSub(req), Instant::now());
    let sid = specter_core::testkit::first_attached_sub(&attach_out).expect("attach_sub succeeded");
    let pid = e.subs().get(sid).unwrap().profile;
    let corr = first_probe_corr(&attach_out).expect("descent probe");

    let out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
            correlation: corr,
            outcome: ProbeOutcome::Failed { errno: 13 },
        }),
        Instant::now(),
    );

    assert!(
        out.diagnostics
            .iter()
            .any(|d| matches!(d, Diagnostic::PendingPathProbeFailed { errno: 13, .. }))
    );
    // Profile still pending (descent state lives on
    // `ProfileState::Pending`, not on a separate SecondaryMap).
    assert!(matches!(
        e.profiles().get(pid).unwrap().state(),
        ProfileState::Pending(_),
    ));
}

#[test]
fn pending_path_event_at_prefix_emits_fresh_probe() {
    // Pending descent waiting for /var/missing/. Drain in-flight probe
    // with a no-progress response, then inject FsEvent at /var (the
    // prefix) to trigger a fresh probe (no settle).
    let mut e = Engine::new();
    let var = e
        .tree_mut()
        .ensure_path(&["/", "var"], ResourceRole::User)
        .expect("non-empty fixture");
    e.tree_mut().set_kind(var, ResourceKind::Dir);

    let req = SubAttachRequest::for_anchor(
        "watch".into(),
        SubAttachAnchor::Path(PathBuf::from("/var/missing")),
        ScanConfig::builder().build(),
        MAX_SETTLE,
        SETTLE,
        empty_program(),
        EffectScope::SubtreeRoot,
        NO_EVENTS,
        false,
    );
    let attach_out = e.step(Input::AttachSub(req), Instant::now());
    let sid = specter_core::testkit::first_attached_sub(&attach_out).expect("attach_sub succeeded");
    let pid = e.subs().get(sid).unwrap().profile;
    let corr = first_probe_corr(&attach_out).expect("descent probe");

    // No-progress response — descent stays pending.
    let _ = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
            correlation: corr,
            outcome: ProbeOutcome::DirEnumerated(dir_snap_with(vec![(
                "other",
                EntryKind::File,
                99,
            )])),
        }),
        Instant::now(),
    );

    // FsEvent at /var triggers a fresh descent probe.
    let out = e.step(
        Input::FsEvent {
            resource: var,
            event: FsEvent::StructureChanged,
        },
        Instant::now(),
    );
    let probes = out
        .probe_ops()
        .iter()
        .filter(|op| matches!(op, ProbeOp::Probe { request } if request.owner() == ProbeOwner::Profile(pid)))
        .count();
    assert_eq!(probes, 1, "FsEvent at prefix triggers fresh descent probe");
    let _ = e.cancel_all_in_flight_probes();
}

#[test]
fn anchor_disappears_re_enters_pending_via_watch_root_parent() {
    // "Watch root deletion": Sub at /src; / is the
    // watch_root_parent. Anchor is removed; Profile → Idle with
    // current=None. Then a StructureChanged at / triggers the recovery
    // path which re-enters pending descent.
    let mut e = Engine::new();
    // Both / and /src exist; /src is the anchor.
    let root_dir = e.tree_mut().ensure_root("root", ResourceRole::User);
    e.tree_mut().set_kind(root_dir, ResourceKind::Dir);
    let src = e
        .tree_mut()
        .ensure_child(root_dir, "src", ResourceRole::User)
        .expect("test live parent");
    e.tree_mut().set_kind(src, ResourceKind::Dir);

    let req = SubAttachRequest::for_anchor(
        "watch".into(),
        SubAttachAnchor::Resource(src),
        ScanConfig::builder().recursive(true).build(),
        MAX_SETTLE,
        SETTLE,
        empty_program(),
        EffectScope::SubtreeRoot,
        NO_EVENTS,
        false,
    );
    let now = Instant::now();
    let attach_out = e.step(Input::AttachSub(req), now);
    let sid = specter_core::testkit::first_attached_sub(&attach_out).expect("attach_sub succeeded");
    let pid = e.subs().get(sid).unwrap().profile;
    // The immediate Seed is Batching-first: no probe at attach.
    assert!(
        first_probe_corr(&attach_out).is_none(),
        "Batching-first Seed emits no probe at attach",
    );
    // Drive the N=2 Seed proof → Idle (`t0` is the attach instant).
    complete_seed_burst(&mut e, pid, dir_snap_with(vec![]), now);
    assert!(matches!(
        e.profiles().get(pid).unwrap().state(),
        ProfileState::Idle,
    ));
    assert!(e.profiles().get(pid).unwrap().watch_root_parent() == Some(root_dir));

    // The Seed proof consumed two settle windows; keep instants
    // monotonic for the recovery sequence that follows.
    let after_seed = now + SETTLE * 3;

    // Anchor gone (Removed event at /src).
    e.step(
        Input::FsEvent {
            resource: src,
            event: FsEvent::Removed,
        },
        after_seed,
    );
    // Profile is Idle with current=None now.
    let p = e.profiles().get(pid).unwrap();
    assert!(matches!(p.state(), ProfileState::Idle));
    assert!(p.current().is_none());

    // StructureChanged at / triggers recovery: Profile re-enters pending
    // descent with prefix=/, remaining=[src].
    let out = e.step(
        Input::FsEvent {
            resource: root_dir,
            event: FsEvent::StructureChanged,
        },
        after_seed,
    );
    let recovery_probe = out
        .probe_ops()
        .iter()
        .any(|op| matches!(op, ProbeOp::Probe { request } if request.owner() == ProbeOwner::Profile(pid)));
    assert!(
        recovery_probe,
        "recovery emits descent probe at watch_root_parent",
    );
    let _ = e.cancel_all_in_flight_probes();
}

// Detach Pending Profile with in-flight descent probe
#[test]
fn detach_pending_profile_with_inflight_descent_emits_cancel() {
    let mut e = Engine::new();
    let var = e
        .tree_mut()
        .ensure_path(&["/", "var"], ResourceRole::User)
        .expect("non-empty fixture");
    e.tree_mut().set_kind(var, ResourceKind::Dir);

    let req = SubAttachRequest::for_anchor(
        "watch".into(),
        SubAttachAnchor::Path(PathBuf::from("/var/log/myapp")),
        ScanConfig::builder().recursive(true).build(),
        MAX_SETTLE,
        SETTLE,
        empty_program(),
        EffectScope::SubtreeRoot,
        NO_EVENTS,
        false,
    );
    let now = Instant::now();
    let attach_out = e.step(Input::AttachSub(req), now);
    let sid = specter_core::testkit::first_attached_sub(&attach_out).expect("attach_sub succeeded");
    let pid = e.subs().get(sid).unwrap().profile;

    // Profile is Pending with an in-flight descent probe.
    let initial_corr = first_probe_corr(&attach_out).expect("descent probe at attach");
    let is_pending = matches!(
        e.profiles().get(pid).expect("Profile attached").state(),
        ProfileState::Pending(_)
    );
    assert!(is_pending, "Profile is in Pending state");
    assert_eq!(
        e.pending_probe_for(ProbeOwner::Profile(pid)),
        Some(initial_corr),
        "descent state carries the outstanding probe correlation",
    );

    // Detach without delivering a probe response.
    let detach_out = e.step(Input::DetachSub(sid), Instant::now());

    // Profile is reaped.
    assert!(
        e.profiles().get(pid).is_none(),
        "Profile reaped on detach (Pending + last Sub detached)",
    );
    // ProbeOp::Cancel emitted for the in-flight descent probe.
    let cancel_present = detach_out
        .probe_ops()
        .iter()
        .any(|op| matches!(op, ProbeOp::Cancel { owner: ProbeOwner::Profile(profile)} if *profile == pid));
    assert!(
        cancel_present,
        "ProbeOp::Cancel emitted for in-flight descent probe; got {:?}",
        detach_out.probe_ops(),
    );
}

// Anchor terminal event on a Pending Profile pins the no-consumer
// routing. An absolute attach against an empty Tree puts the FS-root
// bootstrap between prefix and anchor: prefix is the synthetic `/`,
// anchor is the scaffolded `/foo`. The two are distinct slots, and the
// anchor's `watch_demand` is zero (descent hasn't materialized it yet),
// so a `Removed` at the anchor lands in `EventOnUnwatchedResource`
// rather than coercing the Pending Profile through
// `finalize_anchor_lost` / `finish_burst_to_idle`.
#[test]
fn pending_profile_event_at_anchor_lands_in_no_consumer_branch() {
    let mut e = Engine::new();
    // Absolute path against an empty Tree: bootstrap creates `/`, anchor
    // `/foo` is scaffolded under `/`. Profile lands Pending with
    // current_prefix = `/`, anchor = /foo (different slots).
    let req = SubAttachRequest::for_anchor(
        "watch".into(),
        SubAttachAnchor::Path(PathBuf::from("/foo")),
        ScanConfig::builder().recursive(true).build(),
        MAX_SETTLE,
        SETTLE,
        empty_program(),
        EffectScope::SubtreeRoot,
        NO_EVENTS,
        false,
    );
    let now = Instant::now();
    let attach_out = e.step(Input::AttachSub(req), now);
    let sid = specter_core::testkit::first_attached_sub(&attach_out).expect("attach_sub succeeded");
    let pid = e.subs().get(sid).unwrap().profile;

    let p = e.profiles().get(pid).expect("Profile attached");
    let anchor = p.resource;
    let prefix = match p.state() {
        ProfileState::Pending(d) => d.current_prefix(),
        s => panic!("expected Pending, got {s:?}"),
    };
    assert_ne!(
        prefix, anchor,
        "FS-root bootstrap separates prefix from anchor"
    );
    assert_eq!(
        e.tree().get(prefix).unwrap().watch_demand(),
        1,
        "descent prefix `/` carries the +1 STRUCTURE contribution",
    );
    assert_eq!(
        e.tree().get(anchor).unwrap().watch_demand(),
        0,
        "anchor scaffold is not yet bumped (descent hasn't materialized it)",
    );

    // Dispatch FsEvent::Removed at the anchor (/foo). The anchor's
    // `watch_demand == 0` short-circuits at the `EventOnUnwatchedResource`
    // head guard in `on_fs_event` before any classifier work runs.
    // Earlier this same Profile shape (a degenerate `prefix == anchor`
    // fixture from a relative-path attach against an empty Tree) routed
    // through `finish_burst_to_idle` and underflowed a Resource
    // refcount. The FS-root bootstrap rules out that degenerate shape
    // entirely.
    let out = e.step(
        Input::FsEvent {
            resource: anchor,
            event: FsEvent::Removed,
        },
        now,
    );

    // Profile remains Pending (no covering-profile fan-out touched it).
    let still_pending = matches!(
        e.profiles().get(pid).unwrap().state(),
        ProfileState::Pending(_),
    );
    assert!(
        still_pending,
        "Pending Profile not coerced through anchor-terminal-event path",
    );
    // The head guard short-circuits before any classifier work, so it
    // emits no watch op (no spurious Unwatch/Watch on this path).
    assert!(
        out.watch_ops.is_empty(),
        "EventOnUnwatchedResource head guard emits no watch op; got {:?}",
        out.watch_ops,
    );
    // The event landed in the no-consumer head guard.
    assert!(
        out.diagnostics.iter().any(|d| matches!(
            d,
            specter_core::Diagnostic::EventOnUnwatchedResource { resource, .. } if *resource == anchor,
        )),
        "anchor terminal event on Pending Profile lands in EventOnUnwatchedResource diagnostic; got {:?}",
        out.diagnostics,
    );
    let _ = e.cancel_all_in_flight_probes();
}

// Behavioral parity: a single FsEvent at one Resource fans out to a
// Pending Profile (descent dispatch) AND an Idle Profile with absent
// anchor (recovery dispatch), without disturbing an unrelated Profile.
#[test]
#[allow(clippy::similar_names)]
fn classifier_routes_descent_and_recovery_in_single_pass() {
    // /root and /root/bar exist; /root/foo does not. /elsewhere exists.
    let mut e = Engine::new();
    let root_dir = e
        .tree_mut()
        .ensure_path(&["/", "root"], ResourceRole::User)
        .expect("non-empty fixture");
    e.tree_mut().set_kind(root_dir, ResourceKind::Dir);
    let bar = e
        .tree_mut()
        .ensure_child(root_dir, "bar", ResourceRole::User)
        .expect("test live parent");
    e.tree_mut().set_kind(bar, ResourceKind::Dir);
    let elsewhere = e
        .tree_mut()
        .ensure_path(&["/", "elsewhere"], ResourceRole::User)
        .expect("non-empty fixture");
    e.tree_mut().set_kind(elsewhere, ResourceKind::Dir);

    // Profile A: Pending at /root, descending toward `foo` (does not
    // exist). Drain its initial descent probe with a no-progress
    // response so its `pending_probe` slot is empty before the test
    // event — `on_descent_event` short-circuits on a busy slot.
    let req_a = SubAttachRequest::for_anchor(
        "watch-a".into(),
        SubAttachAnchor::Path(PathBuf::from("/root/foo")),
        ScanConfig::builder().recursive(true).build(),
        MAX_SETTLE,
        SETTLE,
        empty_program(),
        EffectScope::SubtreeRoot,
        NO_EVENTS,
        false,
    );
    let now = Instant::now();
    let attach_a_out = e.step(Input::AttachSub(req_a), now);
    let sid_a =
        specter_core::testkit::first_attached_sub(&attach_a_out).expect("attach_sub succeeded");
    let pid_a = e.subs().get(sid_a).unwrap().profile;
    let a_corr = first_probe_corr(&attach_a_out).expect("descent probe at attach");
    e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid_a),
            correlation: a_corr,
            outcome: ProbeOutcome::DirEnumerated(dir_snap_with(vec![("bar", EntryKind::Dir, 1)])),
        }),
        now,
    );
    assert!(
        matches!(
            e.profiles().get(pid_a).unwrap().state(),
            ProfileState::Pending(_),
        ),
        "A still Pending after no-progress response",
    );

    // Profile B: anchor at /root/bar; drive Seed → Idle, then Removed
    // at /root/bar → Idle with current=None and watch_root_parent=/root.
    let req_b = SubAttachRequest::for_anchor(
        "watch-b".into(),
        SubAttachAnchor::Resource(bar),
        ScanConfig::builder().recursive(true).build(),
        MAX_SETTLE,
        SETTLE,
        empty_program(),
        EffectScope::SubtreeRoot,
        NO_EVENTS,
        false,
    );
    let attach_b_out = e.step(Input::AttachSub(req_b), now);
    let sid_b =
        specter_core::testkit::first_attached_sub(&attach_b_out).expect("attach_sub succeeded");
    let pid_b = e.subs().get(sid_b).unwrap().profile;
    assert!(
        first_probe_corr(&attach_b_out).is_none(),
        "Batching-first Seed emits no probe at attach",
    );
    // Drive B's N=2 Seed proof → Idle (`t0` is B's attach instant). A
    // is Pending with an empty descent slot (no settle timer), so its
    // timers do not interfere with B's settle drain.
    complete_seed_burst(&mut e, pid_b, dir_snap_with(vec![]), now);
    assert_eq!(
        e.profiles().get(pid_b).unwrap().watch_root_parent(),
        Some(root_dir),
        "B watches its parent /root for anchor recovery",
    );
    // B's Seed consumed two settle windows; keep instants monotonic.
    let after_b_seed = now + SETTLE * 3;
    e.step(
        Input::FsEvent {
            resource: bar,
            event: FsEvent::Removed,
        },
        after_b_seed,
    );
    let p_b = e.profiles().get(pid_b).unwrap();
    assert!(matches!(p_b.state(), ProfileState::Idle));
    assert!(p_b.current().is_none(), "B's anchor is gone");
    assert_eq!(p_b.watch_root_parent(), Some(root_dir));

    // Profile C: anchor at /elsewhere; Seed → Idle. Unrelated to /root.
    let req_c = SubAttachRequest::for_anchor(
        "watch-c".into(),
        SubAttachAnchor::Resource(elsewhere),
        ScanConfig::builder().recursive(true).build(),
        MAX_SETTLE,
        SETTLE,
        empty_program(),
        EffectScope::SubtreeRoot,
        NO_EVENTS,
        false,
    );
    // C attaches after B's Seed; keep instants strictly monotonic.
    let c_attach = after_b_seed + SETTLE;
    let attach_c_out = e.step(Input::AttachSub(req_c), c_attach);
    let sid_c =
        specter_core::testkit::first_attached_sub(&attach_c_out).expect("attach_sub succeeded");
    let pid_c = e.subs().get(sid_c).unwrap().profile;
    assert!(
        first_probe_corr(&attach_c_out).is_none(),
        "Batching-first Seed emits no probe at attach",
    );
    // Drive C's N=2 Seed proof → Idle (`t0` is C's attach instant).
    complete_seed_burst(&mut e, pid_c, dir_snap_with(vec![]), c_attach);
    assert!(matches!(
        e.profiles().get(pid_c).unwrap().state(),
        ProfileState::Idle,
    ));

    // The trigger: a single StructureChanged event at /root.
    // - A's `current_prefix == /root` ⇒ descent dispatch.
    // - B's `watch_root_parent == /root && current.is_none()` ⇒ recovery
    //   dispatch (Idle → Pending).
    // - C is anchored at /elsewhere ⇒ untouched.
    // Strictly after both Seed proofs (B and C each consumed two settle
    // windows since `now`).
    let trigger = c_attach + SETTLE * 3;
    let out = e.step(
        Input::FsEvent {
            resource: root_dir,
            event: FsEvent::StructureChanged,
        },
        trigger,
    );

    // A: a fresh descent probe minted (slot was empty after drain).
    let a_probes = out
        .probe_ops()
        .iter()
        .filter(|op| matches!(op, ProbeOp::Probe { request } if request.owner() == ProbeOwner::Profile(pid_a)))
        .count();
    assert_eq!(a_probes, 1, "A's descent advance emits one probe");
    assert!(
        matches!(
            e.profiles().get(pid_a).unwrap().state(),
            ProfileState::Pending(_),
        ),
        "A remains Pending",
    );

    // B: re-entered Pending (recovery descent) and emitted a probe.
    let b_probes = out
        .probe_ops()
        .iter()
        .filter(|op| matches!(op, ProbeOp::Probe { request } if request.owner() == ProbeOwner::Profile(pid_b)))
        .count();
    assert_eq!(b_probes, 1, "B's recovery emits one descent probe");
    assert!(
        matches!(
            e.profiles().get(pid_b).unwrap().state(),
            ProfileState::Pending(_),
        ),
        "B transitioned Idle → Pending",
    );

    // C: untouched. No probe; state still Idle.
    let c_probes = out
        .probe_ops()
        .iter()
        .filter(|op| matches!(op, ProbeOp::Probe { request } if request.owner() == ProbeOwner::Profile(pid_c)))
        .count();
    assert_eq!(c_probes, 0, "C is unrelated to /root; no probe");
    assert!(matches!(
        e.profiles().get(pid_c).unwrap().state(),
        ProfileState::Idle,
    ));
    let _ = e.cancel_all_in_flight_probes();
}
