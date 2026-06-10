//! Cross-cutting discovery lifecycle — the `MatchChain` Profile role end-to-end.
//!
//! Drives one `Engine` per scenario through attach → cold-Seed reconcile (mint per terminus ×
//! template) → Standard re-reconcile → vanish/recovery/overflow/cascade with synthetic
//! [`ProbeResponse`] injections. The inline tests (`src/discovery_tests.rs`) pin the pure
//! collector and the attach-boundary asserts; this file pins the composed behaviour: consequence
//! routing, dedup convergence, lifecycle diagnostics, and the Draining-gate shape filter.
//!
//! Assertions are **converged-state** (the minted registry after quiescence), never step traces —
//! probe cadence is an implementation detail; only the resulting registry is the contract.

use specter_core::testkit::{
    anchor_ok, covered, dir_snap, dir_snap_nested, file_leaf, leaf, proven, uncovered,
};
use specter_core::{
    ActiveBurst, ClassSet, DetachReason, Diagnostic, DirSnapshot, EffectScope, EntryKind, FsEvent,
    Input, OverflowScope, ProbeOp, ProbeResponse, ProfileId, ProfileState, ResourceId,
    ResourceKind, ScanConfig, StepOutput, SubAttachAnchor, SubId,
};
use specter_engine::Engine;
use specter_engine::testkit::{
    DEFAULT_EVENTS, MAX_SETTLE, SETTLE, attach, attach_discovery, attach_discovery_returning,
    descent_advance, discovery_subs_of, drain_due, is_draining, mint_template, pid_of,
    pre_place_dir, seed_to_idle,
};
use std::collections::BTreeSet;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Answer `pid`'s single in-flight probe with `proven(snap)` at `at`.
fn respond(e: &mut Engine, pid: ProfileId, snap: &Arc<DirSnapshot>, at: Instant) -> StepOutput {
    let corr = e.pending_probe_for(pid).expect("probe in flight");
    e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid,
            correlation: corr,
            outcome: proven(Arc::clone(snap)),
        }),
        at,
    )
}

/// Open a Standard burst with a `StructureChanged` at `resource`, drain the settle window, answer
/// the verify probe with `snap` — one event, one probe, one reconcile (`EventsReliable`, N = 1).
fn burst_and_respond(
    e: &mut Engine,
    pid: ProfileId,
    resource: ResourceId,
    snap: &Arc<DirSnapshot>,
    now: Instant,
) -> (StepOutput, Instant) {
    let _ = e.step(
        Input::FsEvent {
            resource,
            event: FsEvent::StructureChanged,
        },
        now,
    );
    let at = now + SETTLE * 2;
    drain_due(e, at);
    (respond(e, pid, snap, at), at)
}

/// Drive every freshly-minted Profile's cold Seed to Idle so later timeline drains never collide
/// with minted-burst timers: File anchors answer `AnchorOk`, Dir anchors an empty subtree.
fn settle_minted_seeds(e: &mut Engine, source: SubId, at: Instant) {
    let minted: Vec<SubId> = discovery_subs_of(e, source).into_values().collect();
    for mid in minted {
        let pid = pid_of(e, mid);
        let Some(corr) = e.pending_probe_for(pid) else {
            continue;
        };
        let outcome = match e.profiles().get(pid).expect("minted Profile live").kind() {
            Some(ResourceKind::File) => anchor_ok(file_leaf(EntryKind::File, 1)),
            _ => proven(dir_snap(&[])),
        };
        let _ = e.step(
            Input::ProbeResponse(ProbeResponse {
                owner: pid,
                correlation: corr,
                outcome,
            }),
            at,
        );
    }
}

/// The `DiscoveryMinted` narration in emission order, projected to `(source, path)`.
fn minted_paths(out: &StepOutput) -> Vec<(SubId, String)> {
    out.diagnostics
        .iter()
        .filter_map(|d| match d {
            Diagnostic::DiscoveryMinted { source, path, .. } => {
                Some((*source, path.display().to_string()))
            }
            _ => None,
        })
        .collect()
}

fn assert_idle_or_pre_fire(e: &Engine, pid: ProfileId, ctx: &str) {
    match e
        .profiles()
        .get(pid)
        .expect("discovery Profile live")
        .state()
    {
        ProfileState::Idle | ProfileState::Active(ActiveBurst::PreFire(_), _) => {}
        other => panic!("{ctx}: discovery Profile must stay Idle | PreFire, got {other:?}"),
    }
}

/// Cold-Seed first reconcile mints one Sub per chain terminus (both terminus kinds), with
/// deterministic synthesised names, source attribution, and the minted Profiles' own cold-Seed
/// probes in the same `StepOutput`; a Standard re-reconcile over the same tree is a dedup no-op.
#[test]
fn cold_seed_reconcile_mints_per_terminus_then_re_reconcile_dedups() {
    let mut e = Engine::new();
    let srv = pre_place_dir(&mut e, &["srv"]);
    let now = Instant::now();
    let (sid, pid) = attach_discovery(
        &mut e,
        "disc",
        SubAttachAnchor::Resource(srv),
        "/srv/*/log",
        mint_template(),
        now,
    );

    // Pruned chain shape: `a/log` a Dir terminus (Uncovered), `b/log` a File terminus (Leaf).
    let chain = dir_snap_nested(&[
        ("a", covered(dir_snap_nested(&[("log", uncovered(10))]))),
        (
            "b",
            covered(dir_snap_nested(&[("log", leaf(EntryKind::File, 11))])),
        ),
    ]);
    let out = respond(&mut e, pid, &chain, now);

    assert!(
        matches!(e.profiles().get(pid).unwrap().state(), ProfileState::Idle),
        "reconcile seals the discovery burst to Idle",
    );
    assert!(out.effects().is_empty(), "discovery never fires Effects");

    let minted = discovery_subs_of(&e, sid);
    assert_eq!(minted.len(), 2, "one mint per terminus");
    let names: BTreeSet<String> = minted
        .values()
        .map(|&mid| e.subs().get(mid).unwrap().name.to_string())
        .collect();
    assert_eq!(
        names,
        BTreeSet::from(["disc@/srv/a/log".to_string(), "disc@/srv/b/log".to_string()]),
        "synthesised names are {{template}}@{{abs path}}",
    );
    assert_eq!(
        minted_paths(&out),
        vec![
            (sid, "/srv/a/log".to_string()),
            (sid, "/srv/b/log".to_string()),
        ],
        "DiscoveryMinted narrates each mint, lexicographic terminus order",
    );
    let kinds: Vec<ResourceKind> = out
        .diagnostics
        .iter()
        .filter_map(|d| match d {
            Diagnostic::DiscoveryMinted { kind, .. } => Some(*kind),
            _ => None,
        })
        .collect();
    assert_eq!(
        kinds,
        vec![ResourceKind::Dir, ResourceKind::File],
        "minted kinds fold the snapshot's terminus kinds",
    );

    // Each minted Profile entered its own cold Seed in the same step — probe ops present.
    for &mid in minted.values() {
        let mp = pid_of(&e, mid);
        assert!(
            out.probe_ops().iter().any(|op| matches!(
                op,
                ProbeOp::Probe { request } if request.owner() == mp
            )),
            "minted Profile {mp:?} cold-Seed probe rides the reconcile step",
        );
        let s = e.subs().get(mid).unwrap();
        assert_eq!(s.source_discovery, Some(sid), "mint carries its source");
        assert!(s.template.is_none(), "minted Subs are never templates");
    }
    let attached = out
        .diagnostics
        .iter()
        .filter(|d| matches!(d, Diagnostic::SubAttached { source_discovery, .. } if *source_discovery == Some(sid)))
        .count();
    assert_eq!(attached, 2, "one SubAttached per mint, keyed to its source");

    settle_minted_seeds(&mut e, sid, now);

    // Standard re-reconcile over the same tree: dedup makes it a no-op.
    let (out2, _) = burst_and_respond(&mut e, pid, srv, &chain, now + Duration::from_millis(10));
    assert!(minted_paths(&out2).is_empty(), "re-reconcile mints nothing");
    assert_eq!(
        discovery_subs_of(&e, sid),
        minted,
        "minted set unchanged — same SubIds, no churn",
    );
    assert!(
        matches!(e.profiles().get(pid).unwrap().state(), ProfileState::Idle),
        "re-reconcile seals back to Idle",
    );
}

/// A storm of chain events inside one settle window coalesces into one Batching window, one
/// probe, one reconcile that mints everything — never one probe per event.
#[test]
fn storm_coalesces_to_one_probe_and_one_reconcile() {
    let mut e = Engine::new();
    let data = pre_place_dir(&mut e, &["data"]);
    let now = Instant::now();
    let (sid, pid) = attach_discovery(
        &mut e,
        "disc",
        SubAttachAnchor::Resource(data),
        "/data/*",
        mint_template(),
        now,
    );
    let _ = respond(&mut e, pid, &dir_snap(&[]), now);

    // Three events, 10ms apart — all inside the 100ms settle window.
    let mut at = now + Duration::from_millis(10);
    for _ in 0..3 {
        let _ = e.step(
            Input::FsEvent {
                resource: data,
                event: FsEvent::StructureChanged,
            },
            at,
        );
        assert!(
            e.pending_probe_for(pid).is_none(),
            "still Batching mid-storm — no probe until the window settles",
        );
        at += Duration::from_millis(10);
    }
    let drain_at = at + SETTLE * 2;
    drain_due(&mut e, drain_at);
    let untar = dir_snap(&[
        ("x", EntryKind::Dir, 1),
        ("y", EntryKind::Dir, 2),
        ("z.log", EntryKind::File, 3),
    ]);
    let out = respond(&mut e, pid, &untar, drain_at);
    assert_eq!(
        minted_paths(&out).len(),
        3,
        "one settle-debounced reconcile mints the whole batch",
    );
    assert_eq!(discovery_subs_of(&e, sid).len(), 3);
    let _ = e.cancel_all_in_flight_probes();
}

/// Terminus-internal churn drives a benign, terminus-scoped no-op reconcile — and across the whole
/// scenario the discovery Profile never leaves `Idle | Active(PreFire)`: no Draining, no PostFire,
/// structurally.
#[test]
fn terminus_churn_is_a_noop_reconcile_and_never_leaves_pre_fire() {
    let mut e = Engine::new();
    let srv = pre_place_dir(&mut e, &["srv"]);
    let now = Instant::now();
    let (sid, pid) = attach_discovery(
        &mut e,
        "disc",
        SubAttachAnchor::Resource(srv),
        "/srv/*/log",
        mint_template(),
        now,
    );
    let chain = dir_snap_nested(&[("a", covered(dir_snap_nested(&[("log", uncovered(10))])))]);
    let _ = respond(&mut e, pid, &chain, now);
    settle_minted_seeds(&mut e, sid, now);
    let a = e.tree().lookup(Some(srv), "a").expect("chain dir slot");
    let log = e.tree().lookup(Some(a), "log").expect("terminus slot");

    // Churn inside the terminus surfaces as STRUCTURE at the terminus slot (the minted Profile's
    // FD); the discovery Profile covers it (depth == td) and bursts alongside the minted one.
    let t1 = now + Duration::from_millis(10);
    let _ = e.step(
        Input::FsEvent {
            resource: log,
            event: FsEvent::StructureChanged,
        },
        t1,
    );
    assert_idle_or_pre_fire(&e, pid, "after terminus event");
    let t2 = t1 + SETTLE * 2;
    drain_due(&mut e, t2);
    assert_idle_or_pre_fire(&e, pid, "after settle drain");
    // The dirty LCA is the terminus itself, so the probe is terminus-scoped; a chain walk targeted
    // there prunes everything beyond the bound — an empty Dir is the honest payload.
    let out = respond(&mut e, pid, &dir_snap(&[]), t2);
    assert!(
        minted_paths(&out).is_empty(),
        "no re-mint on internal churn"
    );
    assert_idle_or_pre_fire(&e, pid, "after no-op reconcile");
    assert_eq!(discovery_subs_of(&e, sid).len(), 1, "minted set stable");
    let _ = e.cancel_all_in_flight_probes();
}

/// A vanished terminus reaps its minted Sub through the minted Profile's own anchor-terminal path
/// (`DiscoverySubReaped` + `SubDetached(AnchorLost)`); the next reconcile mints nothing; the path's
/// reappearance re-mints under a fresh `SubId`.
#[test]
fn terminus_vanish_reaps_minted_and_reappearance_remints_fresh() {
    let mut e = Engine::new();
    let data = pre_place_dir(&mut e, &["data"]);
    let now = Instant::now();
    let (sid, pid) = attach_discovery(
        &mut e,
        "disc",
        SubAttachAnchor::Resource(data),
        "/data/*",
        mint_template(),
        now,
    );
    let _ = respond(
        &mut e,
        pid,
        &dir_snap(&[("x.log", EntryKind::File, 5)]),
        now,
    );
    settle_minted_seeds(&mut e, sid, now);
    let minted = discovery_subs_of(&e, sid);
    let (&terminus, &old_mid) = minted.iter().next().expect("one mint");

    // Kernel-faithful vanish pair: `Removed` on the terminus FD (anchor-terminal for the minted
    // Profile; class CONTENT for a File, so the discovery mask drops it as a descendant event) +
    // `StructureChanged` on the parent (drives the discovery reconcile).
    let t1 = now + Duration::from_millis(10);
    let rm = e.step(
        Input::FsEvent {
            resource: terminus,
            event: FsEvent::Removed,
        },
        t1,
    );
    assert!(
        rm.diagnostics.iter().any(|d| matches!(
            d,
            Diagnostic::DiscoverySubReaped { source, sub, .. }
                if *source == sid && *sub == old_mid
        )),
        "source-keyed reap narration; got {:?}",
        rm.diagnostics,
    );
    assert!(
        rm.diagnostics.iter().any(|d| matches!(
            d,
            Diagnostic::SubDetached { sub, reason: DetachReason::AnchorLost, .. }
                if *sub == old_mid
        )),
        "the minted Sub's lifecycle signal carries AnchorLost",
    );
    assert!(e.subs().get(old_mid).is_none(), "minted Sub removed");

    let (gone, t2) = burst_and_respond(&mut e, pid, data, &dir_snap(&[]), t1);
    assert!(
        minted_paths(&gone).is_empty(),
        "vanished match mints nothing"
    );
    assert!(discovery_subs_of(&e, sid).is_empty());

    let (back, _) = burst_and_respond(
        &mut e,
        pid,
        data,
        &dir_snap(&[("x.log", EntryKind::File, 6)]),
        t2 + Duration::from_millis(10),
    );
    assert_eq!(minted_paths(&back).len(), 1, "reappearance re-mints");
    let new_mid = *discovery_subs_of(&e, sid)
        .values()
        .next()
        .expect("re-minted");
    assert_ne!(new_mid, old_mid, "a re-mint is a fresh Sub, not a revival");
    assert_eq!(
        e.subs().get(new_mid).unwrap().name,
        "disc@/data/x.log",
        "same path ⇒ same synthesised name (the old entry freed it)",
    );
    let _ = e.cancel_all_in_flight_probes();
}

/// Detaching the discovery template cascades: the template detaches under the caller's reason,
/// every minted Sub under `DiscoverySourceDetached`, and every Profile reaps.
#[test]
fn detach_discovery_template_cascades_to_minted_subs() {
    let mut e = Engine::new();
    let data = pre_place_dir(&mut e, &["data"]);
    let now = Instant::now();
    let (sid, pid) = attach_discovery(
        &mut e,
        "disc",
        SubAttachAnchor::Resource(data),
        "/data/*",
        mint_template(),
        now,
    );
    let _ = respond(
        &mut e,
        pid,
        &dir_snap(&[("x", EntryKind::Dir, 1), ("y", EntryKind::Dir, 2)]),
        now,
    );
    settle_minted_seeds(&mut e, sid, now);
    let minted = discovery_subs_of(&e, sid);
    let minted_pids: Vec<ProfileId> = minted.values().map(|&m| pid_of(&e, m)).collect();

    let out = e.step(Input::DetachSub(sid), now + Duration::from_millis(10));
    assert!(
        out.diagnostics.iter().any(|d| matches!(
            d,
            Diagnostic::SubDetached { sub, reason: DetachReason::IpcDisabled, .. } if *sub == sid
        )),
        "template detaches under the caller's reason",
    );
    for &mid in minted.values() {
        assert!(
            out.diagnostics.iter().any(|d| matches!(
                d,
                Diagnostic::SubDetached {
                    sub,
                    reason: DetachReason::DiscoverySourceDetached,
                    ..
                } if *sub == mid
            )),
            "minted Sub {mid:?} cascades under DiscoverySourceDetached",
        );
        assert!(e.subs().get(mid).is_none());
    }
    assert!(e.profiles().get(pid).is_none(), "discovery Profile reaped");
    for mp in minted_pids {
        assert!(e.profiles().get(mp).is_none(), "minted Profile reaped");
    }
    assert_eq!(e.subs().iter().count(), 0, "registry fully unwound");
}

/// Two templates on the same pattern share one discovery Profile and one walk; the reconcile mints
/// per (terminus × template), and identical minted identities share minted Profiles.
#[test]
fn second_template_shares_profile_and_walk_mints_per_template() {
    let mut e = Engine::new();
    let data = pre_place_dir(&mut e, &["data"]);
    let now = Instant::now();
    let (sid1, pid) = attach_discovery(
        &mut e,
        "disc1",
        SubAttachAnchor::Resource(data),
        "/data/*",
        mint_template(),
        now,
    );
    let (sid2, pid2) = attach_discovery(
        &mut e,
        "disc2",
        SubAttachAnchor::Resource(data),
        "/data/*",
        mint_template(),
        now,
    );
    assert_eq!(pid, pid2, "same pattern ⇒ one discovery Profile");

    let out = respond(&mut e, pid, &dir_snap(&[("x", EntryKind::Dir, 1)]), now);
    assert_eq!(
        minted_paths(&out),
        vec![(sid1, "/data/x".to_string()), (sid2, "/data/x".to_string())],
        "one terminus × two templates, template order sorted by SubId",
    );
    let m1 = discovery_subs_of(&e, sid1);
    let m2 = discovery_subs_of(&e, sid2);
    assert_eq!((m1.len(), m2.len()), (1, 1));
    let p1 = pid_of(&e, *m1.values().next().unwrap());
    let p2 = pid_of(&e, *m2.values().next().unwrap());
    assert_eq!(
        p1, p2,
        "identical minted identity ⇒ the two minted Subs share one minted Profile",
    );
    let _ = e.cancel_all_in_flight_probes();
}

/// Prefix `rm -rf` recovery: minted Subs reap bottom-up via their own anchors, the discovery
/// Profile recovers through `watch_root_parent` descent, and the recovery Seed's consequence is a
/// re-minting Reconcile — with **no** `PerFileDriftDroppedOnRecovery` even though the template
/// stores a per-file scope (the template's reaction is minting, not a per-file Effect).
#[test]
fn prefix_rm_rf_recovery_remints_without_per_file_warning() {
    let mut e = Engine::new();
    let data = pre_place_dir(&mut e, &["data"]);
    let root = e.tree().parent(data).expect("FS root");
    let now = Instant::now();
    let (sid, pid, _) = attach_discovery_returning(
        &mut e,
        "disc",
        SubAttachAnchor::Resource(data),
        "/data/*",
        mint_template(),
        EffectScope::PerStableFile,
        now,
    );
    let _ = respond(&mut e, pid, &dir_snap(&[("x", EntryKind::Dir, 1)]), now);
    settle_minted_seeds(&mut e, sid, now);
    let old_mid = *discovery_subs_of(&e, sid).values().next().expect("minted");
    let x = e.tree().lookup(Some(data), "x").expect("terminus slot");

    // rm -rf bottom-up: terminus first (minted all-dynamic teardown), then the anchor
    // (anchor-terminal for the discovery Profile — template is operator-declared, so the
    // mixed/static recovery path preserves watch_root_parent).
    let t1 = now + Duration::from_millis(10);
    let mut outs: Vec<StepOutput> = Vec::new();
    outs.push(e.step(
        Input::FsEvent {
            resource: x,
            event: FsEvent::Removed,
        },
        t1,
    ));
    outs.push(e.step(
        Input::FsEvent {
            resource: data,
            event: FsEvent::Removed,
        },
        t1,
    ));
    assert!(e.subs().get(old_mid).is_none(), "minted Sub reaped");
    assert!(e.subs().get(sid).is_some(), "template survives anchor loss");

    // Reappearance: the parent's StructureChanged re-enters descent; one hop materialises the
    // anchor and opens the recovery Seed.
    let t2 = t1 + Duration::from_millis(10);
    outs.push(e.step(
        Input::FsEvent {
            resource: root,
            event: FsEvent::StructureChanged,
        },
        t2,
    ));
    outs.push(descent_advance(
        &mut e,
        pid,
        &dir_snap(&[("data", EntryKind::Dir, 7)]),
        t2,
    ));
    let recovered = respond(&mut e, pid, &dir_snap(&[("x", EntryKind::Dir, 9)]), t2);
    assert_eq!(
        minted_paths(&recovered).len(),
        1,
        "recovery Seed reconciles and re-mints (never a SilentPin)",
    );
    let new_mid = *discovery_subs_of(&e, sid).values().next().expect("re-mint");
    assert_ne!(new_mid, old_mid, "fresh SubId after the loss window");
    outs.push(recovered);
    for out in &outs {
        assert!(
            !out.diagnostics
                .iter()
                .any(|d| matches!(d, Diagnostic::PerFileDriftDroppedOnRecovery { .. })),
            "a per-file template scope never trips the recovery-drop warn; got {:?}",
            out.diagnostics,
        );
    }
    let _ = e.cancel_all_in_flight_probes();
}

/// Overflow force-reseeds the discovery Profile (Idle and mid-burst arms); the reseed's reconcile
/// recovers mints missed inside the unreliable window.
#[test]
fn overflow_reseed_recovers_missed_mints() {
    let mut e = Engine::new();
    let data = pre_place_dir(&mut e, &["data"]);
    let now = Instant::now();
    let (sid, pid) = attach_discovery(
        &mut e,
        "disc",
        SubAttachAnchor::Resource(data),
        "/data/*",
        mint_template(),
        now,
    );
    let _ = respond(&mut e, pid, &dir_snap(&[("x", EntryKind::Dir, 1)]), now);
    settle_minted_seeds(&mut e, sid, now);

    // Idle arm: overflow reseeds; the fresh cold walk surfaces a match the lost events hid.
    let t1 = now + Duration::from_millis(10);
    let _ = e.step(
        Input::SensorOverflow {
            scope: OverflowScope::Global,
        },
        t1,
    );
    assert!(
        e.pending_probe_for(pid).is_some(),
        "Idle overflow arm reseeds with a fresh cold probe",
    );
    let out = respond(
        &mut e,
        pid,
        &dir_snap(&[("x", EntryKind::Dir, 1), ("y", EntryKind::Dir, 2)]),
        t1,
    );
    assert_eq!(
        minted_paths(&out),
        vec![(sid, "/data/y".to_string())],
        "the missed match mints on the reseed reconcile",
    );

    // Mid-burst arm: overflow lands while Batching — disarm, finish, reseed, same recovery.
    let t2 = t1 + Duration::from_millis(10);
    let _ = e.step(
        Input::FsEvent {
            resource: data,
            event: FsEvent::StructureChanged,
        },
        t2,
    );
    assert!(e.pending_probe_for(pid).is_none());
    let _ = e.step(
        Input::SensorOverflow {
            scope: OverflowScope::Global,
        },
        t2,
    );
    assert!(
        e.pending_probe_for(pid).is_some(),
        "mid-burst overflow arm reseeds immediately",
    );
    let out2 = respond(
        &mut e,
        pid,
        &dir_snap(&[
            ("x", EntryKind::Dir, 1),
            ("y", EntryKind::Dir, 2),
            ("z", EntryKind::Dir, 3),
        ]),
        t2,
    );
    assert_eq!(minted_paths(&out2), vec![(sid, "/data/z".to_string())]);
    assert_eq!(discovery_subs_of(&e, sid).len(), 3);
    let _ = e.cancel_all_in_flight_probes();
}

/// An armed operator `absorb` window is inert for discovery: the reconcile proceeds (mints land)
/// and no `QuiescenceAbsorbed` is narrated — the early classify return precedes the fold override,
/// so the latch is never consumed.
#[test]
fn absorb_window_is_inert_for_discovery() {
    let mut e = Engine::new();
    let data = pre_place_dir(&mut e, &["data"]);
    let now = Instant::now();
    let (sid, pid) = attach_discovery(
        &mut e,
        "disc",
        SubAttachAnchor::Resource(data),
        "/data/*",
        mint_template(),
        now,
    );
    let _ = respond(&mut e, pid, &dir_snap(&[]), now);

    let t1 = now + Duration::from_millis(10);
    let armed = e.step(
        Input::ArmAbsorb {
            profile: pid,
            duration: Some(Duration::from_mins(1)),
        },
        t1,
    );
    assert!(
        armed
            .diagnostics
            .iter()
            .any(|d| matches!(d, Diagnostic::AbsorbArmed { profile, .. } if *profile == pid)),
        "window armed",
    );
    let (out, _) = burst_and_respond(
        &mut e,
        pid,
        data,
        &dir_snap(&[("x", EntryKind::Dir, 1)]),
        t1,
    );
    assert_eq!(minted_paths(&out).len(), 1, "minting proceeds under absorb");
    assert_eq!(discovery_subs_of(&e, sid).len(), 1, "the mint registered");
    assert!(
        !out.diagnostics
            .iter()
            .any(|d| matches!(d, Diagnostic::QuiescenceAbsorbed { .. })),
        "Reconcile is non-firing — nothing for the window to fold",
    );
    let _ = e.cancel_all_in_flight_probes();
}

/// `max_settle` ceiling expiry under churn forces the probe and the forced verdict still reconciles
/// and seals — promotion latency stays bounded.
#[test]
fn forced_ceiling_reconciles_and_seals() {
    let mut e = Engine::new();
    let data = pre_place_dir(&mut e, &["data"]);
    let now = Instant::now();
    let (sid, pid) = attach_discovery(
        &mut e,
        "disc",
        SubAttachAnchor::Resource(data),
        "/data/*",
        mint_template(),
        now,
    );
    let _ = respond(&mut e, pid, &dir_snap(&[]), now);

    let t1 = now + Duration::from_millis(10);
    let _ = e.step(
        Input::FsEvent {
            resource: data,
            event: FsEvent::StructureChanged,
        },
        t1,
    );
    // Drain at the burst-deadline horizon: the settle expiry fires first (Verifying), then the
    // ceiling forces a fresh probe through `force_pending`.
    let horizon = t1 + MAX_SETTLE;
    drain_due(&mut e, horizon);
    let out = respond(&mut e, pid, &dir_snap(&[("x", EntryKind::Dir, 1)]), horizon);
    assert_eq!(
        minted_paths(&out).len(),
        1,
        "forced verdict reconciles from the forced graft",
    );
    assert!(
        matches!(e.profiles().get(pid).unwrap().state(), ProfileState::Idle),
        "forced reconcile seals to Idle",
    );
    assert_eq!(discovery_subs_of(&e, sid).len(), 1);
    let _ = e.cancel_all_in_flight_probes();
}

/// Draining-gate filter, exclusion direction: a mid-burst **discovery** descendant does not hold a
/// covering user Profile in Draining — the gate protects against descendant *command* activity and
/// discovery fires none.
#[test]
fn mid_burst_discovery_descendant_does_not_drain_covering_profile() {
    let mut e = Engine::new();
    let srv = pre_place_dir(&mut e, &["srv"]);
    let apps = pre_place_dir(&mut e, &["srv", "apps"]);
    let now = Instant::now();
    let (_, outer_pid) = attach(
        &mut e,
        "outer",
        SubAttachAnchor::Resource(srv),
        ScanConfig::builder().recursive(true).build(),
        DEFAULT_EVENTS,
        MAX_SETTLE,
        now,
    );
    let outer_base = dir_snap(&[("apps", EntryKind::Dir, 1)]);
    let seed_done = seed_to_idle(&mut e, outer_pid, &outer_base, now);
    let (tpl_sid, disc_pid) = attach_discovery(
        &mut e,
        "disc",
        SubAttachAnchor::Resource(apps),
        "/srv/apps/*/log",
        mint_template(),
        seed_done,
    );
    let chain = dir_snap_nested(&[("a", covered(dir_snap_nested(&[("log", uncovered(10))])))]);
    let _ = respond(&mut e, disc_pid, &chain, seed_done);
    settle_minted_seeds(&mut e, tpl_sid, seed_done);

    // One event at the discovery anchor opens both bursts (outer covers it recursively); the minted
    // Profile (deeper) stays Idle. Answer the outer verify first, while discovery is still
    // mid-burst Verifying.
    let t1 = seed_done + Duration::from_millis(10);
    let _ = e.step(
        Input::FsEvent {
            resource: apps,
            event: FsEvent::StructureChanged,
        },
        t1,
    );
    let t2 = t1 + SETTLE * 2;
    drain_due(&mut e, t2);
    assert!(
        e.pending_probe_for(disc_pid).is_some(),
        "discovery mid-burst at the outer verdict",
    );
    let out = respond(&mut e, outer_pid, &outer_base, t2);
    assert!(
        !is_draining(&e, outer_pid),
        "a chain-shaped descendant burst never holds the gate",
    );
    assert!(
        !out.effects().is_empty(),
        "outer fires through — the filter, not the gate, decided",
    );
    let _ = e.cancel_all_in_flight_probes();
}

/// Draining-gate filter, transitivity direction: a mid-burst **minted** Standard descendant still
/// drains the covering user Profile, its chain resolving *through* the discovery Profile —
/// `chain_reaches` stays shape-agnostic.
#[test]
fn mid_burst_minted_descendant_still_drains_through_discovery() {
    let mut e = Engine::new();
    let srv = pre_place_dir(&mut e, &["srv"]);
    let apps = pre_place_dir(&mut e, &["srv", "apps"]);
    let now = Instant::now();
    let (_, outer_pid) = attach(
        &mut e,
        "outer",
        SubAttachAnchor::Resource(srv),
        ScanConfig::builder().recursive(true).build(),
        DEFAULT_EVENTS,
        MAX_SETTLE,
        now,
    );
    let outer_base = dir_snap(&[("apps", EntryKind::Dir, 1)]);
    let seed_done = seed_to_idle(&mut e, outer_pid, &outer_base, now);
    let (tpl_sid, disc_pid) = attach_discovery(
        &mut e,
        "disc",
        SubAttachAnchor::Resource(apps),
        "/srv/apps/*/log",
        mint_template(),
        seed_done,
    );
    let chain = dir_snap_nested(&[("a", covered(dir_snap_nested(&[("log", uncovered(10))])))]);
    let _ = respond(&mut e, disc_pid, &chain, seed_done);
    settle_minted_seeds(&mut e, tpl_sid, seed_done);
    let a = e.tree().lookup(Some(apps), "a").expect("chain dir");
    let log = e.tree().lookup(Some(a), "log").expect("minted anchor");

    // One event at the minted anchor opens all three bursts (anchor bypass for the minted one).
    let t1 = seed_done + Duration::from_millis(10);
    let _ = e.step(
        Input::FsEvent {
            resource: log,
            event: FsEvent::StructureChanged,
        },
        t1,
    );
    let t2 = t1 + SETTLE * 2;
    drain_due(&mut e, t2);
    let minted_pid = pid_of(&e, *discovery_subs_of(&e, tpl_sid).values().next().unwrap());
    assert!(
        e.pending_probe_for(minted_pid).is_some(),
        "minted Profile mid-burst at the outer verdict",
    );
    let out = respond(&mut e, outer_pid, &outer_base, t2);
    assert!(
        is_draining(&e, outer_pid),
        "a minted Standard descendant gates the outer Profile through the discovery hop",
    );
    assert!(out.effects().is_empty(), "deferred, not fired");
    let _ = e.cancel_all_in_flight_probes();
}

/// A static Sub joining the minted identity makes the minted Profile mixed: anchor loss routes to
/// the static recovery path (minted Sub survives), and the next reconcile's dedup finds the
/// still-attached Sub — no double-mint after recovery.
#[test]
fn mixed_profile_at_minted_anchor_survives_anchor_loss_without_remint() {
    let mut e = Engine::new();
    let data = pre_place_dir(&mut e, &["data"]);
    let now = Instant::now();
    let (sid, pid) = attach_discovery(
        &mut e,
        "disc",
        SubAttachAnchor::Resource(data),
        "/data/*",
        mint_template(),
        now,
    );
    let _ = respond(&mut e, pid, &dir_snap(&[("x", EntryKind::Dir, 1)]), now);
    settle_minted_seeds(&mut e, sid, now);
    let mid = *discovery_subs_of(&e, sid).values().next().expect("minted");
    let x = e.tree().lookup(Some(data), "x").expect("terminus slot");

    // Join the minted Profile with a static Sub of the identical identity (the mint_template
    // fixture's: recursive Subtree, EMPTY mask, MAX_SETTLE).
    let (_, co_pid) = attach(
        &mut e,
        "co-resident",
        SubAttachAnchor::Resource(x),
        ScanConfig::builder().recursive(true).build(),
        ClassSet::EMPTY,
        MAX_SETTLE,
        now,
    );
    assert_eq!(
        co_pid,
        pid_of(&e, mid),
        "static Sub joins the minted Profile"
    );

    let t1 = now + Duration::from_millis(10);
    let rm = e.step(
        Input::FsEvent {
            resource: x,
            event: FsEvent::Removed,
        },
        t1,
    );
    assert!(
        e.subs().get(mid).is_some(),
        "mixed Profile routes to finalize_anchor_lost — the minted Sub survives",
    );
    assert!(
        !rm.diagnostics
            .iter()
            .any(|d| matches!(d, Diagnostic::DiscoverySubReaped { .. })),
        "no wholesale teardown narration on the mixed path",
    );

    // The terminus reappears: dedup finds the still-attached minted Sub — no second mint.
    let (out, _) = burst_and_respond(
        &mut e,
        pid,
        data,
        &dir_snap(&[("x", EntryKind::Dir, 2)]),
        t1 + Duration::from_millis(10),
    );
    assert!(minted_paths(&out).is_empty(), "no re-mint past a live Sub");
    assert_eq!(
        *discovery_subs_of(&e, sid).values().next().unwrap(),
        mid,
        "the surviving Sub is the same Sub",
    );
    let _ = e.cancel_all_in_flight_probes();
}

/// One reconcile minting past the fan-out threshold warns exactly once; the latch silences every
/// later reconcile for the template's lifetime.
#[test]
fn fanout_threshold_warns_exactly_once() {
    let mut e = Engine::new();
    let big = pre_place_dir(&mut e, &["big"]);
    let now = Instant::now();
    let (sid, pid) = attach_discovery(
        &mut e,
        "disc",
        SubAttachAnchor::Resource(big),
        "/big/*",
        mint_template(),
        now,
    );
    let names: Vec<String> = (0..=1000).map(|i| format!("d{i:04}")).collect();
    let entries: Vec<(&str, EntryKind, u64)> = names
        .iter()
        .enumerate()
        .map(|(i, n)| (n.as_str(), EntryKind::Dir, i as u64 + 1))
        .collect();
    let huge = dir_snap(&entries);

    let out = respond(&mut e, pid, &huge, now);
    let warns = |o: &StepOutput| {
        o.diagnostics
            .iter()
            .filter(|d| {
                matches!(
                    d,
                    Diagnostic::DiscoveryFanoutThreshold { source, count }
                        if *source == sid && *count == 1001
                )
            })
            .count()
    };
    assert_eq!(minted_paths(&out).len(), 1001, "everything mints");
    assert_eq!(warns(&out), 1, "threshold crossing warns exactly once");

    let (out2, _) = burst_and_respond(&mut e, pid, big, &huge, now + Duration::from_millis(10));
    assert!(minted_paths(&out2).is_empty());
    assert_eq!(warns(&out2), 0, "the latch silences later reconciles");
    let _ = e.cancel_all_in_flight_probes();
}

/// Two identically-driven engines produce identical mint narration and identical minted-Sub / anchor
/// id sequences — reconcile order is deterministic (lexicographic termini × SubId-sorted templates).
#[test]
fn identically_driven_engines_mint_identically() {
    let drive = || {
        let mut e = Engine::new();
        let data = pre_place_dir(&mut e, &["data"]);
        let now = Instant::now();
        let (sid1, pid) = attach_discovery(
            &mut e,
            "alpha",
            SubAttachAnchor::Resource(data),
            "/data/*",
            mint_template(),
            now,
        );
        let (sid2, _) = attach_discovery(
            &mut e,
            "beta",
            SubAttachAnchor::Resource(data),
            "/data/*",
            mint_template(),
            now,
        );
        let out = respond(
            &mut e,
            pid,
            &dir_snap(&[("x", EntryKind::Dir, 1), ("y", EntryKind::Dir, 2)]),
            now,
        );
        let mints = minted_paths(&out);
        let m1 = discovery_subs_of(&e, sid1);
        let m2 = discovery_subs_of(&e, sid2);
        let _ = e.cancel_all_in_flight_probes();
        (mints, m1, m2)
    };
    let (mints_a, a1, a2) = drive();
    let (mints_b, b1, b2) = drive();
    assert_eq!(mints_a, mints_b, "identical mint narration order");
    assert_eq!((a1, a2), (b1, b2), "identical minted ids and anchors");
}

/// A dynamic pattern whose literal prefix doesn't exist yet attaches `Pending`, walks the ordinary
/// descent, and the materialisation Seed's consequence is the first reconcile.
#[test]
fn pending_prefix_descends_then_first_reconcile_mints() {
    let mut e = Engine::new();
    let now = Instant::now();
    let (sid, pid, attach_out) = attach_discovery_returning(
        &mut e,
        "disc",
        SubAttachAnchor::Path("/data/x".into()),
        "/data/x/*",
        mint_template(),
        EffectScope::SubtreeRoot,
        now,
    );
    assert!(
        matches!(
            e.profiles().get(pid).unwrap().state(),
            ProfileState::Pending(_)
        ),
        "absent literal prefix ⇒ Pending descent",
    );
    assert!(
        attach_out.probe_ops().iter().any(|op| matches!(
            op,
            ProbeOp::Probe { request } if request.owner() == pid
        )),
        "descent probe emitted at attach",
    );
    let _ = descent_advance(&mut e, pid, &dir_snap(&[("data", EntryKind::Dir, 1)]), now);
    let _ = descent_advance(&mut e, pid, &dir_snap(&[("x", EntryKind::Dir, 2)]), now);
    let out = respond(&mut e, pid, &dir_snap(&[("m", EntryKind::Dir, 3)]), now);
    assert_eq!(
        minted_paths(&out),
        vec![(sid, "/data/x/m".to_string())],
        "materialisation Seed reconciles and mints",
    );
    assert_eq!(
        e.subs()
            .get(*discovery_subs_of(&e, sid).values().next().unwrap())
            .unwrap()
            .name,
        "disc@/data/x/m",
    );
    let _ = e.cancel_all_in_flight_probes();
}
