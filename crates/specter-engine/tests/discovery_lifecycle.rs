//! Cross-cutting discovery lifecycle — the `MatchChain` Profile role end-to-end.
//!
//! Drives one `Engine` per scenario through attach → cold-Seed reconcile (mint per terminus ×
//! template) → Standard re-reconcile → vanish/recovery/overflow/cascade with synthetic
//! [`ProbeResponse`] injections. The inline tests (`src/discovery.rs`) pin the pure collector and
//! the attach-boundary asserts; this file pins the composed behaviour: consequence routing, dedup
//! convergence, lifecycle diagnostics, and the Draining-gate shape filter.
//!
//! Assertions are **converged-state** (the minted registry after quiescence), never step traces —
//! probe cadence is an implementation detail; only the resulting registry is the contract.

use specter_core::testkit::{
    anchor_ok, covered, dir_snap, dir_snap_nested, empty_program, file_leaf, leaf, proven,
    uncovered,
};
use specter_core::{
    ActiveBurst, ClassSet, ContribKey, DetachReason, Diagnostic, DirSnapshot, EffectCompletion,
    EffectOutcome, EffectScope, EntryKind, FsEvent, Input, MintTemplate, OverflowScope, ProbeOp,
    ProbeOutcome, ProbeResponse, ProfileId, ProfileIdentity, ProfileState, ResourceId,
    ResourceKind, ScanConfig, SpawnSpec, StepOutput, SubAttachAnchor, SubId, TimerKind, WatchOp,
};
use specter_engine::Engine;
use specter_engine::testkit::{
    DEFAULT_EVENTS, MAX_SETTLE, SETTLE, attach, attach_discovery, attach_discovery_returning,
    complete_effect_to_rebasing, descent_advance, discovery_subs_of, drain_due, fire_standard_once,
    is_draining, last_probe_path, mint_template, mint_template_scoped, pid_of, pre_place_dir,
    respond_anchor_file, seed_to_idle, seed_to_idle_with,
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

/// Drive one minted Profile to Idle from whatever entry state it holds. A parked recovery descent
/// advances first (answer the in-flight descent probe with `listing`, the parent enumeration —
/// required for that entry alone); the convergence loop then answers whatever probe the burst
/// surfaces (cold walk, verify, re-sample, rebase) with a byte-identical kind-appropriate sample
/// (File anchors `AnchorOk`, Dir anchors an empty subtree), completes any fired effect, and advances
/// the clock a settle window when the burst waits on a timer. The EMPTY-mask `mint_template` folds
/// `HashChannel` verdicts, so each certification needs two equal samples; a cold Seed pins on its
/// single sample without a clock advance. Returns the instant the Profile rests at Idle.
fn settle_minted(
    e: &mut Engine,
    mid: SubId,
    listing: Option<&Arc<DirSnapshot>>,
    mut at: Instant,
) -> Instant {
    let pid = pid_of(e, mid);
    if matches!(
        e.profiles().get(pid).expect("minted Profile live").state(),
        ProfileState::Pending(_)
    ) {
        let listing = listing.expect("a parked descent advances on the parent listing");
        let _ = descent_advance(e, pid, listing, at);
    }
    for _ in 0..10 {
        if matches!(e.profiles().get(pid).unwrap().state(), ProfileState::Idle) {
            return at;
        }
        if let Some(correlation) = e.pending_probe_for(pid) {
            let outcome = match e.profiles().get(pid).expect("minted Profile live").kind() {
                Some(ResourceKind::File) => anchor_ok(file_leaf(EntryKind::File, 1)),
                _ => proven(dir_snap(&[])),
            };
            let out = e.step(
                Input::ProbeResponse(ProbeResponse {
                    owner: pid,
                    correlation,
                    outcome,
                }),
                at,
            );
            for eff in out.effects().iter() {
                let _ = e.step(
                    Input::EffectComplete(EffectCompletion {
                        sub: mid,
                        key: eff.key(),
                        outcome: EffectOutcome::Ok,
                    }),
                    at,
                );
            }
        } else {
            at += SETTLE * 2;
            drain_due(e, at);
        }
    }
    panic!("minted Profile did not converge to Idle");
}

/// Drive every freshly-minted Profile's cold Seed to Idle so later timeline drains never collide
/// with minted-burst timers. Probe-less minted Profiles are skipped (already settled), so the pass
/// never advances the clock — a cold Seed pins on its single sample.
fn settle_minted_seeds(e: &mut Engine, source: SubId, at: Instant) {
    let minted: Vec<SubId> = discovery_subs_of(e, source).into_values().collect();
    for mid in minted {
        if e.pending_probe_for(pid_of(e, mid)).is_some() {
            let _ = settle_minted(e, mid, None, at);
        }
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

/// The `DiscoveryUnsupportedAnchorKind` narration in emission order, projected to `(source, path,
/// kind)`.
fn unsupported_warns(out: &StepOutput) -> Vec<(SubId, String, EntryKind)> {
    out.diagnostics
        .iter()
        .filter_map(|d| match d {
            Diagnostic::DiscoveryUnsupportedAnchorKind { source, path, kind } => {
                Some((*source, path.display().to_string(), *kind))
            }
            _ => None,
        })
        .collect()
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
        assert_eq!(s.minted_by(), Some(sid), "mint carries its source");
        assert!(!s.is_template(), "minted Subs are never templates");
    }
    let attached = out
        .diagnostics
        .iter()
        .filter(
            |d| matches!(d, Diagnostic::SubAttached { minted_by, .. } if *minted_by == Some(sid)),
        )
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

/// A storm of chain events inside one settle window coalesces into one Batching window, one probe,
/// one reconcile that mints everything — never one probe per event.
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

/// Terminus-internal churn cannot move the discovery proof object — the terminus is the chain's
/// boundary, and only its membership in the parent's enumeration folds — so the event drops at
/// routing for the discovery Profile (`EventOutsideProofObject`) and never drives it: no burst, no
/// terminus-scoped probe, no splice breach. The event reaches the slot at all only through the
/// minted Profile's co-located anchor FD, and that Profile correctly drives its own burst from it.
#[test]
fn terminus_churn_drops_for_discovery_and_drives_only_the_minted_profile() {
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
    let mid = *discovery_subs_of(&e, sid)
        .values()
        .next()
        .expect("one mint");
    let minted_pid = pid_of(&e, mid);

    // Churn inside the terminus surfaces as STRUCTURE at the terminus slot — the minted Profile's
    // anchor FD (the discovery Profile holds no contribution at its boundary).
    let t1 = now + Duration::from_millis(10);
    let out = e.step(
        Input::FsEvent {
            resource: log,
            event: FsEvent::StructureChanged,
        },
        t1,
    );
    assert!(
        out.diagnostics.iter().any(|d| matches!(
            d,
            Diagnostic::EventOutsideProofObject { resource, profile, .. }
                if *resource == log && *profile == pid,
        )),
        "the discovery Profile's boundary view drops the event; got {:?}",
        out.diagnostics,
    );
    assert!(
        matches!(e.profiles().get(pid).unwrap().state(), ProfileState::Idle),
        "terminus churn never drives the discovery Profile",
    );
    assert!(
        matches!(
            e.profiles().get(minted_pid).unwrap().state(),
            ProfileState::Active(_, _),
        ),
        "the minted Profile drives its own burst from the same event",
    );
    assert!(
        !out.diagnostics
            .iter()
            .any(|d| matches!(d, Diagnostic::SpliceCrossedUncovered { .. })),
        "no splice breach on a healthy system",
    );

    // The minted burst settles on its own; the registry converges with nothing re-minted.
    let _ = settle_minted(&mut e, mid, None, t1);
    assert!(
        matches!(e.profiles().get(pid).unwrap().state(), ProfileState::Idle),
        "discovery Profile still Idle after the minted burst settles",
    );
    assert_eq!(discovery_subs_of(&e, sid).len(), 1, "minted set stable");
}

/// A terminus **delete** is an identity event at the chain's boundary slot — exempt from the
/// proof-relevance drop (it folds to STRUCTURE at a Dir slot and *is* a membership change), so it
/// must drive the discovery reconcile. The pre-fire target clamps to the mid-chain parent (the
/// deepest descend-chain node — probing the vanished terminus itself could not graft), the graft at
/// that parent is clean, and the reconcile reaps the minted Sub whose terminus left the certified
/// set.
#[test]
fn terminus_delete_drives_clamped_reconcile_and_reaps_minted() {
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
    let mid = *discovery_subs_of(&e, sid)
        .values()
        .next()
        .expect("one mint");
    let minted_pid = pid_of(&e, mid);

    // `rmdir` of the Dir terminus: `Removed` on the minted Profile's anchor FD. Anchor-terminal for
    // the minted Profile (loss-step descent); identity-at-boundary for the discovery Profile
    // (drives the reconcile — removal authority stays with the reconcile, not the terminal).
    let t1 = now + Duration::from_millis(10);
    let _ = e.step(
        Input::FsEvent {
            resource: log,
            event: FsEvent::Removed,
        },
        t1,
    );
    assert!(
        matches!(
            e.profiles().get(minted_pid).unwrap().state(),
            ProfileState::Pending(_)
        ),
        "the minted Profile re-enters descent in the loss step",
    );
    assert!(
        matches!(
            e.profiles().get(pid).unwrap().state(),
            ProfileState::Active(_, _)
        ),
        "the identity exemption lets the terminus delete drive the discovery Profile",
    );

    // Settle expiry mints the verify probe; its target is the mid-chain parent, not the vanished
    // terminus — the clamp stops the dirty-LCA resolution above the boundary.
    let t2 = t1 + SETTLE * 2;
    let mut probe_path = None;
    while let Some(en) = e.pop_expired(t2) {
        let out = e.step(
            Input::TimerExpired {
                profile: en.profile,
                kind: en.kind,
                id: en.id,
            },
            t2,
        );
        if let Some(p) = last_probe_path(&out) {
            probe_path = Some(p);
        }
    }
    let a_path = e.tree().path_of(a).expect("chain dir live");
    assert_eq!(
        probe_path.as_deref(),
        Some(&*a_path),
        "verify probe targets the deepest descend-chain ancestor",
    );

    // The honest payload at the parent: `log` is gone from its enumeration. Clean graft —
    // membership, not content, is the proof — and the reconcile reaps the minted Sub.
    let out = respond(&mut e, pid, &dir_snap(&[]), t2);
    assert!(
        !out.diagnostics
            .iter()
            .any(|d| matches!(d, Diagnostic::SpliceCrossedUncovered { .. })),
        "the clamped target grafts cleanly; got {:?}",
        out.diagnostics,
    );
    assert!(
        out.diagnostics.iter().any(|d| matches!(
            d,
            Diagnostic::DiscoverySubReaped { source, sub, .. }
                if *source == sid && *sub == mid
        )),
        "reap narration rides the reconcile",
    );
    assert!(
        out.diagnostics.iter().any(|d| matches!(
            d,
            Diagnostic::SubDetached { sub, reason: DetachReason::MatchVanished, .. }
                if *sub == mid
        )),
        "the minted Sub's lifecycle signal carries MatchVanished",
    );
    assert!(e.subs().get(mid).is_none(), "minted Sub removed");
    assert!(
        e.profiles().get(minted_pid).is_none(),
        "the Pending minted Profile reaped from the removal pass",
    );
    assert!(discovery_subs_of(&e, sid).is_empty());
    assert!(
        matches!(e.profiles().get(pid).unwrap().state(), ProfileState::Idle),
        "discovery Profile seals back to Idle",
    );
    let _ = e.cancel_all_in_flight_probes();
}

/// A `Symlink` / `Other` chain terminus skips the mint wholesale — no Tree slot, no minted Sub, no
/// mint→reap churn — narrated once per template lifetime while a real-file sibling in the same pass
/// mints normally; replacing the symlink with a regular file at the same path then mints (the latch
/// gates only the diagnostic; kind is read fresh off the snapshot each pass).
#[test]
fn unsupported_terminus_skips_mint_and_warns_once() {
    let mut e = Engine::new();
    let srv = pre_place_dir(&mut e, &["srv"]);
    let now = Instant::now();
    let (sid, pid) = attach_discovery(
        &mut e,
        "disc",
        SubAttachAnchor::Resource(srv),
        "/srv/*/current",
        mint_template(),
        now,
    );

    // `a/current` carries the parameterised kind, `b/current` is a fifo-shaped `Other`, `c/current`
    // a regular file — the exact shape a symlink-farm pattern matches.
    let chain = |a_kind: EntryKind| {
        dir_snap_nested(&[
            (
                "a",
                covered(dir_snap_nested(&[("current", leaf(a_kind, 10))])),
            ),
            (
                "b",
                covered(dir_snap_nested(&[("current", leaf(EntryKind::Other, 11))])),
            ),
            (
                "c",
                covered(dir_snap_nested(&[("current", leaf(EntryKind::File, 12))])),
            ),
        ])
    };
    let out = respond(&mut e, pid, &chain(EntryKind::Symlink), now);

    assert_eq!(
        minted_paths(&out),
        vec![(sid, "/srv/c/current".to_string())],
        "the skip is per-terminus — the real-file sibling mints in the same pass",
    );
    assert_eq!(
        unsupported_warns(&out),
        vec![(sid, "/srv/a/current".to_string(), EntryKind::Symlink)],
        "one-shot narration carrying the snapshot's EntryKind: the first unsupported terminus \
         (lexicographic) warns; the fifo sibling skips silently under the same latch",
    );
    let a = e
        .tree()
        .lookup(Some(srv), "a")
        .expect("chain dir slot (reconciler-watched, independent of minting)");
    // The post-graft reconciler's diff bookkeeping holds a slot for every created entry — the
    // skip's claim lives at the registry: nothing anchors there.
    let skipped = e
        .tree()
        .lookup(Some(a), "current")
        .expect("reconciler bookkeeping slot for the skipped entry");
    assert!(
        e.profiles().iter().all(|(_, p)| p.resource() != skipped),
        "no Profile anchored at a skipped terminus",
    );
    assert_eq!(
        e.profiles().iter().count(),
        2,
        "discovery Profile + the one real-file mint — nothing for the skipped termini",
    );
    settle_minted_seeds(&mut e, sid, now);

    // Same tree again: the latch silences the narration and nothing churns — the converged minted
    // set is exactly the real-file mint.
    let (out2, t2) = burst_and_respond(
        &mut e,
        pid,
        srv,
        &chain(EntryKind::Symlink),
        now + Duration::from_millis(10),
    );
    assert!(
        unsupported_warns(&out2).is_empty(),
        "latched: the second pass narrates nothing",
    );
    assert!(minted_paths(&out2).is_empty(), "no re-mint, no churn");
    assert_eq!(discovery_subs_of(&e, sid).len(), 1, "minted set stable");

    // The symlink is replaced by a regular file at the same path: mints normally.
    let (out3, _) = burst_and_respond(
        &mut e,
        pid,
        srv,
        &chain(EntryKind::File),
        t2 + Duration::from_millis(10),
    );
    assert_eq!(
        minted_paths(&out3),
        vec![(sid, "/srv/a/current".to_string())],
        "kind is read fresh — the latch gated only the diagnostic, never the mint",
    );
    assert_eq!(discovery_subs_of(&e, sid).len(), 2);
    let _ = e.cancel_all_in_flight_probes();
}

/// A vanished terminus reaps its minted Sub through the reconcile's removal pass
/// (`DiscoverySubReaped` + `SubDetached(MatchVanished)`), not through its own anchor terminal — the
/// loss step parks the minted Profile in a recovery descent (`Pending`), and the certified empty
/// listing then reaps it from there in one step: the graft deletes the leaf entry and its try-reap
/// is refused on the minted back-ref, the removal pass detaches the Sub, and the Profile's reap
/// cascades the slot. The path's reappearance re-mints under a fresh `SubId`.
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
    let minted_pid = pid_of(&e, old_mid);

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
        matches!(
            e.profiles().get(minted_pid).unwrap().state(),
            ProfileState::Pending(_)
        ),
        "the loss step re-enters descent — removal authority is the reconcile, not the terminal",
    );
    assert!(
        e.subs().get(old_mid).is_some(),
        "the minted Sub survives its anchor terminal",
    );
    assert!(
        !rm.diagnostics
            .iter()
            .any(|d| matches!(d, Diagnostic::DiscoverySubReaped { .. })),
        "no reap narration in the loss step; got {:?}",
        rm.diagnostics,
    );

    let (gone, t2) = burst_and_respond(&mut e, pid, data, &dir_snap(&[]), t1);
    assert!(
        gone.diagnostics.iter().any(|d| matches!(
            d,
            Diagnostic::DiscoverySubReaped { source, sub, .. }
                if *source == sid && *sub == old_mid
        )),
        "source-keyed reap narration rides the reconcile; got {:?}",
        gone.diagnostics,
    );
    assert!(
        gone.diagnostics.iter().any(|d| matches!(
            d,
            Diagnostic::SubDetached { sub, reason: DetachReason::MatchVanished, .. }
                if *sub == old_mid
        )),
        "the minted Sub's lifecycle signal carries MatchVanished",
    );
    assert!(
        minted_paths(&gone).is_empty(),
        "vanished match mints nothing"
    );
    assert!(e.subs().get(old_mid).is_none(), "minted Sub removed");
    assert!(
        e.profiles().get(minted_pid).is_none(),
        "the Pending minted Profile reaped from the removal pass",
    );
    assert!(
        e.tree().lookup(Some(data), "x.log").is_none(),
        "the graft's refused try-reap resolves in the same step once the removal pass \
         drops the minted back-ref — no orphan slot",
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

/// An ancestor directory rename produces **no terminal at the minted anchor** (the terminus inode
/// is untouched; watch FDs are inode-bound), so only the reconcile can observe the move.
/// Still-matching direction (`a/log → b/log`): one pass reaps the old minted Sub from Idle
/// (`MatchVanished`, narrated with the *old* path — path identity changed, so fresh history is
/// correct) and mints the new path. Non-matching direction (the renamed dir leaves the pattern):
/// reap only. Without the removal pass this shape leaked the old Sub as a permanent anchorless
/// zombie alongside the double mint.
#[test]
fn ancestor_rename_reaps_old_minted_and_mints_new() {
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
    let chain_a = dir_snap_nested(&[("a", covered(dir_snap_nested(&[("log", uncovered(10))])))]);
    let _ = respond(&mut e, pid, &chain_a, now);
    settle_minted_seeds(&mut e, sid, now);
    let old_mid = *discovery_subs_of(&e, sid).values().next().expect("minted");

    // mv /srv/a /srv/b — kernel-faithful for the minted FD: nothing at all (the minted Profile
    // stays Idle throughout; this is the no-terminal leak shape). Only the discovery anchor's
    // STRUCTURE notification fires, and the certified listing then shows the move.
    let chain_b = dir_snap_nested(&[("b", covered(dir_snap_nested(&[("log", uncovered(11))])))]);
    let t1 = now + Duration::from_millis(10);
    let (moved, t2) = burst_and_respond(&mut e, pid, srv, &chain_b, t1);
    assert!(
        moved.diagnostics.iter().any(|d| matches!(
            d,
            Diagnostic::DiscoverySubReaped { source, sub, path }
                if *source == sid && *sub == old_mid
                    && path.display().to_string() == "/srv/a/log"
        )),
        "the old path reaps, narrated with the OLD path; got {:?}",
        moved.diagnostics,
    );
    assert!(
        moved.diagnostics.iter().any(|d| matches!(
            d,
            Diagnostic::SubDetached { sub, reason: DetachReason::MatchVanished, .. }
                if *sub == old_mid
        )),
        "the reap carries MatchVanished",
    );
    assert_eq!(
        minted_paths(&moved),
        vec![(sid, "/srv/b/log".to_string())],
        "the new path mints in the same pass",
    );
    assert!(
        e.subs().get(old_mid).is_none(),
        "no zombie Sub at the old path"
    );
    let new_mid = *discovery_subs_of(&e, sid)
        .values()
        .next()
        .expect("re-minted");
    assert_ne!(
        new_mid, old_mid,
        "path identity changed — fresh history is correct",
    );
    settle_minted_seeds(&mut e, sid, t2);

    // mv /srv/b out of the pattern: reap only, nothing mints.
    let (gone, _) = burst_and_respond(
        &mut e,
        pid,
        srv,
        &dir_snap(&[]),
        t2 + Duration::from_millis(10),
    );
    assert!(
        gone.diagnostics.iter().any(|d| matches!(
            d,
            Diagnostic::SubDetached { sub, reason: DetachReason::MatchVanished, .. }
                if *sub == new_mid
        )),
        "non-matching rename reaps without minting",
    );
    assert!(minted_paths(&gone).is_empty());
    assert!(discovery_subs_of(&e, sid).is_empty());
    let _ = e.cancel_all_in_flight_probes();
}

/// An atomic replace of a matched terminus is identical to static: the terminal re-enters descent in
/// the loss step (Sub and Profile survive with identity intact), the descent finds the replacement,
/// the triggered Seed re-fires (`RecoveryFire` — fire history preserved), and the same-step reconcile
/// neither reaps nor double-mints the still-live terminus (anchor-slot membership keeps it in `M ∩ T`
/// while the minted Profile is mid-recovery). The template carries a content mask (the user's
/// file-watch shape) — single-sample verdicts, unlike the EMPTY-mask fixture template.
#[test]
fn terminus_replace_keeps_minted_sub_and_re_fires() {
    let mut e = Engine::new();
    let data = pre_place_dir(&mut e, &["data"]);
    let now = Instant::now();

    let template = Arc::new(MintTemplate {
        identity: ProfileIdentity::new(
            ScanConfig::builder().recursive(true).build(),
            MAX_SETTLE,
            DEFAULT_EVENTS,
        ),
        settle: SETTLE,
        spawn: SpawnSpec::new(empty_program(), EffectScope::SubtreeRoot, false),
    });
    let (src, dpid) = attach_discovery(
        &mut e,
        "disc",
        SubAttachAnchor::Resource(data),
        "/data/*",
        template,
        now,
    );
    // Discovery cold Seed: enumerate one terminus -> mint.
    let out = respond(
        &mut e,
        dpid,
        &dir_snap(&[("x.log", EntryKind::File, 5)]),
        now,
    );
    assert!(out.effects().is_empty(), "discovery never fires");
    let minted = discovery_subs_of(&e, src);
    assert_eq!(minted.len(), 1);
    let (&terminus, &mid) = minted.iter().next().unwrap();
    let minted_pid = pid_of(&e, mid);
    assert!(
        e.profiles()
            .get(minted_pid)
            .unwrap()
            .watch_root_parent()
            .is_some(),
        "minted Profile holds the watch_root_parent recovery channel from attach",
    );
    // Minted cold Seed pins silently.
    let t0 = seed_to_idle_with(
        &mut e,
        minted_pid,
        || anchor_ok(file_leaf(EntryKind::File, 5)),
        now,
    );

    // The minted Sub fires once (in-place content change).
    let t1 = fire_standard_once(&mut e, mid, terminus, 6, t0 + SETTLE);
    assert!(e.subs().get(mid).unwrap().has_fired());

    // Atomic replace: the terminal re-enters descent in the loss step — Sub and Profile survive.
    let t2 = t1 + Duration::from_millis(10);
    let out = e.step(
        Input::FsEvent {
            resource: terminus,
            event: FsEvent::Removed,
        },
        t2,
    );
    assert!(out.effects().is_empty());
    assert!(
        !out.diagnostics
            .iter()
            .any(|d| matches!(d, Diagnostic::DiscoverySubReaped { .. })),
        "no reap on the terminal — reconcile is the removal authority",
    );
    assert!(e.subs().get(mid).is_some(), "minted Sub survives");
    assert!(
        matches!(
            e.profiles().get(minted_pid).unwrap().state(),
            ProfileState::Pending(_)
        ),
        "the loss step re-enters descent",
    );

    // The rename's parent STRUCTURE event drives the discovery burst; the minted Profile's live
    // descent absorbs the same event (I5 — its probe is already in flight).
    let t3 = t2 + Duration::from_millis(1);
    let _ = e.step(
        Input::FsEvent {
            resource: data,
            event: FsEvent::StructureChanged,
        },
        t3,
    );
    // The minted descent finds the replacement (inode 7 != 6 — a fire-worthy change) and
    // materializes into a triggered Seed, Batching-first.
    let out = descent_advance(
        &mut e,
        minted_pid,
        &dir_snap(&[("x.log", EntryKind::File, 7)]),
        t3,
    );
    assert!(out.effects().is_empty(), "descent itself never fires");

    // Both settle windows expire in one drain: the discovery burst and the minted Seed go Verifying
    // together.
    let t4 = t3 + SETTLE * 2;
    drain_due(&mut e, t4);
    let out = respond(
        &mut e,
        dpid,
        &dir_snap(&[("x.log", EntryKind::File, 7)]),
        t4,
    );
    assert!(out.effects().is_empty(), "reconcile never fires");
    assert!(
        !out.diagnostics.iter().any(|d| {
            matches!(
                d,
                Diagnostic::DiscoverySubReaped { .. } | Diagnostic::DiscoveryMinted { .. }
            )
        }),
        "the replaced terminus kept its slot — no reap, no double-mint; got {:?}",
        out.diagnostics,
    );
    assert_eq!(
        discovery_subs_of(&e, src).into_values().collect::<Vec<_>>(),
        vec![mid],
        "same SubId across the replace",
    );

    // The minted Seed's verify folds the recovery verdict: fired before + witness drift ⇒
    // RecoveryFire. Full parity with the static replace.
    let out = respond_anchor_file(&mut e, minted_pid, 7, t4);
    assert_eq!(
        out.effects().len(),
        1,
        "the replace re-fires through the surviving Sub",
    );
    assert!(
        e.subs().get(mid).unwrap().has_fired(),
        "fire history preserved across the replace",
    );
    // Drain the fire cycle: effect Ok -> rebase -> Idle.
    let key = out.effects()[0].key();
    let _ = complete_effect_to_rebasing(&mut e, mid, key, t4);
    let _ = respond_anchor_file(&mut e, minted_pid, 7, t4);
    assert!(matches!(
        e.profiles().get(minted_pid).unwrap().state(),
        ProfileState::Idle
    ));
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

/// Prefix `rm -rf` recovery: every Profile re-enters its own recovery descent (anchor loss is uniform
/// — the minted Sub *survives*, parked `Pending`), the discovery Profile recovers through
/// `watch_root_parent` descent, and the recovery Seed's reconcile finds the reappeared terminus live
/// (slot identity) — no reap, no re-mint — with **no** `PerFileDriftDroppedOnRecovery` even though
/// the template stores a per-file scope (the template's reaction is minting, not a per-file Effect).
#[test]
fn prefix_rm_rf_recovery_preserves_minted_sub_without_per_file_warning() {
    let mut e = Engine::new();
    let data = pre_place_dir(&mut e, &["data"]);
    let root = e.tree().parent(data).expect("FS root");
    let now = Instant::now();
    let (sid, pid, _) = attach_discovery_returning(
        &mut e,
        "disc",
        SubAttachAnchor::Resource(data),
        "/data/*",
        mint_template_scoped(EffectScope::PerStableFile),
        now,
    );
    let _ = respond(&mut e, pid, &dir_snap(&[("x", EntryKind::Dir, 1)]), now);
    settle_minted_seeds(&mut e, sid, now);
    let old_mid = *discovery_subs_of(&e, sid).values().next().expect("minted");
    let x = e.tree().lookup(Some(data), "x").expect("terminus slot");

    // rm -rf bottom-up: terminus first (the minted Profile re-enters descent at its
    // watch_root_parent), then the anchor (same uniform loss path for the discovery Profile;
    // watch_root_parent is preserved either way).
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
    assert!(
        e.subs().get(old_mid).is_some(),
        "minted Sub survives the loss window — reconcile is the removal authority",
    );
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
    // The witnessed loss re-entry materializes into a triggered Seed — Batching-first; drain the
    // settle window to surface the verify probe.
    let t3 = t2 + SETTLE * 2;
    drain_due(&mut e, t3);
    let recovered = respond(&mut e, pid, &dir_snap(&[("x", EntryKind::Dir, 9)]), t3);
    assert!(
        minted_paths(&recovered).is_empty(),
        "the reappeared terminus kept its slot — the still-attached Sub dedups the mint",
    );
    assert!(
        !recovered
            .diagnostics
            .iter()
            .any(|d| matches!(d, Diagnostic::DiscoverySubReaped { .. })),
        "a live terminus is never a removal victim",
    );
    let survivor = *discovery_subs_of(&e, sid).values().next().expect("minted");
    assert_eq!(
        survivor, old_mid,
        "same Sub across the rm -rf window — identity preserved",
    );
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

/// Overflow membership is **anchors-only**, gap-case direction: a Profile holding descendant-FD
/// contributions *inside* an `OverflowScope::Resource(r)` scope, anchor above `r`, is NOT reseeded
/// — no probe, no burst — even though events under `r` are exactly what the overflow window may
/// have dropped for it. Asserted as chosen: `profiles_in_subtree` walks anchor ancestry only. The
/// alternative a per-stream backend will force is contribution-based membership — "does this
/// Profile hold any contribution at-or-below `r`", enumerable off the Tree's per-Resource
/// contribution map (`ContribKey` is fully Profile-keyed, no `covers` calls). The case stays latent
/// today because the Profile anchored *below* `r` reseeds through the same rule, usually
/// re-observing the shared subtree — the second half of the test pins that reach.
#[test]
fn overflow_resource_scope_skips_profile_holding_descendant_fds_inside_scope() {
    let mut e = Engine::new();
    let srv = pre_place_dir(&mut e, &["srv"]);
    let now = Instant::now();
    let (sid, disc_pid) = attach_discovery(
        &mut e,
        "disc",
        SubAttachAnchor::Resource(srv),
        "/srv/*/log",
        mint_template(),
        now,
    );
    let chain = dir_snap_nested(&[("a", covered(dir_snap_nested(&[("log", uncovered(10))])))]);
    let _ = respond(&mut e, disc_pid, &chain, now);
    settle_minted_seeds(&mut e, sid, now);
    let a = e.tree().lookup(Some(srv), "a").expect("chain dir slot");

    // Premise — without this the pin pins nothing: the discovery Profile holds a descendant-FD
    // contribution at `a` (the chain dir the reconciler watched on its behalf) while its anchor
    // (`/srv`) sits above `a`.
    assert!(
        e.tree()
            .get(a)
            .unwrap()
            .contributions()
            .contains_key(&ContribKey::ProfileDescendant(disc_pid)),
        "premise: discovery Profile holds a chain-dir contribution inside the overflow scope",
    );

    let out = e.step(
        Input::SensorOverflow {
            scope: OverflowScope::Resource(a),
        },
        now + Duration::from_millis(10),
    );

    assert!(
        matches!(
            e.profiles().get(disc_pid).unwrap().state(),
            ProfileState::Idle
        ),
        "anchors-only membership: the contribution-holding Profile is not reseeded",
    );
    assert!(
        !out.probe_ops().iter().any(|op| matches!(
            op,
            ProbeOp::Probe { request } if request.owner() == disc_pid
        )),
        "no probe for the skipped Profile",
    );
    // The minted Profile (anchor at `a/log`, inside the scope) reseeds through the same rule — the
    // scope itself was honoured, just anchor-scoped.
    let minted_pid = pid_of(
        &e,
        *discovery_subs_of(&e, sid).values().next().expect("minted"),
    );
    assert!(
        e.pending_probe_for(minted_pid).is_some(),
        "the Profile anchored below the scope root reseeds",
    );
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

/// A static Sub joining the minted identity makes the minted Profile mixed; anchor loss is the same
/// uniform recovery descent regardless, and the pin is the registry seam: the projection keys
/// minted Subs only (the static co-resident never confuses it), the recovering terminus is back in
/// the certified set so the Sub is no removal victim, and the dedup finds it still attached — no
/// double-mint after recovery.
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
        "anchor loss re-enters descent — the minted Sub survives the terminal",
    );
    assert!(
        !rm.diagnostics
            .iter()
            .any(|d| matches!(d, Diagnostic::DiscoverySubReaped { .. })),
        "no reap narration in the loss step",
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

/// A `/*` root pattern anchors the discovery Profile at the FS root itself: no `watch_root_parent`
/// edge exists (there is no parent to watch), the attach opens the cold Seed directly (the root is
/// always materialised), and a minted terminus joins paths cleanly at `/` — `disc@/x`, never
/// `disc@//x`.
#[test]
fn root_pattern_installs_no_parent_edge() {
    let mut e = Engine::new();
    let now = Instant::now();
    let (sid, pid) = attach_discovery(
        &mut e,
        "disc",
        SubAttachAnchor::Path("/".into()),
        "/*",
        mint_template(),
        now,
    );
    let p = e.profiles().get(pid).expect("discovery Profile live");
    assert!(
        p.watch_root_parent().is_none(),
        "root anchor has no parent edge",
    );
    assert!(
        matches!(p.state(), ProfileState::Active(ActiveBurst::PreFire(_), _)),
        "root always exists — cold Seed, never Pending",
    );

    let out = respond(&mut e, pid, &dir_snap(&[("x", EntryKind::Dir, 1)]), now);
    assert_eq!(
        minted_paths(&out),
        vec![(sid, "/x".to_string())],
        "terminus path joins at the root without a doubled separator",
    );
    assert_eq!(
        e.subs()
            .get(*discovery_subs_of(&e, sid).values().next().unwrap())
            .unwrap()
            .name,
        "disc@/x",
    );
    let _ = e.cancel_all_in_flight_probes();
}

/// Detaching one of two same-pattern templates cascades only its own minted set: the sibling
/// template, its minted Sub, the shared discovery Profile, and the shared minted Profile all
/// survive, and the next reconcile is still a dedup no-op for the survivor.
#[test]
fn second_template_detach_leaves_siblings_minted_set_intact() {
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
    let (sid2, _) = attach_discovery(
        &mut e,
        "disc2",
        SubAttachAnchor::Resource(data),
        "/data/*",
        mint_template(),
        now,
    );
    let snap = dir_snap(&[("x", EntryKind::Dir, 1)]);
    let _ = respond(&mut e, pid, &snap, now);
    settle_minted_seeds(&mut e, sid1, now);
    settle_minted_seeds(&mut e, sid2, now);
    let mid1 = *discovery_subs_of(&e, sid1).values().next().expect("minted");
    let mid2 = *discovery_subs_of(&e, sid2).values().next().expect("minted");
    let minted_pid = pid_of(&e, mid1);
    assert_eq!(
        minted_pid,
        pid_of(&e, mid2),
        "identical minted identity shares one minted Profile",
    );

    let out = e.step(Input::DetachSub(sid1), now + Duration::from_millis(10));
    assert!(
        out.diagnostics.iter().any(|d| matches!(
            d,
            Diagnostic::SubDetached { sub, reason: DetachReason::DiscoverySourceDetached, .. }
                if *sub == mid1
        )),
        "detached template's mint cascades; got {:?}",
        out.diagnostics,
    );
    assert!(e.subs().get(mid1).is_none());
    assert!(e.subs().get(sid2).is_some(), "sibling template survives");
    assert!(
        e.subs().get(mid2).is_some(),
        "sibling's minted Sub survives"
    );
    assert!(
        e.profiles().get(pid).is_some(),
        "shared discovery Profile survives on the sibling template",
    );
    assert!(
        e.profiles().get(minted_pid).is_some(),
        "shared minted Profile survives on the sibling's mint",
    );

    let (out2, _) = burst_and_respond(&mut e, pid, data, &snap, now + Duration::from_millis(20));
    assert!(
        minted_paths(&out2).is_empty(),
        "survivor reconcile is a dedup no-op — no re-mint for the detached template either",
    );
    assert_eq!(discovery_subs_of(&e, sid2).len(), 1);
    let _ = e.cancel_all_in_flight_probes();
}

/// The minted Sub's cold-Seed burst deadline derives from reconcile-time `now` — the instant the
/// mint happened — not from the discovery template's attach instant. The timer-heap boundary is the
/// witness: nothing referenced is due one tick before `reconcile_now + max_settle`, and the minted
/// Profile's `BurstDeadline` is due exactly at it.
#[test]
fn minted_seed_burst_deadline_derives_from_reconcile_now() {
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

    // Answer the cold probe a full second after attach: a deadline threaded from any earlier
    // instant (the attach, the event) would fall due before the boundary probed below.
    let t_mint = now + Duration::from_secs(1);
    let _ = respond(&mut e, pid, &dir_snap(&[("x", EntryKind::Dir, 1)]), t_mint);
    let mid = *discovery_subs_of(&e, sid).values().next().expect("minted");
    let minted_pid = pid_of(&e, mid);

    // The attach-derived instant: a deadline threaded from the template's attach `now` would fall
    // due exactly here, a full second early.
    assert!(
        e.pop_expired(now + MAX_SETTLE).is_none(),
        "nothing due at attach-now + max_settle — the deadline is not attach-derived",
    );
    let entry = e
        .pop_expired(t_mint + MAX_SETTLE)
        .expect("minted Seed deadline due exactly at reconcile-now + max_settle");
    assert_eq!(entry.profile, minted_pid);
    assert_eq!(entry.kind, TimerKind::BurstDeadline);
    let _ = e.cancel_all_in_flight_probes();
}

/// A `Vanished` descent response releases only the descender's prefix claim: a discovery Profile
/// anchored at the shared prefix keeps its watch (no `Unwatch`, demand drops by exactly the
/// descent's contribution) and its own lifecycle is untouched, while the descender rewinds to the
/// FS root and re-probes.
#[test]
fn descent_vanish_preserves_co_resident_discovery_contribution() {
    let mut e = Engine::new();
    let data = pre_place_dir(&mut e, &["data"]);
    let now = Instant::now();
    let (_, disc_pid) = attach_discovery(
        &mut e,
        "disc",
        SubAttachAnchor::Resource(data),
        "/data/*",
        mint_template(),
        now,
    );
    let _ = respond(&mut e, disc_pid, &dir_snap(&[]), now);
    let demand_resident = e.tree().get(data).unwrap().watch_demand();

    // A static Sub pends below the discovery anchor; its descent contributes to the shared prefix.
    let (_, static_pid) = attach(
        &mut e,
        "deep",
        SubAttachAnchor::Path("/data/missing/deep".into()),
        ScanConfig::builder().recursive(true).build(),
        ClassSet::EMPTY,
        MAX_SETTLE,
        now,
    );
    assert!(matches!(
        e.profiles().get(static_pid).unwrap().state(),
        ProfileState::Pending(_)
    ));
    assert_eq!(
        e.tree().get(data).unwrap().watch_demand(),
        demand_resident + 1,
        "descent claim joins the discovery anchor's",
    );

    let corr = e.pending_probe_for(static_pid).expect("descent in flight");
    let out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: static_pid,
            correlation: corr,
            outcome: ProbeOutcome::Vanished,
        }),
        now + Duration::from_millis(10),
    );

    assert!(
        !out.watch_ops
            .iter()
            .any(|op| matches!(op, WatchOp::Unwatch { resource } if *resource == data)),
        "the co-resident's watch holds the shared prefix — no Unwatch",
    );
    assert_eq!(
        e.tree().get(data).unwrap().watch_demand(),
        demand_resident,
        "only the descent's own contribution released",
    );
    assert!(
        matches!(
            e.profiles().get(disc_pid).unwrap().state(),
            ProfileState::Idle
        ),
        "discovery Profile untouched by the sibling's rewind",
    );
    assert_eq!(
        last_probe_path(&out),
        Some("/".into()),
        "descender rewound to the FS root and re-probed",
    );
    assert!(matches!(
        e.profiles().get(static_pid).unwrap().state(),
        ProfileState::Pending(_)
    ));
    let _ = e.cancel_all_in_flight_probes();
}

/// `rm -rf` above the anchor: recovery descent rewinds one level per `Vanished` response, the chain
/// terminating at the FS root (where a probe is always answerable), then re-advances component by
/// component and the recovery Seed's reconcile finds the reappeared terminus live — the minted Sub
/// survives the whole cascade. The probe-target ladder `/a/b → /a → / → /a → /a/b → /a/b/c` is the
/// witness.
#[test]
fn recovery_cascade_rewinds_through_parent_to_fs_root() {
    let mut e = Engine::new();
    let c = pre_place_dir(&mut e, &["a", "b", "c"]);
    let now = Instant::now();
    let (sid, pid) = attach_discovery(
        &mut e,
        "disc",
        SubAttachAnchor::Resource(c),
        "/a/b/c/*",
        mint_template(),
        now,
    );
    let _ = respond(&mut e, pid, &dir_snap(&[("x", EntryKind::Dir, 1)]), now);
    settle_minted_seeds(&mut e, sid, now);
    let old_mid = *discovery_subs_of(&e, sid).values().next().expect("minted");
    let x = e.tree().lookup(Some(c), "x").expect("terminus slot");

    // rm -rf /a, bottom-up: both anchors go terminal and both Profiles re-enter their own recovery
    // descents (the minted Sub stays attached throughout — reconcile is its removal authority, and
    // the terminus reappears below).
    let t1 = now + Duration::from_millis(10);
    let _ = e.step(
        Input::FsEvent {
            resource: x,
            event: FsEvent::Removed,
        },
        t1,
    );
    let loss = e.step(
        Input::FsEvent {
            resource: c,
            event: FsEvent::Removed,
        },
        t1,
    );
    assert!(
        e.subs().get(old_mid).is_some(),
        "minted Sub survives the cascade's terminals",
    );

    // The loss step itself re-enters descent at the watch_root_parent — no entry event needed. /a/b
    // and /a are gone too, so each probe answers Vanished and the descent rewinds one level,
    // re-arming at the parent.
    assert_eq!(last_probe_path(&loss), Some("/a/b".into()));
    let vanish = |e: &mut Engine, at| {
        let corr = e.pending_probe_for(pid).expect("descent probe in flight");
        e.step(
            Input::ProbeResponse(ProbeResponse {
                owner: pid,
                correlation: corr,
                outcome: ProbeOutcome::Vanished,
            }),
            at,
        )
    };
    let r1 = vanish(&mut e, t1);
    assert_eq!(last_probe_path(&r1), Some("/a".into()), "first rewind");
    let r2 = vanish(&mut e, t1);
    assert_eq!(
        last_probe_path(&r2),
        Some("/".into()),
        "second rewind terminates at the FS root",
    );

    // Forward again: the tree reappears one component per response, the anchor materialises, and
    // the recovery Seed reconciles into a fresh mint.
    let a1 = descent_advance(&mut e, pid, &dir_snap(&[("a", EntryKind::Dir, 2)]), t1);
    assert_eq!(last_probe_path(&a1), Some("/a".into()));
    let a2 = descent_advance(&mut e, pid, &dir_snap(&[("b", EntryKind::Dir, 3)]), t1);
    assert_eq!(last_probe_path(&a2), Some("/a/b".into()));
    let a3 = descent_advance(&mut e, pid, &dir_snap(&[("c", EntryKind::Dir, 4)]), t1);
    assert_eq!(
        last_probe_path(&a3),
        None,
        "anchor materialised into a triggered Seed — Batching-first, no cold walk",
    );
    // The ladder's last rung surfaces at settle expiry: the recovery Seed probes the subtree.
    let t2 = t1 + SETTLE * 2;
    let mut rung = None;
    while let Some(en) = e.pop_expired(t2) {
        let o = e.step(
            Input::TimerExpired {
                profile: en.profile,
                kind: en.kind,
                id: en.id,
            },
            t2,
        );
        if last_probe_path(&o).is_some() {
            rung = Some(o);
        }
    }
    let rung = rung.expect("recovery Seed probe after settle expiry");
    assert_eq!(
        last_probe_path(&rung),
        Some("/a/b/c".into()),
        "anchor materialised — recovery Seed probes the subtree",
    );
    let recovered = respond(&mut e, pid, &dir_snap(&[("x", EntryKind::Dir, 5)]), t2);
    assert!(
        minted_paths(&recovered).is_empty(),
        "the reappeared terminus kept its slot — no re-mint past the surviving Sub",
    );
    let survivor = *discovery_subs_of(&e, sid).values().next().expect("minted");
    assert_eq!(survivor, old_mid, "same Sub across the rm -rf cascade");
    let _ = e.cancel_all_in_flight_probes();
}

/// Three loss → recovery cycles leave every shared refcount where one cycle leaves it: the parent
/// edge's demand, the anchor's demand, and the minted set are invariant across cycles — the minted
/// Sub survives each cycle through its own recovery descent (cycle 0's witnessed recovery owes and
/// fires the first fire; later cycles pin silently on the unchanged witness), and repeated recovery
/// neither leaks nor double-releases parent-edge contributions.
#[test]
fn repeated_loss_recovery_cycles_keep_prefix_parent_refcount_invariant() {
    let mut e = Engine::new();
    let data = pre_place_dir(&mut e, &["data"]);
    let root = e.tree().parent(data).expect("FS root");
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
    let root_demand = e.tree().get(root).unwrap().watch_demand();
    let anchor_demand = e.tree().get(data).unwrap().watch_demand();

    let mut at = now;
    for cycle in 0..3u64 {
        at += Duration::from_millis(10);
        let anchor = e
            .tree()
            .lookup(Some(root), "data")
            .expect("anchor slot live this cycle");
        let x = e.tree().lookup(Some(anchor), "x").expect("terminus slot");
        let _ = e.step(
            Input::FsEvent {
                resource: x,
                event: FsEvent::Removed,
            },
            at,
        );
        let _ = e.step(
            Input::FsEvent {
                resource: anchor,
                event: FsEvent::Removed,
            },
            at,
        );
        let _ = e.step(
            Input::FsEvent {
                resource: root,
                event: FsEvent::StructureChanged,
            },
            at,
        );
        let _ = descent_advance(
            &mut e,
            pid,
            &dir_snap(&[("data", EntryKind::Dir, 10 + cycle)]),
            at,
        );
        // Witnessed loss re-entry ⇒ triggered Seed, Batching-first: drain the settle window to
        // surface the verify probe.
        at += SETTLE * 2;
        drain_due(&mut e, at);
        let recovered = respond(
            &mut e,
            pid,
            &dir_snap(&[("x", EntryKind::Dir, 20 + cycle)]),
            at,
        );
        assert!(
            minted_paths(&recovered).is_empty(),
            "cycle {cycle}: the surviving Sub dedups the mint",
        );
        assert!(
            !recovered
                .diagnostics
                .iter()
                .any(|d| matches!(d, Diagnostic::DiscoverySubReaped { .. })),
            "cycle {cycle}: a live terminus is never a removal victim",
        );
        // The minted Profile recovers through its own descent — same parent listing.
        at = settle_minted(
            &mut e,
            mid,
            Some(&dir_snap(&[("x", EntryKind::Dir, 20 + cycle)])),
            at,
        );

        assert_eq!(
            e.tree().get(root).unwrap().watch_demand(),
            root_demand,
            "cycle {cycle}: parent-edge demand invariant",
        );
        let anchor = e.tree().lookup(Some(root), "data").expect("anchor live");
        assert_eq!(
            e.tree().get(anchor).unwrap().watch_demand(),
            anchor_demand,
            "cycle {cycle}: anchor demand invariant",
        );
        assert_eq!(
            discovery_subs_of(&e, sid).into_values().collect::<Vec<_>>(),
            vec![mid],
            "cycle {cycle}: same minted Sub across cycles",
        );
        assert_eq!(
            e.profiles().get(pid).unwrap().watch_root_parent(),
            Some(root),
            "cycle {cycle}: parent edge points at the FS root",
        );
    }
    let _ = e.cancel_all_in_flight_probes();
}
