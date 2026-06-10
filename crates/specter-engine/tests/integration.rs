//! Cross-module integration tests for `specter-engine`.
//!
//! Two suites:
//! - **P3-era primitives**: the `covers` predicate against a real `Tree` + `ProfileMap`. Its
//!   transitive derivation (`nearest_covering_ancestor`) and the reconfirm query built on it
//!   (`has_active_standard_descendant`) are engine-internal, so their units live inline in
//!   `coverage::tests`; the `Draining → Verifying` reconfirm is exercised end-to-end through the
//!   burst lifecycle in `tests/multi_profile.rs`.
//! - **P4 lifecycle**: full `Idle ↔ Active(Burst)` flows driven through `Engine::attach_sub` and
//!   `Engine::step` against a `MockSensor`-style harness (assertions read from `StepOutput`).

use specter_core::testkit::{dir_snap, proven};
use specter_core::{
    BurstIntent, ClassSet, Diagnostic, DirSnapshot, EntryKind, FsEvent, Input, ProbeCorrelation,
    ProbeOp, ProbeOutcome, ProbeResponse, Profile, ProfileId, ProfileIdentity, ProfileMap,
    ProfileState, ResourceId, ResourceKind, ResourceRole, ScanConfig, StepOutput, SubAttachAnchor,
    Tree, WatchOp,
};
use specter_engine::testkit::{
    anchor_dir, assert_seed_verifying, attach_returning, complete_effect_to_rebasing,
    first_probe_correlation, rebase_post_fire_to_idle, seed_to_idle,
};
use specter_engine::{Engine, covers};
use std::path::PathBuf;
use std::time::{Duration, Instant};

const SETTLE: Duration = Duration::from_millis(100);
const MAX_SETTLE: Duration = Duration::from_secs(6);
const NO_EVENTS: ClassSet = ClassSet::EMPTY;
/// Production-realistic `EffectScope::SubtreeRoot` events mask — CONTENT in the mask sets
/// `events_witness_quiescence == true`, so a single Authoritative sample closes the verdict floor's
/// hash-equality obligation. Tests that drive the N=2 hash channel directly opt into `NO_EVENTS`
/// (or a CONTENT-free mask) inline.
const DEFAULT_EVENTS: ClassSet = ClassSet::DEFAULT_SUBTREE_ROOT;

fn cfg_recursive() -> ScanConfig {
    ScanConfig::builder().recursive(true).build()
}

fn mark_dir(tree: &mut Tree, id: ResourceId) {
    tree.set_kind(id, ResourceKind::Dir);
}

#[test]
fn engine_default_constructible() {
    let e = Engine::new();
    assert!(e.next_deadline().is_none());
}

#[test]
fn covers_handles_pattern_with_dir_bypass_in_engine_context() {
    let mut tree = Tree::new();
    let mut profiles = ProfileMap::new();
    let root = tree.ensure_root("root", ResourceRole::User);
    let src = tree
        .ensure_child(root, "src", ResourceRole::User)
        .expect("test live parent");
    let lib_rs = tree
        .ensure_child(src, "lib.rs", ResourceRole::User)
        .expect("test live parent");
    let lib_c = tree
        .ensure_child(src, "lib.c", ResourceRole::User)
        .expect("test live parent");
    mark_dir(&mut tree, root);
    mark_dir(&mut tree, src);
    tree.set_kind(lib_rs, ResourceKind::File);
    tree.set_kind(lib_c, ResourceKind::File);

    let p = profiles.attach(
        &mut tree,
        Profile::new(
            root,
            ProfileIdentity {
                config: ScanConfig::builder()
                    .recursive(true)
                    .pattern(specter_core::GlobPattern::compile("*.rs").unwrap())
                    .build(),
                max_settle: MAX_SETTLE,
                events: NO_EVENTS,
            },
            SETTLE,
            None,
        ),
    );
    let profile = profiles.get(p).unwrap();

    assert!(
        covers(profile, src, &tree, &mut PathBuf::new()),
        "Dir bypasses pattern"
    );
    assert!(
        covers(profile, lib_rs, &tree, &mut PathBuf::new()),
        "matching File covered"
    );
    assert!(
        !covers(profile, lib_c, &tree, &mut PathBuf::new()),
        "non-matching File uncovered"
    );
}

// ---------- P4 single-Profile lifecycle scenarios ----------

/// Walk through a Standard burst — drains settle timers, injects probe responses with `snap` until
/// the burst stabilizes and emits an Effect. Returns the `StepOutput` containing the Effect.
///
/// `t0` is the moment the `FsEvent` fired. The walker advances time by large strides each iteration
/// so backoff catches up.
fn drive_standard_burst_to_stable(
    e: &mut Engine,
    pid: specter_core::ProfileId,
    snap: &std::sync::Arc<DirSnapshot>,
    t0: Instant,
) -> StepOutput {
    let mut t = t0;
    for _ in 0..8 {
        t += SETTLE * 4;
        let correlation = drain_to_probe_correlation(e, t);
        if let Some(c) = correlation {
            let out = e.step(
                Input::ProbeResponse(ProbeResponse {
                    owner: pid,
                    correlation: c,
                    outcome: proven(std::sync::Arc::clone(snap)),
                }),
                t,
            );
            if !out.effects().is_empty() {
                return out;
            }
        }
    }
    panic!("Standard burst failed to stabilize within drive iterations");
}

/// Drain timers and return the most recent probe's correlation, if any fired in the process.
fn drain_to_probe_correlation(e: &mut Engine, t: Instant) -> Option<ProbeCorrelation> {
    let mut last_correlation = None;
    while let Some(entry) = e.pop_expired(t) {
        let out = e.step(
            Input::TimerExpired {
                profile: entry.profile,
                kind: entry.kind,
                id: entry.id,
            },
            t,
        );
        if let Some(c) = first_probe_correlation(&out) {
            last_correlation = Some(c);
        }
    }
    last_correlation
}

#[test]
fn golden_path_full_lifecycle() {
    // The whole V4 spine: attach_sub → Seed → Idle → FsEvent → Standard → Effect → EffectComplete →
    // Seed → Idle. Each transition observable in the StepOutputs.
    let mut e = Engine::new();
    let r = anchor_dir(&mut e, "src");
    let now = Instant::now();
    let t0 = now;
    let (sid, pid, attach_out) = attach_returning(
        &mut e,
        "build",
        SubAttachAnchor::Resource(r),
        cfg_recursive(),
        DEFAULT_EVENTS,
        MAX_SETTLE,
        t0,
    );

    // attach_sub emits Watch (anchor) but NO Probe — a Seed is Batching-first; the first Seed probe
    // fires only on settle expiry.
    assert!(
        attach_out
            .watch_ops
            .iter()
            .any(|op| matches!(op, WatchOp::Watch { .. }))
    );
    assert!(
        first_probe_correlation(&attach_out).is_some(),
        "cold-arm Seed: probe emitted at burst construction",
    );
    assert!(attach_out.effects().is_empty());

    // Seed quiescence proof → baseline = current = empty snapshot; → Idle. Never emits Effects
    // (fresh Profile).
    let snap_seed = dir_snap(&[]);
    let _ = seed_to_idle(&mut e, pid, &snap_seed, t0);

    // FsEvent on anchor → Standard Settling. The Seed consumed its settle window; keep instants
    // monotonic by opening the Standard burst strictly after.
    let t1 = t0 + SETTLE * 2 + Duration::from_millis(10);
    let fs_out = e.step(
        Input::FsEvent {
            resource: r,
            event: FsEvent::ContentChanged,
        },
        t1,
    );
    assert!(fs_out.effects().is_empty());

    // Drive the Standard burst to a stable verdict (probing against the empty snapshot; the burst
    // stabilizes when current matches the response). The walker advances time and injects probe
    // responses.
    let stable_out = drive_standard_burst_to_stable(&mut e, pid, &snap_seed, t1);
    assert_eq!(stable_out.effects().len(), 1);
    assert!(!stable_out.effects()[0].forced);

    // EffectComplete::Ok drives the burst out of Awaiting and directly into Rebasing (probe-first) —
    // the WholeSubtree rebase probe is emitted in this very step to capture the post-command tree.
    let _ =
        complete_effect_to_rebasing(&mut e, sid, stable_out.effects()[0].key(), t1 + SETTLE * 16);

    // Post-fire rebase (idempotent — same snapshot) → Idle; baseline := current (the post-command
    // tree).
    let _ = rebase_post_fire_to_idle(
        &mut e,
        pid,
        &snap_seed,
        t1 + SETTLE * 16 + Duration::from_millis(1),
    );
    assert!(
        matches!(e.profiles().get(pid).unwrap().state(), ProfileState::Idle),
        "idempotent rebase loop closes Stable → Idle",
    );
}

#[test]
fn trailing_latched_anchor_event_does_not_double_fire() {
    // After a fire cycle completes (Profile → Idle, baseline := the post-command tree), a trailing
    // latched anchor FsEvent — the late event the watcher now delivers post-burst — opens a spurious
    // Standard burst. Its verdict is `baseline == current` (the rebase just set it) and the Sub has
    // already fired, so B1 dedup suppresses: the burst finishes to Idle with no second Effect.
    let mut e = Engine::new();
    let r = anchor_dir(&mut e, "src");
    let now = Instant::now();
    let t0 = now;
    let (sid, pid, attach_out) = attach_returning(
        &mut e,
        "build",
        SubAttachAnchor::Resource(r),
        cfg_recursive(),
        DEFAULT_EVENTS,
        MAX_SETTLE,
        t0,
    );

    // Seed → Idle with an empty baseline. The cold-arm Seed answers its in-flight probe and pins
    // the empty baseline directly.
    assert!(
        first_probe_correlation(&attach_out).is_some(),
        "cold-arm Seed: probe emitted at burst construction",
    );
    let snap = dir_snap(&[]);
    let _ = seed_to_idle(&mut e, pid, &snap, t0);

    // First fire cycle: FsEvent → Standard → Effect → EffectComplete → Rebase Ok → Idle. The Sub's
    // has_fired is set by the emission. Open the Standard burst after the Seed's settle window so
    // instants stay monotonic.
    let t1 = t0 + SETTLE * 2 + Duration::from_millis(10);
    e.step(
        Input::FsEvent {
            resource: r,
            event: FsEvent::ContentChanged,
        },
        t1,
    );
    let stable_out = drive_standard_burst_to_stable(&mut e, pid, &snap, t1);
    assert_eq!(stable_out.effects().len(), 1, "first fire emits one Effect");
    let _ =
        complete_effect_to_rebasing(&mut e, sid, stable_out.effects()[0].key(), t1 + SETTLE * 16);
    // Post-fire rebase (idempotent) → Idle; baseline := the post-command tree. The loop closes to
    // Idle before the trailing event below.
    let _ = rebase_post_fire_to_idle(
        &mut e,
        pid,
        &snap,
        t1 + SETTLE * 16 + Duration::from_millis(1),
    );
    assert!(
        matches!(e.profiles().get(pid).unwrap().state(), ProfileState::Idle),
        "idempotent rebase loop closes Stable → Idle",
    );

    // Trailing latched anchor event opens a spurious Standard burst.
    let t2 = t1 + SETTLE * 20;
    e.step(
        Input::FsEvent {
            resource: r,
            event: FsEvent::ContentChanged,
        },
        t2,
    );

    // Drive the spurious burst: every verify responds with the same unchanged tree. B1 dedup must
    // suppress — no Effect ever — and the burst must finish cleanly back to Idle.
    let mut t = t2;
    let mut returned_to_idle = false;
    for _ in 0..8 {
        t += SETTLE * 4;
        if let Some(c) = drain_to_probe_correlation(&mut e, t) {
            let out = e.step(
                Input::ProbeResponse(ProbeResponse {
                    owner: pid,
                    correlation: c,
                    outcome: proven(snap.clone()),
                }),
                t,
            );
            assert!(
                out.effects().is_empty(),
                "trailing latched anchor event must not re-fire (B1 dedup); got {:?}",
                out.effects(),
            );
        }
        if matches!(
            e.profiles().get(pid).unwrap().state(),
            specter_core::ProfileState::Idle,
        ) {
            returned_to_idle = true;
            break;
        }
    }
    assert!(
        returned_to_idle,
        "spurious burst from the trailing anchor event finished cleanly to Idle — no double-fire",
    );
}

#[test]
fn vanished_during_seed_clears_baseline_and_diagnoses() {
    let mut e = Engine::new();
    let r = anchor_dir(&mut e, "log.txt");
    e.tree_mut().set_kind(r, ResourceKind::File);
    let t0 = Instant::now();
    let (_sid, pid, out) = attach_returning(
        &mut e,
        "fmt",
        SubAttachAnchor::Resource(r),
        ScanConfig::builder().build(),
        NO_EVENTS,
        MAX_SETTLE,
        t0,
    );

    // The cold-arm Seed answers its in-flight Verify probe with a Vanished — terminal, no further
    // sample needed.
    assert!(
        first_probe_correlation(&out).is_some(),
        "cold-arm Seed: probe emitted at burst construction",
    );
    let (correlation, _) = assert_seed_verifying(&mut e, pid, t0);

    let resp_out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid,
            correlation,
            outcome: ProbeOutcome::Vanished,
        }),
        t0 + SETTLE + Duration::from_millis(1),
    );
    assert!(resp_out.diagnostics.iter().any(|d| matches!(
        d,
        Diagnostic::ProbeVanished {
            intent: BurstIntent::Seed,
            ..
        }
    )));
}

#[test]
fn pending_event_race_late_probe_response_discarded() {
    let mut e = Engine::new();
    let r = anchor_dir(&mut e, "src");
    let now = Instant::now();
    let t0 = now;
    let (_sid, pid, attach_out) = attach_returning(
        &mut e,
        "build",
        SubAttachAnchor::Resource(r),
        cfg_recursive(),
        NO_EVENTS,
        MAX_SETTLE,
        t0,
    );

    // The Seed is Batching-first (no probe at attach). The first Seed probe materializes only after
    // the initial settle expiry; its correlation is the one the intervening FsEvent will render
    // stale.
    assert!(
        first_probe_correlation(&attach_out).is_some(),
        "cold-arm Seed: probe emitted at burst construction",
    );
    let (stale_correlation, _) = assert_seed_verifying(&mut e, pid, t0);

    // Inject FsEvent while the first Seed probe is in flight (Verifying). `event_drives_batching`
    // Cancels + disarms that verify and re-arms Batching, preserving Seed intent — the correlation
    // above is now stale.
    let evt_t = t0 + SETTLE + Duration::from_millis(1);
    let _evt_out = e.step(
        Input::FsEvent {
            resource: r,
            event: FsEvent::ContentChanged,
        },
        evt_t,
    );

    // Late ProbeResponse with the now-stale correlation arrives.
    let late_resp = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid,
            correlation: stale_correlation,
            outcome: proven(dir_snap(&[])),
        }),
        evt_t + Duration::from_millis(1),
    );
    assert!(
        late_resp
            .diagnostics
            .iter()
            .any(|d| matches!(d, Diagnostic::StaleProbeResponse { .. }))
    );
    // No baseline change; Profile still Active (re-batched as Seed).
    assert!(
        matches!(
            e.profiles().get(pid).unwrap().state(),
            specter_core::ProfileState::Active(..),
        ),
        "Profile remains Active after the stale late response is discarded",
    );
    assert!(
        e.profiles().get(pid).unwrap().baseline().is_none(),
        "stale late response must not commit a baseline",
    );
}

#[test]
fn seed_burst_descendants_watched_via_first_probe() {
    let mut e = Engine::new();
    let r = anchor_dir(&mut e, "src");
    let t0 = Instant::now();
    let (_sid, pid, attach_out) = attach_returning(
        &mut e,
        "build",
        SubAttachAnchor::Resource(r),
        cfg_recursive(),
        NO_EVENTS,
        MAX_SETTLE,
        t0,
    );

    // The Seed is Batching-first: descendants are watched via the *first* Seed probe, which
    // materializes only after the initial settle expiry — not at attach. The first sample's verdict
    // is `Retry` (no prior), but `apply_snapshot` still runs the reconcile / Watch side effects on
    // that first response.
    assert!(
        first_probe_correlation(&attach_out).is_some(),
        "cold-arm Seed: probe emitted at burst construction",
    );
    let (correlation, _) = assert_seed_verifying(&mut e, pid, t0);

    let snap = dir_snap(&[("foo.rs", EntryKind::File, 1), ("bar", EntryKind::Dir, 2)]);
    let resp_out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid,
            correlation,
            outcome: proven(snap),
        }),
        t0 + SETTLE + Duration::from_millis(1),
    );
    let watches = resp_out
        .watch_ops
        .iter()
        .filter(|op| matches!(op, WatchOp::Watch { .. }))
        .count();
    // Files don't get Watch ops; only the Dir descendant contributes. The File still materializes
    // as a Resource (for PerStableFile DedupKey support), no FD.
    assert_eq!(watches, 1, "one Watch for Dir descendant only");
}

#[test]
fn force_fire_emits_effect_with_forced_true() {
    let mut e = Engine::new();
    let r = anchor_dir(&mut e, "src");
    let now = Instant::now();
    let t0 = now;
    let (_sid, pid, attach_out) = attach_returning(
        &mut e,
        "build",
        SubAttachAnchor::Resource(r),
        cfg_recursive(),
        NO_EVENTS,
        MAX_SETTLE,
        t0,
    );

    // Complete the Seed burst with an empty baseline. The cold-arm Seed answers its in-flight
    // Verify probe and pins directly.
    assert!(
        first_probe_correlation(&attach_out).is_some(),
        "cold-arm Seed: probe emitted at burst construction",
    );
    let snap = dir_snap(&[]);
    let _ = seed_to_idle(&mut e, pid, &snap, t0);

    // FsEvent → Standard Settling. Open it after the Seed's settle window so instants stay monotonic.
    let t1 = t0 + SETTLE * 2 + Duration::from_millis(10);
    e.step(
        Input::FsEvent {
            resource: r,
            event: FsEvent::ContentChanged,
        },
        t1,
    );

    // Advance past max_settle so burst_deadline fires.
    let deadline_t = t1 + MAX_SETTLE + Duration::from_millis(1);
    let probe_corr = drain_to_probe_correlation(&mut e, deadline_t);

    if let Some(corr) = probe_corr {
        // Inject a not-stable response — different snapshot.
        let snap = dir_snap(&[("x", EntryKind::File, 99)]);
        let out = e.step(
            Input::ProbeResponse(ProbeResponse {
                owner: pid,
                correlation: corr,
                outcome: proven(snap),
            }),
            deadline_t,
        );
        assert_eq!(out.effects().len(), 1);
        assert!(
            out.effects()[0].forced,
            "force-fired Effect carries forced=true"
        );
    } else {
        panic!("burst_deadline did not produce a probe");
    }
}

#[test]
fn step_output_is_sorted() {
    // Build a multi-Watch scenario (descendants reconciled on first probe) and confirm
    // StepOutput.watch_ops is sorted by ResourceId.
    let mut e = Engine::new();
    let r = anchor_dir(&mut e, "root");
    let t0 = Instant::now();
    let (_sid, pid, attach_out) = attach_returning(
        &mut e,
        "build",
        SubAttachAnchor::Resource(r),
        cfg_recursive(),
        NO_EVENTS,
        MAX_SETTLE,
        t0,
    );

    // The attach output is probe-less (Batching-first Seed) and carries no reconciled descendants.
    // The multi-Watch StepOutput is the *first* Seed probe response (post initial settle expiry),
    // where descendants reconcile — assert sortedness there.
    assert!(
        first_probe_correlation(&attach_out).is_some(),
        "cold-arm Seed: probe emitted at burst construction",
    );
    let (correlation, _) = assert_seed_verifying(&mut e, pid, t0);
    let leaves: Vec<(String, EntryKind, u64)> = (0..5)
        .map(|i| (format!("dir-{i}"), EntryKind::Dir, 100 + i))
        .collect();
    let snap = dir_snap(
        &leaves
            .iter()
            .map(|(s, k, i)| (s.as_str(), *k, *i))
            .collect::<Vec<_>>(),
    );
    let out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid,
            correlation,
            outcome: proven(snap),
        }),
        t0 + SETTLE + Duration::from_millis(1),
    );
    let resources: Vec<ResourceId> = out
        .watch_ops
        .iter()
        .map(|op| match op {
            WatchOp::Watch { resource, .. } => *resource,
            WatchOp::Unwatch { resource } => *resource,
        })
        .collect();
    let mut sorted = resources.clone();
    sorted.sort();
    assert_eq!(resources, sorted, "watch_ops sorted by ResourceId");
}

#[test]
fn cancel_all_in_flight_probes_returns_sealed_output() {
    // Graceful-shutdown probe drain. Two Profiles in Verifying — one probe in flight each —
    // exercise the multi-owner cancel path; the returned `StepOutput` must carry only
    // `ProbeOp::Cancel` ops, in owner order, with `watch_ops` and `effects` empty (the live
    // falsifier of `bin/driver.rs`'s shutdown debug_assert).
    let mut e = Engine::new();
    let r1 = anchor_dir(&mut e, "src");
    let r2 = anchor_dir(&mut e, "dist");
    let t0 = Instant::now();
    let (_, pid1, _) = attach_returning(
        &mut e,
        "build_src",
        SubAttachAnchor::Resource(r1),
        cfg_recursive(),
        NO_EVENTS,
        MAX_SETTLE,
        t0,
    );
    let (_, pid2, _) = attach_returning(
        &mut e,
        "build_dist",
        SubAttachAnchor::Resource(r2),
        cfg_recursive(),
        NO_EVENTS,
        MAX_SETTLE,
        t0,
    );

    // Expire each Profile's own Batching settle so both reach Verifying with a Seed probe in flight.
    let _ = assert_seed_verifying(&mut e, pid1, t0);
    let _ = assert_seed_verifying(&mut e, pid2, t0);

    let out = e.cancel_all_in_flight_probes();

    assert!(out.watch_ops.is_empty(), "cancel_all emits no watch ops");
    assert!(out.effects().is_empty(), "cancel_all emits no effects");
    let owners: Vec<ProfileId> = out
        .probe_ops()
        .iter()
        .map(|op| match op {
            ProbeOp::Cancel { owner } => *owner,
            ProbeOp::Probe { .. } => panic!("cancel_all must emit only Cancel ops"),
        })
        .collect();
    assert_eq!(
        owners,
        vec![pid1, pid2],
        "Cancels emitted in ProfileId order",
    );
}
