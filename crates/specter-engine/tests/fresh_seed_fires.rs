//! Fresh-attach Seed-with-activity fire contract.
//!
//! A fresh-attach **Seed** burst that witnessed filesystem activity (its
//! `PreFireBurst.dirty` provenance, populated by `event_drives_batching`,
//! is non-empty) fires its Sub's command on the stable verdict, routing
//! through the same Standard stable consequence — including the Draining
//! gate when a covered Standard descendant is mid-burst. For a
//! `SubtreeRoot` Sub it fires **exactly one** Effect, establishes a
//! baseline, then behaves as a Standard burst thereafter.
//!
//! A fresh Seed that witnessed **no** activity pins its baseline
//! silently (restart-safe: Specter persists no baseline, so a daemon
//! restart over a static tree must not re-fire); a recovery Seed
//! drift-fires.

use compact_str::CompactString;
use specter_core::testkit::{dir_snap, proven};
use specter_core::{
    ActiveBurst, BurstFinish, BurstIntent, ChildEntry, ClassSet, DedupKey, DirMeta, DirSnapshot,
    EntryKind, FsEvent, FsIdentity, Input, LeafEntry, PreFireBurst, PreFirePhase, ProbeCorrelation,
    ProbeOp, ProbeOutcome, ProbeOwner, ProbeResponse, ProfileId, ProfileState, ProofAuthority,
    ResourceId, ResourceKind, ResourceRole, ScanConfig, StepOutput, SubAttachAnchor, SubId,
    TimerKind,
};
use specter_engine::Engine;
use specter_engine::testkit::{
    anchor_dir, attach_returning, attach_structure_only, batching_settle_id,
    complete_effect_to_settling, first_probe_correlation, rebase_post_fire_to_idle, seed_to_idle,
};
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, Instant, UNIX_EPOCH};

const SETTLE: Duration = Duration::from_millis(100);
const MAX_SETTLE: Duration = Duration::from_secs(6);
/// Production-realistic `EffectScope::SubtreeRoot` events mask — CONTENT in
/// the mask sets `events_witness_quiescence == true`, so a single
/// Authoritative sample closes the verdict floor's hash-equality channel.
const DEFAULT_EVENTS: ClassSet = ClassSet::DEFAULT_SUBTREE_ROOT;

/// `pid`'s pre-fire `burst_deadline` (`BurstDeadline`) timer id, or
/// panic with the actual state. Used to fire the max-settle ceiling
/// deterministically for the forced-terminal setup (test c).
fn burst_deadline_id(e: &Engine, pid: ProfileId) -> specter_core::TimerId {
    match e.profiles().get(pid).unwrap().state() {
        ProfileState::Active(ActiveBurst::PreFire(pre), _) => pre.burst_deadline,
        other => panic!("expected {pid:?} in Active(PreFire(_)), got {other:?}"),
    }
}

/// Single-file directory snapshot with an explicit `size` so two
/// snapshots over the same name + inode hash distinctly when the leaf
/// grows in place — the canonical scp / in-place-write shape the N=2
/// hash channel exists to catch. `dir_snap` bakes `size = 0` and offers
/// no override, so this distinct sized fixture stays file-local; the
/// growing-leaf test is the sole consumer here.
fn dir_snap_sized_file(name: &str, inode: u64, size: u64) -> Arc<DirSnapshot> {
    let mut map: BTreeMap<CompactString, ChildEntry> = BTreeMap::new();
    map.insert(
        CompactString::new(name),
        ChildEntry::Leaf(LeafEntry::synthetic(
            EntryKind::File,
            size,
            UNIX_EPOCH,
            FsIdentity::synthetic(inode, 0),
        )),
    );
    Arc::new(DirSnapshot::new(
        DirMeta::synthetic(UNIX_EPOCH, FsIdentity::synthetic(0, 0)),
        0,
        map,
    ))
}

/// Deliver one Seed probe response for `pid`: if the burst is in
/// Batching, expire its own settle timer to advance Batching →
/// Verifying (emits the Seed probe); if already in cold-arm
/// Verifying, deliver the response directly. Returns the response
/// `StepOutput`.
fn seed_cycle(e: &mut Engine, pid: ProfileId, snap: &Arc<DirSnapshot>, at: Instant) -> StepOutput {
    // Cold-arm Verifying-first: the first Seed sample is delivered
    // directly to the construct-armed slot — no Batching to expire.
    // A Batching re-entry (e.g. after an Undischarged !terminal retry)
    // needs the settle-timer expiry to advance back to Verifying.
    if !matches!(
        e.profiles().get(pid).unwrap().state(),
        ProfileState::Active(
            ActiveBurst::PreFire(PreFireBurst {
                phase: PreFirePhase::Verifying(_),
                ..
            }),
            _,
        )
    ) {
        let settle_id = batching_settle_id(e, pid);
        e.step(
            Input::TimerExpired {
                profile: pid,
                kind: TimerKind::Settle,
                id: settle_id,
            },
            at,
        );
    }
    let correlation = e
        .pending_probe_for(ProbeOwner::Profile(pid))
        .expect("Seed Verifying probe in flight");
    e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
            correlation,
            outcome: proven(Arc::clone(snap)),
        }),
        at,
    )
}

/// Drive a settled, idle Profile through one full Standard fire→rebase
/// cycle (event → settle → verify → fire → EffectComplete → rebase →
/// Idle). Asserts exactly one Effect fired. Used post-baseline to prove
/// the Profile now behaves as a Standard burst. `snap_new` differs
/// from the established baseline so the Standard verdict is a genuine
/// fire (not B1-dedup-suppressed).
fn drive_standard_fire_once(
    e: &mut Engine,
    pid: ProfileId,
    sid: SubId,
    r: ResourceId,
    snap_new: &Arc<DirSnapshot>,
    t0: Instant,
) {
    e.step(
        Input::FsEvent {
            resource: r,
            event: FsEvent::Modified,
        },
        t0,
    );
    // The verify response against the seed baseline fires directly on
    // its first Authoritative sample. The drain loop is defensive in
    // case the first response routes through an Undischarged retry.
    let mut t = t0;
    let mut stable_out: Option<StepOutput> = None;
    for _ in 0..8 {
        t += SETTLE * 4;
        let mut probe_corr: Option<ProbeCorrelation> = None;
        while let Some(entry) = e.pop_expired(t) {
            let s = e.step(
                Input::TimerExpired {
                    profile: entry.profile,
                    kind: entry.kind,
                    id: entry.id,
                },
                t,
            );
            if let Some(c) = first_probe_correlation(&s) {
                probe_corr = Some(c);
            }
        }
        if let Some(c) = probe_corr {
            let out = e.step(
                Input::ProbeResponse(ProbeResponse {
                    owner: ProbeOwner::Profile(pid),
                    correlation: c,
                    outcome: proven(Arc::clone(snap_new)),
                }),
                t,
            );
            if !out.effects().is_empty() {
                stable_out = Some(out);
                break;
            }
        }
    }
    let stable_out = stable_out.expect("Standard burst stabilised and fired");
    assert_eq!(
        stable_out.effects().len(),
        1,
        "post-baseline Standard burst fires exactly one Effect (now behaves as Standard)",
    );
    let key = stable_out.effects()[0].key();
    let _co = complete_effect_to_settling(e, sid, key, t + Duration::from_millis(1));
    let _r = rebase_post_fire_to_idle(e, pid, snap_new, t + Duration::from_millis(2));
    assert!(
        matches!(e.profiles().get(pid).unwrap().state(), ProfileState::Idle),
        "idempotent rebase loop closes Stable → Idle",
    );
}

/// The cold-Seed bypass — Layer-B's cold-attach latency win preserved
/// on the worst-case mask. A `STRUCTURE`-only Profile fails
/// `events_witness_quiescence`, so any *fire-bearing* burst on it owes
/// the hash channel. But a cold Seed (no events, never fired) does NOT
/// owe a quiescence proof — `burst_owes_quiescence_proof` projects to
/// `false` on `(BurstIntent::Seed, dirty.is_empty(), !any_fired)`, and
/// `certify_probe_response` skips the hash channel entirely (witness =
/// `EventsReliable`). The single Authoritative response folds to
/// `Stable(Natural)` and the burst finishes in one sample, exactly as on
/// an events-reliable Profile.
///
/// Pins the regression guard for the cold-attach latency win: a
/// structure-only Profile must not accidentally engage the hash channel
/// for cold-Seed quiet (no extra settle window, no carrier loop).
/// [`seed_to_idle`] asserts internally that one Authoritative sample
/// pins to Idle.
#[test]
fn cold_seed_on_structure_only_pins_in_one_sample_without_loop() {
    let mut e = Engine::new();
    let r = anchor_dir(&mut e, "src");
    let now = Instant::now();
    let (_sid, pid) = attach_structure_only(&mut e, r, now);
    assert!(
        !e.profiles().get(pid).unwrap().events_witness_quiescence(),
        "fixture sanity: structure-only Profile fails events_witness_quiescence \
         (so the hash channel would engage for fire-bearing bursts)",
    );

    let snap = dir_snap(&[]);
    let done = seed_to_idle(&mut e, pid, &snap, now);
    assert_eq!(
        done,
        now + SETTLE,
        "cold-Seed bypass: one Authoritative sample (no retry loop)",
    );
    assert!(
        e.profiles().get(pid).unwrap().baseline_is_some(),
        "cold-Seed bypass establishes the baseline on the first sample",
    );
}

#[test]
fn fresh_seed_with_activity_fires_exactly_one_effect() {
    let snap = dir_snap(&[("a.rs", EntryKind::File, 11)]);
    let mut e = Engine::new();
    let r = anchor_dir(&mut e, "src");
    let now = Instant::now();
    let (sid, pid, out) = attach_returning(
        &mut e,
        "test",
        SubAttachAnchor::Resource(r),
        ScanConfig::builder().recursive(true).build(),
        DEFAULT_EVENTS,
        MAX_SETTLE,
        now,
    );
    assert!(
        first_probe_correlation(&out).is_some(),
        "cold-arm Seed: attach emits the cold walk probe at burst construction",
    );
    assert!(
        !e.profiles().get(pid).unwrap().baseline_is_some(),
        "fresh attach has no baseline yet",
    );

    // Witness real activity: an anchor Modified during the cold-arm
    // Verifying phase. Anchor events bypass the class filter, so the
    // NO_EVENTS mask still routes it through `event_drives_batching`,
    // which Cancels the cold-arm verify slot and reschedules Batching
    // with the trigger in `dirty`.
    let act_out = e.step(
        Input::FsEvent {
            resource: r,
            event: FsEvent::Modified,
        },
        now + Duration::from_millis(1),
    );
    assert!(
        matches!(
            e.profiles().get(pid).unwrap().state(),
            ProfileState::Active(
                ActiveBurst::PreFire(PreFireBurst {
                    phase: PreFirePhase::Batching { .. },
                    intent: BurstIntent::Seed,
                    ..
                }),
                _
            )
        ),
        "anchor event during cold-arm Verifying re-enters PreFire(Batching) (Cancel + re-armed settle)",
    );
    assert!(
        act_out.effects().is_empty(),
        "the bare event does not itself fire",
    );

    // The first Authoritative response fires: `seed_owes_first_fire`
    // reads `!dirty.is_empty()`, routing the fresh-with-activity Seed
    // through `FreshSeedFire`.
    let t1 = now + Duration::from_millis(1) + SETTLE;
    let stable_out = seed_cycle(&mut e, pid, &snap, t1);

    assert_eq!(
        stable_out.effects().len(),
        1,
        "fresh Seed that witnessed an FsEvent fires exactly one Effect on the Authoritative verdict",
    );
    let eff = &stable_out.effects()[0];
    assert!(
        matches!(eff.key(), DedupKey::Subtree { sub, .. } if sub == sid),
        "the single Effect is the SubtreeRoot Sub's Subtree effect",
    );

    // Complete the fire cycle and prove the post-fire baseline is now
    // established and the Profile behaves as Standard thereafter.
    let key = eff.key();
    let _co = complete_effect_to_settling(&mut e, sid, key, t1 + Duration::from_millis(1));
    let _r = rebase_post_fire_to_idle(&mut e, pid, &snap, t1 + Duration::from_millis(2));
    assert!(
        matches!(e.profiles().get(pid).unwrap().state(), ProfileState::Idle),
        "idempotent rebase loop closes Stable → Idle",
    );

    assert!(
        e.profiles().get(pid).unwrap().baseline_is_some(),
        "post-fire rebase establishes a settled baseline",
    );
    assert!(
        e.subs().any_fired(pid),
        "the Sub recorded its fire (no longer a fresh Profile)",
    );

    // A subsequent FsEvent + settle now drives a *Standard* fire (a
    // second Effect), proving the Profile is no longer fresh.
    let snap_changed = dir_snap(&[("late.rs", EntryKind::File, 77)]);
    drive_standard_fire_once(&mut e, pid, sid, r, &snap_changed, t1 + SETTLE * 4);
}

/// Growing-leaf scp variant of the fresh-with-activity contract on an
/// events-incomplete (`STRUCTURE`-only) Profile: the same fresh-Seed
/// witnessed-activity path, except in-place writes between samples make
/// the first two reads hash distinctly. The verdict floor's hash channel
/// holds the fire until the leaf stabilises, then fires exactly one
/// Effect — the user-reported scp regression scenario, reduced to its
/// Seed-with-activity shape.
///
/// `attach_structure_only` gives a `STRUCTURE`-only mask: in-place
/// `CONTENT` writes are invisible (no per-file FDs wired, and the
/// per-Profile class filter drops descendant CONTENT events even if
/// they arrived), so settle-window silence is **not** a quiescence
/// witness — the burst owes the hash-equality channel for fire-bearing
/// consequences. Anchor events bypass the class filter, so a single
/// anchor `FsEvent::Modified` Cancels the cold-arm verify slot and
/// re-enters Batching with `dirty` non-empty — a triggered (not cold)
/// Seed whose `burst_owes_quiescence_proof` is `true`.
///
/// The three settle-spaced samples: read1=S1 (prior None ⇒ Unstable,
/// carrier := hash(S1)), read2=S2 ≠ S1 (prior hash(S1) ⇒ Unstable,
/// carrier := hash(S2)), read3=S2 (prior hash(S2) ⇒ Stable ⇒ fire).
#[test]
fn fresh_seed_with_activity_growing_leaf_fires_one() {
    let mut e = Engine::new();
    let r = anchor_dir(&mut e, "src");
    let now = Instant::now();
    let (sid, pid) = attach_structure_only(&mut e, r, now);

    // Anchor event during the cold-arm Verifying phase: bypasses the
    // class filter, Cancels the cold-arm verify slot, re-enters Batching
    // with `dirty` non-empty (triggered Seed — owes a quiescence proof).
    e.step(
        Input::FsEvent {
            resource: r,
            event: FsEvent::Modified,
        },
        now + Duration::from_millis(1),
    );
    assert!(
        matches!(
            e.profiles().get(pid).unwrap().state(),
            ProfileState::Active(
                ActiveBurst::PreFire(PreFireBurst {
                    phase: PreFirePhase::Batching { .. },
                    intent: BurstIntent::Seed,
                    ..
                }),
                _
            )
        ),
        "anchor event during cold-arm Verifying re-enters PreFire(Batching) as a triggered Seed",
    );

    let s1 = dir_snap_sized_file("big.bin", 21, 10);
    let s2 = dir_snap_sized_file("big.bin", 21, 4096);
    assert_ne!(
        s1.dir_hash(),
        s2.dir_hash(),
        "the growing-leaf samples must hash distinctly so the carrier observes disagreement",
    );

    // Two re-batch samples (still-moving — no fire); the third is the
    // stable verdict. Each iteration advances time by SETTLE: the prior
    // step's `retry_drives_batching` armed a fresh `Settle` deadline at
    // `prior + SETTLE`, so the next `seed_cycle` expires it cleanly.
    let mut at = now + Duration::from_millis(1);
    for read in [&s1, &s2] {
        at += SETTLE;
        let out = seed_cycle(&mut e, pid, read, at);
        assert!(
            out.effects().is_empty(),
            "no fire before the carrier observes two consecutive equal samples (still moving)",
        );
    }
    at += SETTLE;
    let stable_out = seed_cycle(&mut e, pid, &s2, at);

    assert_eq!(
        stable_out.effects().len(),
        1,
        "the third sample (carrier prior == response) fires exactly one Effect at the stable verdict",
    );
    assert!(
        matches!(stable_out.effects()[0].key(), DedupKey::Subtree { sub, .. } if sub == sid),
        "the single Effect is the SubtreeRoot Sub's Subtree effect",
    );
}

/// Drive a fresh Seed to the
/// `Undischarged + forced` ceiling terminal so the Profile ends Idle
/// `Undischarged + forced` ceiling terminal so the Profile ends Idle
/// with NO baseline (no FsEvents, expire the BurstDeadline so
/// `forced=true`, answer the verify with an `Undischarged` authority —
/// `undischarged_consequence` + `forced` ⇒ `finish_burst_to_idle`
/// WITHOUT `apply_snapshot` / `rebase_baseline`). Then inject a *single*
/// `FsEvent` (Idle + `!baseline_is_some()` ⇒ `start_seed_burst`) and
/// drive a clean Seed proof to `Stable`. Asserts exactly one Effect —
/// this pins the constructor-symmetry contract specifically: a fresh
/// Seed re-opened by a single event after a forced-ceiling terminal
/// still carries its witnessed activity into the fire path.
#[test]
fn fresh_seed_after_forced_ceiling_single_event_fires_one() {
    let mut e = Engine::new();
    let r = anchor_dir(&mut e, "src");
    let now = Instant::now();
    let (sid, pid, _) = attach_returning(
        &mut e,
        "ceil",
        SubAttachAnchor::Resource(r),
        ScanConfig::builder().recursive(true).build(),
        DEFAULT_EVENTS,
        MAX_SETTLE,
        now,
    );

    // Cold-arm: the Seed verify probe is in flight at burst
    // construction (no settle window to expire). Expire the
    // BurstDeadline (max-settle ceiling) while the verify is in flight
    // → `force_pending` sets `forced=true`; the phase stays Verifying
    // (a probe is in flight) and the in-flight response will dispatch
    // with `forced` observed.
    let bd_id = burst_deadline_id(&e, pid);
    e.step(
        Input::TimerExpired {
            profile: pid,
            kind: TimerKind::BurstDeadline,
            id: bd_id,
        },
        now + MAX_SETTLE + Duration::from_millis(1),
    );
    let corr = e
        .pending_probe_for(ProbeOwner::Profile(pid))
        .expect("forced Seed verify probe still in flight");
    // Undischarged authority: a non-observation lies on the obligation
    // chain. `undischarged_consequence` + forced ⇒ ceiling terminal:
    // finish to Idle, NO apply_snapshot, NO rebase_baseline.
    let unread: Arc<std::path::Path> = Arc::from(std::path::Path::new("/src/unreadable"));
    let ceil_out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
            correlation: corr,
            outcome: ProbeOutcome::SubtreeProven {
                snapshot: dir_snap(&[]),
                authority: ProofAuthority::Undischarged {
                    first_unread: Arc::clone(&unread),
                },
            },
        }),
        now + MAX_SETTLE + Duration::from_millis(2),
    );
    assert!(
        ceil_out.effects().is_empty(),
        "the forced ceiling terminal fires nothing",
    );
    assert!(
        matches!(e.profiles().get(pid).unwrap().state(), ProfileState::Idle),
        "Undischarged + forced is a terminal — the burst finishes to Idle",
    );
    assert!(
        !e.profiles().get(pid).unwrap().baseline_is_some(),
        "the forced ceiling terminal never establishes a baseline (no rebase_baseline)",
    );

    // A single FsEvent: Idle + !baseline ⇒ `start_seed_burst` (a fresh
    // Seed). The trigger is recorded in `dirty`, so this
    // fresh-with-activity Seed fires one Effect at its stable verdict.
    let trigger_at = now + MAX_SETTLE + Duration::from_millis(3);
    let trig_out = e.step(
        Input::FsEvent {
            resource: r,
            event: FsEvent::Modified,
        },
        trigger_at,
    );
    assert!(
        trig_out.effects().is_empty(),
        "the bare trigger event does not itself fire",
    );
    assert!(
        matches!(
            e.profiles().get(pid).unwrap().state(),
            ProfileState::Active(
                ActiveBurst::PreFire(PreFireBurst {
                    phase: PreFirePhase::Batching { .. },
                    intent: BurstIntent::Seed,
                    ..
                }),
                _
            )
        ),
        "the single event re-opened a fresh Seed burst (Idle + !baseline → start_seed_burst)",
    );

    // settle expiry → Verifying → Authoritative response → fire.
    // Single sample.
    let snap = dir_snap(&[("post.rs", EntryKind::File, 31)]);
    let stable_out = seed_cycle(&mut e, pid, &snap, trigger_at + SETTLE);

    assert_eq!(
        stable_out.effects().len(),
        1,
        "fresh Seed re-opened by a single post-ceiling event fires exactly one Effect",
    );
    assert!(
        matches!(stable_out.effects()[0].key(), DedupKey::Subtree { sub, .. } if sub == sid),
        "the single Effect is the SubtreeRoot Sub's Subtree effect",
    );
}

/// (d) Draining gate on a fresh-with-activity Seed. Parent Dir Profile
/// covering a child Dir Profile. Child reaches a settled baseline, then
/// enters an Active **Standard** burst (gating the parent). The parent's
/// fresh-with-activity Seed is driven to `Stable` while the child is
/// mid-Standard-burst: the parent must enter `Draining` and emit **no**
/// Effect yet. Finishing the child's burst then driving the parent's
/// reconfirm to `Stable` makes the parent fire **exactly one** Effect.
#[test]
fn fresh_seed_with_activity_gated_by_draining_then_fires_one() {
    let mut e = Engine::new();
    let src = anchor_dir(&mut e, "src");
    let foo = e
        .tree_mut()
        .ensure_child(src, "foo", ResourceRole::User)
        .expect("test live parent");
    e.tree_mut().set_kind(foo, ResourceKind::Dir);

    let now = Instant::now();
    // The parent's view of /src carries `foo` so the engine covers it as
    // a descendant Profile.
    let parent_snap = dir_snap(&[("foo", EntryKind::Dir, 7)]);
    let child_snap = dir_snap(&[]);

    // Parent: recursive @ /src, DEFAULT_EVENTS. Covers /src/foo.
    let (sid_p, pid_parent, _) = attach_returning(
        &mut e,
        "parent",
        SubAttachAnchor::Resource(src),
        ScanConfig::builder().recursive(true).build(),
        DEFAULT_EVENTS,
        MAX_SETTLE,
        now,
    );

    // Child: recursive @ /src/foo, DEFAULT_EVENTS.
    let (sid_c, pid_child, _) = attach_returning(
        &mut e,
        "child",
        SubAttachAnchor::Resource(foo),
        ScanConfig::builder().recursive(true).build(),
        DEFAULT_EVENTS,
        MAX_SETTLE,
        now,
    );

    // Drive the child's Seed to a pinned Idle baseline (a single
    // Authoritative sample pins → SilentPin → Idle).
    let child_at = now + SETTLE;
    let _ = seed_cycle(&mut e, pid_child, &child_snap, child_at);
    assert!(
        matches!(
            e.profiles().get(pid_child).unwrap().state(),
            ProfileState::Idle
        ),
        "child Seed pinned → Idle",
    );
    assert!(
        e.profiles().get(pid_child).unwrap().baseline_is_some(),
        "child has a settled baseline",
    );

    // Child enters an Active Standard burst (FsEvent at its anchor),
    // then advance it to Verifying so it parks (no expirable settle
    // timer) and keeps gating the parent.
    let t_child_burst = child_at + Duration::from_millis(5);
    e.step(
        Input::FsEvent {
            resource: foo,
            event: FsEvent::Modified,
        },
        t_child_burst,
    );
    let child_settle = batching_settle_id(&e, pid_child);
    e.step(
        Input::TimerExpired {
            profile: pid_child,
            kind: TimerKind::Settle,
            id: child_settle,
        },
        t_child_burst + SETTLE,
    );
    assert!(
        e.profiles()
            .get(pid_child)
            .unwrap()
            .state()
            .in_active_standard_burst(),
        "child is mid-Standard-burst and gates the parent",
    );

    // The parent witnesses activity (anchor Modified during its Seed
    // Batching), then its verify is driven to Stable while the child
    // gates.
    let t_parent_act = t_child_burst + SETTLE + Duration::from_millis(1);
    e.step(
        Input::FsEvent {
            resource: src,
            event: FsEvent::Modified,
        },
        t_parent_act,
    );
    assert!(
        matches!(
            e.profiles().get(pid_parent).unwrap().state(),
            ProfileState::Active(
                ActiveBurst::PreFire(PreFireBurst {
                    phase: PreFirePhase::Batching { .. },
                    intent: BurstIntent::Seed,
                    ..
                }),
                _
            )
        ),
        "parent witnessed activity and is a fresh Seed in Batching",
    );

    // The parent's single Authoritative response folds to fire-or-park
    // via the Draining gate. The child has no expirable settle timer,
    // so stepping the parent's own settle id does not advance it.
    let p2 = t_parent_act + SETTLE;
    let parent_stable_out = seed_cycle(&mut e, pid_parent, &parent_snap, p2);

    // The parent's Stable Seed step itself emits NO Effect: the fire is
    // withheld by the Draining gate because a covered child is
    // mid-Standard-burst. Zero Effects at this step.
    assert!(
        parent_stable_out.effects().is_empty(),
        "parent emits no Effect at the Stable step while a covered child gates it",
    );

    // The child is genuinely a covered, Active-Standard descendant of
    // the parent at this instant — the exact topology
    // `has_active_standard_descendant` detects. This confirms the gate
    // input is valid: `gated_fire` consults it for the parent's
    // fresh-with-activity Seed (`FreshSeedFire`).
    assert!(
        e.profiles()
            .get(pid_child)
            .unwrap()
            .state()
            .in_active_standard_burst(),
        "child is still mid-Standard-burst when the parent stabilises (gate input is valid)",
    );

    // A fresh Seed that witnessed activity routes through the same
    // fire-gate as Standard, so a covered descendant mid-Standard-burst
    // parks the parent in `Draining` (fire withheld, not silently
    // pinned); `parent_entered_draining` is therefore TRUE here and the
    // guarded reconfirm→fire path below runs. Recorded as a non-fatal
    // check (not a hard `assert!`) so the canonical single-Effect
    // assertion at the end — identical in shape to tests (a)/(c) — is
    // the terminal signal rather than a state-machine panic here.
    let parent_entered_draining = matches!(
        e.profiles().get(pid_parent).unwrap().state(),
        ProfileState::Active(
            ActiveBurst::PreFire(PreFireBurst {
                phase: PreFirePhase::Draining,
                ..
            }),
            BurstFinish::ReturnToIdle
        )
    );

    // Finish the child's Standard burst, then drive the parent's
    // Draining → Verifying reconfirm to a firing stable verdict. Guarded
    // on `parent_entered_draining` so the block is structurally tied to
    // the gate state it asserts (a parent already Idle would skip it and
    // the test would fail cleanly at the unconditional single-Effect
    // assertion below rather than panicking on a missing parent probe).
    let mut parent_fire_count = 0usize;
    let mut parent_fire_key: Option<DedupKey> = None;
    if parent_entered_draining {
        let parent_reconfirmed = |out: &StepOutput| {
            out.probe_ops().iter().any(|op| {
                matches!(op, ProbeOp::Probe { request }
                    if request.owner() == ProbeOwner::Profile(pid_parent))
            })
        };

        // ── Drive the child's Standard burst to a genuine Idle. ──
        //
        // The child Sub (recursive @ /src/foo, DEFAULT_EVENTS) had its Seed
        // pinned **silently** over an empty tree (a no-activity Seed
        // never fires ⇒ `has_fired == false`). B1
        // `SuppressDedup` requires `!forced && nothing_changed &&
        // already_fired` (`fire_decision`); the child's never-fired, so
        // even though its Standard confirm is hash-equal to its
        // baseline the verdict is `Emit`, NOT `SuppressDedup`. The
        // child therefore FIRES one Subtree Effect on its `Stable`,
        // enters the post-fire tail, and keeps gating the parent
        // (`has_active_standard_descendant` counts post-fire phases too)
        // until its rebase loop closes to Idle. We must run the full
        // fire → EffectComplete → rebase → Idle cycle, not assume a
        // straight finish-to-Idle.
        //
        // Timing rule (`on_settle_expired`): a step that expires a
        // re-armed Batching settle timer must use an instant `≥
        // last_event_time + SETTLE`, else the handler reschedules
        // (stays Batching, no probe). `retry_drives_batching` pins
        // `last_event_time = <response step instant>`. The child first
        // response is stepped at `p2` exactly (no `+1ms`), so the child
        // re-batches with `last_event_time = p2` and a fresh settle
        // deadline at `p2 + SETTLE`; the child's next-sample settle is
        // then expired at exactly `p2 + SETTLE` (== `last_event_time +
        // SETTLE`), which satisfies `now − last ≥ settle`.

        // A single Authoritative verify response fires the child's
        // Effect. The child never fired ⇒ Emit (not B1 dedup).
        let child_corr = e
            .pending_probe_for(ProbeOwner::Profile(pid_child))
            .expect("child Verifying probe in flight");
        let child_fire_out = e.step(
            Input::ProbeResponse(ProbeResponse {
                owner: ProbeOwner::Profile(pid_child),
                correlation: child_corr,
                outcome: proven(child_snap.clone()),
            }),
            p2,
        );
        assert_eq!(
            child_fire_out.effects().len(),
            1,
            "never-fired child Standard Authoritative fires one Subtree Effect (Emit, not B1 dedup)",
        );
        assert!(
            !parent_reconfirmed(&child_fire_out),
            "parent does not reconfirm while the child is still post-fire (gate held)",
        );
        let child_effect_key = child_fire_out.effects()[0].key();
        assert!(
            matches!(child_effect_key, DedupKey::Subtree { sub, .. } if sub == sid_c),
            "the child's Effect is its own SubtreeRoot Subtree effect",
        );

        // Child EffectComplete::Ok → Rebasing (idempotent `/bin/true`).
        let child_rebase_out = complete_effect_to_settling(
            &mut e,
            sid_c,
            child_effect_key,
            p2 + SETTLE + Duration::from_millis(1),
        );
        assert!(
            !parent_reconfirmed(&child_rebase_out),
            "parent does not reconfirm while the child is still Rebasing (gate held)",
        );

        // Post-fire rebase → child Idle. The terminal `Stable` step
        // calls `finish_burst_to_idle`, whose Draining sweep
        // re-evaluates the parent's now-false covered-descendant query
        // and transitions the parent Draining → Verifying *in the same
        // step* — the reconfirm probe is read back off this output.
        let child_terminal_out = rebase_post_fire_to_idle(
            &mut e,
            pid_child,
            &child_snap,
            p2 + SETTLE + Duration::from_millis(2),
        )
        .finish;
        assert!(
            matches!(
                e.profiles().get(pid_child).unwrap().state(),
                ProfileState::Idle
            ),
            "idempotent rebase loop closes Stable → Idle",
        );
        assert!(
            matches!(
                e.profiles().get(pid_child).unwrap().state(),
                ProfileState::Idle
            ),
            "child Standard burst completed its full fire → rebase cycle and reached Idle",
        );
        assert!(
            !e.profiles()
                .get(pid_child)
                .unwrap()
                .state()
                .in_active_standard_burst(),
            "child no longer gates the parent (Draining gate input is now false)",
        );

        // The child's `finish_burst_to_idle` Draining sweep reconfirms
        // the parent (Draining → Verifying) in that same terminal step.
        // If a topology nuance defers it, drive the parent's re-armed
        // settle to reach the reconfirm Verifying.
        let parent_reconfirm_corr = if parent_reconfirmed(&child_terminal_out) {
            first_probe_correlation(&child_terminal_out).expect("parent reconfirm probe in flight")
        } else {
            // The parent re-armed a Batching settle on a witnessed
            // event; expire it well past `last_event_time + SETTLE`
            // (a wide margin keeps `now − last ≥ settle` regardless of
            // which witnessed-event instant pinned `last_event_time`).
            let p_settle = batching_settle_id(&e, pid_parent);
            let rc = e.step(
                Input::TimerExpired {
                    profile: pid_parent,
                    kind: TimerKind::Settle,
                    id: p_settle,
                },
                p2 + SETTLE * 4,
            );
            first_probe_correlation(&rc)
                .expect("parent reconfirm probe after Draining→Verifying via settle")
        };

        // The child reached Idle at `(p2 + SETTLE + 2ms) + SETTLE`
        // (the rebase loop's terminal step); the parent's reconfirm
        // probe was emitted there. Answer it at `p2 + SETTLE * 3`
        // (strictly later than that instant, so the timeline stays
        // monotonic). With the gate lifted and `dirty` still
        // carrying the witnessed `/src` event, the reconfirm's stable
        // verdict must fire exactly one Effect.
        let reconfirm_out = e.step(
            Input::ProbeResponse(ProbeResponse {
                owner: ProbeOwner::Profile(pid_parent),
                correlation: parent_reconfirm_corr,
                outcome: proven(parent_snap.clone()),
            }),
            p2 + SETTLE * 3,
        );

        // Defensive: if the reconfirm did not fire on its first
        // Authoritative response (e.g. a transient re-Draining or an
        // Undischarged retry), drive successive settle windows until
        // an Effect is emitted. The `pop_expired` drain advances `t`
        // by `SETTLE * 4` per iteration, always well past
        // `last_event_time + SETTLE`, so each re-armed settle expiry
        // transitions to Verifying rather than rescheduling.
        let mut fire_out = if reconfirm_out.effects().is_empty() {
            None
        } else {
            Some(reconfirm_out)
        };
        if fire_out.is_none() {
            let mut t = p2 + SETTLE * 3;
            for _ in 0..6 {
                t += SETTLE * 4;
                let mut probe_corr: Option<ProbeCorrelation> = None;
                while let Some(entry) = e.pop_expired(t) {
                    let s = e.step(
                        Input::TimerExpired {
                            profile: entry.profile,
                            kind: entry.kind,
                            id: entry.id,
                        },
                        t,
                    );
                    if let Some(c) = first_probe_correlation(&s)
                        && matches!(
                            e.profiles()
                                .get(pid_parent)
                                .map(specter_core::Profile::state),
                            Some(ProfileState::Active(ActiveBurst::PreFire(_), _))
                        )
                    {
                        probe_corr = Some(c);
                    }
                }
                if let Some(c) = probe_corr {
                    let out = e.step(
                        Input::ProbeResponse(ProbeResponse {
                            owner: ProbeOwner::Profile(pid_parent),
                            correlation: c,
                            outcome: proven(parent_snap.clone()),
                        }),
                        t,
                    );
                    if !out.effects().is_empty() {
                        fire_out = Some(out);
                        break;
                    }
                }
            }
        }
        if let Some(out) = fire_out {
            parent_fire_count = out.effects().len();
            parent_fire_key = out.effects().first().map(specter_core::Effect::key);
        }
    }

    // Canonical contract assertion (unconditional). Across the whole
    // scenario the parent fires exactly one Effect: a fresh Seed that
    // witnessed activity, gated by a covered Standard-burst child, fires
    // once after the gate lifts.
    assert_eq!(
        parent_fire_count, 1,
        "parent fires exactly one Effect once the Draining gate lifts \
         (fresh-with-activity Seed)",
    );
    assert!(
        matches!(parent_fire_key, Some(DedupKey::Subtree { sub, .. }) if sub == sid_p),
        "the single Effect is the parent SubtreeRoot Sub's Subtree effect",
    );
}
