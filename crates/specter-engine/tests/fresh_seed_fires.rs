//! Fresh-attach Seed-with-activity fire contract.
//!
//! A fresh-attach **Seed** burst that witnessed filesystem activity (its
//! `PreFireBurst.dirty_resources`, populated by `event_drives_batching`,
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

#![allow(
    clippy::items_after_statements,
    clippy::manual_let_else,
    clippy::match_wildcard_for_single_variants,
    clippy::missing_const_for_fn,
    clippy::needless_pass_by_value,
    clippy::option_if_let_else,
    clippy::redundant_clone,
    clippy::single_match_else,
    clippy::too_many_lines,
    dead_code
)]

use compact_str::CompactString;
use specter_core::testkit::single_exec_program;
use specter_core::{
    ActionProgram, ActiveBurst, ArgPart, ArgTemplate, BurstFinish, BurstIntent, ChildEntry,
    ClassSet, DedupKey, DirChild, DirMeta, DirSnapshot, EffectOutcome, EffectScope, EntryKind,
    FsEvent, FsIdentity, Input, LeafEntry, PostFireBurst, PostFirePhase, PreFireBurst,
    PreFirePhase, ProbeCorrelation, ProbeOp, ProbeOutcome, ProbeOwner, ProbeResponse, ProfileId,
    ProfileState, ProofAuthority, ResourceId, ResourceKind, ResourceRole, ScanConfig, StepOutput,
    SubAttachAnchor, SubAttachRequest, SubId, TimerKind,
};
use specter_engine::Engine;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, Instant, UNIX_EPOCH};

const SETTLE: Duration = Duration::from_millis(100);
const MAX_SETTLE: Duration = Duration::from_secs(6);
const NO_EVENTS: ClassSet = ClassSet::EMPTY;

fn empty_program() -> Arc<ActionProgram> {
    single_exec_program([ArgTemplate::new([ArgPart::literal("/bin/true")])])
}

/// Flat single-component `DirSnapshot`. Mirrors the `dir_snap` helper
/// shared by `fire_cycle.rs` / `multi_profile.rs`.
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

/// Like [`dir_snap`] but the lone file carries an explicit `size`, so
/// two snapshots of the same shape but different sizes hash distinctly
/// (mirrors a growing-in-place file across N=2 reads).
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

fn first_probe_correlation(out: &StepOutput) -> Option<ProbeCorrelation> {
    out.probe_ops().iter().find_map(|op| match op {
        ProbeOp::Probe { request } => Some(request.correlation()),
        ProbeOp::Cancel { .. } => None,
    })
}

fn anchor(e: &mut Engine, name: &str) -> ResourceId {
    let r = e.tree_mut().ensure_root(name, ResourceRole::User);
    e.tree_mut().set_kind(r, ResourceKind::Dir);
    r
}

fn pid_of(e: &Engine, sid: SubId) -> ProfileId {
    e.subs().get(sid).expect("sub exists").profile
}

/// Subtree-root attach request (recursive Sub, `/bin/true`).
fn subtree_request(name: &str, r: ResourceId, events: ClassSet) -> SubAttachRequest {
    SubAttachRequest::for_anchor(
        name.into(),
        SubAttachAnchor::Resource(r),
        ScanConfig::builder().recursive(true).build(),
        MAX_SETTLE,
        SETTLE,
        empty_program(),
        EffectScope::SubtreeRoot,
        events,
        false,
    )
}

/// Read `pid`'s `Active(PreFire(Batching))` settle-timer id, or panic
/// with the actual state. Stepping by id keeps a drive scoped to one
/// Profile (a blanket `pop_expired` would advance siblings).
fn batching_settle_id(e: &Engine, pid: ProfileId) -> specter_core::TimerId {
    match e.profiles().get(pid).unwrap().state() {
        ProfileState::Active(
            ActiveBurst::PreFire(PreFireBurst {
                phase: PreFirePhase::Batching { settle_timer },
                ..
            }),
            _,
        ) => *settle_timer,
        other => panic!("expected {pid:?} in Active(PreFire(Batching)), got {other:?}"),
    }
}

/// `pid`'s pre-fire `burst_deadline` (`BurstDeadline`) timer id, or
/// panic with the actual state. Used to fire the max-settle ceiling
/// deterministically for the §4 forced-terminal setup (test c).
fn burst_deadline_id(e: &Engine, pid: ProfileId) -> specter_core::TimerId {
    match e.profiles().get(pid).unwrap().state() {
        ProfileState::Active(ActiveBurst::PreFire(pre), _) => pre.burst_deadline,
        other => panic!("expected {pid:?} in Active(PreFire(_)), got {other:?}"),
    }
}

/// One Seed N=2 cycle scoped to `pid`: expire its own Batching settle
/// timer (Batching → Verifying, Seed probe emitted) then answer the
/// probe with `snap`. Returns the response `StepOutput`.
fn seed_cycle(e: &mut Engine, pid: ProfileId, snap: &Arc<DirSnapshot>, at: Instant) -> StepOutput {
    let settle_id = batching_settle_id(e, pid);
    e.step(
        Input::TimerExpired {
            profile: pid,
            kind: TimerKind::Settle,
            id: settle_id,
        },
        at,
    );
    let correlation = e
        .pending_probe_for(ProbeOwner::Profile(pid))
        .expect("Seed Verifying probe in flight after settle expiry");
    e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
            correlation,
            outcome: ProbeOutcome::SubtreeProven {
                snapshot: Arc::clone(snap),
                authority: ProofAuthority::Authoritative,
            },
        }),
        at,
    )
}

/// Drive the post-fire rebase loop to its terminal (idempotent
/// command). Mirror of `fire_cycle.rs::complete_rebase_loop`: sample 1
/// (`prior == None`) ⇒ Unstable ⇒ RebaseSettling; spacing-timer expiry
/// ⇒ Rebasing; sample 2 hash-equal ⇒ Stable ⇒ rebase + finish to Idle.
///
/// Returns the terminal (`Stable`-step) `StepOutput`. That step calls
/// `finish_burst_to_idle`, whose Draining sweep reconfirms any covering
/// Profile parked in `Draining` in the *same* step — test (d) reads the
/// parent's reconfirm probe back off this output.
fn complete_rebase_loop(
    e: &mut Engine,
    pid: ProfileId,
    snap: &Arc<DirSnapshot>,
    first_rebase_corr: ProbeCorrelation,
    t0: Instant,
) -> StepOutput {
    e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
            correlation: first_rebase_corr,
            outcome: ProbeOutcome::SubtreeProven {
                snapshot: Arc::clone(snap),
                authority: ProofAuthority::Authoritative,
            },
        }),
        t0,
    );
    let spacing_timer = match e.profiles().get(pid).unwrap().state() {
        ProfileState::Active(
            ActiveBurst::PostFire(PostFireBurst {
                phase: PostFirePhase::RebaseSettling { spacing_timer },
                ..
            }),
            _,
        ) => *spacing_timer,
        other => panic!("rebase sample 1 must loop to RebaseSettling; got {other:?}"),
    };
    let t1 = t0 + SETTLE;
    let rearm_out = e.step(
        Input::TimerExpired {
            profile: pid,
            kind: TimerKind::RebaseSettle,
            id: spacing_timer,
        },
        t1,
    );
    let corr2 =
        first_probe_correlation(&rearm_out).expect("RebaseSettle expiry re-arms Rebasing probe");
    let stable_out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
            correlation: corr2,
            outcome: ProbeOutcome::SubtreeProven {
                snapshot: Arc::clone(snap),
                authority: ProofAuthority::Authoritative,
            },
        }),
        t1,
    );
    assert!(
        matches!(e.profiles().get(pid).unwrap().state(), ProfileState::Idle),
        "idempotent rebase loop closes Stable → Idle",
    );
    stable_out
}

/// Drive a settled, idle Profile through one full Standard fire→rebase
/// cycle (event → settle → N=2 verify → fire → EffectComplete → rebase
/// N=2 → Idle). Asserts exactly one Effect fired. Used post-baseline to
/// prove the Profile now behaves as a Standard burst. `snap_new` differs
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
    // Standard burst is also N=2 here: first verify diffs the fresh
    // response against the seed baseline (Unstable ⇒ re-batch), the
    // second hash-equal sample stabilises and fires.
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
                    outcome: ProbeOutcome::SubtreeProven {
                        snapshot: Arc::clone(snap_new),
                        authority: ProofAuthority::Authoritative,
                    },
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
    let rebase_out = e.step(
        Input::EffectComplete {
            sub: sid,
            key,
            result: EffectOutcome::Ok,
        },
        t + Duration::from_millis(1),
    );
    let rebase_corr =
        first_probe_correlation(&rebase_out).expect("rebase probe emitted on EffectComplete::Ok");
    complete_rebase_loop(e, pid, snap_new, rebase_corr, t + Duration::from_millis(2));
}

/// Shared body for tests (a) and (b): fresh attach (SubtreeRoot,
/// NO_EVENTS), an anchor `FsEvent::Modified` injected during the Seed
/// Batching window (anchor events bypass the class filter, so NO_EVENTS
/// still records the event into `dirty_resources`), then the N=2 proof
/// driven to `Stable` with the two supplied reads. Asserts exactly one
/// Effect fired at the stable verdict, then completes the fire cycle and
/// asserts the Profile established a baseline + fired + now behaves as a
/// Standard burst.
fn fresh_seed_with_activity_fires_one(read1: &Arc<DirSnapshot>, read2: &Arc<DirSnapshot>) {
    let mut e = Engine::new();
    let r = anchor(&mut e, "src");
    let now = Instant::now();
    let out = e.step(Input::AttachSub(subtree_request("test", r, NO_EVENTS)), now);
    let sid = specter_core::testkit::first_attached_sub(&out).expect("attach_sub succeeded");
    let pid = pid_of(&e, sid);
    assert!(
        first_probe_correlation(&out).is_none(),
        "Seed is Batching-first: attach emits no probe",
    );
    assert!(
        !e.profiles().get(pid).unwrap().baseline_is_some(),
        "fresh attach has no baseline yet",
    );

    // Witness real activity: an anchor Modified during Batching. Anchor
    // events bypass the class filter, so the NO_EVENTS mask still routes
    // it through `event_drives_batching`, populating `dirty_resources`.
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
        "anchor event keeps the fresh Seed in PreFire(Batching) (re-armed settle)",
    );
    assert!(
        act_out.effects().is_empty(),
        "the bare event does not itself fire",
    );

    // N=2 proof. read1 (prior None ⇒ Unstable) re-batches; read2
    // hash-equal-to-read1 ⇒ Stable ⇒ the Seed pins. Sequenced strictly
    // past the re-armed settle window so each cycle's settle expiry is
    // clean.
    let t1 = now + Duration::from_millis(1) + SETTLE;
    let _ = seed_cycle(&mut e, pid, read1, t1);
    let t2 = t1 + SETTLE;
    let stable_out = seed_cycle(&mut e, pid, read2, t2);

    // A fresh Seed that witnessed activity fires exactly one Effect at
    // the stable verdict: `dispatch_seed_ok` consults `dirty_resources`
    // and a non-empty witness routes through the Standard fire path.
    assert_eq!(
        stable_out.effects().len(),
        1,
        "fresh Seed that witnessed an FsEvent fires exactly one Effect at the stable verdict",
    );
    let eff = &stable_out.effects()[0];
    assert!(
        matches!(eff.key(), DedupKey::Subtree { sub, .. } if sub == sid),
        "the single Effect is the SubtreeRoot Sub's Subtree effect",
    );

    // Complete the fire cycle and prove the post-fire baseline is now
    // established and the Profile behaves as Standard thereafter.
    let key = eff.key();
    let rebase_out = e.step(
        Input::EffectComplete {
            sub: sid,
            key,
            result: EffectOutcome::Ok,
        },
        t2 + Duration::from_millis(1),
    );
    let rebase_corr = first_probe_correlation(&rebase_out)
        .expect("EffectComplete::Ok drives the burst to Rebasing with a probe");
    complete_rebase_loop(
        &mut e,
        pid,
        read2,
        rebase_corr,
        t2 + Duration::from_millis(2),
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
    let snap_changed = dir_snap(vec![("late.rs", EntryKind::File, 77)]);
    drive_standard_fire_once(&mut e, pid, sid, r, &snap_changed, t2 + SETTLE * 4);
}

/// (a) Core repro: fresh attach + anchor activity, equal N=2 reads.
#[test]
fn fresh_seed_with_activity_fires_exactly_one_effect() {
    let snap = dir_snap(vec![("a.rs", EntryKind::File, 11)]);
    fresh_seed_with_activity_fires_one(&snap, &snap);
}

/// (b) Growing-leaf scp variant: the N=2 reads differ first (smaller
/// file) then stabilise on the larger snapshot, mirroring an in-place
/// growing file. `CertifiedPrior::advance` is `Stable` iff the prior ==
/// the response hash, else `Unstable` (and re-bases the prior). So:
/// read1=S1 (prior None ⇒ Unstable, prior:=S1); read2=S2≠S1 (prior S1 ⇒
/// Unstable, prior:=S2); read3=S2 (prior S2 ⇒ Stable ⇒ pin).
#[test]
fn fresh_seed_with_activity_growing_leaf_fires_one() {
    let mut e = Engine::new();
    let r = anchor(&mut e, "src");
    let now = Instant::now();
    let out = e.step(Input::AttachSub(subtree_request("scp", r, NO_EVENTS)), now);
    let sid = specter_core::testkit::first_attached_sub(&out).expect("attach_sub succeeded");
    let pid = pid_of(&e, sid);

    // Anchor activity during Batching populates `dirty_resources`.
    e.step(
        Input::FsEvent {
            resource: r,
            event: FsEvent::Modified,
        },
        now + Duration::from_millis(1),
    );

    let s1 = dir_snap_sized_file("big.bin", 21, 10);
    let s2 = dir_snap_sized_file("big.bin", 21, 4096);
    assert_ne!(
        s1.dir_hash(),
        s2.dir_hash(),
        "the growing-file reads must hash distinctly so read2 stays Unstable",
    );

    // Three samples: read1=S1 (prior None ⇒ Unstable, prior:=S1),
    // read2=S2 (≠S1 ⇒ Unstable, prior:=S2), read3=S2 (prior S2 ⇒
    // Stable ⇒ pin). The two re-batch samples are still-moving (no
    // fire); the stable one is the pin.
    let mut at = now + Duration::from_millis(1);
    for read in [&s1, &s2] {
        at += SETTLE;
        let out = seed_cycle(&mut e, pid, read, at);
        assert!(
            out.effects().is_empty(),
            "no fire before the tree stabilises (still moving)",
        );
    }
    at += SETTLE;
    let stable_out = seed_cycle(&mut e, pid, &s2, at);

    assert_eq!(
        stable_out.effects().len(),
        1,
        "fresh Seed that witnessed activity fires exactly one Effect once the growing file stabilises",
    );
    assert!(
        matches!(stable_out.effects()[0].key(), DedupKey::Subtree { sub, .. } if sub == sid),
        "the single Effect is the SubtreeRoot Sub's Subtree effect",
    );
}

/// (c) §4 post-ceiling single-event. Drive a fresh Seed to the
/// `Undischarged + forced` ceiling terminal so the Profile ends Idle
/// with NO baseline (no FsEvents, expire the BurstDeadline so
/// `forced=true`, answer the verify with an `Undischarged` authority —
/// `undischarged_consequence` + `forced` ⇒ `finish_burst_to_idle`
/// WITHOUT `apply_snapshot` / `rebase_baseline`). Then inject a *single*
/// `FsEvent` (Idle + `!baseline_is_some()` ⇒ `start_seed_burst`) and
/// drive a clean N=2 to `Stable`. Asserts exactly one Effect — this
/// pins the constructor-symmetry contract specifically: a fresh Seed
/// re-opened by a single event after a forced-ceiling terminal still
/// carries its witnessed activity into the fire path.
#[test]
fn fresh_seed_after_forced_ceiling_single_event_fires_one() {
    let mut e = Engine::new();
    let r = anchor(&mut e, "src");
    let now = Instant::now();
    let out = e.step(Input::AttachSub(subtree_request("ceil", r, NO_EVENTS)), now);
    let sid = specter_core::testkit::first_attached_sub(&out).expect("attach_sub succeeded");
    let pid = pid_of(&e, sid);

    // No FsEvents. Expire the settle window → Verifying (Seed probe in
    // flight, not yet forced).
    let settle_id = batching_settle_id(&e, pid);
    e.step(
        Input::TimerExpired {
            profile: pid,
            kind: TimerKind::Settle,
            id: settle_id,
        },
        now + SETTLE,
    );
    // Expire the BurstDeadline (max-settle ceiling) while the verify is
    // in flight → `force_pending` sets `forced=true`; the phase stays
    // Verifying (a probe is in flight) and the in-flight response will
    // dispatch with `forced` observed.
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
                snapshot: dir_snap(vec![]),
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
    // Seed). The trigger is recorded in `dirty_resources`, so this
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

    // Clean N=2 (no further events) → Stable.
    let snap = dir_snap(vec![("post.rs", EntryKind::File, 31)]);
    let c1 = trigger_at + SETTLE;
    let _ = seed_cycle(&mut e, pid, &snap, c1);
    let c2 = c1 + SETTLE;
    let stable_out = seed_cycle(&mut e, pid, &snap, c2);

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
    let src = e.tree_mut().ensure_root("src", ResourceRole::User);
    e.tree_mut().set_kind(src, ResourceKind::Dir);
    let foo = e
        .tree_mut()
        .ensure_child(src, "foo", ResourceRole::User)
        .expect("test live parent");
    e.tree_mut().set_kind(foo, ResourceKind::Dir);

    let now = Instant::now();
    // The parent's view of /src carries `foo` so the engine covers it as
    // a descendant Profile.
    let parent_snap = dir_snap(vec![("foo", EntryKind::Dir, 7)]);
    let child_snap = dir_snap(vec![]);

    // Parent: recursive @ /src, NO_EVENTS. Covers /src/foo.
    let out_p = e.step(
        Input::AttachSub(subtree_request("parent", src, NO_EVENTS)),
        now,
    );
    let sid_p = specter_core::testkit::first_attached_sub(&out_p).expect("attach_sub succeeded");
    let pid_parent = pid_of(&e, sid_p);

    // Child: recursive @ /src/foo, NO_EVENTS.
    let out_c = e.step(
        Input::AttachSub(subtree_request("child", foo, NO_EVENTS)),
        now,
    );
    let sid_c = specter_core::testkit::first_attached_sub(&out_c).expect("attach_sub succeeded");
    let pid_child = pid_of(&e, sid_c);

    // Drive the child's Seed N=2 to a pinned Idle baseline (scoped by
    // its own settle timer so the parent's Seed is untouched).
    let mut child_at = now;
    for _ in 0..2 {
        child_at += SETTLE;
        let _ = seed_cycle(&mut e, pid_child, &child_snap, child_at);
    }
    assert!(
        matches!(
            e.profiles().get(pid_child).unwrap().state(),
            ProfileState::Idle
        ),
        "child Seed N=2 pinned → Idle",
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
    // Batching), then its N=2 is driven to Stable while the child gates.
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

    // Parent N=2 (prime ⇒ re-batch; confirm ⇒ Stable). The child has no
    // expirable settle timer, so stepping the parent's own settle id
    // does not advance it.
    let p1 = t_parent_act + SETTLE;
    let _ = seed_cycle(&mut e, pid_parent, &parent_snap, p1);
    let p2 = p1 + SETTLE;
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
    // input is valid: `dispatch_seed_ok` consults it for the parent's
    // fresh-with-activity Seed.
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
        // The child Sub is `subtree_request("child", foo, NO_EVENTS)`
        // and its Seed pinned **silently** over an empty tree (a
        // no-activity Seed never fires ⇒ `has_fired == false`). B1
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
        // (stays Batching, no probe). `unstable_response_drives_batching`
        // pins `last_event_time = <response step instant>`. The child
        // prime response is stepped at `p2` exactly (no `+1ms`), so the
        // child re-batches with `last_event_time = p2` and a fresh
        // settle deadline at `p2 + SETTLE`; the child confirm settle is
        // then expired at exactly `p2 + SETTLE` (== `last_event_time +
        // SETTLE`), which satisfies `now − last ≥ settle`.

        // Child Standard sample 1 (prior None ⇒ Unstable ⇒ re-batch).
        let child_prime = e
            .pending_probe_for(ProbeOwner::Profile(pid_child))
            .expect("child Verifying probe in flight (prime sample)");
        let cp = e.step(
            Input::ProbeResponse(ProbeResponse {
                owner: ProbeOwner::Profile(pid_child),
                correlation: child_prime,
                outcome: ProbeOutcome::SubtreeProven {
                    snapshot: child_snap.clone(),
                    authority: ProofAuthority::Authoritative,
                },
            }),
            p2,
        );
        assert!(
            !parent_reconfirmed(&cp),
            "parent does not reconfirm at the child's prime sample",
        );

        // Re-arm the child's settle (instant == last_event_time + SETTLE
        // ⇒ Batching → Verifying, no reschedule) then answer the confirm
        // sample. Hash-equal ⇒ Stable; the child never fired ⇒ `Emit`
        // (NOT B1 `SuppressDedup`) ⇒ the child fires one Subtree Effect
        // and enters the post-fire tail (still gating the parent).
        let child_settle2 = batching_settle_id(&e, pid_child);
        e.step(
            Input::TimerExpired {
                profile: pid_child,
                kind: TimerKind::Settle,
                id: child_settle2,
            },
            p2 + SETTLE,
        );
        let child_confirm = e
            .pending_probe_for(ProbeOwner::Profile(pid_child))
            .expect("child Verifying probe in flight (confirm sample)");
        let child_fire_out = e.step(
            Input::ProbeResponse(ProbeResponse {
                owner: ProbeOwner::Profile(pid_child),
                correlation: child_confirm,
                outcome: ProbeOutcome::SubtreeProven {
                    snapshot: child_snap.clone(),
                    authority: ProofAuthority::Authoritative,
                },
            }),
            p2 + SETTLE,
        );
        assert_eq!(
            child_fire_out.effects().len(),
            1,
            "never-fired child Standard Stable fires one Subtree Effect (Emit, not B1 dedup)",
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
        let child_rebase_out = e.step(
            Input::EffectComplete {
                sub: sid_c,
                key: child_effect_key,
                result: EffectOutcome::Ok,
            },
            p2 + SETTLE + Duration::from_millis(1),
        );
        assert!(
            !parent_reconfirmed(&child_rebase_out),
            "parent does not reconfirm while the child is still Rebasing (gate held)",
        );
        let child_rebase_corr = first_probe_correlation(&child_rebase_out)
            .expect("child EffectComplete::Ok drives Rebasing with a probe");

        // Post-fire N=2 rebase loop → child Idle. The terminal `Stable`
        // step calls `finish_burst_to_idle`, whose Draining sweep
        // re-evaluates the parent's now-false covered-descendant query
        // and transitions the parent Draining → Verifying *in the same
        // step* — the reconfirm probe is read back off this output.
        let child_terminal_out = complete_rebase_loop(
            &mut e,
            pid_child,
            &child_snap,
            child_rebase_corr,
            p2 + SETTLE + Duration::from_millis(2),
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
        // monotonic). With the gate lifted and `dirty_resources` still
        // carrying the witnessed `/src` event, the reconfirm's stable
        // verdict must fire exactly one Effect.
        let reconfirm_out = e.step(
            Input::ProbeResponse(ProbeResponse {
                owner: ProbeOwner::Profile(pid_parent),
                correlation: parent_reconfirm_corr,
                outcome: ProbeOutcome::SubtreeProven {
                    snapshot: parent_snap.clone(),
                    authority: ProofAuthority::Authoritative,
                },
            }),
            p2 + SETTLE * 3,
        );

        // The reconfirm may itself be N=2 (fresh `CertifiedPrior` on
        // the reconfirm carrier ⇒ first sample Unstable ⇒ re-batch).
        // Drive to the firing stable verdict. The `pop_expired` drain
        // advances `t` by `SETTLE * 4` per iteration — always well past
        // `last_event_time + SETTLE` (set by
        // `unstable_response_drives_batching` to the prior response
        // instant), so each re-armed settle expiry transitions to
        // Verifying rather than rescheduling.
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
                            outcome: ProbeOutcome::SubtreeProven {
                                snapshot: parent_snap.clone(),
                                authority: ProofAuthority::Authoritative,
                            },
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
