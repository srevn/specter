//! Per-input dispatch tests. Each `(state, input)` cell of the transition table gets a focused
//! test. Goes hand-in-hand with the integration suite at `tests/integration.rs` which covers
//! full-lifecycle flows.

// Tests prioritize readability over the workspace's pedantic style budget.
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
    clippy::unnecessary_wraps
)]

use crate::Engine;
use compact_str::CompactString;
use specter_core::program::SpawnBody;
use specter_core::testkit::single_exec_program;
use specter_core::{
    ActionProgram, ActiveBurst, AnchorClaim, ArgPart, ArgTemplate, BurstFinish, BurstIntent,
    CeilingState, ChildEntry, ClaimKind, ClassSet, DedupKey, Diagnostic, DirChild, DirMeta,
    DirSnapshot, DirtyProvenance, EffectCompletion, EffectOutcome, EffectScope, EntryKind,
    FS_ROOT_SEGMENT, FsEvent, FsIdentity, Input, LeafEntry, OverflowScope, Placeholder,
    PostFireBurst, PostFirePhase, PreFireBurst, PreFirePhase, ProbeFailure, ProbeOp, ProbeOutcome,
    ProbeRequest, ProbeResponse, ProbeSlot, ProfileIdentity, ProfileState, ProofAuthority,
    ProofObligation, QuiescenceVerdict, ResourceId, ResourceKind, ResourceRole, ScanConfig,
    StableReason, StepOutput, SubAttachAnchor, SubAttachRequest, SubId, SubParams, Termination,
    TimerKind, TreeSnapshot, WatchOp,
};
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const SETTLE: Duration = Duration::from_millis(100);
const MAX_SETTLE: Duration = Duration::from_secs(6);
/// Default events mask for the shared `engine_with_attached_sub` fixture — `STRUCTURE | CONTENT`
/// matches production `EffectScope::SubtreeRoot` Profiles. CONTENT in the mask sets
/// `events_witness_quiescence == true`, so a single Authoritative sample closes the N=2 hash-channel
/// obligation (witness = `EventsReliable`). It also sets `has_per_file_fds = true`, so the reconciler
/// emits per-file FD `WatchOp`s for File children. Tests that drive the N=2 hash channel directly or
/// that pin "no per-file FDs" opt into `ClassSet::EMPTY` (or a CONTENT-free mask) inline.
const DEFAULT_EVENTS: ClassSet = ClassSet::DEFAULT_SUBTREE_ROOT;
const NO_EVENTS: ClassSet = ClassSet::EMPTY;

fn empty_program() -> Arc<ActionProgram> {
    single_exec_program([ArgTemplate::new([ArgPart::literal("/bin/true")])])
}

fn diff_program() -> Arc<ActionProgram> {
    single_exec_program([ArgTemplate::new([
        ArgPart::literal("fmt"),
        ArgPart::Placeholder(Placeholder::Created),
    ])])
}

/// Engine + Sub attached at `/anchor` (Dir, recursive). Returns the engine, `ProfileId`, `SubId`.
fn engine_with_attached_sub() -> (
    Engine,
    specter_core::ProfileId,
    specter_core::SubId,
    ResourceId,
    Instant,
) {
    let mut e = Engine::new();
    let r = e.tree.ensure_root("anchor", ResourceRole::User);
    e.tree.set_kind(r, ResourceKind::Dir);
    let now = Instant::now();
    let req = SubAttachRequest {
        anchor: SubAttachAnchor::Resource(r),
        identity: ProfileIdentity::new(
            ScanConfig::builder().recursive(true).build(),
            MAX_SETTLE,
            DEFAULT_EVENTS,
        ),
        params: SubParams::spawn(
            "test-sub".into(),
            empty_program(),
            EffectScope::SubtreeRoot,
            SETTLE,
            false,
        ),
    };
    let out = e.step(Input::AttachSub(req), now);
    let sid = specter_core::testkit::first_attached_sub(&out).expect("attach_sub succeeded");
    let pid = e.subs.get(sid).unwrap().profile();
    (e, pid, sid, r, now)
}

/// V5-native test helper: build a `TreeSnapshot::Dir` with the supplied single-component children.
/// Each child is `(name, EntryKind, inode)`; Dirs are emitted as `DirChild::Uncovered(_)` (the walker
/// stored the entry but did not recurse). Tests that need nested subtrees should use
/// `dir_with_subtree`. Returns `Arc<DirSnapshot>` directly — the typed `ProbeOutcome::SubtreeProven`
/// variant carries an `Arc<DirSnapshot>`, not a wrapping `TreeSnapshot`.
fn dir_tree_snap(children: Vec<(&str, EntryKind, u64)>) -> Arc<DirSnapshot> {
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

/// `LeafEntry` for File-anchored Profiles. Consumed directly by `ProbeOutcome::AnchorOk`; the
/// wrapping `TreeSnapshot::File` lives on the engine-internal `Profile.current`, not the wire
/// response.
fn file_tree_snap(kind: EntryKind, size: u64, mtime: SystemTime, inode: u64) -> LeafEntry {
    LeafEntry::synthetic(kind, size, mtime, FsIdentity::synthetic(inode, 0))
}

/// Drive a fresh-attach cold-arm Seed burst from `Active(PreFire(Verifying))` through its
/// quiescence verdict to pinned `Idle`, pinning against `snap`. After this, `Profile.current` and
/// `Profile.baseline` are set to `snap`.
///
/// The cold-arm Seed burst pins on the first `Authoritative` response: a cold-Seed `SilentPin`
/// consequence does not owe quiescence proof, so the witness is
/// [`QuiescenceWitness::EventsReliable`] and the fold folds to `Stable(StableReason::Natural)`;
/// dispatch reaches `SilentPin` (no fired Subs, no drift) and finishes to Idle. The cold-arm
/// Verifying-first contract puts the probe in flight at burst construction, so the helper answers
/// it directly — no Batching settle expiry, no second sample.
///
/// Returns the pinning response's [`StepOutput`] so callers can assert the Seed-completion emission
/// (a fresh Seed never fires, so it is effect-empty).
fn complete_seed_burst_with(
    e: &mut Engine,
    pid: specter_core::ProfileId,
    snap: Arc<DirSnapshot>,
) -> StepOutput {
    let at = Instant::now();
    let correlation = e
        .pending_probe_for(pid)
        .expect("cold-Seed Verifying probe in flight from start_seed_burst(None)");
    let last = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid,
            correlation,
            outcome: ProbeOutcome::SubtreeProven {
                snapshot: Arc::clone(&snap),
                authority: ProofAuthority::Authoritative,
            },
        }),
        at,
    );
    assert!(
        matches!(e.profiles.get(pid).unwrap().state(), ProfileState::Idle),
        "one Authoritative Seed sample pins the baseline → Idle",
    );
    last
}

/// [`complete_seed_burst_with`] against an empty baseline — the common case for attach→Idle setups
/// that don't pin specific children.
fn complete_seed_burst(e: &mut Engine, pid: specter_core::ProfileId) -> StepOutput {
    complete_seed_burst_with(e, pid, dir_tree_snap(vec![]))
}

/// Assert every Seed Profile is in `Active(PreFire(Verifying))` with a probe in flight at burst
/// construction — the cold-arm contract.
///
/// Pure state projection over the whole `ProfileMap`: `start_seed_burst` puts every cold-arm Seed
/// in `Verifying` at construction with its probe in flight, so this helper does not advance time.
/// For the full Seed pin use [`complete_seed_burst`].
fn assert_seed_verifying(e: &Engine) {
    let pids: Vec<_> = e.profiles.iter().map(|(pid, _)| pid).collect();
    for pid in pids {
        let Some(p) = e.profiles.get(pid) else {
            continue;
        };
        let ProfileState::Active(ActiveBurst::PreFire(pre), _) = p.state() else {
            continue;
        };
        if pre.intent == BurstIntent::Seed {
            assert!(
                matches!(pre.phase, PreFirePhase::Verifying { .. }),
                "cold-arm Seed: expected Verifying at burst construction, got {:?}",
                pre.phase,
            );
        }
    }
}

// ---- attach_sub ----

#[test]
fn attach_sub_fresh_profile_emits_watch_probe() {
    let (mut e, _pid, _sid, r, _now) = engine_with_attached_sub();
    // After attach: anchor watch_demand=1, Profile is Active(Seed Verifying).
    assert_eq!(e.tree.get(r).unwrap().watch_demand(), 1);
    let _ = e.cancel_all_in_flight_probes();
}

/// `Profile.kind` is the cached witness of the anchor's classification: `transition_to_verifying`'s
/// probe-target dispatch and `emit_effects`'s `compute_cwd` dispatch read this rather than
/// re-deriving the kind from the Tree on every call. A resource-based attach against a
/// kind-classified slot must populate the field at the `attach_sub_inner` post-`Profile::new` write.
#[test]
fn attach_sub_caches_anchor_kind_for_classified_resource() {
    let (mut e, pid, _sid, _r, _now) = engine_with_attached_sub();
    assert_eq!(
        e.profiles.get(pid).and_then(specter_core::Profile::kind),
        Some(ResourceKind::Dir),
        "resource-based attach reads the classified anchor's kind into Profile.kind",
    );
    let _ = e.cancel_all_in_flight_probes();
}

/// Resource-based attach against an `Unknown` slot leaves `Profile.kind = None` until the first probe
/// response classifies the anchor. The `dispatch_quiescence_ok` fallback writes the field from the
/// response shape — the rare unprobed-attach path's only signal of the anchor's classification.
#[test]
fn attach_sub_unprobed_anchor_seeds_kind_on_first_response() {
    let mut e = Engine::new();
    // Resource exists but kind is left Unknown — the rare path where a caller passes a resource-based
    // attach against a freshly-`ensure`'d slot whose kind hasn't been classified by any prior probe.
    let r = e.tree.ensure_root("anchor", ResourceRole::User);
    let now = Instant::now();
    let req = SubAttachRequest {
        anchor: SubAttachAnchor::Resource(r),
        identity: ProfileIdentity::new(
            ScanConfig::builder().recursive(true).build(),
            MAX_SETTLE,
            NO_EVENTS,
        ),
        params: SubParams::spawn(
            "test-sub".into(),
            empty_program(),
            EffectScope::SubtreeRoot,
            SETTLE,
            false,
        ),
    };
    let attach_out = e.step(Input::AttachSub(req), now);
    let sid = specter_core::testkit::first_attached_sub(&attach_out).expect("attach_sub succeeded");
    let pid = e.subs.get(sid).unwrap().profile();

    assert_eq!(
        e.profiles.get(pid).and_then(specter_core::Profile::kind),
        None,
        "unprobed anchor → Profile.kind starts as None",
    );

    // The Seed burst is Batching-first; expire the settle window so it advances to Verifying and
    // emits its first probe.
    assert_seed_verifying(&e);

    // Drive the first Seed verify with a Dir-shaped response. The kind-classification fallback in
    // `dispatch_burst_outcome` caches the anchor kind from the response shape on the *first*
    // response, before the first-sample `Retry` verdict re-batches.
    let correlation = e
        .pending_probe_for(pid)
        .expect("Seed verify probe in flight after settle expiry");
    let snap = dir_tree_snap(vec![]);
    let _ = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid,
            correlation,
            outcome: ProbeOutcome::SubtreeProven {
                snapshot: snap,
                authority: ProofAuthority::Authoritative,
            },
        }),
        now,
    );
    assert_eq!(
        e.profiles.get(pid).and_then(specter_core::Profile::kind),
        Some(ResourceKind::Dir),
        "Seed-Ok fallback caches the anchor kind from the response shape",
    );
}

/// `dispatch_burst_outcome` is the unified fan-out for both Seed and Standard intents, so the
/// kind-classification fallback fires from every burst arm — not just Seed. Companion to
/// `attach_sub_unprobed_anchor_seeds_kind_on_first_response`: that test pins the Seed-Ok /
/// SubtreeProven path; this one pins it explicitly through the same outcome shape and asserts the
/// Profile reaches its first classification before any subsequent dispatcher work runs.
#[test]
fn dispatch_burst_outcome_classifies_kind_on_first_seed_subtree() {
    let mut e = Engine::new();
    let r = e.tree.ensure_root("anchor", ResourceRole::User);
    // Leave the Resource Unknown — anchor_kind from `Resource::kind()` collapses Unknown to None,
    // so Profile.kind starts as None.
    let req = SubAttachRequest {
        anchor: SubAttachAnchor::Resource(r),
        identity: ProfileIdentity::new(
            ScanConfig::builder().recursive(true).build(),
            MAX_SETTLE,
            NO_EVENTS,
        ),
        params: SubParams::spawn(
            "test-sub".into(),
            empty_program(),
            EffectScope::SubtreeRoot,
            SETTLE,
            false,
        ),
    };
    let now = Instant::now();
    let attach_out = e.step(Input::AttachSub(req), now);
    let sid = specter_core::testkit::first_attached_sub(&attach_out).expect("attach_sub succeeded");
    let pid = e.subs.get(sid).unwrap().profile();
    assert_eq!(
        e.profiles.get(pid).and_then(specter_core::Profile::kind),
        None,
        "unprobed anchor → Profile.kind starts as None",
    );

    assert_seed_verifying(&e);
    let correlation = e
        .pending_probe_for(pid)
        .expect("Seed verify probe in flight after settle expiry");
    let snap = dir_tree_snap(vec![]);
    let _ = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid,
            correlation,
            outcome: ProbeOutcome::SubtreeProven {
                snapshot: snap,
                authority: ProofAuthority::Authoritative,
            },
        }),
        now,
    );
    assert_eq!(
        e.profiles.get(pid).and_then(specter_core::Profile::kind),
        Some(ResourceKind::Dir),
        "SubtreeProven on a kind-None Profile classifies as Dir at dispatch_burst_outcome",
    );
}

/// Mirror of the SubtreeProven test for the AnchorOk arm: an `AnchorOk(leaf)` response on a Profile
/// whose `kind` was None classifies the anchor as `File`. The walker's response variant is the
/// canonical witness, so the fallback cannot be specialised to one shape.
#[test]
fn dispatch_burst_outcome_classifies_kind_on_first_seed_anchor() {
    let mut e = Engine::new();
    let r = e.tree.ensure_root("anchor", ResourceRole::User);
    // Resource is Unknown ⇒ Profile.kind starts as None. The Seed burst emits `ProbeRequest::Subtree`
    // per the unified fallback (Subtree is the safe default for unclassified anchors). The walker,
    // finding a regular file at the path, replies with `Vanished` in production (kind mismatch). For
    // this test we synthesise an `AnchorOk(leaf)` response — a deliberate deviation that exercises
    // the dispatch_burst_outcome classification path for AnchorOk; the walker never produces this
    // response shape against a Subtree request, but the engine's classification logic must still fall
    // out correctly if it ever does (defense-in-depth + symmetry with the SubtreeProven arm).
    let req = SubAttachRequest {
        anchor: SubAttachAnchor::Resource(r),
        identity: ProfileIdentity::new(
            ScanConfig::builder().recursive(true).build(),
            MAX_SETTLE,
            NO_EVENTS,
        ),
        params: SubParams::spawn(
            "test-sub".into(),
            empty_program(),
            EffectScope::SubtreeRoot,
            SETTLE,
            false,
        ),
    };
    let now = Instant::now();
    let attach_out = e.step(Input::AttachSub(req), now);
    let sid = specter_core::testkit::first_attached_sub(&attach_out).expect("attach_sub succeeded");
    let pid = e.subs.get(sid).unwrap().profile();
    assert_eq!(
        e.profiles.get(pid).and_then(specter_core::Profile::kind),
        None,
        "unprobed anchor → Profile.kind starts as None",
    );

    assert_seed_verifying(&e);
    let correlation = e
        .pending_probe_for(pid)
        .expect("Seed verify probe in flight after settle expiry");
    let leaf = file_tree_snap(EntryKind::File, 0, UNIX_EPOCH, 1);
    let _ = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid,
            correlation,
            outcome: ProbeOutcome::AnchorOk(leaf),
        }),
        now,
    );
    assert_eq!(
        e.profiles.get(pid).and_then(specter_core::Profile::kind),
        Some(ResourceKind::File),
        "AnchorOk on a kind-None Profile classifies as File at dispatch_burst_outcome",
    );
}

/// Walker contract: a `Pending` Profile (descent state) probes a Dir prefix with
/// `ProbeRequest::Descent`; the only valid responses are `SegmentObserved`, `Vanished`, or
/// `Failed`. An `AnchorOk` in this slot is a walker-side bug — descent never queries an anchor's
/// `lstat` shape. `DescentOutcome::try_from` rejects it at the demux seam and the Descent arm
/// routes to `walker_contract_violated_descent`, which fires a `debug_assert!` in dev/CI and, in
/// release, emits `WalkerContractViolated` and abandons the descent prefix (no re-probe loop
/// against a buggy walker). The test pins the dev/CI panic.
///
/// Disabled in release builds via the standard `cfg_attr` discipline, mirroring
/// `mint_probe_correlation_panics_on_double_open`.
#[test]
#[cfg_attr(
    not(debug_assertions),
    ignore = "debug_assert! is compiled out in release"
)]
#[should_panic(expected = "walker contract violated")]
fn dispatch_descent_with_anchor_outcome_is_walker_contract_violation() {
    let mut e = Engine::new();
    let foo = e
        .tree
        .ensure_path(&[FS_ROOT_SEGMENT, "foo"], ResourceRole::User)
        .expect("non-empty fixture");
    e.tree.set_kind(foo, ResourceKind::Dir);
    let req = SubAttachRequest::for_anchor(
        "guard".into(),
        SubAttachAnchor::Path(std::path::PathBuf::from("/foo/bar")),
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
    let pid = e.subs.get(sid).unwrap().profile();
    assert!(
        matches!(
            e.profiles.get(pid).map(specter_core::Profile::state),
            Some(ProfileState::Pending(_)),
        ),
        "path-based attach against an absent leaf goes Pending",
    );
    let correlation = e
        .pending_probe_for(pid)
        .expect("descent probe in flight at the prefix");

    // `AnchorOk` from a Descent probe is structurally impossible from the production walker —
    // `probe_descent` answers a structural query and can only yield `SegmentObserved` / `Vanished`
    // / `Failed`. We synthesise the breach to exercise the walker-contract debug_assert in
    // `walker_contract_violated_descent`.
    let leaf = file_tree_snap(EntryKind::File, 0, UNIX_EPOCH, 1);
    let _ = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid,
            correlation,
            outcome: ProbeOutcome::AnchorOk(leaf),
        }),
        now,
    );
}

/// Walker contract: a Verifying / Rebasing probe response carries a quiescence observation
/// (`AnchorOk` / `SubtreeProven` / `Vanished` / `Failed`). A `SegmentObserved` outcome — the
/// descent-route shape — is a walker-side bug: the request was a `Subtree` quiescence read, not a
/// structural segment query. `ProofOutcome::try_from` rejects it at the demux seam and the
/// Verifying arm routes to `walker_contract_violated_burst`, which fires a `debug_assert!` in
/// dev/CI and, in release, emits `WalkerContractViolated` and finishes the burst to Idle
/// (anchor/baseline preserved). The test pins the dev/CI panic.
///
/// `profile_probe_gate` does not filter on outcome variant (it routes on owner state +
/// correlation), so this payload-shape violation is the kind the public test surface can
/// synthesise. The owner-split removed every cross-owner defensive arm at the type level (no
/// `Enumerating` on the Profile side), so the only surviving runtime guards are these
/// `WalkerContractViolation` payload-shape parses.
///
/// Disabled in release builds via the standard `cfg_attr` discipline, mirroring
/// `dispatch_descent_with_anchor_outcome_is_walker_contract_violation`.
#[test]
#[cfg_attr(
    not(debug_assertions),
    ignore = "debug_assert! is compiled out in release"
)]
#[should_panic(expected = "walker contract violated")]
fn certify_segment_observed_outcome_is_walker_contract_violation() {
    let (mut e, pid, _sid, _r, now) = engine_with_attached_sub();
    // Cold-arm Seed → Verifying probe in flight at burst construction.
    assert_seed_verifying(&e);
    let correlation = e
        .pending_probe_for(pid)
        .expect("cold-Seed Verifying probe in flight from start_seed_burst");
    // `SegmentObserved` from a quiescence probe is structurally impossible from the production walker
    // — the Subtree request answers a content proof, not a structural segment query. We synthesise
    // the breach to exercise the walker-contract debug_assert in `walker_contract_violated_burst`.
    let _ = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid,
            correlation,
            outcome: ProbeOutcome::SegmentObserved { kind: None },
        }),
        now,
    );
}

/// `certify_probe_response`'s inline kind guard: a `Profile.kind = Some(File)` receiving a
/// Dir-shaped response is a structurally unreachable kind divergence (the typed `ProbeRequest`
/// chain emits `AnchorFile` for File-kinded Profiles, and the walker collapses any Dir↔File swap to
/// `Vanished` by construction). The guard catches the case at the verdict floor and routes through
/// [`Engine::finalize_anchor_lost`] rather than misroute the Dir snapshot onto a File-kinded
/// Profile (which would leak watch contributions and break the cross-field invariant).
#[test]
#[cfg_attr(
    not(debug_assertions),
    ignore = "debug_assert! is compiled out in release"
)]
#[should_panic(expected = "walker contract violated")]
fn kind_mismatched_response_routes_through_finalize_anchor_lost_debug() {
    // Set up a File-kinded Profile in Active(Verifying) and inject a SubtreeProven (Dir) response.
    let mut e = Engine::new();
    let r = e.tree.ensure_root("anchor", ResourceRole::User);
    e.tree.set_kind(r, ResourceKind::File);
    let req = SubAttachRequest {
        anchor: SubAttachAnchor::Resource(r),
        identity: ProfileIdentity::new(
            ScanConfig::builder().recursive(true).build(),
            MAX_SETTLE,
            NO_EVENTS,
        ),
        params: SubParams::spawn(
            "test-sub".into(),
            empty_program(),
            EffectScope::SubtreeRoot,
            SETTLE,
            false,
        ),
    };
    let now = Instant::now();
    let attach_out = e.step(Input::AttachSub(req), now);
    let sid = specter_core::testkit::first_attached_sub(&attach_out).expect("attach_sub succeeded");
    let pid = e.subs.get(sid).unwrap().profile();
    // Cold-arm Seed Verifying-first: a single Authoritative sample pins → SilentPin → Idle.
    let mut at = now + SETTLE;
    let correlation = e
        .pending_probe_for(pid)
        .expect("cold-arm Seed Verify probe in flight at burst construction");
    e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid,
            correlation,
            outcome: ProbeOutcome::AnchorOk(file_tree_snap(EntryKind::File, 0, UNIX_EPOCH, 1)),
        }),
        at,
    );
    assert_eq!(
        e.profiles.get(pid).and_then(specter_core::Profile::kind),
        Some(ResourceKind::File),
    );

    // Drive a Standard burst (FsEvent at the anchor) and let the settle timer fire so a Verifying
    // probe is in flight.
    at += SETTLE;
    let _ = e.step(
        Input::FsEvent {
            resource: r,
            event: FsEvent::MetadataChanged,
        },
        at,
    );
    let settle_at = at + SETTLE;
    while let Some(entry) = e.pop_expired(settle_at) {
        e.step(
            Input::TimerExpired {
                id: entry.id,
                kind: entry.kind,
                profile: entry.profile,
            },
            settle_at,
        );
    }
    let correlation = e
        .pending_probe_for(pid)
        .expect("Standard Verifying probe in flight");

    // Inject the kind-mismatched response: a SubtreeProven (Dir) for a File-kinded Profile. The
    // boundary check fires the debug_assert.
    let dir = dir_tree_snap(vec![]);
    let _ = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid,
            correlation,
            outcome: ProbeOutcome::SubtreeProven {
                snapshot: dir,
                authority: ProofAuthority::Authoritative,
            },
        }),
        settle_at,
    );
}

#[test]
fn attach_sub_existing_profile_bumps_refcount() {
    let (mut e, pid, _sid, r, now) = engine_with_attached_sub();
    let pre_state = matches!(
        e.profiles.get(pid).unwrap().state(),
        ProfileState::Active(_, _)
    );
    assert!(pre_state, "first attach went Active");
    let pre_count = e.subs.at(pid).len();

    // Second attach with the same config_hash — must match the fixture's events mask, since
    // `events` folds into `config_hash`.
    let req = SubAttachRequest {
        anchor: SubAttachAnchor::Resource(r),
        identity: ProfileIdentity::new(
            ScanConfig::builder().recursive(true).build(),
            MAX_SETTLE,
            DEFAULT_EVENTS,
        ),
        params: SubParams::spawn(
            "second".into(),
            empty_program(),
            EffectScope::SubtreeRoot,
            SETTLE,
            false,
        ),
    };
    let out = e.step(Input::AttachSub(req), now);
    let sid2 = specter_core::testkit::first_attached_sub(&out).expect("attach_sub succeeded");
    assert_eq!(e.subs.at(pid).len(), pre_count + 1);
    assert_eq!(e.subs.get(sid2).unwrap().profile(), pid, "shared Profile");
    // No fresh watch/probe emitted: existing Profile already has them.
    assert!(out.watch_ops.is_empty());
    assert!(out.probe_ops().is_empty());
    let _ = e.cancel_all_in_flight_probes();
}

// ---- ProbeResponse dispatch ----

/// Smoke test: a `TreeSnapshot::Dir(...)` with one Leaf entry lands as a Seed-Ok on the Profile (no
/// Effect, baseline set). Pins the dispatch wiring; the rest of the engine test suite is the broad
/// coverage.
#[test]
fn engine_dispatch_through_shim_matches_v4_behaviour() {
    let (mut e, pid, _sid, _root, _now) = engine_with_attached_sub();
    // Drive the Seed burst to a pinned Idle baseline.
    let out = complete_seed_burst(&mut e, pid);
    let p = e.profiles.get(pid).unwrap();
    assert!(matches!(p.state(), ProfileState::Idle));
    assert!(p.current().is_some(), "Seed-Ok sets current");
    assert!(out.effects().is_empty(), "Seed bursts never fire Effects");
}

#[test]
fn probe_response_seed_ok_sets_baseline_and_idles_no_effect() {
    let (mut e, pid, _sid, _root, _now) = engine_with_attached_sub();
    let out = complete_seed_burst(&mut e, pid);

    let p = e.profiles.get(pid).unwrap();
    assert!(matches!(p.state(), ProfileState::Idle));
    assert!(p.baseline().is_some());
    assert!(p.current().is_some());
    assert!(out.effects().is_empty(), "Seed bursts never fire Effects");
}

#[test]
fn probe_response_seed_vanished_clears_baseline_and_diagnoses() {
    let (mut e, pid, _sid, _r, now) = engine_with_attached_sub();
    // Batching-first: drive to the first Seed Verify probe, then answer it Vanished — a terminal
    // outcome regardless of verdict.
    assert_seed_verifying(&e);
    let correlation = e
        .pending_probe_for(pid)
        .expect("Seed Verifying probe in flight after settle expiry");
    let out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid,
            correlation,
            outcome: ProbeOutcome::Vanished,
        }),
        now + SETTLE,
    );
    let p = e.profiles.get(pid).unwrap();
    // Root anchor — no recovery parent, so the observed-loss wrapper's fallback parks.
    assert!(matches!(p.state(), ProfileState::Parked));
    assert!(p.baseline().is_none());
    assert!(p.current().is_none());
    let has_diag = out.diagnostics.iter().any(|d| {
        matches!(
            d,
            Diagnostic::ProbeVanished {
                intent: BurstIntent::Seed,
                ..
            }
        )
    });
    assert!(has_diag);
}

#[test]
fn probe_response_seed_failed_clears_baseline_and_diagnoses() {
    let (mut e, pid, _sid, _r, now) = engine_with_attached_sub();
    // Batching-first: drive to the first Seed Verify probe, then answer it Failed — a terminal
    // outcome regardless of verdict.
    assert_seed_verifying(&e);
    let correlation = e
        .pending_probe_for(pid)
        .expect("Seed Verifying probe in flight after settle expiry");
    let out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid,
            correlation,
            outcome: ProbeOutcome::Failed(ProbeFailure::Anchor { errno: 13 }),
        }),
        now + SETTLE,
    );
    let has_diag = out.diagnostics.iter().any(|d| {
        matches!(
            d,
            Diagnostic::ProbeFailed {
                intent: BurstIntent::Seed,
                failure: ProbeFailure::Anchor { errno: 13 },
                ..
            },
        )
    });
    assert!(has_diag);
}

#[test]
fn probe_response_correlation_mismatch_drops_with_diagnostic() {
    let (mut e, pid, _sid, _root, _now) = engine_with_attached_sub();
    // Inject a response with the wrong correlation.
    let bogus = specter_core::ProbeCorrelation::from(99_999);
    let snap = dir_tree_snap(vec![]);
    let out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid,
            correlation: bogus,
            outcome: ProbeOutcome::SubtreeProven {
                snapshot: snap,
                authority: ProofAuthority::Authoritative,
            },
        }),
        Instant::now(),
    );
    let stale = out
        .diagnostics
        .iter()
        .any(|d| matches!(d, Diagnostic::StaleProbeResponse { .. }));
    assert!(stale);
    // State unchanged: still Active(Seed Verifying).
    assert!(matches!(
        e.profiles.get(pid).unwrap().state(),
        ProfileState::Active(_, _),
    ));
    let _ = e.cancel_all_in_flight_probes();
}

#[test]
fn probe_response_for_idle_profile_drops_with_diagnostic() {
    let (mut e, pid, _sid, _root, _now) = engine_with_attached_sub();
    complete_seed_burst(&mut e, pid);
    // Profile is Idle; injecting a ProbeResponse drops with diagnostic.
    let snap = dir_tree_snap(vec![]);
    let out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid,
            correlation: specter_core::ProbeCorrelation::from(1),
            outcome: ProbeOutcome::SubtreeProven {
                snapshot: snap,
                authority: ProofAuthority::Authoritative,
            },
        }),
        Instant::now(),
    );
    let stale = out
        .diagnostics
        .iter()
        .any(|d| matches!(d, Diagnostic::StaleProbeResponse { .. }));
    assert!(stale);
}

// I5-breach panic/diagnostic tests deleted: the forged "probe armed in non-mint phase" shape they
// exercised is structurally unrepresentable once probe correlation lives on a state-resident
// `ProbeSlot` — a slot can only be armed via its owning phase's typed transition, so the (state,
// phase)-mismatch arm cannot be reached without forging an invalid state. Structural property tests
// for `ProbeSlot` live in `specter-core`'s `probe.rs` `#[cfg(test)] mod tests`.

// ---- Standard burst dispatch ----

#[test]
fn standard_burst_stable_emits_effect_and_awaits() {
    // Stable verdict emits the Effect and transitions to `PostFirePhase::Awaiting`; the engine
    // waits for the completion before returning to Idle. Idle means "nothing in flight" —
    // outstanding Effects keep the burst Active until they report back.
    let (mut e, pid, _sid, root, _now) = engine_with_attached_sub();
    complete_seed_burst(&mut e, pid);
    let now = Instant::now();
    // FsEvent at anchor → Standard Settling.
    e.step(
        Input::FsEvent {
            resource: root,
            event: FsEvent::ContentChanged,
        },
        now,
    );
    // Settle fires.
    while let Some(entry) = e.pop_expired(now + SETTLE) {
        e.step(
            Input::TimerExpired {
                profile: entry.profile,
                kind: entry.kind,
                id: entry.id,
            },
            now + SETTLE,
        );
    }
    // The verify response folds through `quiescence_verdict` to `Stable(StableReason::Natural)` on
    // the first sample — single dispatch fires the Effect.
    let correlation = e.pending_probe_for(pid).expect("Verifying probe in flight");
    let out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid,
            correlation,
            outcome: ProbeOutcome::SubtreeProven {
                snapshot: dir_tree_snap(vec![]),
                authority: ProofAuthority::Authoritative,
            },
        }),
        now + SETTLE + Duration::from_millis(1),
    );
    let burst = match e.profiles.get(pid).unwrap().state() {
        ProfileState::Active(ActiveBurst::PostFire(post), _) => post,
        _ => panic!("expected Active(Awaiting) after firing"),
    };
    assert_eq!(burst.intent, BurstIntent::Standard);
    assert!(
        matches!(burst.phase, PostFirePhase::Awaiting { outstanding: 1, .. }),
        "stable verdict transitions to Awaiting with one outstanding Effect; got {:?}",
        burst.phase,
    );
    assert_eq!(
        out.effects().len(),
        1,
        "one Effect emitted at stable verdict"
    );
    let eff = &out.effects()[0];
    assert!(!eff.forced);
    // Engine carries the lowered ActionProgram; the actuator resolves argv at spawn time. Assert on
    // the template's literal-only first arg instead of the resolved argv. (`/bin/true` is the
    // test's stub command — see `empty_program()`.)
    let SpawnBody::Exec(exec) = &eff.program.ops()[0].body() else {
        panic!("expected SpawnBody::Exec");
    };
    assert_eq!(exec.argv().len(), 1);
    assert!(matches!(
        exec.argv()[0].parts(),
        [specter_core::ArgPart::Literal(s)] if s.as_str() == "/bin/true"
    ));
    // Substitution-domain inputs that the actuator-side resolver renders to SPECTER_PATH /
    // SPECTER_WATCH / SPECTER_FORCED / SPECTER_EVENT_KIND.
    assert!(
        !eff.anchor_path.as_os_str().is_empty(),
        "anchor_path populated; resolver derives target_path from anchor + relative"
    );
    assert!(
        !eff.sub_name.is_empty(),
        "sub_name populated for ${{specter.watch}} / SPECTER_WATCH"
    );
    assert!(
        !eff.forced,
        "SPECTER_FORCED == \"0\" derives from forced=false"
    );
    assert!(
        matches!(eff.key(), specter_core::DedupKey::Subtree { .. }),
        "EVENT_KIND=dir-subtree derives from DedupKey::Subtree variant",
    );
    // SPECTER_DIFF_PATH is an actuator-side augmentation; engine's Effect doesn't carry it. The
    // structural witness is `eff.diff()`:
    assert!(
        eff.diff().is_none(),
        "engine doesn't include diff for non-needs_diff Sub"
    );
    // cwd derives from (anchor_path, anchor_kind) at spawn time. Pin both:
    assert_eq!(eff.anchor_path.as_os_str(), "anchor");
    assert_eq!(eff.anchor_kind, specter_core::ResourceKind::Dir);
}

/// The Subtree suppress decision is `nothing_changed && fired_subs.contains(&dk)` — two gates in
/// conjunction. The per-Profile half (`baseline.hash() == current.hash()`) covers the "fired then
/// noop" arm; the per-Sub half (`fired_subs` existence) is the "Sub has fired before" discriminator
/// that distinguishes a fresh Sub (must fire even on an unchanged tree — first emission) from a
/// repeat fire (suppress on an unchanged tree).
///
/// A noise FsEvent on a fresh Profile drives a phantom Standard burst whose stable verdict observes
/// `baseline.hash() == current.hash()`. Without the `fired_subs.contains` gate, the phantom would
/// suppress the very first Effect — the Sub would never fire and the user's command would never
/// run. This test pins the gate so any future flattening of the suppress derivation (e.g., dropping
/// back to the per-Profile signal alone) fails here, not in production.
#[test]
fn b1_dedup_fresh_sub_fires_on_phantom_standard_burst() {
    let (mut e, pid, _sid, root, _now) = engine_with_attached_sub();
    complete_seed_burst(&mut e, pid);

    // Precondition: post-Seed Profile has baseline = current and no fire history. The phantom
    // condition (`baseline.hash() == current.hash()`) is structurally satisfied; the fresh-Sub
    // condition (`fired_subs.is_empty()`) is by construction.
    let baseline_hash = match e.profiles.get(pid).unwrap().baseline() {
        Some(TreeSnapshot::Dir(arc)) => arc.dir_hash(),
        _ => panic!("post-Seed baseline must be Some(Dir)"),
    };
    assert!(!e.subs.any_fired(pid), "fresh Sub: no fire history");

    // Drive a Standard burst whose probe response equals the Seed baseline byte-for-byte — a
    // phantom (noise FsEvent, no actual disk change). One probe suffices because the stability
    // verdict compares the response against `current.subtree_at(target)`, which equals baseline
    // immediately post-Seed.
    let now = Instant::now();
    e.step(
        Input::FsEvent {
            resource: root,
            event: FsEvent::ContentChanged,
        },
        now,
    );
    while let Some(entry) = e.pop_expired(now + SETTLE) {
        e.step(
            Input::TimerExpired {
                profile: entry.profile,
                kind: entry.kind,
                id: entry.id,
            },
            now + SETTLE,
        );
    }
    let phantom_snap = dir_tree_snap(vec![]);
    assert_eq!(
        phantom_snap.dir_hash(),
        baseline_hash,
        "test setup: probe response must hash-match baseline (phantom)",
    );
    // The verify response folds through `quiescence_verdict` to `Stable(StableReason::Natural)` on
    // the first sample — single dispatch, no prime-then-confirm.
    let corr = e.pending_probe_for(pid).expect("Verifying probe in flight");
    let out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid,
            correlation: corr,
            outcome: ProbeOutcome::SubtreeProven {
                snapshot: dir_tree_snap(vec![]),
                authority: ProofAuthority::Authoritative,
            },
        }),
        now + SETTLE + Duration::from_millis(1),
    );

    assert_eq!(
        out.effects().len(),
        1,
        "fresh Sub must fire on the Authoritative stable verdict, \
         even on a phantom (`baseline.hash() == current.hash()`); \
         `fired_subs.contains` is the discriminator that prevents the \
         noop-suppress path",
    );
}

#[test]
fn emit_effects_subtree_root_uses_parent_dir_for_file_profile() {
    // Contract: SubtreeRoot Sub anchored at a File-kind Profile derives the Effect's `cwd` from the
    // file's parent dir (not the file itself — `Command::current_dir` requires a directory). The
    // surrounding burst flow (probe target, current-shape preservation, graft path) is exercised by
    // `standard_burst_on_file_anchor_targets_anchor_not_parent_dir`; this test asserts only the cwd
    // / env-var contract.
    let mut e = Engine::new();
    let parent = e.tree.ensure_root("parentdir", ResourceRole::User);
    e.tree.set_kind(parent, ResourceKind::Dir);
    let file_anchor = e
        .tree
        .ensure_child(parent, "main.rs", ResourceRole::User)
        .expect("test live parent");
    e.tree.set_kind(file_anchor, ResourceKind::File);
    let now = Instant::now();
    let req = SubAttachRequest {
        anchor: SubAttachAnchor::Resource(file_anchor),
        identity: ProfileIdentity::new(
            ScanConfig::builder().recursive(false).build(),
            MAX_SETTLE,
            DEFAULT_EVENTS,
        ),
        params: SubParams::spawn(
            "build".into(),
            empty_program(),
            EffectScope::SubtreeRoot,
            SETTLE,
            false,
        ),
    };
    let attach_out = e.step(Input::AttachSub(req), now);
    let sid = specter_core::testkit::first_attached_sub(&attach_out).expect("attach_sub succeeded");
    let pid = e.subs.get(sid).unwrap().profile();
    // Seed → Idle: cold-arm Verifying-first, a single Authoritative response pins → SilentPin →
    // Idle. No Batching settle expiry needed (the cold walk is in flight at burst construction).
    let snap = file_tree_snap(EntryKind::File, 0, std::time::UNIX_EPOCH, 1);
    let seed_corr = e
        .pending_probe_for(pid)
        .expect("cold-Seed Verify probe in flight at burst construction");
    e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid,
            correlation: seed_corr,
            outcome: ProbeOutcome::AnchorOk(snap.clone()),
        }),
        now + SETTLE,
    );
    // Standard burst with the same snap (stable). Start it after the Seed pin so the timeline stays
    // monotonic.
    let t1 = now + SETTLE * 3;
    e.step(
        Input::FsEvent {
            resource: file_anchor,
            event: FsEvent::ContentChanged,
        },
        t1,
    );
    let t2 = t1 + SETTLE * 2;
    while let Some(entry) = e.pop_expired(t2) {
        e.step(
            Input::TimerExpired {
                profile: entry.profile,
                kind: entry.kind,
                id: entry.id,
            },
            t2,
        );
    }
    // The verify response folds to `Stable(StableReason::Natural)` on the first sample — single
    // dispatch fires the Effect.
    let std_corr = e.pending_probe_for(pid).expect("Verifying probe in flight");
    let out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid,
            correlation: std_corr,
            outcome: ProbeOutcome::AnchorOk(snap),
        }),
        t2,
    );
    assert_eq!(out.effects().len(), 1);
    let eff = &out.effects()[0];
    // File-kind anchor: actuator's `compute_cwd` returns parent dir. The engine's job here is to
    // pin (anchor_path, anchor_kind) so the actuator's compute_cwd reaches "parentdir". The
    // original cwd assertion ("File-kind anchor uses parent dir as cwd") is now structural:
    // anchor_path is the file, anchor_kind is File ⇒ compute_cwd returns parent.
    assert_eq!(eff.anchor_path.as_os_str(), "parentdir/main.rs");
    assert_eq!(eff.anchor_kind, specter_core::ResourceKind::File);
    // SPECTER_PATH and SPECTER_ANCHOR both derive from `anchor_path` for a File-anchor Subtree
    // Effect: the resolver returns `Cow::Borrowed(&anchor_path)` when `relative()` is empty, so
    // both env values share the same byte sequence.
    assert_eq!(eff.relative(), "");
}

#[test]
fn standard_burst_on_file_anchor_targets_anchor_not_parent_dir() {
    // Realistic Standard-burst-on-File-anchor flow. A real Sensor probing a File anchor returns
    // `TreeSnapshot::File(leaf)`; the engine must (1) probe the anchor itself rather than the
    // parent dir and (2) preserve `Profile.current` as `TreeSnapshot::File(_)` post-dispatch — the
    // snapshot navigation invariant `current` is anchor-shaped breaks if a Standard burst graft
    // wholesale-replaces with a Dir snapshot rooted at the parent.
    let mut e = Engine::new();
    let parent = e.tree.ensure_root("parentdir", ResourceRole::User);
    e.tree.set_kind(parent, ResourceKind::Dir);
    let file_anchor = e
        .tree
        .ensure_child(parent, "main.rs", ResourceRole::User)
        .expect("test live parent");
    e.tree.set_kind(file_anchor, ResourceKind::File);
    let now = Instant::now();
    let req = SubAttachRequest {
        anchor: SubAttachAnchor::Resource(file_anchor),
        identity: ProfileIdentity::new(
            ScanConfig::builder().recursive(false).build(),
            MAX_SETTLE,
            DEFAULT_EVENTS,
        ),
        params: SubParams::spawn(
            "build".into(),
            empty_program(),
            EffectScope::SubtreeRoot,
            SETTLE,
            false,
        ),
    };
    let attach_out = e.step(Input::AttachSub(req), now);
    let sid = specter_core::testkit::first_attached_sub(&attach_out).expect("attach_sub succeeded");
    let pid = e.subs.get(sid).unwrap().profile();

    // Cold-arm Seed Verifying-first: a single Authoritative sample pins → SilentPin → Idle.
    let snap = file_tree_snap(EntryKind::File, 0, UNIX_EPOCH, 1);
    let seed_corr = e
        .pending_probe_for(pid)
        .expect("cold-arm Seed Verify probe in flight at burst construction");
    e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid,
            correlation: seed_corr,
            outcome: ProbeOutcome::AnchorOk(snap.clone()),
        }),
        now + SETTLE,
    );

    // Drive a Standard burst from an FsEvent at the file. Capture the probe request emitted on the
    // settle-timer expiry step. Start it after the Seed burst's two settle windows (monotonic
    // timeline).
    let t1 = now + SETTLE * 3;
    e.step(
        Input::FsEvent {
            resource: file_anchor,
            event: FsEvent::ContentChanged,
        },
        t1,
    );
    let t2 = t1 + SETTLE * 2;
    let mut probe_request: Option<ProbeRequest> = None;
    while let Some(entry) = e.pop_expired(t2) {
        let out = e.step(
            Input::TimerExpired {
                profile: entry.profile,
                kind: entry.kind,
                id: entry.id,
            },
            t2,
        );
        for op in out.probe_ops().iter() {
            if let ProbeOp::Probe { request } = op {
                probe_request = Some(request.clone());
            }
        }
    }

    // (1) The Standard probe is `AnchorFile` and its `target_path` is the anchor's filesystem path.
    // The two assertions are the structural witnesses for the v1 design: File anchors take the
    // typed `AnchorFile` arm (single-`lstat` walker dispatch) and never promote past the anchor to
    // the parent dir.
    let anchor_path = e.tree.path_of(file_anchor).expect("anchor path resolves");
    match probe_request.as_ref() {
        Some(ProbeRequest::AnchorFile { target_path, .. }) => {
            assert_eq!(
                *target_path, anchor_path,
                "AnchorFile target_path is the anchor's filesystem path",
            );
        }
        other => panic!(
            "Standard burst on a File-anchored Profile must emit ProbeRequest::AnchorFile; \
             got {other:?}",
        ),
    }

    // (2) Inject a realistic File response (kqueue per-file FD path). The verify response folds
    // through `quiescence_verdict` to `Stable(StableReason::Natural)` on the first sample — single
    // dispatch, grafts the leaf into `current` and fires the Effect.
    let std_corr = e
        .pending_probe_for(pid)
        .expect("Standard verify probe in flight");
    let out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid,
            correlation: std_corr,
            outcome: ProbeOutcome::AnchorOk(snap),
        }),
        t2,
    );

    let p = e.profiles.get(pid).expect("Profile alive");
    match p.current() {
        Some(TreeSnapshot::File(_)) => {} // navigation invariant preserved
        Some(TreeSnapshot::Dir(arc)) => panic!(
            "Profile.current corrupted to Dir(root_meta={:?}); expected File(leaf)",
            arc.root_meta(),
        ),
        None => panic!("Profile.current must be Some(File(leaf)) post-Standard"),
    }

    // (3) Stable verdict (same leaf hash) + dirty=0 ⇒ exactly one Effect fires.
    assert_eq!(
        out.effects().len(),
        1,
        "stable verdict + dirty=0 ⇒ exactly one Effect fires",
    );

    // (4) The anchor's `watch_demand` is exactly 1 (Profile claim only).
    assert_eq!(
        e.tree
            .get(file_anchor)
            .map(specter_core::Resource::watch_demand),
        Some(1),
        "no spurious watch_demand bump on the anchor",
    );
}

#[test]
fn standard_burst_force_fires_on_max_settle() {
    let (mut e, pid, _sid, root, _now) = engine_with_attached_sub();
    complete_seed_burst(&mut e, pid);
    let now = Instant::now();
    e.step(
        Input::FsEvent {
            resource: root,
            event: FsEvent::ContentChanged,
        },
        now,
    );
    // Burst deadline fires before settle.
    let deadline = now + MAX_SETTLE + Duration::from_millis(1);
    while let Some(entry) = e.pop_expired(deadline) {
        e.step(
            Input::TimerExpired {
                profile: entry.profile,
                kind: entry.kind,
                id: entry.id,
            },
            deadline,
        );
    }
    // After force-fire, we're either in Verifying (forced=true) or already Awaiting if the deadline
    // race resolved both timers. Drive the response back if needed.
    if let Some(correlation) = e.pending_probe_for(pid) {
        // Inject a not-stable response to test the forced effect emission.
        let snap = dir_tree_snap(vec![("new.rs", EntryKind::File, 99)]);
        let out = e.step(
            Input::ProbeResponse(ProbeResponse {
                owner: pid,
                correlation,
                outcome: ProbeOutcome::SubtreeProven {
                    snapshot: snap,
                    authority: ProofAuthority::Authoritative,
                },
            }),
            deadline,
        );
        // Forced fire transitions to Awaiting (Effect in flight). The post-fire rebase happens when
        // the eventual EffectComplete drives the Awaiting → Rebasing transition.
        let phase = match e.profiles.get(pid).unwrap().state() {
            ProfileState::Active(ActiveBurst::PostFire(post), _) => &post.phase,
            _ => panic!("expected Active(Awaiting)"),
        };
        assert!(
            matches!(phase, PostFirePhase::Awaiting { outstanding: 1, .. }),
            "force-fired stable verdict transitions to Awaiting; got {phase:?}",
        );
        assert_eq!(out.effects().len(), 1);
        assert!(
            out.effects()[0].forced,
            "force-fired Effect must carry forced=true",
        );
    }
}

#[test]
fn fs_event_content_changed_during_seed_probing_preserves_intent() {
    let (mut e, pid, _sid, root, _) = engine_with_attached_sub();
    // The Seed burst is Batching-first; expire the settle window so it reaches Verifying with a
    // probe in flight, then inject an FsEvent — it should cancel that probe and return to
    // Active(Seed Batching) with the Seed intent preserved.
    assert_seed_verifying(&e);
    let out = e.step(
        Input::FsEvent {
            resource: root,
            event: FsEvent::ContentChanged,
        },
        Instant::now(),
    );
    let burst = match e.profiles.get(pid).unwrap().state() {
        ProfileState::Active(ActiveBurst::PreFire(pre), _) => pre,
        _ => panic!(),
    };
    assert_eq!(
        burst.intent,
        BurstIntent::Seed,
        "intent preserved across Verifying → Batching",
    );
    assert!(matches!(burst.phase, PreFirePhase::Batching { .. }));
    let cancels = out
        .probe_ops()
        .iter()
        .filter(|op| matches!(op, ProbeOp::Cancel { .. }))
        .count();
    assert_eq!(cancels, 1);
}

/// Field-discipline pin for `event_drives_batching`: an FsEvent during Verifying disarms the verify
/// slot atomically with the Cancel emission, so the `Verifying → Batching` rewrite cannot leave an
/// armed slot behind for the just-cancelled probe.
#[test]
fn event_drives_batching_clears_pending_probe() {
    let (mut e, pid, _sid, root, _) = engine_with_attached_sub();
    // The Seed burst is Batching-first; drive it to Verifying so a probe is actually in flight to
    // be cleared.
    assert_seed_verifying(&e);
    assert!(
        e.pending_probe_for(pid).is_some(),
        "Seed probe in flight after settle expiry",
    );

    let _ = e.step(
        Input::FsEvent {
            resource: root,
            event: FsEvent::ContentChanged,
        },
        Instant::now(),
    );

    assert!(
        e.pending_probe_for(pid).is_none(),
        "slot disarmed atomically with Verifying → Batching transition",
    );
}

/// Hash-channel carrier survival across the pre-fire phase swaps, pinned through the **verdict** —
/// `last_certified_hash` is `pub(crate)` on core, so this crate cannot read the field directly and
/// pins survival black-box. The Layer-C channel needs the prior sample to ride through every
/// pre-fire swap for a *second, equal* Authoritative sample to fold `Stable(Natural)` and fire; if
/// any phase helper reset the carrier the second sample's `prior` reverts to `None`, folding
/// `Unstable` forever (never fires). The fire is therefore the executable witness that the carrier
/// survived `transition_to_verifying` (Batching → Verifying), `retry_drives_batching` (the first
/// sample's `Unstable` re-Batch), and `event_drives_batching` (an FsEvent during Verifying).
///
/// Requires a CONTENT-free (`STRUCTURE`-only) Profile so the channel engages
/// (`events_witness_quiescence == false`); a CONTENT-bearing Profile closes the obligation on one
/// `EventsReliable` sample and never threads the carrier.
#[test]
fn hash_channel_carrier_survives_pre_fire_swaps_to_stable_resample() {
    let mut e = Engine::new();
    let anchor = e.tree.ensure_root("anchor", ResourceRole::User);
    e.tree.set_kind(anchor, ResourceKind::Dir);
    let t0 = Instant::now();
    // STRUCTURE-only ⇒ the N=2 hash channel engages on the Standard burst.
    let (_sid, pid) = crate::testkit::attach_structure_only(&mut e, anchor, t0);
    // Cold-arm Seed pins on its first Authoritative sample (SilentPin owes no quiescence proof ⇒
    // EventsReliable), independent of the mask; baseline := empty dir.
    complete_seed_burst(&mut e, pid);

    // Both Standard samples share one non-empty snapshot: their certified hashes are equal (the
    // fold's `prior == response` input) and differ from the empty baseline (so the fire is not
    // B1-deduped).
    let sample = || dir_tree_snap(vec![("a.rs", EntryKind::File, 1)]);

    // Idle → Batching on the driving event.
    let _ = e.step(
        Input::FsEvent {
            resource: anchor,
            event: FsEvent::ContentChanged,
        },
        t0,
    );

    // Batching → Verifying (#1) via settle expiry.
    let t1 = t0 + SETTLE * 2;
    crate::testkit::drain_due(&mut e, t1);
    let corr1 = e
        .pending_probe_for(pid)
        .expect("settle expiry reaches the first Verifying probe");

    // Sample 1: Authoritative hash H. prior = None ⇒ Unstable ⇒ retry_drives_batching re-Batches;
    // carrier := Some(H).
    let out1 = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid,
            correlation: corr1,
            outcome: ProbeOutcome::SubtreeProven {
                snapshot: sample(),
                authority: ProofAuthority::Authoritative,
            },
        }),
        t1,
    );
    assert!(
        out1.effects().is_empty(),
        "first hash-channel sample folds Unstable (prior None) — no fire",
    );
    assert!(
        matches!(
            e.profiles.get(pid).unwrap().state(),
            ProfileState::Active(ActiveBurst::PreFire(pre), _)
                if matches!(pre.phase, PreFirePhase::Batching { .. })
        ),
        "Unstable re-Batches via retry_drives_batching",
    );

    // Batching → Verifying (#2) via settle expiry.
    let t2 = t1 + SETTLE * 2;
    crate::testkit::drain_due(&mut e, t2);
    assert!(
        e.pending_probe_for(pid).is_some(),
        "settle expiry reaches the second Verifying probe",
    );

    // FsEvent during Verifying ⇒ event_drives_batching cancels the probe and re-Batches. The
    // carrier (Some(H)) must ride through this swap.
    let _ = e.step(
        Input::FsEvent {
            resource: anchor,
            event: FsEvent::ContentChanged,
        },
        t2,
    );
    assert!(
        matches!(
            e.profiles.get(pid).unwrap().state(),
            ProfileState::Active(ActiveBurst::PreFire(pre), _)
                if matches!(pre.phase, PreFirePhase::Batching { .. })
        ),
        "event_drives_batching cancels the verify and re-Batches",
    );

    // Batching → Verifying (#3) via settle expiry.
    let t3 = t2 + SETTLE * 2;
    crate::testkit::drain_due(&mut e, t3);
    let corr3 = e
        .pending_probe_for(pid)
        .expect("settle expiry reaches the third Verifying probe");

    // Sample 2: Authoritative, *equal* hash H. prior = Some(H) == response ⇒ Stable(Natural) ⇒ FIRE
    // — iff the carrier survived swaps #1–#3.
    let out2 = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid,
            correlation: corr3,
            outcome: ProbeOutcome::SubtreeProven {
                snapshot: sample(),
                authority: ProofAuthority::Authoritative,
            },
        }),
        t3,
    );
    assert!(
        !out2.effects().is_empty(),
        "second equal Authoritative sample folds Stable(Natural) and fires — \
         the surviving carrier; a reset would fold Unstable forever",
    );
    assert!(
        matches!(
            e.profiles.get(pid).unwrap().state(),
            ProfileState::Active(ActiveBurst::PostFire(_), _)
        ),
        "the fire transitions the burst into the post-fire tail",
    );
}

/// Field-discipline pin for `finalize_anchor_lost`: an anchor terminal event during Verifying
/// cancels the in-flight probe and clears the channel.
#[test]
fn finalize_anchor_lost_during_verifying_clears_pending_probe() {
    let (mut e, pid, _sid, root, _) = engine_with_attached_sub();
    // The Seed burst is Batching-first; drive it to Verifying so an anchor-terminal event has an
    // in-flight probe to cancel.
    assert_seed_verifying(&e);
    assert!(
        e.pending_probe_for(pid).is_some(),
        "Seed probe in flight after settle expiry",
    );

    let out = e.step(
        Input::FsEvent {
            resource: root,
            event: FsEvent::Removed,
        },
        Instant::now(),
    );

    assert!(
        e.pending_probe_for(pid).is_none(),
        "anchor terminal during Verifying disarms the slot",
    );
    let cancels = out
        .probe_ops()
        .iter()
        .filter(|op| matches!(op, ProbeOp::Cancel { owner: profile} if *profile == pid))
        .count();
    assert_eq!(
        cancels,
        1,
        "exactly one Cancel emitted; got {:?}",
        out.probe_ops()
    );
}

/// Single-diagnostic guarantee for stale `ProbeResponse`. The top-level `pending_probe ==
/// Some(received)` check is the sole stale gate — exactly one diagnostic per stale response, with
/// no second state-shape or inner-correlation layer to double-fire on degenerate inputs.
#[test]
fn stale_probe_response_emits_exactly_one_diagnostic() {
    let (mut e, pid, _sid, _root, _) = engine_with_attached_sub();
    // The Seed burst is Batching-first; drive it to Verifying so a legitimate Seed probe is live
    // when the stale response lands.
    assert_seed_verifying(&e);
    let bogus = specter_core::ProbeCorrelation::from(99_999);
    let snap = dir_tree_snap(vec![]);

    let out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid,
            correlation: bogus,
            outcome: ProbeOutcome::SubtreeProven {
                snapshot: snap,
                authority: ProofAuthority::Authoritative,
            },
        }),
        Instant::now(),
    );

    let stale_count = out
        .diagnostics
        .iter()
        .filter(|d| matches!(d, Diagnostic::StaleProbeResponse { owner: profile, .. } if *profile == pid))
        .count();
    assert_eq!(
        stale_count, 1,
        "exactly one StaleProbeResponse diagnostic; got {:?}",
        out.diagnostics,
    );
    // Live channel untouched: the legitimate Seed probe is still in flight.
    assert!(
        e.pending_probe_for(pid).is_some(),
        "live channel untouched by stale response",
    );
    let _ = e.cancel_all_in_flight_probes();
}

/// Anchor events bypass the class filter unconditionally. Profile has events = EMPTY (nothing in
/// the mask); a `MetadataChanged` at the anchor still drives the lifecycle path (burst start), and
/// no `EventClassDropped` is emitted. This guards the lifecycle-continuity invariant: anchor events
/// never get filtered out by user mask choice.
#[test]
fn fs_event_metadatachanged_at_anchor_bypasses_class_filter() {
    let (mut e, pid, _sid, root, _now) = engine_with_attached_sub();
    complete_seed_burst(&mut e, pid);
    let out = e.step(
        Input::FsEvent {
            resource: root,
            event: FsEvent::MetadataChanged,
        },
        Instant::now(),
    );
    assert!(
        !out.diagnostics
            .iter()
            .any(|d| matches!(d, Diagnostic::EventClassDropped { .. })),
        "anchor events bypass the class filter",
    );
    assert!(
        matches!(
            e.profiles.get(pid).unwrap().state(),
            ProfileState::Active(_, _),
        ),
        "MetadataChanged at the anchor drives a burst even on EMPTY mask",
    );
}

/// Descendant events whose class is not in the covering Profile's `events_union` drop with
/// `EventClassDropped` BEFORE driving the burst. Profile has events = EMPTY ⇒ `intersects(any_class)
/// == false`, so a `MetadataChanged` on a covered descendant drops cleanly without state mutation.
#[test]
fn fs_event_metadatachanged_at_descendant_drops_with_event_class_dropped() {
    let (mut e, pid, _sid, root, _now) = engine_with_attached_sub();
    complete_seed_burst(&mut e, pid);

    // Materialize a covered descendant. Bump `watch_demand` so the event passes the
    // `EventOnUnwatchedResource` head guard. The Profile's ScanConfig has `recursive(true)` so
    // `covers(profile, child, tree)` is satisfied.
    let child = e
        .tree
        .ensure_child(root, "child.txt", ResourceRole::User)
        .expect("test live parent");
    e.tree.set_kind(child, ResourceKind::File);
    e.tree.get_mut(child).unwrap().insert_contribution(
        specter_core::ContribKey::ProfileDescendant(pid),
        ClassSet::CONTENT,
    );

    let out = e.step(
        Input::FsEvent {
            resource: child,
            event: FsEvent::MetadataChanged,
        },
        Instant::now(),
    );
    let has_class_drop = out.diagnostics.iter().any(|d| {
        matches!(
            d,
            Diagnostic::EventClassDropped {
                resource,
                event: FsEvent::MetadataChanged,
                profile,
            } if *resource == child && *profile == pid,
        )
    });
    assert!(
        has_class_drop,
        "descendant MetadataChanged drops with EventClassDropped on EMPTY mask",
    );
    // No `MetadataChangedIgnored` lingers — the variant was deleted. No state mutation: the filter
    // `continue`s before drive_burst.
    assert!(matches!(
        e.profiles.get(pid).unwrap().state(),
        ProfileState::Idle,
    ));
}

/// Identity events on a *descendant File* fold into the CONTENT class. A Profile excluding CONTENT
/// (here: STRUCTURE-only on a Dir anchor) drops the descendant File `Removed` with
/// `EventClassDropped`. The dropped event is not routed through `on_anchor_terminal_event` — that
/// routing is anchor-only.
#[test]
fn fs_event_terminal_on_descendant_file_folds_to_content_and_drops() {
    let mut e = Engine::new();
    let r = e.tree.ensure_root("anchor", ResourceRole::User);
    e.tree.set_kind(r, ResourceKind::Dir);
    let now = Instant::now();
    let req = SubAttachRequest {
        anchor: SubAttachAnchor::Resource(r),
        identity: ProfileIdentity::new(
            ScanConfig::builder().recursive(true).build(),
            MAX_SETTLE,
            ClassSet::STRUCTURE,
        ),
        params: SubParams::spawn(
            "test-sub".into(),
            empty_program(),
            EffectScope::SubtreeRoot,
            SETTLE,
            false,
        ),
    };
    let out = e.step(Input::AttachSub(req), now);
    let sid = specter_core::testkit::first_attached_sub(&out).expect("attach_sub succeeded");
    let pid = e.subs.get(sid).unwrap().profile();
    complete_seed_burst(&mut e, pid);

    let child = e
        .tree
        .ensure_child(r, "f.txt", ResourceRole::User)
        .expect("test live parent");
    e.tree.set_kind(child, ResourceKind::File);
    e.tree.get_mut(child).unwrap().insert_contribution(
        specter_core::ContribKey::ProfileDescendant(pid),
        ClassSet::STRUCTURE,
    );

    let out = e.step(
        Input::FsEvent {
            resource: child,
            event: FsEvent::Removed,
        },
        Instant::now(),
    );
    let has_class_drop = out.diagnostics.iter().any(|d| {
        matches!(
            d,
            Diagnostic::EventClassDropped {
                event: FsEvent::Removed,
                ..
            },
        )
    });
    assert!(
        has_class_drop,
        "Removed on a descendant File folds to CONTENT and drops on STRUCTURE-only mask",
    );
    // Profile remains Idle: dropped events do not extend dirty sets.
    assert!(matches!(
        e.profiles.get(pid).unwrap().state(),
        ProfileState::Idle,
    ));
    // Sanity: anchor's contribution is intact (we did NOT terminate).
    assert_eq!(
        e.profiles.get(pid).unwrap().anchor_claim(),
        AnchorClaim::Held,
    );
    let _ = sid;
}

/// Terminal events on the anchor route through `on_anchor_terminal_event` regardless of the
/// Profile's `events_union`. Anchor is a Dir, events = EMPTY: the kqexec class for `Removed` on a
/// Dir is STRUCTURE — not in the EMPTY mask — but anchor events bypass the filter. After the call,
/// `anchor_claim` is cleared to `None` and `baseline` / `current` are dropped.
#[test]
fn fs_event_anchor_terminal_bypasses_class_filter() {
    let (mut e, pid, _sid, root, _now) = engine_with_attached_sub();
    complete_seed_burst(&mut e, pid);
    assert_eq!(
        e.profiles.get(pid).unwrap().anchor_claim(),
        AnchorClaim::Held,
        "anchor_claim set to Held after attach_sub_inner",
    );

    let out = e.step(
        Input::FsEvent {
            resource: root,
            event: FsEvent::Removed,
        },
        Instant::now(),
    );
    assert!(
        !out.diagnostics
            .iter()
            .any(|d| matches!(d, Diagnostic::EventClassDropped { .. })),
        "anchor terminal events bypass the class filter",
    );

    let p = e.profiles.get(pid).unwrap();
    assert_eq!(
        p.anchor_claim(),
        AnchorClaim::None,
        "anchor_claim cleared by on_anchor_terminal_event",
    );
    assert!(p.baseline().is_none());
    assert!(p.current().is_none());
    // Root anchor — no recovery parent, so the observed-loss wrapper's fallback parks.
    assert!(matches!(p.state(), ProfileState::Parked));
}

#[test]
fn fs_event_for_unwatched_resource_emits_diagnostic() {
    let mut e = Engine::new();
    let r = e.tree.ensure_root("ghost", ResourceRole::User);
    let out = e.step(
        Input::FsEvent {
            resource: r,
            event: FsEvent::ContentChanged,
        },
        Instant::now(),
    );
    let has_diag = out
        .diagnostics
        .iter()
        .any(|d| matches!(d, Diagnostic::EventOnUnwatchedResource { .. }));
    assert!(has_diag);
}

#[test]
fn fs_event_at_watched_resource_with_no_consumer_emits_event_no_consumer_not_unwatched() {
    // A WatchRootParent fires `StructureChanged` (e.g., a sibling directory was created / renamed)
    // and no Profile in the engine cares. The event must NOT be diagnosed as "unwatched resource" —
    // the Resource IS Watched. The new `EventNoConsumer` variant signals this benign case so the
    // bin can log it at TRACE rather than WARN.
    let mut e = Engine::new();
    // Materialize an unrelated Watched resource (e.g., a parent that someone else holds open).
    // watch_demand > 0 ensures the event isn't routed through the `EventOnUnwatchedResource` path.
    let r = e.tree.ensure_root("lonely", ResourceRole::User);
    e.tree.set_kind(r, ResourceKind::Dir);
    e.tree.get_mut(r).unwrap().insert_contribution(
        specter_core::ContribKey::ProfileAnchor(specter_core::ProfileId::default()),
        ClassSet::STRUCTURE,
    );

    let out = e.step(
        Input::FsEvent {
            resource: r,
            event: FsEvent::StructureChanged,
        },
        Instant::now(),
    );

    let no_consumer = out
        .diagnostics
        .iter()
        .any(|d| matches!(d, Diagnostic::EventNoConsumer { .. }));
    let unwatched = out
        .diagnostics
        .iter()
        .any(|d| matches!(d, Diagnostic::EventOnUnwatchedResource { .. }));
    assert!(no_consumer, "should emit EventNoConsumer");
    assert!(
        !unwatched,
        "should NOT emit EventOnUnwatchedResource — the resource IS Watched"
    );
}

#[test]
fn fs_event_removed_at_anchor_active_terminates() {
    let (mut e, pid, _sid, root, _now) = engine_with_attached_sub();
    complete_seed_burst(&mut e, pid);
    // Drive a Standard burst.
    let now = Instant::now();
    e.step(
        Input::FsEvent {
            resource: root,
            event: FsEvent::ContentChanged,
        },
        now,
    );
    assert!(matches!(
        e.profiles.get(pid).unwrap().state(),
        ProfileState::Active(_, _),
    ));
    // Now Removed at anchor → terminate.
    let out = e.step(
        Input::FsEvent {
            resource: root,
            event: FsEvent::Removed,
        },
        now,
    );
    let p = e.profiles.get(pid).unwrap();
    // Root anchor — no recovery parent, so the loss wrapper's fallback parks.
    assert!(matches!(p.state(), ProfileState::Parked));
    assert!(p.baseline().is_none());
    assert!(p.current().is_none());
    // watch_demand on anchor → 0; one Unwatch op emitted.
    assert_eq!(e.tree.get(root).unwrap().watch_demand(), 0);
    let unwatches = out
        .watch_ops
        .iter()
        .filter(|op| matches!(op, WatchOp::Unwatch { .. }))
        .count();
    assert!(unwatches >= 1);
}

#[test]
fn fs_event_removed_at_anchor_idle_releases_watch_and_clears_baseline() {
    // FsEvent: Removed/Renamed/Revoked on an idle-but-anchored profile transitions idempotently. We
    // additionally release the watch contribution and drop baseline/current — they refer to a
    // now-vanished slot. This root anchor has no recovery parent, so the loss wrapper's fallback
    // parks the Profile; the `Parked` state is what the event-scan recovery arm selects on later
    // activity at the slot.
    let (mut e, pid, _sid, root, _now) = engine_with_attached_sub();
    complete_seed_burst(&mut e, pid);
    assert_eq!(e.tree.get(root).unwrap().watch_demand(), 1);
    assert!(e.profiles.get(pid).unwrap().current().is_some());

    let _ = e.step(
        Input::FsEvent {
            resource: root,
            event: FsEvent::Removed,
        },
        Instant::now(),
    );

    // Root anchor — no recovery parent, so the loss wrapper's fallback parks.
    assert!(matches!(
        e.profiles.get(pid).unwrap().state(),
        ProfileState::Parked,
    ));
    // watch_demand released; baseline/current cleared.
    assert_eq!(e.tree.get(root).unwrap().watch_demand(), 0);
    let p = e.profiles.get(pid).unwrap();
    assert!(p.baseline().is_none());
    assert!(p.current().is_none());
}

#[test]
fn count_gate_zero_iff_no_carrier_and_anchor_loss_while_idle_balances_nonsteady() {
    // Oracle (b): the O(1) carrier gate is sound. `nonsteady() == 0` ⇒ `classify_event_carriers`
    // empty ∀ r — a healthy *anchored* Idle Profile is excluded by the pure state predicate
    // (`Pending ∨ Parked`), so a quiet watcher never pins the gate. And the count stays balanced
    // across the anchor-loss-while-Idle park (the loss wrapper's `Idle → Parked` edge through the
    // `ProfileMap` chokepoint). The debug count-vs-full-scan tripwire inside
    // `classify_event_carriers` runs on every covering scan here (and across the whole suite), so a
    // desync panics in debug regardless of the explicit asserts below — this test pins the
    // *implication* and the loss balance the tripwire alone does not state.
    let (mut e, pid, _sid, root, _now) = engine_with_attached_sub();
    complete_seed_burst(&mut e, pid);

    // A carrier is empty iff both carrier classes are empty.
    let empty = |e: &Engine, r: ResourceId| {
        let c = e.classify_event_carriers(r);
        c.descents.is_empty() && c.recoveries.is_empty()
    };

    // Healthy anchored Idle ⇒ carrier-free steady state.
    {
        let p = e.profiles().get(pid).unwrap();
        assert!(p.current().is_some() && matches!(p.state(), ProfileState::Idle));
    }
    assert_eq!(
        e.profiles().nonsteady(),
        0,
        "a healthy anchored Idle Profile is excluded from the carrier count",
    );
    assert!(empty(&e, root), "gate zero ⇒ no carrier at the anchor");
    if let Some(par) = e.tree.parent(root) {
        assert!(empty(&e, par), "gate zero ⇒ no carrier at the parent");
    }

    // Anchor lost while Idle: `Removed @ anchor` ⇒ the observed-loss wrapper. This root anchor has
    // no recovery parent, so the fallback parks — a chokepointed `Idle → Parked` edge that must
    // move the count 0 → 1.
    e.step(
        Input::FsEvent {
            resource: root,
            event: FsEvent::Removed,
        },
        Instant::now(),
    );

    {
        let p = e.profiles().get(pid).unwrap();
        assert!(matches!(p.state(), ProfileState::Parked) && p.current().is_none());
    }
    assert_eq!(
        e.profiles().nonsteady(),
        1,
        "the park's state edge recorded the carrier count (a zero here would \
         false-skip the recovery scan)",
    );
    // The gated scan finds the park through its anchor-slot channel: the slot is the Profile's own
    // anchor, so an event there is a recovery signal even with no watch_root_parent cached.
    assert!(
        !empty(&e, root),
        "the parked Profile is a recovery carrier at its own anchor slot",
    );
}

// ---- TimerExpired dispatch ----

#[test]
fn timer_expired_settle_in_settling_transitions_to_probing() {
    let (mut e, pid, _sid, root, _now) = engine_with_attached_sub();
    complete_seed_burst(&mut e, pid);
    let now = Instant::now();
    e.step(
        Input::FsEvent {
            resource: root,
            event: FsEvent::ContentChanged,
        },
        now,
    );
    // Pop the settle timer.
    let entry = e.pop_expired(now + SETTLE).expect("settle timer ready");
    let out = e.step(
        Input::TimerExpired {
            profile: entry.profile,
            kind: entry.kind,
            id: entry.id,
        },
        now + SETTLE,
    );
    let p = e.profiles.get(pid).unwrap();
    assert!(matches!(
        p.state(),
        ProfileState::Active(_, _) // Verifying
    ));
    let probes = out
        .probe_ops()
        .iter()
        .filter(|op| matches!(op, ProbeOp::Probe { .. }))
        .count();
    assert_eq!(probes, 1);
    let _ = e.cancel_all_in_flight_probes();
}

/// An external `FsEvent` on the burst's own anchor, arriving mid-Batching, is processed (not
/// silently dropped): it lands in the pre-fire burst's `dirty` and advances `last_event_time`, so
/// the next settle expiry **reschedules** the settle timer (debounce) instead of verifying. The two
/// anchor events plus the reschedule collapse into a single fire — no double-fire. Deterministic
/// via explicit clock control; no soak.
#[test]
fn pre_fire_anchor_event_rearms_settle_and_fires_once() {
    let (mut e, pid, _sid, root, _now) = engine_with_attached_sub();
    complete_seed_burst(&mut e, pid);
    let t0 = Instant::now();

    // First anchor event opens the Standard burst (Idle → Batching).
    let out_a = e.step(
        Input::FsEvent {
            resource: root,
            event: FsEvent::ContentChanged,
        },
        t0,
    );
    assert!(
        matches!(
            e.profiles.get(pid).unwrap().state(),
            ProfileState::Active(ActiveBurst::PreFire(pre), _)
                if matches!(pre.phase, PreFirePhase::Batching { .. })
        ),
        "first anchor event opens Batching",
    );
    assert!(out_a.effects().is_empty(), "no fire at burst open");

    // Second anchor event, still inside the settle window. Assert it is recorded — it enters the
    // burst accumulator and advances `last_event_time`.
    let t1 = t0 + SETTLE / 2;
    let out_b = e.step(
        Input::FsEvent {
            resource: root,
            event: FsEvent::ContentChanged,
        },
        t1,
    );
    {
        let pre = match e.profiles.get(pid).unwrap().state() {
            ProfileState::Active(ActiveBurst::PreFire(pre), _) => pre,
            other => panic!("expected Active(PreFire) mid-burst, got {other:?}"),
        };
        assert!(
            matches!(pre.phase, PreFirePhase::Batching { .. }),
            "second anchor event keeps Batching (timer reused, not re-minted)",
        );
        let root_path = e.tree.path_of(root).expect("anchor path resolves");
        assert!(
            pre.dirty.chains().contains(&root_path),
            "anchor event's path tracked in dirty (the obligation basis) — \
             not silently dropped",
        );
        assert_eq!(
            pre.last_event_time,
            Some(t1),
            "anchor event advanced last_event_time (debounce basis)",
        );
    }
    assert!(out_b.effects().is_empty(), "no fire on the second event");

    // The original settle timer expires at its deadline (t0 + SETTLE), but the last event was at t1
    // (< SETTLE ago) → debounce: stay Batching, reschedule a fresh settle at last_event_time +
    // SETTLE.
    let ta = t0 + SETTLE;
    let entry = e.pop_expired(ta).expect("first settle timer ready");
    let out_c = e.step(
        Input::TimerExpired {
            profile: entry.profile,
            kind: entry.kind,
            id: entry.id,
        },
        ta,
    );
    assert!(
        matches!(
            e.profiles.get(pid).unwrap().state(),
            ProfileState::Active(ActiveBurst::PreFire(pre), _)
                if matches!(pre.phase, PreFirePhase::Batching { .. })
        ),
        "debounce: still Batching after the first settle expiry",
    );
    assert!(
        e.pending_probe_for(pid).is_none(),
        "debounce did not verify — no probe in flight",
    );
    assert!(
        out_c.probe_ops().is_empty() && out_c.effects().is_empty(),
        "debounce emitted neither probe nor effect",
    );

    // Quiet for ≥ SETTLE past the last event: the rescheduled timer expires and transitions to
    // Verifying with exactly one probe.
    let tb = t0 + SETTLE * 2;
    let entry = e
        .pop_expired(tb)
        .expect("rescheduled settle timer ready — proves the re-arm");
    let out_d = e.step(
        Input::TimerExpired {
            profile: entry.profile,
            kind: entry.kind,
            id: entry.id,
        },
        tb,
    );
    assert!(
        e.pop_expired(tb).is_none(),
        "exactly one settle timer pending after the re-arm",
    );
    let probes = out_d
        .probe_ops()
        .iter()
        .filter(|op| matches!(op, ProbeOp::Probe { .. }))
        .count();
    assert_eq!(probes, 1, "rescheduled settle transitions to one verify");
    assert!(
        out_d.effects().is_empty(),
        "no fire before the verify responds"
    );

    // The verify response folds to `Stable(StableReason::Natural)` on the first sample — single
    // dispatch fires the Effect. The Sub never fired (Seed does not fire), so B1 does not suppress.
    let correlation = e.pending_probe_for(pid).expect("Verifying probe in flight");
    let out_e = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid,
            correlation,
            outcome: ProbeOutcome::SubtreeProven {
                snapshot: dir_tree_snap(vec![]),
                authority: ProofAuthority::Authoritative,
            },
        }),
        tb + Duration::from_millis(1),
    );
    assert_eq!(
        out_e.effects().len(),
        1,
        "two anchor events + a debounce reschedule fire exactly once",
    );
    assert!(
        matches!(
            e.profiles.get(pid).unwrap().state(),
            ProfileState::Active(ActiveBurst::PostFire(post), _)
                if matches!(post.phase, PostFirePhase::Awaiting { outstanding: 1, .. })
        ),
        "single fire cycle — one outstanding Effect, no second burst",
    );
}

#[test]
fn timer_expired_stale_id_emits_diagnostic() {
    let mut e = Engine::new();
    let bogus = specter_core::TimerId::from(99_999);
    let out = e.step(
        Input::TimerExpired {
            profile: specter_core::ProfileId::default(),
            kind: specter_core::TimerKind::Settle,
            id: bogus,
        },
        Instant::now(),
    );
    let stale = out
        .diagnostics
        .iter()
        .any(|d| matches!(d, Diagnostic::StaleTimer { .. }));
    assert!(stale);
}

// ---- EffectComplete dispatch ----
//
// The engine does not return to Idle after firing Effects: the burst stays `Active(Awaiting)` until
// each completion reports back, and the post-Effect rebase happens in `PostFirePhase::Rebasing` as
// a phase of the same burst. `EffectComplete` arrivals route by phase: Awaiting decrements /
// transitions; non-Awaiting emits `EffectCompleteOutsideAwaiting`.

#[test]
fn effect_complete_ok_in_idle_diagnoses_outside_awaiting() {
    // No path leaves Idle with an outstanding EffectComplete: the burst stays Active(Awaiting)
    // until completions arrive. A completion landing in Idle is therefore unexpected (gate-deadline
    // force- transition or anchor-loss) — emit `EffectCompleteOutsideAwaiting` and drop without
    // state change.
    let (mut e, pid, sid, _root, _now) = engine_with_attached_sub();
    complete_seed_burst(&mut e, pid);
    let out = e.step(
        Input::EffectComplete(EffectCompletion {
            sub: sid,
            key: DedupKey::Subtree {
                sub: sid,
                profile: pid,
            },
            outcome: EffectOutcome::Ok,
        }),
        Instant::now(),
    );
    let has_diag = out.diagnostics.iter().any(|d| {
        matches!(
            d,
            Diagnostic::EffectCompleteOutsideAwaiting { sub: s, profile: p }
                if *s == sid && *p == pid,
        )
    });
    assert!(
        has_diag,
        "EffectComplete::Ok in Idle is a late completion and diagnoses",
    );
    // No probe emitted — the Profile stays Idle.
    assert!(out.probe_ops().is_empty());
    assert!(matches!(
        e.profiles.get(pid).unwrap().state(),
        ProfileState::Idle,
    ));
}

#[test]
fn effect_complete_failed_in_idle_clears_hash_and_diagnoses() {
    // Failed always clears `fired_subs[key]` regardless of phase — a failed Effect leaves no
    // observable state to dedupe against. In Idle the completion is also "late" (the engine isn't
    // tracking it), so it diagnoses with EffectCompleteOutsideAwaiting.
    let (mut e, pid, sid, _root, _now) = engine_with_attached_sub();
    complete_seed_burst(&mut e, pid);
    let pre_baseline = e.profiles.get(pid).unwrap().baseline().is_some();
    let out = e.step(
        Input::EffectComplete(EffectCompletion {
            sub: sid,
            key: DedupKey::Subtree {
                sub: sid,
                profile: pid,
            },
            outcome: EffectOutcome::Failed(Termination::Exit(1)),
        }),
        Instant::now(),
    );
    let has_diag = out.diagnostics.iter().any(|d| {
        matches!(
            d,
            Diagnostic::EffectCompleteOutsideAwaiting { sub: s, profile: p }
                if *s == sid && *p == pid,
        )
    });
    assert!(has_diag, "Failed in Idle diagnoses as late completion");
    assert!(out.effects().is_empty());
    assert!(out.probe_ops().is_empty());
    assert_eq!(
        e.profiles.get(pid).unwrap().baseline().is_some(),
        pre_baseline,
        "baseline unchanged on Failed",
    );
}

// ---- Effect needs_diff carries Diff ----

#[test]
fn effect_emission_carries_diff_when_needs_diff() {
    let mut e = Engine::new();
    let r = e.tree.ensure_root("anchor", ResourceRole::User);
    e.tree.set_kind(r, ResourceKind::Dir);
    let now = Instant::now();
    let req = SubAttachRequest {
        anchor: SubAttachAnchor::Resource(r),
        identity: ProfileIdentity::new(
            ScanConfig::builder().recursive(true).build(),
            MAX_SETTLE,
            NO_EVENTS,
        ),
        params: SubParams::spawn(
            "fmt".into(),
            diff_program(), // references ${specter.created}
            EffectScope::SubtreeRoot,
            SETTLE,
            false,
        ),
    };
    let out = e.step(Input::AttachSub(req), now);
    let sid = specter_core::testkit::first_attached_sub(&out).expect("attach_sub succeeded");
    let pid = e.subs.get(sid).unwrap().profile();
    assert!(e.subs.get(sid).unwrap().spawn_spec().unwrap().needs_diff());

    // Seed burst → baseline = empty snapshot.
    complete_seed_burst(&mut e, pid);

    // Standard burst, first round: FsEvent → settle → probe → snapshot with a new entry. The first
    // response is *not stable* (current was empty), so the Engine reschedules another settle cycle.
    e.step(
        Input::FsEvent {
            resource: r,
            event: FsEvent::ContentChanged,
        },
        now,
    );
    let snap_with_entry = dir_tree_snap(vec![("new.rs", EntryKind::File, 5)]);

    // Iteratively drain settle timers and inject probe responses until the burst stabilizes (one
    // Effect emitted).
    let mut t = now;
    let mut effect_out = None;
    for _ in 0..6 {
        t += SETTLE * 4; // big enough to cover backoff
        while let Some(entry) = e.pop_expired(t) {
            e.step(
                Input::TimerExpired {
                    profile: entry.profile,
                    kind: entry.kind,
                    id: entry.id,
                },
                t,
            );
        }
        let correlation = e.pending_probe_for(pid).expect("Verifying probe in flight");
        let out = e.step(
            Input::ProbeResponse(ProbeResponse {
                owner: pid,
                correlation,
                outcome: ProbeOutcome::SubtreeProven {
                    snapshot: snap_with_entry.clone(),
                    authority: ProofAuthority::Authoritative,
                },
            }),
            t,
        );
        if !out.effects().is_empty() {
            effect_out = Some(out);
            break;
        }
    }
    let out = effect_out.expect("burst stabilized and emitted an Effect");
    assert_eq!(out.effects().len(), 1);
    let effect = &out.effects()[0];
    assert!(
        effect.diff().is_some(),
        "needs_diff Effect carries the Diff"
    );
    let diff = effect.diff().unwrap();
    assert_eq!(diff.created.len(), 1);
    assert_eq!(diff.created[0].segment.as_str(), "new.rs");
}

// ---- Descent integration ----

#[test]
fn seed_burst_descendants_watched_via_first_probe() {
    // Uses an events-incomplete (`STRUCTURE`-only) Profile so the graft can be exercised on the
    // first sample without per-file FDs landing on the File child — CONTENT in the mask would set
    // `has_per_file_fds = true`, doubling the Watch count.
    let mut e = Engine::new();
    let anchor = e.tree.ensure_root("anchor", ResourceRole::User);
    e.tree.set_kind(anchor, ResourceKind::Dir);
    let now = Instant::now();
    let (_sid, pid) = crate::testkit::attach_structure_only(&mut e, anchor, now);
    // Cold-arm Seed: the first Seed probe is in flight directly after `attach_sub` — no settle
    // expiry needed to reach Verifying.
    assert_seed_verifying(&e);
    let correlation = e
        .pending_probe_for(pid)
        .expect("cold-arm Seed Verifying probe in flight at attach");
    // First-probe response with one File and one Dir descendant. Only the Dir gets a Watch op; the
    // File materializes without an FD contribution. The graft (and thus the descendant Watch ops)
    // runs on the first response even though its verdict is Retry.
    let snap = dir_tree_snap(vec![
        ("a.rs", EntryKind::File, 1),
        ("subdir", EntryKind::Dir, 2),
    ]);
    let out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid,
            correlation,
            outcome: ProbeOutcome::SubtreeProven {
                snapshot: snap,
                authority: ProofAuthority::Authoritative,
            },
        }),
        now + SETTLE,
    );
    // 1 Watch op (subdir Dir). File doesn't contribute Watch.
    let watches = out
        .watch_ops
        .iter()
        .filter(|op| matches!(op, WatchOp::Watch { .. }))
        .count();
    assert_eq!(watches, 1);
}

// ---- Probe kind selection ----

#[test]
fn probe_op_for_file_anchor_is_file_kind() {
    let mut e = Engine::new();
    let r = e.tree.ensure_root("log.txt", ResourceRole::User);
    e.tree.set_kind(r, ResourceKind::File);
    let req = SubAttachRequest {
        anchor: SubAttachAnchor::Resource(r),
        identity: ProfileIdentity::new(ScanConfig::builder().build(), MAX_SETTLE, NO_EVENTS),
        params: SubParams::spawn(
            "file-sub".into(),
            empty_program(),
            EffectScope::SubtreeRoot,
            SETTLE,
            false,
        ),
    };
    let now = Instant::now();
    let attach_out = e.step(Input::AttachSub(req), now);
    // Cold-arm Seed: the probe emits at burst construction during `attach_sub`, not on settle-timer
    // expiry.
    let probe_request = attach_out.probe_ops().iter().find_map(|op| match op {
        ProbeOp::Probe { request } => Some(request.clone()),
        _ => None,
    });
    assert!(
        matches!(probe_request, Some(ProbeRequest::AnchorFile { .. })),
        "File-anchored Profile's cold-arm seed burst must emit ProbeRequest::AnchorFile \
         at attach",
    );
    let _ = e.cancel_all_in_flight_probes();
}

// ---- on_watch_op_rejected ----

#[test]
fn watch_op_rejected_clamps_watch_demand_to_zero() {
    // Build a Resource with watch_demand=2 (multi-Profile co-located). Inject WatchOpRejected.
    // Expect watch_demand → 0, Unwatch emitted, Diagnostic.
    let mut e = Engine::new();
    let r = e.tree.ensure_root("x", ResourceRole::User);
    e.tree.set_kind(r, ResourceKind::Dir);
    let mut out = StepOutput::default();
    let pid = specter_core::ProfileId::default();
    crate::refcounts::add_watch(
        &mut e.tree,
        r,
        specter_core::ContribKey::ProfileAnchor(pid),
        NO_EVENTS,
        &mut out,
    );
    crate::refcounts::add_watch(
        &mut e.tree,
        r,
        specter_core::ContribKey::ProfileParent(pid),
        NO_EVENTS,
        &mut out,
    );
    assert_eq!(e.tree.get(r).unwrap().watch_demand(), 2);

    let result = e.step(
        Input::WatchOpRejected {
            resource: r,
            failure: specter_core::WatchFailure::Pressure { errno: 24 },
        },
        Instant::now(),
    );

    assert_eq!(e.tree.get(r).unwrap().watch_demand(), 0);
    assert!(
        result
            .watch_ops
            .iter()
            .any(|op| matches!(op, WatchOp::Unwatch { .. }))
    );
    assert!(result.diagnostics.iter().any(|d| matches!(
        d,
        Diagnostic::WatchOpRejected {
            failure: specter_core::WatchFailure::Pressure { errno: 24 },
            ..
        }
    )),);
}

#[test]
fn watch_op_rejected_already_unwatched_emits_diagnostic_only() {
    let mut e = Engine::new();
    let r = e.tree.ensure_root("x", ResourceRole::User);
    let result = e.step(
        Input::WatchOpRejected {
            resource: r,
            failure: specter_core::WatchFailure::Pressure { errno: 24 },
        },
        Instant::now(),
    );
    assert!(result.watch_ops.is_empty());
    assert!(
        result
            .diagnostics
            .iter()
            .any(|d| matches!(d, Diagnostic::WatchOpRejected { .. }))
    );
}

// ---- P11.0: WatchOpRejected purges descents at the rejected resource ----

#[test]
fn watch_op_rejected_purges_pending_descent_at_rejected_prefix() {
    // Set up: pre-existing /foo, attach `/foo/bar` (descent at /foo).
    let mut e = Engine::new();
    let foo = e
        .tree
        .ensure_path(&[FS_ROOT_SEGMENT, "foo"], ResourceRole::User)
        .expect("non-empty fixture");
    e.tree.set_kind(foo, ResourceKind::Dir);
    let req = SubAttachRequest::for_anchor(
        "guard".into(),
        SubAttachAnchor::Path(std::path::PathBuf::from("/foo/bar")),
        ScanConfig::builder().recursive(true).build(),
        MAX_SETTLE,
        SETTLE,
        empty_program(),
        EffectScope::SubtreeRoot,
        NO_EVENTS,
        false,
    );
    let _ = e.step(Input::AttachSub(req), Instant::now());
    let pid = {
        let mut iter = e.profiles.iter();
        iter.next().expect("profile exists").0
    };
    assert!(e.descent_state(pid).is_some());
    let initial_corr = e.pending_probe_for(pid).expect("first probe in flight");
    let initial_demand = e.tree.get(foo).unwrap().watch_demand();
    assert_eq!(initial_demand, 1);

    // Inject WatchOpRejected (e.g., EMFILE) for the descent prefix.
    let result = e.step(
        Input::WatchOpRejected {
            resource: foo,
            failure: specter_core::WatchFailure::Pressure { errno: 24 },
        },
        Instant::now(),
    );

    // The clamp zeroed watch_demand; the descent has been purged.
    assert_eq!(e.tree.get(foo).unwrap().watch_demand(), 0);
    assert!(
        e.descent_state(pid).is_none(),
        "descent purged on rejection",
    );

    // A Cancel for the in-flight probe was emitted.
    assert!(
        result
            .probe_ops()
            .iter()
            .any(|op| matches!(op, ProbeOp::Cancel { owner: profile} if *profile == pid)),
        "ProbeOp::Cancel emitted for the in-flight descent probe",
    );

    // ProfileClaimPurged{DescentPrefix} surfaces (in addition to WatchOpRejected).
    assert!(
        result
            .diagnostics
            .iter()
            .any(|d| matches!(d, Diagnostic::ProfileClaimPurged {
                profile, claim, resource, failure
            } if *profile == pid
                && *claim == ClaimKind::DescentPrefix
                && *resource == foo
                && *failure == specter_core::WatchFailure::Pressure { errno: 24 })),
        "ProfileClaimPurged{{DescentPrefix}} diagnostic emitted",
    );

    // Late `ProbeResponse` for the cancelled correlation arrives — must be silently discarded
    // (descent removed, correlation no longer matches anything).
    let late = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid,
            correlation: initial_corr,
            outcome: ProbeOutcome::Vanished,
        }),
        Instant::now(),
    );
    // Stale-response diagnostic, no panic.
    assert!(
        late.diagnostics
            .iter()
            .any(|d| matches!(d, Diagnostic::StaleProbeResponse { .. })),
    );
}

#[test]
fn watch_op_rejected_for_anchored_profile_emits_anchor_claim_purged() {
    // Materialized Profile, WatchOpRejected at its anchor — emits ProfileClaimPurged{Anchor} +
    // WatchOpRejected.
    let (mut e, pid, _sid, r, _now) = engine_with_attached_sub();
    let result = e.step(
        Input::WatchOpRejected {
            resource: r,
            failure: specter_core::WatchFailure::Pressure { errno: 24 },
        },
        Instant::now(),
    );
    assert!(
        result
            .diagnostics
            .iter()
            .any(|d| matches!(d, Diagnostic::ProfileClaimPurged {
            profile, claim, resource, ..
        } if *profile == pid && *claim == ClaimKind::Anchor && *resource == r)),
        "ProfileClaimPurged{{Anchor}} emitted for anchored Profile",
    );
    assert!(
        result
            .diagnostics
            .iter()
            .any(|d| matches!(d, Diagnostic::WatchOpRejected { .. })),
    );
    // Anchor claim cleared.
    assert_eq!(
        e.profiles.get(pid).unwrap().anchor_claim(),
        AnchorClaim::None,
        "anchor_claim cleared by purge",
    );
}

#[test]
fn watch_op_rejected_purges_multiple_descents_at_same_prefix() {
    // Two Profiles share a descent prefix (e.g., two Subs anchored at siblings under the same
    // scaffold). WatchOpRejected purges both.
    let mut e = Engine::new();
    let foo = e
        .tree
        .ensure_path(&[FS_ROOT_SEGMENT, "foo"], ResourceRole::User)
        .expect("non-empty fixture");
    e.tree.set_kind(foo, ResourceKind::Dir);
    let req_a = SubAttachRequest::for_anchor(
        "a".into(),
        SubAttachAnchor::Path(std::path::PathBuf::from("/foo/sib_a")),
        ScanConfig::builder().recursive(true).build(),
        MAX_SETTLE,
        SETTLE,
        empty_program(),
        EffectScope::SubtreeRoot,
        NO_EVENTS,
        false,
    );
    let req_b = SubAttachRequest::for_anchor(
        "b".into(),
        SubAttachAnchor::Path(std::path::PathBuf::from("/foo/sib_b")),
        ScanConfig::builder().recursive(true).build(),
        MAX_SETTLE,
        SETTLE,
        empty_program(),
        EffectScope::SubtreeRoot,
        NO_EVENTS,
        false,
    );
    let attach_out = e.step(Input::AttachSub(req_a), Instant::now());
    let sid_a =
        specter_core::testkit::first_attached_sub(&attach_out).expect("attach_sub succeeded");
    let attach_out = e.step(Input::AttachSub(req_b), Instant::now());
    let sid_b =
        specter_core::testkit::first_attached_sub(&attach_out).expect("attach_sub succeeded");
    let pid_a = e.subs.get(sid_a).unwrap().profile();
    let pid_b = e.subs.get(sid_b).unwrap().profile();
    // Both descents at /foo (different anchors).
    assert!(e.descent_state(pid_a).is_some());
    assert!(e.descent_state(pid_b).is_some());
    assert_eq!(e.tree.get(foo).unwrap().watch_demand(), 2);

    let result = e.step(
        Input::WatchOpRejected {
            resource: foo,
            failure: specter_core::WatchFailure::Pressure { errno: 24 },
        },
        Instant::now(),
    );

    assert!(e.descent_state(pid_a).is_none());
    assert!(e.descent_state(pid_b).is_none());
    let purged_count = result
        .diagnostics
        .iter()
        .filter(|d| {
            matches!(
                d,
                Diagnostic::ProfileClaimPurged {
                    claim: ClaimKind::DescentPrefix,
                    ..
                }
            )
        })
        .count();
    assert_eq!(
        purged_count, 2,
        "one ProfileClaimPurged{{DescentPrefix}} per descent",
    );
}

// ---- SensorOverflow reseeds in-scope Profiles ----

#[test]
fn sensor_overflow_global_idle_reseeds_to_active_seed() {
    // Idle Profile (post-`complete_seed_burst`): an overflow drives a direct `start_seed_burst`
    // call; the Profile transitions to `Active(Seed)` Batching-first — no probe yet, a fresh settle
    // window opens (the probe emits one settle expiry later).
    let (mut e, pid, _sid, _root, _now) = engine_with_attached_sub();
    complete_seed_burst(&mut e, pid);
    assert!(matches!(
        e.profiles.get(pid).unwrap().state(),
        ProfileState::Idle
    ));

    let out = e.step(
        Input::SensorOverflow {
            scope: OverflowScope::Global,
        },
        Instant::now(),
    );

    let burst = match e.profiles.get(pid).unwrap().state() {
        ProfileState::Active(ActiveBurst::PreFire(pre), _) => pre,
        s => panic!("expected Active(Seed) after overflow; got {s:?}"),
    };
    assert_eq!(burst.intent, BurstIntent::Seed);
    assert!(matches!(burst.phase, PreFirePhase::Verifying { .. }));
    assert!(
        burst.dirty.is_empty(),
        "reseed starts a fresh Seed quiescence sequence (cold-arm Verifying-first)",
    );
    assert!(
        e.pending_probe_for(pid).is_some(),
        "cold-arm Seed reseed: probe armed at burst construction",
    );
    assert!(
        out.diagnostics.iter().any(|d| matches!(
            d,
            Diagnostic::SensorOverflow {
                scope: OverflowScope::Global
            }
        )),
        "Diagnostic::SensorOverflow{{Global}} emitted exactly once per overflow input",
    );
    let _ = e.cancel_all_in_flight_probes();
}

#[test]
fn sensor_overflow_active_standard_transitions_to_active_seed() {
    // Active(Standard) Profile: an overflow `finish_burst_to_idle` + `start_seed_burst` round-trip
    // transitions the burst to `Active(Seed)`. The Standard burst's `dirty` provenance and
    // quiescence prior are discarded — the seed re-baselines.
    let (mut e, pid, _sid, root, _now) = engine_with_attached_sub();
    complete_seed_burst(&mut e, pid);
    let now = Instant::now();
    e.step(
        Input::FsEvent {
            resource: root,
            event: FsEvent::ContentChanged,
        },
        now,
    );
    // Now in Active(Standard) Batching.
    let burst = match e.profiles.get(pid).unwrap().state() {
        ProfileState::Active(ActiveBurst::PreFire(pre), _) => pre,
        s => panic!("expected Active(Standard) after FsEvent; got {s:?}"),
    };
    assert_eq!(burst.intent, BurstIntent::Standard);
    assert!(matches!(burst.phase, PreFirePhase::Batching { .. }));

    let out = e.step(
        Input::SensorOverflow {
            scope: OverflowScope::Global,
        },
        now,
    );

    let burst = match e.profiles.get(pid).unwrap().state() {
        ProfileState::Active(ActiveBurst::PreFire(pre), _) => pre,
        s => panic!("expected Active(Seed) after overflow; got {s:?}"),
    };
    assert_eq!(
        burst.intent,
        BurstIntent::Seed,
        "overflow abandoned the Standard burst and re-seeded",
    );
    assert!(matches!(burst.phase, PreFirePhase::Verifying { .. }));
    assert!(
        burst.dirty.is_empty(),
        "seed burst starts with an empty dirty set and a fresh quiescence \
         sequence — Standard's accumulators discarded",
    );
    assert!(out.diagnostics.iter().any(|d| matches!(
        d,
        Diagnostic::SensorOverflow {
            scope: OverflowScope::Global
        }
    )),);
    let _ = e.cancel_all_in_flight_probes();
}

/// Overflow over an armed verify slot disarms rather than drops it: overflow on a *genuinely armed*
/// `Active(PreFire(Verifying))` — the verify slot is in flight and was NOT pre-consumed — must NOT
/// panic. Dropping the armed slot through `finish_burst_to_idle` would trip `ProbeSlot`'s Drop
/// tripwire, so the reseed must disarm it first. Under the cold-arm Verifying-first contract, the
/// genuinely-armed Verifying reproduction state is reached at attach (the probe is armed at burst
/// construction, never pre-consumed). Reseed (no Reap): disarm-only via `take_owner_probe` (no wire
/// `Cancel`), then `start_seed_burst` arms a fresh cold-Verifying — one fresh `Probe` emits this
/// step. The guard is that overflow over the armed slot disarms rather than drops it; owner-scoped,
/// exactly one `Probe`, zero `Cancel`.
#[test]
fn sensor_overflow_armed_verifying_reseeds_no_cancel() {
    let (mut e, pid, _sid, _root, _) = engine_with_attached_sub();
    // Cold-arm Seed: the verify probe is in flight directly after attach. Asserting it here is the
    // whole point — a pre-consumed slot would not reproduce the armed-slot drop hazard.
    assert_seed_verifying(&e);
    let burst = match e.profiles.get(pid).unwrap().state() {
        ProfileState::Active(ActiveBurst::PreFire(pre), _) => pre,
        s => panic!("fixture: expected Active(PreFire(Verifying)); got {s:?}"),
    };
    assert!(matches!(burst.phase, PreFirePhase::Verifying { .. }));
    assert!(
        e.pending_probe_for(pid).is_some(),
        "fixture: Verifying slot genuinely armed (NOT pre-consumed)",
    );

    let out = e.step(
        Input::SensorOverflow {
            scope: OverflowScope::Global,
        },
        Instant::now(),
    );

    // (b) Owner-scoped probe ops: one Probe (cold-arm reseed emits the fresh cold walk), zero
    // Cancel (reseed disarms the engine slot only via take_owner_probe — no wire Cancel).
    let owner = pid;
    let probes = out
        .probe_ops()
        .iter()
        .filter(|op| op.owner() == owner && matches!(op, ProbeOp::Probe { .. }))
        .count();
    let cancels = out
        .probe_ops()
        .iter()
        .filter(|op| op.owner() == owner && matches!(op, ProbeOp::Cancel { .. }))
        .count();
    assert_eq!(
        probes, 1,
        "cold-arm Seed reseed emits the fresh cold walk Probe"
    );
    assert_eq!(
        cancels, 0,
        "reseed disarms the engine slot only — no wire Cancel"
    );

    // (c) Profile back in Active(PreFire(Verifying)) with Seed intent — a fresh cold-arm quiescence
    // sequence. Reaching here without tripping ProbeSlot's Drop tripwire is the armed-slot-drop
    // guard.
    let burst = match e.profiles.get(pid).unwrap().state() {
        ProfileState::Active(ActiveBurst::PreFire(pre), _) => pre,
        s => panic!("expected Active(Seed) after overflow; got {s:?}"),
    };
    assert_eq!(burst.intent, BurstIntent::Seed);
    assert!(matches!(burst.phase, PreFirePhase::Verifying { .. }));
    assert!(
        burst.dirty.is_empty(),
        "reseed starts a fresh Seed quiescence sequence",
    );
    let _ = e.cancel_all_in_flight_probes();
}

/// Armed-slot overflow, Rebasing variant: same as above but on a *genuinely armed*
/// `Active(PostFire(Rebasing))`. The Rebasing slot is the post-effect rebase probe minted by
/// `transition_to_rebasing` — armed, never pre-consumed. Overflow must reseed without panicking,
/// disarming the slot only; the superseding Seed burst is Batching-first. Owner-scoped: zero
/// `Probe`, zero `Cancel`.
#[test]
fn sensor_overflow_armed_rebasing_reseeds_no_cancel() {
    let (mut e, pid, sid, root, _now0) = engine_with_attached_sub();
    let now = Instant::now();
    let stable_out = drive_to_first_effect(&mut e, pid, root, now);
    let key = stable_out.effects()[0].key();
    // EffectComplete::Ok drives Awaiting → Rebasing directly (probe-first), where
    // transition_to_rebasing minted a fresh probe in the same step.
    let _ = e.step(
        Input::EffectComplete(EffectCompletion {
            sub: sid,
            key,
            outcome: EffectOutcome::Ok,
        }),
        now + SETTLE * 3,
    );
    let burst = match e.profiles.get(pid).unwrap().state() {
        ProfileState::Active(ActiveBurst::PostFire(post), _) => post,
        s => panic!("fixture: expected Active(PostFire(Rebasing)); got {s:?}"),
    };
    assert!(matches!(burst.phase, PostFirePhase::Rebasing(_)));
    assert!(
        e.pending_probe_for(pid).is_some(),
        "fixture: Rebasing slot genuinely armed (NOT pre-consumed)",
    );

    let out = e.step(
        Input::SensorOverflow {
            scope: OverflowScope::Global,
        },
        now + SETTLE * 4,
    );

    let owner = pid;
    let probes = out
        .probe_ops()
        .iter()
        .filter(|op| op.owner() == owner && matches!(op, ProbeOp::Probe { .. }))
        .count();
    let cancels = out
        .probe_ops()
        .iter()
        .filter(|op| op.owner() == owner && matches!(op, ProbeOp::Cancel { .. }))
        .count();
    assert_eq!(
        probes, 1,
        "cold-arm Seed reseed emits the fresh cold walk Probe"
    );
    assert_eq!(
        cancels, 0,
        "reseed disarms the engine slot only — no wire Cancel"
    );

    // Reseed re-enters Active(PreFire(Verifying)) with Seed intent — the prior PostFire(Rebasing)
    // burst was abandoned, a fresh quiescence sequence opened (cold-arm Verifying-first). Reaching
    // here without tripping ProbeSlot's Drop tripwire is the armed-slot-drop guard.
    let burst = match e.profiles.get(pid).unwrap().state() {
        ProfileState::Active(ActiveBurst::PreFire(pre), _) => pre,
        s => panic!("expected Active(Seed) after overflow; got {s:?}"),
    };
    assert_eq!(burst.intent, BurstIntent::Seed);
    assert!(matches!(burst.phase, PreFirePhase::Verifying { .. }));
    assert!(
        burst.dirty.is_empty(),
        "reseed starts a fresh Seed quiescence sequence",
    );
    let _ = e.cancel_all_in_flight_probes();
}

/// Armed-slot overflow, reap arm: overflow on a *genuinely armed* `Active(PreFire(Verifying))`
/// whose `BurstFinish` is `Reap` (the last Sub was detached mid-burst). Here `will_reap == true`,
/// so the arm emits the wire `Cancel` via `cancel_owner_probe` (no superseding submit follows —
/// `start_seed_burst` no-ops on the detached Profile), then `finish_burst_to_idle` reaps the
/// Profile. Owner-scoped: exactly one `Cancel`, zero `Probe`; Profile gone.
#[test]
fn sensor_overflow_armed_verifying_reap_emits_cancel_only() {
    let (mut e, pid, sid, _root, _) = engine_with_attached_sub();
    // A Seed burst is Batching-first; expire its settle timer to reach the genuinely-armed
    // Active(Seed, Verifying) reproduction state.
    assert_seed_verifying(&e);
    assert!(
        e.pending_probe_for(pid).is_some(),
        "fixture: Verifying slot genuinely armed (NOT pre-consumed)",
    );
    // Detach the sole Sub mid-burst → BurstFinish flips to Reap.
    let _ = e.step(Input::DetachSub(sid), Instant::now());
    assert!(
        matches!(
            e.profiles.get(pid).unwrap().state().burst_finish(),
            Some(BurstFinish::Reap)
        ),
        "fixture: burst marked Reap by detaching the last Sub",
    );

    let out = e.step(
        Input::SensorOverflow {
            scope: OverflowScope::Global,
        },
        Instant::now(),
    );

    // (b) Owner-scoped: exactly one Cancel, zero Probe. start_seed_burst no-ops on the now-detached
    // Profile, so no fresh submit follows — the wire Cancel spares the worker a doomed walk.
    let owner = pid;
    let probes = out
        .probe_ops()
        .iter()
        .filter(|op| op.owner() == owner && matches!(op, ProbeOp::Probe { .. }))
        .count();
    let cancels = out
        .probe_ops()
        .iter()
        .filter(|op| op.owner() == owner && matches!(op, ProbeOp::Cancel { .. }))
        .count();
    assert_eq!(
        cancels, 1,
        "reap arm emits exactly one Cancel for the owner"
    );
    assert_eq!(probes, 0, "reap arm emits NO Probe (Profile detached)");

    // (c) Profile reaped.
    assert!(
        e.profiles.get(pid).is_none(),
        "reap arm tears the Profile down on overflow",
    );
    let _ = e.cancel_all_in_flight_probes();
}

#[test]
fn sensor_overflow_pending_descent_latches_and_repays() {
    // A Pending descent whose probe is still in flight when an overflow lands. The overflow cannot
    // be dropped: the in-flight walk may predate the overflow window, so its response cannot
    // witness the edges the kernel lost. `on_descent_event` latches a re-probe-owed debt instead —
    // no second probe at overflow time, the armed slot untouched — and the descent's own response
    // dispatch repays the debt with a fresh probe that reads the post-overflow tree. Pins both
    // halves: the latch (overflow step) and the repay (response step).
    let mut e = Engine::new();
    let req = SubAttachRequest::for_anchor(
        "guard".into(),
        SubAttachAnchor::Path(std::path::PathBuf::from("/missing/anchor")),
        ScanConfig::builder().recursive(true).build(),
        MAX_SETTLE,
        SETTLE,
        empty_program(),
        EffectScope::SubtreeRoot,
        NO_EVENTS,
        false,
    );
    let _ = e.step(Input::AttachSub(req), Instant::now());
    let pid = {
        let mut iter = e.profiles.iter();
        iter.next().expect("profile exists").0
    };
    assert!(
        e.descent_state(pid).is_some(),
        "fixture: profile is in Pending(_)",
    );
    let in_flight_corr = e
        .pending_probe_for(pid)
        .expect("fixture: descent probe in flight");

    let pre_prefix = e.descent_state(pid).unwrap().current_prefix();
    let out = e.step(
        Input::SensorOverflow {
            scope: OverflowScope::Global,
        },
        Instant::now(),
    );

    let descent = e
        .descent_state(pid)
        .expect("Pending preserved across overflow");
    assert_eq!(
        descent.current_prefix(),
        pre_prefix,
        "descent position preserved across overflow",
    );
    assert!(
        !descent.witnessed(),
        "overflow never writes the appearance witness — dropped events prove churn \
         somewhere in scope, not that the awaited segment appeared; the in-flight \
         probe's own observations carry whatever witness is due",
    );
    assert!(
        !out.probe_ops()
            .iter()
            .any(|op| matches!(op, ProbeOp::Probe { .. })),
        "overflow latches the debt rather than emitting a second probe",
    );
    assert_eq!(
        e.pending_probe_for(pid),
        Some(in_flight_corr),
        "in-flight correlation preserved across overflow",
    );
    assert!(
        out.diagnostics
            .iter()
            .any(|d| matches!(d, Diagnostic::SensorOverflow { .. })),
        "diagnostic still emitted regardless of per-Profile dispatch",
    );

    // The in-flight probe's (pre-overflow) response lands: the awaited segment is still absent, so
    // the descent parks. Without the latch the dropped overflow would wedge here — the segment's
    // creation edge is gone and nothing would re-probe. The latch repays it: a fresh probe that
    // postdates the overflow is emitted in this very dispatch step.
    let out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid,
            correlation: in_flight_corr,
            outcome: ProbeOutcome::SegmentObserved { kind: None },
        }),
        Instant::now(),
    );
    assert!(
        e.descent_state(pid).is_some(),
        "descent still live after the stale park",
    );
    let repay_corr = e
        .pending_probe_for(pid)
        .expect("re-probe-owed debt repaid: a fresh descent probe is in flight");
    assert_ne!(
        repay_corr, in_flight_corr,
        "the repay probe carries a fresh correlation that postdates the overflow",
    );
    assert!(
        out.probe_ops()
            .iter()
            .any(|op| matches!(op, ProbeOp::Probe { request } if request.owner() == pid)),
        "the repay probe is emitted in the response-dispatch step",
    );
    let _ = e.cancel_all_in_flight_probes();
}

#[test]
fn sensor_overflow_pending_profile_reprobes() {
    // Pending(_) Profile with a DISARMED descent slot (a prior probe found the next segment absent
    // / failed, so the descent is waiting on an IN_CREATE): overflow re-probes the current prefix.
    // Without the re-probe, an IN_CREATE lost to the unreliable window wedges the descent forever —
    // the stall this pins against.
    let mut e = Engine::new();
    let a = e
        .tree
        .ensure_path(&[FS_ROOT_SEGMENT, "a"], ResourceRole::User)
        .expect("non-empty fixture");
    e.tree.set_kind(a, ResourceKind::Dir);
    let req = SubAttachRequest::for_anchor(
        "guard".into(),
        SubAttachAnchor::Path(std::path::PathBuf::from("/a/b")),
        ScanConfig::builder().recursive(true).build(),
        MAX_SETTLE,
        SETTLE,
        empty_program(),
        EffectScope::SubtreeRoot,
        NO_EVENTS,
        false,
    );
    let _ = e.step(Input::AttachSub(req), Instant::now());
    let pid = {
        let mut iter = e.profiles.iter();
        iter.next().expect("profile exists").0
    };
    let corr = e
        .pending_probe_for(pid)
        .expect("fixture: descent probe in flight at /a");

    // Failed response: descent retains state (current_prefix = /a) but the slot disarms — the exact
    // "awaiting an event the kernel may have dropped" shape the overflow re-probe exists for.
    let _ = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid,
            correlation: corr,
            outcome: ProbeOutcome::Failed(ProbeFailure::Anchor { errno: 13 }),
        }),
        Instant::now(),
    );
    assert!(
        e.pending_probe_for(pid).is_none(),
        "fixture: slot disarmed after Failed response",
    );
    assert!(e.descent_state(pid).is_some(), "fixture: still descending");

    let out = e.step(
        Input::SensorOverflow {
            scope: OverflowScope::Global,
        },
        Instant::now(),
    );

    let a_path = e.tree.path_of(a).expect("a path resolves");
    assert!(
        out.probe_ops().iter().any(|op| {
            matches!(
                op,
                ProbeOp::Probe { request }
                    if request.owner() == pid
                        && *request.target_path() == *a_path,
            )
        }),
        "fresh descent probe at the current prefix /a",
    );
    assert!(
        e.pending_probe_for(pid).is_some(),
        "descent slot re-armed post-overflow",
    );
    let _ = e.cancel_all_in_flight_probes();
}

#[test]
fn sensor_overflow_resource_scope_filters_profiles() {
    // OverflowScope::Resource(r) reseeds only Profiles whose anchor lies in the subtree rooted at r
    // — the FSEvents per-stream signal. Set up two siblings under one root; overflow at the first
    // sibling's resource reseeds only the first.
    let mut e = Engine::new();
    let parent = e.tree.ensure_root("parent", ResourceRole::User);
    e.tree.set_kind(parent, ResourceKind::Dir);
    let a = e
        .tree
        .ensure_child(parent, "a", ResourceRole::User)
        .expect("test live parent");
    e.tree.set_kind(a, ResourceKind::Dir);
    let b = e
        .tree
        .ensure_child(parent, "b", ResourceRole::User)
        .expect("test live parent");
    e.tree.set_kind(b, ResourceKind::Dir);
    let now = Instant::now();
    let req_a = SubAttachRequest {
        anchor: SubAttachAnchor::Resource(a),
        identity: ProfileIdentity::new(
            ScanConfig::builder().recursive(true).build(),
            MAX_SETTLE,
            NO_EVENTS,
        ),
        params: SubParams::spawn(
            "sub-a".into(),
            empty_program(),
            EffectScope::SubtreeRoot,
            SETTLE,
            false,
        ),
    };
    let req_b = SubAttachRequest {
        anchor: SubAttachAnchor::Resource(b),
        identity: req_a.identity.clone(),
        params: SubParams {
            name: "sub-b".into(),
            ..req_a.params.clone()
        },
    };
    let attach_out = e.step(Input::AttachSub(req_a), now);
    let sid_a =
        specter_core::testkit::first_attached_sub(&attach_out).expect("attach_sub succeeded");
    let attach_out = e.step(Input::AttachSub(req_b), now);
    let sid_b =
        specter_core::testkit::first_attached_sub(&attach_out).expect("attach_sub succeeded");
    let pid_a = e.subs.get(sid_a).unwrap().profile();
    let pid_b = e.subs.get(sid_b).unwrap().profile();
    complete_seed_burst(&mut e, pid_a);
    complete_seed_burst(&mut e, pid_b);
    assert!(matches!(
        e.profiles.get(pid_a).unwrap().state(),
        ProfileState::Idle
    ));
    assert!(matches!(
        e.profiles.get(pid_b).unwrap().state(),
        ProfileState::Idle
    ));

    // Overflow scoped to `a` — only Profile A reseeds.
    let _ = e.step(
        Input::SensorOverflow {
            scope: OverflowScope::Resource(a),
        },
        Instant::now(),
    );

    assert!(
        matches!(
            e.profiles.get(pid_a).unwrap().state(),
            ProfileState::Active(ActiveBurst::PreFire(pre), _) if pre.intent == BurstIntent::Seed
        ),
        "Profile A (anchor at a) reseeded",
    );
    assert!(
        matches!(e.profiles.get(pid_b).unwrap().state(), ProfileState::Idle),
        "Profile B (anchor at b, sibling of a) untouched",
    );
    let _ = e.cancel_all_in_flight_probes();
}

// ---- P11.0: anchor_claim drives reap correctness ----

#[test]
fn seed_vanished_then_reap_releases_anchor_via_claim() {
    let (mut e, pid, sid, r, _) = engine_with_attached_sub();
    // Anchor watch_demand is 1, anchor_claim is Held.
    assert_eq!(e.tree.get(r).unwrap().watch_demand(), 1);
    assert_eq!(
        e.profiles.get(pid).unwrap().anchor_claim(),
        AnchorClaim::Held,
    );

    // A Seed burst is Batching-first; expire its settle timer so a verify probe is in flight to
    // drive the Vanished response below.
    assert_seed_verifying(&e);

    // Detach the Sub mid-burst → reap_pending = true.
    let _ = e.step(Input::DetachSub(sid), Instant::now());
    assert!(matches!(
        e.profiles.get(pid).unwrap().state().burst_finish(),
        Some(BurstFinish::Reap)
    ));

    // Drive Seed Vanished to fire the reap.
    let correlation = e.pending_probe_for(pid).expect("Verifying probe in flight");
    let out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid,
            correlation,
            outcome: ProbeOutcome::Vanished,
        }),
        Instant::now(),
    );

    // Profile reaped; anchor's contribution released → Unwatch emitted.
    assert!(e.profiles.get(pid).is_none(), "Profile reaped");
    let saw_unwatch = out
        .watch_ops
        .iter()
        .any(|op| matches!(op, WatchOp::Unwatch { resource } if *resource == r));
    assert!(
        saw_unwatch,
        "Unwatch for the anchor emitted on reap (anchor_claim drove the release)",
    );
}

#[test]
fn anchor_terminal_event_clears_anchor_claim() {
    // After on_anchor_terminal_event releases the anchor, a subsequent reap must NOT double-release
    // it (the claim is cleared to None).
    let (mut e, pid, _sid, r, _now) = engine_with_attached_sub();
    complete_seed_burst(&mut e, pid);
    assert_eq!(
        e.profiles.get(pid).unwrap().anchor_claim(),
        AnchorClaim::Held,
    );

    // Inject a Removed event at the anchor: the terminal event releases the anchor's contribution
    // and clears the claim.
    let _ = e.step(
        Input::FsEvent {
            resource: r,
            event: FsEvent::Removed,
        },
        Instant::now(),
    );
    assert_eq!(
        e.profiles.get(pid).unwrap().anchor_claim(),
        AnchorClaim::None,
        "anchor_claim cleared by terminal event",
    );
    assert_eq!(
        e.tree.get(r).unwrap().watch_demand(),
        0,
        "anchor's watch_demand released",
    );
}

// ---- detach_sub ----

#[test]
fn detach_sub_idle_profile_reaps_immediately() {
    let (mut e, pid, sid, r, _now) = engine_with_attached_sub();
    complete_seed_burst(&mut e, pid);
    // Profile is now Idle.
    assert!(matches!(
        e.profiles.get(pid).unwrap().state(),
        ProfileState::Idle,
    ));
    let out = e.step(Input::DetachSub(sid), Instant::now());
    // Profile reaped; anchor unwatched.
    assert!(e.profiles.get(pid).is_none());
    assert!(
        out.watch_ops
            .iter()
            .any(|op| matches!(op, WatchOp::Unwatch { resource } if *resource == r))
    );
    assert!(out.diagnostics.iter().any(|d| matches!(
        d,
        Diagnostic::ProfileReaped {
            via: specter_core::ReapTrigger::Immediate,
            ..
        }
    )));
}

#[test]
fn detach_sub_active_profile_marks_reap_pending() {
    let (mut e, pid, sid, _r, _now) = engine_with_attached_sub();
    // Profile is Active(Seed Verifying) — Seed-burst still in flight.
    let _ = e.step(Input::DetachSub(sid), Instant::now());
    let p = e.profiles.get(pid).expect("profile alive until burst ends");
    assert!(matches!(p.state().burst_finish(), Some(BurstFinish::Reap)));
    assert!(
        e.subs.at(pid).is_empty(),
        "no Subs remain after detaching the sole Sub"
    );
    let _ = e.cancel_all_in_flight_probes();
}

#[test]
fn reap_pending_burst_completion_skips_effects_and_reaps() {
    // Sub on Active(Standard, stable) Profile; detach mid-burst; finish burst — no Effect emitted;
    // Profile reaped.
    let (mut e, pid, sid, root, _now) = engine_with_attached_sub();
    complete_seed_burst(&mut e, pid);

    // Drive Standard burst.
    let t1 = Instant::now();
    e.step(
        Input::FsEvent {
            resource: root,
            event: FsEvent::ContentChanged,
        },
        t1,
    );

    // Detach the Sub mid-burst.
    let _ = e.step(Input::DetachSub(sid), Instant::now());
    assert!(matches!(
        e.profiles.get(pid).unwrap().state().burst_finish(),
        Some(BurstFinish::Reap)
    ));

    // Drain the settle timer to advance to Verifying.
    let t2 = t1 + SETTLE * 2;
    while let Some(entry) = e.pop_expired(t2) {
        e.step(
            Input::TimerExpired {
                profile: entry.profile,
                kind: entry.kind,
                id: entry.id,
            },
            t2,
        );
    }
    let correlation = e.pending_probe_for(pid).expect("Verifying probe in flight");

    // The verify response folds to `Stable(StableReason::Natural)` on the first sample — single
    // dispatch. A reap-pending burst suppresses the Effect and finishes by reaping. No Effect is
    // emitted.
    let out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid,
            correlation,
            outcome: ProbeOutcome::SubtreeProven {
                snapshot: dir_tree_snap(vec![]),
                authority: ProofAuthority::Authoritative,
            },
        }),
        t2,
    );
    assert!(out.effects().is_empty(), "reap_pending suppresses Effect");
    assert!(e.profiles.get(pid).is_none(), "Profile reaped at burst end");
}

#[test]
fn detach_sub_settle_recomputed_when_subs_remain() {
    // Profile with two Subs of different settle; detach the faster one; remaining Sub's settle wins.
    let mut e = Engine::new();
    let r = e.tree.ensure_root("anchor", ResourceRole::User);
    e.tree.set_kind(r, ResourceKind::Dir);
    let now = Instant::now();
    let cfg = ScanConfig::builder().recursive(true).build();
    let attach_out = e.step(
        Input::AttachSub(SubAttachRequest {
            anchor: SubAttachAnchor::Resource(r),
            identity: ProfileIdentity::new(cfg.clone(), MAX_SETTLE, NO_EVENTS),
            params: SubParams::spawn(
                "fast".into(),
                empty_program(),
                EffectScope::SubtreeRoot,
                Duration::from_millis(50),
                false,
            ),
        }),
        now,
    );
    let sid_fast =
        specter_core::testkit::first_attached_sub(&attach_out).expect("attach_sub succeeded");
    let pid = e.subs.get(sid_fast).unwrap().profile();
    let _ = e.step(
        Input::AttachSub(SubAttachRequest {
            anchor: SubAttachAnchor::Resource(r),
            identity: ProfileIdentity::new(cfg, MAX_SETTLE, NO_EVENTS),
            params: SubParams::spawn(
                "slow".into(),
                empty_program(),
                EffectScope::SubtreeRoot,
                Duration::from_millis(200),
                false,
            ),
        }),
        now,
    );
    // Fast Sub's settle wins on attach.
    assert_eq!(
        e.profiles.get(pid).unwrap().settle,
        Duration::from_millis(50)
    );

    // Detach the fast Sub. Remaining settle is the slow one's.
    let _ = e.step(Input::DetachSub(sid_fast), Instant::now());
    assert_eq!(
        e.profiles.get(pid).unwrap().settle,
        Duration::from_millis(200)
    );
    let _ = e.cancel_all_in_flight_probes();
}

// ---- on_config_diff ----

#[test]
fn config_diff_added_only_attaches_subs() {
    let mut e = Engine::new();
    let r = e.tree.ensure_root("anchor", ResourceRole::User);
    e.tree.set_kind(r, ResourceKind::Dir);

    let req = SubAttachRequest {
        anchor: SubAttachAnchor::Resource(r),
        identity: ProfileIdentity::new(ScanConfig::builder().build(), MAX_SETTLE, NO_EVENTS),
        params: SubParams::spawn(
            "added".into(),
            empty_program(),
            EffectScope::SubtreeRoot,
            SETTLE,
            false,
        ),
    };
    let mut diff = specter_core::SubRegistryDiff::default();
    diff.added.push(req);

    let out = e.step(Input::ConfigDiff(diff), Instant::now());
    assert!(
        out.watch_ops
            .iter()
            .any(|op| matches!(op, WatchOp::Watch { .. }))
    );
    // Cold-arm Seed: the attach starts the burst AND emits the cold walk probe at construction.
    assert!(
        out.probe_ops()
            .iter()
            .any(|op| matches!(op, ProbeOp::Probe { .. })),
        "cold-arm Seed burst emits the cold walk Probe at attach",
    );
    let (_pid, p) = e.profiles.iter().next().expect("one Profile attached");
    assert!(
        matches!(
            p.state(),
            ProfileState::Active(
                ActiveBurst::PreFire(PreFireBurst {
                    phase: PreFirePhase::Verifying { .. },
                    intent: BurstIntent::Seed,
                    ..
                }),
                _,
            )
        ),
        "ConfigDiff.added attaches the Sub and starts its cold-arm Verifying-first Seed burst",
    );
    assert_eq!(e.subs().len(), 1);
    let _ = e.cancel_all_in_flight_probes();
}

#[test]
fn config_diff_removed_then_added_atomic() {
    // Engine has Sub A at /anchor; ConfigDiff removes A and adds B (path-based, anchored at /anchor
    // — re-creates the slot if A's detach reaped it).
    let (mut e, pid_a, sid_a, _r, _now) = engine_with_attached_sub();
    complete_seed_burst(&mut e, pid_a);

    // Path-based add — the engine re-materializes if needed.
    let req_b = SubAttachRequest::for_anchor(
        "B".into(),
        SubAttachAnchor::Path(std::path::PathBuf::from("/anchor")),
        ScanConfig::builder().build(), // different config_hash (non-recursive)
        MAX_SETTLE,
        SETTLE,
        empty_program(),
        EffectScope::SubtreeRoot,
        NO_EVENTS,
        false,
    );
    let mut diff = specter_core::SubRegistryDiff::default();
    diff.removed.push(CompactString::from("test-sub"));
    diff.added.push(req_b);

    let out = e.step(Input::ConfigDiff(diff), Instant::now());
    // A reaped (sub registry no longer has it); B added.
    assert!(e.subs().get(sid_a).is_none());
    assert_eq!(e.subs().len(), 1);
    // Single sorted StepOutput; multiple watch_ops merged.
    assert!(!out.watch_ops.is_empty());
    let _ = e.cancel_all_in_flight_probes();
}

/// The name-keyed shim resolves `removed` / `modified_*` against the engine's own registry:
///
/// - a `removed` name the engine never attached emits `Diagnostic::ConfigDiffUnknownSub` — not a
///   silent skip, and not a stale-id `DetachUnknownSub`;
/// - a `modified_params` name the engine never attached degrades to an attach-only retry, narrated
///   by `ConfigDiffRebindFallbackAttach`. A watch whose earlier attach failed (`AttachPathInvalid`)
///   can recover on a later reload through this path rather than being skipped forever.
///
#[test]
fn config_diff_unknown_removed_diagnoses_unknown_modified_retries_as_attach() {
    let mut e = Engine::new();

    let revenant = SubAttachRequest::for_anchor(
        "revenant".into(),
        SubAttachAnchor::Path(std::path::PathBuf::from("/revenant-anchor")),
        ScanConfig::builder().build(),
        MAX_SETTLE,
        SETTLE,
        empty_program(),
        EffectScope::SubtreeRoot,
        NO_EVENTS,
        false,
    );
    let mut subs = specter_core::SubRegistryDiff::default();
    subs.removed.push(CompactString::from("ghost")); // never attached
    subs.modified_params.push(revenant); // never attached ⇒ attach-only fallback

    let out = e.step(Input::ConfigDiff(subs), Instant::now());

    assert!(
        out.diagnostics.iter().any(|d| matches!(
            d,
            Diagnostic::ConfigDiffUnknownSub { name } if name == "ghost"
        )),
        "unresolved `removed` name must emit ConfigDiffUnknownSub; got {:?}",
        out.diagnostics,
    );
    assert!(
        out.diagnostics.iter().any(|d| matches!(
            d,
            Diagnostic::ConfigDiffRebindFallbackAttach { name } if name == "revenant"
        )),
        "modified_params name with no live Sub must narrate \
         ConfigDiffRebindFallbackAttach; got {:?}",
        out.diagnostics,
    );
    assert!(
        !out.diagnostics
            .iter()
            .any(|d| matches!(d, Diagnostic::DetachUnknownSub { .. })),
        "the shim must not route an unknown removed name through the \
         stale-id detach path",
    );
    assert!(
        e.subs().find_by_name("revenant").is_some(),
        "a `modified_params` name the engine never attached retries as \
         a fresh attach (registered, not skipped forever)",
    );
    assert!(
        e.subs().find_by_name("ghost").is_none(),
        "the unknown `removed` name attached nothing",
    );
    let _ = e.cancel_all_in_flight_probes();
}

// ---- emit_effects PerStableFile ----

#[test]
fn per_stable_file_fires_one_effect_per_created_entry() {
    // Profile with PerStableFile Sub; burst stabilizes with 2 created file entries.
    // `DEFAULT_PER_FILE` matches the production default for PerStableFile and carries CONTENT —
    // events_witness_quiescence ⇒ a single Authoritative sample fires.
    let mut e = Engine::new();
    let r = e.tree.ensure_root("anchor", ResourceRole::User);
    e.tree.set_kind(r, ResourceKind::Dir);
    let now = Instant::now();
    let req = SubAttachRequest {
        anchor: SubAttachAnchor::Resource(r),
        identity: ProfileIdentity::new(
            ScanConfig::builder().recursive(true).build(),
            MAX_SETTLE,
            ClassSet::DEFAULT_PER_FILE,
        ),
        params: SubParams::spawn(
            "fmt".into(),
            diff_program(),
            EffectScope::PerStableFile,
            SETTLE,
            false,
        ),
    };
    let attach_out = e.step(Input::AttachSub(req), now);
    let sid = specter_core::testkit::first_attached_sub(&attach_out).expect("attach_sub succeeded");
    let pid = e.subs.get(sid).unwrap().profile();

    // Complete Seed with empty baseline.
    complete_seed_burst(&mut e, pid);

    // FsEvent → Standard burst.
    let t1 = now + Duration::from_millis(10);
    e.step(
        Input::FsEvent {
            resource: r,
            event: FsEvent::ContentChanged,
        },
        t1,
    );

    // Drain settle.
    let t2 = t1 + SETTLE * 2;
    while let Some(entry) = e.pop_expired(t2) {
        e.step(
            Input::TimerExpired {
                profile: entry.profile,
                kind: entry.kind,
                id: entry.id,
            },
            t2,
        );
    }
    let std_corr = e.pending_probe_for(pid).expect("Verifying probe in flight");

    // Inject an Authoritative response with 2 file entries — the CONTENT-subscribed Profile reaches
    // `EventsReliable` on the single sample, so the fold yields `Stable(Natural)` and fires inline.
    let snap = dir_tree_snap(vec![
        ("a.rs", EntryKind::File, 1),
        ("b.rs", EntryKind::File, 2),
    ]);
    let out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid,
            correlation: std_corr,
            outcome: ProbeOutcome::SubtreeProven {
                snapshot: snap,
                authority: ProofAuthority::Authoritative,
            },
        }),
        t2,
    );
    // Stable; baseline is empty, current has 2 files → diff.created has 2.
    let per_file_effects: Vec<&specter_core::Effect> = out
        .effects()
        .iter()
        .filter(|e| matches!(&e.key(), DedupKey::PerFile { sub, .. } if *sub == sid))
        .collect();
    assert_eq!(per_file_effects.len(), 2, "one Effect per created file");
    for eff in &per_file_effects {
        // Engine carries the unresolved ActionProgram; the resolver runs in the actuator. Assert
        // the template references the diff-derived `${specter.created}` placeholder (the test
        // fixture's `diff_program()`).
        let SpawnBody::Exec(exec) = &eff.program.ops()[0].body() else {
            panic!("expected SpawnBody::Exec");
        };
        assert!(
            exec.argv()
                .iter()
                .any(|a| a.parts().iter().any(|p| matches!(
                    p,
                    specter_core::ArgPart::Placeholder(specter_core::Placeholder::Created)
                ))),
            "diff_program's template references ${{specter.created}}"
        );
        // anchor_path + anchor_kind ⇒ actuator's compute_cwd("anchor", Dir) = "anchor".
        assert_eq!(eff.anchor_path.as_os_str(), "anchor");
        assert_eq!(eff.anchor_kind, specter_core::ResourceKind::Dir);
        // relative() ⇒ SPECTER_RELATIVE_PATH source. The resolver derives SPECTER_PATH =
        // anchor_path.join(relative()) at spawn time; this assertion implicitly pins target_path to
        // "anchor/a.rs" or "anchor/b.rs".
        assert!(
            eff.relative() == "a.rs" || eff.relative() == "b.rs",
            "relative() = {:?}",
            eff.relative(),
        );
        // SPECTER_EVENT_KIND="file" derives from the DedupKey::PerFile variant.
        assert!(matches!(eff.key(), specter_core::DedupKey::PerFile { .. }));
    }
}

#[test]
fn per_stable_file_skips_dir_entries() {
    // Mixed Diff: 1 created File, 1 created Dir, 1 modified Dir. PerStableFile must fire ONE Effect
    // (the File), not three.
    let mut e = Engine::new();
    let r = e.tree.ensure_root("anchor", ResourceRole::User);
    e.tree.set_kind(r, ResourceKind::Dir);
    let now = Instant::now();
    let req = SubAttachRequest {
        anchor: SubAttachAnchor::Resource(r),
        identity: ProfileIdentity::new(
            ScanConfig::builder().recursive(true).build(),
            MAX_SETTLE,
            ClassSet::DEFAULT_PER_FILE,
        ),
        params: SubParams::spawn(
            "fmt".into(),
            diff_program(),
            EffectScope::PerStableFile,
            SETTLE,
            false,
        ),
    };
    let attach_out = e.step(Input::AttachSub(req), now);
    let sid = specter_core::testkit::first_attached_sub(&attach_out).expect("attach_sub succeeded");
    let pid = e.subs.get(sid).unwrap().profile();

    // Seed completes against a snapshot already containing one Dir (`subdir`). After Seed, `subdir`
    // is in the baseline and won't re-appear as `created` later.
    complete_seed_burst_with(
        &mut e,
        pid,
        dir_tree_snap(vec![("subdir", EntryKind::Dir, 10)]),
    );

    // FsEvent → Standard burst.
    let t1 = now + Duration::from_millis(10);
    e.step(
        Input::FsEvent {
            resource: r,
            event: FsEvent::ContentChanged,
        },
        t1,
    );
    let t2 = t1 + SETTLE * 2;
    while let Some(entry) = e.pop_expired(t2) {
        e.step(
            Input::TimerExpired {
                profile: entry.profile,
                kind: entry.kind,
                id: entry.id,
            },
            t2,
        );
    }
    let std_corr = e.pending_probe_for(pid).expect("Verifying probe in flight");

    // Mixed snapshot: subdir (modified — different mtime), newdir (new Dir), main.rs (new File).
    // Diff = created=[newdir, main.rs], modified=[subdir]. Only main.rs should fire.
    let mixed_snap = dir_tree_snap(vec![
        ("main.rs", EntryKind::File, 1),
        ("newdir", EntryKind::Dir, 11),
        // subdir has different mtime ⇒ counted as Modified.
        ("subdir", EntryKind::Dir, 10),
    ]);
    // The verify response folds to `Stable(StableReason::Natural)` on the first sample — single
    // dispatch fires the per-file Effects.
    let out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid,
            correlation: std_corr,
            outcome: ProbeOutcome::SubtreeProven {
                snapshot: mixed_snap,
                authority: ProofAuthority::Authoritative,
            },
        }),
        t2,
    );

    let per_file_effects: Vec<&specter_core::Effect> = out
        .effects()
        .iter()
        .filter(|e| matches!(&e.key(), DedupKey::PerFile { sub, .. } if *sub == sid))
        .collect();
    assert_eq!(
        per_file_effects.len(),
        1,
        "exactly ONE Effect for the File entry; Dir entries skipped"
    );
    // SPECTER_RELATIVE_PATH source.
    assert_eq!(per_file_effects[0].relative(), "main.rs");
}

// ---------- Dedup-hash + drift suppression ----------

/// Drive a complete attach + Seed-Ok + FsEvent + single Standard-Ok response and return the
/// StepOutput that contains the Effect emission. Common harness for SubtreeRoot dedup-hash tests.
///
/// The verify response folds through `quiescence_verdict` to `Stable(StableReason::Natural)` — the
/// CONTENT-subscribed `DEFAULT_EVENTS` mask makes the witness `EventsReliable`, so one
/// Authoritative sample fires on classify-consequence Standard.
fn drive_to_first_effect(
    e: &mut Engine,
    pid: specter_core::ProfileId,
    root: ResourceId,
    now: Instant,
) -> StepOutput {
    // Complete Seed.
    complete_seed_burst(e, pid);
    // Inject FsEvent → Standard burst at root.
    let _ = e.step(
        Input::FsEvent {
            resource: root,
            event: FsEvent::StructureChanged,
        },
        now,
    );
    // Drain settle timer → Verifying.
    let settle_timer = match e.profiles.get(pid).unwrap().state() {
        ProfileState::Active(ActiveBurst::PreFire(pre), _) => match &pre.phase {
            PreFirePhase::Batching { settle_timer } => *settle_timer,
            _ => panic!("expected Batching"),
        },
        _ => panic!("expected Active"),
    };
    let _ = e.step(
        Input::TimerExpired {
            profile: pid,
            kind: TimerKind::Settle,
            id: settle_timer,
        },
        now + SETTLE,
    );
    let correlation = e.pending_probe_for(pid).expect("Verifying probe in flight");
    // Single Authoritative probe response ⇒ fire (Consequence::StandardFire).
    let snap = dir_tree_snap(vec![("a.rs", EntryKind::File, 1)]);
    e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid,
            correlation,
            outcome: ProbeOutcome::SubtreeProven {
                snapshot: snap,
                authority: ProofAuthority::Authoritative,
            },
        }),
        now + SETTLE,
    )
}

#[test]
fn records_fired_subs_after_subtree_effect() {
    let (mut e, pid, sid, root, _now) = engine_with_attached_sub();
    let now = Instant::now();
    let out = drive_to_first_effect(&mut e, pid, root, now);

    // First Effect fires (no prior emission).
    assert_eq!(out.effects().len(), 1, "first Standard-Ok fires Effect");
    // The Subtree fire is recorded on this Sub — the post-emit fire-history flag that gates later
    // B1 suppression.
    assert!(
        e.subs.get(sid).is_some_and(specter_core::Sub::has_fired),
        "post-emit: Subtree fire recorded for this Sub",
    );
}

/// `Effect.target` for a `Subtree`-keyed Effect is the Profile anchor — captured from
/// `Profile.resource` at emit time. The sort-key extractor pulls `target` directly without a
/// `&Engine` lookup; this pins the emission-side capture so a future refactor that drops the
/// `target` assignment surfaces here.
#[test]
fn subtree_effect_target_is_anchor_at_emission() {
    let (mut e, pid, _sid, root, _now) = engine_with_attached_sub();
    let now = Instant::now();
    let out = drive_to_first_effect(&mut e, pid, root, now);

    assert_eq!(out.effects().len(), 1, "Subtree-Ok fires one Effect");
    assert!(
        matches!(&out.effects()[0].key(), DedupKey::Subtree { profile, .. } if *profile == pid),
        "Effect is keyed Subtree at the burst's Profile",
    );
    assert_eq!(
        out.effects()[0].sort_key().1,
        root,
        "Subtree.target is the Profile anchor at emission time",
    );
    assert_eq!(
        e.profiles.get(pid).unwrap().resource(),
        root,
        "anchor is unchanged post-emit (sanity)",
    );
}

#[test]
fn clears_fired_subs_on_effect_complete_failed() {
    let (mut e, pid, sid, root, _now) = engine_with_attached_sub();
    let now = Instant::now();
    let _ = drive_to_first_effect(&mut e, pid, root, now);

    // Confirm the fire-history flag was set.
    assert!(e.subs.any_fired(pid));

    // EffectComplete::Failed clears the dedup-hash entry for this DedupKey.
    let dk = DedupKey::Subtree {
        sub: sid,
        profile: pid,
    };
    let _ = e.step(
        Input::EffectComplete(EffectCompletion {
            sub: sid,
            key: dk,
            outcome: EffectOutcome::Failed(Termination::Exit(1)),
        }),
        now,
    );
    assert!(
        !e.subs.any_fired(pid),
        "Failed Effect clears the suppression entry",
    );
    let _ = e.cancel_all_in_flight_probes();
}

#[test]
fn fresh_seed_without_activity_does_not_fire() {
    // Fresh attach, **no FsEvents witnessed** → silent Seed pin. With an empty `dirty`,
    // `seed_owes_first_fire` is false and `seed_drift_observed` is false (never-fired), so the Seed
    // pins silently (restart-safe: Specter persists no baseline, so a daemon restart over an
    // unchanged tree must not re-fire). This is strictly the no-activity path; the witnessed-activity
    // case (a fresh Seed that *did* see events fires exactly one Effect) is covered by the
    // `fresh_seed_fires::*` tests. `complete_seed_burst` returns the pinning response's StepOutput.
    let (mut e, pid, _sid, _root, _now) = engine_with_attached_sub();
    let out = complete_seed_burst(&mut e, pid);
    assert!(
        out.effects().is_empty(),
        "fresh Seed that witnessed no activity fires no Effect"
    );
}

/// Standard burst with a per-stable-file Sub: drift filter is `None`, PerFile keys still emit per
/// matching diff entry. This pins that the SeedDrift-path narrowing (PerFile Subs skipped on
/// `EmitMode::SeedDrift`) doesn't accidentally skip PerFile emission on the unrelated Standard
/// burst path.
#[test]
fn b3_per_key_filter_does_not_affect_standard_burst_perfile_emission() {
    let mut e = Engine::new();
    let r = e.tree.ensure_root("anchor", ResourceRole::User);
    e.tree.set_kind(r, ResourceKind::Dir);
    let now = Instant::now();
    let req = SubAttachRequest {
        anchor: SubAttachAnchor::Resource(r),
        identity: ProfileIdentity::new(
            ScanConfig::builder().recursive(true).build(),
            MAX_SETTLE,
            ClassSet::CONTENT,
        ),
        params: SubParams::spawn(
            "fmt".into(),
            empty_program(),
            EffectScope::PerStableFile,
            SETTLE,
            false,
        ),
    };
    let _ = e.step(Input::AttachSub(req), now);
    let pid = e.profiles.iter().next().unwrap().0;
    // Seed → Idle (establishes the baseline before the Standard burst below).
    complete_seed_burst(&mut e, pid);

    // Standard burst with a created file → PerFile Effect emits.
    e.step(
        Input::FsEvent {
            resource: r,
            event: FsEvent::ContentChanged,
        },
        now,
    );
    let mut t = now;
    let mut effect_out = None;
    let snap_with_file = dir_tree_snap(vec![("new.rs", EntryKind::File, 5)]);
    for _ in 0..6 {
        t += SETTLE * 4;
        while let Some(entry) = e.pop_expired(t) {
            e.step(
                Input::TimerExpired {
                    profile: entry.profile,
                    kind: entry.kind,
                    id: entry.id,
                },
                t,
            );
        }
        let correlation = e.pending_probe_for(pid).expect("Verifying probe in flight");
        let out = e.step(
            Input::ProbeResponse(ProbeResponse {
                owner: pid,
                correlation,
                outcome: ProbeOutcome::SubtreeProven {
                    snapshot: snap_with_file.clone(),
                    authority: ProofAuthority::Authoritative,
                },
            }),
            t,
        );
        if !out.effects().is_empty() {
            effect_out = Some(out);
            break;
        }
    }
    let out = effect_out.expect("Standard burst stabilised and emitted");
    assert_eq!(
        out.effects().len(),
        1,
        "Standard burst with PerFile Sub fires one Effect for the new file",
    );
}

#[test]
fn has_per_file_fds_is_invariant_for_profile_lifetime() {
    // The events mask folds into `config_hash`, so every Sub on a Profile shares the same events by
    // construction. `has_per_file_fds` is derived once at `Profile::new` from the events mask and
    // never flips during the Profile's lifetime.
    //
    // This test pins the new invariant: a Profile constructed with a mask containing CONTENT (or
    // METADATA) starts with the flag set, and a Sub attaching via the same `(resource,
    // config_hash)` does not change it.
    let mut e = Engine::new();
    let r = e.tree.ensure_root("anchor", ResourceRole::User);
    e.tree.set_kind(r, ResourceKind::Dir);
    let req = SubAttachRequest {
        anchor: SubAttachAnchor::Resource(r),
        identity: ProfileIdentity::new(
            ScanConfig::builder().recursive(true).build(),
            MAX_SETTLE,
            ClassSet::CONTENT,
        ),
        params: SubParams::spawn(
            "formatter".into(),
            empty_program(),
            EffectScope::PerStableFile,
            SETTLE,
            false,
        ),
    };
    let out = e.step(Input::AttachSub(req), Instant::now());
    let sid = specter_core::testkit::first_attached_sub(&out).expect("attach_sub succeeded");
    let pid = e.subs.get(sid).unwrap().profile();
    assert!(
        e.profiles.get(pid).unwrap().has_per_file_fds(),
        "CONTENT-mask Profile has has_per_file_fds = true at construction",
    );

    // A Sub with the same `(resource, max_settle, scan, events)` shares the existing Profile; the
    // flag stays true.
    let req2 = SubAttachRequest {
        anchor: SubAttachAnchor::Resource(r),
        identity: ProfileIdentity::new(
            ScanConfig::builder().recursive(true).build(),
            MAX_SETTLE,
            ClassSet::CONTENT,
        ),
        params: SubParams::spawn(
            "formatter-2".into(),
            empty_program(),
            EffectScope::PerStableFile,
            SETTLE,
            false,
        ),
    };
    let _ = e.step(Input::AttachSub(req2), Instant::now());
    assert!(e.profiles.get(pid).unwrap().has_per_file_fds());

    // Detaching the second Sub leaves the Profile alive (a Sub still remains before detach); the
    // flag still doesn't flip because the Profile's events mask is invariant.
    let _ = e.step(Input::DetachSub(sid), Instant::now());
    assert!(e.profiles.get(pid).unwrap().has_per_file_fds());
    let _ = e.cancel_all_in_flight_probes();
}

#[test]
fn structure_only_profile_has_per_file_fds_false() {
    // Inverse case: a STRUCTURE-only mask leaves `has_per_file_fds` false. The reconciler
    // (`apply_diff_to_tree`) then doesn't bump per-leaf watch_demand for covered files.
    let mut e = Engine::new();
    let r = e.tree.ensure_root("anchor", ResourceRole::User);
    e.tree.set_kind(r, ResourceKind::Dir);
    let req = SubAttachRequest {
        anchor: SubAttachAnchor::Resource(r),
        identity: ProfileIdentity::new(
            ScanConfig::builder().recursive(true).build(),
            MAX_SETTLE,
            ClassSet::STRUCTURE,
        ),
        params: SubParams::spawn(
            "ls-only".into(),
            empty_program(),
            EffectScope::SubtreeRoot,
            SETTLE,
            false,
        ),
    };
    let attach_out = e.step(Input::AttachSub(req), Instant::now());
    let sid = specter_core::testkit::first_attached_sub(&attach_out).expect("attach_sub succeeded");
    let pid = e.subs.get(sid).unwrap().profile();
    assert!(!e.profiles.get(pid).unwrap().has_per_file_fds());
    let _ = e.cancel_all_in_flight_probes();
}

// ---------- Anchor-loss kind-cache invalidation ----------
//
// Per-origin assertions that every anchor-loss dispatch path through `Engine::discard_anchor_state`
// clears the cached `Profile.kind`. The helper unit tests in `claims.rs` pin the contract in
// isolation; these tests pin the integration at every dispatch origin — both intents through each
// merged pre-fire helper (`dispatch_pre_fire_{vanished,failed}`, now intent-parametric), both
// rebase routes (`dispatch_rebase_{vanished,failed}`), and the anchor-terminal event — so the
// kind-clear cannot regress at any one of them without a test failure.

/// Drive a Profile from fresh-attach into `Active(Standard, Verifying)` with
/// `pending_probe.is_some()`. Returns the live correlation.
fn drive_to_standard_verifying(
    e: &mut Engine,
    pid: specter_core::ProfileId,
    root: ResourceId,
    now: Instant,
) -> specter_core::ProbeCorrelation {
    complete_seed_burst(e, pid);
    let _ = e.step(
        Input::FsEvent {
            resource: root,
            event: FsEvent::StructureChanged,
        },
        now,
    );
    let settle_timer = match e.profiles.get(pid).unwrap().state() {
        ProfileState::Active(ActiveBurst::PreFire(pre), _) => match &pre.phase {
            PreFirePhase::Batching { settle_timer } => *settle_timer,
            _ => panic!("expected Standard Batching"),
        },
        _ => panic!("expected Active"),
    };
    let _ = e.step(
        Input::TimerExpired {
            profile: pid,
            kind: TimerKind::Settle,
            id: settle_timer,
        },
        now + SETTLE,
    );
    e.pending_probe_for(pid).expect("Verifying probe in flight")
}

/// Drive into `Active(_, Rebasing)` by completing a Standard burst's stable verdict + Effect →
/// EffectComplete::Ok. Returns the rebase probe correlation so the caller can drive the rebase
/// response.
///
/// The last `EffectComplete::Ok` goes probe-first: `Awaiting → Rebasing` directly, minting the
/// `WholeSubtree` rebase probe in the same step. There is no first `Settling` window before the
/// first sample.
fn drive_to_rebasing(
    e: &mut Engine,
    pid: specter_core::ProfileId,
    sid: SubId,
    root: ResourceId,
    now: Instant,
) -> specter_core::ProbeCorrelation {
    let stable_out = drive_to_first_effect(e, pid, root, now);
    assert_eq!(
        stable_out.effects().len(),
        1,
        "Standard stable verdict fires one Effect; got {:?}",
        stable_out.effects(),
    );
    let key = stable_out.effects()[0].key();
    // EffectComplete::Ok drives Awaiting → Rebasing directly, with the WholeSubtree rebase probe
    // already in flight.
    let _ = e.step(
        Input::EffectComplete(EffectCompletion {
            sub: sid,
            key,
            outcome: EffectOutcome::Ok,
        }),
        now + SETTLE * 3,
    );
    match e.profiles.get(pid).unwrap().state() {
        ProfileState::Active(ActiveBurst::PostFire(post), _) => assert!(
            matches!(post.phase, PostFirePhase::Rebasing(_)),
            "expected Active(PostFire(Rebasing)) after EffectComplete::Ok, got {:?}",
            post.phase,
        ),
        other => panic!("expected Active(PostFire) after EffectComplete::Ok, got {other:?}"),
    }
    e.pending_probe_for(pid)
        .expect("rebase probe in flight after EffectComplete drove Awaiting → Rebasing")
}

#[test]
fn dispatch_pre_fire_vanished_seed_clears_profile_kind() {
    let (mut e, pid, _sid, _r, _) = engine_with_attached_sub();
    assert_eq!(
        e.profiles.get(pid).unwrap().kind(),
        Some(ResourceKind::Dir),
        "fresh attach caches anchor's classified kind",
    );
    // Seed is Batching-first; expire the settle timer to put a verify probe in flight, then answer
    // it Vanished.
    assert_seed_verifying(&e);
    let correlation = e.pending_probe_for(pid).expect("Seed Verifying probe");
    let _ = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid,
            correlation,
            outcome: ProbeOutcome::Vanished,
        }),
        Instant::now(),
    );
    assert!(
        e.profiles.get(pid).unwrap().kind().is_none(),
        "Seed-Vanished must clear the cached anchor kind",
    );
}

#[test]
fn dispatch_pre_fire_failed_seed_clears_profile_kind() {
    let (mut e, pid, _sid, _r, _) = engine_with_attached_sub();
    // Seed is Batching-first; expire the settle timer to put a verify probe in flight, then answer
    // it Failed.
    assert_seed_verifying(&e);
    let correlation = e.pending_probe_for(pid).expect("Seed Verifying probe");
    let _ = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid,
            correlation,
            outcome: ProbeOutcome::Failed(ProbeFailure::Anchor { errno: 5 }),
        }),
        Instant::now(),
    );
    assert!(
        e.profiles.get(pid).unwrap().kind().is_none(),
        "Seed-Failed must clear the cached anchor kind",
    );
}

#[test]
fn dispatch_pre_fire_vanished_standard_clears_profile_kind() {
    let (mut e, pid, _sid, root, now) = engine_with_attached_sub();
    let correlation = drive_to_standard_verifying(&mut e, pid, root, now);
    assert_eq!(
        e.profiles.get(pid).unwrap().kind(),
        Some(ResourceKind::Dir),
        "kind cached pre-dispatch",
    );
    let _ = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid,
            correlation,
            outcome: ProbeOutcome::Vanished,
        }),
        now + SETTLE,
    );
    assert!(
        e.profiles.get(pid).unwrap().kind().is_none(),
        "Standard-Vanished must clear the cached anchor kind",
    );
}

#[test]
fn dispatch_pre_fire_failed_standard_clears_profile_kind() {
    let (mut e, pid, _sid, root, now) = engine_with_attached_sub();
    let correlation = drive_to_standard_verifying(&mut e, pid, root, now);
    let _ = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid,
            correlation,
            outcome: ProbeOutcome::Failed(ProbeFailure::Anchor { errno: 13 }),
        }),
        now + SETTLE,
    );
    assert!(
        e.profiles.get(pid).unwrap().kind().is_none(),
        "Standard-Failed must clear the cached anchor kind",
    );
}

#[test]
fn dispatch_rebase_vanished_clears_profile_kind() {
    let (mut e, pid, sid, root, now) = engine_with_attached_sub();
    let correlation = drive_to_rebasing(&mut e, pid, sid, root, now);
    assert_eq!(
        e.profiles.get(pid).unwrap().kind(),
        Some(ResourceKind::Dir),
        "kind cached pre-rebase",
    );
    let _ = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid,
            correlation,
            outcome: ProbeOutcome::Vanished,
        }),
        now + SETTLE * 4,
    );
    assert!(
        e.profiles.get(pid).unwrap().kind().is_none(),
        "Rebase-Vanished must clear the cached anchor kind",
    );
}

#[test]
fn dispatch_rebase_failed_clears_profile_kind() {
    let (mut e, pid, sid, root, now) = engine_with_attached_sub();
    let correlation = drive_to_rebasing(&mut e, pid, sid, root, now);
    let _ = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid,
            correlation,
            outcome: ProbeOutcome::Failed(ProbeFailure::Anchor { errno: 5 }),
        }),
        now + SETTLE * 4,
    );
    assert!(
        e.profiles.get(pid).unwrap().kind().is_none(),
        "Rebase-Failed must clear the cached anchor kind",
    );
}

// ---- ProbeFailure::Transient fork ----
//
// A `Transient` probe failure (FD / kernel-resource pressure — `EMFILE`, errno 24) is the epistemic
// twin of an `Undischarged` proof: the probe observed nothing, so the engine takes the same
// consequence the verdict floor gives `Undischarged` — retry the window while the deadline holds,
// finish to Idle once it forces — never tearing the anchor down. Only `Anchor`-class failures tear
// the anchor down, pinned by the `*_clears_profile_kind` tests above.

/// (Standard, Transient). FD pressure on a Standard verify re-batches for another window
/// (`!forced`) — the anchor watch and baseline retained, the pressure surfaced — then, once the
/// `BurstDeadline` forces, finishes to Idle with the anchor watch and baseline **still** retained,
/// so the anchor's own next event re-bursts without waiting on the parent (an `Anchor`-class
/// failure would instead tear it down). Pins both Standard-route Transient arms in one lifecycle;
/// the retained baseline is the central regression guard a baseline-less Seed cannot assert.
#[test]
fn standard_transient_retries_then_forced_finishes_retaining_anchor() {
    let (mut e, pid, _sid, root, now) = engine_with_attached_sub();
    let correlation = drive_to_standard_verifying(&mut e, pid, root, now);
    let baseline = baseline_hash(&e, pid);
    assert!(
        baseline.is_some(),
        "the Standard burst inherits the Seed baseline"
    );

    // !forced: re-batch for another window; baseline + anchor watch retained, pressure surfaced.
    let retry_out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid,
            correlation,
            outcome: ProbeOutcome::Failed(ProbeFailure::Transient { errno: 24 }),
        }),
        now + SETTLE,
    );
    match e.profiles.get(pid).unwrap().state() {
        ProfileState::Active(ActiveBurst::PreFire(pre), _) => assert!(
            matches!(pre.phase, PreFirePhase::Batching { .. }),
            "Transient !forced re-batches for another window; got {:?}",
            pre.phase,
        ),
        other => panic!("expected Active(PreFire(Batching)); got {other:?}"),
    }
    assert_eq!(
        baseline_hash(&e, pid),
        baseline,
        "the retry never tears down the baseline",
    );
    assert_eq!(
        e.tree.get(root).unwrap().watch_demand(),
        1,
        "FD pressure is not anchor loss — the anchor watch is retained",
    );
    assert!(
        retry_out.diagnostics.iter().any(|d| matches!(
            d,
            Diagnostic::ProbeFailed { profile, intent, failure }
                if *profile == pid
                    && *intent == BurstIntent::Standard
                    && matches!(failure, ProbeFailure::Transient { errno: 24 }),
        )),
        "the pressure signal is surfaced on the retry window; got {:?}",
        retry_out.diagnostics,
    );

    // forced: the BurstDeadline forces the next window's response to the bounded terminal (set-only
    // while a probe is in flight — the re-verify the settle expiry drove carries forced=true).
    let deadline = now + MAX_SETTLE + Duration::from_millis(1);
    while let Some(entry) = e.pop_expired(deadline) {
        e.step(
            Input::TimerExpired {
                profile: entry.profile,
                kind: entry.kind,
                id: entry.id,
            },
            deadline,
        );
    }
    let correlation = e
        .pending_probe_for(pid)
        .expect("the forced re-verify is in flight");
    let _ = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid,
            correlation,
            outcome: ProbeOutcome::Failed(ProbeFailure::Transient { errno: 24 }),
        }),
        deadline,
    );

    assert!(
        matches!(e.profiles.get(pid).unwrap().state(), ProfileState::Idle),
        "forced Transient is the bounded terminal — the burst finishes",
    );
    assert_eq!(
        baseline_hash(&e, pid),
        baseline,
        "the terminal retains the baseline — FD pressure said nothing about the tree",
    );
    assert_eq!(
        e.tree.get(root).unwrap().watch_demand(),
        1,
        "the terminal retains the anchor watch — recovery is the anchor's own next event",
    );
}

/// (Seed, Transient). A cold Seed under FD pressure re-batches with no baseline established
/// (`!forced`), then — once the `BurstDeadline` forces — finishes to Idle still baseline-less but
/// with the anchor watch retained: the next event at the anchor re-seeds. Pins both Seed-route
/// Transient arms in one lifecycle.
#[test]
fn seed_transient_retries_then_forced_finishes_without_baseline() {
    let (mut e, pid, _sid, root, now) = engine_with_attached_sub();
    // Cold-arm Seed: Verifying-first with a probe in flight, no baseline yet.
    let correlation = e.pending_probe_for(pid).expect("cold Seed Verifying probe");
    assert_eq!(baseline_hash(&e, pid), None, "a fresh Seed has no baseline");

    // !forced: re-batch, still baseline-less, anchor watch retained.
    let retry_out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid,
            correlation,
            outcome: ProbeOutcome::Failed(ProbeFailure::Transient { errno: 24 }),
        }),
        now,
    );
    match e.profiles.get(pid).unwrap().state() {
        ProfileState::Active(ActiveBurst::PreFire(pre), _) => {
            assert_eq!(
                pre.intent,
                BurstIntent::Seed,
                "the Seed intent is preserved across the retry",
            );
            assert!(
                matches!(pre.phase, PreFirePhase::Batching { .. }),
                "Seed Transient !forced re-batches; got {:?}",
                pre.phase,
            );
        }
        other => panic!("expected Active(PreFire(Batching)); got {other:?}"),
    }
    assert_eq!(
        baseline_hash(&e, pid),
        None,
        "the retry establishes no baseline",
    );
    assert!(
        retry_out.diagnostics.iter().any(|d| matches!(
            d,
            Diagnostic::ProbeFailed { profile, intent, .. }
                if *profile == pid && *intent == BurstIntent::Seed,
        )),
        "the Seed retry surfaces the pressure signal; got {:?}",
        retry_out.diagnostics,
    );

    // forced: the BurstDeadline forces the next window's response to terminal.
    let deadline = now + MAX_SETTLE + Duration::from_millis(1);
    while let Some(entry) = e.pop_expired(deadline) {
        e.step(
            Input::TimerExpired {
                profile: entry.profile,
                kind: entry.kind,
                id: entry.id,
            },
            deadline,
        );
    }
    let correlation = e
        .pending_probe_for(pid)
        .expect("the forced re-verify is in flight");
    let _ = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid,
            correlation,
            outcome: ProbeOutcome::Failed(ProbeFailure::Transient { errno: 24 }),
        }),
        deadline,
    );

    assert!(
        matches!(e.profiles.get(pid).unwrap().state(), ProfileState::Idle),
        "forced Transient finishes the Seed burst",
    );
    assert_eq!(
        baseline_hash(&e, pid),
        None,
        "a Seed that never observed pins no baseline",
    );
    assert_eq!(
        e.tree.get(root).unwrap().watch_demand(),
        1,
        "the anchor watch is retained — the anchor's next event re-seeds",
    );
}

/// (Rebase, Transient, `!forced`). FD pressure on a rebase loops back through `Rebasing → Settling`
/// — the post-fire mirror of the Undischarged Retry — bounded by the already-armed `RebaseCeiling`.
#[test]
fn rebase_transient_not_forced_settles() {
    let (mut e, pid, sid, root, now) = engine_with_attached_sub();
    let correlation = drive_to_rebasing(&mut e, pid, sid, root, now);

    let out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid,
            correlation,
            outcome: ProbeOutcome::Failed(ProbeFailure::Transient { errno: 24 }),
        }),
        now + SETTLE * 4,
    );

    match e.profiles.get(pid).unwrap().state() {
        ProfileState::Active(ActiveBurst::PostFire(post), _) => assert!(
            matches!(post.phase, PostFirePhase::Settling { .. }),
            "Transient !forced settle-spaces the next rebase sample; got {:?}",
            post.phase,
        ),
        other => panic!("expected Active(PostFire(Settling)); got {other:?}"),
    }
    assert!(
        out.diagnostics.iter().any(|d| matches!(
            d,
            Diagnostic::ProbeFailed { profile, failure, .. }
                if *profile == pid && matches!(failure, ProbeFailure::Transient { errno: 24 }),
        )),
        "the rebase pressure signal is surfaced; got {:?}",
        out.diagnostics,
    );
}

/// (Rebase, Transient, `forced`). Once the `RebaseCeiling` forces, FD pressure finishes the burst
/// with the prior baseline frozen in place — never rebased blind, mirroring the
/// `RebaseCeilingUnreadable` refusal.
#[test]
fn rebase_transient_forced_finishes_with_baseline_frozen() {
    let (mut e, pid, sid, root, now) = engine_with_attached_sub();
    let correlation = drive_to_rebasing(&mut e, pid, sid, root, now);
    let baseline_before = baseline_hash(&e, pid);

    // Latch the ceiling while the rebase probe is in flight (set-only — the in-flight response
    // carries the terminal as forced=true).
    let ceiling = rebase_ceiling_timer(&e, pid);
    e.step(
        Input::TimerExpired {
            profile: pid,
            kind: TimerKind::RebaseCeiling,
            id: ceiling,
        },
        now + SETTLE * 4,
    );

    let out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid,
            correlation,
            outcome: ProbeOutcome::Failed(ProbeFailure::Transient { errno: 24 }),
        }),
        now + SETTLE * 5,
    );

    assert!(
        matches!(e.profiles.get(pid).unwrap().state(), ProfileState::Idle),
        "forced Transient is the post-fire terminal — the burst finishes",
    );
    assert_eq!(
        baseline_hash(&e, pid),
        baseline_before,
        "the terminal never rebases blind — the prior baseline is frozen",
    );
    assert!(
        out.diagnostics.iter().any(|d| matches!(
            d,
            Diagnostic::ProbeFailed { profile, .. } if *profile == pid,
        )),
        "the forced rebase terminal surfaces the pressure signal; got {:?}",
        out.diagnostics,
    );
}

#[test]
fn finalize_anchor_lost_clears_profile_kind() {
    // Anchor terminal event during a materialised burst.
    let (mut e, pid, _sid, root, _now) = engine_with_attached_sub();
    complete_seed_burst(&mut e, pid);
    assert_eq!(
        e.profiles.get(pid).unwrap().kind(),
        Some(ResourceKind::Dir),
        "kind cached post-Seed-Ok",
    );
    let _ = e.step(
        Input::FsEvent {
            resource: root,
            event: FsEvent::Removed,
        },
        Instant::now(),
    );
    assert!(
        e.profiles.get(pid).unwrap().kind().is_none(),
        "anchor terminal event must clear the cached anchor kind",
    );
}

/// Pin `finalize_anchor_lost`'s ordering invariant: `was_active` is captured BEFORE
/// `discard_anchor_state` runs. Exercises the Active-burst path and asserts the burst is finished
/// to Idle (i.e. the `was_active = true` branch ran). A future helper change that flips `state`
/// mid-helper would otherwise silently break the burst-end pathway.
#[test]
fn finalize_anchor_lost_was_active_pre_helper_ordering() {
    let (mut e, pid, _sid, root, _now) = engine_with_attached_sub();
    complete_seed_burst(&mut e, pid);
    // Re-enter Active by injecting an FsEvent → Standard Batching.
    let _ = e.step(
        Input::FsEvent {
            resource: root,
            event: FsEvent::StructureChanged,
        },
        Instant::now(),
    );
    assert!(
        matches!(
            e.profiles.get(pid).unwrap().state(),
            ProfileState::Active(_, _)
        ),
        "harness pre-condition: Profile is Active",
    );

    let mut out = StepOutput::default();
    e.finalize_anchor_lost(pid, &mut out);

    let p = e.profiles.get(pid).expect("Profile lives");
    assert!(
        matches!(p.state(), ProfileState::Idle),
        "was_active=true ⇒ finish_burst_to_idle ran ⇒ state is Idle; got {:?}",
        p.state(),
    );
    assert!(p.kind().is_none(), "kind cleared by discard_anchor_state");
    assert_eq!(
        p.anchor_claim(),
        AnchorClaim::None,
        "anchor claim released by discard_anchor_state",
    );
}

// ---------- rebase probes the whole subtree; residual is reset ----------

/// The rebase probe's obligation is unconditionally `WholeSubtree`, even when an Awaiting absorb
/// populated `dirty`: the command just mutated the tree, so there is no trustworthy prior to scope
/// a `Chains` walk against — an in-place descendant edit need not bump an ancestor mtime, so a
/// chains/mtime skip would re-clone a stale subtree and certify a false quiet. `dirty` is not a
/// post-fire obligation source; `transition_to_rebasing` clears it at the loop entry, so an
/// Awaiting-absorbed event is folded into the `WholeSubtree` read itself rather than carried as a
/// restart seed.
///
/// Sub uses `ClassSet::CONTENT` so the descendant `ContentChanged` event passes both gates: (1) a
/// per-file FD is wired up by the standard burst's reconcile (`has_per_file_fds = true`), bumping
/// the leaf's `watch_demand` past `on_fs_event`'s zero-gate, and (2) the per-Profile class filter
/// (which sits BEFORE `drive_burst`'s absorb arm) admits the CONTENT-classed event.
#[test]
fn rebasing_probes_whole_subtree_and_resets_awaiting_absorbed_residual() {
    let mut e = Engine::new();
    let root = e.tree.ensure_root("anchor", ResourceRole::User);
    e.tree.set_kind(root, ResourceKind::Dir);
    let now = Instant::now();
    let req = SubAttachRequest {
        anchor: SubAttachAnchor::Resource(root),
        identity: ProfileIdentity::new(
            ScanConfig::builder().recursive(true).build(),
            MAX_SETTLE,
            ClassSet::CONTENT,
        ),
        params: SubParams::spawn(
            "test-sub".into(),
            empty_program(),
            EffectScope::SubtreeRoot,
            SETTLE,
            false,
        ),
    };
    let attach_out = e.step(Input::AttachSub(req), now);
    let sid = specter_core::testkit::first_attached_sub(&attach_out).expect("attach_sub succeeded");
    let pid = e.subs.get(sid).unwrap().profile();

    let stable_out = drive_to_first_effect(&mut e, pid, root, now);
    assert_eq!(stable_out.effects().len(), 1, "stable verdict fires Effect");
    let key = stable_out.effects()[0].key();

    // Look up the descendant the standard burst's reconcile created. `drive_to_first_effect` ships
    // `[("a.rs", File, 1)]` as the probe response; the engine's graft creates an `a.rs` Resource
    // under root and bumps its watch_demand (per-file FD) because the Profile carries CONTENT in
    // its events_union.
    let descendant = e
        .tree
        .lookup(Some(root), "a.rs")
        .expect("standard burst's reconcile created a.rs");
    assert!(
        e.tree.get(descendant).is_some_and(|r| r.watch_demand() > 0),
        "per-file FD must be wired up for the descendant — otherwise \
         the ContentChanged event drops at on_fs_event's watch_demand gate \
         before reaching the absorb arm",
    );

    // Inject an FsEvent during Awaiting → absorb arm. `ContentChanged` is the in-place content-edit
    // class — the same FsEvent kqueue emits for a `write(2)` against a per-file FD, which is the
    // carve-out scenario this test pins (the parent dir's mtime is unchanged).
    let absorb_out = e.step(
        Input::FsEvent {
            resource: descendant,
            event: FsEvent::ContentChanged,
        },
        now + SETTLE * 2,
    );
    assert!(
        absorb_out.diagnostics.iter().any(|d| matches!(
            d,
            Diagnostic::EventAbsorbedByFireTail { resource, .. } if *resource == descendant,
        )),
        "Awaiting absorb must emit EventAbsorbedByFireTail",
    );
    let descendant_path = e
        .tree
        .path_of(descendant)
        .expect("descendant path resolves");
    let burst = match e.profiles.get(pid).unwrap().state() {
        ProfileState::Active(ActiveBurst::PostFire(post), _) => post,
        _ => panic!("expected Active(Awaiting)"),
    };
    assert!(
        burst
            .final_window_residual
            .chains()
            .contains(&descendant_path),
        "Awaiting absorb must accumulate the event's path into \
         the fire-tail residual for the next Rebasing probe; got {:?}",
        burst.final_window_residual.chains(),
    );

    // EffectComplete::Ok drives Awaiting → Rebasing directly (probe-first), minting the
    // WholeSubtree rebase probe in the same step.
    let rebase_out = e.step(
        Input::EffectComplete(EffectCompletion {
            sub: sid,
            key,
            outcome: EffectOutcome::Ok,
        }),
        now + SETTLE * 3,
    );

    let req = rebase_out
        .probe_ops()
        .iter()
        .find_map(|op| match op {
            ProbeOp::Probe { request } => Some(request),
            ProbeOp::Cancel { .. } => None,
        })
        .expect("Rebase probe minted on EffectComplete (Awaiting → Rebasing)");
    match req {
        ProbeRequest::Subtree {
            obligation, forced, ..
        } => {
            assert!(
                matches!(obligation, ProofObligation::WholeSubtree),
                "rebase probes the whole subtree even with an \
                 Awaiting-absorbed residual — no trustworthy prior to \
                 scope a Chains walk against (the command just mutated \
                 the tree); got {obligation:?}",
            );
            assert!(!forced, "rebase is never forced");
        }
        other => panic!("Rebasing on Dir-anchored Profile must emit Subtree probe; got {other:?}"),
    }

    let burst = match e.profiles.get(pid).unwrap().state() {
        ProfileState::Active(ActiveBurst::PostFire(post), _) => post,
        _ => panic!("expected Active(Rebasing)"),
    };
    assert!(matches!(burst.phase, PostFirePhase::Rebasing(_)));
    assert!(
        burst.final_window_residual.is_empty(),
        "transition_to_rebasing resets the fire-tail residual at the \
         loop entry — the Awaiting-absorbed event is folded into the \
         WholeSubtree read, not carried as a restart seed",
    );
    let _ = e.cancel_all_in_flight_probes();
}

/// Idempotent fire-tail: even with no FsEvent absorbed during Awaiting, the rebase probe ships
/// `WholeSubtree` and is never `forced`. The post-command tree has no trustworthy prior — an
/// in-place descendant edit need not bump an ancestor mtime, so the walker must re-read the whole
/// subtree regardless of mtime or any (now-absent) scoped chain. Pins that the rebase obligation is
/// a soundness floor, not an absorb-conditioned optimization.
#[test]
fn rebasing_without_absorbs_still_probes_whole_subtree() {
    let (mut e, pid, sid, root, _now0) = engine_with_attached_sub();
    let now = Instant::now();
    let stable_out = drive_to_first_effect(&mut e, pid, root, now);
    let key = stable_out.effects()[0].key();

    // EffectComplete::Ok drives Awaiting → Rebasing directly (probe-first), minting the
    // WholeSubtree rebase probe in the same step.
    let rebase_out = e.step(
        Input::EffectComplete(EffectCompletion {
            sub: sid,
            key,
            outcome: EffectOutcome::Ok,
        }),
        now + SETTLE * 3,
    );

    let req = rebase_out
        .probe_ops()
        .iter()
        .find_map(|op| match op {
            ProbeOp::Probe { request } => Some(request),
            ProbeOp::Cancel { .. } => None,
        })
        .expect("Rebase probe minted on EffectComplete (Awaiting → Rebasing)");
    match req {
        ProbeRequest::Subtree {
            obligation, forced, ..
        } => {
            assert!(
                matches!(obligation, ProofObligation::WholeSubtree),
                "rebase ships WholeSubtree even with no absorbs — the \
                 post-command tree has no trustworthy prior to mtime-skip \
                 against; got {obligation:?}",
            );
            assert!(!forced, "rebase is never forced");
        }
        other => panic!("expected Subtree probe; got {other:?}"),
    }
    let _ = e.cancel_all_in_flight_probes();
}

// ---------- post-fire Settling debounce & loop-back ----------

/// An `FsEvent` absorbed during the post-fire `Settling` window updates
/// `PostFireBurst.last_event_time`, and the next `PostFireSettle` expiry reschedules (now −
/// last_event_time < settle) instead of transitioning to `Rebasing` — the post-fire mirror of
/// pre-fire's `event_drives_batching` reschedule. A subsequent expiry past the quiet window then
/// completes the natural Settling → Rebasing advance.
#[test]
fn post_fire_settling_reschedules_on_absorbed_event() {
    // Use a CONTENT-mask Sub so a ContentChanged event at the anchor's covered descendant reaches
    // the absorb arm. The anchor itself also accepts events unconditionally (anchor events bypass
    // the class filter).
    let mut e = Engine::new();
    let root = e.tree.ensure_root("anchor", ResourceRole::User);
    e.tree.set_kind(root, ResourceKind::Dir);
    let req = SubAttachRequest {
        anchor: SubAttachAnchor::Resource(root),
        identity: ProfileIdentity::new(
            ScanConfig::builder().recursive(true).build(),
            MAX_SETTLE,
            ClassSet::CONTENT,
        ),
        params: SubParams::spawn(
            "test-sub".into(),
            empty_program(),
            EffectScope::SubtreeRoot,
            SETTLE,
            false,
        ),
    };
    let attach_out = e.step(Input::AttachSub(req), Instant::now());
    let sid = specter_core::testkit::first_attached_sub(&attach_out).expect("attach_sub succeeded");
    let pid = e.subs.get(sid).unwrap().profile();

    let now = Instant::now();
    let stable_out = drive_to_first_effect(&mut e, pid, root, now);
    let key = stable_out.effects()[0].key();

    // EffectComplete::Ok → Awaiting → Rebasing directly (probe-first), with the WholeSubtree rebase
    // probe in flight.
    let _ = e.step(
        Input::EffectComplete(EffectCompletion {
            sub: sid,
            key,
            outcome: EffectOutcome::Ok,
        }),
        now + SETTLE * 3,
    );
    let rebase_corr = e
        .pending_probe_for(pid)
        .expect("rebase probe in flight after EffectComplete drove Awaiting → Rebasing");

    // Fold the first rebase response to Retry (Undischarged + !terminal walker refusal) so the
    // burst enters the spacing Settling window — the only post-fire Settling entry. `now_a` is the
    // Retry-response instant: `last_event_time` and the PostFireSettle timer are set here,
    // scheduled at `now_a + SETTLE`.
    let now_a = now + SETTLE * 3;
    let unread: std::sync::Arc<std::path::Path> =
        std::sync::Arc::from(std::path::Path::new("anchor/opaque"));
    let degraded = dir_tree_snap(vec![("ghost", EntryKind::File, 9)]);
    let _ = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid,
            correlation: rebase_corr,
            outcome: ProbeOutcome::SubtreeProven {
                snapshot: degraded,
                authority: ProofAuthority::Undischarged {
                    first_unread: unread,
                },
            },
        }),
        now_a,
    );
    let settle_timer_1 = match e.profiles.get(pid).unwrap().state() {
        ProfileState::Active(ActiveBurst::PostFire(post), _) => match &post.phase {
            PostFirePhase::Settling { settle_timer } => *settle_timer,
            other => panic!("Retry must loop into Settling; got {other:?}"),
        },
        other => panic!("expected Active(PostFire(Settling)); got {other:?}"),
    };

    // Absorb an anchor FsEvent strictly inside the settle window (now_a + 5ms ≪ SETTLE). The absorb
    // updates last_event_time and notes into final_window_residual.
    let now_b = now_a + Duration::from_millis(5);
    let absorb_out = e.step(
        Input::FsEvent {
            resource: root,
            event: FsEvent::ContentChanged,
        },
        now_b,
    );
    assert!(
        absorb_out.diagnostics.iter().any(|d| matches!(
            d,
            Diagnostic::EventAbsorbedByFireTail { resource, .. } if *resource == root,
        )),
        "Settling absorb must emit EventAbsorbedByFireTail",
    );
    match e.profiles.get(pid).unwrap().state() {
        ProfileState::Active(ActiveBurst::PostFire(post), _) => {
            assert!(
                matches!(post.phase, PostFirePhase::Settling { .. }),
                "absorb keeps the burst in Settling",
            );
            assert!(
                !post.final_window_residual.is_empty(),
                "absorb noted the anchor event into final_window_residual",
            );
            assert_eq!(
                post.last_event_time,
                Some(now_b),
                "absorb advanced last_event_time to the event instant",
            );
        }
        other => panic!("expected Active(PostFire(Settling)) after absorb; got {other:?}"),
    }

    // First PostFireSettle expiry lands at the original deadline (now_a + SETTLE). The reschedule
    // check: now_c − last_event_time = now_a + SETTLE − now_b < SETTLE (since now_b > now_a). The
    // handler must schedule a fresh PostFireSettle timer and stay in Settling; no rebase probe.
    let now_c = now_a + SETTLE;
    let out_resched = e.step(
        Input::TimerExpired {
            profile: pid,
            kind: TimerKind::PostFireSettle,
            id: settle_timer_1,
        },
        now_c,
    );
    let settle_timer_2 = match e.profiles.get(pid).unwrap().state() {
        ProfileState::Active(ActiveBurst::PostFire(post), _) => match &post.phase {
            PostFirePhase::Settling { settle_timer } => *settle_timer,
            other => {
                panic!("reschedule must keep Settling; got phase {other:?}")
            }
        },
        other => panic!("expected Active(PostFire(Settling)) after reschedule; got {other:?}"),
    };
    assert_ne!(
        settle_timer_1, settle_timer_2,
        "reschedule mints a fresh PostFireSettle timer id (the old id is no \
         longer referenced)",
    );
    assert!(
        out_resched.probe_ops().is_empty(),
        "reschedule emits no rebase probe — the quiet window has not closed",
    );

    // Second PostFireSettle expiry at the new deadline (now_b + SETTLE). Now the quiet window has
    // closed; the handler transitions Settling → Rebasing and mints the rebase probe.
    let now_d = now_b + SETTLE;
    let out_rebase = e.step(
        Input::TimerExpired {
            profile: pid,
            kind: TimerKind::PostFireSettle,
            id: settle_timer_2,
        },
        now_d,
    );
    assert!(
        matches!(
            e.profiles.get(pid).unwrap().state(),
            ProfileState::Active(
                ActiveBurst::PostFire(PostFireBurst {
                    phase: PostFirePhase::Rebasing(_),
                    ..
                }),
                _,
            ),
        ),
        "second expiry (past the rescheduled deadline) transitions Settling → Rebasing",
    );
    assert!(
        out_rebase.probe_ops().iter().any(|op| matches!(
            op,
            ProbeOp::Probe { request } if request.owner() == pid,
        )),
        "Settling → Rebasing mints a fresh rebase probe",
    );
    let _ = e.cancel_all_in_flight_probes();
}

/// A `Retry` rebase response loops back through `transition_to_settling` (no commit), the next
/// `PostFireSettle` expiry re-enters `Rebasing` with a fresh probe, and a follow-up `Authoritative`
/// response commits and finishes — the only surviving post-fire loop. Pairs with
/// `rebase_retry_does_not_poison_current` (which pins the first loop-back); this test pins the
/// COMPLETE retry path: loop entry → settle expiry → re-Rebasing → Authoritative commit. The
/// `Retry` verdict has two origins — channel disagreement and walker refusal with `terminal: false`
/// — that share this dispatcher arm; the test exercises the walker-refusal origin.
#[test]
fn post_fire_retry_loops_via_settling() {
    let (mut e, pid, sid, root, now) = engine_with_attached_sub();
    let rebase_corr = drive_to_rebasing(&mut e, pid, sid, root, now);
    let baseline_before = baseline_hash(&e, pid);

    // First Rebasing response: Undischarged + !terminal authority folds to Retry → Settling. No
    // commit; the baseline must not move.
    let unread: std::sync::Arc<std::path::Path> =
        std::sync::Arc::from(std::path::Path::new("anchor/opaque"));
    let degraded = dir_tree_snap(vec![("ghost", EntryKind::File, 9)]);
    let now_loop = now + SETTLE * 4;
    let _ = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid,
            correlation: rebase_corr,
            outcome: ProbeOutcome::SubtreeProven {
                snapshot: degraded,
                authority: ProofAuthority::Undischarged {
                    first_unread: unread,
                },
            },
        }),
        now_loop,
    );
    let retry_settle_id = match e.profiles.get(pid).unwrap().state() {
        ProfileState::Active(ActiveBurst::PostFire(post), _) => match &post.phase {
            PostFirePhase::Settling { settle_timer } => *settle_timer,
            other => panic!("Retry must loop into Settling; got {other:?}"),
        },
        other => panic!("expected Active(PostFire(Settling)); got {other:?}"),
    };
    assert_eq!(
        baseline_hash(&e, pid),
        baseline_before,
        "Retry must not commit — baseline stays put",
    );

    // PostFireSettle expiry past the settle window: Settling → Rebasing with a fresh probe
    // (different correlation from the first).
    let now_retry = now_loop + SETTLE * 2;
    let out_retry = e.step(
        Input::TimerExpired {
            profile: pid,
            kind: TimerKind::PostFireSettle,
            id: retry_settle_id,
        },
        now_retry,
    );
    let retry_corr = out_retry
        .probe_ops()
        .iter()
        .find_map(|op| match op {
            ProbeOp::Probe { request } => Some(request.correlation()),
            ProbeOp::Cancel { .. } => None,
        })
        .expect("retry mints a fresh Rebasing probe correlation");
    assert_ne!(
        retry_corr, rebase_corr,
        "retry's correlation is fresh (the loop-back disarmed before re-emitting)",
    );
    assert!(
        matches!(
            e.profiles.get(pid).unwrap().state(),
            ProfileState::Active(
                ActiveBurst::PostFire(PostFireBurst {
                    phase: PostFirePhase::Rebasing(_),
                    ..
                }),
                _,
            ),
        ),
        "retry transitioned Settling → Rebasing",
    );

    // Authoritative on the retry: commit + finish. The baseline now moves to the fresh snapshot.
    let fresh = dir_tree_snap(vec![("a.rs", EntryKind::File, 1)]);
    let _ = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid,
            correlation: retry_corr,
            outcome: ProbeOutcome::SubtreeProven {
                snapshot: fresh.clone(),
                authority: ProofAuthority::Authoritative,
            },
        }),
        now_retry,
    );
    assert!(
        matches!(e.profiles.get(pid).unwrap().state(), ProfileState::Idle),
        "Authoritative on the retry commits and finishes the burst",
    );
    assert_eq!(
        baseline_hash(&e, pid),
        Some(fresh.dir_hash()),
        "retry's commit rebases the baseline to the fresh snapshot",
    );
}

/// The gate-deadline non-zombie recovery skips the rebase-loop ceiling entirely:
/// `handle_gate_deadline` calls `force_pending_post_fire` (lockstep `forced := true; rebase_ceiling
/// := None`) then `transition_to_rebasing` directly — there is no `Settling` window between
/// `Awaiting` and `Rebasing`. Pairs with `fire_cycle_gate_deadline_force_transitions_to_rebasing`
/// (which pins the phase and probe emission); this test additionally pins the field-level shape
/// (`forced == true`, `rebase_ceiling.is_none()`) and the absence of a `PostFireSettle` schedule on
/// the path.
#[test]
fn gate_deadline_non_zombie_drives_rebase_with_forced_directly() {
    let (mut e, pid, _sid, root, _now) = engine_with_attached_sub();
    let now = Instant::now();
    let stable_out = drive_to_first_effect(&mut e, pid, root, now);
    assert_eq!(
        stable_out.effects().len(),
        1,
        "stable verdict fires one Effect"
    );

    // Confirm Awaiting (precondition for gate-deadline recovery).
    let gate_deadline_id = match e.profiles.get(pid).unwrap().state() {
        ProfileState::Active(ActiveBurst::PostFire(post), _) => match &post.phase {
            PostFirePhase::Awaiting { gate_deadline, .. } => *gate_deadline,
            other => panic!("expected Active(PostFire(Awaiting)) post-fire; got {other:?}"),
        },
        other => panic!("expected Active(PostFire); got {other:?}"),
    };

    // Fire the AwaitGateDeadline timer past the gate window. Use `step` directly with the captured
    // id so the test runs the single transition under inspection (not a multi-timer drain).
    let now_gate = now + MAX_SETTLE * 8;
    let out_gate = e.step(
        Input::TimerExpired {
            profile: pid,
            kind: TimerKind::AwaitGateDeadline,
            id: gate_deadline_id,
        },
        now_gate,
    );

    // Phase: Awaiting → Rebasing directly (no Settling in between). CeilingState: gate-deadline
    // latches directly from NotStarted to Reached without arming a timer (the prior `forced = true`
    // + `rebase_ceiling = None` lockstep, now a single state).
    match e.profiles.get(pid).unwrap().state() {
        ProfileState::Active(ActiveBurst::PostFire(post), _) => {
            assert!(
                matches!(post.phase, PostFirePhase::Rebasing(_)),
                "gate-deadline transitions Awaiting → Rebasing directly; got {:?}",
                post.phase,
            );
            assert!(
                matches!(post.ceiling, CeilingState::Reached),
                "gate-deadline latches CeilingState::Reached directly from \
                 NotStarted without arming a timer; got {:?}",
                post.ceiling,
            );
        }
        other => panic!("expected Active(PostFire(Rebasing)); got {other:?}"),
    }

    // The Rebasing burst carries no PostFireSettle token: the gate-deadline went Awaiting →
    // Rebasing directly, never opening a Settling window.
    assert!(
        e.profiles
            .get(pid)
            .unwrap()
            .state()
            .timer_token(TimerKind::PostFireSettle)
            .is_none(),
        "gate-deadline skips Settling entirely — no PostFireSettle armed",
    );

    // Rebase probe emitted; respond Authoritative to confirm the forced=true commit terminal.
    let retry_corr = out_gate
        .probe_ops()
        .iter()
        .find_map(|op| match op {
            ProbeOp::Probe { request } => Some(request.correlation()),
            ProbeOp::Cancel { .. } => None,
        })
        .expect("gate-deadline emitted the rebase probe");
    let fresh = dir_tree_snap(vec![("a.rs", EntryKind::File, 1)]);
    let final_out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid,
            correlation: retry_corr,
            outcome: ProbeOutcome::SubtreeProven {
                snapshot: fresh.clone(),
                authority: ProofAuthority::Authoritative,
            },
        }),
        now_gate + Duration::from_millis(1),
    );
    assert!(
        matches!(e.profiles.get(pid).unwrap().state(), ProfileState::Idle),
        "Authoritative + forced=true commits and finishes",
    );
    assert_eq!(
        baseline_hash(&e, pid),
        Some(fresh.dir_hash()),
        "forced commit rebases the baseline to the freshest observation",
    );
    assert!(
        final_out.diagnostics.iter().any(|d| matches!(
            d,
            Diagnostic::RebaseCeilingForced {
                profile,
                observed_change: false,
                ..
            } if *profile == pid,
        )),
        "forced=true commit emits RebaseCeilingForced; the single gate-deadline sample \
         (prior=None) saw no disagreement ⇒ observed_change=false",
    );
}

// ---------- post-fire rebase-loop consequence table ----------

/// The `RebaseCeiling` timer armed on the in-flight post-fire loop.
fn rebase_ceiling_timer(e: &Engine, pid: specter_core::ProfileId) -> specter_core::TimerId {
    match e.profiles.get(pid).unwrap().state() {
        ProfileState::Active(ActiveBurst::PostFire(post), _) => post
            .timer_token(TimerKind::RebaseCeiling)
            .expect("RebaseCeiling armed at the loop's first entry"),
        other => panic!("expected Active(PostFire); got {other:?}"),
    }
}

fn baseline_hash(e: &Engine, pid: specter_core::ProfileId) -> Option<u128> {
    e.profiles.get(pid).unwrap().baseline().map(|s| s.hash())
}

fn current_hash(e: &Engine, pid: specter_core::ProfileId) -> Option<u128> {
    e.profiles.get(pid).unwrap().current().map(|s| s.hash())
}

/// `Authoritative`, ceiling not reached: an `Stable(StableReason::Natural)` rebase response commits
/// the snapshot, rebases the baseline, and finishes — no loop, no settle-spaced sample.
#[test]
fn rebase_authoritative_commits_baseline_and_finishes() {
    let (mut e, pid, sid, root, now) = engine_with_attached_sub();
    let rebase_corr = drive_to_rebasing(&mut e, pid, sid, root, now);

    let observed = dir_tree_snap(vec![
        ("a.rs", EntryKind::File, 1),
        ("b.rs", EntryKind::File, 2),
    ]);
    e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid,
            correlation: rebase_corr,
            outcome: ProbeOutcome::SubtreeProven {
                snapshot: observed.clone(),
                authority: ProofAuthority::Authoritative,
            },
        }),
        now + SETTLE * 4,
    );

    assert!(
        matches!(e.profiles.get(pid).unwrap().state(), ProfileState::Idle),
        "Authoritative + !forced is the natural fire arm — finish to Idle",
    );
    assert_eq!(
        baseline_hash(&e, pid),
        Some(observed.dir_hash()),
        "Authoritative + !forced commits the snapshot and rebases the baseline",
    );
}

/// A `Retry` rebase response — produced by the walker refusing on some chain with `terminal:
/// false`, or by the hash channel disagreeing at this sample — loops back through
/// `transition_to_settling`: an unread region must never poison `current` — **no** `apply_snapshot`
/// — and the carrier prior is withheld. The loop settle-spaces for another sample. The test
/// exercises the walker-refusal origin; the channel disagreement origin folds identically.
#[test]
fn rebase_retry_does_not_poison_current() {
    let (mut e, pid, sid, root, now) = engine_with_attached_sub();
    let rebase_corr = drive_to_rebasing(&mut e, pid, sid, root, now);
    let baseline_before = baseline_hash(&e, pid);
    let current_before = current_hash(&e, pid);

    // An unread response: the walker could not discharge its obligation at `first_unread`. Fold:
    // Undischarged + !forced ⇒ Retry.
    let unread: std::sync::Arc<std::path::Path> =
        std::sync::Arc::from(std::path::Path::new("anchor/opaque"));
    let degraded = dir_tree_snap(vec![("ghost", EntryKind::File, 9)]);
    e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid,
            correlation: rebase_corr,
            outcome: ProbeOutcome::SubtreeProven {
                snapshot: degraded,
                authority: ProofAuthority::Undischarged {
                    first_unread: unread,
                },
            },
        }),
        now + SETTLE * 4,
    );

    match e.profiles.get(pid).unwrap().state() {
        ProfileState::Active(ActiveBurst::PostFire(post), _) => assert!(
            matches!(post.phase, PostFirePhase::Settling { .. }),
            "Retry loops back through Settling; got {:?}",
            post.phase,
        ),
        other => panic!("expected Active(PostFire(Settling)); got {other:?}"),
    }
    assert_eq!(
        current_hash(&e, pid),
        current_before,
        "Retry must NOT apply_snapshot — an unread region cannot poison current",
    );
    assert_eq!(
        baseline_hash(&e, pid),
        baseline_before,
        "Retry never rebases the baseline",
    );
}

/// `Authoritative`, ceiling **reached** (forced=true): the `RebaseCeiling` fired but the walker
/// still certified. Pin the freshest observation as the new baseline anyway (a deliberate, loud
/// terminal — not a wedge) and finish.
#[test]
fn rebase_authoritative_at_ceiling_pins_freshest_and_diagnoses() {
    let (mut e, pid, sid, root, now) = engine_with_attached_sub();
    let rebase_corr = drive_to_rebasing(&mut e, pid, sid, root, now);

    // Latch the ceiling while the probe is in flight (set-only — the in-flight response carries the
    // terminal as forced=true).
    let ceiling = rebase_ceiling_timer(&e, pid);
    e.step(
        Input::TimerExpired {
            profile: pid,
            kind: TimerKind::RebaseCeiling,
            id: ceiling,
        },
        now + SETTLE * 4,
    );

    let freshest = dir_tree_snap(vec![
        ("a.rs", EntryKind::File, 1),
        ("late.rs", EntryKind::File, 5),
    ]);
    let out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid,
            correlation: rebase_corr,
            outcome: ProbeOutcome::SubtreeProven {
                snapshot: freshest.clone(),
                authority: ProofAuthority::Authoritative,
            },
        }),
        now + SETTLE * 5,
    );

    assert!(
        matches!(e.profiles.get(pid).unwrap().state(), ProfileState::Idle),
        "Authoritative + forced=true is a terminal — the burst finishes",
    );
    assert_eq!(
        baseline_hash(&e, pid),
        Some(freshest.dir_hash()),
        "Authoritative + forced=true pins the freshest observation as the rebased baseline",
    );
    assert!(
        out.diagnostics.iter().any(|d| matches!(
            d,
            Diagnostic::RebaseCeilingForced {
                profile,
                intent,
                observed_change: false,
            } if *profile == pid && *intent == BurstIntent::Standard,
        )),
        "Authoritative + forced=true emits the loud RebaseCeilingForced; the single \
         sample (prior=None) saw no disagreement ⇒ observed_change=false; got {:?}",
        out.diagnostics,
    );
}

/// An `Abandon { first_unread }` rebase response — Undischarged authority with the ceiling
/// **reached** (forced=true) — refuses to rebase blind. No commit, no rebase — the prior baseline
/// stays in place — plus the loud `RebaseCeilingUnreadable` carrying `first_unread`.
#[test]
fn rebase_abandon_refuses_blind_rebase_and_diagnoses() {
    let (mut e, pid, sid, root, now) = engine_with_attached_sub();
    let rebase_corr = drive_to_rebasing(&mut e, pid, sid, root, now);
    let baseline_before = baseline_hash(&e, pid);

    let ceiling = rebase_ceiling_timer(&e, pid);
    e.step(
        Input::TimerExpired {
            profile: pid,
            kind: TimerKind::RebaseCeiling,
            id: ceiling,
        },
        now + SETTLE * 4,
    );

    let unread: std::sync::Arc<std::path::Path> =
        std::sync::Arc::from(std::path::Path::new("anchor/opaque"));
    let out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid,
            correlation: rebase_corr,
            outcome: ProbeOutcome::SubtreeProven {
                snapshot: dir_tree_snap(vec![("ghost", EntryKind::File, 9)]),
                authority: ProofAuthority::Undischarged {
                    first_unread: std::sync::Arc::clone(&unread),
                },
            },
        }),
        now + SETTLE * 5,
    );

    assert!(
        matches!(e.profiles.get(pid).unwrap().state(), ProfileState::Idle),
        "Abandon is a terminal — the burst finishes",
    );
    assert_eq!(
        baseline_hash(&e, pid),
        baseline_before,
        "Abandon never rebases blind — the prior baseline stays in place",
    );
    assert!(
        out.diagnostics.iter().any(|d| matches!(
            d,
            Diagnostic::RebaseCeilingUnreadable { profile, first_unread, intent }
                if *profile == pid
                    && first_unread.as_ref() == std::path::Path::new("anchor/opaque")
                    && *intent == BurstIntent::Standard,
        )),
        "Abandon emits RebaseCeilingUnreadable carrying first_unread; got {:?}",
        out.diagnostics,
    );
}

/// B4 mirror — ceiling expiry **in `Rebasing`** (a probe in flight, the `Verifying` analogue):
/// set-only. No immediate re-drive, no new probe; the in-flight response applies the terminal via
/// `dispatch_rebase_ok`'s `reached` read.
#[test]
fn rebase_ceiling_in_rebasing_is_set_only_inflight_response_applies_terminal() {
    let (mut e, pid, sid, root, now) = engine_with_attached_sub();
    let rebase_corr = drive_to_rebasing(&mut e, pid, sid, root, now);
    let ceiling = rebase_ceiling_timer(&e, pid);

    let ceiling_out = e.step(
        Input::TimerExpired {
            profile: pid,
            kind: TimerKind::RebaseCeiling,
            id: ceiling,
        },
        now + SETTLE * 4,
    );
    assert!(
        ceiling_out
            .probe_ops()
            .iter()
            .all(|op| !matches!(op, ProbeOp::Probe { .. })),
        "ceiling in Rebasing is set-only — no fresh probe driven; got {:?}",
        ceiling_out.probe_ops(),
    );
    match e.profiles.get(pid).unwrap().state() {
        ProfileState::Active(ActiveBurst::PostFire(post), _) => {
            assert!(
                matches!(post.phase, PostFirePhase::Rebasing(_)),
                "still Rebasing — the in-flight probe was not cancelled; got {:?}",
                post.phase,
            );
            assert!(
                matches!(post.ceiling, CeilingState::Reached),
                "the ceiling latched: CeilingState::Reached raised by \
                 force_pending_post_fire; got {:?}",
                post.ceiling,
            );
        }
        other => panic!("expected Active(PostFire(Rebasing)); got {other:?}"),
    }

    // The original in-flight probe's response applies the terminal.
    e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid,
            correlation: rebase_corr,
            outcome: ProbeOutcome::SubtreeProven {
                snapshot: dir_tree_snap(vec![("a.rs", EntryKind::File, 1)]),
                authority: ProofAuthority::Authoritative,
            },
        }),
        now + SETTLE * 5,
    );
    assert!(
        matches!(e.profiles.get(pid).unwrap().state(), ProfileState::Idle),
        "the in-flight response, seeing reached, applies the terminal → Idle",
    );
}

/// B4 mirror — `RebaseCeiling` expiry while the burst is in `Settling` (no probe in flight) latches
/// `forced = true` and drives `Rebasing` in the same step — the post-fire mirror of
/// `handle_burst_deadline` driving a verify when no Verifying probe is in flight. The
/// `Stable(StableReason::Forced)` response then commits + emits `RebaseCeilingForced` (the
/// Undischarged-retry withheld the carrier, so the forced sample sees `prior == None` ⇒
/// `observed_change == false`).
#[test]
fn rebase_ceiling_in_settling_drives_rebasing_with_forced() {
    let (mut e, pid, sid, root, now) = engine_with_attached_sub();
    let rebase_corr = drive_to_rebasing(&mut e, pid, sid, root, now);

    // Loop back through Settling via a Retry response — produced by an Undischarged !forced
    // authority (the only surviving post-fire loop). The `Stable(Natural)` arm finishes
    // immediately; we need the burst still alive in Settling to fire the ceiling there.
    let unread: std::sync::Arc<std::path::Path> =
        std::sync::Arc::from(std::path::Path::new("anchor/opaque"));
    e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid,
            correlation: rebase_corr,
            outcome: ProbeOutcome::SubtreeProven {
                snapshot: dir_tree_snap(vec![]),
                authority: ProofAuthority::Undischarged {
                    first_unread: unread,
                },
            },
        }),
        now + SETTLE * 4,
    );
    assert!(
        matches!(
            e.profiles.get(pid).unwrap().state(),
            ProfileState::Active(
                ActiveBurst::PostFire(PostFireBurst {
                    phase: PostFirePhase::Settling { .. },
                    ..
                }),
                _,
            ),
        ),
        "Retry loops the post-fire burst back through Settling",
    );

    // Now fire the RebaseCeiling in Settling.
    let ceiling = rebase_ceiling_timer(&e, pid);
    let ceiling_out = e.step(
        Input::TimerExpired {
            profile: pid,
            kind: TimerKind::RebaseCeiling,
            id: ceiling,
        },
        now + SETTLE * 5,
    );
    let driven = ceiling_out.probe_ops().iter().find_map(|op| match op {
        ProbeOp::Probe { request } => Some(request.correlation()),
        ProbeOp::Cancel { .. } => None,
    });
    assert!(
        driven.is_some(),
        "ceiling in Settling latches forced=true and drives Rebasing — fresh probe emitted",
    );
    match e.profiles.get(pid).unwrap().state() {
        ProfileState::Active(ActiveBurst::PostFire(post), _) => {
            assert!(
                matches!(post.phase, PostFirePhase::Rebasing(_)),
                "driven immediately back into Rebasing; got {:?}",
                post.phase,
            );
            assert!(
                matches!(post.ceiling, CeilingState::Reached),
                "the ceiling latched: CeilingState::Reached; got {:?}",
                post.ceiling,
            );
        }
        other => panic!("expected Active(PostFire(Rebasing)); got {other:?}"),
    }

    // That driven probe's `Authoritative` response folds with forced=true to the ceiling-pin
    // terminal: commit + diagnose + finish.
    let freshest = dir_tree_snap(vec![("a.rs", EntryKind::File, 1)]);
    let final_out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid,
            correlation: driven.unwrap(),
            outcome: ProbeOutcome::SubtreeProven {
                snapshot: freshest.clone(),
                authority: ProofAuthority::Authoritative,
            },
        }),
        now + SETTLE * 6,
    );
    assert!(
        matches!(e.profiles.get(pid).unwrap().state(), ProfileState::Idle),
        "Authoritative + forced=true is the rebase-ceiling terminal → Idle",
    );
    assert_eq!(
        baseline_hash(&e, pid),
        Some(freshest.dir_hash()),
        "Authoritative + forced=true pins the freshest observation",
    );
    assert!(
        final_out.diagnostics.iter().any(|d| matches!(
            d,
            Diagnostic::RebaseCeilingForced {
                profile,
                observed_change: false,
                ..
            } if *profile == pid,
        )),
        "ceiling terminal emits RebaseCeilingForced (observed_change=false — the \
         Undischarged-retry withheld the carrier, so prior=None); got {:?}",
        final_out.diagnostics,
    );
}

/// Spacing soundness — the reason post-fire `Settling` settle-spaces consecutive rebase samples on
/// an events-incomplete Profile. A writer whose period exceeds the probe round-trip but is shorter
/// than `settle` would let two back-to-back `WholeSubtree` reads catch the same transient
/// byte-state and manufacture a premature `Stable`. The loop instead settle-separates consecutive
/// samples through `Settling`: a mid-loop change makes sample N+1 differ from sample N ⇒ `Retry` ⇒
/// keep looping; only two settle-spaced *equal* samples close `Stable`.
///
/// Pins, on a `STRUCTURE`-only Profile (the hash channel is active for the rebase loop): a
/// differing second sample does NOT prematurely finish, a `Settling` settle-spacer always sits
/// between samples, and the third equal sample is what closes the loop. The events-reliable rebase
/// path skips the channel entirely (a single sample fires), so this scenario is reachable only on
/// events-incomplete masks.
#[test]
fn rebase_loop_spacing_defeats_a_premature_stable_on_a_slow_writer() {
    let mut e = Engine::new();
    let r = e.tree.ensure_root("anchor", ResourceRole::User);
    e.tree.set_kind(r, ResourceKind::Dir);
    let now = Instant::now();
    let (sid, pid) = crate::testkit::attach_structure_only(&mut e, r, now);

    // Cold-Seed bypass — a never-fired Profile with `dirty.is_empty()` does not owe a quiescence
    // proof; one Authoritative sample pins the baseline → Idle even on an events-incomplete mask.
    let baseline = dir_tree_snap(vec![]);
    let seed_done = crate::testkit::seed_to_idle(&mut e, pid, &baseline, now);

    // Open the post-fire loop. The Standard fire owes the hash channel (structure-only mask,
    // fire-bearing burst): drive two equal samples to a single `Stable` and Effect. The pre-fire
    // path is exercised mechanically — the test's claim is the post-fire loop below.
    let fire_snap = dir_tree_snap(vec![("draft", EntryKind::File, 1)]);
    let fire_start = seed_done + Duration::from_millis(1);
    let stable_out = crate::testkit::drive_standard_n2_until_stable(
        &mut e,
        pid,
        r,
        &[Arc::clone(&fire_snap), Arc::clone(&fire_snap)],
        fire_start,
    );
    let key = stable_out.effects()[0].key();

    // EffectComplete → Awaiting → Rebasing directly (probe-first; the rebase loop's natural entry).
    // `arm_rebase_loop_ceiling` armed `RebaseCeiling` here for the whole loop's bound — well past
    // the settle windows this test walks through. Sample 1's WholeSubtree probe is already in
    // flight. The helper advances `at += SETTLE * 2` per sample and fired on the second equal
    // sample, so we are at `fire_start + SETTLE * 4`.
    let after_fire = fire_start + SETTLE * 4 + Duration::from_millis(1);
    e.step(
        Input::EffectComplete(EffectCompletion {
            sub: sid,
            key,
            outcome: EffectOutcome::Ok,
        }),
        after_fire,
    );
    let corr1 = match e.profiles.get(pid).unwrap().state() {
        ProfileState::Active(ActiveBurst::PostFire(post), _) => {
            assert!(
                matches!(post.phase, PostFirePhase::Rebasing(_)),
                "expected Rebasing after EffectComplete::Ok; got {:?}",
                post.phase,
            );
            e.pending_probe_for(pid)
                .expect("sample 1's rebase probe in flight after EffectComplete drove Rebasing")
        }
        other => panic!("expected Active(PostFire) after EffectComplete::Ok; got {other:?}"),
    };

    // Sample 1: the writer's state at this instant (carrier prior None ⇒ Retry). Records carrier :=
    // hash(A); the response routes through `transition_to_settling` (no commit).
    let a = dir_tree_snap(vec![("draft", EntryKind::File, 1)]);
    let t_s1 = after_fire + SETTLE * 2;
    e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid,
            correlation: corr1,
            outcome: ProbeOutcome::SubtreeProven {
                snapshot: Arc::clone(&a),
                authority: ProofAuthority::Authoritative,
            },
        }),
        t_s1,
    );
    let settle1 = match e.profiles.get(pid).unwrap().state() {
        ProfileState::Active(ActiveBurst::PostFire(post), _) => match &post.phase {
            PostFirePhase::Settling { settle_timer } => *settle_timer,
            other => panic!("sample 1 must settle-space before sample 2; got {other:?}"),
        },
        other => panic!("expected Active(PostFire(Settling)); got {other:?}"),
    };

    // PostFireSettle expiry → Rebasing for sample 2.
    let rearm1 = e.step(
        Input::TimerExpired {
            profile: pid,
            kind: TimerKind::PostFireSettle,
            id: settle1,
        },
        t_s1 + SETTLE,
    );
    let corr2 = rearm1
        .probe_ops()
        .iter()
        .find_map(|op| match op {
            ProbeOp::Probe { request } => Some(request.correlation()),
            ProbeOp::Cancel { .. } => None,
        })
        .expect("PostFireSettle re-arms Rebasing #2");

    // Sample 2, settle-spaced: the slow writer advanced, so this read differs (B ≠ A). carrier prior
    // hash(A) ≠ hash(B) ⇒ Retry — NOT a premature Stable just because it is the "second" sample.
    let b = dir_tree_snap(vec![
        ("draft", EntryKind::File, 1),
        ("more", EntryKind::File, 2),
    ]);
    let t_s2 = t_s1 + SETTLE * 2;
    e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid,
            correlation: corr2,
            outcome: ProbeOutcome::SubtreeProven {
                snapshot: Arc::clone(&b),
                authority: ProofAuthority::Authoritative,
            },
        }),
        t_s2,
    );
    let settle2 = match e.profiles.get(pid).unwrap().state() {
        ProfileState::Active(ActiveBurst::PostFire(post), _) => match &post.phase {
            PostFirePhase::Settling { settle_timer } => *settle_timer,
            other => panic!(
                "a differing settle-spaced sample must NOT prematurely Stable — keep looping; got {other:?}",
            ),
        },
        other => panic!("expected Active(PostFire(Settling)); got {other:?}"),
    };

    // PostFireSettle expiry → Rebasing for sample 3.
    let rearm2 = e.step(
        Input::TimerExpired {
            profile: pid,
            kind: TimerKind::PostFireSettle,
            id: settle2,
        },
        t_s2 + SETTLE,
    );
    let corr3 = rearm2
        .probe_ops()
        .iter()
        .find_map(|op| match op {
            ProbeOp::Probe { request } => Some(request.correlation()),
            ProbeOp::Cancel { .. } => None,
        })
        .expect("PostFireSettle re-arms Rebasing #3");

    // Sample 3, settle-spaced: the writer has now quiesced (== B). carrier prior hash(B) == hash(B)
    // ⇒ Stable ⇒ commit + finish. Two settle-spaced *equal* samples close the loop — never a sub-
    // settle coincidence.
    let t_s3 = t_s2 + SETTLE * 2;
    e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid,
            correlation: corr3,
            outcome: ProbeOutcome::SubtreeProven {
                snapshot: Arc::clone(&b),
                authority: ProofAuthority::Authoritative,
            },
        }),
        t_s3,
    );
    assert!(
        matches!(e.profiles.get(pid).unwrap().state(), ProfileState::Idle),
        "two settle-spaced equal samples (B, B) finally close Stable → Idle",
    );
    assert_eq!(
        baseline_hash(&e, pid),
        Some(b.dir_hash()),
        "the natural Stable terminal rebases the baseline to the freshest equal sample",
    );
}

/// The strong-signal pre-fire ceiling diagnostic: `QuiescenceCeilingForcedDespiteChange` fires only
/// when the hash channel was active AND observed concrete `prior != response` disagreement before
/// the `BurstDeadline` expired. On the natural `Stable(Forced)` arm (channel inactive OR
/// first-sample `prior=None` OR `prior == response`), pre-fire stays silent — the `forced` bit is
/// already operator-visible on `Effect.forced`.
///
/// Drives a structure-only Profile through Sample 1 (carrier prior = None → Retry, carrier :=
/// hash(A)), then expires the `BurstDeadline` mid-Batching to set `forced = true` and drive a fresh
/// Verifying probe; Sample 2 (hash(B) ≠ hash(A)) folds to `Stable(Forced { hash_channel_disagreed:
/// true })` ⇒ the strong-signal diagnostic emits exactly once alongside the bounded fire.
#[test]
fn pre_fire_forced_ceiling_with_hash_disagreement_emits_strong_signal_diagnostic() {
    let mut e = Engine::new();
    let r = e.tree.ensure_root("anchor", ResourceRole::User);
    e.tree.set_kind(r, ResourceKind::Dir);
    let now = Instant::now();
    let (_sid, pid) = crate::testkit::attach_structure_only(&mut e, r, now);
    let baseline = dir_tree_snap(vec![]);
    let seed_done = crate::testkit::seed_to_idle(&mut e, pid, &baseline, now);

    // Open Standard burst; `start_standard_burst` schedules the `BurstDeadline` at `burst_start +
    // MAX_SETTLE`.
    let burst_start = seed_done + Duration::from_millis(1);
    e.step(
        Input::FsEvent {
            resource: r,
            event: FsEvent::StructureChanged,
        },
        burst_start,
    );

    // Sample 1: Settle expires → Verifying probe → response (snap A). Carrier prior=None ⇒ Retry ⇒
    // `retry_drives_batching` re-arms Batching with carrier := hash(A).
    let a = dir_tree_snap(vec![("draft", EntryKind::File, 1)]);
    let s1_at = burst_start + SETTLE * 2;
    crate::testkit::drain_due(&mut e, s1_at);
    let corr1 = e
        .pending_probe_for(pid)
        .expect("Verifying probe in flight after settle expiry");
    let s1_out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid,
            correlation: corr1,
            outcome: ProbeOutcome::SubtreeProven {
                snapshot: Arc::clone(&a),
                authority: ProofAuthority::Authoritative,
            },
        }),
        s1_at,
    );
    assert!(
        s1_out.effects().is_empty(),
        "Sample 1: carrier prior=None ⇒ Retry ⇒ no fire (re-Batching)",
    );

    // Fire the BurstDeadline mid-Batching: `force_pending` sets `forced=true`, then
    // `handle_burst_deadline` drives a fresh Verifying probe (Batching phase: no probe in flight).
    let bd_id = match e.profiles.get(pid).unwrap().state() {
        ProfileState::Active(ActiveBurst::PreFire(pre), _) => pre.burst_deadline,
        other => panic!("expected Active(PreFire) post-Sample-1; got {other:?}"),
    };
    let bd_out = e.step(
        Input::TimerExpired {
            profile: pid,
            kind: TimerKind::BurstDeadline,
            id: bd_id,
        },
        burst_start + MAX_SETTLE + Duration::from_millis(1),
    );
    let corr2 = bd_out
        .probe_ops()
        .iter()
        .find_map(|op| match op {
            ProbeOp::Probe { request } => Some(request.correlation()),
            ProbeOp::Cancel { .. } => None,
        })
        .expect("BurstDeadline drives a fresh forced Verifying probe");

    // Sample 2: forced=true + carrier prior=Some(hash(A)), response=hash(B) (≠) ⇒
    // Stable(Forced{disagreed=true}) ⇒ QuiescenceCeilingForcedDespiteChange + fire.
    let b = dir_tree_snap(vec![
        ("draft", EntryKind::File, 1),
        ("late", EntryKind::File, 5),
    ]);
    assert_ne!(
        a.dir_hash(),
        b.dir_hash(),
        "the diagnostic-selection bit needs distinct sample hashes",
    );
    let s2_out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid,
            correlation: corr2,
            outcome: ProbeOutcome::SubtreeProven {
                snapshot: Arc::clone(&b),
                authority: ProofAuthority::Authoritative,
            },
        }),
        burst_start + MAX_SETTLE + Duration::from_millis(2),
    );

    assert_eq!(
        s2_out.effects().len(),
        1,
        "Stable(Forced) fires (ceiling bypass) — fire path is the same regardless of diagnostic selection",
    );
    assert!(
        s2_out.diagnostics.iter().any(|d| matches!(
            d,
            Diagnostic::QuiescenceCeilingForcedDespiteChange { profile, intent }
                if *profile == pid && *intent == BurstIntent::Standard,
        )),
        "forced=true + observed disagreement ⇒ strong-signal pre-fire ceiling diagnostic; got {:?}",
        s2_out.diagnostics,
    );
}

/// U2 — the post-fire forced-ceiling diagnostic carries the carrier's disagreement bit in
/// `observed_change`, end to end. Drives a structure-only Profile through the post-fire loop until
/// the `RebaseCeiling` forces a commit, then varies only the second (forced) sample relative to the
/// carried first sample:
///
/// - second sample DIFFERS ⇒ `Stable(Forced{disagreed=true})` ⇒ `RebaseCeilingForced {
///   observed_change: true }` (the strong "tree visibly still moving at ceiling expiry" signal).
/// - second sample EQUALS the first ⇒ `Stable(Forced{disagreed=false})` ⇒ `RebaseCeilingForced {
///   observed_change: false }` — the same-step race the ceiling resolves: the samples agreed, but
///   the ceiling ran out before two consecutive equal reads could fold a natural `Stable`, so the
///   freshest observation is pinned anyway.
///
/// Exactly one `RebaseCeilingForced` per forced rebase either way; only the bit varies. The
/// equal-sample (`prior == response`) arm is the path no other end-to-end test exercises — the case
/// the retired `RebaseCeilingStillChanging` name libeled.
#[test]
fn post_fire_forced_ceiling_observed_change_tracks_carrier_disagreement() {
    let first = dir_tree_snap(vec![("draft", EntryKind::File, 1)]);
    let distinct = dir_tree_snap(vec![
        ("draft", EntryKind::File, 1),
        ("late", EntryKind::File, 5),
    ]);
    assert_ne!(
        first.dir_hash(),
        distinct.dir_hash(),
        "the disagreement arm needs a distinct second-sample hash",
    );

    for (second, expected_observed_change) in
        [(Arc::clone(&distinct), true), (Arc::clone(&first), false)]
    {
        let mut e = Engine::new();
        let r = e.tree.ensure_root("anchor", ResourceRole::User);
        e.tree.set_kind(r, ResourceKind::Dir);
        let now = Instant::now();
        let (sid, pid) = crate::testkit::attach_structure_only(&mut e, r, now);
        let baseline = dir_tree_snap(vec![]);
        let seed_done = crate::testkit::seed_to_idle(&mut e, pid, &baseline, now);

        // Fire the Standard burst (events-incomplete pre-fire: N=2 with two equal samples) →
        // EffectComplete → Awaiting → Rebasing directly (probe-first; rebase probe in flight;
        // RebaseCeiling armed at the natural Awaiting→Rebasing entry).
        let fire_snap = dir_tree_snap(vec![("draft", EntryKind::File, 1)]);
        let fire_start = seed_done + Duration::from_millis(1);
        let stable_out = crate::testkit::drive_standard_n2_until_stable(
            &mut e,
            pid,
            r,
            &[Arc::clone(&fire_snap), Arc::clone(&fire_snap)],
            fire_start,
        );
        let key = stable_out.effects()[0].key();
        let after_fire = fire_start + SETTLE * 4 + Duration::from_millis(1);
        e.step(
            Input::EffectComplete(EffectCompletion {
                sub: sid,
                key,
                outcome: EffectOutcome::Ok,
            }),
            after_fire,
        );
        let corr1 = match e.profiles.get(pid).unwrap().state() {
            ProfileState::Active(ActiveBurst::PostFire(post), _) => {
                assert!(
                    matches!(post.phase, PostFirePhase::Rebasing(_)),
                    "expected Rebasing after EffectComplete::Ok; got {:?}",
                    post.phase,
                );
                e.pending_probe_for(pid)
                    .expect("sample 1's rebase probe in flight after EffectComplete drove Rebasing")
            }
            other => panic!("expected Active(PostFire); got {other:?}"),
        };

        // Sample 1 in Rebasing: `first` ⇒ carrier prior=None ⇒ Retry ⇒ transition_to_settling
        // (carrier := hash(first)).
        let t_s1 = after_fire + SETTLE * 2;
        let s1_out = e.step(
            Input::ProbeResponse(ProbeResponse {
                owner: pid,
                correlation: corr1,
                outcome: ProbeOutcome::SubtreeProven {
                    snapshot: Arc::clone(&first),
                    authority: ProofAuthority::Authoritative,
                },
            }),
            t_s1,
        );
        assert!(
            s1_out
                .diagnostics
                .iter()
                .all(|d| !matches!(d, Diagnostic::RebaseCeilingForced { .. })),
            "Sample 1 (Retry + ceiling not yet reached) emits no rebase ceiling \
             diagnostic; got {:?}",
            s1_out.diagnostics,
        );

        // Fire the RebaseCeiling mid-Settling: `force_pending_post_fire` sets `forced=true` + drops
        // `rebase_ceiling = None`; `handle_rebase_ceiling` drives Rebasing (Settling phase: no
        // probe in flight).
        let ceiling = rebase_ceiling_timer(&e, pid);
        let ceiling_out = e.step(
            Input::TimerExpired {
                profile: pid,
                kind: TimerKind::RebaseCeiling,
                id: ceiling,
            },
            t_s1 + Duration::from_millis(1),
        );
        let corr2 = ceiling_out
            .probe_ops()
            .iter()
            .find_map(|op| match op {
                ProbeOp::Probe { request } => Some(request.correlation()),
                ProbeOp::Cancel { .. } => None,
            })
            .expect("RebaseCeiling in Settling drives a fresh forced Rebasing probe");

        // Sample 2 (forced): carrier prior=Some(hash(first)), response=hash(second). `second ==
        // first` ⇒ p == response ⇒ disagreed=false; `second != first` ⇒ disagreed=true.
        let s2_out = e.step(
            Input::ProbeResponse(ProbeResponse {
                owner: pid,
                correlation: corr2,
                outcome: ProbeOutcome::SubtreeProven {
                    snapshot: Arc::clone(&second),
                    authority: ProofAuthority::Authoritative,
                },
            }),
            t_s1 + Duration::from_millis(2),
        );

        assert!(
            matches!(e.profiles.get(pid).unwrap().state(), ProfileState::Idle),
            "Stable(Forced) is the rebase-ceiling terminal → Idle",
        );
        assert_eq!(
            baseline_hash(&e, pid),
            Some(second.dir_hash()),
            "Stable(Forced) pins the freshest observation as the rebased baseline",
        );
        let forced: Vec<bool> = s2_out
            .diagnostics
            .iter()
            .filter_map(|d| match d {
                Diagnostic::RebaseCeilingForced {
                    profile,
                    intent,
                    observed_change,
                } if *profile == pid && *intent == BurstIntent::Standard => Some(*observed_change),
                _ => None,
            })
            .collect();
        assert_eq!(
            forced,
            vec![expected_observed_change],
            "exactly one RebaseCeilingForced per forced rebase; observed_change carries \
             the carrier-disagreement bit; got {:?}",
            s2_out.diagnostics,
        );
    }
}

/// The mask-blindspot upgrade at the pre-fire ceiling: a touch-storm under `events = STRUCTURE`.
/// Every settle window is event-silent (no STRUCTURE event ever arrives, so `retry_streak` never
/// resets) yet every sample hashes differently — in-place writes / mtime churn fold into
/// `leaf_hash` while the subscribed class stays quiet. After `CHANGE_OUTSIDE_MASK_RETRY_FLOOR`
/// consecutive Retry windows, the forced-ceiling disagreement emits `ChangeOutsideEventMask`
/// (carrying the streak) **instead of** the generic `QuiescenceCeilingForcedDespiteChange`; the
/// bounded fire itself is unchanged.
#[test]
fn pre_fire_forced_ceiling_after_event_silent_retry_streak_emits_mask_hint() {
    let floor = crate::transitions::CHANGE_OUTSIDE_MASK_RETRY_FLOOR;
    let mut e = Engine::new();
    let r = e.tree.ensure_root("anchor", ResourceRole::User);
    e.tree.set_kind(r, ResourceKind::Dir);
    let now = Instant::now();
    let (_sid, pid) = crate::testkit::attach_structure_only(&mut e, r, now);
    let baseline = dir_tree_snap(vec![]);
    let seed_done = crate::testkit::seed_to_idle(&mut e, pid, &baseline, now);

    let burst_start = seed_done + Duration::from_millis(1);
    e.step(
        Input::FsEvent {
            resource: r,
            event: FsEvent::StructureChanged,
        },
        burst_start,
    );

    // One priming round plus `floor` disagreement rounds: each settle expiry drives a verify whose
    // sample hashes differently from the carrier's prior (distinct sizes ⇒ distinct leaf/dir
    // hashes). The first round folds `Retry { observed_motion: false }` (`prior = None` is absence
    // of confirmation) and holds the streak; every subsequent round observes concrete disagreement
    // and bumps it. No FsEvent arrives anywhere in the loop: the streak never resets.
    let mut at = burst_start;
    for round in 0..=floor {
        at += SETTLE * 2;
        crate::testkit::drain_due(&mut e, at);
        let corr = e
            .pending_probe_for(pid)
            .expect("settle expiry drives a Verifying probe each round");
        let sample = dir_tree_snap(vec![("draft", EntryKind::File, u64::from(round) + 1)]);
        let out = e.step(
            Input::ProbeResponse(ProbeResponse {
                owner: pid,
                correlation: corr,
                outcome: ProbeOutcome::SubtreeProven {
                    snapshot: sample,
                    authority: ProofAuthority::Authoritative,
                },
            }),
            at,
        );
        assert!(
            out.effects().is_empty(),
            "round {round}: Retry must not fire",
        );
    }

    // BurstDeadline mid-Batching: `forced = true` + a fresh Verifying probe.
    let bd_id = match e.profiles.get(pid).unwrap().state() {
        ProfileState::Active(ActiveBurst::PreFire(pre), _) => pre.burst_deadline,
        other => panic!("expected Active(PreFire) after the retry rounds; got {other:?}"),
    };
    let bd_out = e.step(
        Input::TimerExpired {
            profile: pid,
            kind: TimerKind::BurstDeadline,
            id: bd_id,
        },
        burst_start + MAX_SETTLE + Duration::from_millis(1),
    );
    let corr_forced = bd_out
        .probe_ops()
        .iter()
        .find_map(|op| match op {
            ProbeOp::Probe { request } => Some(request.correlation()),
            ProbeOp::Cancel { .. } => None,
        })
        .expect("BurstDeadline drives a fresh forced Verifying probe");

    // The forced sample still disagrees with the carrier (the storm never quiesced) — the terminal
    // pairs `hash_channel_disagreed = true` with `retry_streak == floor`, selecting the hint.
    let final_sample = dir_tree_snap(vec![("draft", EntryKind::File, u64::from(floor) + 2)]);
    let s_out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid,
            correlation: corr_forced,
            outcome: ProbeOutcome::SubtreeProven {
                snapshot: final_sample,
                authority: ProofAuthority::Authoritative,
            },
        }),
        burst_start + MAX_SETTLE + Duration::from_millis(2),
    );

    assert_eq!(
        s_out.effects().len(),
        1,
        "the hint changes the diagnostic, never the bounded fire",
    );
    assert!(
        s_out.diagnostics.iter().any(|d| matches!(
            d,
            Diagnostic::ChangeOutsideEventMask { profile, intent, retries }
                if *profile == pid && *intent == BurstIntent::Standard && *retries == floor,
        )),
        "streak at the floor upgrades the forced-ceiling disagreement to the mask hint; got {:?}",
        s_out.diagnostics,
    );
    assert!(
        s_out
            .diagnostics
            .iter()
            .all(|d| !matches!(d, Diagnostic::QuiescenceCeilingForcedDespiteChange { .. })),
        "the hint replaces the generic despite-change diagnostic, not alongside it; got {:?}",
        s_out.diagnostics,
    );
}

/// The mask-blindspot streak is disagreement-denominated: a window that observed *nothing* — a
/// transient probe failure (FD pressure) — holds the streak rather than inflating it. A burst that
/// suffered `floor` such windows and then one genuine disagreement at the forced ceiling reports
/// the generic `QuiescenceCeilingForcedDespiteChange`, **not** `ChangeOutsideEventMask`: the story
/// was pressure, not motion outside the mask. (The walker-refusal Retry origin folds
/// `observed_motion: false` identically — `quiescence_verdict_folds_three_axes` pins the fold; this
/// pins the streak consequence through the engine's failure arm.)
#[test]
fn pre_fire_forced_ceiling_after_transient_windows_keeps_generic_diagnostic() {
    let floor = crate::transitions::CHANGE_OUTSIDE_MASK_RETRY_FLOOR;
    let mut e = Engine::new();
    let r = e.tree.ensure_root("anchor", ResourceRole::User);
    e.tree.set_kind(r, ResourceKind::Dir);
    let now = Instant::now();
    let (_sid, pid) = crate::testkit::attach_structure_only(&mut e, r, now);
    let baseline = dir_tree_snap(vec![]);
    let seed_done = crate::testkit::seed_to_idle(&mut e, pid, &baseline, now);

    let burst_start = seed_done + Duration::from_millis(1);
    e.step(
        Input::FsEvent {
            resource: r,
            event: FsEvent::StructureChanged,
        },
        burst_start,
    );

    // Round 0 primes the carrier: one Authoritative sample (`prior = None` ⇒ Retry with no motion
    // observed — streak stays 0, carrier := the sample's hash).
    let mut at = burst_start + SETTLE * 2;
    crate::testkit::drain_due(&mut e, at);
    let corr = e
        .pending_probe_for(pid)
        .expect("first Verifying probe in flight");
    let primed = dir_tree_snap(vec![("draft", EntryKind::File, 1)]);
    e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid,
            correlation: corr,
            outcome: ProbeOutcome::SubtreeProven {
                snapshot: primed,
                authority: ProofAuthority::Authoritative,
            },
        }),
        at,
    );

    // `floor` pressure windows: each verify fails Transient (EMFILE), observing nothing — every
    // window re-batches, none counts toward the streak.
    for round in 0..floor {
        at += SETTLE * 2;
        crate::testkit::drain_due(&mut e, at);
        let corr = e
            .pending_probe_for(pid)
            .expect("settle expiry re-drives the verify after the Transient re-batch");
        let out = e.step(
            Input::ProbeResponse(ProbeResponse {
                owner: pid,
                correlation: corr,
                outcome: ProbeOutcome::Failed(ProbeFailure::Transient { errno: 24 }),
            }),
            at,
        );
        assert!(
            out.effects().is_empty(),
            "round {round}: a failed window must not fire",
        );
    }

    // BurstDeadline mid-Batching: `forced = true` + a fresh Verifying probe.
    let bd_id = match e.profiles.get(pid).unwrap().state() {
        ProfileState::Active(ActiveBurst::PreFire(pre), _) => pre.burst_deadline,
        other => panic!("expected Active(PreFire) after the pressure rounds; got {other:?}"),
    };
    let bd_out = e.step(
        Input::TimerExpired {
            profile: pid,
            kind: TimerKind::BurstDeadline,
            id: bd_id,
        },
        burst_start + MAX_SETTLE + Duration::from_millis(1),
    );
    let corr_forced = bd_out
        .probe_ops()
        .iter()
        .find_map(|op| match op {
            ProbeOp::Probe { request } => Some(request.correlation()),
            ProbeOp::Cancel { .. } => None,
        })
        .expect("BurstDeadline drives a fresh forced Verifying probe");

    // The forced sample disagrees with the primed carrier — but the streak witnessed only pressure,
    // so the terminal pairs `hash_channel_disagreed = true` with `retry_streak == 0`: the generic
    // despite-change diagnostic, never the mask hint.
    let final_sample = dir_tree_snap(vec![("draft", EntryKind::File, 2)]);
    let s_out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid,
            correlation: corr_forced,
            outcome: ProbeOutcome::SubtreeProven {
                snapshot: final_sample,
                authority: ProofAuthority::Authoritative,
            },
        }),
        burst_start + MAX_SETTLE + Duration::from_millis(2),
    );

    assert_eq!(s_out.effects().len(), 1, "the bounded fire is unchanged");
    assert!(
        s_out.diagnostics.iter().any(|d| matches!(
            d,
            Diagnostic::QuiescenceCeilingForcedDespiteChange { profile, intent }
                if *profile == pid && *intent == BurstIntent::Standard,
        )),
        "a disagreement tailing a pressure streak keeps the generic diagnostic; got {:?}",
        s_out.diagnostics,
    );
    assert!(
        s_out
            .diagnostics
            .iter()
            .all(|d| !matches!(d, Diagnostic::ChangeOutsideEventMask { .. })),
        "FD pressure must not masquerade as motion outside the event mask; got {:?}",
        s_out.diagnostics,
    );
}

/// The post-fire mirror of the mask-blindspot upgrade: the rebase loop's own `retry_streak`
/// (incremented at every `Rebasing → Settling` loop-back, fresh across the fire boundary) reaches
/// the floor with every window event-silent, and the `RebaseCeiling` terminal's disagreement then
/// emits `ChangeOutsideEventMask` **instead of** `RebaseCeilingForced`. The commit semantics are
/// unchanged: the freshest observation is still pinned as the rebased baseline.
#[test]
fn post_fire_forced_ceiling_after_retry_streak_emits_mask_hint() {
    let floor = crate::transitions::CHANGE_OUTSIDE_MASK_RETRY_FLOOR;
    let mut e = Engine::new();
    let r = e.tree.ensure_root("anchor", ResourceRole::User);
    e.tree.set_kind(r, ResourceKind::Dir);
    let now = Instant::now();
    let (sid, pid) = crate::testkit::attach_structure_only(&mut e, r, now);
    let baseline = dir_tree_snap(vec![]);
    let seed_done = crate::testkit::seed_to_idle(&mut e, pid, &baseline, now);

    // Fire a Standard burst, complete its Effect → Rebasing probe-first.
    let fire_snap = dir_tree_snap(vec![("draft", EntryKind::File, 1)]);
    let fire_start = seed_done + Duration::from_millis(1);
    let stable_out = crate::testkit::drive_standard_n2_until_stable(
        &mut e,
        pid,
        r,
        &[Arc::clone(&fire_snap), Arc::clone(&fire_snap)],
        fire_start,
    );
    let key = stable_out.effects()[0].key();
    let mut at = fire_start + SETTLE * 4 + Duration::from_millis(1);
    e.step(
        Input::EffectComplete(EffectCompletion {
            sub: sid,
            key,
            outcome: EffectOutcome::Ok,
        }),
        at,
    );

    // One priming loop-back plus `floor` disagreement loop-backs: each Rebasing sample hashes
    // differently from the carrier's prior, folding Retry → Settling; the settle expiry re-arms
    // Rebasing for the next sample. The first round's `prior = None` folds `observed_motion: false`
    // and holds the streak; each subsequent round's concrete disagreement bumps it. No FsEvent is
    // absorbed anywhere: the streak never resets.
    for round in 0..=floor {
        let corr = e
            .pending_probe_for(pid)
            .expect("Rebasing probe in flight each round");
        let sample = dir_tree_snap(vec![("draft", EntryKind::File, u64::from(round) + 2)]);
        at += SETTLE * 2;
        let out = e.step(
            Input::ProbeResponse(ProbeResponse {
                owner: pid,
                correlation: corr,
                outcome: ProbeOutcome::SubtreeProven {
                    snapshot: sample,
                    authority: ProofAuthority::Authoritative,
                },
            }),
            at,
        );
        assert!(
            out.diagnostics
                .iter()
                .all(|d| !matches!(d, Diagnostic::RebaseCeilingForced { .. })),
            "round {round}: Retry before the ceiling emits no forced diagnostic",
        );
        // Settling → Rebasing for the next sample — except after the last round, where the ceiling
        // (not the settle timer) drives the terminal probe below.
        if round < floor {
            let settle_id = match e.profiles.get(pid).unwrap().state() {
                ProfileState::Active(ActiveBurst::PostFire(post), _) => match &post.phase {
                    PostFirePhase::Settling { settle_timer } => *settle_timer,
                    other => panic!("expected Settling after Retry; got {other:?}"),
                },
                other => panic!("expected Active(PostFire); got {other:?}"),
            };
            at += SETTLE;
            e.step(
                Input::TimerExpired {
                    profile: pid,
                    kind: TimerKind::PostFireSettle,
                    id: settle_id,
                },
                at,
            );
        }
    }

    // RebaseCeiling mid-Settling: latch the forced terminal + drive the final Rebasing probe.
    let ceiling = rebase_ceiling_timer(&e, pid);
    at += Duration::from_millis(1);
    let ceiling_out = e.step(
        Input::TimerExpired {
            profile: pid,
            kind: TimerKind::RebaseCeiling,
            id: ceiling,
        },
        at,
    );
    let corr_forced = ceiling_out
        .probe_ops()
        .iter()
        .find_map(|op| match op {
            ProbeOp::Probe { request } => Some(request.correlation()),
            ProbeOp::Cancel { .. } => None,
        })
        .expect("RebaseCeiling in Settling drives a fresh forced Rebasing probe");

    // The forced sample still disagrees — `hash_channel_disagreed = true` at `retry_streak ==
    // floor` selects the hint over the generic RebaseCeilingForced.
    let final_sample = dir_tree_snap(vec![("draft", EntryKind::File, u64::from(floor) + 3)]);
    let s_out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid,
            correlation: corr_forced,
            outcome: ProbeOutcome::SubtreeProven {
                snapshot: Arc::clone(&final_sample),
                authority: ProofAuthority::Authoritative,
            },
        }),
        at + Duration::from_millis(1),
    );

    assert!(
        matches!(e.profiles.get(pid).unwrap().state(), ProfileState::Idle),
        "the forced terminal still finishes the burst to Idle",
    );
    assert_eq!(
        baseline_hash(&e, pid),
        Some(final_sample.dir_hash()),
        "the hint changes the diagnostic, never the freshest-observation pin",
    );
    assert!(
        s_out.diagnostics.iter().any(|d| matches!(
            d,
            Diagnostic::ChangeOutsideEventMask { profile, intent, retries }
                if *profile == pid && *intent == BurstIntent::Standard && *retries == floor,
        )),
        "post-fire streak at the floor upgrades the forced terminal to the mask hint; got {:?}",
        s_out.diagnostics,
    );
    assert!(
        s_out
            .diagnostics
            .iter()
            .all(|d| !matches!(d, Diagnostic::RebaseCeilingForced { .. })),
        "the hint replaces RebaseCeilingForced, not alongside it; got {:?}",
        s_out.diagnostics,
    );
}

// ---------- rebase_baseline witness clears at every site ----------

/// Construct an `Active(PreFire)` state populated with default empty per-burst sets and the
/// supplied phase / intent / probe target. Used by witness-clear tests that drive `dispatch_*_ok`
/// directly with a pre-staged Profile.
fn active_pre_fire_burst(
    e: &mut Engine,
    pid: specter_core::ProfileId,
    phase: PreFirePhase,
    intent: BurstIntent,
    now: Instant,
) -> ProfileState {
    let burst_deadline = e
        .timers
        .schedule(now + MAX_SETTLE, pid, TimerKind::BurstDeadline);
    ProfileState::Active(
        ActiveBurst::PreFire(PreFireBurst::new(
            burst_deadline,
            phase,
            intent,
            DirtyProvenance::new(),
            None,
            false,
        )),
        BurstFinish::ReturnToIdle,
    )
}

/// Construct an `Active(PostFire)` state with the supplied phase / intent. Used by witness-clear
/// tests that drive `dispatch_rebase_*` directly with a pre-staged Profile.
fn active_post_fire_burst(
    _e: &mut Engine,
    _pid: specter_core::ProfileId,
    phase: PostFirePhase,
    intent: BurstIntent,
    _now: Instant,
) -> ProfileState {
    ProfileState::Active(
        ActiveBurst::PostFire(PostFireBurst::new(intent, phase, DirtyProvenance::new())),
        BurstFinish::ReturnToIdle,
    )
}

/// Drive an attached Profile into **survival mode** — the post anchor-loss shape: the anchor
/// collapses to `Unclassified` with the pre-loss baseline hash retained as the survival witness
/// (`settled_hash() == Some(witness_snap.dir_hash())`; `baseline()` and `current()` both `None`).
/// Built only from the production `Profile` API, mirroring the engine's take-then-clear loss
/// sequence, so the captured witness is exactly `witness_snap`'s hash.
fn enter_survival_mode(
    e: &mut Engine,
    pid: specter_core::ProfileId,
    witness_snap: Arc<DirSnapshot>,
) {
    let p = e.profiles.get_mut(pid).expect("Profile lives");
    p.install_dir_current(witness_snap);
    p.rebase_baseline();
    p.take_current();
    p.clear_anchor_classification();
}

/// Drive an attached Profile into **active mode**: `baseline = Dir(baseline_snap)`, `current =
/// Dir(current_snap)`, witness `None`, `kind = Some(Dir)` — built only from the production setters.
fn enter_active_mode(
    e: &mut Engine,
    pid: specter_core::ProfileId,
    baseline_snap: Arc<DirSnapshot>,
    current_snap: Arc<DirSnapshot>,
) {
    let p = e.profiles.get_mut(pid).expect("Profile lives");
    p.install_dir_current(baseline_snap);
    p.rebase_baseline();
    p.install_dir_current(current_snap);
}

#[test]
fn dispatch_rebase_ok_consumes_survival_witness() {
    let (mut e, pid, _sid, _anchor, now) = engine_with_attached_sub();
    // Drain the attach-time Seed-Verifying probe before the manual `transition_state` clobber below
    // drops that armed state.
    let _ = e.cancel_all_in_flight_probes();

    let witness_snap = dir_tree_snap(vec![]);
    let witness_hash = witness_snap.dir_hash();
    enter_survival_mode(&mut e, pid, witness_snap);
    let state = active_post_fire_burst(
        &mut e,
        pid,
        PostFirePhase::Rebasing(ProbeSlot::empty()),
        BurstIntent::Standard,
        now,
    );
    if let Some(p) = e.profiles.get_mut(pid) {
        assert_eq!(
            p.settled_hash(),
            Some(witness_hash),
            "precondition: survival mode retains the pre-loss hash as the witness",
        );
        assert!(
            p.baseline().is_none(),
            "precondition: survival mode has no live baseline",
        );
        p.transition_state(state);
    }

    // Rebase grafts a snapshot distinct from the witness so the Witness → Snapshot consume is
    // observable in `settled_hash()`.
    let rebased = dir_tree_snap(vec![("rebased", EntryKind::File, 7)]);
    let rebased_hash = rebased.dir_hash();
    assert_ne!(
        rebased_hash, witness_hash,
        "test setup: rebased snapshot must differ from the witness",
    );
    // An `Stable(StableReason::Natural)` rebase verdict is the consume-the-witness arm:
    // apply_snapshot + rebase_baseline + finish. (The looping / ceiling-terminal arms are exercised
    // by the rebase-loop tests.)
    let mut out = StepOutput::default();
    e.dispatch_rebase_ok(
        pid,
        TreeSnapshot::Dir(rebased),
        QuiescenceVerdict::Stable(StableReason::Natural),
        now,
        &mut out,
    );

    let p = e.profiles.get(pid).expect("Profile lives");
    assert!(
        p.baseline().is_some(),
        "rebase settled the grafted current as the new baseline",
    );
    assert_eq!(
        p.settled_hash(),
        Some(rebased_hash),
        "rebase consumed the survival witness: the settled reference now \
         tracks the new baseline, not the stale pre-loss witness",
    );
}

/// `dispatch_quiescence_ok` contract — pre-fire `Stable(Forced)` arm. The pre-fire forced ceiling
/// fires through the Draining gate regardless of the disagreement bit; the bit selects only the
/// diagnostic:
/// - `hash_channel_disagreed = true` ⇒ exactly one
///   [`Diagnostic::QuiescenceCeilingForcedDespiteChange`].
/// - `hash_channel_disagreed = false` ⇒ zero diagnostics from this arm (the `forced` bit is already
///   operator-visible on the emitted Effect; pre-fire's quiet ceiling stays silent).
///
/// Pins audit Test Gap #3 for the pre-fire forced dispatch.
#[test]
fn dispatch_quiescence_ok_stable_forced_emits_diagnostic_only_when_disagreed() {
    for (disagreed, label) in [(false, "quiet forced"), (true, "loud forced")] {
        let (mut e, pid, _sid, anchor, now) = engine_with_attached_sub();
        let _ = e.cancel_all_in_flight_probes();

        let state = active_pre_fire_burst(
            &mut e,
            pid,
            PreFirePhase::Verifying {
                slot: ProbeSlot::empty(),
                target: anchor,
            },
            BurstIntent::Standard,
            now,
        );
        if let Some(p) = e.profiles.get_mut(pid) {
            p.transition_state(state);
        }

        let snap = dir_tree_snap(vec![("a", EntryKind::File, 1)]);
        let mut out = StepOutput::default();
        e.dispatch_quiescence_ok(
            pid,
            TreeSnapshot::Dir(snap),
            QuiescenceVerdict::Stable(StableReason::Forced {
                hash_channel_disagreed: disagreed,
            }),
            BurstIntent::Standard,
            now,
            &mut out,
        );

        let ceiling_count = out
            .diagnostics
            .iter()
            .filter(|d| matches!(d, Diagnostic::QuiescenceCeilingForcedDespiteChange { .. },))
            .count();
        assert_eq!(
            ceiling_count,
            usize::from(disagreed),
            "{label}: QuiescenceCeilingForcedDespiteChange emits iff \
             hash_channel_disagreed; got diagnostics {:?}",
            out.diagnostics,
        );
        assert!(
            !out.diagnostics
                .iter()
                .any(|d| matches!(d, Diagnostic::QuiescenceCeilingUnreadable { .. },)),
            "{label}: Stable(Forced) never emits QuiescenceCeilingUnreadable; \
             got {:?}",
            out.diagnostics,
        );
    }
}

/// `dispatch_quiescence_ok` contract — pre-fire `Abandon` arm. The bounded ceiling already fired
/// and the walker still refused on `first_unread`; the dispatch must emit exactly one
/// [`Diagnostic::QuiescenceCeilingUnreadable`] carrying the unread path, finish the burst to Idle,
/// and **not** commit (an unread region must never become the dedup / Seed baseline).
#[test]
fn dispatch_quiescence_ok_abandon_emits_unreadable_and_finishes() {
    let (mut e, pid, _sid, anchor, now) = engine_with_attached_sub();
    let _ = e.cancel_all_in_flight_probes();

    let state = active_pre_fire_burst(
        &mut e,
        pid,
        PreFirePhase::Verifying {
            slot: ProbeSlot::empty(),
            target: anchor,
        },
        BurstIntent::Standard,
        now,
    );
    if let Some(p) = e.profiles.get_mut(pid) {
        p.transition_state(state);
    }

    let unread: Arc<std::path::Path> = Arc::from(std::path::Path::new("anchor/sealed"));
    let snap = dir_tree_snap(vec![]);
    let mut out = StepOutput::default();
    e.dispatch_quiescence_ok(
        pid,
        TreeSnapshot::Dir(snap),
        QuiescenceVerdict::Abandon {
            first_unread: Arc::clone(&unread),
        },
        BurstIntent::Standard,
        now,
        &mut out,
    );

    let unreadable: Vec<_> = out
        .diagnostics
        .iter()
        .filter_map(|d| match d {
            Diagnostic::QuiescenceCeilingUnreadable { first_unread, .. } => Some(first_unread),
            _ => None,
        })
        .collect();
    assert_eq!(
        unreadable.len(),
        1,
        "Abandon emits exactly one QuiescenceCeilingUnreadable; got {:?}",
        out.diagnostics,
    );
    assert_eq!(
        unreadable[0].as_ref(),
        unread.as_ref(),
        "Abandon's diagnostic carries the verdict's first_unread verbatim",
    );
    assert!(
        matches!(e.profiles.get(pid).unwrap().state(), ProfileState::Idle),
        "Abandon finishes the burst to Idle",
    );
}

/// `dispatch_rebase_ok` contract — post-fire `Stable(Forced)` arm. The post-fire forced ceiling
/// always commits (graft + `rebase_baseline`) AND emits exactly one
/// [`Diagnostic::RebaseCeilingForced`], carrying the disagreement bit as `observed_change` — loud
/// on both, because no `Effect` records the forced fallback downstream (the principled asymmetry
/// with the pre-fire mirror, which stays silent on the quiet path).
#[test]
fn dispatch_rebase_ok_stable_forced_emits_exactly_one_ceiling_diagnostic() {
    for disagreed in [false, true] {
        let (mut e, pid, _sid, _anchor, now) = engine_with_attached_sub();
        let _ = e.cancel_all_in_flight_probes();

        let baseline = dir_tree_snap(vec![("seed", EntryKind::File, 1)]);
        let current = dir_tree_snap(vec![("seed", EntryKind::File, 2)]);
        enter_active_mode(&mut e, pid, baseline, current);
        let state = active_post_fire_burst(
            &mut e,
            pid,
            PostFirePhase::Rebasing(ProbeSlot::empty()),
            BurstIntent::Standard,
            now,
        );
        if let Some(p) = e.profiles.get_mut(pid) {
            p.transition_state(state);
        }

        let snap = dir_tree_snap(vec![("seed", EntryKind::File, 7)]);
        let snap_hash = snap.dir_hash();
        let mut out = StepOutput::default();
        e.dispatch_rebase_ok(
            pid,
            TreeSnapshot::Dir(snap),
            QuiescenceVerdict::Stable(StableReason::Forced {
                hash_channel_disagreed: disagreed,
            }),
            now,
            &mut out,
        );

        let forced: Vec<bool> = out
            .diagnostics
            .iter()
            .filter_map(|d| match d {
                Diagnostic::RebaseCeilingForced {
                    observed_change, ..
                } => Some(*observed_change),
                _ => None,
            })
            .collect();
        assert_eq!(
            forced,
            vec![disagreed],
            "disagreed={disagreed}: exactly one RebaseCeilingForced, observed_change \
             carries the bit; got {:?}",
            out.diagnostics,
        );
        assert_eq!(
            baseline_hash(&e, pid),
            Some(snap_hash),
            "Stable(Forced) commits + rebases the baseline (the prelude \
             absorbs into the outer Stable(_) arm)",
        );
        assert!(
            matches!(e.profiles.get(pid).unwrap().state(), ProfileState::Idle),
            "Stable(Forced) finishes the burst to Idle",
        );
    }
}

#[test]
fn seed_recovery_seal_consumes_survival_witness() {
    let (mut e, pid, _sid, anchor, now) = engine_with_attached_sub();
    // Drain the attach-time Seed-Verifying probe before the manual `transition_state` clobber below
    // drops that armed state.
    let _ = e.cancel_all_in_flight_probes();

    // Survival mode at entry (no live baseline, witness populated); empty fired_subs ⇒ no drift —
    // but Seed still rebases, consuming the witness.
    let witness_snap = dir_tree_snap(vec![]);
    let witness_hash = witness_snap.dir_hash();
    enter_survival_mode(&mut e, pid, witness_snap);
    let state = active_pre_fire_burst(
        &mut e,
        pid,
        PreFirePhase::Verifying {
            slot: ProbeSlot::empty(),
            target: anchor,
        },
        BurstIntent::Seed,
        now,
    );
    if let Some(p) = e.profiles.get_mut(pid) {
        p.transition_state(state);
    }

    let rebased = dir_tree_snap(vec![("rebased", EntryKind::File, 7)]);
    let rebased_hash = rebased.dir_hash();
    assert_ne!(
        rebased_hash, witness_hash,
        "test setup: rebased snapshot must differ from the witness",
    );
    let mut out = StepOutput::default();
    e.dispatch_quiescence_ok(
        pid,
        TreeSnapshot::Dir(rebased),
        QuiescenceVerdict::Stable(StableReason::Natural),
        BurstIntent::Seed,
        now,
        &mut out,
    );

    let p = e.profiles.get(pid).unwrap();
    assert!(
        p.baseline().is_some(),
        "no-drift Seed-Ok rebased the grafted current",
    );
    assert_eq!(
        p.settled_hash(),
        Some(rebased_hash),
        "no-drift Seed-Ok consumed the survival witness (settled now \
         tracks the new baseline)",
    );
}

#[test]
fn seed_recovery_fire_consumes_survival_witness_eagerly() {
    // Sub fired pre-loss; anchor lost; recovery Seed-Ok with drift. The eager consume on the
    // recovery-drift fire (the `EmitMode::SeedDrift` seal in `fire_and_settle`, before
    // transition_to_awaiting) keeps the baseline ⊕ witness exclusivity holding at every step
    // boundary, not just at later consume sites.
    let (mut e, pid, sid, anchor, now) = engine_with_attached_sub();
    // Drain the attach-time Seed-Verifying probe before the manual `transition_state` clobber below
    // drops that armed state.
    let _ = e.cancel_all_in_flight_probes();

    // Survival-mode drift setup: a witness snapshot whose hash won't match the post-graft current ⇒
    // the drift signal triggers; pre-loss fire history on `sid` narrows the SeedDrift filter to
    // this Sub. `mark_fired` is idempotent — "sid fired pre-loss" needs exactly one mark.
    let witness_snap = dir_tree_snap(vec![("pre", EntryKind::File, 1)]);
    let witness_hash = witness_snap.dir_hash();
    enter_survival_mode(&mut e, pid, witness_snap);
    let state = active_pre_fire_burst(
        &mut e,
        pid,
        PreFirePhase::Verifying {
            slot: ProbeSlot::empty(),
            target: anchor,
        },
        BurstIntent::Seed,
        now,
    );
    e.subs.mark_fired(sid);
    if let Some(p) = e.profiles.get_mut(pid) {
        p.transition_state(state);
    }

    let regrafted = dir_tree_snap(vec![]);
    let regrafted_hash = regrafted.dir_hash();
    assert_ne!(
        regrafted_hash, witness_hash,
        "test setup: post-graft current must differ from the witness to drift",
    );
    let mut out = StepOutput::default();
    e.dispatch_quiescence_ok(
        pid,
        TreeSnapshot::Dir(regrafted),
        QuiescenceVerdict::Stable(StableReason::Natural),
        BurstIntent::Seed,
        now,
        &mut out,
    );

    // The recovery-drift fire must have emitted one Effect.
    assert_eq!(
        out.effects().len(),
        1,
        "recovery-drift fire emitted one Effect"
    );

    let p = e.profiles.get(pid).unwrap();
    assert!(
        p.baseline().is_some(),
        "drift branch rebased the grafted current as baseline",
    );
    assert_eq!(
        p.settled_hash(),
        Some(regrafted_hash),
        "drift Seed-Ok eagerly consumed the survival witness (settled now \
         tracks the new baseline)",
    );
}

// ---- PerFileDriftDroppedOnRecovery: loss→recovery honesty signal ----

/// Attach a `PerStableFile` Sub at the same `(anchor, config_hash)` as `engine_with_attached_sub`'s
/// `SubtreeRoot` Sub so both share one Profile. Identical `ProfileIdentity` (config / max_settle /
/// events) ⇒ same `ProfileId`; the scope differs, which is the only axis the
/// `has_per_stable_file_sub` gate reads.
fn attach_per_stable_file_sibling(
    e: &mut Engine,
    anchor: ResourceId,
    pid: specter_core::ProfileId,
    now: Instant,
) -> specter_core::SubId {
    let req = SubAttachRequest {
        anchor: SubAttachAnchor::Resource(anchor),
        identity: ProfileIdentity::new(
            ScanConfig::builder().recursive(true).build(),
            MAX_SETTLE,
            DEFAULT_EVENTS,
        ),
        params: SubParams::spawn(
            "per-file-sibling".into(),
            empty_program(),
            EffectScope::PerStableFile,
            SETTLE,
            false,
        ),
    };
    let out = e.step(Input::AttachSub(req), now);
    let sid = specter_core::testkit::first_attached_sub(&out).expect("per-file sibling attached");
    assert_eq!(
        e.subs.get(sid).unwrap().profile(),
        pid,
        "per-file sibling shares the original Profile (same anchor + config_hash)",
    );
    assert!(
        e.subs.has_per_stable_file_sub(pid),
        "precondition: Profile now carries a PerStableFile Sub",
    );
    sid
}

/// (a) Positive: anchor loss → real content drift across the loss window → recovery Seed-Ok with a
/// `PerStableFile` Sub attached ⇒ exactly one `PerFileDriftDroppedOnRecovery` for the Profile. No
/// fired Sub is required — the diagnostic is scope+drift gated, not drift-branch gated (a
/// PerFile-only Profile never records a fire yet is exactly the case to flag).
#[test]
fn per_file_drift_dropped_on_recovery_emits_once_on_real_drift() {
    let (mut e, pid, _sid, anchor, now) = engine_with_attached_sub();
    let _ = e.cancel_all_in_flight_probes();
    attach_per_stable_file_sibling(&mut e, anchor, pid, now);
    let _ = e.cancel_all_in_flight_probes();

    // Survival mode: pre-loss hash retained as the witness; the recovery probe lands a `current`
    // whose hash differs ⇒ real drift.
    let witness_snap = dir_tree_snap(vec![("pre", EntryKind::File, 1)]);
    let witness_hash = witness_snap.dir_hash();
    enter_survival_mode(&mut e, pid, witness_snap);
    let state = active_pre_fire_burst(
        &mut e,
        pid,
        PreFirePhase::Verifying {
            slot: ProbeSlot::empty(),
            target: anchor,
        },
        BurstIntent::Seed,
        now,
    );
    if let Some(p) = e.profiles.get_mut(pid) {
        p.transition_state(state);
    }

    let recovered = dir_tree_snap(vec![("post", EntryKind::File, 2)]);
    assert_ne!(
        recovered.dir_hash(),
        witness_hash,
        "test setup: recovered tree must differ from the pre-loss witness to drift",
    );
    let mut out = StepOutput::default();
    e.dispatch_quiescence_ok(
        pid,
        TreeSnapshot::Dir(recovered),
        QuiescenceVerdict::Stable(StableReason::Natural),
        BurstIntent::Seed,
        now,
        &mut out,
    );

    assert_eq!(
        out.diagnostics
            .iter()
            .filter(|d| matches!(
                d,
                Diagnostic::PerFileDriftDroppedOnRecovery { profile } if *profile == pid
            ))
            .count(),
        1,
        "real loss-window drift + PerStableFile Sub ⇒ exactly one \
         PerFileDriftDroppedOnRecovery for the Profile",
    );
}

/// (b) Byte-identical recovery: same loss→recovery shape, but the recovered tree hash equals the
/// pre-loss witness ⇒ zero `PerFileDriftDroppedOnRecovery` (a byte-identical recovery dropped
/// nothing). Pins the `current.hash() != witness` gate.
#[test]
fn per_file_drift_dropped_on_recovery_silent_on_byte_identical_recovery() {
    let (mut e, pid, _sid, anchor, now) = engine_with_attached_sub();
    let _ = e.cancel_all_in_flight_probes();
    attach_per_stable_file_sibling(&mut e, anchor, pid, now);
    let _ = e.cancel_all_in_flight_probes();

    let witness_snap = dir_tree_snap(vec![("same", EntryKind::File, 1)]);
    let witness_hash = witness_snap.dir_hash();
    enter_survival_mode(&mut e, pid, witness_snap);
    let state = active_pre_fire_burst(
        &mut e,
        pid,
        PreFirePhase::Verifying {
            slot: ProbeSlot::empty(),
            target: anchor,
        },
        BurstIntent::Seed,
        now,
    );
    if let Some(p) = e.profiles.get_mut(pid) {
        p.transition_state(state);
    }

    // Same shape ⇒ identical dir_hash (synthetic ctors are deterministic): a byte-identical recovery.
    let recovered = dir_tree_snap(vec![("same", EntryKind::File, 1)]);
    assert_eq!(
        recovered.dir_hash(),
        witness_hash,
        "test setup: recovered tree must be byte-identical to the witness",
    );
    let mut out = StepOutput::default();
    e.dispatch_quiescence_ok(
        pid,
        TreeSnapshot::Dir(recovered),
        QuiescenceVerdict::Stable(StableReason::Natural),
        BurstIntent::Seed,
        now,
        &mut out,
    );

    assert_eq!(
        out.diagnostics
            .iter()
            .filter(|d| matches!(
                d,
                Diagnostic::PerFileDriftDroppedOnRecovery { profile } if *profile == pid
            ))
            .count(),
        0,
        "byte-identical recovery dropped nothing ⇒ no \
         PerFileDriftDroppedOnRecovery",
    );
}

/// (c) Scope gate: same loss→drift→recovery, but the Profile has only a `SubtreeRoot` Sub (no
/// `PerStableFile`) ⇒ zero `PerFileDriftDroppedOnRecovery`. Regression guard against collapsing the
/// scope scan into `Profile::has_per_file_fds` — that predicate is events-mask derived and would
/// false-positive a content-watching Subtree-only Profile.
#[test]
fn per_file_drift_dropped_on_recovery_gated_by_per_stable_file_scope() {
    let (mut e, pid, _sid, anchor, now) = engine_with_attached_sub();
    let _ = e.cancel_all_in_flight_probes();
    assert!(
        !e.subs.has_per_stable_file_sub(pid),
        "precondition: Profile has only the SubtreeRoot Sub",
    );

    let witness_snap = dir_tree_snap(vec![("pre", EntryKind::File, 1)]);
    let witness_hash = witness_snap.dir_hash();
    enter_survival_mode(&mut e, pid, witness_snap);
    let state = active_pre_fire_burst(
        &mut e,
        pid,
        PreFirePhase::Verifying {
            slot: ProbeSlot::empty(),
            target: anchor,
        },
        BurstIntent::Seed,
        now,
    );
    if let Some(p) = e.profiles.get_mut(pid) {
        p.transition_state(state);
    }

    let recovered = dir_tree_snap(vec![("post", EntryKind::File, 2)]);
    assert_ne!(
        recovered.dir_hash(),
        witness_hash,
        "test setup: recovered tree must differ from the witness (real drift)",
    );
    let mut out = StepOutput::default();
    e.dispatch_quiescence_ok(
        pid,
        TreeSnapshot::Dir(recovered),
        QuiescenceVerdict::Stable(StableReason::Natural),
        BurstIntent::Seed,
        now,
        &mut out,
    );

    assert_eq!(
        out.diagnostics
            .iter()
            .filter(|d| matches!(
                d,
                Diagnostic::PerFileDriftDroppedOnRecovery { profile } if *profile == pid
            ))
            .count(),
        0,
        "real drift but no PerStableFile Sub ⇒ the scope gate suppresses \
         PerFileDriftDroppedOnRecovery",
    );
}

// ---- seed_drift_observed: baseline-or-witness contract ----

/// Fresh Profile reports no drift: `fired_subs` is empty by construction and there is no settled
/// reference yet. Pins the "fresh Seed never fires Effect" contract — without the
/// `fired_subs.is_empty()` short-circuit, a Profile with no settled reference AND a Some current
/// would still fall through to the `match p.settled_hash()` arm; the short-circuit is the
/// load-bearing guard for the fresh-attach case.
#[test]
fn seed_drift_observed_returns_false_for_fresh_profile() {
    let (mut e, pid, _sid, _anchor, _now) = engine_with_attached_sub();
    assert!(!e.subs.any_fired(pid), "precondition: no fire history");
    let p = e.profiles.get(pid).expect("Profile lives post-attach");
    assert!(
        p.settled_hash().is_none(),
        "precondition: no settled reference"
    );
    assert!(
        p.baseline().is_none(),
        "precondition: baseline cleared post-attach"
    );

    assert!(
        !e.seed_drift_observed(pid),
        "fresh Profile (no fire history, no settled state) reports no drift",
    );
    let _ = e.cancel_all_in_flight_probes();
}

/// Survival-mode drift: anchor loss collapsed the anchor to `Unclassified`, retaining the pre-loss
/// baseline hash as the survival witness. The recovery Seed-Ok lands a new `current` whose hash
/// differs from that witness — drift detected, conservative re-fire required. Pins the witness arm
/// of `settled_hash()`.
#[test]
fn seed_drift_observed_returns_true_on_post_recovery_drift() {
    let (mut e, pid, sid, _anchor, _now) = engine_with_attached_sub();
    let snap = dir_tree_snap(vec![("file", EntryKind::File, 1)]);
    let witness_snap = dir_tree_snap(vec![]);
    assert_ne!(
        witness_snap.dir_hash(),
        snap.dir_hash(),
        "test setup: pre-loss witness and recovery current must differ",
    );

    // Survival mode: anchor loss stashed the pre-loss baseline.hash() into the witness; the
    // recovery probe lands a `current` whose hash differs from that witness ⇒ drift.
    enter_survival_mode(&mut e, pid, witness_snap);
    e.subs.mark_fired(sid);
    if let Some(p) = e.profiles.get_mut(pid) {
        p.install_dir_current(snap);
    }

    assert!(
        e.seed_drift_observed(pid),
        "survival mode: witness != current.hash() ⇒ drift detected",
    );
    let _ = e.cancel_all_in_flight_probes();
}

/// Active-mode drift: `baseline().is_some()` (no anchor loss has occurred), so `settled_hash()`
/// returns the live baseline's hash — a separate survival witness alongside a held baseline is not
/// representable in the anchor sum. Drift derives from `baseline.hash() != current.hash()`. Covers
/// the `on_sensor_overflow` reseed path: overflow does not go through `discard_anchor_state`, so
/// the baseline (hence the settled reference) persists and the single `settled_hash()` oracle still
/// yields the conservative re-fire verdict.
#[test]
fn seed_drift_observed_returns_true_on_active_mode_drift() {
    let (mut e, pid, sid, _anchor, _now) = engine_with_attached_sub();
    let baseline_snap = dir_tree_snap(vec![("a", EntryKind::File, 1)]);
    let current_snap = dir_tree_snap(vec![("b", EntryKind::File, 1)]);
    assert_ne!(
        baseline_snap.dir_hash(),
        current_snap.dir_hash(),
        "test setup: baseline and current must have distinct hashes",
    );

    enter_active_mode(&mut e, pid, baseline_snap, current_snap);
    e.subs.mark_fired(sid);

    assert!(
        e.seed_drift_observed(pid),
        "active-mode (overflow) drift: baseline.hash() != current.hash() ⇒ \
         drift detected ⇒ conservative re-fire",
    );
    let _ = e.cancel_all_in_flight_probes();
}

/// Active-mode no-drift: `baseline.hash() == current.hash()` — overflow happened to coincide with
/// no actual disk change. The witness is `None`, the baseline arm runs and reports no drift. No
/// conservative re-fire — `baseline` still represents reality.
#[test]
fn seed_drift_observed_returns_false_when_active_mode_baseline_matches_current() {
    let (mut e, pid, sid, _anchor, _now) = engine_with_attached_sub();
    let snap = dir_tree_snap(vec![("file", EntryKind::File, 1)]);

    enter_active_mode(&mut e, pid, snap.clone(), snap);
    e.subs.mark_fired(sid);

    assert!(
        !e.seed_drift_observed(pid),
        "active mode, baseline.hash() == current.hash() ⇒ no drift \
         (overflow without disk change)",
    );
    let _ = e.cancel_all_in_flight_probes();
}

// ---- fire-history relocation invariant (per-Sub `Sub.has_fired`) ----

/// Pins the three load-bearing properties of relocating the Effect fire-history from a per-Profile
/// container to a per-Sub `bool`:
///
///  1. **B1 / SeedDrift read `Sub.has_fired`.** A real burst that fires sets the emitting Sub's
///     flag; `SubRegistry::any_fired` / `fired_in` (the B1-suppress and SeedDrift-filter bases)
///     observe it through the registry, not through any Profile container.
///  2. **A detached Sub's flag dies with its slotmap entry — no purge needed.** After the Sub
///     fired, detach it (a sibling Sub keeps the Profile alive) and drive a survival-mode *drift*
///     Seed-Ok. It must NOT re-fire: `fired_in(pid)` is empty because the detached Sub's
///     `has_fired` died with `subs.remove`, and the surviving sibling never fired. There is no
///     per-Profile fire container to purge, and none is touched.
///  3. **A fresh attach starts unfired.** The sibling attached after the original has `has_fired ==
///     false`.
#[test]
fn fire_history_is_per_sub_detach_drops_it_no_purge() {
    let (mut e, pid, sid_a, anchor, now) = engine_with_attached_sub();
    // Drain the attach-time Seed-Verifying probe before later manual state manipulation.
    let _ = e.cancel_all_in_flight_probes();

    // Property 3 (part a): the freshly attached Sub starts unfired.
    assert!(
        !e.subs.get(sid_a).unwrap().has_fired(),
        "fresh attach: sid_a starts unfired",
    );
    assert!(
        !e.subs.any_fired(pid),
        "fresh Profile: no Sub has fired (B1/SeedDrift basis is empty)",
    );

    // Attach a second Sub at the *same* (anchor, config_hash) so it shares this Profile — it keeps
    // the Profile alive across sid_a's detach below. Identical request shape ⇒ same ProfileId.
    let req_b = SubAttachRequest {
        anchor: SubAttachAnchor::Resource(anchor),
        identity: ProfileIdentity::new(
            ScanConfig::builder().recursive(true).build(),
            MAX_SETTLE,
            DEFAULT_EVENTS,
        ),
        params: SubParams::spawn(
            "sibling".into(),
            empty_program(),
            EffectScope::SubtreeRoot,
            SETTLE,
            false,
        ),
    };
    let out_b = e.step(Input::AttachSub(req_b), now);
    let sid_b = specter_core::testkit::first_attached_sub(&out_b).expect("sibling attached");
    assert_eq!(
        e.subs.get(sid_b).unwrap().profile(),
        pid,
        "sibling shares the original Profile (same anchor + config_hash)",
    );
    // Property 3 (part b): the sibling attaches unfired.
    assert!(
        !e.subs.get(sid_b).unwrap().has_fired(),
        "fresh attach: sibling starts unfired",
    );
    let _ = e.cancel_all_in_flight_probes();

    // Property 1: mark sid_a fired (the post-emit B1 bookkeeping the SubtreeRoot emit arm performs)
    // and observe it through the registry — the exact signal `seed_drift_observed` / B1 read.
    e.subs.mark_fired(sid_a);
    assert!(
        e.subs.get(sid_a).unwrap().has_fired(),
        "Property 1: B1/SeedDrift fire-history reads Sub.has_fired",
    );
    assert!(
        e.subs.any_fired(pid),
        "Property 1: any_fired observes the per-Sub flag (B1 basis)",
    );
    assert_eq!(
        e.subs.fired_in(pid).as_slice(),
        &[sid_a],
        "Property 1: fired_in yields exactly the fired Sub (SeedDrift filter)",
    );

    // Set up a survival-mode *drift* scenario: the survival witness carries a pre-loss hash; the
    // post-recovery `current` differs from it ⇒ `seed_drift_observed` is true *while sid_a is
    // fired*. (The witness must differ from `current` for drift; matches the working
    // `seed_drift_observed_returns_true_on_post_recovery_drift` setup.)
    let witness_snap = dir_tree_snap(vec![("pre", EntryKind::File, 1)]);
    let current_snap = dir_tree_snap(vec![("post", EntryKind::File, 2)]);
    assert_ne!(
        witness_snap.dir_hash(),
        current_snap.dir_hash(),
        "test setup: witness and recovery current must differ",
    );
    enter_survival_mode(&mut e, pid, witness_snap);
    if let Some(p) = e.profiles.get_mut(pid) {
        p.install_dir_current(current_snap);
    }
    assert!(
        e.seed_drift_observed(pid),
        "precondition: survival-mode drift detected while sid_a is fired",
    );

    // Property 2: detach sid_a. The Profile survives (sibling sid_b remains). sid_a's `has_fired`
    // died with its slotmap entry — no per-Profile purge exists or runs.
    let _ = e.step(Input::DetachSub(sid_a), now);
    assert!(
        e.profiles.get(pid).is_some(),
        "Profile survives sid_a detach (sibling keeps it alive)",
    );
    assert!(
        e.subs.get(sid_a).is_none(),
        "sid_a's slotmap entry (and its has_fired) is gone",
    );
    assert!(
        !e.subs.any_fired(pid),
        "Property 2: no live Sub has fired — sid_a's flag died with it, \
         sibling never fired; no purge was needed",
    );
    assert!(
        e.subs.fired_in(pid).is_empty(),
        "Property 2: SeedDrift filter is empty post-detach",
    );

    // A recovery Seed-Ok now must NOT re-fire: the tree still differs from the witness, but
    // `any_fired` is false post-detach (sid_a's flag died with it; sid_b never fired), so
    // `classify_consequence` yields `SilentPin` — seal-and-finish, no fire. This is the behavioural
    // proof that a detached Sub cannot be re-fired and that no purge is required to achieve it.
    let regrafted = dir_tree_snap(vec![]);
    let mut out = StepOutput::default();
    e.dispatch_quiescence_ok(
        pid,
        TreeSnapshot::Dir(regrafted),
        QuiescenceVerdict::Stable(StableReason::Natural),
        BurstIntent::Seed,
        now,
        &mut out,
    );
    assert!(
        out.effects().is_empty(),
        "Property 2: drift Seed-Ok fires nothing — the only Sub that \
         had fired was detached; its flag died with it, no purge needed",
    );
    let _ = e.cancel_all_in_flight_probes();
}

// ---------- absorb: runtime fold-without-fire ----------

/// Drive an already-Idle Profile through one Standard burst to its quiescence verdict, pinning
/// against `snap`. FsEvent at `root` → Settle expiry → single Authoritative response. Returns the
/// verdict `StepOutput` (the fire/fold emission) so callers assert effects + diagnostics. Births
/// the burst at `fs_event_at`; the caller arms the window before this call (birth-latch). The
/// retro-latch path arms mid-Batching and drives the steps inline.
fn drive_standard_verdict(
    e: &mut Engine,
    pid: specter_core::ProfileId,
    root: ResourceId,
    snap: Arc<DirSnapshot>,
    fs_event_at: Instant,
) -> StepOutput {
    let _ = e.step(
        Input::FsEvent {
            resource: root,
            event: FsEvent::StructureChanged,
        },
        fs_event_at,
    );
    let settle_timer = match e.profiles.get(pid).unwrap().state() {
        ProfileState::Active(ActiveBurst::PreFire(pre), _) => match &pre.phase {
            PreFirePhase::Batching { settle_timer } => *settle_timer,
            other => panic!("expected Standard Batching; got {other:?}"),
        },
        other => panic!("expected Active(PreFire); got {other:?}"),
    };
    let _ = e.step(
        Input::TimerExpired {
            profile: pid,
            kind: TimerKind::Settle,
            id: settle_timer,
        },
        fs_event_at + SETTLE,
    );
    let correlation = e
        .pending_probe_for(pid)
        .expect("Verifying probe in flight after Settle expiry");
    e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid,
            correlation,
            outcome: ProbeOutcome::SubtreeProven {
                snapshot: snap,
                authority: ProofAuthority::Authoritative,
            },
        }),
        fs_event_at + SETTLE + Duration::from_millis(1),
    )
}

/// `baseline().hash() == current().hash()` for a Dir-anchored Profile — the post-fold witness that
/// the silent seal advanced the baseline onto the folded sample (the rebase family, not the
/// suppress family).
fn baseline_equals_current(e: &Engine, pid: specter_core::ProfileId) -> bool {
    let p = e.profiles.get(pid).unwrap();
    match (p.baseline(), p.current()) {
        (Some(b), Some(c)) => b.hash() == c.hash(),
        _ => false,
    }
}

/// A Standard burst born under a live `absorb` window folds instead of firing: the firing `base`
/// (`StandardFire`) is overridden to `AbsorbFold` at `classify_consequence`. No Effect; one
/// `QuiescenceAbsorbed`; the baseline advances onto the (drifted) folded sample; the Sub's fire
/// history is untouched; the Profile's `absorb_count` bumps to 1.
#[test]
fn standard_burst_born_under_window_folds_without_firing() {
    let (mut e, pid, sid, root, _now) = engine_with_attached_sub();
    complete_seed_burst(&mut e, pid);
    let now = Instant::now();

    // Arm the window BEFORE the FsEvent births the Standard burst, so the birth consult latches it.
    e.step(
        Input::ArmAbsorb {
            profile: pid,
            duration: None,
        },
        now,
    );

    // A drifted snapshot — a new child the baseline lacks. A fresh Sub on this burst would fire
    // (`StandardFire`); the fold advances the baseline onto the drift without emitting.
    let drift = dir_tree_snap(vec![("echo.rs", EntryKind::File, 7)]);
    let out = drive_standard_verdict(&mut e, pid, root, Arc::clone(&drift), now);

    assert!(
        out.effects().is_empty(),
        "fold-latched burst emits no Effect; got {:?}",
        out.effects(),
    );
    assert!(
        out.diagnostics
            .iter()
            .any(|d| matches!(d, Diagnostic::QuiescenceAbsorbed { profile } if *profile == pid)),
        "fold emits QuiescenceAbsorbed",
    );
    assert!(
        matches!(e.profiles.get(pid).unwrap().state(), ProfileState::Idle),
        "fold finishes the burst to Idle via the silent seal",
    );
    assert!(
        baseline_equals_current(&e, pid),
        "fold advances the baseline onto the folded drift (rebase family)",
    );
    let post_baseline = match e.profiles.get(pid).unwrap().baseline() {
        Some(TreeSnapshot::Dir(arc)) => arc.dir_hash(),
        _ => panic!("baseline is Some(Dir) post-fold"),
    };
    assert_eq!(
        post_baseline,
        drift.dir_hash(),
        "the advanced baseline is the drifted sample, not the pre-fold empty tree",
    );
    let sub = e.subs.get(sid).unwrap();
    assert!(
        !sub.has_fired(),
        "a fold is not a fire — has_fired untouched"
    );
    assert_eq!(
        sub.fire_history().unwrap().fire_count,
        0,
        "a fold does not bump fire_count"
    );
    assert_eq!(
        e.profiles.get(pid).unwrap().absorb_count(),
        1,
        "the fold bumps the Profile's absorb_count",
    );
}

/// Reverse race: the window is armed AFTER the events arrive (the burst is already in Batching).
/// `arm_absorb` retro-latches the in-flight pre-fire burst, so it folds at its verdict rather than
/// firing.
#[test]
fn arming_window_retro_latches_in_flight_batching_burst() {
    let (mut e, pid, sid, root, _now) = engine_with_attached_sub();
    complete_seed_burst(&mut e, pid);
    let now = Instant::now();

    // FsEvent births the Standard burst (in Batching) BEFORE any window.
    let _ = e.step(
        Input::FsEvent {
            resource: root,
            event: FsEvent::StructureChanged,
        },
        now,
    );
    assert!(
        !e.profiles.get(pid).unwrap().state().burst_fold_latched(),
        "burst is unlatched before the arm (born without a window)",
    );
    let settle_timer = crate::testkit::batching_settle_id(&e, pid);

    // Arm the window while the burst is in Batching — the retro-latch.
    e.step(
        Input::ArmAbsorb {
            profile: pid,
            duration: None,
        },
        now,
    );
    assert!(
        e.profiles.get(pid).unwrap().state().burst_fold_latched(),
        "arm_absorb retro-latches the in-flight Batching burst",
    );

    // Drive the retro-latched burst to its verdict → folds.
    let _ = e.step(
        Input::TimerExpired {
            profile: pid,
            kind: TimerKind::Settle,
            id: settle_timer,
        },
        now + SETTLE,
    );
    let correlation = e
        .pending_probe_for(pid)
        .expect("Verifying probe in flight after Settle expiry");
    let drift = dir_tree_snap(vec![("echo.rs", EntryKind::File, 7)]);
    let out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid,
            correlation,
            outcome: ProbeOutcome::SubtreeProven {
                snapshot: drift,
                authority: ProofAuthority::Authoritative,
            },
        }),
        now + SETTLE + Duration::from_millis(1),
    );

    assert!(
        out.effects().is_empty(),
        "retro-latched burst folds, no Effect"
    );
    assert!(
        out.diagnostics
            .iter()
            .any(|d| matches!(d, Diagnostic::QuiescenceAbsorbed { profile } if *profile == pid)),
        "retro-latched fold emits QuiescenceAbsorbed",
    );
    assert_eq!(e.profiles.get(pid).unwrap().absorb_count(), 1);
    assert!(!e.subs.get(sid).unwrap().has_fired());
}

/// `ConsumeOnFirst` (the `None`-duration default) retires the window on the first fold. A second,
/// separate episode then fires normally — the one-shot cover is spent.
#[test]
fn consume_on_first_folds_once_then_fires() {
    let (mut e, pid, sid, root, _now) = engine_with_attached_sub();
    complete_seed_burst(&mut e, pid);
    let t0 = Instant::now();

    e.step(
        Input::ArmAbsorb {
            profile: pid,
            duration: None,
        },
        t0,
    );
    let drift1 = dir_tree_snap(vec![("a.rs", EntryKind::File, 1)]);
    let fold_out = drive_standard_verdict(&mut e, pid, root, drift1, t0);
    assert!(fold_out.effects().is_empty(), "first episode folds");
    assert!(
        e.profiles.get(pid).unwrap().absorb_window().is_none(),
        "ConsumeOnFirst retires the window on the first fold",
    );
    assert_eq!(e.profiles.get(pid).unwrap().absorb_count(), 1);
    assert!(!e.subs.get(sid).unwrap().has_fired(), "fold did not fire");

    // Second, separate episode — no live window ⇒ fires normally.
    let t1 = t0 + SETTLE * 10;
    let drift2 = dir_tree_snap(vec![
        ("a.rs", EntryKind::File, 1),
        ("b.rs", EntryKind::File, 2),
    ]);
    let fire_out = drive_standard_verdict(&mut e, pid, root, drift2, t1);
    assert_eq!(
        fire_out.effects().len(),
        1,
        "the consumed window does not cover the next episode — it fires",
    );
    assert_eq!(
        e.profiles.get(pid).unwrap().absorb_count(),
        1,
        "a real fire does not bump absorb_count",
    );
    assert!(e.subs.get(sid).unwrap().has_fired(), "second episode fired");
}

/// `PersistUntil` (a `Some(duration)` window) survives folds: two separate episodes within the
/// window both fold. Once the clock passes `expiry`, the window reads inert and a burst fires.
#[test]
fn persist_until_folds_repeatedly_then_fires_after_expiry() {
    let (mut e, pid, sid, root, _now) = engine_with_attached_sub();
    complete_seed_burst(&mut e, pid);
    let t0 = Instant::now();
    // A window wide enough to span both fold episodes but expire before the third. Each episode
    // consumes ~SETTLE of clock in the helper.
    let window = SETTLE * 5;

    e.step(
        Input::ArmAbsorb {
            profile: pid,
            duration: Some(window),
        },
        t0,
    );

    let drift1 = dir_tree_snap(vec![("a.rs", EntryKind::File, 1)]);
    let fold1 = drive_standard_verdict(&mut e, pid, root, drift1, t0);
    assert!(
        fold1.effects().is_empty(),
        "first episode folds under PersistUntil"
    );
    assert!(
        e.profiles.get(pid).unwrap().absorb_window().is_some(),
        "PersistUntil survives the first fold",
    );

    // Second episode, still inside the window (t0 + SETTLE < expiry).
    let t1 = t0 + SETTLE + Duration::from_millis(1);
    let drift2 = dir_tree_snap(vec![
        ("a.rs", EntryKind::File, 1),
        ("b.rs", EntryKind::File, 2),
    ]);
    let fold2 = drive_standard_verdict(&mut e, pid, root, drift2, t1);
    assert!(
        fold2.effects().is_empty(),
        "second episode folds under PersistUntil"
    );
    assert_eq!(
        e.profiles.get(pid).unwrap().absorb_count(),
        2,
        "both folds bumped the count",
    );
    assert!(!e.subs.get(sid).unwrap().has_fired(), "no fold fired");

    // Third episode, born after expiry ⇒ window inert ⇒ fires.
    let t2 = t0 + window + Duration::from_millis(1);
    let drift3 = dir_tree_snap(vec![
        ("a.rs", EntryKind::File, 1),
        ("c.rs", EntryKind::File, 3),
    ]);
    let fire_out = drive_standard_verdict(&mut e, pid, root, drift3, t2);
    assert_eq!(
        fire_out.effects().len(),
        1,
        "a burst born past the PersistUntil expiry fires",
    );
    assert!(
        e.subs.get(sid).unwrap().has_fired(),
        "post-expiry episode fired"
    );
}

/// Cold-Seed redundancy: a fold-latched Cold Seed (no driving event) resolves to `SilentPin`, not
/// `AbsorbFold` — a Seed that owes no first-fire is not a firing `base`, so the override never
/// engages. The window is therefore NOT consumed and survives for the first genuinely-fireable
/// burst, which then consumes it.
#[test]
fn cold_seed_under_window_stays_silent_pin_and_preserves_window() {
    let (mut e, pid, sid, root, _now) = engine_with_attached_sub();
    complete_seed_burst(&mut e, pid);
    let now = Instant::now();

    // Arm a one-shot window on the now-Idle Profile.
    e.step(
        Input::ArmAbsorb {
            profile: pid,
            duration: None,
        },
        now,
    );

    // SensorOverflow reseeds the Idle Profile into a cold Verifying-first Seed burst, born
    // fold-latched (the window is live at its birth).
    let overflow_out = e.step(
        Input::SensorOverflow {
            scope: OverflowScope::Global,
        },
        now,
    );
    assert!(
        e.profiles.get(pid).unwrap().state().burst_fold_latched(),
        "the cold Seed is born fold-latched under the live window",
    );
    let _ = overflow_out;

    // Answer the cold Seed's probe Authoritative — no driving event, so the base is SilentPin (not
    // firing) and the fold override is inert.
    let correlation = e
        .pending_probe_for(pid)
        .expect("cold Seed Verifying probe in flight");
    let seed_out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid,
            correlation,
            outcome: ProbeOutcome::SubtreeProven {
                snapshot: dir_tree_snap(vec![]),
                authority: ProofAuthority::Authoritative,
            },
        }),
        now,
    );
    assert!(seed_out.effects().is_empty(), "a Seed never fires");
    assert!(
        !seed_out
            .diagnostics
            .iter()
            .any(|d| matches!(d, Diagnostic::QuiescenceAbsorbed { .. })),
        "a redundant Cold Seed pins silently — no QuiescenceAbsorbed",
    );
    assert!(
        e.profiles.get(pid).unwrap().absorb_window().is_some(),
        "the SilentPin did NOT consume the window",
    );
    assert_eq!(
        e.profiles.get(pid).unwrap().absorb_count(),
        0,
        "a redundant Cold Seed records no fold",
    );

    // The surviving window now covers the first genuinely-fireable burst: a Standard drift burst
    // folds and consumes it.
    let drift = dir_tree_snap(vec![("echo.rs", EntryKind::File, 9)]);
    let fold_out = drive_standard_verdict(&mut e, pid, root, drift, now);
    assert!(
        fold_out.effects().is_empty(),
        "the preserved window folds the first fireable burst",
    );
    assert_eq!(e.profiles.get(pid).unwrap().absorb_count(), 1);
    assert!(
        e.profiles.get(pid).unwrap().absorb_window().is_none(),
        "the first fireable fold finally consumes the ConsumeOnFirst window",
    );
    assert!(
        !e.subs.get(sid).unwrap().has_fired(),
        "the fireable burst folded, not fired"
    );
}

/// Residual-restart latch: a window armed during post-fire applies to the burst the residual
/// restart produces. The restarted Standard burst (born via `restart_burst_from_fire_tail_residual`
/// → `into_pre_fire_residual`) freezes the live window at its birth and folds at its verdict.
#[test]
fn residual_restart_under_window_folds() {
    let (mut e, pid, sid, root, _now) = engine_with_attached_sub();
    let now = Instant::now();
    // Fire once (real Effect), then drive the post-fire loop into Rebasing with an absorbed
    // residual so the rebase response restarts a fresh Standard burst.
    let stable_out = drive_to_first_effect(&mut e, pid, root, now);
    assert_eq!(
        stable_out.effects().len(),
        1,
        "first burst fires a real Effect"
    );
    let key = stable_out.effects()[0].key();

    // The descendant the fire created — events on it route to the fire-tail residual.
    let child = e
        .tree
        .lookup(Some(root), "a.rs")
        .expect("the standard burst's reconcile created a.rs");

    // EffectComplete::Ok → Awaiting → Rebasing directly (probe-first; probe in flight, dirty
    // cleared at the loop entry).
    let rebasing_at = now + SETTLE * 2;
    let rearm_out = e.step(
        Input::EffectComplete(EffectCompletion {
            sub: sid,
            key,
            outcome: EffectOutcome::Ok,
        }),
        rebasing_at,
    );
    let rebase_corr = crate::testkit::first_probe_correlation(&rearm_out)
        .expect("Rebasing probe in flight after EffectComplete drove Awaiting → Rebasing");

    // Arm the window DURING post-fire (Rebasing) — it cannot retro-latch a PostFire burst (no-op),
    // but it stands for the next pre-fire birth, including the residual restart.
    e.step(
        Input::ArmAbsorb {
            profile: pid,
            duration: None,
        },
        rebasing_at,
    );

    // FsEvent during the Rebasing round-trip → absorbed into the final-window residual.
    let _ = e.step(
        Input::FsEvent {
            resource: child,
            event: FsEvent::ContentChanged,
        },
        rebasing_at + Duration::from_millis(1),
    );

    // Authoritative rebase response with a non-empty residual ⇒ restart.
    let restart_at = rebasing_at + Duration::from_millis(2);
    let _ = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid,
            correlation: rebase_corr,
            outcome: ProbeOutcome::SubtreeProven {
                snapshot: dir_tree_snap(vec![("a.rs", EntryKind::File, 1)]),
                authority: ProofAuthority::Authoritative,
            },
        }),
        restart_at,
    );

    // The restarted burst is a fresh Standard Batching burst, born under the live window ⇒
    // fold-latched. The latch read is sealed to core; assert it through the public
    // `burst_fold_latched()` accessor before destructuring the burst.
    assert!(
        e.profiles.get(pid).unwrap().state().burst_fold_latched(),
        "the restarted burst froze the live window at its birth",
    );
    let settle_timer = match e.profiles.get(pid).unwrap().state() {
        ProfileState::Active(ActiveBurst::PreFire(pre), _) => {
            assert_eq!(
                pre.intent,
                BurstIntent::Standard,
                "residual restart is Standard"
            );
            match &pre.phase {
                PreFirePhase::Batching { settle_timer } => *settle_timer,
                other => panic!("residual restart re-enters Batching; got {other:?}"),
            }
        }
        other => panic!("expected a restarted Active(PreFire) burst; got {other:?}"),
    };
    let absorb_count_before = e.profiles.get(pid).unwrap().absorb_count();
    let fire_count_before = e.subs.get(sid).unwrap().fire_history().unwrap().fire_count;

    // Drive the restarted burst to its verdict → folds.
    let _ = e.step(
        Input::TimerExpired {
            profile: pid,
            kind: TimerKind::Settle,
            id: settle_timer,
        },
        restart_at + SETTLE,
    );
    let restart_probe = e
        .pending_probe_for(pid)
        .expect("restarted burst's Verifying probe in flight");
    let fold_out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid,
            correlation: restart_probe,
            outcome: ProbeOutcome::SubtreeProven {
                snapshot: dir_tree_snap(vec![("a.rs", EntryKind::File, 1)]),
                authority: ProofAuthority::Authoritative,
            },
        }),
        restart_at + SETTLE + Duration::from_millis(1),
    );
    assert!(
        fold_out.effects().is_empty(),
        "the fold-latched residual restart emits no Effect; got {:?}",
        fold_out.effects(),
    );
    assert!(
        fold_out
            .diagnostics
            .iter()
            .any(|d| matches!(d, Diagnostic::QuiescenceAbsorbed { profile } if *profile == pid)),
        "the residual-restart fold emits QuiescenceAbsorbed",
    );
    assert_eq!(
        e.profiles.get(pid).unwrap().absorb_count(),
        absorb_count_before + 1,
        "the residual-restart fold bumps absorb_count",
    );
    assert_eq!(
        e.subs.get(sid).unwrap().fire_history().unwrap().fire_count,
        fire_count_before,
        "the fold did not fire — fire_count unchanged across the restart fold",
    );
    let _ = e.cancel_all_in_flight_probes();
}

/// `on_arm_absorb` window math, asserted via `Profile::absorb_window()` and the `AbsorbArmed`
/// diagnostic. `None` ⇒ `(now + settle, ConsumeOnFirst)`; `Some(d)` ⇒ `(now + d, PersistUntil)`.
#[test]
fn on_arm_absorb_derives_window_from_duration() {
    let (mut e, pid, _sid, _root, _now) = engine_with_attached_sub();
    complete_seed_burst(&mut e, pid);
    let now = Instant::now();

    // None ⇒ default one-shot, expiry = now + settle.
    let none_out = e.step(
        Input::ArmAbsorb {
            profile: pid,
            duration: None,
        },
        now,
    );
    assert_eq!(
        e.profiles.get(pid).unwrap().absorb_window(),
        Some(&specter_core::AbsorbWindow {
            expiry: now + SETTLE,
            mode: specter_core::AbsorbMode::ConsumeOnFirst,
        }),
        "None ⇒ (now + settle, ConsumeOnFirst)",
    );
    assert!(
        none_out.diagnostics.iter().any(|d| matches!(
            d,
            Diagnostic::AbsorbArmed { profile, mode: specter_core::AbsorbMode::ConsumeOnFirst }
                if *profile == pid,
        )),
        "None arm emits AbsorbArmed{{ConsumeOnFirst}}",
    );

    // Some(d) ⇒ time-boxed, expiry = now + d, PersistUntil (last-writer-wins).
    let d = Duration::from_secs(30);
    let some_out = e.step(
        Input::ArmAbsorb {
            profile: pid,
            duration: Some(d),
        },
        now,
    );
    assert_eq!(
        e.profiles.get(pid).unwrap().absorb_window(),
        Some(&specter_core::AbsorbWindow {
            expiry: now + d,
            mode: specter_core::AbsorbMode::PersistUntil,
        }),
        "Some(d) ⇒ (now + d, PersistUntil)",
    );
    assert!(
        some_out.diagnostics.iter().any(|d| matches!(
            d,
            Diagnostic::AbsorbArmed { profile, mode: specter_core::AbsorbMode::PersistUntil }
                if *profile == pid,
        )),
        "Some(d) arm emits AbsorbArmed{{PersistUntil}}",
    );
}

// ---------- Property tests ----------

mod props {
    use super::*;
    use proptest::prelude::*;

    /// Each property generates a sequence of opaque "actions" for the engine to dispatch, then
    /// asserts a global invariant. We don't generate random `ResourceIds` — the resource always
    /// exists from a fresh attach.
    #[derive(Clone, Debug)]
    enum Action {
        FsEvent(FsEvent),
        AdvanceTime(u32), // ms to advance
        Probe,            // accept whatever probe is in flight, return Ok
        ProbeVanished,
        ProbeFailed(i32),
        EffectComplete,
    }

    fn arb_fsevent() -> impl Strategy<Value = FsEvent> {
        prop_oneof![
            Just(FsEvent::ContentChanged),
            Just(FsEvent::StructureChanged),
            Just(FsEvent::Removed),
            Just(FsEvent::Renamed),
        ]
    }

    fn arb_action() -> impl Strategy<Value = Action> {
        prop_oneof![
            arb_fsevent().prop_map(Action::FsEvent),
            (1u32..200).prop_map(Action::AdvanceTime),
            Just(Action::Probe),
            Just(Action::ProbeVanished),
            (1i32..5).prop_map(Action::ProbeFailed),
            Just(Action::EffectComplete),
        ]
    }

    /// Apply `action` to a freshly-attached single-Profile engine; collect the `StepOutput` from
    /// each step. Returns the latest correlation seen so the next `Probe` action can target it.
    fn run_action(
        e: &mut Engine,
        sid: specter_core::SubId,
        r: ResourceId,
        action: Action,
        t: &mut Instant,
        last_correlation: &mut Option<specter_core::ProbeCorrelation>,
    ) -> StepOutput {
        let pid = e.subs.get(sid).unwrap().profile();
        let out = match action {
            Action::FsEvent(event) => e.step(Input::FsEvent { resource: r, event }, *t),
            Action::AdvanceTime(ms) => {
                *t += Duration::from_millis(u64::from(ms));
                let mut combined = StepOutput::default();
                while let Some(entry) = e.pop_expired(*t) {
                    let s = e.step(
                        Input::TimerExpired {
                            profile: entry.profile,
                            kind: entry.kind,
                            id: entry.id,
                        },
                        *t,
                    );
                    for c in s.probe_ops().iter().filter_map(|op| match op {
                        ProbeOp::Probe { request } => Some(request.correlation()),
                        _ => None,
                    }) {
                        *last_correlation = Some(c);
                    }
                    extend_step_output(&mut combined, s);
                }
                combined
            }
            Action::Probe => {
                let snap = dir_tree_snap(vec![]);
                let corr = last_correlation.unwrap_or(specter_core::ProbeCorrelation::from(0));
                e.step(
                    Input::ProbeResponse(ProbeResponse {
                        owner: pid,
                        correlation: corr,
                        outcome: ProbeOutcome::SubtreeProven {
                            snapshot: snap,
                            authority: ProofAuthority::Authoritative,
                        },
                    }),
                    *t,
                )
            }
            Action::ProbeVanished => {
                let corr = last_correlation.unwrap_or(specter_core::ProbeCorrelation::from(0));
                e.step(
                    Input::ProbeResponse(ProbeResponse {
                        owner: pid,
                        correlation: corr,
                        outcome: ProbeOutcome::Vanished,
                    }),
                    *t,
                )
            }
            Action::ProbeFailed(errno) => {
                let corr = last_correlation.unwrap_or(specter_core::ProbeCorrelation::from(0));
                e.step(
                    Input::ProbeResponse(ProbeResponse {
                        owner: pid,
                        correlation: corr,
                        outcome: ProbeOutcome::Failed(ProbeFailure::Anchor { errno }),
                    }),
                    *t,
                )
            }
            Action::EffectComplete => e.step(
                Input::EffectComplete(EffectCompletion {
                    sub: sid,
                    key: DedupKey::Subtree {
                        sub: sid,
                        profile: pid,
                    },
                    outcome: EffectOutcome::Ok,
                }),
                *t,
            ),
        };

        // Update last_correlation from any Probe in the output.
        for c in out.probe_ops().iter().filter_map(|op| match op {
            ProbeOp::Probe { request } => Some(request.correlation()),
            _ => None,
        }) {
            *last_correlation = Some(c);
        }

        out
    }

    fn extend_step_output(dst: &mut StepOutput, src: StepOutput) {
        let (watch_ops, probe_ops, effects, cancel_effects, diagnostics) = src.into_parts();
        dst.watch_ops.extend(watch_ops);
        for op in probe_ops.into_values() {
            dst.push_probe_op(op);
        }
        for ef in effects {
            dst.push_effect(ef);
        }
        for profile in cancel_effects {
            dst.push_cancel_effect(profile);
        }
        dst.diagnostics.extend(diagnostics);
    }

    fn fresh_engine_with_sub() -> (
        Engine,
        specter_core::SubId,
        ResourceId,
        Instant,
        Option<specter_core::ProbeCorrelation>,
    ) {
        let mut e = Engine::new();
        let r = e.tree.ensure_root("anchor", ResourceRole::User);
        e.tree.set_kind(r, ResourceKind::Dir);
        let now = Instant::now();
        let req = SubAttachRequest {
            anchor: SubAttachAnchor::Resource(r),
            identity: ProfileIdentity::new(
                ScanConfig::builder().recursive(true).build(),
                MAX_SETTLE,
                NO_EVENTS,
            ),
            params: SubParams::spawn(
                "test".into(),
                empty_program(),
                EffectScope::SubtreeRoot,
                SETTLE,
                false,
            ),
        };
        let out = e.step(Input::AttachSub(req), now);
        let sid = specter_core::testkit::first_attached_sub(&out).expect("attach_sub succeeded");
        let last_correlation = out.probe_ops().iter().find_map(|op| match op {
            ProbeOp::Probe { request } => Some(request.correlation()),
            _ => None,
        });
        (e, sid, r, now, last_correlation)
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 64,
            ..ProptestConfig::default()
        })]

        /// Every StepOutput is sorted canonically. Run a random sequence of inputs and verify after
        /// each step.
        #[test]
        fn prop_step_output_sorted_after_every_step(
            actions in prop::collection::vec(arb_action(), 0..16),
        ) {
            let (mut e, sid, r, mut t, mut last_correlation) =
                fresh_engine_with_sub();

            for action in actions {
                let out = run_action(&mut e, sid, r, action, &mut t, &mut last_correlation);

                // watch_ops sorted by ResourceId.
                let watch_keys: Vec<_> = out
                    .watch_ops
                    .iter()
                    .map(WatchOp::resource)
                    .collect();
                let mut sorted = watch_keys.clone();
                sorted.sort();
                prop_assert_eq!(watch_keys, sorted);

                // probe_ops sorted by ProfileId.
                let probe_keys: Vec<_> = out
                    .probe_ops()
                    .iter()
                    .map(ProbeOp::owner)
                    .collect();
                let mut sorted_p = probe_keys.clone();
                sorted_p.sort();
                prop_assert_eq!(probe_keys, sorted_p);
            }
            let _ = e.cancel_all_in_flight_probes();
        }

        /// I5: at most one outstanding ProbeRequest per Profile. Track outstanding probes via
        /// emit/cancel/respond; assert ≤ 1.
        #[test]
        fn prop_at_most_one_outstanding_probe(
            actions in prop::collection::vec(arb_action(), 0..16),
        ) {
            let (mut e, sid, r, mut t, mut last_correlation) =
                fresh_engine_with_sub();
            let pid = e.subs.get(sid).unwrap().profile();

            // attach_sub starts a Batching-first Seed burst — no probe until its settle expires, so
            // outstanding = 0.
            let mut outstanding: u32 = 0;

            for action in actions {
                let was_probe = matches!(action, Action::Probe | Action::ProbeVanished | Action::ProbeFailed(_));
                let out = run_action(&mut e, sid, r, action, &mut t, &mut last_correlation);

                // Each Probe op increments; each Cancel and each accepted ProbeResponse decrements.
                // (We treat "any Probe op emitted" as +1 and "any Cancel emitted" as -1; the test
                // doesn't care about the difference, only that the running count stays ≤ 1.)
                let probes_emitted = out
                    .probe_ops()
                    .iter()
                    .filter(|op| matches!(op, ProbeOp::Probe { .. }))
                    .count();
                let cancels_emitted = out
                    .probe_ops()
                    .iter()
                    .filter(|op| matches!(op, ProbeOp::Cancel { .. }))
                    .count();

                outstanding = outstanding
                    .saturating_add(u32::try_from(probes_emitted).unwrap_or(0))
                    .saturating_sub(u32::try_from(cancels_emitted).unwrap_or(0));

                // If a ProbeResponse action was injected and didn't cause a stale-diagnostic, the
                // outstanding probe is consumed.
                if was_probe {
                    let stale = out.diagnostics.iter().any(|d| {
                        matches!(d, Diagnostic::StaleProbeResponse { .. })
                    });
                    if !stale {
                        outstanding = outstanding.saturating_sub(1);
                    }
                }

                prop_assert!(
                    outstanding <= 1,
                    "I5: outstanding probes per profile = {} > 1",
                    outstanding,
                );

                // Slot-discipline I5: at most one outstanding probe per Profile, expressed as a
                // single state-resident probe slot reachable via `pid`. `pending_probe_for` returns
                // `Option<ProbeCorrelation>` so `<= 1` is trivially true; the assertion is a
                // regression guard against a future widening of the per-owner slot shape.
                let probing_count =
                    u32::from(e.pending_probe_for(pid).is_some());
                prop_assert!(
                    probing_count <= 1,
                    "I5 representability: a Profile's single state-resident ProbeSlot carries at most one in-flight probe",
                );
            }
            let _ = e.cancel_all_in_flight_probes();
        }

        /// `prop_seed_burst_without_activity_emits_no_effects`: from a fresh attach with **no
        /// FsEvents witnessed**, the Seed-burst's eventual ProbeResponse path never produces an
        /// Effect. This is strictly the no-activity path: with no events injected and `dirty`
        /// empty, the verdict routes to `SilentPin` and the burst finishes without emission. It
        /// does NOT assert anything about a fresh Seed that *witnessed* activity — that case fires
        /// and is covered by the `fresh_seed_fires::*` tests.
        #[test]
        fn prop_seed_burst_without_activity_emits_no_effects(
            seed_outcome in prop_oneof![
                Just(0),  // Ok
                Just(1),  // Vanished
                Just(2),  // Failed
            ],
        ) {
            let (mut e, sid, _r, now, _last_correlation) =
                fresh_engine_with_sub();
            let pid = e.subs.get(sid).unwrap().profile();
            // Batching-first Seed: expire the settle timer so the verify probe is in flight, then
            // answer it with the random outcome.
            assert_seed_verifying(&e);
            let corr = e
                .pending_probe_for(pid)
                .expect("seed verify probe in flight after settle expiry");
            let outcome = match seed_outcome {
                0 => ProbeOutcome::SubtreeProven { snapshot: dir_tree_snap(vec![]), authority: ProofAuthority::Authoritative },
                1 => ProbeOutcome::Vanished,
                _ => ProbeOutcome::Failed(ProbeFailure::Anchor { errno: 13 }),
            };
            let out = e.step(
                Input::ProbeResponse(ProbeResponse { owner: pid,
                    correlation: corr,
                    outcome,
                }),
                now + SETTLE,
            );
            prop_assert!(
                out.effects().is_empty(),
                "a fresh Seed that witnessed no activity emits no Effects"
            );
        }

        /// `prop_single_profile_never_has_active_standard_descendant` — the derived I4 floor. A
        /// single-Profile engine has no covered descendant, so the fresh reconfirm query that gates
        /// `gated_fire`'s fire (`coverage::has_active_standard_descendant`) must stay false after
        /// *any* input sequence. This also pins the query's self-exclusion: the lone Profile is the
        /// ancestor under test and must never count itself, through every burst phase the random
        /// actions drive it into.
        #[test]
        fn prop_single_profile_never_has_active_standard_descendant(
            actions in prop::collection::vec(arb_action(), 0..16),
        ) {
            let (mut e, sid, r, mut t, mut last_correlation) =
                fresh_engine_with_sub();
            let pid = e.subs.get(sid).unwrap().profile();
            for action in actions {
                let _ = run_action(&mut e, sid, r, action, &mut t, &mut last_correlation);
                if e.profiles.get(pid).is_some() {
                    prop_assert!(!crate::coverage::has_active_standard_descendant(
                        &e.tree,
                        &e.profiles,
                        pid,
                        &mut std::path::PathBuf::new(),
                    ));
                }
            }
            let _ = e.cancel_all_in_flight_probes();
        }

        /// `prop_step_is_total` — for any input sequence on a fresh engine, no panic in release.
        /// Implicit by reaching this assertion. Keep as a smoke test for the random-input fuzzer.
        #[test]
        fn prop_step_is_total(
            actions in prop::collection::vec(arb_action(), 0..32),
        ) {
            let (mut e, sid, r, mut t, mut last_correlation) =
                fresh_engine_with_sub();
            for action in actions {
                let _ = run_action(&mut e, sid, r, action, &mut t, &mut last_correlation);
            }
            prop_assert!(true);
            let _ = e.cancel_all_in_flight_probes();
        }
    }

    /// Reference-only: avoid an "unused field" warning for `BurstIntent`.
    const _: BurstIntent = BurstIntent::Standard;
}
