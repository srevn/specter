//! Per-input dispatch tests. Each `(state, input)` cell of the transition
//! table gets a focused test. Goes hand-in-hand with the integration suite
//! at `tests/integration.rs` which covers full-lifecycle flows.

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
    ChildEntry, ClaimKind, ClassSet, DedupKey, Diagnostic, DirChild, DirMeta, DirSnapshot,
    DirtyProvenance, EffectCompletion, EffectOutcome, EffectScope, EntryKind, FS_ROOT_SEGMENT,
    FsEvent, FsIdentity, Input, LeafEntry, OverflowScope, PatternSpec, Placeholder, PostFireBurst,
    PostFirePhase, PreFireBurst, PreFirePhase, ProbeOp, ProbeOutcome, ProbeOwner, ProbeRequest,
    ProbeResponse, ProbeSlot, ProfileIdentity, ProfileState, Promoter, PromoterAttachRequest,
    PromoterRegistryDiff, PromoterState, ProofAuthority, ProofObligation, QuiescenceVerdict,
    ResourceId, ResourceKind, ResourceRole, ScanConfig, StepOutput, SubAttachAnchor,
    SubAttachRequest, SubId, SubParams, Termination, TimerKind, TreeSnapshot, WatchOp,
    WatchRegistryDiff,
};
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const SETTLE: Duration = Duration::from_millis(100);
const MAX_SETTLE: Duration = Duration::from_secs(6);
/// Default events mask for transition tests. Empty mask gives a Profile
/// with `has_per_file_fds = false`, matching the prior tests' assumption
/// that per-file FDs are out of scope unless the test specifically opts
/// into PerStableFile + a CONTENT/METADATA mask. Events fold into
/// `config_hash` so two test fixtures differing only on `events` fork
/// separate Profiles (intentional partition).
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

/// Engine + Sub attached at `/anchor` (Dir, recursive). Returns the
/// engine, `ProfileId`, `SubId`.
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
        identity: ProfileIdentity {
            config: ScanConfig::builder().recursive(true).build(),
            max_settle: MAX_SETTLE,
            events: NO_EVENTS,
        },
        params: SubParams {
            name: "test-sub".into(),
            program: empty_program(),
            scope: EffectScope::SubtreeRoot,
            settle: SETTLE,
            log_output: false,
            source_promoter: None,
        },
    };
    let out = e.step(Input::AttachSub(req), now);
    let sid = specter_core::testkit::first_attached_sub(&out).expect("attach_sub succeeded");
    let pid = e.subs.get(sid).unwrap().profile();
    (e, pid, sid, r, now)
}

/// V5-native test helper: build a `TreeSnapshot::Dir` with the supplied
/// single-component children. Each child is `(name, EntryKind, inode)`;
/// Dirs are emitted as `DirChild::Uncovered(_)` (the walker stored the
/// entry but did not recurse). Tests that need nested subtrees should
/// use `dir_with_subtree`. Returns `Arc<DirSnapshot>` directly — the
/// typed `ProbeOutcome::SubtreeProven` / `DirEnumerated` variants carry
/// an `Arc<DirSnapshot>`, not a wrapping `TreeSnapshot`.
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

/// `LeafEntry` for File-anchored Profiles. Consumed directly by
/// `ProbeOutcome::AnchorOk`; the wrapping `TreeSnapshot::File` lives on
/// the engine-internal `Profile.current`, not the wire response.
fn file_tree_snap(kind: EntryKind, size: u64, mtime: SystemTime, inode: u64) -> LeafEntry {
    LeafEntry::synthetic(kind, size, mtime, FsIdentity::synthetic(inode, 0))
}

/// Drive a fresh-attach cold-arm Seed burst from
/// `Active(PreFire(Verifying))` through its quiescence verdict to
/// pinned `Idle`, pinning against `snap`. After this,
/// `Profile.current` and `Profile.baseline` are set to `snap`.
///
/// The cold-arm Seed burst pins on the first `Authoritative` response:
/// `quiescence_verdict(Authoritative, !forced)` folds to `Authoritative
/// { forced: false }`, dispatch reaches `SilentPin` (no fired Subs, no
/// drift) and finishes to Idle. The cold-arm Verifying-first contract
/// puts the probe in flight at burst construction, so the helper
/// answers it directly — no Batching settle expiry, no second sample.
///
/// Returns the pinning response's [`StepOutput`] so callers can assert
/// the Seed-completion emission (a fresh Seed never fires, so it is
/// effect-empty).
fn complete_seed_burst_with(
    e: &mut Engine,
    pid: specter_core::ProfileId,
    snap: Arc<DirSnapshot>,
) -> StepOutput {
    let at = Instant::now();
    let correlation = e
        .pending_probe_for(ProbeOwner::Profile(pid))
        .expect("cold-Seed Verifying probe in flight from start_seed_burst(None)");
    let last = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
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

/// [`complete_seed_burst_with`] against an empty baseline — the common
/// case for attach→Idle setups that don't pin specific children.
fn complete_seed_burst(e: &mut Engine, pid: specter_core::ProfileId) -> StepOutput {
    complete_seed_burst_with(e, pid, dir_tree_snap(vec![]))
}

/// Assert every Seed Profile is in `Active(PreFire(Verifying))` with a
/// probe in flight at burst construction — the cold-arm contract.
///
/// Under the cold-arm Verifying-first shape, `start_seed_burst(None)`
/// arms the verify slot at construction and emits the cold walk
/// immediately, so this helper does not advance state. It exists for
/// readability and to keep call-sites symmetric with the prior
/// Batching-first helper; `attach_now` is retained for signature
/// parity. For the full Seed pin use [`complete_seed_burst`].
fn seed_settle_to_verifying(e: &Engine, _attach_now: Instant) {
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
                matches!(pre.phase, PreFirePhase::Verifying(_)),
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
    // After attach: anchor watch_demand=1, Profile is
    // Active(Seed Verifying).
    assert_eq!(e.tree.get(r).unwrap().watch_demand(), 1);
    let _ = e.cancel_all_in_flight_probes();
}

/// `Profile.kind` is the cached witness of the anchor's classification:
/// `transition_to_verifying`'s probe-target dispatch and
/// `emit_effects`'s `compute_cwd` dispatch read this rather than
/// re-deriving the kind from the Tree on every call. A resource-based
/// attach against a kind-classified slot must populate the field at the
/// `attach_sub_inner` post-`Profile::new` write.
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

/// Resource-based attach against an `Unknown` slot leaves `Profile.kind
/// = None` until the first probe response classifies the anchor. The
/// `dispatch_quiescence_ok` fallback writes the field from the response shape
/// — the rare unprobed-attach path's only signal of the anchor's
/// classification.
#[test]
fn attach_sub_unprobed_anchor_seeds_kind_on_first_response() {
    let mut e = Engine::new();
    // Resource exists but kind is left Unknown — the rare path where a
    // caller passes a resource-based attach against a freshly-`ensure`'d
    // slot whose kind hasn't been classified by any prior probe.
    let r = e.tree.ensure_root("anchor", ResourceRole::User);
    let now = Instant::now();
    let req = SubAttachRequest {
        anchor: SubAttachAnchor::Resource(r),
        identity: ProfileIdentity {
            config: ScanConfig::builder().recursive(true).build(),
            max_settle: MAX_SETTLE,
            events: NO_EVENTS,
        },
        params: SubParams {
            name: "test-sub".into(),
            program: empty_program(),
            scope: EffectScope::SubtreeRoot,
            settle: SETTLE,
            log_output: false,
            source_promoter: None,
        },
    };
    let attach_out = e.step(Input::AttachSub(req), now);
    let sid = specter_core::testkit::first_attached_sub(&attach_out).expect("attach_sub succeeded");
    let pid = e.subs.get(sid).unwrap().profile();

    assert_eq!(
        e.profiles.get(pid).and_then(specter_core::Profile::kind),
        None,
        "unprobed anchor → Profile.kind starts as None",
    );

    // The Seed burst is Batching-first; expire the settle
    // window so it advances to Verifying and emits its first probe.
    seed_settle_to_verifying(&e, now);

    // Drive the first Seed verify with a Dir-shaped response. The
    // kind-classification fallback in `dispatch_burst_outcome` caches
    // the anchor kind from the response shape on the *first* response,
    // before the first-sample `Unstable` verdict re-batches.
    let correlation = e
        .pending_probe_for(ProbeOwner::Profile(pid))
        .expect("Seed verify probe in flight after settle expiry");
    let snap = dir_tree_snap(vec![]);
    let _ = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
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

/// `dispatch_burst_outcome` is the unified fan-out for both Seed and
/// Standard intents, so the kind-classification fallback fires from every
/// burst arm — not just Seed. Companion to
/// `attach_sub_unprobed_anchor_seeds_kind_on_first_response`: that test
/// pins the Seed-Ok / SubtreeProven path; this one pins it explicitly through
/// the same outcome shape and asserts the Profile reaches its first
/// classification before any subsequent dispatcher work runs.
#[test]
fn dispatch_burst_outcome_classifies_kind_on_first_seed_subtree() {
    let mut e = Engine::new();
    let r = e.tree.ensure_root("anchor", ResourceRole::User);
    // Leave the Resource Unknown — anchor_kind from `Resource::kind()`
    // collapses Unknown to None, so Profile.kind starts as None.
    let req = SubAttachRequest {
        anchor: SubAttachAnchor::Resource(r),
        identity: ProfileIdentity {
            config: ScanConfig::builder().recursive(true).build(),
            max_settle: MAX_SETTLE,
            events: NO_EVENTS,
        },
        params: SubParams {
            name: "test-sub".into(),
            program: empty_program(),
            scope: EffectScope::SubtreeRoot,
            settle: SETTLE,
            log_output: false,
            source_promoter: None,
        },
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

    seed_settle_to_verifying(&e, now);
    let correlation = e
        .pending_probe_for(ProbeOwner::Profile(pid))
        .expect("Seed verify probe in flight after settle expiry");
    let snap = dir_tree_snap(vec![]);
    let _ = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
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

/// Mirror of the SubtreeProven test for the AnchorOk arm: an `AnchorOk(leaf)`
/// response on a Profile whose `kind` was None classifies the anchor as
/// `File`. The walker's response variant is the canonical witness, so the
/// fallback cannot be specialised to one shape.
#[test]
fn dispatch_burst_outcome_classifies_kind_on_first_seed_anchor() {
    let mut e = Engine::new();
    let r = e.tree.ensure_root("anchor", ResourceRole::User);
    // Resource is Unknown ⇒ Profile.kind starts as None. The Seed burst
    // emits `ProbeRequest::Subtree` per the unified fallback (Subtree
    // is the safe default for unclassified anchors). The walker, finding a
    // regular file at the path, replies with `Vanished` in production
    // (kind mismatch). For this test we synthesise an `AnchorOk(leaf)`
    // response — a deliberate deviation that exercises the
    // dispatch_burst_outcome classification path for AnchorOk; the walker
    // never produces this response shape against a Subtree request, but
    // the engine's classification logic must still fall out correctly if
    // it ever does (defense-in-depth + symmetry with the SubtreeProven arm).
    let req = SubAttachRequest {
        anchor: SubAttachAnchor::Resource(r),
        identity: ProfileIdentity {
            config: ScanConfig::builder().recursive(true).build(),
            max_settle: MAX_SETTLE,
            events: NO_EVENTS,
        },
        params: SubParams {
            name: "test-sub".into(),
            program: empty_program(),
            scope: EffectScope::SubtreeRoot,
            settle: SETTLE,
            log_output: false,
            source_promoter: None,
        },
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

    seed_settle_to_verifying(&e, now);
    let correlation = e
        .pending_probe_for(ProbeOwner::Profile(pid))
        .expect("Seed verify probe in flight after settle expiry");
    let leaf = file_tree_snap(EntryKind::File, 0, UNIX_EPOCH, 1);
    let _ = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
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

/// Walker contract: a `Pending` Profile (descent state) probes a Dir
/// prefix with `ProbeRequest::Descent`; the only valid responses are
/// `DirEnumerated`, `Vanished`, or `Failed`. An `AnchorOk` in this slot is a
/// walker-side bug — descent never queries an anchor's `lstat` shape. The
/// `(DispatchTarget::Descent, ProbeOutcome::AnchorOk(_))` arm fires a
/// `debug_assert!` in dev/CI and falls through to `StaleProbeResponse` in
/// release. The test pins the dev/CI behaviour.
///
/// Disabled in release builds via the standard `cfg_attr` discipline,
/// mirroring `mint_probe_correlation_panics_on_double_open`.
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
        .pending_probe_for(ProbeOwner::Profile(pid))
        .expect("descent probe in flight at the prefix");

    // `AnchorOk` from a Descent probe is structurally impossible from the
    // production walker — `probe_descent` calls `probe_subtree`, whose
    // root-`lstat` rejects non-Dir paths via `Vanished`. We synthesise the
    // breach to exercise the walker-contract debug_assert in
    // `on_probe_response`.
    let leaf = file_tree_snap(EntryKind::File, 0, UNIX_EPOCH, 1);
    let _ = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
            correlation,
            outcome: ProbeOutcome::AnchorOk(leaf),
        }),
        now,
    );
}

/// `Engine::kind_agrees_or_finalize` boundary check: a `Profile.kind =
/// Some(File)` receiving a Dir-shaped response is a structurally
/// unreachable walker-contract violation (the typed `ProbeRequest`
/// chain emits `AnchorFile` for File-kinded Profiles, and the walker's
/// `ProbeOutcome` variant matches the request by construction). The
/// boundary catches the case at dispatch time and routes through
/// [`Engine::finalize_anchor_lost`] rather than misroute the Dir
/// snapshot onto a File-kinded Profile (which would leak watch
/// contributions and break the cross-field invariant).
#[test]
#[cfg_attr(
    not(debug_assertions),
    ignore = "debug_assert! is compiled out in release"
)]
#[should_panic(expected = "walker contract violated")]
fn kind_mismatched_response_routes_through_finalize_anchor_lost_debug() {
    // Set up a File-kinded Profile in Active(Verifying) and inject a
    // SubtreeProven (Dir) response.
    let mut e = Engine::new();
    let r = e.tree.ensure_root("anchor", ResourceRole::User);
    e.tree.set_kind(r, ResourceKind::File);
    let req = SubAttachRequest {
        anchor: SubAttachAnchor::Resource(r),
        identity: ProfileIdentity {
            config: ScanConfig::builder().recursive(true).build(),
            max_settle: MAX_SETTLE,
            events: NO_EVENTS,
        },
        params: SubParams {
            name: "test-sub".into(),
            program: empty_program(),
            scope: EffectScope::SubtreeRoot,
            settle: SETTLE,
            log_output: false,
            source_promoter: None,
        },
    };
    let now = Instant::now();
    let attach_out = e.step(Input::AttachSub(req), now);
    let sid = specter_core::testkit::first_attached_sub(&attach_out).expect("attach_sub succeeded");
    let pid = e.subs.get(sid).unwrap().profile();
    // Cold-arm Seed Verifying-first: a single Authoritative sample pins
    // → SilentPin → Idle.
    let mut at = now + SETTLE;
    let correlation = e
        .pending_probe_for(ProbeOwner::Profile(pid))
        .expect("cold-arm Seed Verify probe in flight at burst construction");
    e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
            correlation,
            outcome: ProbeOutcome::AnchorOk(file_tree_snap(EntryKind::File, 0, UNIX_EPOCH, 1)),
        }),
        at,
    );
    assert_eq!(
        e.profiles.get(pid).and_then(specter_core::Profile::kind),
        Some(ResourceKind::File),
    );

    // Drive a Standard burst (FsEvent at the anchor) and let the settle
    // timer fire so a Verifying probe is in flight.
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
        .pending_probe_for(ProbeOwner::Profile(pid))
        .expect("Standard Verifying probe in flight");

    // Inject the kind-mismatched response: a SubtreeProven (Dir) for a
    // File-kinded Profile. The boundary check fires the debug_assert.
    let dir = dir_tree_snap(vec![]);
    let _ = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
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

    // Second attach with the same config_hash.
    let req = SubAttachRequest {
        anchor: SubAttachAnchor::Resource(r),
        identity: ProfileIdentity {
            config: ScanConfig::builder().recursive(true).build(),
            max_settle: MAX_SETTLE,
            events: NO_EVENTS,
        },
        params: SubParams {
            name: "second".into(),
            program: empty_program(),
            scope: EffectScope::SubtreeRoot,
            settle: SETTLE,
            log_output: false,
            source_promoter: None,
        },
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

/// Smoke test: a `TreeSnapshot::Dir(...)` with one Leaf entry
/// lands as a Seed-Ok on the Profile (no Effect, baseline set). Pins
/// the dispatch wiring; the rest of the engine test suite is the broad
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
    // Batching-first: drive to the first Seed Verify probe, then
    // answer it Vanished — a terminal outcome regardless of verdict.
    seed_settle_to_verifying(&e, now);
    let correlation = e
        .pending_probe_for(ProbeOwner::Profile(pid))
        .expect("Seed Verifying probe in flight after settle expiry");
    let out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
            correlation,
            outcome: ProbeOutcome::Vanished,
        }),
        now + SETTLE,
    );
    let p = e.profiles.get(pid).unwrap();
    assert!(matches!(p.state(), ProfileState::Idle));
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
    // Batching-first: drive to the first Seed Verify probe, then
    // answer it Failed — a terminal outcome regardless of verdict.
    seed_settle_to_verifying(&e, now);
    let correlation = e
        .pending_probe_for(ProbeOwner::Profile(pid))
        .expect("Seed Verifying probe in flight after settle expiry");
    let out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
            correlation,
            outcome: ProbeOutcome::Failed { errno: 13 },
        }),
        now + SETTLE,
    );
    let has_diag = out.diagnostics.iter().any(|d| {
        matches!(
            d,
            Diagnostic::ProbeFailed {
                intent: BurstIntent::Seed,
                errno: 13,
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
            owner: ProbeOwner::Profile(pid),
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
            owner: ProbeOwner::Profile(pid),
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

// I5-breach panic/diagnostic tests deleted: the forged "probe armed in
// non-mint phase" shape they exercised is structurally unrepresentable
// once probe correlation lives on a state-resident `ProbeSlot` — a
// slot can only be armed via its owning phase's typed transition, so
// the (state, phase)-mismatch arm cannot be reached without forging an
// invalid state. Structural property tests for `ProbeSlot` live in
// `specter-core`'s `probe.rs` `#[cfg(test)] mod tests`.

// ---- Standard burst dispatch ----

#[test]
fn standard_burst_stable_emits_effect_and_awaits() {
    // Stable verdict emits the Effect and transitions to
    // `PostFirePhase::Awaiting`; the engine waits for the completion before
    // returning to Idle. Idle means "nothing in flight" — outstanding
    // Effects keep the burst Active until they report back.
    let (mut e, pid, _sid, root, _now) = engine_with_attached_sub();
    complete_seed_burst(&mut e, pid);
    let now = Instant::now();
    // FsEvent at anchor → Standard Settling.
    e.step(
        Input::FsEvent {
            resource: root,
            event: FsEvent::Modified,
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
    // The verify response folds through `quiescence_verdict` to
    // `Authoritative { forced: false }` on the first sample — single
    // dispatch fires the Effect.
    let correlation = e
        .pending_probe_for(ProbeOwner::Profile(pid))
        .expect("Verifying probe in flight");
    let out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
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
    // Engine carries the lowered ActionProgram; the actuator resolves
    // argv at spawn time. Assert on the template's literal-only first arg
    // instead of the resolved argv. (`/bin/true` is the test's stub
    // command — see `empty_program()`.)
    let SpawnBody::Exec(exec) = &eff.program.ops()[0].body() else {
        panic!("expected SpawnBody::Exec");
    };
    assert_eq!(exec.argv().len(), 1);
    assert!(matches!(
        exec.argv()[0].parts(),
        [specter_core::ArgPart::Literal(s)] if s.as_str() == "/bin/true"
    ));
    // Substitution-domain inputs that the actuator-side resolver renders
    // to SPECTER_PATH / SPECTER_WATCH / SPECTER_FORCED / SPECTER_EVENT_KIND.
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
    // SPECTER_DIFF_PATH is an actuator-side augmentation; engine's Effect
    // doesn't carry it. The structural witness is `eff.diff()`:
    assert!(
        eff.diff().is_none(),
        "engine doesn't include diff for non-needs_diff Sub"
    );
    // cwd derives from (anchor_path, anchor_kind) at spawn time. Pin both:
    assert_eq!(eff.anchor_path.as_os_str(), "anchor");
    assert_eq!(eff.anchor_kind, specter_core::ResourceKind::Dir);
}

/// The Subtree suppress decision is `nothing_changed &&
/// fired_subs.contains(&dk)` — two gates in conjunction. The
/// per-Profile half (`baseline.hash() == current.hash()`) covers
/// the "fired then noop" arm; the per-Sub half (`fired_subs`
/// existence) is the "Sub has fired before" discriminator that
/// distinguishes a fresh Sub (must fire even on an unchanged tree —
/// first emission) from a repeat fire (suppress on an unchanged
/// tree).
///
/// A noise FsEvent on a fresh Profile drives a phantom Standard
/// burst whose stable verdict observes `baseline.hash() ==
/// current.hash()`. Without the `fired_subs.contains` gate, the
/// phantom would suppress the very first Effect — the Sub would
/// never fire and the user's command would never run. This test
/// pins the gate so any future flattening of the suppress
/// derivation (e.g., dropping back to the per-Profile signal
/// alone) fails here, not in production.
#[test]
fn b1_dedup_fresh_sub_fires_on_phantom_standard_burst() {
    let (mut e, pid, _sid, root, _now) = engine_with_attached_sub();
    complete_seed_burst(&mut e, pid);

    // Precondition: post-Seed Profile has baseline = current and no
    // fire history. The phantom condition (`baseline.hash() ==
    // current.hash()`) is structurally satisfied; the fresh-Sub
    // condition (`fired_subs.is_empty()`) is by construction.
    let baseline_hash = match e.profiles.get(pid).unwrap().baseline() {
        Some(TreeSnapshot::Dir(arc)) => arc.dir_hash(),
        _ => panic!("post-Seed baseline must be Some(Dir)"),
    };
    assert!(!e.subs.any_fired(pid), "fresh Sub: no fire history");

    // Drive a Standard burst whose probe response equals the Seed
    // baseline byte-for-byte — a phantom (noise FsEvent, no actual
    // disk change). One probe suffices because the stability verdict
    // compares the response against `current.subtree_at(target)`,
    // which equals baseline immediately post-Seed.
    let now = Instant::now();
    e.step(
        Input::FsEvent {
            resource: root,
            event: FsEvent::Modified,
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
    // The verify response folds through `quiescence_verdict` to
    // `Authoritative { forced: false }` on the first sample — single
    // dispatch, no prime-then-confirm.
    let corr = e
        .pending_probe_for(ProbeOwner::Profile(pid))
        .expect("Verifying probe in flight");
    let out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
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
    // Contract: SubtreeRoot Sub anchored at a File-kind Profile derives
    // the Effect's `cwd` from the file's parent dir (not the file
    // itself — `Command::current_dir` requires a directory). The
    // surrounding burst flow (probe target, current-shape preservation,
    // graft path) is exercised by
    // `standard_burst_on_file_anchor_targets_anchor_not_parent_dir`;
    // this test asserts only the cwd / env-var contract.
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
        identity: ProfileIdentity {
            config: ScanConfig::builder().recursive(false).build(),
            max_settle: MAX_SETTLE,
            events: NO_EVENTS,
        },
        params: SubParams {
            name: "build".into(),
            program: empty_program(),
            scope: EffectScope::SubtreeRoot,
            settle: SETTLE,
            log_output: false,
            source_promoter: None,
        },
    };
    let attach_out = e.step(Input::AttachSub(req), now);
    let sid = specter_core::testkit::first_attached_sub(&attach_out).expect("attach_sub succeeded");
    let pid = e.subs.get(sid).unwrap().profile();
    // Seed → Idle: cold-arm Verifying-first, a single Authoritative
    // response pins → SilentPin → Idle. No Batching settle expiry
    // needed (the cold walk is in flight at burst construction).
    let snap = file_tree_snap(EntryKind::File, 0, std::time::UNIX_EPOCH, 1);
    let seed_corr = e
        .pending_probe_for(ProbeOwner::Profile(pid))
        .expect("cold-Seed Verify probe in flight at burst construction");
    e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
            correlation: seed_corr,
            outcome: ProbeOutcome::AnchorOk(snap.clone()),
        }),
        now + SETTLE,
    );
    // Standard burst with the same snap (stable). Start it after the
    // Seed pin so the timeline stays monotonic.
    let t1 = now + SETTLE * 3;
    e.step(
        Input::FsEvent {
            resource: file_anchor,
            event: FsEvent::Modified,
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
    // The verify response folds to `Authoritative { forced: false }`
    // on the first sample — single dispatch fires the Effect.
    let std_corr = e
        .pending_probe_for(ProbeOwner::Profile(pid))
        .expect("Verifying probe in flight");
    let out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
            correlation: std_corr,
            outcome: ProbeOutcome::AnchorOk(snap),
        }),
        t2,
    );
    assert_eq!(out.effects().len(), 1);
    let eff = &out.effects()[0];
    // File-kind anchor: actuator's `compute_cwd` returns parent dir.
    // The engine's job here is to pin (anchor_path, anchor_kind) so the
    // actuator's compute_cwd reaches "parentdir". The original cwd
    // assertion ("File-kind anchor uses parent dir as cwd") is now
    // structural: anchor_path is the file, anchor_kind is File ⇒
    // compute_cwd returns parent.
    assert_eq!(eff.anchor_path.as_os_str(), "parentdir/main.rs");
    assert_eq!(eff.anchor_kind, specter_core::ResourceKind::File);
    // SPECTER_PATH and SPECTER_ANCHOR both derive from `anchor_path`
    // for a File-anchor Subtree Effect: the resolver returns
    // `Cow::Borrowed(&anchor_path)` when `relative()` is empty,
    // so both env values share the same byte sequence.
    assert_eq!(eff.relative(), "");
}

#[test]
fn standard_burst_on_file_anchor_targets_anchor_not_parent_dir() {
    // Realistic Standard-burst-on-File-anchor flow. A real Sensor
    // probing a File anchor returns `TreeSnapshot::File(leaf)`; the
    // engine must (1) probe the anchor itself rather than the parent
    // dir and (2) preserve `Profile.current` as `TreeSnapshot::File(_)`
    // post-dispatch — the snapshot navigation invariant
    // `current` is anchor-shaped breaks if a Standard burst graft
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
        identity: ProfileIdentity {
            config: ScanConfig::builder().recursive(false).build(),
            max_settle: MAX_SETTLE,
            events: NO_EVENTS,
        },
        params: SubParams {
            name: "build".into(),
            program: empty_program(),
            scope: EffectScope::SubtreeRoot,
            settle: SETTLE,
            log_output: false,
            source_promoter: None,
        },
    };
    let attach_out = e.step(Input::AttachSub(req), now);
    let sid = specter_core::testkit::first_attached_sub(&attach_out).expect("attach_sub succeeded");
    let pid = e.subs.get(sid).unwrap().profile();

    // Cold-arm Seed Verifying-first: a single Authoritative sample pins
    // → SilentPin → Idle.
    let snap = file_tree_snap(EntryKind::File, 0, UNIX_EPOCH, 1);
    let seed_corr = e
        .pending_probe_for(ProbeOwner::Profile(pid))
        .expect("cold-arm Seed Verify probe in flight at burst construction");
    e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
            correlation: seed_corr,
            outcome: ProbeOutcome::AnchorOk(snap.clone()),
        }),
        now + SETTLE,
    );

    // Drive a Standard burst from an FsEvent at the file. Capture the
    // probe request emitted on the settle-timer expiry step. Start it
    // after the Seed burst's two settle windows (monotonic timeline).
    let t1 = now + SETTLE * 3;
    e.step(
        Input::FsEvent {
            resource: file_anchor,
            event: FsEvent::Modified,
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

    // (1) The Standard probe is `AnchorFile` and its `target_path` is
    // the anchor's filesystem path. The two assertions are the structural
    // witnesses for the v1 design: File anchors take the typed
    // `AnchorFile` arm (single-`lstat` walker dispatch) and never promote
    // past the anchor to the parent dir.
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

    // (2) Inject a realistic File response (kqueue per-file FD path).
    // The verify response folds through `quiescence_verdict` to
    // `Authoritative { forced: false }` on the first sample — single
    // dispatch, grafts the leaf into `current` and fires the Effect.
    let std_corr = e
        .pending_probe_for(ProbeOwner::Profile(pid))
        .expect("Standard verify probe in flight");
    let out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
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
            event: FsEvent::Modified,
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
    // After force-fire, we're either in Verifying (forced=true) or already
    // Awaiting if the deadline race resolved both timers. Drive the
    // response back if needed.
    if let Some(correlation) = e.pending_probe_for(ProbeOwner::Profile(pid)) {
        // Inject a not-stable response to test the forced effect emission.
        let snap = dir_tree_snap(vec![("new.rs", EntryKind::File, 99)]);
        let out = e.step(
            Input::ProbeResponse(ProbeResponse {
                owner: ProbeOwner::Profile(pid),
                correlation,
                outcome: ProbeOutcome::SubtreeProven {
                    snapshot: snap,
                    authority: ProofAuthority::Authoritative,
                },
            }),
            deadline,
        );
        // Forced fire transitions to Awaiting (Effect in flight). The
        // post-fire rebase happens when the eventual EffectComplete
        // drives the Awaiting → Rebasing transition.
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
fn fs_event_modified_during_seed_probing_preserves_intent() {
    let (mut e, pid, _sid, root, now) = engine_with_attached_sub();
    // The Seed burst is Batching-first; expire the settle
    // window so it reaches Verifying with a probe in flight, then
    // inject an FsEvent — it should cancel that probe and return to
    // Active(Seed Batching) with the Seed intent preserved.
    seed_settle_to_verifying(&e, now);
    let out = e.step(
        Input::FsEvent {
            resource: root,
            event: FsEvent::Modified,
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

/// Field-discipline pin for `event_drives_batching`: an FsEvent during
/// Verifying disarms the verify slot atomically with the Cancel
/// emission, so the `Verifying → Batching` rewrite cannot leave an
/// armed slot behind for the just-cancelled probe.
#[test]
fn event_drives_batching_clears_pending_probe() {
    let (mut e, pid, _sid, root, now) = engine_with_attached_sub();
    // The Seed burst is Batching-first; drive it to Verifying
    // so a probe is actually in flight to be cleared.
    seed_settle_to_verifying(&e, now);
    assert!(
        e.pending_probe_for(ProbeOwner::Profile(pid)).is_some(),
        "Seed probe in flight after settle expiry",
    );

    let _ = e.step(
        Input::FsEvent {
            resource: root,
            event: FsEvent::Modified,
        },
        Instant::now(),
    );

    assert!(
        e.pending_probe_for(ProbeOwner::Profile(pid)).is_none(),
        "slot disarmed atomically with Verifying → Batching transition",
    );
}

/// Field-discipline pin for `finalize_anchor_lost`: an anchor terminal
/// event during Verifying cancels the in-flight probe and clears the
/// channel. Replaces the pre-refactor `was_verifying` snapshot's role.
#[test]
fn finalize_anchor_lost_during_verifying_clears_pending_probe() {
    let (mut e, pid, _sid, root, now) = engine_with_attached_sub();
    // The Seed burst is Batching-first; drive it to Verifying
    // so an anchor-terminal event has an in-flight probe to cancel.
    seed_settle_to_verifying(&e, now);
    assert!(
        e.pending_probe_for(ProbeOwner::Profile(pid)).is_some(),
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
        e.pending_probe_for(ProbeOwner::Profile(pid)).is_none(),
        "anchor terminal during Verifying disarms the slot",
    );
    let cancels = out
        .probe_ops()
        .iter()
        .filter(|op| matches!(op, ProbeOp::Cancel { owner: ProbeOwner::Profile(profile)} if *profile == pid))
        .count();
    assert_eq!(
        cancels,
        1,
        "exactly one Cancel emitted; got {:?}",
        out.probe_ops()
    );
}

/// Single-diagnostic guarantee for stale `ProbeResponse`. Pre-refactor
/// the dispatch had two stale-detection layers (state-shape mismatch and
/// inner-correlation mismatch) that could both fire on degenerate inputs.
/// Post-refactor the top-level `pending_probe == Some(received)` check is
/// the sole gate — exactly one diagnostic per stale response.
#[test]
fn stale_probe_response_emits_exactly_one_diagnostic() {
    let (mut e, pid, _sid, _root, now) = engine_with_attached_sub();
    // The Seed burst is Batching-first; drive it to Verifying
    // so a legitimate Seed probe is live when the stale response lands.
    seed_settle_to_verifying(&e, now);
    let bogus = specter_core::ProbeCorrelation::from(99_999);
    let snap = dir_tree_snap(vec![]);

    let out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
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
        .filter(|d| matches!(d, Diagnostic::StaleProbeResponse { owner: ProbeOwner::Profile(profile), .. } if *profile == pid))
        .count();
    assert_eq!(
        stale_count, 1,
        "exactly one StaleProbeResponse diagnostic; got {:?}",
        out.diagnostics,
    );
    // Live channel untouched: the legitimate Seed probe is still in flight.
    assert!(
        e.pending_probe_for(ProbeOwner::Profile(pid)).is_some(),
        "live channel untouched by stale response",
    );
    let _ = e.cancel_all_in_flight_probes();
}

/// Anchor events bypass the class filter unconditionally.
/// Profile has events = EMPTY (nothing in the mask); a `MetadataChanged`
/// at the anchor still drives the lifecycle path (burst start), and no
/// `EventClassDropped` is emitted. This guards the lifecycle-continuity
/// invariant: anchor events never get filtered out by user mask choice.
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

/// Descendant events whose class is not in the covering Profile's
/// `events_union` drop with `EventClassDropped` BEFORE driving the burst.
/// Profile has events = EMPTY ⇒ `intersects(any_class) == false`, so a
/// `MetadataChanged` on a covered descendant drops cleanly without state
/// mutation.
#[test]
fn fs_event_metadatachanged_at_descendant_drops_with_event_class_dropped() {
    let (mut e, pid, _sid, root, _now) = engine_with_attached_sub();
    complete_seed_burst(&mut e, pid);

    // Materialize a covered descendant. Bump `watch_demand` so the event
    // passes the `EventOnUnwatchedResource` head guard. The Profile's
    // ScanConfig has `recursive(true)` so `covers(profile, child, tree)`
    // is satisfied.
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
    // No `MetadataChangedIgnored` lingers — the variant was deleted.
    // No state mutation: the filter `continue`s before drive_burst.
    assert!(matches!(
        e.profiles.get(pid).unwrap().state(),
        ProfileState::Idle,
    ));
}

/// Identity events on a *descendant File* fold into the CONTENT
/// class. A Profile excluding CONTENT (here: STRUCTURE-only on a Dir
/// anchor) drops the descendant File `Removed` with `EventClassDropped`.
/// The dropped event is not routed through `on_anchor_terminal_event`
/// — that routing is anchor-only.
#[test]
fn fs_event_terminal_on_descendant_file_folds_to_content_and_drops() {
    let mut e = Engine::new();
    let r = e.tree.ensure_root("anchor", ResourceRole::User);
    e.tree.set_kind(r, ResourceKind::Dir);
    let now = Instant::now();
    let req = SubAttachRequest {
        anchor: SubAttachAnchor::Resource(r),
        identity: ProfileIdentity {
            config: ScanConfig::builder().recursive(true).build(),
            max_settle: MAX_SETTLE,
            events: ClassSet::STRUCTURE,
        },
        params: SubParams {
            name: "test-sub".into(),
            program: empty_program(),
            scope: EffectScope::SubtreeRoot,
            settle: SETTLE,
            log_output: false,
            source_promoter: None,
        },
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

/// Terminal events on the anchor route through
/// `on_anchor_terminal_event` regardless of the Profile's `events_union`.
/// Anchor is a Dir, events = EMPTY: the kqexec class for `Removed` on a
/// Dir is STRUCTURE — not in the EMPTY mask — but anchor events bypass
/// the filter. After the call, `anchor_claim` is cleared to `None` and
/// `baseline` / `current` are dropped.
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
    assert!(matches!(p.state(), ProfileState::Idle));
}

#[test]
fn fs_event_for_unwatched_resource_emits_diagnostic() {
    let mut e = Engine::new();
    let r = e.tree.ensure_root("ghost", ResourceRole::User);
    let out = e.step(
        Input::FsEvent {
            resource: r,
            event: FsEvent::Modified,
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
    // A WatchRootParent fires
    // `StructureChanged` (e.g., a sibling directory was created /
    // renamed) and no Profile in the engine cares. The event must NOT
    // be diagnosed as "unwatched resource" — the Resource IS Watched.
    // The new `EventNoConsumer` variant signals this benign case so the
    // bin can log it at TRACE rather than WARN.
    let mut e = Engine::new();
    // Materialize an unrelated Watched resource (e.g., a parent that
    // someone else holds open). watch_demand > 0 ensures the event isn't
    // routed through the `EventOnUnwatchedResource` path.
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
            event: FsEvent::Modified,
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
    assert!(matches!(p.state(), ProfileState::Idle));
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
    // FsEvent: Removed/Renamed/Revoked on an Idle profile transitions
    // idempotently. We additionally release the watch contribution and
    // drop baseline/current — they refer to a now-vanished slot, and
    // clearing them lets the watch-root-parent recovery path
    // (`on_fs_event`'s `start_pending_recovery`) detect "anchor is gone"
    // via `current.is_none()`.
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

    // Profile state stays Idle (no Active transition).
    assert!(matches!(
        e.profiles.get(pid).unwrap().state(),
        ProfileState::Idle,
    ));
    // watch_demand released; baseline/current cleared.
    assert_eq!(e.tree.get(root).unwrap().watch_demand(), 0);
    let p = e.profiles.get(pid).unwrap();
    assert!(p.baseline().is_none());
    assert!(p.current().is_none());
}

#[test]
fn count_gate_zero_iff_no_carrier_and_anchor_loss_while_idle_balances_nonsteady() {
    // Oracle (b): the O(1) carrier gate is sound. `nonsteady() == 0`
    // ⇒ `classify_event_carriers` empty ∀ r — a healthy *anchored*
    // Idle Profile is excluded by the tight predicate, so a quiet
    // watcher never pins the gate. And the count stays balanced
    // across the anchor-loss-while-Idle reconcile
    // (`discard_anchor_state` → `ProfileMap::reconcile_nonsteady`),
    // the loss direction of the plan's subtlest count point. The
    // debug count-vs-full-scan tripwire inside
    // `classify_event_carriers` runs on every covering scan here
    // (and across the whole suite), so a desync panics in debug
    // regardless of the explicit asserts below — this test pins the
    // *implication* and the loss/recovery balance the tripwire alone
    // does not state.
    let (mut e, pid, _sid, root, _now) = engine_with_attached_sub();
    complete_seed_burst(&mut e, pid);

    // A carrier is empty iff all three carrier classes are empty.
    let empty = |e: &Engine, r: ResourceId| {
        let c = e.classify_event_carriers(r);
        c.descents.is_empty() && c.recoveries.is_empty() && c.promoter_recoveries.is_empty()
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

    // Anchor lost while Idle: `Removed @ anchor` ⇒ finalize_anchor_lost
    // with `was_active == false` ⇒ `discard_anchor_state` clears the
    // anchor with no state edge. The reconcile must move the count
    // 0 → 1 (pre-fix this desynced — the Profile is now an
    // Idle-anchorless recovery carrier the gate would have hidden).
    e.step(
        Input::FsEvent {
            resource: root,
            event: FsEvent::Removed,
        },
        Instant::now(),
    );

    {
        let p = e.profiles().get(pid).unwrap();
        assert!(matches!(p.state(), ProfileState::Idle) && p.current().is_none());
    }
    assert_eq!(
        e.profiles().nonsteady(),
        1,
        "anchor-loss-while-Idle reconciled the carrier count via reconcile_nonsteady \
         (pre-fix this desynced to 0 — a false-skip of the recovery scan)",
    );
    // Sound over-approximation: the predicate is `Idle ∧ ¬current`
    // (state + anchor), *not* the precise recovery predicate (which
    // also needs `watch_root_parent`). This root-anchor harness has
    // no parent recovery channel, so the scan finds no carrier here
    // even though the count is 1 — the gate over-counts harmlessly
    // (the scan runs and returns empty) but, critically, *never
    // under-counts* (it would never have wrongly skipped the scan).
    assert!(
        empty(&e, root),
        "over-approx: count gated the scan ON; no precise carrier in this harness",
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
            event: FsEvent::Modified,
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

/// An external `FsEvent` on the burst's own anchor, arriving mid-Batching,
/// is processed (not silently dropped): it lands in the pre-fire burst's
/// `dirty` and advances
/// `last_event_time`, so the next settle expiry **reschedules** the
/// settle timer (debounce) instead of verifying. The two anchor events
/// plus the reschedule collapse into a single fire — no double-fire.
/// Deterministic via explicit clock control; no soak.
#[test]
fn pre_fire_anchor_event_rearms_settle_and_fires_once() {
    let (mut e, pid, _sid, root, _now) = engine_with_attached_sub();
    complete_seed_burst(&mut e, pid);
    let t0 = Instant::now();

    // First anchor event opens the Standard burst (Idle → Batching).
    let out_a = e.step(
        Input::FsEvent {
            resource: root,
            event: FsEvent::Modified,
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

    // Second anchor event, still inside the settle window. This is the
    // event the watcher used to silence; assert it is recorded — it
    // enters the burst accumulator and advances `last_event_time`.
    let t1 = t0 + SETTLE / 2;
    let out_b = e.step(
        Input::FsEvent {
            resource: root,
            event: FsEvent::Modified,
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

    // The original settle timer expires at its deadline (t0 + SETTLE),
    // but the last event was at t1 (< SETTLE ago) → debounce: stay
    // Batching, reschedule a fresh settle at last_event_time + SETTLE.
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
        e.pending_probe_for(ProbeOwner::Profile(pid)).is_none(),
        "debounce did not verify — no probe in flight",
    );
    assert!(
        out_c.probe_ops().is_empty() && out_c.effects().is_empty(),
        "debounce emitted neither probe nor effect",
    );

    // Quiet for ≥ SETTLE past the last event: the rescheduled timer
    // expires and transitions to Verifying with exactly one probe.
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

    // The verify response folds to `Authoritative { forced: false }`
    // on the first sample — single dispatch fires the Effect. The Sub
    // never fired (Seed does not fire), so B1 does not suppress.
    let correlation = e
        .pending_probe_for(ProbeOwner::Profile(pid))
        .expect("Verifying probe in flight");
    let out_e = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
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
// The engine does not return to Idle after firing Effects: the burst
// stays `Active(Awaiting)` until each completion reports back, and the
// post-Effect rebase happens in `PostFirePhase::Rebasing` as a phase of
// the same burst. `EffectComplete` arrivals route by phase: Awaiting
// decrements / transitions; non-Awaiting emits
// `EffectCompleteOutsideAwaiting`.

#[test]
fn effect_complete_ok_in_idle_diagnoses_outside_awaiting() {
    // No path leaves Idle with an outstanding EffectComplete: the burst
    // stays Active(Awaiting) until completions arrive. A completion
    // landing in Idle is therefore unexpected (gate-deadline force-
    // transition or anchor-loss) — emit `EffectCompleteOutsideAwaiting`
    // and drop without state change.
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
    // Failed always clears `fired_subs[key]` regardless of
    // phase — a failed Effect leaves no observable state to dedupe
    // against. In Idle the completion is also "late" (the engine isn't
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
        identity: ProfileIdentity {
            config: ScanConfig::builder().recursive(true).build(),
            max_settle: MAX_SETTLE,
            events: NO_EVENTS,
        },
        params: SubParams {
            name: "fmt".into(),
            program: diff_program(), // references ${specter.created}
            scope: EffectScope::SubtreeRoot,
            settle: SETTLE,
            log_output: false,
            source_promoter: None,
        },
    };
    let out = e.step(Input::AttachSub(req), now);
    let sid = specter_core::testkit::first_attached_sub(&out).expect("attach_sub succeeded");
    let pid = e.subs.get(sid).unwrap().profile();
    assert!(e.subs.get(sid).unwrap().needs_diff);

    // Seed burst → baseline = empty snapshot.
    complete_seed_burst(&mut e, pid);

    // Standard burst, first round: FsEvent → settle → probe → snapshot with
    // a new entry. The first response is *not stable* (current was empty),
    // so the Engine reschedules another settle cycle.
    e.step(
        Input::FsEvent {
            resource: r,
            event: FsEvent::Modified,
        },
        now,
    );
    let snap_with_entry = dir_tree_snap(vec![("new.rs", EntryKind::File, 5)]);

    // Iteratively drain settle timers and inject probe responses until the
    // burst stabilizes (one Effect emitted).
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
        let correlation = e
            .pending_probe_for(ProbeOwner::Profile(pid))
            .expect("Verifying probe in flight");
        let out = e.step(
            Input::ProbeResponse(ProbeResponse {
                owner: ProbeOwner::Profile(pid),
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
    let (mut e, pid, _sid, _root, now) = engine_with_attached_sub();
    // Cold-arm Seed: the first Seed probe is in flight directly after
    // `attach_sub` — no settle expiry needed to reach Verifying.
    seed_settle_to_verifying(&e, now);
    let correlation = e
        .pending_probe_for(ProbeOwner::Profile(pid))
        .expect("cold-arm Seed Verifying probe in flight at attach");
    // First-probe response with one File and one Dir descendant.
    // Only the Dir gets a Watch op; the File materializes without an FD
    // contribution. The graft (and thus the descendant Watch ops) runs
    // on the first response even though its verdict is Unstable.
    let snap = dir_tree_snap(vec![
        ("a.rs", EntryKind::File, 1),
        ("subdir", EntryKind::Dir, 2),
    ]);
    let out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
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
        identity: ProfileIdentity {
            config: ScanConfig::builder().build(),
            max_settle: MAX_SETTLE,
            events: NO_EVENTS,
        },
        params: SubParams {
            name: "file-sub".into(),
            program: empty_program(),
            scope: EffectScope::SubtreeRoot,
            settle: SETTLE,
            log_output: false,
            source_promoter: None,
        },
    };
    let now = Instant::now();
    let attach_out = e.step(Input::AttachSub(req), now);
    // Cold-arm Seed: the probe emits at burst construction during
    // `attach_sub`, not on settle-timer expiry.
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
    // Build a Resource with watch_demand=2 (multi-Profile co-located).
    // Inject WatchOpRejected. Expect watch_demand → 0, Unwatch emitted,
    // Diagnostic.
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
    assert!(e.descent_state(ProbeOwner::Profile(pid)).is_some());
    let initial_corr = e
        .pending_probe_for(ProbeOwner::Profile(pid))
        .expect("first probe in flight");
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
        e.descent_state(ProbeOwner::Profile(pid)).is_none(),
        "descent purged on rejection",
    );

    // A Cancel for the in-flight probe was emitted.
    assert!(
        result
            .probe_ops()
            .iter()
            .any(|op| matches!(op, ProbeOp::Cancel { owner: ProbeOwner::Profile(profile)} if *profile == pid)),
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

    // Late `ProbeResponse` for the cancelled correlation arrives — must
    // be silently discarded (descent removed, correlation no longer
    // matches anything).
    let late = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
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
    // Materialized Profile, WatchOpRejected at its anchor — emits
    // ProfileClaimPurged{Anchor} + WatchOpRejected.
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
    // Two Profiles share a descent prefix (e.g., two Subs anchored at
    // siblings under the same scaffold). WatchOpRejected purges both.
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
    assert!(e.descent_state(ProbeOwner::Profile(pid_a)).is_some());
    assert!(e.descent_state(ProbeOwner::Profile(pid_b)).is_some());
    assert_eq!(e.tree.get(foo).unwrap().watch_demand(), 2);

    let result = e.step(
        Input::WatchOpRejected {
            resource: foo,
            failure: specter_core::WatchFailure::Pressure { errno: 24 },
        },
        Instant::now(),
    );

    assert!(e.descent_state(ProbeOwner::Profile(pid_a)).is_none());
    assert!(e.descent_state(ProbeOwner::Profile(pid_b)).is_none());
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
    // Idle Profile (post-`complete_seed_burst`): an overflow drives a
    // direct `start_seed_burst` call; the Profile transitions to
    // `Active(Seed)` Batching-first — no probe yet, a fresh settle
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
    assert!(matches!(burst.phase, PreFirePhase::Verifying(_)));
    assert!(
        burst.dirty.is_empty(),
        "reseed starts a fresh Seed quiescence sequence (cold-arm Verifying-first)",
    );
    assert!(
        e.pending_probe_for(ProbeOwner::Profile(pid)).is_some(),
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
    // Active(Standard) Profile: an overflow `finish_burst_to_idle` +
    // `start_seed_burst` round-trip transitions the burst to
    // `Active(Seed)`. The Standard burst's `dirty` provenance and
    // quiescence prior are discarded — the seed re-baselines.
    let (mut e, pid, _sid, root, _now) = engine_with_attached_sub();
    complete_seed_burst(&mut e, pid);
    let now = Instant::now();
    e.step(
        Input::FsEvent {
            resource: root,
            event: FsEvent::Modified,
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
    assert!(matches!(burst.phase, PreFirePhase::Verifying(_)));
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

/// β★ regression (F-CRIT-1): overflow on a *genuinely armed*
/// `Active(PreFire(Verifying))` — the verify slot is in flight and was
/// NOT pre-consumed — must NOT panic. Pre-β★ the `Active(_, finish)`
/// arm unconditionally dropped the armed slot through
/// `finish_burst_to_idle` and tripped `ProbeSlot`'s Drop tripwire.
/// Under the cold-arm Verifying-first contract, the genuinely-armed
/// Verifying reproduction state is reached at attach (the probe is
/// armed at burst construction, never pre-consumed). Reseed (no Reap):
/// disarm-only via `take_owner_probe` (no wire `Cancel`), then
/// `start_seed_burst` arms a fresh cold-Verifying — one fresh `Probe`
/// emits this step. The guard is that overflow over the armed slot
/// disarms rather than drops it; owner-scoped, exactly one `Probe`,
/// zero `Cancel`.
#[test]
fn sensor_overflow_armed_verifying_reseeds_no_cancel() {
    let (mut e, pid, _sid, _root, now) = engine_with_attached_sub();
    // Cold-arm Seed: the verify probe is in flight directly after
    // attach. Asserting it here is the whole point — a pre-consumed
    // slot would not reproduce F-CRIT-1.
    seed_settle_to_verifying(&e, now);
    let burst = match e.profiles.get(pid).unwrap().state() {
        ProfileState::Active(ActiveBurst::PreFire(pre), _) => pre,
        s => panic!("fixture: expected Active(PreFire(Verifying)); got {s:?}"),
    };
    assert!(matches!(burst.phase, PreFirePhase::Verifying(_)));
    assert!(
        e.pending_probe_for(ProbeOwner::Profile(pid)).is_some(),
        "fixture: Verifying slot genuinely armed (NOT pre-consumed)",
    );

    let out = e.step(
        Input::SensorOverflow {
            scope: OverflowScope::Global,
        },
        Instant::now(),
    );

    // (b) Owner-scoped probe ops: one Probe (cold-arm reseed emits the
    // fresh cold walk), zero Cancel (reseed disarms the engine slot
    // only via take_owner_probe — no wire Cancel).
    let owner = ProbeOwner::Profile(pid);
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

    // (c) Profile back in Active(PreFire(Verifying)) with Seed intent —
    // a fresh cold-arm quiescence sequence. Reaching here without
    // tripping ProbeSlot's Drop tripwire is the F-CRIT-1 guard.
    let burst = match e.profiles.get(pid).unwrap().state() {
        ProfileState::Active(ActiveBurst::PreFire(pre), _) => pre,
        s => panic!("expected Active(Seed) after overflow; got {s:?}"),
    };
    assert_eq!(burst.intent, BurstIntent::Seed);
    assert!(matches!(burst.phase, PreFirePhase::Verifying(_)));
    assert!(
        burst.dirty.is_empty(),
        "reseed starts a fresh Seed quiescence sequence",
    );
    let _ = e.cancel_all_in_flight_probes();
}

/// β★ regression: same as above but on a *genuinely armed*
/// `Active(PostFire(Rebasing))`. The Rebasing slot is the post-effect
/// rebase probe minted by `transition_to_rebasing` — armed, never
/// pre-consumed. Overflow must reseed without panicking, disarming the
/// slot only; the superseding Seed burst is Batching-first.
/// Owner-scoped: zero `Probe`, zero `Cancel`.
#[test]
fn sensor_overflow_armed_rebasing_reseeds_no_cancel() {
    let (mut e, pid, sid, root, _now0) = engine_with_attached_sub();
    let now = Instant::now();
    let stable_out = drive_to_first_effect(&mut e, pid, root, now);
    let key = stable_out.effects()[0].key();
    // EffectComplete::Ok lands the burst in Settling; the PostFireSettle
    // expiry then drives Rebasing where transition_to_rebasing mints a
    // fresh probe.
    let _ = e.step(
        Input::EffectComplete(EffectCompletion {
            sub: sid,
            key,
            outcome: EffectOutcome::Ok,
        }),
        now + SETTLE * 3,
    );
    let settle_id = match e.profiles.get(pid).unwrap().state() {
        ProfileState::Active(ActiveBurst::PostFire(post), _) => match &post.phase {
            PostFirePhase::Settling { settle_timer } => *settle_timer,
            s => panic!("expected Active(PostFire(Settling)) post-EffectComplete; got {s:?}"),
        },
        s => panic!("expected Active(PostFire); got {s:?}"),
    };
    let _ = e.step(
        Input::TimerExpired {
            profile: pid,
            kind: TimerKind::PostFireSettle,
            id: settle_id,
        },
        now + SETTLE * 4,
    );
    let burst = match e.profiles.get(pid).unwrap().state() {
        ProfileState::Active(ActiveBurst::PostFire(post), _) => post,
        s => panic!("fixture: expected Active(PostFire(Rebasing)); got {s:?}"),
    };
    assert!(matches!(burst.phase, PostFirePhase::Rebasing(_)));
    assert!(
        e.pending_probe_for(ProbeOwner::Profile(pid)).is_some(),
        "fixture: Rebasing slot genuinely armed (NOT pre-consumed)",
    );

    let out = e.step(
        Input::SensorOverflow {
            scope: OverflowScope::Global,
        },
        now + SETTLE * 4,
    );

    let owner = ProbeOwner::Profile(pid);
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

    // Reseed re-enters Active(PreFire(Verifying)) with Seed intent —
    // the prior PostFire(Rebasing) burst was abandoned, a fresh
    // quiescence sequence opened (cold-arm Verifying-first). Reaching
    // here without tripping ProbeSlot's Drop tripwire is the β★ guard.
    let burst = match e.profiles.get(pid).unwrap().state() {
        ProfileState::Active(ActiveBurst::PreFire(pre), _) => pre,
        s => panic!("expected Active(Seed) after overflow; got {s:?}"),
    };
    assert_eq!(burst.intent, BurstIntent::Seed);
    assert!(matches!(burst.phase, PreFirePhase::Verifying(_)));
    assert!(
        burst.dirty.is_empty(),
        "reseed starts a fresh Seed quiescence sequence",
    );
    let _ = e.cancel_all_in_flight_probes();
}

/// β★ reap arm: overflow on a *genuinely armed*
/// `Active(PreFire(Verifying))` whose `BurstFinish` is `Reap` (the
/// last Sub was detached mid-burst). Here `will_reap == true`, so the
/// arm emits the wire `Cancel` via `cancel_owner_probe` (no
/// superseding submit follows — `start_seed_burst` no-ops on the
/// detached Profile), then `finish_burst_to_idle` reaps the Profile.
/// Owner-scoped: exactly one `Cancel`, zero `Probe`; Profile gone.
#[test]
fn sensor_overflow_armed_verifying_reap_emits_cancel_only() {
    let (mut e, pid, sid, _root, now) = engine_with_attached_sub();
    // A Seed burst is Batching-first; expire its settle timer to reach
    // the genuinely-armed Active(Seed, Verifying) reproduction state.
    seed_settle_to_verifying(&e, now);
    assert!(
        e.pending_probe_for(ProbeOwner::Profile(pid)).is_some(),
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

    // (b) Owner-scoped: exactly one Cancel, zero Probe. start_seed_burst
    // no-ops on the now-detached Profile, so no fresh submit follows —
    // the wire Cancel spares the worker a doomed walk.
    let owner = ProbeOwner::Profile(pid);
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
fn sensor_overflow_pending_profile_is_skipped() {
    // Pending(_) Profile: descent in flight; no baseline to drift-test.
    // Overflow is a no-op for the Profile state but still emits the
    // diagnostic.
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
        e.descent_state(ProbeOwner::Profile(pid)).is_some(),
        "fixture: profile is in Pending(_)",
    );

    let pre_state = format!("{:?}", e.profiles.get(pid).unwrap().state());
    let out = e.step(
        Input::SensorOverflow {
            scope: OverflowScope::Global,
        },
        Instant::now(),
    );
    let post_state = format!("{:?}", e.profiles.get(pid).unwrap().state());

    assert_eq!(
        pre_state, post_state,
        "Pending Profile state preserved across overflow",
    );
    assert!(
        e.descent_state(ProbeOwner::Profile(pid)).is_some(),
        "descent still in flight after overflow",
    );
    assert!(
        out.diagnostics
            .iter()
            .any(|d| matches!(d, Diagnostic::SensorOverflow { .. })),
        "diagnostic still emitted regardless of per-Profile dispatch",
    );
    let _ = e.cancel_all_in_flight_probes();
}

#[test]
fn sensor_overflow_resource_scope_filters_profiles() {
    // OverflowScope::Resource(r) reseeds only Profiles whose anchor
    // lies in the subtree rooted at r — the FSEvents per-stream signal.
    // Set up two siblings under one root; overflow at the first
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
        identity: ProfileIdentity {
            config: ScanConfig::builder().recursive(true).build(),
            max_settle: MAX_SETTLE,
            events: NO_EVENTS,
        },
        params: SubParams {
            name: "sub-a".into(),
            program: empty_program(),
            scope: EffectScope::SubtreeRoot,
            settle: SETTLE,
            log_output: false,
            source_promoter: None,
        },
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
    let (mut e, pid, sid, r, now) = engine_with_attached_sub();
    // Anchor watch_demand is 1, anchor_claim is Held.
    assert_eq!(e.tree.get(r).unwrap().watch_demand(), 1);
    assert_eq!(
        e.profiles.get(pid).unwrap().anchor_claim(),
        AnchorClaim::Held,
    );

    // A Seed burst is Batching-first; expire its settle timer so a
    // verify probe is in flight to drive the Vanished response below.
    seed_settle_to_verifying(&e, now);

    // Detach the Sub mid-burst → reap_pending = true.
    let _ = e.step(Input::DetachSub(sid), Instant::now());
    assert!(matches!(
        e.profiles.get(pid).unwrap().state().burst_finish(),
        Some(BurstFinish::Reap)
    ));

    // Drive Seed Vanished to fire the reap.
    let correlation = e
        .pending_probe_for(ProbeOwner::Profile(pid))
        .expect("Verifying probe in flight");
    let out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
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
    // After on_anchor_terminal_event releases the anchor, a subsequent
    // reap must NOT double-release it (the claim is cleared to None).
    let (mut e, pid, _sid, r, _now) = engine_with_attached_sub();
    complete_seed_burst(&mut e, pid);
    assert_eq!(
        e.profiles.get(pid).unwrap().anchor_claim(),
        AnchorClaim::Held,
    );

    // Inject a Removed event at the anchor: the terminal event releases
    // the anchor's contribution and clears the claim.
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

/// Defensive guard: a Profile already gone from the map reaching
/// `on_anchor_terminal_all_dynamic` (the caller filters empty-Subs,
/// not a vanished Profile) returns cleanly. Pre-guard this fell
/// through to `path_of(default-id)` → `None` → a `debug_assert!`
/// that panics in test builds with a "live Profile anchor" message —
/// the opposite of the real state.
#[test]
fn on_anchor_terminal_all_dynamic_on_vanished_profile_is_noop() {
    let mut e = Engine::new();
    let mut out = StepOutput::default();
    // The slotmap never yields the null/default key, so this id was
    // never inserted — `profiles.get` is `None`.
    e.on_anchor_terminal_all_dynamic(specter_core::ProfileId::default(), &mut out);
    assert!(out.diagnostics.is_empty(), "no diagnostics");
    assert!(out.probe_ops().is_empty(), "no probe ops");
    assert!(out.effects().is_empty(), "no effects");
    assert!(out.watch_ops.is_empty(), "no watch ops");
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
    // Sub on Active(Standard, stable) Profile; detach mid-burst; finish
    // burst — no Effect emitted; Profile reaped.
    let (mut e, pid, sid, root, _now) = engine_with_attached_sub();
    complete_seed_burst(&mut e, pid);

    // Drive Standard burst.
    let t1 = Instant::now();
    e.step(
        Input::FsEvent {
            resource: root,
            event: FsEvent::Modified,
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
    let correlation = e
        .pending_probe_for(ProbeOwner::Profile(pid))
        .expect("Verifying probe in flight");

    // The verify response folds to `Authoritative { forced: false }`
    // on the first sample — single dispatch. A reap-pending burst
    // suppresses the Effect and finishes by reaping. No Effect is
    // emitted.
    let out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
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
    // Profile with two Subs of different settle; detach the faster one;
    // remaining Sub's settle wins.
    let mut e = Engine::new();
    let r = e.tree.ensure_root("anchor", ResourceRole::User);
    e.tree.set_kind(r, ResourceKind::Dir);
    let now = Instant::now();
    let cfg = ScanConfig::builder().recursive(true).build();
    let attach_out = e.step(
        Input::AttachSub(SubAttachRequest {
            anchor: SubAttachAnchor::Resource(r),
            identity: ProfileIdentity {
                config: cfg.clone(),
                max_settle: MAX_SETTLE,
                events: NO_EVENTS,
            },
            params: SubParams {
                name: "fast".into(),
                program: empty_program(),
                scope: EffectScope::SubtreeRoot,
                settle: Duration::from_millis(50),
                log_output: false,
                source_promoter: None,
            },
        }),
        now,
    );
    let sid_fast =
        specter_core::testkit::first_attached_sub(&attach_out).expect("attach_sub succeeded");
    let pid = e.subs.get(sid_fast).unwrap().profile();
    let _ = e.step(
        Input::AttachSub(SubAttachRequest {
            anchor: SubAttachAnchor::Resource(r),
            identity: ProfileIdentity {
                config: cfg,
                max_settle: MAX_SETTLE,
                events: NO_EVENTS,
            },
            params: SubParams {
                name: "slow".into(),
                program: empty_program(),
                scope: EffectScope::SubtreeRoot,
                settle: Duration::from_millis(200),
                log_output: false,
                source_promoter: None,
            },
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
        identity: ProfileIdentity {
            config: ScanConfig::builder().build(),
            max_settle: MAX_SETTLE,
            events: NO_EVENTS,
        },
        params: SubParams {
            name: "added".into(),
            program: empty_program(),
            scope: EffectScope::SubtreeRoot,
            settle: SETTLE,
            log_output: false,
            source_promoter: None,
        },
    };
    let mut diff = specter_core::SubRegistryDiff::default();
    diff.added.push(req);

    let out = e.step(
        Input::ConfigDiff(WatchRegistryDiff {
            subs: diff,
            ..Default::default()
        }),
        Instant::now(),
    );
    assert!(
        out.watch_ops
            .iter()
            .any(|op| matches!(op, WatchOp::Watch { .. }))
    );
    // Cold-arm Seed: the attach starts the burst AND emits the cold
    // walk probe at construction.
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
                    phase: PreFirePhase::Verifying(_),
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
    // Engine has Sub A at /anchor; ConfigDiff removes A and adds B
    // (path-based, anchored at /anchor — re-creates the slot if A's
    // detach reaped it).
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

    let out = e.step(
        Input::ConfigDiff(WatchRegistryDiff {
            subs: diff,
            ..Default::default()
        }),
        Instant::now(),
    );
    // A reaped (sub registry no longer has it); B added.
    assert!(e.subs().get(sid_a).is_none());
    assert_eq!(e.subs().len(), 1);
    // Single sorted StepOutput; multiple watch_ops merged.
    assert!(!out.watch_ops.is_empty());
    let _ = e.cancel_all_in_flight_probes();
}

/// The name-keyed shim resolves `removed` / `modified_*` against the
/// engine's own registry:
///
/// - a `removed` name the engine never attached emits
///   `Diagnostic::ConfigDiffUnknownSub` — not a silent skip, and not a
///   stale-id `DetachUnknownSub`;
/// - a `modified_params` name the engine never attached degrades to an
///   attach-only retry, narrated by `ConfigDiffRebindFallbackAttach`.
///   A watch whose earlier attach failed (`AttachPathInvalid`) can
///   recover on a later reload through this path rather than being
///   skipped forever.
///
/// Pins the Sub side; the Promoter twin is
/// `config_diff_promoter_removed_with_unknown_name_emits_diagnostic`.
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

    let out = e.step(
        Input::ConfigDiff(WatchRegistryDiff {
            subs,
            ..Default::default()
        }),
        Instant::now(),
    );

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

// ---- on_config_diff: promoter half ----

fn promoter_req(name: &str, pattern: &str) -> PromoterAttachRequest {
    PromoterAttachRequest {
        name: name.into(),
        pattern_spec: PatternSpec::parse(pattern).expect("valid test pattern"),
        identity: ProfileIdentity {
            config: ScanConfig::builder().recursive(true).build(),
            max_settle: MAX_SETTLE,
            events: ClassSet::EMPTY,
        },
        settle: SETTLE,
        program: empty_program(),
        scope: EffectScope::SubtreeRoot,
        log_output: false,
    }
}

/// `promoters.added` runs `attach_promoter_inner` for each request,
/// registering the Promoter and emitting `PromoterAttached`.
#[test]
fn config_diff_promoter_added_attaches_promoter() {
    let mut e = Engine::new();
    // Pre-place the literal-prefix dir so the Promoter lands in
    // immediate-Active mode (no descent state to inspect for this
    // test).
    let _var_log = {
        let r = e
            .tree_mut()
            .ensure_path(&[FS_ROOT_SEGMENT, "var", "log"], ResourceRole::User)
            .expect("non-empty fixture");
        e.tree_mut().set_kind(r, ResourceKind::Dir);
        r
    };

    let diff = WatchRegistryDiff {
        promoters: PromoterRegistryDiff {
            added: vec![promoter_req("logs", "/var/log/*.log")],
            removed: Vec::new(),
            modified: Vec::new(),
        },
        ..Default::default()
    };

    let out = e.step(Input::ConfigDiff(diff), Instant::now());

    assert_eq!(e.promoters.len(), 1, "Promoter registered");
    assert!(
        out.diagnostics.iter().any(|d| matches!(
            d,
            Diagnostic::PromoterAttached { name, .. } if name == "logs"
        )),
        "PromoterAttached diagnostic emitted; got {:?}",
        out.diagnostics,
    );
    let _ = e.cancel_all_in_flight_probes();
}

/// `promoters.removed` runs `reap_promoter_inner` for each id.
/// Cancels any in-flight probe, drops the registry entry, emits
/// `PromoterReaped`.
#[test]
fn config_diff_promoter_removed_reaps_promoter() {
    let mut e = Engine::new();
    let var_log = e
        .tree_mut()
        .ensure_path(&[FS_ROOT_SEGMENT, "var", "log"], ResourceRole::User)
        .expect("non-empty fixture");
    e.tree_mut().set_kind(var_log, ResourceKind::Dir);

    let attach_out = e.step(
        Input::AttachPromoter(promoter_req("logs", "/var/log/*.log")),
        Instant::now(),
    );
    let pid = specter_core::testkit::first_attached_promoter(&attach_out)
        .expect("attach_promoter succeeded");
    assert_eq!(e.promoters.len(), 1);

    let diff = WatchRegistryDiff {
        promoters: PromoterRegistryDiff {
            added: Vec::new(),
            removed: vec![CompactString::from("logs")],
            modified: Vec::new(),
        },
        ..Default::default()
    };

    let out = e.step(Input::ConfigDiff(diff), Instant::now());

    assert!(e.promoters.get(pid).is_none(), "Promoter reaped");
    assert!(
        out.diagnostics.iter().any(|d| matches!(
            d,
            Diagnostic::PromoterReaped { promoter } if *promoter == pid
        )),
        "PromoterReaped diagnostic emitted; got {:?}",
        out.diagnostics,
    );
}

/// `promoters.modified` is wholesale: reap then attach. The new
/// PromoterId differs from the old (the registry mints a fresh slot
/// on attach).
#[test]
fn config_diff_promoter_modified_reaps_and_attaches() {
    let mut e = Engine::new();
    let var_log = e
        .tree_mut()
        .ensure_path(&[FS_ROOT_SEGMENT, "var", "log"], ResourceRole::User)
        .expect("non-empty fixture");
    e.tree_mut().set_kind(var_log, ResourceKind::Dir);

    let attach_out = e.step(
        Input::AttachPromoter(promoter_req("logs", "/var/log/*.log")),
        Instant::now(),
    );
    let old_pid = specter_core::testkit::first_attached_promoter(&attach_out)
        .expect("attach_promoter succeeded");

    let diff = WatchRegistryDiff {
        promoters: PromoterRegistryDiff {
            added: Vec::new(),
            removed: Vec::new(),
            modified: vec![promoter_req("logs", "/var/log/*.json")],
        },
        ..Default::default()
    };

    let out = e.step(Input::ConfigDiff(diff), Instant::now());

    assert!(e.promoters.get(old_pid).is_none(), "old Promoter reaped");
    assert_eq!(e.promoters.len(), 1, "fresh Promoter attached");
    let new_pid = e
        .promoters
        .find_by_name("logs")
        .expect("name re-registered");
    assert_ne!(new_pid, old_pid, "PromoterId minted fresh on modify");

    // Both diagnostics emitted in order: reap then attach.
    let mut saw_reap = false;
    let mut saw_attach_after_reap = false;
    for d in &out.diagnostics {
        match d {
            Diagnostic::PromoterReaped { promoter } if *promoter == old_pid => {
                saw_reap = true;
            }
            Diagnostic::PromoterAttached { promoter, name } if name == "logs" => {
                assert!(saw_reap, "attach must come after reap");
                assert_eq!(*promoter, new_pid);
                saw_attach_after_reap = true;
            }
            _ => {}
        }
    }
    assert!(
        saw_reap,
        "PromoterReaped emitted; got {:?}",
        out.diagnostics
    );
    assert!(
        saw_attach_after_reap,
        "PromoterAttached emitted after reap; got {:?}",
        out.diagnostics,
    );
    let _ = e.cancel_all_in_flight_probes();
}

/// Sub side runs before Promoter side. Confirms ordering by combining
/// a Sub add with a Promoter add and asserting both land in one step.
/// Also exercises that promoter and sub diagnostics merge into the
/// same `StepOutput`.
#[test]
fn config_diff_applies_both_halves_in_one_step() {
    let mut e = Engine::new();
    // Anchor for the static Sub.
    let r = e.tree_mut().ensure_root("anchor", ResourceRole::User);
    e.tree_mut().set_kind(r, ResourceKind::Dir);
    // Literal prefix for the Promoter.
    let var_log = e
        .tree_mut()
        .ensure_path(&[FS_ROOT_SEGMENT, "var", "log"], ResourceRole::User)
        .expect("non-empty fixture");
    e.tree_mut().set_kind(var_log, ResourceKind::Dir);

    let sub_req = SubAttachRequest {
        anchor: SubAttachAnchor::Resource(r),
        identity: ProfileIdentity {
            config: ScanConfig::builder().build(),
            max_settle: MAX_SETTLE,
            events: NO_EVENTS,
        },
        params: SubParams {
            name: "static_a".into(),
            program: empty_program(),
            scope: EffectScope::SubtreeRoot,
            settle: SETTLE,
            log_output: false,
            source_promoter: None,
        },
    };

    let mut sub_diff = specter_core::SubRegistryDiff::default();
    sub_diff.added.push(sub_req);

    let diff = WatchRegistryDiff {
        subs: sub_diff,
        promoters: PromoterRegistryDiff {
            added: vec![promoter_req("dyn_a", "/var/log/*.log")],
            removed: Vec::new(),
            modified: Vec::new(),
        },
    };

    let out = e.step(Input::ConfigDiff(diff), Instant::now());

    assert_eq!(e.subs().len(), 1, "Sub attached");
    assert_eq!(e.promoters.len(), 1, "Promoter attached");
    assert!(
        e.subs().find_by_name("static_a").is_some(),
        "static_a registered by name",
    );
    let promoter_attached = out
        .diagnostics
        .iter()
        .any(|d| matches!(d, Diagnostic::PromoterAttached { name, .. } if name == "dyn_a"));
    assert!(
        promoter_attached,
        "PromoterAttached emitted; got {:?}",
        out.diagnostics,
    );
    let _ = e.cancel_all_in_flight_probes();
}

/// An unresolved `removed` Promoter name is handled gracefully: the
/// name-resolution shim emits `ConfigDiffUnknownPromoter` (never a
/// reap) and the engine doesn't panic. The bin can race a ConfigDiff
/// against an in-flight reap, leaving a dangling `removed` name; this
/// test confirms the benign path.
#[test]
fn config_diff_promoter_removed_with_unknown_name_emits_diagnostic() {
    let mut e = Engine::new();

    let diff = WatchRegistryDiff {
        promoters: PromoterRegistryDiff {
            added: Vec::new(),
            removed: vec![CompactString::from("never-attached")],
            modified: Vec::new(),
        },
        ..Default::default()
    };

    let out = e.step(Input::ConfigDiff(diff), Instant::now());

    // No PromoterReaped diagnostic — the name resolves to nothing,
    // so the shim emits the dedicated unknown diagnostic instead.
    assert!(
        !out.diagnostics
            .iter()
            .any(|d| matches!(d, Diagnostic::PromoterReaped { .. })),
        "unknown name must not emit PromoterReaped; got {:?}",
        out.diagnostics,
    );
    assert!(
        out.diagnostics.iter().any(|d| matches!(
            d,
            Diagnostic::ConfigDiffUnknownPromoter { name } if name == "never-attached"
        )),
        "unresolved removed name must emit ConfigDiffUnknownPromoter; got {:?}",
        out.diagnostics,
    );
}

/// PrefixPending Promoter modify: v1 in PrefixPending (literal
/// prefix doesn't exist); reload modifies. Reap unwinds the
/// PrefixPending state, attach mints a fresh Promoter against the
/// new pattern (which may or may not be PrefixPending depending on
/// disk reality). The pre-flip state-branch in
/// `reap_promoter_inner` handles the PrefixPending arm cleanly.
#[test]
fn config_diff_promoter_modify_during_prefix_pending() {
    let mut e = Engine::new();
    // Don't pre-create the literal prefix → Promoter lands in
    // PrefixPending.
    let attach_out = e.step(
        Input::AttachPromoter(promoter_req("logs", "/missing/dir/*.log")),
        Instant::now(),
    );
    let old_pid = specter_core::testkit::first_attached_promoter(&attach_out)
        .expect("attach_promoter succeeded");
    assert!(matches!(
        e.promoters.get(old_pid).map(Promoter::state),
        Some(PromoterState::PrefixPending(_)),
    ));
    // Descent probe was emitted.
    assert!(
        attach_out
            .probe_ops()
            .iter()
            .any(|op| matches!(op, ProbeOp::Probe { .. })),
        "attach in PrefixPending must emit a descent probe",
    );

    let diff = WatchRegistryDiff {
        promoters: PromoterRegistryDiff {
            added: Vec::new(),
            removed: Vec::new(),
            modified: vec![promoter_req("logs", "/different/dir/*.log")],
        },
        ..Default::default()
    };

    let out = e.step(Input::ConfigDiff(diff), Instant::now());

    assert!(e.promoters.get(old_pid).is_none(), "v1 reaped");
    assert_eq!(e.promoters.len(), 1, "v2 attached");
    // Reap of an in-flight descent emits a Cancel.
    assert!(
        out.probe_ops()
            .iter()
            .any(|op| matches!(op, ProbeOp::Cancel { .. })),
        "PrefixPending reap must Cancel the in-flight descent probe",
    );
    let _ = e.cancel_all_in_flight_probes();
}

// ---- emit_effects PerStableFile ----

#[test]
fn per_stable_file_fires_one_effect_per_created_entry() {
    // Profile with PerStableFile Sub; burst stabilizes with 2 created
    // file entries.
    let mut e = Engine::new();
    let r = e.tree.ensure_root("anchor", ResourceRole::User);
    e.tree.set_kind(r, ResourceKind::Dir);
    let now = Instant::now();
    let req = SubAttachRequest {
        anchor: SubAttachAnchor::Resource(r),
        identity: ProfileIdentity {
            config: ScanConfig::builder().recursive(true).build(),
            max_settle: MAX_SETTLE,
            events: NO_EVENTS,
        },
        params: SubParams {
            name: "fmt".into(),
            program: diff_program(),
            scope: EffectScope::PerStableFile,
            settle: SETTLE,
            log_output: false,
            source_promoter: None,
        },
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
            event: FsEvent::Modified,
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
    let std_corr = e
        .pending_probe_for(ProbeOwner::Profile(pid))
        .expect("Verifying probe in flight");

    // Inject an Authoritative response with 2 file entries — the first
    // sample fires (no prime/confirm dance).
    let snap = dir_tree_snap(vec![
        ("a.rs", EntryKind::File, 1),
        ("b.rs", EntryKind::File, 2),
    ]);
    let out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
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
        // Engine carries the unresolved ActionProgram; the resolver
        // runs in the actuator. Assert the template references the
        // diff-derived `${specter.created}` placeholder (the test fixture's
        // `diff_program()`).
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
        // relative() ⇒ SPECTER_RELATIVE_PATH source. The resolver
        // derives SPECTER_PATH = anchor_path.join(relative()) at
        // spawn time; this assertion implicitly pins target_path to
        // "anchor/a.rs" or "anchor/b.rs".
        assert!(
            eff.relative() == "a.rs" || eff.relative() == "b.rs",
            "relative() = {:?}",
            eff.relative(),
        );
        // SPECTER_EVENT_KIND="file" derives from the DedupKey::PerFile
        // variant.
        assert!(matches!(eff.key(), specter_core::DedupKey::PerFile { .. }));
    }
}

#[test]
fn per_stable_file_skips_dir_entries() {
    // Mixed Diff: 1 created File, 1 created Dir, 1 modified Dir.
    // PerStableFile must fire ONE Effect (the File), not three.
    let mut e = Engine::new();
    let r = e.tree.ensure_root("anchor", ResourceRole::User);
    e.tree.set_kind(r, ResourceKind::Dir);
    let now = Instant::now();
    let req = SubAttachRequest {
        anchor: SubAttachAnchor::Resource(r),
        identity: ProfileIdentity {
            config: ScanConfig::builder().recursive(true).build(),
            max_settle: MAX_SETTLE,
            events: NO_EVENTS,
        },
        params: SubParams {
            name: "fmt".into(),
            program: diff_program(),
            scope: EffectScope::PerStableFile,
            settle: SETTLE,
            log_output: false,
            source_promoter: None,
        },
    };
    let attach_out = e.step(Input::AttachSub(req), now);
    let sid = specter_core::testkit::first_attached_sub(&attach_out).expect("attach_sub succeeded");
    let pid = e.subs.get(sid).unwrap().profile();

    // Seed completes against a snapshot already containing one Dir
    // (`subdir`). After Seed, `subdir` is in the baseline and won't
    // re-appear as `created` later.
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
            event: FsEvent::Modified,
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
    let std_corr = e
        .pending_probe_for(ProbeOwner::Profile(pid))
        .expect("Verifying probe in flight");

    // Mixed snapshot: subdir (modified — different mtime), newdir (new
    // Dir), main.rs (new File). Diff = created=[newdir, main.rs],
    // modified=[subdir]. Only main.rs should fire.
    let mixed_snap = dir_tree_snap(vec![
        ("main.rs", EntryKind::File, 1),
        ("newdir", EntryKind::Dir, 11),
        // subdir has different mtime ⇒ counted as Modified.
        ("subdir", EntryKind::Dir, 10),
    ]);
    // The verify response folds to `Authoritative { forced: false }`
    // on the first sample — single dispatch fires the per-file Effects.
    let out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
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

/// Drive a complete attach + Seed-Ok + FsEvent + single Standard-Ok
/// response and return the StepOutput that contains the Effect
/// emission. Common harness for SubtreeRoot dedup-hash tests.
///
/// The verify response folds through `quiescence_verdict` to
/// `Authoritative { forced: false }` — single sample, fire on
/// classify-consequence Standard. No prime/confirm dance.
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
    let correlation = e
        .pending_probe_for(ProbeOwner::Profile(pid))
        .expect("Verifying probe in flight");
    // Single Authoritative probe response ⇒ fire (Consequence::StandardFire).
    let snap = dir_tree_snap(vec![("a.rs", EntryKind::File, 1)]);
    e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
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
    // The Subtree fire is recorded on this Sub — the post-emit
    // fire-history flag that gates later B1 suppression.
    assert!(
        e.subs.get(sid).is_some_and(|s| s.has_fired),
        "post-emit: Subtree fire recorded for this Sub",
    );
}

/// `Effect.target` for a `Subtree`-keyed Effect is the Profile anchor
/// — captured from `Profile.resource` at emit time. The sort-key
/// extractor pulls `target` directly without a `&Engine` lookup; this
/// pins the emission-side capture so a future refactor that drops the
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
    // Fresh attach, **no FsEvents witnessed** → silent Seed pin. With
    // an empty `dirty`, `seed_owes_first_fire` is false and
    // `seed_drift_observed` is false (never-fired), so the Seed pins
    // silently (restart-safe: Specter persists no baseline, so a daemon
    // restart over an unchanged tree must not re-fire). This is
    // strictly the no-activity path; the witnessed-activity case (a
    // fresh Seed that *did* see events fires exactly one Effect) is
    // covered by the `fresh_seed_fires::*` tests.
    // `complete_seed_burst` returns the pinning response's StepOutput.
    let (mut e, pid, _sid, _root, _now) = engine_with_attached_sub();
    let out = complete_seed_burst(&mut e, pid);
    assert!(
        out.effects().is_empty(),
        "fresh Seed that witnessed no activity fires no Effect"
    );
}

/// Standard burst with a per-stable-file Sub: drift filter is `None`,
/// PerFile keys still emit per matching diff entry. This pins that the
/// SeedDrift-path narrowing (PerFile Subs skipped on `EmitMode::SeedDrift`)
/// doesn't accidentally skip PerFile emission on the unrelated Standard
/// burst path.
#[test]
fn b3_per_key_filter_does_not_affect_standard_burst_perfile_emission() {
    let mut e = Engine::new();
    let r = e.tree.ensure_root("anchor", ResourceRole::User);
    e.tree.set_kind(r, ResourceKind::Dir);
    let now = Instant::now();
    let req = SubAttachRequest {
        anchor: SubAttachAnchor::Resource(r),
        identity: ProfileIdentity {
            config: ScanConfig::builder().recursive(true).build(),
            max_settle: MAX_SETTLE,
            events: ClassSet::CONTENT,
        },
        params: SubParams {
            name: "fmt".into(),
            program: empty_program(),
            scope: EffectScope::PerStableFile,
            settle: SETTLE,
            log_output: false,
            source_promoter: None,
        },
    };
    let _ = e.step(Input::AttachSub(req), now);
    let pid = e.profiles.iter().next().unwrap().0;
    // Seed → Idle (establishes the baseline before the Standard burst
    // below).
    complete_seed_burst(&mut e, pid);

    // Standard burst with a created file → PerFile Effect emits.
    e.step(
        Input::FsEvent {
            resource: r,
            event: FsEvent::Modified,
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
        let correlation = e
            .pending_probe_for(ProbeOwner::Profile(pid))
            .expect("Verifying probe in flight");
        let out = e.step(
            Input::ProbeResponse(ProbeResponse {
                owner: ProbeOwner::Profile(pid),
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
    // The events mask folds into `config_hash`, so every Sub on a Profile
    // shares the same events by construction. `has_per_file_fds` is
    // derived once at `Profile::new` from the events mask and never flips
    // during the Profile's lifetime.
    //
    // This test pins the new invariant: a Profile constructed with a
    // mask containing CONTENT (or METADATA) starts with the flag set,
    // and a Sub attaching via the same `(resource, config_hash)` does
    // not change it.
    let mut e = Engine::new();
    let r = e.tree.ensure_root("anchor", ResourceRole::User);
    e.tree.set_kind(r, ResourceKind::Dir);
    let req = SubAttachRequest {
        anchor: SubAttachAnchor::Resource(r),
        identity: ProfileIdentity {
            config: ScanConfig::builder().recursive(true).build(),
            max_settle: MAX_SETTLE,
            events: ClassSet::CONTENT,
        },
        params: SubParams {
            name: "formatter".into(),
            program: empty_program(),
            scope: EffectScope::PerStableFile,
            settle: SETTLE,
            log_output: false,
            source_promoter: None,
        },
    };
    let out = e.step(Input::AttachSub(req), Instant::now());
    let sid = specter_core::testkit::first_attached_sub(&out).expect("attach_sub succeeded");
    let pid = e.subs.get(sid).unwrap().profile();
    assert!(
        e.profiles.get(pid).unwrap().has_per_file_fds(),
        "CONTENT-mask Profile has has_per_file_fds = true at construction",
    );

    // A Sub with the same `(resource, max_settle, scan, events)` shares
    // the existing Profile; the flag stays true.
    let req2 = SubAttachRequest {
        anchor: SubAttachAnchor::Resource(r),
        identity: ProfileIdentity {
            config: ScanConfig::builder().recursive(true).build(),
            max_settle: MAX_SETTLE,
            events: ClassSet::CONTENT,
        },
        params: SubParams {
            name: "formatter-2".into(),
            program: empty_program(),
            scope: EffectScope::PerStableFile,
            settle: SETTLE,
            log_output: false,
            source_promoter: None,
        },
    };
    let _ = e.step(Input::AttachSub(req2), Instant::now());
    assert!(e.profiles.get(pid).unwrap().has_per_file_fds());

    // Detaching the second Sub leaves the Profile alive (a Sub still
    // remains before detach); the flag still doesn't flip because the
    // Profile's events mask is invariant.
    let _ = e.step(Input::DetachSub(sid), Instant::now());
    assert!(e.profiles.get(pid).unwrap().has_per_file_fds());
    let _ = e.cancel_all_in_flight_probes();
}

#[test]
fn structure_only_profile_has_per_file_fds_false() {
    // Inverse case: a STRUCTURE-only mask leaves `has_per_file_fds`
    // false. walk_pair then doesn't bump per-leaf watch_demand for
    // covered files.
    let mut e = Engine::new();
    let r = e.tree.ensure_root("anchor", ResourceRole::User);
    e.tree.set_kind(r, ResourceKind::Dir);
    let req = SubAttachRequest {
        anchor: SubAttachAnchor::Resource(r),
        identity: ProfileIdentity {
            config: ScanConfig::builder().recursive(true).build(),
            max_settle: MAX_SETTLE,
            events: ClassSet::STRUCTURE,
        },
        params: SubParams {
            name: "ls-only".into(),
            program: empty_program(),
            scope: EffectScope::SubtreeRoot,
            settle: SETTLE,
            log_output: false,
            source_promoter: None,
        },
    };
    let attach_out = e.step(Input::AttachSub(req), Instant::now());
    let sid = specter_core::testkit::first_attached_sub(&attach_out).expect("attach_sub succeeded");
    let pid = e.subs.get(sid).unwrap().profile();
    assert!(!e.profiles.get(pid).unwrap().has_per_file_fds());
    let _ = e.cancel_all_in_flight_probes();
}

// ---------- Anchor-loss kind-cache invalidation ----------
//
// Per-site assertions that every dispatch path through
// `Engine::discard_anchor_state` clears the cached `Profile.kind`. The
// helper unit tests in `claims_tests.rs` pin the contract in
// isolation; these tests pin the integration at the seven production
// call sites so the kind-clear cannot regress at any one of them
// without a test failure.

/// Drive a Profile from fresh-attach into `Active(Standard, Verifying)`
/// with `pending_probe.is_some()`. Returns the live correlation.
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
    e.pending_probe_for(ProbeOwner::Profile(pid))
        .expect("Verifying probe in flight")
}

/// Drive into `Active(_, Rebasing)` by completing a Standard burst's
/// stable verdict + Effect → EffectComplete::Ok + PostFireSettle
/// expiry. Returns the rebase probe correlation so the caller can
/// drive the rebase response.
///
/// The natural post-`Awaiting` phase is `Settling`, with `Rebasing`
/// reached via `PostFireSettle` expiry. The helper drives both
/// transitions so the caller retains a rebase correlation.
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
    // EffectComplete::Ok lands the burst in Settling.
    let _ = e.step(
        Input::EffectComplete(EffectCompletion {
            sub: sid,
            key,
            outcome: EffectOutcome::Ok,
        }),
        now + SETTLE * 3,
    );
    // Drive PostFireSettle expiry to advance Settling → Rebasing,
    // where the rebase probe is in flight.
    let settle_id = match e.profiles.get(pid).unwrap().state() {
        ProfileState::Active(ActiveBurst::PostFire(post), _) => match &post.phase {
            PostFirePhase::Settling { settle_timer } => *settle_timer,
            other => panic!(
                "expected Active(PostFire(Settling)) after EffectComplete::Ok, got {other:?}",
            ),
        },
        other => panic!("expected Active(PostFire) after EffectComplete::Ok, got {other:?}"),
    };
    let _ = e.step(
        Input::TimerExpired {
            profile: pid,
            kind: TimerKind::PostFireSettle,
            id: settle_id,
        },
        now + SETTLE * 4,
    );
    e.pending_probe_for(ProbeOwner::Profile(pid))
        .expect("rebase probe in flight after PostFireSettle drove Settling → Rebasing")
}

#[test]
fn dispatch_seed_vanished_clears_profile_kind() {
    let (mut e, pid, _sid, _r, now) = engine_with_attached_sub();
    assert_eq!(
        e.profiles.get(pid).unwrap().kind(),
        Some(ResourceKind::Dir),
        "fresh attach caches anchor's classified kind",
    );
    // Seed is Batching-first; expire the settle timer to put a verify
    // probe in flight, then answer it Vanished.
    seed_settle_to_verifying(&e, now);
    let correlation = e
        .pending_probe_for(ProbeOwner::Profile(pid))
        .expect("Seed Verifying probe");
    let _ = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
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
fn dispatch_seed_failed_clears_profile_kind() {
    let (mut e, pid, _sid, _r, now) = engine_with_attached_sub();
    // Seed is Batching-first; expire the settle timer to put a verify
    // probe in flight, then answer it Failed.
    seed_settle_to_verifying(&e, now);
    let correlation = e
        .pending_probe_for(ProbeOwner::Profile(pid))
        .expect("Seed Verifying probe");
    let _ = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
            correlation,
            outcome: ProbeOutcome::Failed { errno: 5 },
        }),
        Instant::now(),
    );
    assert!(
        e.profiles.get(pid).unwrap().kind().is_none(),
        "Seed-Failed must clear the cached anchor kind",
    );
}

#[test]
fn dispatch_standard_vanished_clears_profile_kind() {
    let (mut e, pid, _sid, root, now) = engine_with_attached_sub();
    let correlation = drive_to_standard_verifying(&mut e, pid, root, now);
    assert_eq!(
        e.profiles.get(pid).unwrap().kind(),
        Some(ResourceKind::Dir),
        "kind cached pre-dispatch",
    );
    let _ = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
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
fn dispatch_standard_failed_clears_profile_kind() {
    let (mut e, pid, _sid, root, now) = engine_with_attached_sub();
    let correlation = drive_to_standard_verifying(&mut e, pid, root, now);
    let _ = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
            correlation,
            outcome: ProbeOutcome::Failed { errno: 13 },
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
            owner: ProbeOwner::Profile(pid),
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
            owner: ProbeOwner::Profile(pid),
            correlation,
            outcome: ProbeOutcome::Failed { errno: 5 },
        }),
        now + SETTLE * 4,
    );
    assert!(
        e.profiles.get(pid).unwrap().kind().is_none(),
        "Rebase-Failed must clear the cached anchor kind",
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

/// Pin `finalize_anchor_lost`'s ordering invariant: `was_active` is
/// captured BEFORE `discard_anchor_state` runs. Exercises the
/// Active-burst path and asserts the burst is finished to Idle (i.e.
/// the `was_active = true` branch ran). A future helper change that
/// flips `state` mid-helper would otherwise silently break the
/// burst-end pathway.
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

/// The rebase probe's obligation is unconditionally `WholeSubtree`,
/// even when an Awaiting absorb populated `dirty`: the command just
/// mutated the tree, so there is no trustworthy prior to scope a
/// `Chains` walk against — an in-place descendant edit need not bump
/// an ancestor mtime, so a chains/mtime skip would re-clone a stale
/// subtree and certify a false quiet. `dirty` is no longer a
/// post-fire obligation source; `transition_to_rebasing` clears it at
/// the loop entry, so an Awaiting-absorbed event is folded into the
/// `WholeSubtree` read itself rather than carried as a restart seed.
///
/// Sub uses `ClassSet::CONTENT` so the descendant `Modified` event
/// passes both gates: (1) a per-file FD is wired up by the standard
/// burst's reconcile (`has_per_file_fds = true`), bumping the leaf's
/// `watch_demand` past `on_fs_event`'s zero-gate, and (2) the
/// per-Profile class filter (which sits BEFORE `drive_burst`'s absorb
/// arm) admits the CONTENT-classed event.
#[test]
fn rebasing_probes_whole_subtree_and_resets_awaiting_absorbed_residual() {
    let mut e = Engine::new();
    let root = e.tree.ensure_root("anchor", ResourceRole::User);
    e.tree.set_kind(root, ResourceKind::Dir);
    let now = Instant::now();
    let req = SubAttachRequest {
        anchor: SubAttachAnchor::Resource(root),
        identity: ProfileIdentity {
            config: ScanConfig::builder().recursive(true).build(),
            max_settle: MAX_SETTLE,
            events: ClassSet::CONTENT,
        },
        params: SubParams {
            name: "test-sub".into(),
            program: empty_program(),
            scope: EffectScope::SubtreeRoot,
            settle: SETTLE,
            log_output: false,
            source_promoter: None,
        },
    };
    let attach_out = e.step(Input::AttachSub(req), now);
    let sid = specter_core::testkit::first_attached_sub(&attach_out).expect("attach_sub succeeded");
    let pid = e.subs.get(sid).unwrap().profile();

    let stable_out = drive_to_first_effect(&mut e, pid, root, now);
    assert_eq!(stable_out.effects().len(), 1, "stable verdict fires Effect");
    let key = stable_out.effects()[0].key();

    // Look up the descendant the standard burst's reconcile created.
    // `drive_to_first_effect` ships `[("a.rs", File, 1)]` as the
    // probe response; the engine's graft creates an `a.rs` Resource
    // under root and bumps its watch_demand (per-file FD) because the
    // Profile carries CONTENT in its events_union.
    let descendant = e
        .tree
        .lookup(Some(root), "a.rs")
        .expect("standard burst's reconcile created a.rs");
    assert!(
        e.tree.get(descendant).is_some_and(|r| r.watch_demand() > 0),
        "per-file FD must be wired up for the descendant — otherwise \
         the Modified event drops at on_fs_event's watch_demand gate \
         before reaching the absorb arm",
    );

    // Inject an FsEvent during Awaiting → absorb arm. `Modified` is
    // the in-place content-edit class — the same FsEvent kqueue emits
    // for a `write(2)` against a per-file FD, which is the carve-out
    // scenario this test pins (the parent dir's mtime is unchanged).
    let absorb_out = e.step(
        Input::FsEvent {
            resource: descendant,
            event: FsEvent::Modified,
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

    // EffectComplete::Ok lands the burst in Settling; the PostFireSettle
    // expiry then drives Rebasing where the rebase probe is minted.
    let _ = e.step(
        Input::EffectComplete(EffectCompletion {
            sub: sid,
            key,
            outcome: EffectOutcome::Ok,
        }),
        now + SETTLE * 3,
    );
    let settle_id = match e.profiles.get(pid).unwrap().state() {
        ProfileState::Active(ActiveBurst::PostFire(post), _) => match &post.phase {
            PostFirePhase::Settling { settle_timer } => *settle_timer,
            other => panic!("expected Active(PostFire(Settling)); got {other:?}"),
        },
        other => panic!("expected Active(PostFire); got {other:?}"),
    };
    let rebase_out = e.step(
        Input::TimerExpired {
            profile: pid,
            kind: TimerKind::PostFireSettle,
            id: settle_id,
        },
        now + SETTLE * 4,
    );

    let req = rebase_out
        .probe_ops()
        .iter()
        .find_map(|op| match op {
            ProbeOp::Probe { request } => Some(request),
            ProbeOp::Cancel { .. } => None,
        })
        .expect("Rebase probe minted on PostFireSettle expiry (Settling → Rebasing)");
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

/// Idempotent fire-tail: even with no FsEvent absorbed during
/// Awaiting, the rebase probe ships `WholeSubtree` and is never
/// `forced`. The post-command tree has no trustworthy prior — an
/// in-place descendant edit need not bump an ancestor mtime, so the
/// walker must re-read the whole subtree regardless of mtime or any
/// (now-absent) scoped chain. Pins that the rebase obligation is a
/// soundness floor, not an absorb-conditioned optimization.
#[test]
fn rebasing_without_absorbs_still_probes_whole_subtree() {
    let (mut e, pid, sid, root, _now0) = engine_with_attached_sub();
    let now = Instant::now();
    let stable_out = drive_to_first_effect(&mut e, pid, root, now);
    let key = stable_out.effects()[0].key();

    // EffectComplete::Ok lands the burst in Settling. Drive
    // PostFireSettle expiry to advance Settling → Rebasing where the
    // rebase probe is minted.
    let _ = e.step(
        Input::EffectComplete(EffectCompletion {
            sub: sid,
            key,
            outcome: EffectOutcome::Ok,
        }),
        now + SETTLE * 3,
    );
    let settle_id = match e.profiles.get(pid).unwrap().state() {
        ProfileState::Active(ActiveBurst::PostFire(post), _) => match &post.phase {
            PostFirePhase::Settling { settle_timer } => *settle_timer,
            other => panic!("expected Active(PostFire(Settling)); got {other:?}"),
        },
        other => panic!("expected Active(PostFire); got {other:?}"),
    };
    let rebase_out = e.step(
        Input::TimerExpired {
            profile: pid,
            kind: TimerKind::PostFireSettle,
            id: settle_id,
        },
        now + SETTLE * 4,
    );

    let req = rebase_out
        .probe_ops()
        .iter()
        .find_map(|op| match op {
            ProbeOp::Probe { request } => Some(request),
            ProbeOp::Cancel { .. } => None,
        })
        .expect("Rebase probe minted on PostFireSettle expiry (Settling → Rebasing)");
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
/// `PostFireBurst.last_event_time`, and the next `PostFireSettle`
/// expiry reschedules (now − last_event_time < settle) instead of
/// transitioning to `Rebasing` — the post-fire mirror of pre-fire's
/// `event_drives_batching` reschedule. A subsequent expiry past the
/// quiet window then completes the natural Settling → Rebasing advance.
#[test]
fn post_fire_settling_reschedules_on_absorbed_event() {
    // Use a CONTENT-mask Sub so a Modified event at the anchor's covered
    // descendant reaches the absorb arm. The anchor itself also accepts
    // events unconditionally (anchor events bypass the class filter).
    let mut e = Engine::new();
    let root = e.tree.ensure_root("anchor", ResourceRole::User);
    e.tree.set_kind(root, ResourceKind::Dir);
    let req = SubAttachRequest {
        anchor: SubAttachAnchor::Resource(root),
        identity: ProfileIdentity {
            config: ScanConfig::builder().recursive(true).build(),
            max_settle: MAX_SETTLE,
            events: ClassSet::CONTENT,
        },
        params: SubParams {
            name: "test-sub".into(),
            program: empty_program(),
            scope: EffectScope::SubtreeRoot,
            settle: SETTLE,
            log_output: false,
            source_promoter: None,
        },
    };
    let attach_out = e.step(Input::AttachSub(req), Instant::now());
    let sid = specter_core::testkit::first_attached_sub(&attach_out).expect("attach_sub succeeded");
    let pid = e.subs.get(sid).unwrap().profile();

    let now = Instant::now();
    let stable_out = drive_to_first_effect(&mut e, pid, root, now);
    let key = stable_out.effects()[0].key();

    // EffectComplete::Ok → Awaiting → Settling. Capture the initial
    // PostFireSettle timer id and the EffectComplete instant.
    let now_a = now + SETTLE * 3;
    let _ = e.step(
        Input::EffectComplete(EffectCompletion {
            sub: sid,
            key,
            outcome: EffectOutcome::Ok,
        }),
        now_a,
    );
    let settle_timer_1 = match e.profiles.get(pid).unwrap().state() {
        ProfileState::Active(ActiveBurst::PostFire(post), _) => match &post.phase {
            PostFirePhase::Settling { settle_timer } => *settle_timer,
            other => panic!("expected Active(PostFire(Settling)); got {other:?}"),
        },
        other => panic!("expected Active(PostFire); got {other:?}"),
    };

    // Absorb an anchor FsEvent strictly inside the settle window
    // (now_a + 5ms ≪ SETTLE). The absorb updates last_event_time and
    // notes into final_window_residual.
    let now_b = now_a + Duration::from_millis(5);
    let absorb_out = e.step(
        Input::FsEvent {
            resource: root,
            event: FsEvent::Modified,
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

    // First PostFireSettle expiry lands at the original deadline
    // (now_a + SETTLE). The reschedule check: now_c − last_event_time =
    // now_a + SETTLE − now_b < SETTLE (since now_b > now_a). The
    // handler must schedule a fresh PostFireSettle timer and stay in
    // Settling; no rebase probe.
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

    // Second PostFireSettle expiry at the new deadline (now_b + SETTLE).
    // Now the quiet window has closed; the handler transitions
    // Settling → Rebasing and mints the rebase probe.
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
            ProbeOp::Probe { request } if request.owner() == ProbeOwner::Profile(pid),
        )),
        "Settling → Rebasing mints a fresh rebase probe",
    );
    let _ = e.cancel_all_in_flight_probes();
}

/// An `Undischarged { terminal: false }` rebase response loops back
/// through `transition_to_settling` (no commit), the next
/// `PostFireSettle` expiry re-enters `Rebasing` with a fresh probe,
/// and a follow-up `Authoritative` response commits and finishes — the
/// only surviving post-fire loop. Pairs with
/// `rebase_undischarged_not_reached_does_not_poison_current` (which
/// pins the first loop-back); this test pins the COMPLETE retry path:
/// loop entry → settle expiry → re-Rebasing → Authoritative commit.
#[test]
fn post_fire_undischarged_retries_via_settling() {
    let (mut e, pid, sid, root, now) = engine_with_attached_sub();
    let rebase_corr = drive_to_rebasing(&mut e, pid, sid, root, now);
    let baseline_before = baseline_hash(&e, pid);

    // First Rebasing response: Undischarged + !terminal → Settling.
    // No commit; the baseline must not move.
    let unread: std::sync::Arc<std::path::Path> =
        std::sync::Arc::from(std::path::Path::new("anchor/opaque"));
    let degraded = dir_tree_snap(vec![("ghost", EntryKind::File, 9)]);
    let now_loop = now + SETTLE * 4;
    let _ = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
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
            other => panic!("Undischarged !terminal must loop into Settling; got {other:?}"),
        },
        other => panic!("expected Active(PostFire(Settling)); got {other:?}"),
    };
    assert_eq!(
        baseline_hash(&e, pid),
        baseline_before,
        "Undischarged !terminal must not commit — baseline stays put",
    );

    // PostFireSettle expiry past the settle window: Settling → Rebasing
    // with a fresh probe (different correlation from the first).
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

    // Authoritative on the retry: commit + finish. The baseline now
    // moves to the fresh snapshot.
    let fresh = dir_tree_snap(vec![("a.rs", EntryKind::File, 1)]);
    let _ = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
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

/// The gate-deadline non-zombie recovery skips the rebase-loop ceiling
/// entirely: `handle_gate_deadline` calls `force_pending_post_fire`
/// (lockstep `forced := true; rebase_ceiling := None`) then
/// `transition_to_rebasing` directly — there is no `Settling` window
/// between `Awaiting` and `Rebasing`. Pairs with
/// `fire_cycle_gate_deadline_force_transitions_to_rebasing` (which pins
/// the phase and probe emission); this test additionally pins the
/// field-level shape (`forced == true`, `rebase_ceiling.is_none()`)
/// and the absence of a `PostFireSettle` schedule on the path.
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

    // Fire the AwaitGateDeadline timer past the gate window. Use
    // `step` directly with the captured id so the test runs the
    // single transition under inspection (not a multi-timer drain).
    let now_gate = now + MAX_SETTLE * 8;
    let out_gate = e.step(
        Input::TimerExpired {
            profile: pid,
            kind: TimerKind::AwaitGateDeadline,
            id: gate_deadline_id,
        },
        now_gate,
    );

    // Phase: Awaiting → Rebasing directly (no Settling in between).
    // Fields: forced == true (latched), rebase_ceiling == None (skipped
    // — gate-deadline has already waited 4× max_settle).
    match e.profiles.get(pid).unwrap().state() {
        ProfileState::Active(ActiveBurst::PostFire(post), _) => {
            assert!(
                matches!(post.phase, PostFirePhase::Rebasing(_)),
                "gate-deadline transitions Awaiting → Rebasing directly; got {:?}",
                post.phase,
            );
            assert!(post.forced, "force_pending_post_fire latched forced = true");
            assert!(
                post.rebase_ceiling.is_none(),
                "rebase_ceiling skipped on the gate-deadline path (lockstep \
                 with forced = true)",
            );
        }
        other => panic!("expected Active(PostFire(Rebasing)); got {other:?}"),
    }

    // The Rebasing burst carries no PostFireSettle token: the
    // gate-deadline went Awaiting → Rebasing directly, never opening a
    // Settling window.
    assert!(
        e.profiles
            .get(pid)
            .unwrap()
            .state()
            .timer_token(TimerKind::PostFireSettle)
            .is_none(),
        "gate-deadline skips Settling entirely — no PostFireSettle armed",
    );

    // Rebase probe emitted; respond Authoritative to confirm the
    // forced=true commit terminal.
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
            owner: ProbeOwner::Profile(pid),
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
            Diagnostic::RebaseCeilingStillChanging { profile, .. } if *profile == pid,
        )),
        "forced=true commit emits the RebaseCeilingStillChanging diagnostic",
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

/// `Authoritative`, ceiling not reached: an `Authoritative { forced:
/// false }` rebase response commits the snapshot, rebases the baseline,
/// and finishes — no loop, no settle-spaced sample.
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
            owner: ProbeOwner::Profile(pid),
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

/// `Undischarged`, ceiling not reached: an unread region must never
/// poison `current` — **no** `apply_snapshot` — and the carrier prior
/// is withheld. The loop settle-spaces for another sample.
#[test]
fn rebase_undischarged_not_reached_does_not_poison_current() {
    let (mut e, pid, sid, root, now) = engine_with_attached_sub();
    let rebase_corr = drive_to_rebasing(&mut e, pid, sid, root, now);
    let baseline_before = baseline_hash(&e, pid);
    let current_before = current_hash(&e, pid);

    // An unread response: the walker could not discharge its
    // obligation at `first_unread`.
    let unread: std::sync::Arc<std::path::Path> =
        std::sync::Arc::from(std::path::Path::new("anchor/opaque"));
    let degraded = dir_tree_snap(vec![("ghost", EntryKind::File, 9)]);
    e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
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
            "Undischarged + !reached loops back through RebaseSettling; got {:?}",
            post.phase,
        ),
        other => panic!("expected Active(PostFire(RebaseSettling)); got {other:?}"),
    }
    assert_eq!(
        current_hash(&e, pid),
        current_before,
        "Undischarged + !reached must NOT apply_snapshot — an unread region cannot poison current",
    );
    assert_eq!(
        baseline_hash(&e, pid),
        baseline_before,
        "Undischarged + !reached never rebases the baseline",
    );
}

/// `Authoritative`, ceiling **reached** (forced=true): the
/// `RebaseCeiling` fired but the walker still certified. Pin the
/// freshest observation as the new baseline anyway (a deliberate, loud
/// terminal — not a wedge) and finish.
#[test]
fn rebase_authoritative_at_ceiling_pins_freshest_and_diagnoses() {
    let (mut e, pid, sid, root, now) = engine_with_attached_sub();
    let rebase_corr = drive_to_rebasing(&mut e, pid, sid, root, now);

    // Latch the ceiling while the probe is in flight (set-only — the
    // in-flight response carries the terminal as forced=true).
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
            owner: ProbeOwner::Profile(pid),
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
            Diagnostic::RebaseCeilingStillChanging { profile, intent }
                if *profile == pid && *intent == BurstIntent::Standard,
        )),
        "Authoritative + forced=true emits the loud RebaseCeilingStillChanging diagnostic; got {:?}",
        out.diagnostics,
    );
}

/// `Undischarged`, ceiling **reached**: refuse to rebase blind. No
/// commit, no rebase — the prior baseline stays in place — plus the
/// loud `RebaseCeilingUnreadable` carrying `first_unread`.
#[test]
fn rebase_undischarged_at_ceiling_refuses_blind_rebase_and_diagnoses() {
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
            owner: ProbeOwner::Profile(pid),
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
        "Undischarged + Reached is a terminal — the burst finishes",
    );
    assert_eq!(
        baseline_hash(&e, pid),
        baseline_before,
        "Undischarged + Reached never rebases blind — the prior baseline stays in place",
    );
    assert!(
        out.diagnostics.iter().any(|d| matches!(
            d,
            Diagnostic::RebaseCeilingUnreadable { profile, first_unread, intent }
                if *profile == pid
                    && first_unread.as_ref() == std::path::Path::new("anchor/opaque")
                    && *intent == BurstIntent::Standard,
        )),
        "Undischarged + Reached emits RebaseCeilingUnreadable carrying first_unread; got {:?}",
        out.diagnostics,
    );
}

/// B4 mirror — ceiling expiry **in `Rebasing`** (a probe in flight, the
/// `Verifying` analogue): set-only. No immediate re-drive, no new
/// probe; the in-flight response applies the terminal via
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
                post.forced,
                "the ceiling latched: forced raised by force_pending_post_fire",
            );
        }
        other => panic!("expected Active(PostFire(Rebasing)); got {other:?}"),
    }

    // The original in-flight probe's response applies the terminal.
    e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
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

/// B4 mirror — ceiling expiry **in `RebaseSettling`** (no probe in
/// `RebaseCeiling` expiry while the burst is in `Settling` (no probe
/// in flight) latches `forced = true` and drives `Rebasing` in the
/// same step — the post-fire mirror of `handle_burst_deadline`
/// driving a verify when no Verifying probe is in flight. The
/// `Authoritative { forced: true }` response then commits + emits
/// `RebaseCeilingStillChanging`.
#[test]
fn rebase_ceiling_in_settling_drives_rebasing_with_forced() {
    let (mut e, pid, sid, root, now) = engine_with_attached_sub();
    let rebase_corr = drive_to_rebasing(&mut e, pid, sid, root, now);

    // Loop back through Settling via an Undischarged !terminal response
    // (the only surviving post-fire loop). The `Authoritative !forced`
    // arm finishes immediately; we need the burst still alive in
    // Settling to fire the ceiling there.
    let unread: std::sync::Arc<std::path::Path> =
        std::sync::Arc::from(std::path::Path::new("anchor/opaque"));
    e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
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
        "Undischarged !terminal loops the post-fire burst back through Settling",
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
            assert!(post.forced, "the ceiling latched: forced=true");
        }
        other => panic!("expected Active(PostFire(Rebasing)); got {other:?}"),
    }

    // That driven probe's `Authoritative` response folds with forced=true
    // to the ceiling-pin terminal: commit + diagnose + finish.
    let freshest = dir_tree_snap(vec![("a.rs", EntryKind::File, 1)]);
    let final_out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
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
            Diagnostic::RebaseCeilingStillChanging { profile, .. } if *profile == pid,
        )),
        "ceiling terminal emits RebaseCeilingStillChanging; got {:?}",
        final_out.diagnostics,
    );
}

// ---------- rebase_baseline witness clears at every site ----------

/// Construct an `Active(PreFire)` state populated with default empty
/// per-burst sets and the supplied phase / intent / probe target. Used
/// by witness-clear tests that drive `dispatch_*_ok` directly with a
/// pre-staged Profile.
fn active_pre_fire_burst(
    e: &mut Engine,
    pid: specter_core::ProfileId,
    phase: PreFirePhase,
    intent: BurstIntent,
    probe_target: ResourceId,
    now: Instant,
) -> ProfileState {
    let burst_deadline = e
        .timers
        .schedule(now + MAX_SETTLE, pid, TimerKind::BurstDeadline);
    ProfileState::Active(
        ActiveBurst::PreFire(PreFireBurst {
            burst_deadline,
            phase,
            intent,
            forced: false,
            dirty: DirtyProvenance::new(),
            probe_target,
            last_event_time: None,
        }),
        BurstFinish::ReturnToIdle,
    )
}

/// Construct an `Active(PostFire)` state with the supplied phase /
/// intent. Used by witness-clear tests that drive `dispatch_rebase_*`
/// directly with a pre-staged Profile.
fn active_post_fire_burst(
    _e: &mut Engine,
    _pid: specter_core::ProfileId,
    phase: PostFirePhase,
    intent: BurstIntent,
    _probe_target: ResourceId,
    _now: Instant,
) -> ProfileState {
    ProfileState::Active(
        ActiveBurst::PostFire(PostFireBurst::new(intent, phase, DirtyProvenance::new())),
        BurstFinish::ReturnToIdle,
    )
}

/// Drive an attached Profile into **survival mode** — the post
/// anchor-loss shape: the anchor collapses to `Unclassified` with the
/// pre-loss baseline hash retained as the survival witness
/// (`settled_hash() == Some(witness_snap.dir_hash())`; `baseline()` and
/// `current()` both `None`). Built only from the production `Profile`
/// API, mirroring the engine's take-then-clear loss sequence, so the
/// captured witness is exactly `witness_snap`'s hash.
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

/// Drive an attached Profile into **active mode**: `baseline =
/// Dir(baseline_snap)`, `current = Dir(current_snap)`, witness `None`,
/// `kind = Some(Dir)` — built only from the production setters.
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
    let (mut e, pid, _sid, anchor, now) = engine_with_attached_sub();
    // Drain the attach-time Seed-Verifying probe before the manual
    // `transition_state` clobber below drops that armed state.
    let _ = e.cancel_all_in_flight_probes();

    let witness_snap = dir_tree_snap(vec![]);
    let witness_hash = witness_snap.dir_hash();
    enter_survival_mode(&mut e, pid, witness_snap);
    let state = active_post_fire_burst(
        &mut e,
        pid,
        PostFirePhase::Rebasing(ProbeSlot::empty()),
        BurstIntent::Standard,
        anchor,
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

    // Rebase grafts a snapshot distinct from the witness so the
    // Witness → Snapshot consume is observable in `settled_hash()`.
    let rebased = dir_tree_snap(vec![("rebased", EntryKind::File, 7)]);
    let rebased_hash = rebased.dir_hash();
    assert_ne!(
        rebased_hash, witness_hash,
        "test setup: rebased snapshot must differ from the witness",
    );
    // An `Authoritative { forced: false }` rebase verdict is the
    // consume-the-witness arm: apply_snapshot + rebase_baseline + finish.
    // (The looping / ceiling-terminal arms are exercised by the
    // rebase-loop tests.)
    let mut out = StepOutput::default();
    e.dispatch_rebase_ok(
        pid,
        TreeSnapshot::Dir(rebased),
        QuiescenceVerdict::Authoritative { forced: false },
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

#[test]
fn seed_recovery_seal_consumes_survival_witness() {
    let (mut e, pid, _sid, anchor, now) = engine_with_attached_sub();
    // Drain the attach-time Seed-Verifying probe before the manual
    // `transition_state` clobber below drops that armed state.
    let _ = e.cancel_all_in_flight_probes();

    // Survival mode at entry (no live baseline, witness populated);
    // empty fired_subs ⇒ no drift — but Seed still rebases, consuming
    // the witness.
    let witness_snap = dir_tree_snap(vec![]);
    let witness_hash = witness_snap.dir_hash();
    enter_survival_mode(&mut e, pid, witness_snap);
    let state = active_pre_fire_burst(
        &mut e,
        pid,
        PreFirePhase::Verifying(ProbeSlot::empty()),
        BurstIntent::Seed,
        anchor,
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
        QuiescenceVerdict::Authoritative { forced: false },
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
    // Sub fired pre-loss; anchor lost; recovery Seed-Ok with drift.
    // The eager consume on the recovery-drift fire (the
    // `EmitMode::SeedDrift` seal in `fire_and_settle`, before
    // transition_to_awaiting) keeps the baseline ⊕ witness exclusivity
    // holding at every step boundary, not just at later consume sites.
    let (mut e, pid, sid, anchor, now) = engine_with_attached_sub();
    // Drain the attach-time Seed-Verifying probe before the manual
    // `transition_state` clobber below drops that armed state.
    let _ = e.cancel_all_in_flight_probes();

    // Survival-mode drift setup: a witness snapshot whose hash won't
    // match the post-graft current ⇒ the drift signal triggers;
    // pre-loss fire history on `sid` narrows the SeedDrift filter to
    // this Sub. `mark_fired` is idempotent — "sid fired pre-loss"
    // needs exactly one mark.
    let witness_snap = dir_tree_snap(vec![("pre", EntryKind::File, 1)]);
    let witness_hash = witness_snap.dir_hash();
    enter_survival_mode(&mut e, pid, witness_snap);
    let state = active_pre_fire_burst(
        &mut e,
        pid,
        PreFirePhase::Verifying(ProbeSlot::empty()),
        BurstIntent::Seed,
        anchor,
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
        QuiescenceVerdict::Authoritative { forced: false },
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

/// Attach a `PerStableFile` Sub at the same `(anchor, config_hash)` as
/// `engine_with_attached_sub`'s `SubtreeRoot` Sub so both share one
/// Profile. Identical `ProfileIdentity` (config / max_settle / events)
/// ⇒ same `ProfileId`; the scope differs, which is the only axis the
/// `has_per_stable_file_sub` gate reads.
fn attach_per_stable_file_sibling(
    e: &mut Engine,
    anchor: ResourceId,
    pid: specter_core::ProfileId,
    now: Instant,
) -> specter_core::SubId {
    let req = SubAttachRequest {
        anchor: SubAttachAnchor::Resource(anchor),
        identity: ProfileIdentity {
            config: ScanConfig::builder().recursive(true).build(),
            max_settle: MAX_SETTLE,
            events: NO_EVENTS,
        },
        params: SubParams {
            name: "per-file-sibling".into(),
            program: empty_program(),
            scope: EffectScope::PerStableFile,
            settle: SETTLE,
            log_output: false,
            source_promoter: None,
        },
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

/// (a) Positive: anchor loss → real content drift across the loss
/// window → recovery Seed-Ok with a `PerStableFile` Sub attached ⇒
/// exactly one `PerFileDriftDroppedOnRecovery` for the Profile. No
/// fired Sub is required — the diagnostic is scope+drift gated, not
/// drift-branch gated (a PerFile-only Profile never records a fire yet
/// is exactly the case to flag).
#[test]
fn per_file_drift_dropped_on_recovery_emits_once_on_real_drift() {
    let (mut e, pid, _sid, anchor, now) = engine_with_attached_sub();
    let _ = e.cancel_all_in_flight_probes();
    attach_per_stable_file_sibling(&mut e, anchor, pid, now);
    let _ = e.cancel_all_in_flight_probes();

    // Survival mode: pre-loss hash retained as the witness; the
    // recovery probe lands a `current` whose hash differs ⇒ real drift.
    let witness_snap = dir_tree_snap(vec![("pre", EntryKind::File, 1)]);
    let witness_hash = witness_snap.dir_hash();
    enter_survival_mode(&mut e, pid, witness_snap);
    let state = active_pre_fire_burst(
        &mut e,
        pid,
        PreFirePhase::Verifying(ProbeSlot::empty()),
        BurstIntent::Seed,
        anchor,
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
        QuiescenceVerdict::Authoritative { forced: false },
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

/// (b) Byte-identical recovery: same loss→recovery shape, but the
/// recovered tree hash equals the pre-loss witness ⇒ zero
/// `PerFileDriftDroppedOnRecovery` (a byte-identical recovery dropped
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
        PreFirePhase::Verifying(ProbeSlot::empty()),
        BurstIntent::Seed,
        anchor,
        now,
    );
    if let Some(p) = e.profiles.get_mut(pid) {
        p.transition_state(state);
    }

    // Same shape ⇒ identical dir_hash (synthetic ctors are
    // deterministic): a byte-identical recovery.
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
        QuiescenceVerdict::Authoritative { forced: false },
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

/// (c) Scope gate: same loss→drift→recovery, but the Profile has only
/// a `SubtreeRoot` Sub (no `PerStableFile`) ⇒ zero
/// `PerFileDriftDroppedOnRecovery`. Regression guard against collapsing
/// the scope scan into `Profile::has_per_file_fds` — that predicate is
/// events-mask derived and would false-positive a content-watching
/// Subtree-only Profile.
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
        PreFirePhase::Verifying(ProbeSlot::empty()),
        BurstIntent::Seed,
        anchor,
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
        QuiescenceVerdict::Authoritative { forced: false },
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

/// Fresh Profile reports no drift: `fired_subs` is empty by
/// construction and there is no settled reference yet. Pins the
/// "fresh Seed never fires Effect" contract — without the
/// `fired_subs.is_empty()` short-circuit, a Profile with no settled
/// reference AND a Some current would still fall through to the
/// `match p.settled_hash()` arm; the short-circuit is the
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

/// Survival-mode drift: anchor loss collapsed the anchor to
/// `Unclassified`, retaining the pre-loss baseline hash as the
/// survival witness. The recovery Seed-Ok lands a new `current` whose
/// hash differs from that witness — drift detected, conservative
/// re-fire required. Pins the witness arm of `settled_hash()`.
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

    // Survival mode: anchor loss stashed the pre-loss baseline.hash()
    // into the witness; the recovery probe lands a `current` whose hash
    // differs from that witness ⇒ drift.
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

/// Active-mode drift: `baseline().is_some()` (no anchor loss has
/// occurred), so `settled_hash()` returns the live baseline's hash —
/// a separate survival witness alongside a held baseline is not
/// representable in the anchor sum. Drift derives from
/// `baseline.hash() != current.hash()`. Covers the
/// `on_sensor_overflow` reseed path: overflow does not go through
/// `discard_anchor_state`, so the baseline (hence the settled
/// reference) persists and the single `settled_hash()` oracle still
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

/// Active-mode no-drift: `baseline.hash() == current.hash()` —
/// overflow happened to coincide with no actual disk change. The
/// witness is `None`, the baseline arm runs and reports no drift.
/// No conservative re-fire — `baseline` still represents reality.
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

/// Pins the three load-bearing properties of relocating the Effect
/// fire-history from a per-Profile container to a per-Sub `bool`:
///
///  1. **B1 / SeedDrift read `Sub.has_fired`.** A real burst that
///     fires sets the emitting Sub's flag; `SubRegistry::any_fired` /
///     `fired_in` (the B1-suppress and SeedDrift-filter bases) observe
///     it through the registry, not through any Profile container.
///  2. **A detached Sub's flag dies with its slotmap entry — no purge
///     needed.** After the Sub fired, detach it (a sibling Sub keeps
///     the Profile alive) and drive a survival-mode *drift* Seed-Ok.
///     It must NOT re-fire: `fired_in(pid)` is empty because the
///     detached Sub's `has_fired` died with `subs.remove`, and the
///     surviving sibling never fired. There is no per-Profile fire
///     container to purge, and none is touched.
///  3. **A fresh attach starts unfired.** The sibling attached after
///     the original has `has_fired == false`.
#[test]
fn fire_history_is_per_sub_detach_drops_it_no_purge() {
    let (mut e, pid, sid_a, anchor, now) = engine_with_attached_sub();
    // Drain the attach-time Seed-Verifying probe before later manual
    // state manipulation.
    let _ = e.cancel_all_in_flight_probes();

    // Property 3 (part a): the freshly attached Sub starts unfired.
    assert!(
        !e.subs.get(sid_a).unwrap().has_fired,
        "fresh attach: sid_a starts unfired",
    );
    assert!(
        !e.subs.any_fired(pid),
        "fresh Profile: no Sub has fired (B1/SeedDrift basis is empty)",
    );

    // Attach a second Sub at the *same* (anchor, config_hash) so it
    // shares this Profile — it keeps the Profile alive across sid_a's
    // detach below. Identical request shape ⇒ same ProfileId.
    let req_b = SubAttachRequest {
        anchor: SubAttachAnchor::Resource(anchor),
        identity: ProfileIdentity {
            config: ScanConfig::builder().recursive(true).build(),
            max_settle: MAX_SETTLE,
            events: NO_EVENTS,
        },
        params: SubParams {
            name: "sibling".into(),
            program: empty_program(),
            scope: EffectScope::SubtreeRoot,
            settle: SETTLE,
            log_output: false,
            source_promoter: None,
        },
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
        !e.subs.get(sid_b).unwrap().has_fired,
        "fresh attach: sibling starts unfired",
    );
    let _ = e.cancel_all_in_flight_probes();

    // Property 1: mark sid_a fired (the post-emit B1 bookkeeping the
    // SubtreeRoot emit arm performs) and observe it through the
    // registry — the exact signal `seed_drift_observed` / B1 read.
    e.subs.mark_fired(sid_a);
    assert!(
        e.subs.get(sid_a).unwrap().has_fired,
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

    // Set up a survival-mode *drift* scenario: the survival witness
    // carries a pre-loss hash; the post-recovery `current` differs
    // from it ⇒ `seed_drift_observed` is true *while sid_a is fired*.
    // (The witness must differ from `current` for drift; matches the
    // working `seed_drift_observed_returns_true_on_post_recovery_drift`
    // setup.)
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

    // Property 2: detach sid_a. The Profile survives (sibling sid_b
    // remains). sid_a's `has_fired` died with its slotmap entry — no
    // per-Profile purge exists or runs.
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

    // A recovery Seed-Ok now must NOT re-fire: the tree still differs
    // from the witness, but `any_fired` is false post-detach (sid_a's
    // flag died with it; sid_b never fired), so `classify_consequence`
    // yields `SilentPin` — seal-and-finish, no fire. This is the
    // behavioural proof that a detached Sub cannot be re-fired and that
    // no purge is required to achieve it.
    let regrafted = dir_tree_snap(vec![]);
    let mut out = StepOutput::default();
    e.dispatch_quiescence_ok(
        pid,
        TreeSnapshot::Dir(regrafted),
        QuiescenceVerdict::Authoritative { forced: false },
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

// ---------- Property tests ----------

mod props {
    use super::*;
    use proptest::prelude::*;

    /// Each property generates a sequence of opaque "actions" for the
    /// engine to dispatch, then asserts a global invariant. We don't
    /// generate random `ResourceIds` — the resource always exists from a
    /// fresh attach.
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
            Just(FsEvent::Modified),
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

    /// Apply `action` to a freshly-attached single-Profile engine; collect
    /// the `StepOutput` from each step. Returns the latest correlation seen
    /// so the next `Probe` action can target it.
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
                        owner: ProbeOwner::Profile(pid),
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
                        owner: ProbeOwner::Profile(pid),
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
                        owner: ProbeOwner::Profile(pid),
                        correlation: corr,
                        outcome: ProbeOutcome::Failed { errno },
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
            identity: ProfileIdentity {
                config: ScanConfig::builder().recursive(true).build(),
                max_settle: MAX_SETTLE,
                events: NO_EVENTS,
            },
            params: SubParams {
                name: "test".into(),
                program: empty_program(),
                scope: EffectScope::SubtreeRoot,
                settle: SETTLE,
                log_output: false,
                source_promoter: None,
            },
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

        /// Every StepOutput is sorted canonically. Run a random sequence of
        /// inputs and verify after each step.
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

                // probe_ops sorted by ProbeOwner.
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

        /// I5: at most one outstanding ProbeRequest per Profile. Track
        /// outstanding probes via emit/cancel/respond; assert ≤ 1.
        #[test]
        fn prop_at_most_one_outstanding_probe(
            actions in prop::collection::vec(arb_action(), 0..16),
        ) {
            let (mut e, sid, r, mut t, mut last_correlation) =
                fresh_engine_with_sub();
            let pid = e.subs.get(sid).unwrap().profile();

            // attach_sub starts a Batching-first Seed burst — no probe
            // until its settle expires, so outstanding = 0.
            let mut outstanding: u32 = 0;

            for action in actions {
                let was_probe = matches!(action, Action::Probe | Action::ProbeVanished | Action::ProbeFailed(_));
                let out = run_action(&mut e, sid, r, action, &mut t, &mut last_correlation);

                // Each Probe op increments; each Cancel and each accepted
                // ProbeResponse decrements. (We treat "any Probe op
                // emitted" as +1 and "any Cancel emitted" as -1; the test
                // doesn't care about the difference, only that the running
                // count stays ≤ 1.)
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

                // If a ProbeResponse action was injected and didn't cause
                // a stale-diagnostic, the outstanding probe is consumed.
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

                // Slot-discipline I5: at most one outstanding probe per
                // Profile, expressed as a single state-resident probe
                // slot reachable via `ProbeOwner::Profile(pid)`.
                // `pending_probe_for` returns `Option<ProbeCorrelation>`
                // so `<= 1` is trivially true; the assertion is a
                // regression guard against a future widening of the
                // per-owner slot shape.
                let probing_count =
                    u32::from(e.pending_probe_for(ProbeOwner::Profile(pid)).is_some());
                prop_assert!(
                    probing_count <= 1,
                    "I5 representability: a Profile's single state-resident ProbeSlot carries at most one in-flight probe",
                );
            }
            let _ = e.cancel_all_in_flight_probes();
        }

        /// `prop_seed_burst_without_activity_emits_no_effects`: from a
        /// fresh attach with **no FsEvents witnessed**, the Seed-burst's
        /// eventual ProbeResponse path never produces an Effect. This is
        /// strictly the no-activity path: with no events injected and
        /// `dirty` empty, the verdict routes to `SilentPin` and the
        /// burst finishes without emission. It does NOT assert anything
        /// about a fresh Seed that *witnessed* activity — that case
        /// fires and is covered by the `fresh_seed_fires::*` tests.
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
            // Batching-first Seed: expire the settle timer so the verify
            // probe is in flight, then answer it with the random outcome.
            seed_settle_to_verifying(&e, now);
            let corr = e
                .pending_probe_for(ProbeOwner::Profile(pid))
                .expect("seed verify probe in flight after settle expiry");
            let outcome = match seed_outcome {
                0 => ProbeOutcome::SubtreeProven { snapshot: dir_tree_snap(vec![]), authority: ProofAuthority::Authoritative },
                1 => ProbeOutcome::Vanished,
                _ => ProbeOutcome::Failed { errno: 13 },
            };
            let out = e.step(
                Input::ProbeResponse(ProbeResponse { owner: ProbeOwner::Profile(pid),
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

        /// `prop_single_profile_never_has_active_standard_descendant` —
        /// the derived replacement for the deleted `dirty_descendants`
        /// I4 floor. A single-Profile engine has no covered descendant,
        /// so the fresh reconfirm query that now gates `gated_fire`'s
        /// fire (`coverage::has_active_standard_descendant`) must stay
        /// false after *any* input sequence. This also pins the query's
        /// self-exclusion: the lone Profile is the ancestor under test
        /// and must never count itself, through every burst phase the
        /// random actions drive it into.
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
                    ));
                }
            }
            let _ = e.cancel_all_in_flight_probes();
        }

        /// `prop_step_is_total` — for any input sequence on a fresh engine,
        /// no panic in release. Implicit by reaching this assertion. Keep
        /// as a smoke test for the random-input fuzzer.
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
