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

use crate::{Engine, SubAttachRequest};
use compact_str::CompactString;
use specter_core::{
    ArgPart, ArgTemplate, BurstIntent, BurstPhase, ChildEntry, ClaimKind, ClassSet,
    CommandTemplate, DedupKey, Diagnostic, DirChild, DirMeta, DirSnapshot, EffectOutcome,
    EffectScope, EntryKind, FsEvent, Input, LeafEntry, Placeholder, ProbeKind, ProbeOp,
    ProbeResponse, ProbeResult, ProfileState, ResourceId, ResourceKind, ResourceRole, ScanConfig,
    StepOutput, SubId, TimerKind, TreeSnapshot, WatchOp,
};
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const SETTLE: Duration = Duration::from_millis(100);
const MAX_SETTLE: Duration = Duration::from_secs(6);
/// Default events mask for transition tests. Empty mask gives a Profile
/// with `has_per_file_fds = false`, matching the prior tests' assumption
/// that per-file FDs are out of scope unless the test specifically opts
/// into PerStableFile + a CONTENT/METADATA mask. Per design D3, events
/// fold into `config_hash` so two test fixtures differing only on `events`
/// fork separate Profiles (intentional partition).
const NO_EVENTS: ClassSet = ClassSet::EMPTY;

fn empty_command() -> CommandTemplate {
    CommandTemplate::new([ArgTemplate::new([ArgPart::literal("/bin/true")])])
}

fn diff_command() -> CommandTemplate {
    CommandTemplate::new([ArgTemplate::new([
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
    let r = e.tree.ensure(None, "anchor", ResourceRole::User);
    e.tree.get_mut(r).unwrap().kind = ResourceKind::Dir;
    let now = Instant::now();
    let req = SubAttachRequest {
        name: String::from("test-sub"),
        resource: r,
        path: None,
        config: ScanConfig::builder().recursive(true).build(),
        max_settle: MAX_SETTLE,
        settle: SETTLE,
        command: empty_command(),
        scope: EffectScope::SubtreeRoot,
        events: NO_EVENTS,
        log_output: false,
    };
    let (sid, _out) = e.attach_sub(req, now);
    let pid = e.subs.get(sid).unwrap().profile;
    (e, pid, sid, r, now)
}

/// V5-native test helper: build a `TreeSnapshot::Dir` with the supplied
/// single-component children. Each child is `(name, EntryKind, inode)`;
/// Dirs are emitted with `subtree: None` (uncovered). Tests that need
/// nested subtrees should use `dir_with_subtree`.
fn dir_tree_snap(root: ResourceId, children: Vec<(&str, EntryKind, u64)>) -> TreeSnapshot {
    let mut map: BTreeMap<CompactString, ChildEntry> = BTreeMap::new();
    for (name, kind, inode) in children {
        let child = match kind {
            EntryKind::Dir => ChildEntry::Dir(DirChild {
                inode,
                device: 0,
                subtree: None,
            }),
            _ => ChildEntry::Leaf(LeafEntry::new(kind, 0, UNIX_EPOCH, inode, 0)),
        };
        map.insert(CompactString::new(name), child);
    }
    TreeSnapshot::Dir(Arc::new(DirSnapshot::new(
        root,
        DirMeta {
            mtime: UNIX_EPOCH,
            inode: 0,
            device: 0,
        },
        0,
        map,
    )))
}

/// `LeafEntry`-only `TreeSnapshot::File` for File-anchored Profiles.
#[allow(dead_code)]
fn file_tree_snap(kind: EntryKind, size: u64, mtime: SystemTime, inode: u64) -> TreeSnapshot {
    TreeSnapshot::File(LeafEntry::new(kind, size, mtime, inode, 0))
}

/// Drive the Profile from fresh-attach through Seed-Ok → Idle (post-Seed
/// state). Returns the response correlation. After this, Profile.current
/// and Profile.baseline are set.
fn complete_seed_burst(e: &mut Engine, pid: specter_core::ProfileId, root: ResourceId) {
    let correlation = e.pending_probe(pid).expect("Verifying probe in flight");
    let snap = dir_tree_snap(root, vec![]);
    e.step(
        Input::ProbeResponse(ProbeResponse {
            profile: pid,
            correlation,
            result: ProbeResult::Ok(snap),
        }),
        Instant::now(),
    );
    assert!(matches!(
        e.profiles.get(pid).unwrap().state,
        ProfileState::Idle,
    ));
}

// ---- attach_sub ----

#[test]
fn attach_sub_fresh_profile_emits_watch_suppress_probe() {
    let (e, _pid, _sid, r, _now) = engine_with_attached_sub();
    // After attach: anchor watch_demand=1, suppress_count=1, Profile is
    // Active(Seed Probing).
    assert_eq!(e.tree.get(r).unwrap().watch_demand, 1);
    assert_eq!(e.tree.get(r).unwrap().suppress_count, 1);
}

#[test]
fn attach_sub_existing_profile_bumps_refcount() {
    let (mut e, pid, _sid, r, now) = engine_with_attached_sub();
    let pre_state = matches!(e.profiles.get(pid).unwrap().state, ProfileState::Active(_));
    assert!(pre_state, "first attach went Active");
    let pre_refcount = e.profiles.get(pid).unwrap().sub_refcount;

    // Second attach with the same config_hash.
    let req = SubAttachRequest {
        name: String::from("second"),
        resource: r,
        path: None,
        config: ScanConfig::builder().recursive(true).build(),
        max_settle: MAX_SETTLE,
        settle: SETTLE,
        command: empty_command(),
        scope: EffectScope::SubtreeRoot,
        events: NO_EVENTS,
        log_output: false,
    };
    let (sid2, out) = e.attach_sub(req, now);
    assert_eq!(e.profiles.get(pid).unwrap().sub_refcount, pre_refcount + 1);
    assert_eq!(e.subs.get(sid2).unwrap().profile, pid, "shared Profile");
    // No fresh watch/probe/suppress emitted: existing Profile already has
    // them.
    assert!(out.watch_ops.is_empty());
    assert!(out.probe_ops.is_empty());
}

// ---- ProbeResponse dispatch ----

/// Smoke test: a `TreeSnapshot::Dir(...)` with one Leaf entry
/// lands as a Seed-Ok on the Profile (no Effect, baseline set). Pins
/// the dispatch wiring; the rest of the engine test suite is the broad
/// coverage.
#[test]
fn engine_dispatch_through_shim_matches_v4_behaviour() {
    let (mut e, pid, _sid, root, _now) = engine_with_attached_sub();
    let correlation = e.pending_probe(pid).expect("Verifying probe in flight");
    // One-Leaf TreeSnapshot — the shim flattens to one V4 Entry.
    let snap = dir_tree_snap(root, vec![("main.rs", EntryKind::File, 100)]);
    let out = e.step(
        Input::ProbeResponse(ProbeResponse {
            profile: pid,
            correlation,
            result: ProbeResult::Ok(snap),
        }),
        Instant::now(),
    );
    let p = e.profiles.get(pid).unwrap();
    assert!(matches!(p.state, ProfileState::Idle));
    assert!(p.current.is_some(), "Seed-Ok sets current via shim");
    assert!(out.effects.is_empty(), "Seed bursts never fire Effects");
}

#[test]
fn probe_response_seed_ok_sets_baseline_and_idles_no_effect() {
    let (mut e, pid, _sid, root, _now) = engine_with_attached_sub();
    let correlation = e.pending_probe(pid).expect("Verifying probe in flight");
    let snap = dir_tree_snap(root, vec![]);
    let out = e.step(
        Input::ProbeResponse(ProbeResponse {
            profile: pid,
            correlation,
            result: ProbeResult::Ok(snap),
        }),
        Instant::now(),
    );

    let p = e.profiles.get(pid).unwrap();
    assert!(matches!(p.state, ProfileState::Idle));
    assert!(p.baseline.is_some());
    assert!(p.current.is_some());
    assert!(out.effects.is_empty(), "Seed bursts never fire Effects");
    // Anchor unsuppressed.
    let unsuppress = out
        .watch_ops
        .iter()
        .any(|op| matches!(op, WatchOp::Unsuppress { .. }));
    assert!(unsuppress);
}

#[test]
fn probe_response_seed_vanished_clears_baseline_and_diagnoses() {
    let (mut e, pid, _sid, _r, _now) = engine_with_attached_sub();
    let correlation = e.pending_probe(pid).expect("Verifying probe in flight");
    let out = e.step(
        Input::ProbeResponse(ProbeResponse {
            profile: pid,
            correlation,
            result: ProbeResult::Vanished,
        }),
        Instant::now(),
    );
    let p = e.profiles.get(pid).unwrap();
    assert!(matches!(p.state, ProfileState::Idle));
    assert!(p.baseline.is_none());
    assert!(p.current.is_none());
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
    let (mut e, pid, _sid, _r, _now) = engine_with_attached_sub();
    let correlation = e.pending_probe(pid).expect("Verifying probe in flight");
    let out = e.step(
        Input::ProbeResponse(ProbeResponse {
            profile: pid,
            correlation,
            result: ProbeResult::Failed { errno: 13 },
        }),
        Instant::now(),
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
    let (mut e, pid, _sid, root, _now) = engine_with_attached_sub();
    // Inject a response with the wrong correlation.
    let bogus = specter_core::ProbeCorrelation(99_999);
    let snap = dir_tree_snap(root, vec![]);
    let out = e.step(
        Input::ProbeResponse(ProbeResponse {
            profile: pid,
            correlation: bogus,
            result: ProbeResult::Ok(snap),
        }),
        Instant::now(),
    );
    let stale = out
        .diagnostics
        .iter()
        .any(|d| matches!(d, Diagnostic::StaleProbeResponse { .. }));
    assert!(stale);
    // State unchanged: still Active(Seed Probing).
    assert!(matches!(
        e.profiles.get(pid).unwrap().state,
        ProfileState::Active(_),
    ));
}

#[test]
fn probe_response_for_idle_profile_drops_with_diagnostic() {
    let (mut e, pid, _sid, root, _now) = engine_with_attached_sub();
    complete_seed_burst(&mut e, pid, root);
    // Profile is Idle; injecting a ProbeResponse drops with diagnostic.
    let snap = dir_tree_snap(root, vec![]);
    let out = e.step(
        Input::ProbeResponse(ProbeResponse {
            profile: pid,
            correlation: specter_core::ProbeCorrelation(1),
            result: ProbeResult::Ok(snap),
        }),
        Instant::now(),
    );
    let stale = out
        .diagnostics
        .iter()
        .any(|d| matches!(d, Diagnostic::StaleProbeResponse { .. }));
    assert!(stale);
}

// ---- Standard burst dispatch ----

#[test]
fn standard_burst_stable_emits_effect_and_awaits() {
    // Stable verdict emits the Effect and transitions to
    // `BurstPhase::Awaiting`; the engine waits for the completion before
    // returning to Idle. Idle means "nothing in flight" — outstanding
    // Effects keep the burst Active until they report back.
    let (mut e, pid, _sid, root, _now) = engine_with_attached_sub();
    complete_seed_burst(&mut e, pid, root);
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
    // We're in Probing; pick up the correlation.
    let correlation = e.pending_probe(pid).expect("Verifying probe in flight");
    // Reproduce the seed-burst's empty snapshot — both shim to the same
    // V4 content_hash (entries.len() == 0), so `stable_against` holds.
    let snap = dir_tree_snap(root, vec![]);
    let out = e.step(
        Input::ProbeResponse(ProbeResponse {
            profile: pid,
            correlation,
            result: ProbeResult::Ok(snap),
        }),
        now + SETTLE + Duration::from_millis(1),
    );
    let burst = match &e.profiles.get(pid).unwrap().state {
        ProfileState::Active(b) => b,
        _ => panic!("expected Active(Awaiting) after firing"),
    };
    assert_eq!(burst.intent, BurstIntent::Standard);
    assert!(
        matches!(burst.phase, BurstPhase::Awaiting { outstanding: 1, .. }),
        "stable verdict transitions to Awaiting with one outstanding Effect; got {:?}",
        burst.phase,
    );
    assert_eq!(out.effects.len(), 1, "one Effect emitted at stable verdict");
    assert!(!out.effects[0].forced);
    // Resolver populated the command + env (P8: was placeholders before).
    assert_eq!(out.effects[0].command.argv, vec!["/bin/true".to_string()]);
    let env = &out.effects[0].env;
    assert!(env.iter().any(|(k, _)| k == "SPECTER_PATH"));
    assert!(env.iter().any(|(k, _)| k == "SPECTER_SUB"));
    assert!(env.iter().any(|(k, v)| k == "SPECTER_FORCED" && v == "0"));
    assert!(
        env.iter()
            .any(|(k, v)| k == "SPECTER_EVENT_KIND" && v == "dir-subtree")
    );
    // SPECTER_DIFF_PATH is set by the actuator at spawn time, not the engine.
    assert!(env.iter().all(|(k, _)| k != "SPECTER_DIFF_PATH"));
    // cwd is the anchor (Dir Profile).
    assert_eq!(out.effects[0].cwd.as_os_str(), "anchor");
}

#[test]
fn emit_effects_subtree_root_uses_parent_dir_for_file_profile() {
    // SubtreeRoot Sub anchored at a File-kind Profile: cwd should be the
    // parent dir, not the file itself (Command::current_dir requires a
    // directory).
    let mut e = Engine::new();
    let parent = e.tree.ensure(None, "parentdir", ResourceRole::User);
    e.tree.get_mut(parent).unwrap().kind = ResourceKind::Dir;
    let file_anchor = e.tree.ensure(Some(parent), "main.rs", ResourceRole::User);
    e.tree.get_mut(file_anchor).unwrap().kind = ResourceKind::File;
    let now = Instant::now();
    let req = SubAttachRequest {
        name: "build".into(),
        resource: file_anchor,
        path: None,
        config: ScanConfig::builder().recursive(false).build(),
        max_settle: MAX_SETTLE,
        settle: SETTLE,
        command: empty_command(),
        scope: EffectScope::SubtreeRoot,
        events: NO_EVENTS,
        log_output: false,
    };
    let (sid, _) = e.attach_sub(req, now);
    let pid = e.subs.get(sid).unwrap().profile;
    // Seed → Idle.
    let seed_corr = e.pending_probe(pid).expect("Verifying probe in flight");
    let snap = file_tree_snap(EntryKind::File, 0, std::time::UNIX_EPOCH, 1);
    e.step(
        Input::ProbeResponse(ProbeResponse {
            profile: pid,
            correlation: seed_corr,
            result: ProbeResult::Ok(snap.clone()),
        }),
        now,
    );
    // Standard burst with the same snap (stable).
    let t1 = now + Duration::from_millis(10);
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
    let std_corr = e.pending_probe(pid).expect("Verifying probe in flight");
    let out = e.step(
        Input::ProbeResponse(ProbeResponse {
            profile: pid,
            correlation: std_corr,
            result: ProbeResult::Ok(snap),
        }),
        t2,
    );
    assert_eq!(out.effects.len(), 1);
    // Anchor path: parentdir/main.rs. cwd should be parentdir.
    assert_eq!(
        out.effects[0].cwd.as_os_str(),
        "parentdir",
        "File-kind anchor uses parent dir as cwd",
    );
    // SPECTER_PATH is the file itself; SPECTER_ANCHOR is the file too.
    let env = &out.effects[0].env;
    assert_eq!(
        env.iter().find(|(k, _)| k == "SPECTER_PATH").unwrap().1,
        "parentdir/main.rs",
    );
    assert_eq!(
        env.iter().find(|(k, _)| k == "SPECTER_ANCHOR").unwrap().1,
        "parentdir/main.rs",
    );
}

#[test]
fn standard_burst_force_fires_on_max_settle() {
    let (mut e, pid, _sid, root, _now) = engine_with_attached_sub();
    complete_seed_burst(&mut e, pid, root);
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
    // After force-fire, we're either in Probing (forced=true) or already
    // Awaiting if the deadline race resolved both timers. Drive the
    // response back if needed.
    if let Some(correlation) = e.pending_probe(pid) {
        // Inject a not-stable response to test the forced effect emission.
        let snap = dir_tree_snap(root, vec![("new.rs", EntryKind::File, 99)]);
        let out = e.step(
            Input::ProbeResponse(ProbeResponse {
                profile: pid,
                correlation,
                result: ProbeResult::Ok(snap),
            }),
            deadline,
        );
        // Forced fire transitions to Awaiting (Effect in flight). The
        // post-fire rebase happens when the eventual EffectComplete
        // drives the Awaiting → Rebasing transition.
        let phase = match &e.profiles.get(pid).unwrap().state {
            ProfileState::Active(burst) => &burst.phase,
            _ => panic!("expected Active(Awaiting)"),
        };
        assert!(
            matches!(phase, BurstPhase::Awaiting { outstanding: 1, .. }),
            "force-fired stable verdict transitions to Awaiting; got {phase:?}",
        );
        assert_eq!(out.effects.len(), 1);
        assert!(
            out.effects[0].forced,
            "force-fired Effect must carry forced=true",
        );
    }
}

#[test]
fn fs_event_modified_during_seed_probing_preserves_intent() {
    let (mut e, pid, _sid, root, _now) = engine_with_attached_sub();
    // Profile is in Active(Seed Probing) right after attach. Inject an
    // FsEvent — should transition to Active(Seed Settling), emit Cancel.
    let out = e.step(
        Input::FsEvent {
            resource: root,
            event: FsEvent::Modified,
        },
        Instant::now(),
    );
    let burst = match &e.profiles.get(pid).unwrap().state {
        ProfileState::Active(b) => b,
        _ => panic!(),
    };
    assert_eq!(
        burst.intent,
        BurstIntent::Seed,
        "intent preserved across Verifying → Batching",
    );
    assert!(matches!(burst.phase, BurstPhase::Batching { .. }));
    let cancels = out
        .probe_ops
        .iter()
        .filter(|op| matches!(op, ProbeOp::Cancel { .. }))
        .count();
    assert_eq!(cancels, 1);
}

/// Field-discipline pin for `event_drives_batching`: an FsEvent during
/// Verifying closes the probe channel atomically with the Cancel emission.
/// Pre-refactor the close was implicit in the `BurstPhase::Verifying { ... }
/// → Batching { ... }` variant rewrite; post-refactor it must clear the
/// per-Profile `pending_probe` slot explicitly.
#[test]
fn event_drives_batching_clears_pending_probe() {
    let (mut e, pid, _sid, root, _now) = engine_with_attached_sub();
    assert!(
        e.pending_probe(pid).is_some(),
        "Seed probe in flight after attach",
    );

    let _ = e.step(
        Input::FsEvent {
            resource: root,
            event: FsEvent::Modified,
        },
        Instant::now(),
    );

    assert!(
        e.pending_probe(pid).is_none(),
        "channel closed atomically with Verifying → Batching transition",
    );
}

/// Field-discipline pin for `finalize_anchor_lost`: an anchor terminal
/// event during Verifying cancels the in-flight probe and clears the
/// channel. Replaces the pre-refactor `was_verifying` snapshot's role.
#[test]
fn finalize_anchor_lost_during_verifying_clears_pending_probe() {
    let (mut e, pid, _sid, root, _now) = engine_with_attached_sub();
    assert!(
        e.pending_probe(pid).is_some(),
        "Seed probe in flight after attach",
    );

    let out = e.step(
        Input::FsEvent {
            resource: root,
            event: FsEvent::Removed,
        },
        Instant::now(),
    );

    assert!(
        e.pending_probe(pid).is_none(),
        "anchor terminal during Verifying closes the channel",
    );
    let cancels = out
        .probe_ops
        .iter()
        .filter(|op| matches!(op, ProbeOp::Cancel { profile } if *profile == pid))
        .count();
    assert_eq!(
        cancels, 1,
        "exactly one Cancel emitted; got {:?}",
        out.probe_ops
    );
}

/// Single-diagnostic guarantee for stale `ProbeResponse`. Pre-refactor
/// the dispatch had two stale-detection layers (state-shape mismatch and
/// inner-correlation mismatch) that could both fire on degenerate inputs.
/// Post-refactor the top-level `pending_probe == Some(received)` check is
/// the sole gate — exactly one diagnostic per stale response.
#[test]
fn stale_probe_response_emits_exactly_one_diagnostic() {
    let (mut e, pid, _sid, root, _now) = engine_with_attached_sub();
    let bogus = specter_core::ProbeCorrelation(99_999);
    let snap = dir_tree_snap(root, vec![]);

    let out = e.step(
        Input::ProbeResponse(ProbeResponse {
            profile: pid,
            correlation: bogus,
            result: ProbeResult::Ok(snap),
        }),
        Instant::now(),
    );

    let stale_count = out
        .diagnostics
        .iter()
        .filter(|d| matches!(d, Diagnostic::StaleProbeResponse { profile, .. } if *profile == pid))
        .count();
    assert_eq!(
        stale_count, 1,
        "exactly one StaleProbeResponse diagnostic; got {:?}",
        out.diagnostics,
    );
    // Live channel untouched: the legitimate Seed probe is still in flight.
    assert!(
        e.pending_probe(pid).is_some(),
        "live channel untouched by stale response",
    );
}

/// D8 — anchor events bypass the L5 class filter unconditionally.
/// Profile has events = EMPTY (nothing in the mask); a `MetadataChanged`
/// at the anchor still drives the lifecycle path (burst start), and no
/// `EventClassDropped` is emitted. This guards the lifecycle-continuity
/// invariant: anchor events never get filtered out by user mask choice.
#[test]
fn fs_event_metadatachanged_at_anchor_bypasses_class_filter() {
    let (mut e, pid, _sid, root, _now) = engine_with_attached_sub();
    complete_seed_burst(&mut e, pid, root);
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
        "anchor events bypass the L5 class filter (D8)",
    );
    assert!(
        matches!(e.profiles.get(pid).unwrap().state, ProfileState::Active(_),),
        "MetadataChanged at the anchor drives a burst even on EMPTY mask",
    );
}

/// §6.1 — descendant events whose class is not in the covering Profile's
/// `events_union` drop with `EventClassDropped` BEFORE driving the burst.
/// Profile has events = EMPTY ⇒ `intersects(any_class) == false`, so a
/// `MetadataChanged` on a covered descendant drops cleanly without state
/// mutation.
#[test]
fn fs_event_metadatachanged_at_descendant_drops_with_event_class_dropped() {
    let (mut e, pid, _sid, root, _now) = engine_with_attached_sub();
    complete_seed_burst(&mut e, pid, root);

    // Materialize a covered descendant. Bump `watch_demand` so the event
    // passes the `EventOnUnwatchedResource` head guard. The Profile's
    // ScanConfig has `recursive(true)` so `covers(profile, child, tree)`
    // is satisfied.
    let child = e.tree.ensure(Some(root), "child.txt", ResourceRole::User);
    e.tree.get_mut(child).unwrap().kind = ResourceKind::File;
    e.tree.get_mut(child).unwrap().watch_demand = 1;

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
        e.profiles.get(pid).unwrap().state,
        ProfileState::Idle,
    ));
}

/// L5 + D7 — identity events on a *descendant File* fold into the CONTENT
/// class. A Profile excluding CONTENT (here: STRUCTURE-only on a Dir
/// anchor) drops the descendant File `Removed` with `EventClassDropped`.
/// The dropped event is not routed through `on_anchor_terminal_event`
/// — that routing is anchor-only.
#[test]
fn fs_event_terminal_on_descendant_file_folds_to_content_and_drops() {
    let mut e = Engine::new();
    let r = e.tree.ensure(None, "anchor", ResourceRole::User);
    e.tree.get_mut(r).unwrap().kind = ResourceKind::Dir;
    let now = Instant::now();
    let req = SubAttachRequest {
        name: String::from("test-sub"),
        resource: r,
        path: None,
        config: ScanConfig::builder().recursive(true).build(),
        max_settle: MAX_SETTLE,
        settle: SETTLE,
        command: empty_command(),
        scope: EffectScope::SubtreeRoot,
        events: ClassSet::STRUCTURE,
        log_output: false,
    };
    let (sid, _out) = e.attach_sub(req, now);
    let pid = e.subs.get(sid).unwrap().profile;
    complete_seed_burst(&mut e, pid, r);

    let child = e.tree.ensure(Some(r), "f.txt", ResourceRole::User);
    e.tree.get_mut(child).unwrap().kind = ResourceKind::File;
    e.tree.get_mut(child).unwrap().watch_demand = 1;

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
        e.profiles.get(pid).unwrap().state,
        ProfileState::Idle,
    ));
    // Sanity: anchor's contribution is intact (we did NOT terminate).
    assert!(e.profiles.get(pid).unwrap().anchor_contribution);
    let _ = sid;
}

/// D8 — terminal events on the anchor route through
/// `on_anchor_terminal_event` regardless of the Profile's `events_union`.
/// Anchor is a Dir, events = EMPTY: the kqexec class for `Removed` on a
/// Dir is STRUCTURE — not in the EMPTY mask — but anchor events bypass
/// the filter. After the call, `anchor_contribution` is cleared and
/// `baseline` / `current` are dropped.
#[test]
fn fs_event_anchor_terminal_bypasses_class_filter() {
    let (mut e, pid, _sid, root, _now) = engine_with_attached_sub();
    complete_seed_burst(&mut e, pid, root);
    assert!(
        e.profiles.get(pid).unwrap().anchor_contribution,
        "anchor_contribution set after attach_sub_inner",
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
        "anchor terminal events bypass the L5 class filter (D8)",
    );

    let p = e.profiles.get(pid).unwrap();
    assert!(
        !p.anchor_contribution,
        "anchor_contribution cleared by on_anchor_terminal_event",
    );
    assert!(p.baseline.is_none());
    assert!(p.current.is_none());
    assert!(matches!(p.state, ProfileState::Idle));
}

#[test]
fn fs_event_for_unwatched_resource_emits_diagnostic() {
    let mut e = Engine::new();
    let r = e.tree.ensure(None, "ghost", ResourceRole::User);
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
    let r = e.tree.ensure(None, "lonely", ResourceRole::User);
    e.tree.get_mut(r).unwrap().kind = ResourceKind::Dir;
    e.tree.get_mut(r).unwrap().watch_demand = 1;

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
    complete_seed_burst(&mut e, pid, root);
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
        e.profiles.get(pid).unwrap().state,
        ProfileState::Active(_),
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
    assert!(matches!(p.state, ProfileState::Idle));
    assert!(p.baseline.is_none());
    assert!(p.current.is_none());
    // watch_demand on anchor → 0; one Unwatch op emitted.
    assert_eq!(e.tree.get(root).unwrap().watch_demand, 0);
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
    complete_seed_burst(&mut e, pid, root);
    assert_eq!(e.tree.get(root).unwrap().watch_demand, 1);
    assert!(e.profiles.get(pid).unwrap().current.is_some());

    let _out = e.step(
        Input::FsEvent {
            resource: root,
            event: FsEvent::Removed,
        },
        Instant::now(),
    );

    // Profile state stays Idle (no Active transition).
    assert!(matches!(
        e.profiles.get(pid).unwrap().state,
        ProfileState::Idle,
    ));
    // watch_demand released; baseline/current cleared.
    assert_eq!(e.tree.get(root).unwrap().watch_demand, 0);
    let p = e.profiles.get(pid).unwrap();
    assert!(p.baseline.is_none());
    assert!(p.current.is_none());
}

// ---- TimerExpired dispatch ----

#[test]
fn timer_expired_settle_in_settling_transitions_to_probing() {
    let (mut e, pid, _sid, root, _now) = engine_with_attached_sub();
    complete_seed_burst(&mut e, pid, root);
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
        p.state,
        ProfileState::Active(_) // Probing
    ));
    let probes = out
        .probe_ops
        .iter()
        .filter(|op| matches!(op, ProbeOp::Probe { .. }))
        .count();
    assert_eq!(probes, 1);
}

#[test]
fn timer_expired_stale_id_emits_diagnostic() {
    let mut e = Engine::new();
    use slotmap::KeyData;
    let bogus = specter_core::TimerId::from(KeyData::from_ffi(99_999));
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
// post-Effect rebase happens in `BurstPhase::Rebasing` as a phase of
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
    let (mut e, pid, sid, root, _now) = engine_with_attached_sub();
    complete_seed_burst(&mut e, pid, root);
    let out = e.step(
        Input::EffectComplete {
            sub: sid,
            key: DedupKey::Subtree {
                sub: sid,
                profile: pid,
            },
            result: EffectOutcome::Ok,
        },
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
    assert!(out.probe_ops.is_empty());
    assert!(matches!(
        e.profiles.get(pid).unwrap().state,
        ProfileState::Idle,
    ));
}

#[test]
fn effect_complete_failed_in_idle_clears_hash_and_diagnoses() {
    // Failed always clears `last_emitted_dir_hash[key]` regardless of
    // phase — a failed Effect leaves no observable state to dedupe
    // against. In Idle the completion is also "late" (the engine isn't
    // tracking it), so it diagnoses with EffectCompleteOutsideAwaiting.
    let (mut e, pid, sid, root, _now) = engine_with_attached_sub();
    complete_seed_burst(&mut e, pid, root);
    let pre_baseline = e.profiles.get(pid).unwrap().baseline.is_some();
    let out = e.step(
        Input::EffectComplete {
            sub: sid,
            key: DedupKey::Subtree {
                sub: sid,
                profile: pid,
            },
            result: EffectOutcome::Failed {
                exit_code: Some(1),
                signal: None,
            },
        },
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
    assert!(out.effects.is_empty());
    assert!(out.probe_ops.is_empty());
    assert_eq!(
        e.profiles.get(pid).unwrap().baseline.is_some(),
        pre_baseline,
        "baseline unchanged on Failed",
    );
}

// ---- Effect needs_diff carries Diff ----

#[test]
fn effect_emission_carries_diff_when_needs_diff() {
    let mut e = Engine::new();
    let r = e.tree.ensure(None, "anchor", ResourceRole::User);
    e.tree.get_mut(r).unwrap().kind = ResourceKind::Dir;
    let now = Instant::now();
    let req = SubAttachRequest {
        name: String::from("fmt"),
        resource: r,
        path: None,
        config: ScanConfig::builder().recursive(true).build(),
        max_settle: MAX_SETTLE,
        settle: SETTLE,
        command: diff_command(), // references $created
        scope: EffectScope::SubtreeRoot,
        events: NO_EVENTS,
        log_output: false,
    };
    let (sid, _out) = e.attach_sub(req, now);
    let pid = e.subs.get(sid).unwrap().profile;
    assert!(e.subs.get(sid).unwrap().needs_diff);

    // Seed burst → baseline = empty snapshot.
    complete_seed_burst(&mut e, pid, r);

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
    let snap_with_entry = dir_tree_snap(r, vec![("new.rs", EntryKind::File, 5)]);

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
        let correlation = e.pending_probe(pid).expect("Verifying probe in flight");
        let out = e.step(
            Input::ProbeResponse(ProbeResponse {
                profile: pid,
                correlation,
                result: ProbeResult::Ok(snap_with_entry.clone()),
            }),
            t,
        );
        if !out.effects.is_empty() {
            effect_out = Some(out);
            break;
        }
    }
    let out = effect_out.expect("burst stabilized and emitted an Effect");
    assert_eq!(out.effects.len(), 1);
    let effect = &out.effects[0];
    assert!(effect.diff.is_some(), "needs_diff Effect carries the Diff");
    let diff = effect.diff.as_ref().unwrap();
    assert_eq!(diff.created.len(), 1);
    assert_eq!(diff.created[0].segment.as_str(), "new.rs");
}

// ---- Descent integration ----

#[test]
fn seed_burst_descendants_watched_via_first_probe() {
    let (mut e, pid, _sid, root, _now) = engine_with_attached_sub();
    let correlation = e.pending_probe(pid).expect("Verifying probe in flight");
    // First-probe response with one File and one Dir descendant.
    // Only the Dir gets a Watch op; the File materializes without an FD
    // contribution.
    let snap = dir_tree_snap(
        root,
        vec![("a.rs", EntryKind::File, 1), ("subdir", EntryKind::Dir, 2)],
    );
    let out = e.step(
        Input::ProbeResponse(ProbeResponse {
            profile: pid,
            correlation,
            result: ProbeResult::Ok(snap),
        }),
        Instant::now(),
    );
    // 1 Watch op (subdir Dir) + 1 Unsuppress for the anchor. File doesn't
    // contribute Watch.
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
    let r = e.tree.ensure(None, "log.txt", ResourceRole::User);
    e.tree.get_mut(r).unwrap().kind = ResourceKind::File;
    let req = SubAttachRequest {
        name: String::from("file-sub"),
        resource: r,
        path: None,
        config: ScanConfig::builder().build(),
        max_settle: MAX_SETTLE,
        settle: SETTLE,
        command: empty_command(),
        scope: EffectScope::SubtreeRoot,
        events: NO_EVENTS,
        log_output: false,
    };
    let (_sid, out) = e.attach_sub(req, Instant::now());
    let probe_kind = out.probe_ops.iter().find_map(|op| match op {
        ProbeOp::Probe { request } => Some(request.kind),
        _ => None,
    });
    assert_eq!(probe_kind, Some(ProbeKind::File));
}

// ---- on_watch_op_rejected ----

#[test]
fn watch_op_rejected_clamps_watch_demand_to_zero() {
    // Build a Resource with watch_demand=2 (multi-Profile co-located).
    // Inject WatchOpRejected. Expect watch_demand → 0, Unwatch emitted,
    // Diagnostic.
    let mut e = Engine::new();
    let r = e.tree.ensure(None, "x", ResourceRole::User);
    e.tree.get_mut(r).unwrap().kind = ResourceKind::Dir;
    let mut out = StepOutput::default();
    crate::refcounts::add_watch_demand(&mut e.tree, r, NO_EVENTS, &mut out);
    crate::refcounts::add_watch_demand(&mut e.tree, r, NO_EVENTS, &mut out);
    assert_eq!(e.tree.get(r).unwrap().watch_demand, 2);

    let result = e.step(
        Input::WatchOpRejected {
            resource: r,
            op: WatchOp::Watch {
                resource: r,
                path: std::path::PathBuf::new(),
                kind: specter_core::ResourceKind::Unknown,
                events: specter_core::ClassSet::EMPTY,
            },
            errno: 24,
        },
        Instant::now(),
    );

    assert_eq!(e.tree.get(r).unwrap().watch_demand, 0);
    assert!(
        result
            .watch_ops
            .iter()
            .any(|op| matches!(op, WatchOp::Unwatch { .. }))
    );
    assert!(
        result
            .diagnostics
            .iter()
            .any(|d| matches!(d, Diagnostic::WatchOpRejected { errno: 24, .. }))
    );
}

#[test]
fn watch_op_rejected_already_unwatched_emits_diagnostic_only() {
    let mut e = Engine::new();
    let r = e.tree.ensure(None, "x", ResourceRole::User);
    let result = e.step(
        Input::WatchOpRejected {
            resource: r,
            op: WatchOp::Unwatch { resource: r },
            errno: 24,
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
    // Set up: pre-existing /foo, attach `foo/bar` (descent at /foo).
    let mut e = Engine::new();
    let foo = e.tree.ensure(None, "foo", ResourceRole::User);
    e.tree.get_mut(foo).unwrap().kind = ResourceKind::Dir;
    let req = SubAttachRequest::for_path(
        "guard".into(),
        std::path::PathBuf::from("foo/bar"),
        ScanConfig::builder().recursive(true).build(),
        MAX_SETTLE,
        SETTLE,
        empty_command(),
        EffectScope::SubtreeRoot,
        NO_EVENTS,
        false,
    );
    let (_sid, _) = e.attach_sub(req, Instant::now());
    let pid = {
        let mut iter = e.profiles.iter();
        iter.next().expect("profile exists").0
    };
    assert!(e.descent_state(pid).is_some());
    let initial_corr = e.pending_probe(pid).expect("first probe in flight");
    let initial_demand = e.tree.get(foo).unwrap().watch_demand;
    assert_eq!(initial_demand, 1);

    // Inject WatchOpRejected (e.g., EMFILE) for the descent prefix.
    let result = e.step(
        Input::WatchOpRejected {
            resource: foo,
            op: WatchOp::Watch {
                resource: foo,
                path: std::path::PathBuf::from("foo"),
                kind: specter_core::ResourceKind::Unknown,
                events: specter_core::ClassSet::EMPTY,
            },
            errno: 24,
        },
        Instant::now(),
    );

    // The clamp zeroed watch_demand; the descent has been purged.
    assert_eq!(e.tree.get(foo).unwrap().watch_demand, 0);
    assert!(
        e.descent_state(pid).is_none(),
        "descent purged on rejection",
    );

    // A Cancel for the in-flight probe was emitted.
    assert!(
        result
            .probe_ops
            .iter()
            .any(|op| matches!(op, ProbeOp::Cancel { profile } if *profile == pid)),
        "ProbeOp::Cancel emitted for the in-flight descent probe",
    );

    // ProfileClaimPurged{DescentPrefix} surfaces (in addition to WatchOpRejected).
    assert!(
        result
            .diagnostics
            .iter()
            .any(|d| matches!(d, Diagnostic::ProfileClaimPurged {
                profile, claim, resource, errno
            } if *profile == pid
                && *claim == ClaimKind::DescentPrefix
                && *resource == foo
                && *errno == 24)),
        "ProfileClaimPurged{{DescentPrefix}} diagnostic emitted",
    );

    // Late `ProbeResponse` for the cancelled correlation arrives — must
    // be silently discarded (descent removed, correlation no longer
    // matches anything).
    let late = e.step(
        Input::ProbeResponse(ProbeResponse {
            profile: pid,
            correlation: initial_corr,
            result: ProbeResult::Vanished,
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
            op: WatchOp::Watch {
                resource: r,
                path: std::path::PathBuf::new(),
                kind: specter_core::ResourceKind::Unknown,
                events: specter_core::ClassSet::EMPTY,
            },
            errno: 24,
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
    // Anchor flag cleared.
    assert!(
        !e.profiles.get(pid).unwrap().anchor_contribution,
        "anchor_contribution cleared by purge",
    );
}

#[test]
fn watch_op_rejected_purges_multiple_descents_at_same_prefix() {
    // Two Profiles share a descent prefix (e.g., two Subs anchored at
    // siblings under the same scaffold). WatchOpRejected purges both.
    let mut e = Engine::new();
    let foo = e.tree.ensure(None, "foo", ResourceRole::User);
    e.tree.get_mut(foo).unwrap().kind = ResourceKind::Dir;
    let req_a = SubAttachRequest::for_path(
        "a".into(),
        std::path::PathBuf::from("foo/sib_a"),
        ScanConfig::builder().recursive(true).build(),
        MAX_SETTLE,
        SETTLE,
        empty_command(),
        EffectScope::SubtreeRoot,
        NO_EVENTS,
        false,
    );
    let req_b = SubAttachRequest::for_path(
        "b".into(),
        std::path::PathBuf::from("foo/sib_b"),
        ScanConfig::builder().recursive(true).build(),
        MAX_SETTLE,
        SETTLE,
        empty_command(),
        EffectScope::SubtreeRoot,
        NO_EVENTS,
        false,
    );
    let (sid_a, _) = e.attach_sub(req_a, Instant::now());
    let (sid_b, _) = e.attach_sub(req_b, Instant::now());
    let pid_a = e.subs.get(sid_a).unwrap().profile;
    let pid_b = e.subs.get(sid_b).unwrap().profile;
    // Both descents at /foo (different anchors).
    assert!(e.descent_state(pid_a).is_some());
    assert!(e.descent_state(pid_b).is_some());
    assert_eq!(e.tree.get(foo).unwrap().watch_demand, 2);

    let result = e.step(
        Input::WatchOpRejected {
            resource: foo,
            op: WatchOp::Watch {
                resource: foo,
                path: std::path::PathBuf::from("foo"),
                kind: specter_core::ResourceKind::Unknown,
                events: specter_core::ClassSet::EMPTY,
            },
            errno: 24,
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

// ---- P11.0: anchor_contribution flag drives reap correctness ----

#[test]
fn seed_vanished_then_reap_releases_anchor_via_contribution_flag() {
    // Regression: a Profile that had anchor_contribution = true (set on
    // immediate-Seed attach) but Seed Vanished (clearing baseline /
    // current) must still release the anchor's watch_demand on reap.
    // The pre-fix `had_watch_contribution = baseline.is_some() ||
    // current.is_some()` heuristic missed this case.
    let (mut e, pid, sid, r, _now) = engine_with_attached_sub();
    // Anchor watch_demand is 1, anchor_contribution is true.
    assert_eq!(e.tree.get(r).unwrap().watch_demand, 1);
    assert!(e.profiles.get(pid).unwrap().anchor_contribution);

    // Detach the Sub mid-burst → reap_pending = true.
    let _ = e.detach_sub(sid, Instant::now());
    assert!(e.profiles.get(pid).unwrap().reap_pending);

    // Drive Seed Vanished to fire the reap.
    let correlation = e.pending_probe(pid).expect("Verifying probe in flight");
    let out = e.step(
        Input::ProbeResponse(ProbeResponse {
            profile: pid,
            correlation,
            result: ProbeResult::Vanished,
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
        "Unwatch for the anchor emitted on reap (anchor_contribution flag)",
    );
}

#[test]
fn anchor_terminal_event_clears_anchor_contribution_flag() {
    // After on_anchor_terminal_event releases the anchor, a subsequent
    // reap must NOT double-release it (the flag is cleared).
    let (mut e, pid, _sid, r, _now) = engine_with_attached_sub();
    complete_seed_burst(&mut e, pid, r);
    assert!(e.profiles.get(pid).unwrap().anchor_contribution);

    // Inject a Removed event at the anchor: the terminal event releases
    // the anchor's contribution and clears the flag.
    let _ = e.step(
        Input::FsEvent {
            resource: r,
            event: FsEvent::Removed,
        },
        Instant::now(),
    );
    assert!(
        !e.profiles.get(pid).unwrap().anchor_contribution,
        "anchor_contribution cleared by terminal event",
    );
    assert_eq!(
        e.tree.get(r).unwrap().watch_demand,
        0,
        "anchor's watch_demand released",
    );
}

// ---- detach_sub ----

#[test]
fn detach_sub_idle_profile_reaps_immediately() {
    let (mut e, pid, sid, r, _now) = engine_with_attached_sub();
    complete_seed_burst(&mut e, pid, r);
    // Profile is now Idle.
    assert!(matches!(
        e.profiles.get(pid).unwrap().state,
        ProfileState::Idle,
    ));
    let out = e.detach_sub(sid, Instant::now());
    // Profile reaped; anchor unwatched.
    assert!(e.profiles.get(pid).is_none());
    assert!(
        out.watch_ops
            .iter()
            .any(|op| matches!(op, WatchOp::Unwatch { resource } if *resource == r))
    );
    assert!(
        out.diagnostics
            .iter()
            .any(|d| matches!(d, Diagnostic::ReapPendingResolved { .. }))
    );
}

#[test]
fn detach_sub_active_profile_marks_reap_pending() {
    let (mut e, pid, sid, _r, _now) = engine_with_attached_sub();
    // Profile is Active(Seed Probing) — Seed-burst still in flight.
    let _out = e.detach_sub(sid, Instant::now());
    let p = e.profiles.get(pid).expect("profile alive until burst ends");
    assert!(p.reap_pending);
    assert_eq!(p.sub_refcount, 0);
}

#[test]
fn reap_pending_burst_completion_skips_effects_and_reaps() {
    // Sub on Active(Standard, stable) Profile; detach mid-burst; finish
    // burst — no Effect emitted; Profile reaped.
    let (mut e, pid, sid, root, _now) = engine_with_attached_sub();
    complete_seed_burst(&mut e, pid, root);

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
    let _ = e.detach_sub(sid, t1);
    assert!(e.profiles.get(pid).unwrap().reap_pending);

    // Drain the settle timer to advance to Probing.
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
    let correlation = e.pending_probe(pid).expect("Verifying probe in flight");

    // Inject stable response. Profile should reap; no Effect emitted.
    let snap = dir_tree_snap(root, vec![]);
    let out = e.step(
        Input::ProbeResponse(ProbeResponse {
            profile: pid,
            correlation,
            result: ProbeResult::Ok(snap),
        }),
        t2,
    );
    assert!(out.effects.is_empty(), "reap_pending suppresses Effect");
    assert!(e.profiles.get(pid).is_none(), "Profile reaped at burst end");
}

#[test]
fn detach_sub_settle_recomputed_when_subs_remain() {
    // Profile with two Subs of different settle; detach the faster one;
    // remaining Sub's settle wins.
    let mut e = Engine::new();
    let r = e.tree.ensure(None, "anchor", ResourceRole::User);
    e.tree.get_mut(r).unwrap().kind = ResourceKind::Dir;
    let now = Instant::now();
    let cfg = ScanConfig::builder().recursive(true).build();
    let (sid_fast, _) = e.attach_sub(
        SubAttachRequest {
            name: "fast".into(),
            resource: r,
            path: None,
            config: cfg.clone(),
            max_settle: MAX_SETTLE,
            settle: Duration::from_millis(50),
            command: empty_command(),
            scope: EffectScope::SubtreeRoot,
            events: NO_EVENTS,
            log_output: false,
        },
        now,
    );
    let pid = e.subs.get(sid_fast).unwrap().profile;
    let (_sid_slow, _) = e.attach_sub(
        SubAttachRequest {
            name: "slow".into(),
            resource: r,
            path: None,
            config: cfg,
            max_settle: MAX_SETTLE,
            settle: Duration::from_millis(200),
            command: empty_command(),
            scope: EffectScope::SubtreeRoot,
            events: NO_EVENTS,
            log_output: false,
        },
        now,
    );
    // Fast Sub's settle wins on attach.
    assert_eq!(
        e.profiles.get(pid).unwrap().settle,
        Duration::from_millis(50)
    );

    // Detach the fast Sub. Remaining settle is the slow one's.
    let _ = e.detach_sub(sid_fast, now);
    assert_eq!(
        e.profiles.get(pid).unwrap().settle,
        Duration::from_millis(200)
    );
}

// ---- on_config_diff ----

#[test]
fn config_diff_added_only_attaches_subs() {
    let mut e = Engine::new();
    let r = e.tree.ensure(None, "anchor", ResourceRole::User);
    e.tree.get_mut(r).unwrap().kind = ResourceKind::Dir;

    let req = SubAttachRequest {
        name: "added".into(),
        resource: r,
        path: None,
        config: ScanConfig::builder().build(),
        max_settle: MAX_SETTLE,
        settle: SETTLE,
        command: empty_command(),
        scope: EffectScope::SubtreeRoot,
        events: NO_EVENTS,
        log_output: false,
    };
    let mut diff = specter_core::SubRegistryDiff::default();
    diff.added.push(req);

    let out = e.step(Input::ConfigDiff(diff), Instant::now());
    assert!(
        out.watch_ops
            .iter()
            .any(|op| matches!(op, WatchOp::Watch { .. }))
    );
    assert!(
        out.probe_ops
            .iter()
            .any(|op| matches!(op, ProbeOp::Probe { .. }))
    );
    assert_eq!(e.subs().len(), 1);
}

#[test]
fn config_diff_removed_then_added_atomic() {
    // Engine has Sub A at /anchor; ConfigDiff removes A and adds B
    // (path-based, anchored at /anchor — re-creates the slot if A's
    // detach reaped it).
    let (mut e, pid_a, sid_a, r, _now) = engine_with_attached_sub();
    complete_seed_burst(&mut e, pid_a, r);

    // Path-based add — the engine re-materializes if needed.
    let req_b = SubAttachRequest::for_path(
        "B".into(),
        std::path::PathBuf::from("anchor"),
        ScanConfig::builder().build(), // different config_hash (non-recursive)
        MAX_SETTLE,
        SETTLE,
        empty_command(),
        EffectScope::SubtreeRoot,
        NO_EVENTS,
        false,
    );
    let mut diff = specter_core::SubRegistryDiff::default();
    diff.removed.push(sid_a);
    diff.added.push(req_b);

    let out = e.step(Input::ConfigDiff(diff), Instant::now());
    // A reaped (sub registry no longer has it); B added.
    assert!(e.subs().get(sid_a).is_none());
    assert_eq!(e.subs().len(), 1);
    // Single sorted StepOutput; multiple watch_ops merged.
    assert!(!out.watch_ops.is_empty());
}

// ---- emit_effects PerStableFile ----

#[test]
fn per_stable_file_fires_one_effect_per_created_entry() {
    // Profile with PerStableFile Sub; burst stabilizes with 2 created
    // file entries.
    let mut e = Engine::new();
    let r = e.tree.ensure(None, "anchor", ResourceRole::User);
    e.tree.get_mut(r).unwrap().kind = ResourceKind::Dir;
    let now = Instant::now();
    let req = SubAttachRequest {
        name: "fmt".into(),
        resource: r,
        path: None,
        config: ScanConfig::builder().recursive(true).build(),
        max_settle: MAX_SETTLE,
        settle: SETTLE,
        command: diff_command(),
        scope: EffectScope::PerStableFile,
        events: NO_EVENTS,
        log_output: false,
    };
    let (sid, _) = e.attach_sub(req, now);
    let pid = e.subs.get(sid).unwrap().profile;

    // Complete Seed with empty baseline.
    let seed_corr = e.pending_probe(pid).expect("Verifying probe in flight");
    e.step(
        Input::ProbeResponse(ProbeResponse {
            profile: pid,
            correlation: seed_corr,
            result: ProbeResult::Ok(dir_tree_snap(r, vec![])),
        }),
        now,
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
    let std_corr = e.pending_probe(pid).expect("Verifying probe in flight");

    // Inject stable response with 2 file entries.
    let snap = dir_tree_snap(
        r,
        vec![("a.rs", EntryKind::File, 1), ("b.rs", EntryKind::File, 2)],
    );
    // Send same snap twice via state cycling: first probe sees the
    // change; second confirms stable. Simplify by re-routing — the
    // engine's Standard dispatch needs `current` set to the same snap
    // for stability. For this test we drive: snap1 (not stable), then
    // snap1 again (stable).
    let _ = e.step(
        Input::ProbeResponse(ProbeResponse {
            profile: pid,
            correlation: std_corr,
            result: ProbeResult::Ok(snap.clone()),
        }),
        t2,
    );
    // Now drain the rescheduled settle and inject the same snapshot.
    let t3 = t2 + SETTLE * 2;
    while let Some(entry) = e.pop_expired(t3) {
        e.step(
            Input::TimerExpired {
                profile: entry.profile,
                kind: entry.kind,
                id: entry.id,
            },
            t3,
        );
    }
    let std_corr2 = e.pending_probe(pid).expect("Verifying probe in flight");
    let out = e.step(
        Input::ProbeResponse(ProbeResponse {
            profile: pid,
            correlation: std_corr2,
            result: ProbeResult::Ok(snap),
        }),
        t3,
    );
    // Stable; baseline is empty, current has 2 files → diff.created has 2.
    let per_file_effects: Vec<&specter_core::Effect> = out
        .effects
        .iter()
        .filter(|e| matches!(&e.key, DedupKey::PerFile { sub, .. } if *sub == sid))
        .collect();
    assert_eq!(per_file_effects.len(), 2, "one Effect per created file");
    // diff_command = [ArgTemplate([Lit("fmt"), Placeholder($created)])] — the
    // literal "fmt" tiles per emitted multi-value entry. Each PerStableFile
    // Effect carries the same diff (all created entries), so each Effect's
    // argv expands to ["fmta.rs", "fmtb.rs"].
    for eff in &per_file_effects {
        assert_eq!(
            eff.command.argv,
            vec!["fmta.rs".to_string(), "fmtb.rs".to_string()],
            "literal tiles per multi-value entry"
        );
        // cwd is the anchor (a Dir), not the per-entry file path.
        assert_eq!(eff.cwd.as_os_str(), "anchor");
        // SPECTER_PATH is the per-entry path.
        let env = &eff.env;
        let path = &env.iter().find(|(k, _)| k == "SPECTER_PATH").unwrap().1;
        assert!(path.starts_with("anchor/"));
        // SPECTER_REL_PATH is the segment.
        let rel = &env.iter().find(|(k, _)| k == "SPECTER_REL_PATH").unwrap().1;
        assert!(rel == "a.rs" || rel == "b.rs");
        // SPECTER_EVENT_KIND = "file" (PerStableFile scope).
        assert_eq!(
            env.iter()
                .find(|(k, _)| k == "SPECTER_EVENT_KIND")
                .unwrap()
                .1,
            "file"
        );
    }
}

#[test]
fn per_stable_file_skips_dir_entries() {
    // Mixed Diff: 1 created File, 1 created Dir, 1 modified Dir.
    // PerStableFile must fire ONE Effect (the File), not three.
    let mut e = Engine::new();
    let r = e.tree.ensure(None, "anchor", ResourceRole::User);
    e.tree.get_mut(r).unwrap().kind = ResourceKind::Dir;
    let now = Instant::now();
    let req = SubAttachRequest {
        name: "fmt".into(),
        resource: r,
        path: None,
        config: ScanConfig::builder().recursive(true).build(),
        max_settle: MAX_SETTLE,
        settle: SETTLE,
        command: diff_command(),
        scope: EffectScope::PerStableFile,
        events: NO_EVENTS,
        log_output: false,
    };
    let (sid, _) = e.attach_sub(req, now);
    let pid = e.subs.get(sid).unwrap().profile;

    // Seed completes against a snapshot already containing one Dir
    // (`subdir`). After Seed, `subdir` is in the baseline and won't
    // re-appear as `created` later.
    let seed_corr = e.pending_probe(pid).expect("Verifying probe in flight");
    let seed_snap = dir_tree_snap(r, vec![("subdir", EntryKind::Dir, 10)]);
    e.step(
        Input::ProbeResponse(ProbeResponse {
            profile: pid,
            correlation: seed_corr,
            result: ProbeResult::Ok(seed_snap),
        }),
        now,
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
    let std_corr = e.pending_probe(pid).expect("Verifying probe in flight");

    // Mixed snapshot: subdir (modified — different mtime), newdir (new
    // Dir), main.rs (new File). Diff = created=[newdir, main.rs],
    // modified=[subdir]. Only main.rs should fire.
    let mixed_snap = dir_tree_snap(
        r,
        vec![
            ("main.rs", EntryKind::File, 1),
            ("newdir", EntryKind::Dir, 11),
            // subdir has different mtime ⇒ counted as Modified.
            ("subdir", EntryKind::Dir, 10),
        ],
    );
    e.step(
        Input::ProbeResponse(ProbeResponse {
            profile: pid,
            correlation: std_corr,
            result: ProbeResult::Ok(mixed_snap.clone()),
        }),
        t2,
    );
    let t3 = t2 + SETTLE * 2;
    while let Some(entry) = e.pop_expired(t3) {
        e.step(
            Input::TimerExpired {
                profile: entry.profile,
                kind: entry.kind,
                id: entry.id,
            },
            t3,
        );
    }
    let std_corr2 = e.pending_probe(pid).expect("Verifying probe in flight");
    let out = e.step(
        Input::ProbeResponse(ProbeResponse {
            profile: pid,
            correlation: std_corr2,
            result: ProbeResult::Ok(mixed_snap),
        }),
        t3,
    );

    let per_file_effects: Vec<&specter_core::Effect> = out
        .effects
        .iter()
        .filter(|e| matches!(&e.key, DedupKey::PerFile { sub, .. } if *sub == sid))
        .collect();
    assert_eq!(
        per_file_effects.len(),
        1,
        "exactly ONE Effect for the File entry; Dir entries skipped"
    );
    // SPECTER_REL_PATH must be the file, not a directory.
    let rel = &per_file_effects[0]
        .env
        .iter()
        .find(|(k, _)| k == "SPECTER_REL_PATH")
        .unwrap()
        .1;
    assert_eq!(rel, "main.rs");
}

// ---------- Bundles B1, B2, B3 ----------

/// Drive a complete attach + Seed-Ok + FsEvent + two Standard-Ok responses
/// (the second confirming stability) and return the StepOutput that contains
/// the Effect emission. Common harness for B1 SubtreeRoot tests.
///
/// Stability requires `current.subtree_at(target).dir_hash() ==
/// response.dir_hash()`. The first Standard probe diffs against the Seed
/// baseline (different ⇒ Settling); the second probe lands the same
/// response post-graft ⇒ stable ⇒ Effect fires.
fn drive_to_first_effect(
    e: &mut Engine,
    pid: specter_core::ProfileId,
    root: ResourceId,
    now: Instant,
) -> StepOutput {
    // Complete Seed.
    complete_seed_burst(e, pid, root);
    // Inject FsEvent → Standard burst at root.
    let _ = e.step(
        Input::FsEvent {
            resource: root,
            event: FsEvent::StructureChanged,
        },
        now,
    );
    // Drain settle timer → Verifying.
    let settle_timer = match &e.profiles.get(pid).unwrap().state {
        ProfileState::Active(b) => match &b.phase {
            BurstPhase::Batching { settle_timer } => *settle_timer,
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
    let correlation = e.pending_probe(pid).expect("Verifying probe in flight");
    // First probe — response differs from Seed baseline ⇒ not-stable ⇒
    // Batching.
    let snap1 = dir_tree_snap(root, vec![("a.rs", EntryKind::File, 1)]);
    let _ = e.step(
        Input::ProbeResponse(ProbeResponse {
            profile: pid,
            correlation,
            result: ProbeResult::Ok(snap1),
        }),
        now + SETTLE,
    );
    // Drain settle timer → Verifying again.
    let settle_timer2 = match &e.profiles.get(pid).unwrap().state {
        ProfileState::Active(b) => match &b.phase {
            BurstPhase::Batching { settle_timer } => *settle_timer,
            _ => panic!("expected Batching"),
        },
        _ => panic!("expected Active(Batching)"),
    };
    let _ = e.step(
        Input::TimerExpired {
            profile: pid,
            kind: TimerKind::Settle,
            id: settle_timer2,
        },
        now + SETTLE + SETTLE,
    );
    let correlation2 = e.pending_probe(pid).expect("Verifying probe in flight");
    // Second probe — same content ⇒ stable ⇒ Effect.
    let snap2 = dir_tree_snap(root, vec![("a.rs", EntryKind::File, 1)]);
    e.step(
        Input::ProbeResponse(ProbeResponse {
            profile: pid,
            correlation: correlation2,
            result: ProbeResult::Ok(snap2),
        }),
        now + SETTLE + SETTLE,
    )
}

#[test]
fn b1_records_last_emitted_dir_hash_after_subtree_effect() {
    let (mut e, pid, _sid, root, _now) = engine_with_attached_sub();
    let now = Instant::now();
    let out = drive_to_first_effect(&mut e, pid, root, now);

    // First Effect fires (no prior emission).
    assert_eq!(out.effects.len(), 1, "first Standard-Ok fires Effect");
    // last_emitted_dir_hash now has one entry for the SubtreeRoot key.
    let p = e.profiles.get(pid).unwrap();
    assert_eq!(p.last_emitted_dir_hash.len(), 1);
    let (key, _hash) = p.last_emitted_dir_hash.iter().next().unwrap();
    assert!(matches!(
        key,
        DedupKey::Subtree {
            profile,
            ..
        } if *profile == pid,
    ));
}

#[test]
fn b1_clears_last_emitted_dir_hash_on_effect_complete_failed() {
    let (mut e, pid, sid, root, _now) = engine_with_attached_sub();
    let now = Instant::now();
    let _ = drive_to_first_effect(&mut e, pid, root, now);

    // Confirm B1 wrote an entry.
    assert!(
        !e.profiles
            .get(pid)
            .unwrap()
            .last_emitted_dir_hash
            .is_empty()
    );

    // EffectComplete::Failed → B1 clears the entry for this DedupKey.
    let dk = DedupKey::Subtree {
        sub: sid,
        profile: pid,
    };
    let _ = e.step(
        Input::EffectComplete {
            sub: sid,
            key: dk,
            result: EffectOutcome::Failed {
                exit_code: Some(1),
                signal: None,
            },
        },
        now,
    );
    assert!(
        e.profiles
            .get(pid)
            .unwrap()
            .last_emitted_dir_hash
            .is_empty(),
        "Failed Effect clears the suppression entry",
    );
}

#[test]
fn b3_recovery_seed_no_prior_emit_does_not_fire() {
    // Fresh attach → Seed-Ok → no prior `last_emitted_dir_hash` ⇒
    // b3_seed_drift_observed returns an empty key set ⇒ no Effect
    // (preserves "fresh Seed never fires Effect").
    let (mut e, pid, _sid, root, _now) = engine_with_attached_sub();
    let correlation = e.pending_probe(pid).expect("Verifying probe in flight");
    let snap = dir_tree_snap(root, vec![]);
    let out = e.step(
        Input::ProbeResponse(ProbeResponse {
            profile: pid,
            correlation,
            result: ProbeResult::Ok(snap),
        }),
        Instant::now(),
    );
    assert!(out.effects.is_empty(), "fresh-Profile Seed fires no Effect");
}

/// Multi-Sub Profile with two `SubtreeRoot` Subs sharing one
/// `(resource, config_hash)`. Manually populate
/// `last_emitted_dir_hash` so one Sub's recorded hash matches the
/// post-graft `current` and the other's diverges. Trigger a Seed-Ok;
/// the per-key B3 helper must return only the drifted key, and
/// `emit_effects` must fire one Effect — the matched-key Sub stays
/// silent because re-running its command would be a no-op against an
/// unchanged tree.
#[test]
fn b3_recovery_seed_per_key_only_drifted_subs_fire() {
    use specter_core::DedupKey;

    let (mut e, pid, sid_a, root, _now) = engine_with_attached_sub();
    // Seed → Idle so we have a stable baseline for setup.
    complete_seed_burst(&mut e, pid, root);

    // Attach a second SubtreeRoot Sub onto the same Profile (same
    // `(resource, max_settle, config, events)` ⇒ shared config_hash ⇒
    // shared Profile via the existing-Profile path).
    let req_b = SubAttachRequest {
        name: String::from("sub-b"),
        resource: root,
        path: None,
        config: ScanConfig::builder().recursive(true).build(),
        max_settle: MAX_SETTLE,
        settle: SETTLE,
        command: empty_command(),
        scope: EffectScope::SubtreeRoot,
        events: NO_EVENTS,
        log_output: false,
    };
    let (sid_b, _) = e.attach_sub(req_b, Instant::now());
    assert_eq!(
        e.subs.get(sid_b).unwrap().profile,
        pid,
        "Sub B joins the existing Profile via shared config_hash",
    );

    // The post-Seed `current` snapshot's anchor `dir_hash` is what the
    // helper compares against. Seed left `current = baseline = empty
    // dir`, so capture that hash and synthesise per-Sub history:
    //  - Sub A's recorded hash equals `current_hash`         → no drift.
    //  - Sub B's recorded hash differs                       → drifted.
    let curr_hash: u128 = match e.profiles.get(pid).unwrap().current.as_ref().unwrap() {
        TreeSnapshot::Dir(arc) => arc.dir_hash(),
        TreeSnapshot::File(_) => unreachable!("dir-anchored Profile"),
    };
    let stale_hash: u128 = curr_hash.wrapping_add(1);
    let key_a = DedupKey::Subtree {
        sub: sid_a,
        profile: pid,
    };
    let key_b = DedupKey::Subtree {
        sub: sid_b,
        profile: pid,
    };
    {
        let p = e.profiles.get_mut(pid).unwrap();
        p.last_emitted_dir_hash.insert(key_a.clone(), curr_hash);
        p.last_emitted_dir_hash.insert(key_b.clone(), stale_hash);
    }

    // Trigger a recovery-style Seed: clear `current` so the next probe
    // response drives `dispatch_seed_ok`, then synthesise an Active(Seed)
    // burst pointing at the anchor (matches the shape `start_seed_burst`
    // produces). The helper would normally be reached via a real
    // recovery; the manual setup is the cheapest path that exercises the
    // dispatch arm.
    let snap = dir_tree_snap(root, vec![]);
    {
        let p = e.profiles.get_mut(pid).unwrap();
        // Force a Seed dispatch by emptying current; the helper compares
        // against the post-graft hash, which `dispatch_seed_ok` writes
        // from the response.
        p.current = None;
        p.baseline = None;
    }

    // Drive the Seed via `start_seed_burst` + an injected Ok response.
    let mut out = StepOutput::default();
    e.start_seed_burst(pid, Instant::now(), &mut out);
    let seed_corr = e.pending_probe(pid).expect("Seed probe in flight");
    let seed_out = e.step(
        Input::ProbeResponse(ProbeResponse {
            profile: pid,
            correlation: seed_corr,
            result: ProbeResult::Ok(snap),
        }),
        Instant::now(),
    );

    // Exactly one Effect — Sub B (whose key drifted). Sub A's key
    // matches the post-graft hash, so the per-key filter excludes it.
    assert_eq!(
        seed_out.effects.len(),
        1,
        "only the drifted Sub fires; got {} effects",
        seed_out.effects.len(),
    );
    assert_eq!(
        seed_out.effects[0].key, key_b,
        "the drifted SubtreeRoot key fires; the matched key stays silent",
    );
}

/// Per-key drift check is bool-equivalent at the boundaries: empty
/// `last_emitted_dir_hash` ⇒ empty result; all entries match ⇒ empty
/// result; at least one differs ⇒ non-empty result.
#[test]
fn b3_per_key_helper_returns_only_subtree_drifted_keys() {
    use specter_core::DedupKey;

    let (mut e, pid, sid, root, _now) = engine_with_attached_sub();
    complete_seed_burst(&mut e, pid, root);

    // Compute the current dir_hash and inject one matching + one
    // diverging entry, plus one PerFile entry that should be ignored.
    let curr_hash: u128 = match e.profiles.get(pid).unwrap().current.as_ref().unwrap() {
        TreeSnapshot::Dir(arc) => arc.dir_hash(),
        TreeSnapshot::File(_) => unreachable!("dir-anchored Profile"),
    };
    let key_subtree_match = DedupKey::Subtree {
        sub: sid,
        profile: pid,
    };
    // A second subtree key with the same (sub, profile) collides with
    // `key_subtree_match`; use a synthetic SubId that's distinct. Any
    // distinct (sub, profile) is fine — the helper doesn't validate
    // against the live SubRegistry.
    use slotmap::KeyData;
    let synthetic_sub = SubId::from(KeyData::from_ffi(0xdead_beef));
    let key_subtree_drift = DedupKey::Subtree {
        sub: synthetic_sub,
        profile: pid,
    };
    let key_perfile = DedupKey::PerFile {
        sub: sid,
        profile: pid,
        resource: root,
    };
    {
        let p = e.profiles.get_mut(pid).unwrap();
        p.last_emitted_dir_hash
            .insert(key_subtree_match.clone(), curr_hash);
        p.last_emitted_dir_hash
            .insert(key_subtree_drift.clone(), curr_hash.wrapping_add(7));
        p.last_emitted_dir_hash
            .insert(key_perfile, curr_hash.wrapping_add(13));
    }

    let drifted = e.b3_seed_drift_observed(pid);
    assert_eq!(drifted.len(), 1, "only the diverged Subtree key returns");
    assert_eq!(
        drifted[0], key_subtree_drift,
        "PerFile keys are filtered; matched Subtree keys are filtered",
    );
}

/// Standard burst with a per-stable-file Sub: drift filter is `None`,
/// PerFile keys still emit per matching diff entry. This pins that the
/// Commit-4 narrowing of the Seed-drift path didn't accidentally skip
/// PerFile emission on the unrelated Standard burst path.
#[test]
fn b3_per_key_filter_does_not_affect_standard_burst_perfile_emission() {
    let mut e = Engine::new();
    let r = e.tree.ensure(None, "anchor", ResourceRole::User);
    e.tree.get_mut(r).unwrap().kind = ResourceKind::Dir;
    let now = Instant::now();
    let req = SubAttachRequest {
        name: String::from("fmt"),
        resource: r,
        path: None,
        config: ScanConfig::builder().recursive(true).build(),
        max_settle: MAX_SETTLE,
        settle: SETTLE,
        command: empty_command(),
        scope: EffectScope::PerStableFile,
        events: ClassSet::CONTENT,
        log_output: false,
    };
    let (_sid, _) = e.attach_sub(req, now);
    let pid = e.profiles.iter().next().unwrap().0;
    // Seed → Idle.
    let seed_corr = e.pending_probe(pid).expect("Seed probe in flight");
    e.step(
        Input::ProbeResponse(ProbeResponse {
            profile: pid,
            correlation: seed_corr,
            result: ProbeResult::Ok(dir_tree_snap(r, vec![])),
        }),
        now,
    );

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
    let snap_with_file = dir_tree_snap(r, vec![("new.rs", EntryKind::File, 5)]);
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
        let correlation = e.pending_probe(pid).expect("Verifying probe in flight");
        let out = e.step(
            Input::ProbeResponse(ProbeResponse {
                profile: pid,
                correlation,
                result: ProbeResult::Ok(snap_with_file.clone()),
            }),
            t,
        );
        if !out.effects.is_empty() {
            effect_out = Some(out);
            break;
        }
    }
    let out = effect_out.expect("Standard burst stabilised and emitted");
    assert_eq!(
        out.effects.len(),
        1,
        "Standard burst with PerFile Sub fires one Effect for the new file",
    );
}

#[test]
fn has_per_file_fds_is_invariant_for_profile_lifetime() {
    // Under D3 the events mask folds into `config_hash`, so every Sub on
    // a Profile shares the same events by construction. `has_per_file_fds`
    // is derived once at `Profile::new` from the events mask and never
    // flips during the Profile's lifetime — the prior B2 recompute
    // machinery (recompute_has_per_file_sub / update_per_leaf_watch_demand)
    // is removed because it's structurally unreachable.
    //
    // This test pins the new invariant: a Profile constructed with a
    // mask containing CONTENT (or METADATA) starts with the flag set,
    // and a Sub attaching via the same `(resource, config_hash)` does
    // not change it.
    let mut e = Engine::new();
    let r = e.tree.ensure(None, "anchor", ResourceRole::User);
    e.tree.get_mut(r).unwrap().kind = ResourceKind::Dir;
    let req = SubAttachRequest {
        name: String::from("formatter"),
        resource: r,
        path: None,
        config: ScanConfig::builder().recursive(true).build(),
        max_settle: MAX_SETTLE,
        settle: SETTLE,
        command: empty_command(),
        scope: EffectScope::PerStableFile,
        events: ClassSet::CONTENT,
        log_output: false,
    };
    let (sid, _out) = e.attach_sub(req, Instant::now());
    let pid = e.subs.get(sid).unwrap().profile;
    assert!(
        e.profiles.get(pid).unwrap().has_per_file_fds,
        "CONTENT-mask Profile has has_per_file_fds = true at construction",
    );

    // A Sub with the same `(resource, max_settle, scan, events)` shares
    // the existing Profile (D3); the flag stays true.
    let req2 = SubAttachRequest {
        name: String::from("formatter-2"),
        resource: r,
        path: None,
        config: ScanConfig::builder().recursive(true).build(),
        max_settle: MAX_SETTLE,
        settle: SETTLE,
        command: empty_command(),
        scope: EffectScope::PerStableFile,
        events: ClassSet::CONTENT,
        log_output: false,
    };
    let (_sid2, _) = e.attach_sub(req2, Instant::now());
    assert!(e.profiles.get(pid).unwrap().has_per_file_fds);

    // Detaching the second Sub leaves the Profile alive (sub_refcount > 0
    // before detach); the flag still doesn't flip because the Profile's
    // events mask is invariant.
    let _ = e.detach_sub(sid, Instant::now());
    assert!(e.profiles.get(pid).unwrap().has_per_file_fds);
}

#[test]
fn structure_only_profile_has_per_file_fds_false() {
    // Inverse case: a STRUCTURE-only mask leaves `has_per_file_fds`
    // false. walk_pair then doesn't bump per-leaf watch_demand for
    // covered files.
    let mut e = Engine::new();
    let r = e.tree.ensure(None, "anchor", ResourceRole::User);
    e.tree.get_mut(r).unwrap().kind = ResourceKind::Dir;
    let req = SubAttachRequest {
        name: String::from("ls-only"),
        resource: r,
        path: None,
        config: ScanConfig::builder().recursive(true).build(),
        max_settle: MAX_SETTLE,
        settle: SETTLE,
        command: empty_command(),
        scope: EffectScope::SubtreeRoot,
        events: ClassSet::STRUCTURE,
        log_output: false,
    };
    let (sid, _) = e.attach_sub(req, Instant::now());
    let pid = e.subs.get(sid).unwrap().profile;
    assert!(!e.profiles.get(pid).unwrap().has_per_file_fds);
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
        let pid = e.subs.get(sid).unwrap().profile;
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
                    for c in s.probe_ops.iter().filter_map(|op| match op {
                        ProbeOp::Probe { request } => Some(request.correlation),
                        _ => None,
                    }) {
                        *last_correlation = Some(c);
                    }
                    extend_step_output(&mut combined, s);
                }
                combined
            }
            Action::Probe => {
                let snap = dir_tree_snap(r, vec![]);
                let corr = last_correlation.unwrap_or(specter_core::ProbeCorrelation(0));
                e.step(
                    Input::ProbeResponse(ProbeResponse {
                        profile: pid,
                        correlation: corr,
                        result: ProbeResult::Ok(snap),
                    }),
                    *t,
                )
            }
            Action::ProbeVanished => {
                let corr = last_correlation.unwrap_or(specter_core::ProbeCorrelation(0));
                e.step(
                    Input::ProbeResponse(ProbeResponse {
                        profile: pid,
                        correlation: corr,
                        result: ProbeResult::Vanished,
                    }),
                    *t,
                )
            }
            Action::ProbeFailed(errno) => {
                let corr = last_correlation.unwrap_or(specter_core::ProbeCorrelation(0));
                e.step(
                    Input::ProbeResponse(ProbeResponse {
                        profile: pid,
                        correlation: corr,
                        result: ProbeResult::Failed { errno },
                    }),
                    *t,
                )
            }
            Action::EffectComplete => e.step(
                Input::EffectComplete {
                    sub: sid,
                    key: DedupKey::Subtree {
                        sub: sid,
                        profile: pid,
                    },
                    result: EffectOutcome::Ok,
                },
                *t,
            ),
        };

        // Update last_correlation from any Probe in the output.
        for c in out.probe_ops.iter().filter_map(|op| match op {
            ProbeOp::Probe { request } => Some(request.correlation),
            _ => None,
        }) {
            *last_correlation = Some(c);
        }

        out
    }

    fn extend_step_output(dst: &mut StepOutput, src: StepOutput) {
        for op in src.watch_ops {
            dst.watch_ops.push(op);
        }
        for op in src.probe_ops {
            dst.probe_ops.push(op);
        }
        for ef in src.effects {
            dst.effects.push(ef);
        }
        for d in src.diagnostics {
            dst.diagnostics.push(d);
        }
    }

    fn fresh_engine_with_sub() -> (
        Engine,
        specter_core::SubId,
        ResourceId,
        Instant,
        Option<specter_core::ProbeCorrelation>,
    ) {
        let mut e = Engine::new();
        let r = e.tree.ensure(None, "anchor", ResourceRole::User);
        e.tree.get_mut(r).unwrap().kind = ResourceKind::Dir;
        let now = Instant::now();
        let req = SubAttachRequest {
            name: String::from("test"),
            resource: r,
            path: None,
            config: ScanConfig::builder().recursive(true).build(),
            max_settle: MAX_SETTLE,
            settle: SETTLE,
            command: empty_command(),
            scope: EffectScope::SubtreeRoot,
            events: NO_EVENTS,
            log_output: false,
        };
        let (sid, out) = e.attach_sub(req, now);
        let last_correlation = out.probe_ops.iter().find_map(|op| match op {
            ProbeOp::Probe { request } => Some(request.correlation),
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
                    .map(Engine::watch_op_key)
                    .collect();
                let mut sorted = watch_keys.clone();
                sorted.sort();
                prop_assert_eq!(watch_keys, sorted);

                // probe_ops sorted by ProfileId.
                let probe_keys: Vec<_> = out
                    .probe_ops
                    .iter()
                    .map(Engine::probe_op_key)
                    .collect();
                let mut sorted_p = probe_keys.clone();
                sorted_p.sort();
                prop_assert_eq!(probe_keys, sorted_p);
            }
        }

        /// I5: at most one outstanding ProbeRequest per Profile. Track
        /// outstanding probes via emit/cancel/respond; assert ≤ 1.
        #[test]
        fn prop_at_most_one_outstanding_probe(
            actions in prop::collection::vec(arb_action(), 0..16),
        ) {
            let (mut e, sid, r, mut t, mut last_correlation) =
                fresh_engine_with_sub();
            let pid = e.subs.get(sid).unwrap().profile;

            // attach_sub emits the initial Seed probe; outstanding = 1.
            let mut outstanding: u32 = 1;

            for action in actions {
                let was_probe = matches!(action, Action::Probe | Action::ProbeVanished | Action::ProbeFailed(_));
                let out = run_action(&mut e, sid, r, action, &mut t, &mut last_correlation);

                // Each Probe op increments; each Cancel and each accepted
                // ProbeResponse decrements. (We treat "any Probe op
                // emitted" as +1 and "any Cancel emitted" as -1; the test
                // doesn't care about the difference, only that the running
                // count stays ≤ 1.)
                let probes_emitted = out
                    .probe_ops
                    .iter()
                    .filter(|op| matches!(op, ProbeOp::Probe { .. }))
                    .count();
                let cancels_emitted = out
                    .probe_ops
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

                // Field-discipline I5: at most one outstanding probe per
                // Profile, expressed as a single `Option<ProbeCorrelation>`
                // slot. The `Option`-typed field makes `<= 1` trivially
                // true; the assertion is a regression guard for any
                // future change that broadens the slot's shape (e.g.,
                // accidentally introducing per-state probe correlations
                // again).
                if let Some(p) = e.profiles.get(pid) {
                    let probing_count = u32::from(p.pending_probe.is_some());
                    prop_assert!(
                        probing_count <= 1,
                        "I5 field-discipline: pending_probe carries at most one outstanding probe",
                    );
                }
            }
        }

        /// `prop_seed_burst_emits_no_effects`: from a fresh attach, the
        /// Seed-burst's eventual ProbeResponse path never produces an
        /// Effect (fresh Seed bursts never emit Effects).
        #[test]
        fn prop_seed_burst_emits_no_effects(
            seed_outcome in prop_oneof![
                Just(0),  // Ok
                Just(1),  // Vanished
                Just(2),  // Failed
            ],
        ) {
            let (mut e, sid, r, now, last_correlation) =
                fresh_engine_with_sub();
            let pid = e.subs.get(sid).unwrap().profile;
            let corr = last_correlation.expect("seed probe correlation");
            let result = match seed_outcome {
                0 => ProbeResult::Ok(dir_tree_snap(r, vec![])),
                1 => ProbeResult::Vanished,
                _ => ProbeResult::Failed { errno: 13 },
            };
            let _ = now;
            let out = e.step(
                Input::ProbeResponse(ProbeResponse {
                    profile: pid,
                    correlation: corr,
                    result,
                }),
                now,
            );
            prop_assert!(out.effects.is_empty(), "Seed bursts never emit Effects");
        }

        /// `prop_dirty_descendants_clamps_at_zero` — I4 on a single-Profile
        /// engine. After any sequence of inputs, the Profile's
        /// `dirty_descendants` is ≥ 0 (always 0 for a single Profile).
        #[test]
        fn prop_dirty_descendants_clamps_at_zero(
            actions in prop::collection::vec(arb_action(), 0..16),
        ) {
            let (mut e, sid, r, mut t, mut last_correlation) =
                fresh_engine_with_sub();
            let pid = e.subs.get(sid).unwrap().profile;
            for action in actions {
                let _ = run_action(&mut e, sid, r, action, &mut t, &mut last_correlation);
                if let Some(p) = e.profiles.get(pid) {
                    // u32 is ≥ 0 by type. Confirm anyway and tighten:
                    // single Profile with no parent edges → dirty_descendants
                    // is always 0.
                    prop_assert_eq!(p.dirty_descendants, 0);
                }
            }
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
        }
    }

    /// Reference-only: avoid an "unused field" warning for `BurstIntent`.
    const _: BurstIntent = BurstIntent::Standard;
}
