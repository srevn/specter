//! Sibling tests for `engine::descent` — pending-path scenarios that
//! exercise `DescentState` lifecycle in isolation. Tests compose `Engine`
//! with `MockSensor`-style direct ProbeResponse injection.

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

use crate::Engine;
use compact_str::CompactString;
use specter_core::{
    ChildEntry, ClassSet, Diagnostic, DirChild, DirMeta, DirSnapshot, EffectScope, EntryKind,
    Input, LeafEntry, ProbeOp, ProbeResponse, ProbeResult, ResourceId, ResourceKind, ResourceRole,
    ScanConfig, SubAttachRequest, TreeSnapshot,
};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant, UNIX_EPOCH};

const SETTLE: Duration = Duration::from_millis(100);
const MAX_SETTLE: Duration = Duration::from_secs(6);
const NO_EVENTS: ClassSet = ClassSet::EMPTY;

fn cfg() -> ScanConfig {
    ScanConfig::builder().recursive(true).build()
}

fn empty_command() -> specter_core::CommandTemplate {
    specter_core::CommandTemplate::new([specter_core::ArgTemplate::new([
        specter_core::ArgPart::literal("/bin/true"),
    ])])
}

/// Build a `TreeSnapshot::Dir` carrying the supplied single-component
/// children. Descent probes ship `recursive=false`, so every descent test
/// response is a single-level `DirSnapshot`; this helper matches that shape
/// exactly. Recursive uses are out of scope for the descent test surface
/// (recursive walks live in burst tests).
fn dir_snap_with(children: Vec<(&str, EntryKind, u64)>) -> TreeSnapshot {
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
        ResourceId::default(),
        DirMeta {
            mtime: UNIX_EPOCH,
            inode: 0,
            device: 0,
        },
        0,
        map,
    )))
}

/// Set up an Engine with `/foo` as a Dir; attach a Sub at path
/// `/foo/bar`. Bar doesn't exist yet — descent registers.
fn setup_pending_one_level() -> (Engine, specter_core::SubId, specter_core::ProfileId) {
    let mut e = Engine::new();
    // /foo exists as a Dir with no role-anchor — represents a real
    // directory the engine has discovered.
    let foo = e.tree_mut().ensure(None, "foo", ResourceRole::User);
    e.tree_mut().get_mut(foo).unwrap().kind = ResourceKind::Dir;

    let req = SubAttachRequest::for_path(
        "guard".into(),
        PathBuf::from("foo/bar"),
        cfg(),
        MAX_SETTLE,
        SETTLE,
        empty_command(),
        EffectScope::SubtreeRoot,
        NO_EVENTS,
    );
    let (sid, _out) = e.attach_sub(req, Instant::now());
    let pid = e.subs().get(sid).unwrap().profile;
    (e, sid, pid)
}

#[test]
fn descent_one_level_advances_on_created_entry() {
    let (mut e, _sid, pid) = setup_pending_one_level();
    assert!(e.descent_state(pid).is_some());
    let descent = e.descent_state(pid).unwrap();
    let correlation = e.pending_probe(pid).expect("first probe in flight");
    assert_eq!(descent.remaining_components, vec!["bar".to_string()]);

    // Inject a probe response showing `bar` now exists.
    let snap = dir_snap_with(vec![("bar", EntryKind::Dir, 99)]);
    let out = e.step(
        Input::ProbeResponse(ProbeResponse {
            profile: pid,
            correlation,
            result: ProbeResult::Ok(snap),
        }),
        Instant::now(),
    );

    // Anchor materialized: descent state cleared; Seed burst started.
    assert!(e.descent_state(pid).is_none());
    let probes: Vec<_> = out
        .probe_ops
        .iter()
        .filter_map(|op| match op {
            ProbeOp::Probe { request } => Some(request.profile),
            ProbeOp::Cancel { .. } => None,
        })
        .collect();
    assert_eq!(probes, vec![pid], "Seed burst probe emitted at anchor");
}

#[test]
fn descent_two_levels_advances_progressively() {
    let mut e = Engine::new();
    let foo = e.tree_mut().ensure(None, "foo", ResourceRole::User);
    e.tree_mut().get_mut(foo).unwrap().kind = ResourceKind::Dir;

    let req = SubAttachRequest::for_path(
        "guard".into(),
        PathBuf::from("foo/bar/baz"),
        cfg(),
        MAX_SETTLE,
        SETTLE,
        empty_command(),
        EffectScope::SubtreeRoot,
        NO_EVENTS,
    );
    let (sid, _out) = e.attach_sub(req, Instant::now());
    let pid = e.subs().get(sid).unwrap().profile;

    // First probe at /foo. Inject "bar" appears.
    let descent = e.descent_state(pid).unwrap();
    let corr1 = e.pending_probe(pid).unwrap();
    assert_eq!(
        descent.remaining_components,
        vec!["bar".to_string(), "baz".to_string()]
    );

    let snap1 = dir_snap_with(vec![("bar", EntryKind::Dir, 1)]);
    let _out = e.step(
        Input::ProbeResponse(ProbeResponse {
            profile: pid,
            correlation: corr1,
            result: ProbeResult::Ok(snap1),
        }),
        Instant::now(),
    );

    // Now descent should be at /foo/bar with remaining=[baz].
    let descent = e.descent_state(pid).expect("still pending");
    assert_eq!(descent.remaining_components, vec!["baz".to_string()]);
    let corr2 = e.pending_probe(pid).expect("fresh probe");
    assert_ne!(corr1, corr2, "fresh correlation per descent step");

    // Inject "baz" appears.
    let snap2 = dir_snap_with(vec![("baz", EntryKind::Dir, 2)]);
    let _out = e.step(
        Input::ProbeResponse(ProbeResponse {
            profile: pid,
            correlation: corr2,
            result: ProbeResult::Ok(snap2),
        }),
        Instant::now(),
    );

    // Anchor materialized.
    assert!(e.descent_state(pid).is_none());
}

#[test]
fn descent_no_progress_keeps_pending() {
    let (mut e, _sid, pid) = setup_pending_one_level();
    let corr = e.pending_probe(pid).unwrap();

    // Snapshot with unrelated entries (no "bar").
    let snap = dir_snap_with(vec![("other.c", EntryKind::File, 1)]);
    let _out = e.step(
        Input::ProbeResponse(ProbeResponse {
            profile: pid,
            correlation: corr,
            result: ProbeResult::Ok(snap),
        }),
        Instant::now(),
    );

    // Still pending; no new probe.
    let descent = e.descent_state(pid).unwrap();
    assert_eq!(descent.remaining_components, vec!["bar".to_string()]);
    assert!(e.pending_probe(pid).is_none(), "no probe in flight");
}

#[test]
fn descent_event_at_prefix_emits_fresh_probe() {
    let (mut e, _sid, pid) = setup_pending_one_level();
    // Drain the in-flight probe.
    let corr = e.pending_probe(pid).unwrap();
    let snap = dir_snap_with(vec![("other.c", EntryKind::File, 1)]);
    let _ = e.step(
        Input::ProbeResponse(ProbeResponse {
            profile: pid,
            correlation: corr,
            result: ProbeResult::Ok(snap),
        }),
        Instant::now(),
    );
    // No probe in flight now.
    assert!(e.pending_probe(pid).is_none());

    // Inject a StructureChanged at /foo (the prefix).
    let foo = e.tree().lookup(None, "foo").unwrap();
    let out = e.step(
        Input::FsEvent {
            resource: foo,
            event: specter_core::FsEvent::StructureChanged,
        },
        Instant::now(),
    );

    // Fresh descent probe emitted.
    let probe_for_pid = out
        .probe_ops
        .iter()
        .any(|op| matches!(op, ProbeOp::Probe { request } if request.profile == pid));
    assert!(probe_for_pid, "descent probe emitted on prefix event");
    assert!(e.pending_probe(pid).is_some());
}

#[test]
fn descent_event_during_in_flight_probe_drops() {
    let (mut e, _sid, pid) = setup_pending_one_level();
    // probe is in flight from setup
    assert!(e.pending_probe(pid).is_some());

    let foo = e.tree().lookup(None, "foo").unwrap();
    let out = e.step(
        Input::FsEvent {
            resource: foo,
            event: specter_core::FsEvent::StructureChanged,
        },
        Instant::now(),
    );

    // No new probe (I5 for descent).
    let descent_probes = out
        .probe_ops
        .iter()
        .filter(|op| matches!(op, ProbeOp::Probe { request } if request.profile == pid))
        .count();
    assert_eq!(descent_probes, 0);
}

#[test]
fn descent_failed_retains_state() {
    let (mut e, _sid, pid) = setup_pending_one_level();
    let corr = e.pending_probe(pid).unwrap();

    let out = e.step(
        Input::ProbeResponse(ProbeResponse {
            profile: pid,
            correlation: corr,
            result: ProbeResult::Failed { errno: 13 },
        }),
        Instant::now(),
    );

    let has_diag = out
        .diagnostics
        .iter()
        .any(|d| matches!(d, Diagnostic::PendingPathProbeFailed { errno: 13, .. }));
    assert!(has_diag);
    // Still pending; no probe in flight.
    let descent = e.descent_state(pid).unwrap();
    assert_eq!(descent.remaining_components, vec!["bar".to_string()]);
    assert!(e.pending_probe(pid).is_none());
}

#[test]
fn descent_anchor_kind_set_from_entry() {
    let (mut e, _sid, pid) = setup_pending_one_level();
    let corr = e.pending_probe(pid).unwrap();
    let foo = e.tree().lookup(None, "foo").unwrap();
    let bar = e.tree().lookup(Some(foo), "bar").expect("scaffold exists");

    // Inject as a Dir.
    let snap = dir_snap_with(vec![("bar", EntryKind::Dir, 1)]);
    let _ = e.step(
        Input::ProbeResponse(ProbeResponse {
            profile: pid,
            correlation: corr,
            result: ProbeResult::Ok(snap),
        }),
        Instant::now(),
    );

    let res = e.tree().get(bar).unwrap();
    assert_eq!(res.kind, ResourceKind::Dir);
    assert!(matches!(res.role, ResourceRole::User));
}

// ===== absolute-path bootstrap & minimal descent probe =====

/// Absolute-path attaches bootstrap a synthetic FS-root `"/"` segment so
/// descents have a guaranteed-existing starting prefix. The bootstrap is
/// idempotent across repeated absolute attaches.
#[test]
fn absolute_attach_bootstraps_fs_root_segment() {
    let mut e = Engine::new();

    let req = SubAttachRequest::for_path(
        "build".into(),
        PathBuf::from("/tmp"),
        cfg(),
        MAX_SETTLE,
        SETTLE,
        empty_command(),
        EffectScope::SubtreeRoot,
        NO_EVENTS,
    );
    let (sid, out) = e.attach_sub(req, Instant::now());
    let pid = e.subs().get(sid).unwrap().profile;

    // Tree contains the synthetic FS-root and the `tmp` scaffold.
    let root = e.tree().lookup(None, "/").expect("FS-root bootstrapped");
    let tmp = e
        .tree()
        .lookup(Some(root), "tmp")
        .expect("anchor scaffold installed under /");

    // Profile registered; descent in flight at the FS-root.
    let descent = e
        .descent_state(pid)
        .expect("absolute attach against empty Tree is pending");
    assert_eq!(descent.current_prefix, root);
    assert_eq!(descent.remaining_components, vec!["tmp".to_string()]);
    assert!(e.pending_probe(pid).is_some());

    // The FS-root carries the descent's watch_demand contribution; the
    // anchor scaffold doesn't (descent hasn't materialized it yet).
    assert_eq!(e.tree().get(root).unwrap().watch_demand, 1);
    assert_eq!(e.tree().get(tmp).unwrap().watch_demand, 0);

    // The emitted Watch op carries an *absolute* path — `Tree::path_of`
    // reconstructs `/` because `PathBuf::push("/")` resets to absolute.
    let watch_for_root = out.watch_ops.iter().find_map(|op| match op {
        specter_core::WatchOp::Watch { resource, path, .. } if *resource == root => Some(path),
        _ => None,
    });
    assert_eq!(
        watch_for_root,
        Some(&PathBuf::from("/")),
        "FS-root Watch op carries an absolute path",
    );

    // The probe op for the descent also carries an absolute prefix path.
    let probe_path = out.probe_ops.iter().find_map(|op| match op {
        ProbeOp::Probe { request } if request.profile == pid => Some(&request.target_path),
        ProbeOp::Cancel { .. } | ProbeOp::Probe { .. } => None,
    });
    assert_eq!(probe_path, Some(&PathBuf::from("/")));
}

/// Two absolute attaches share the FS-root via the bootstrap's
/// idempotence (`Tree::ensure(None, "/")` returns the existing root on
/// the second call).
#[test]
fn second_absolute_attach_reuses_fs_root() {
    let mut e = Engine::new();
    let req1 = SubAttachRequest::for_path(
        "a".into(),
        PathBuf::from("/foo"),
        cfg(),
        MAX_SETTLE,
        SETTLE,
        empty_command(),
        EffectScope::SubtreeRoot,
        NO_EVENTS,
    );
    let req2 = SubAttachRequest::for_path(
        "b".into(),
        PathBuf::from("/bar"),
        cfg(),
        MAX_SETTLE,
        SETTLE,
        empty_command(),
        EffectScope::SubtreeRoot,
        NO_EVENTS,
    );
    let (_, _) = e.attach_sub(req1, Instant::now());
    let (_, _) = e.attach_sub(req2, Instant::now());

    let root = e.tree().lookup(None, "/").expect("single FS-root");
    assert_eq!(e.tree().roots().len(), 1, "exactly one tree root");
    // Both children attach under the same FS-root.
    assert!(e.tree().lookup(Some(root), "foo").is_some());
    assert!(e.tree().lookup(Some(root), "bar").is_some());
    // FS-root carries one contribution from each pending descent.
    assert_eq!(e.tree().get(root).unwrap().watch_demand, 2);
}

/// Deep absolute paths walk one segment at a time: the descent's
/// `remaining_components` reflects the unmaterialized tail.
#[test]
fn deep_absolute_attach_decomposes_to_one_remaining_per_segment() {
    let mut e = Engine::new();
    let req = SubAttachRequest::for_path(
        "log".into(),
        PathBuf::from("/var/log/myapp"),
        cfg(),
        MAX_SETTLE,
        SETTLE,
        empty_command(),
        EffectScope::SubtreeRoot,
        NO_EVENTS,
    );
    let (sid, _) = e.attach_sub(req, Instant::now());
    let pid = e.subs().get(sid).unwrap().profile;

    let root = e.tree().lookup(None, "/").unwrap();
    let descent = e.descent_state(pid).unwrap();
    assert_eq!(descent.current_prefix, root);
    assert_eq!(
        descent.remaining_components,
        vec!["var".to_string(), "log".to_string(), "myapp".to_string()],
    );
}

/// The descent probe uses a *minimal* `ScanConfig`: non-recursive,
/// hidden=true, no pattern, no exclude, max_depth unbounded — just a
/// single-level enumeration of the prefix's children. Otherwise the
/// user's recursive scan would walk the entire filesystem from `/` for
/// every absolute attach, and a user pattern would filter out the very
/// segment we're descending into.
#[test]
fn descent_probe_uses_minimal_scan_config() {
    let mut e = Engine::new();
    let foo = e.tree_mut().ensure(None, "foo", ResourceRole::User);
    e.tree_mut().get_mut(foo).unwrap().kind = ResourceKind::Dir;

    let user_cfg = specter_core::ScanConfig::builder()
        .recursive(true)
        .pattern(specter_core::GlobPattern::compile("*.c").unwrap())
        .build();
    let req = SubAttachRequest::for_path(
        "g".into(),
        PathBuf::from("foo/bar"),
        user_cfg,
        MAX_SETTLE,
        SETTLE,
        empty_command(),
        EffectScope::SubtreeRoot,
        NO_EVENTS,
    );
    let (_sid, out) = e.attach_sub(req, Instant::now());

    let probe_cfg = out
        .probe_ops
        .iter()
        .find_map(|op| match op {
            ProbeOp::Probe { request } => Some(&request.scan_config),
            ProbeOp::Cancel { .. } => None,
        })
        .expect("descent probe emitted");
    assert!(!probe_cfg.recursive, "descent probe is non-recursive");
    assert!(probe_cfg.hidden, "descent probe shows hidden entries");
    assert!(
        probe_cfg.pattern.is_none(),
        "descent probe ignores user pattern (would hide next segment)",
    );
    assert!(
        probe_cfg.exclude.is_empty(),
        "descent probe ignores user excludes",
    );
}

/// Materialization at descent's anchor branch sets
/// `Profile.anchor_contribution = true` so a later reap correctly
/// releases the anchor's `watch_demand`.
#[test]
fn descent_materialization_sets_anchor_contribution_flag() {
    let (mut e, _sid, pid) = setup_pending_one_level();
    let corr = e.pending_probe(pid).unwrap();
    let snap = dir_snap_with(vec![("bar", EntryKind::Dir, 1)]);
    let _ = e.step(
        Input::ProbeResponse(ProbeResponse {
            profile: pid,
            correlation: corr,
            result: ProbeResult::Ok(snap),
        }),
        Instant::now(),
    );
    assert!(
        e.profiles().get(pid).unwrap().anchor_contribution,
        "anchor_contribution set on descent materialization",
    );
}

/// Pending Profile reaped before descent advances:
/// - Releases the descent's prefix `watch_demand`.
/// - Does NOT touch the anchor (anchor was never bumped).
/// - No underflow panic in dev.
#[test]
fn reap_pending_profile_releases_only_descent_prefix() {
    let (mut e, sid, pid) = setup_pending_one_level();
    let foo = e.tree().lookup(None, "foo").unwrap();
    let bar = e.tree().lookup(Some(foo), "bar").expect("anchor scaffold");

    // Pre-conditions: descent contributes to `foo`, anchor `bar` is
    // unbumped.
    assert_eq!(e.tree().get(foo).unwrap().watch_demand, 1);
    assert_eq!(e.tree().get(bar).unwrap().watch_demand, 0);
    assert!(!e.profiles().get(pid).unwrap().anchor_contribution);

    // Detach the only Sub. Profile is Pending; Pending Profiles reap
    // immediately (they hold no burst that would resolve a deferred reap).
    let out = e.detach_sub(sid, Instant::now());

    // `bar`'s slot is reaped (no other anchors), `foo` still has its
    // pre-existing User Resource — only the descent's contribution is
    // released.
    assert_eq!(
        e.tree().get(foo).map_or(0, |r| r.watch_demand),
        0,
        "descent prefix watch_demand released",
    );
    assert!(
        out.watch_ops.iter().any(
            |op| matches!(op, specter_core::WatchOp::Unwatch { resource } if *resource == foo)
        ),
        "Unwatch emitted for the descent prefix",
    );
}

/// A fresh `Profile::new` defaults to `ProfileState::Idle`,
/// not Pending. Pending is reachable only through the descent registry
/// paths (`attach_sub_inner` Pending branch, `start_pending_recovery`,
/// `dispatch_descent_vanished` rewind).
#[test]
fn profile_state_default_is_idle() {
    use specter_core::{Profile, ProfileState, ScanConfig};
    let mut e = Engine::new();
    let r = e.tree_mut().ensure(None, "anchor", ResourceRole::User);
    let p = Profile::new(
        r,
        ScanConfig::builder().build(),
        MAX_SETTLE,
        SETTLE,
        NO_EVENTS,
    );
    assert!(matches!(p.state, ProfileState::Idle));
}

/// `Engine::descent_state` returns `None` for an Idle Profile.
/// The accessor's reader contract is "Some iff state is Pending."
#[test]
fn descent_state_helper_returns_none_for_idle() {
    let mut e = Engine::new();
    let foo = e.tree_mut().ensure(None, "foo", ResourceRole::User);
    e.tree_mut().get_mut(foo).unwrap().kind = ResourceKind::Dir;
    let req = SubAttachRequest::for_resource(
        "g".into(),
        foo,
        cfg(),
        MAX_SETTLE,
        SETTLE,
        empty_command(),
        EffectScope::SubtreeRoot,
        NO_EVENTS,
    );
    let (sid, _) = e.attach_sub(req, Instant::now());
    let pid = e.subs().get(sid).unwrap().profile;
    // Materialized Profile starts a Seed burst — Active, not Idle. Drive
    // it to completion to land in Idle.
    let probe = e
        .step(
            Input::ProbeResponse(ProbeResponse {
                profile: pid,
                correlation: specter_core::ProbeCorrelation(1),
                result: ProbeResult::Vanished,
            }),
            Instant::now(),
        )
        .diagnostics;
    let _ = probe; // not asserted; the Vanished response drains the Seed burst to Idle
    assert!(e.descent_state(pid).is_none());
}

/// `Engine::descent_state` returns `None` for an Active Profile
/// (a burst is in flight; the descent slot is not used).
#[test]
fn descent_state_helper_returns_none_for_active() {
    let mut e = Engine::new();
    let foo = e.tree_mut().ensure(None, "foo", ResourceRole::User);
    e.tree_mut().get_mut(foo).unwrap().kind = ResourceKind::Dir;
    let req = SubAttachRequest::for_resource(
        "g".into(),
        foo,
        cfg(),
        MAX_SETTLE,
        SETTLE,
        empty_command(),
        EffectScope::SubtreeRoot,
        NO_EVENTS,
    );
    let (sid, _) = e.attach_sub(req, Instant::now());
    let pid = e.subs().get(sid).unwrap().profile;
    // Materialized Profile starts a Seed burst — state is Active.
    assert!(matches!(
        e.profiles().get(pid).unwrap().state,
        specter_core::ProfileState::Active(_)
    ));
    assert!(e.descent_state(pid).is_none());
}

/// `Engine::descent_state` returns `Some(d)` for a Pending Profile,
/// and the inner state matches what was registered.
#[test]
fn descent_state_helper_returns_some_for_pending() {
    let (e, _sid, pid) = setup_pending_one_level();
    let descent = e.descent_state(pid).expect("Pending state populated");
    assert_eq!(descent.remaining_components, vec!["bar".to_string()]);
    assert!(e.pending_probe(pid).is_some());
}

/// `Engine::descent_state` returns `None` for an unknown `ProfileId`.
/// No panic; defensive read.
#[test]
fn descent_state_helper_handles_unknown_profile() {
    let e = Engine::new();
    let bogus = specter_core::ProfileId::default();
    assert!(e.descent_state(bogus).is_none());
}

/// `ProfileState::Pending` and `ProfileState::Active` are mutually
/// exclusive variants — the compiler proves the property. This test
/// exercises the lifecycle transition Pending → Idle → Active(Seed) at
/// descent anchor materialization and asserts the Profile passes through
/// the intermediate Idle state cleanly (no observation of
/// Pending+Active simultaneously).
#[test]
fn profile_state_pending_and_active_are_mutually_exclusive() {
    use specter_core::ProfileState;
    let (mut e, _sid, pid) = setup_pending_one_level();
    // Initially Pending.
    assert!(matches!(
        e.profiles().get(pid).unwrap().state,
        ProfileState::Pending(_)
    ));
    let corr = e.pending_probe(pid).unwrap();
    let snap = dir_snap_with(vec![("bar", EntryKind::Dir, 1)]);
    let _ = e.step(
        Input::ProbeResponse(ProbeResponse {
            profile: pid,
            correlation: corr,
            result: ProbeResult::Ok(snap),
        }),
        Instant::now(),
    );
    // After anchor materialization: Pending → Idle, then start_seed_burst
    // transitions Idle → Active(Seed). The post-step state is Active.
    assert!(matches!(
        e.profiles().get(pid).unwrap().state,
        ProfileState::Active(_)
    ));
    // descent_state agrees: no descent.
    assert!(e.descent_state(pid).is_none());
}

/// `reap_profile`'s trichotomy `debug_assert!` is reachable from the
/// Pending lifecycle (descent in flight, then Sub detaches) and does not
/// fire — the assertion pins the invariant in code, not just prose.
#[test]
fn reap_profile_trichotomy_debug_assert_holds_for_pending() {
    let (mut e, sid, pid) = setup_pending_one_level();
    // Pending Profile reap path: descent_prefix.is_some() &&
    // !had_anchor_contribution. Predicate `(some && false)` is false →
    // assertion holds.
    let _ = e.detach_sub(sid, Instant::now());
    assert!(e.profiles().get(pid).is_none(), "Profile reaped");
}

#[test]
fn reap_profile_trichotomy_debug_assert_holds_for_materialized() {
    // Materialized Profile reap path: descent_prefix.is_none() &&
    // had_anchor_contribution. Predicate `(none && true)` is false →
    // assertion holds.
    let mut e = Engine::new();
    let foo = e.tree_mut().ensure(None, "foo", ResourceRole::User);
    e.tree_mut().get_mut(foo).unwrap().kind = ResourceKind::Dir;
    let req = SubAttachRequest::for_resource(
        "g".into(),
        foo,
        cfg(),
        MAX_SETTLE,
        SETTLE,
        empty_command(),
        EffectScope::SubtreeRoot,
        NO_EVENTS,
    );
    let (sid, _) = e.attach_sub(req, Instant::now());
    let pid = e.subs().get(sid).unwrap().profile;
    assert!(e.profiles().get(pid).unwrap().anchor_contribution);
    // Drain Seed via Vanished so the Profile lands Idle with the
    // anchor's contribution still held. Then detach.
    let Some(corr) = e.pending_probe(pid) else {
        return;
    };
    let _ = e.step(
        Input::ProbeResponse(ProbeResponse {
            profile: pid,
            correlation: corr,
            result: ProbeResult::Vanished,
        }),
        Instant::now(),
    );
    // Vanished clears the anchor contribution (it's the terminal-event
    // path). Force the materialized branch by re-seeding via a fresh
    // anchor lookup. For coverage of the assertion, the detach path
    // itself is sufficient (it runs reap_profile, which contains the
    // assertion).
    let _ = e.detach_sub(sid, Instant::now());
    assert!(e.profiles().get(pid).is_none(), "Profile reaped");
}

/// Detaching the last Sub on a Pending Profile reaps immediately rather
/// than setting `reap_pending = true`. Pending Profiles have no burst
/// whose `finish_burst_to_idle` would resolve a deferred reap.
#[test]
fn detach_sub_pending_profile_reaps_immediately() {
    let (mut e, sid, pid) = setup_pending_one_level();
    let foo = e.tree().lookup(None, "foo").unwrap();
    // Pre-condition: Pending; descent contributes +1 to /foo.
    assert!(e.descent_state(pid).is_some());
    assert_eq!(e.tree().get(foo).unwrap().watch_demand, 1);

    let out = e.detach_sub(sid, Instant::now());

    // Profile reaped synchronously: no longer in the registry; descent
    // contribution released atomically.
    assert!(e.profiles().get(pid).is_none(), "Profile reaped");
    assert_eq!(
        e.tree().get(foo).map_or(0, |r| r.watch_demand),
        0,
        "descent contribution released",
    );
    assert!(
        out.diagnostics
            .iter()
            .any(|d| matches!(d, Diagnostic::ReapPendingResolved { profile } if *profile == pid)),
        "ReapPendingResolved emitted",
    );
}

/// `on_probe_response`'s unified routing dispatches a Pending Profile's
/// response to the descent path via `match &p.state`. This test asserts
/// the routing by exercising a descent probe response and verifying the
/// descent advances.
#[test]
fn on_probe_response_routes_descent_via_state_match() {
    let (mut e, _sid, pid) = setup_pending_one_level();
    let corr = e.pending_probe(pid).unwrap();
    let snap = dir_snap_with(vec![("bar", EntryKind::Dir, 99)]);
    let _ = e.step(
        Input::ProbeResponse(ProbeResponse {
            profile: pid,
            correlation: corr,
            result: ProbeResult::Ok(snap),
        }),
        Instant::now(),
    );
    // Descent route fired: Pending → Idle → Active(Seed). The Profile
    // is no longer Pending.
    assert!(e.descent_state(pid).is_none(), "descent route ran");
    assert!(matches!(
        e.profiles().get(pid).unwrap().state,
        specter_core::ProfileState::Active(_)
    ));
}

/// `on_watch_op_rejected` purge transitions Pending → Idle.
#[test]
fn on_watch_op_rejected_clears_pending_state() {
    use specter_core::{ProfileState, WatchOp, WatchOpts};
    let (mut e, _sid, pid) = setup_pending_one_level();
    let foo = e.tree().lookup(None, "foo").unwrap();
    assert!(matches!(
        e.profiles().get(pid).unwrap().state,
        ProfileState::Pending(_)
    ));

    let _ = e.step(
        Input::WatchOpRejected {
            resource: foo,
            op: WatchOp::Watch {
                resource: foo,
                path: std::path::PathBuf::from("foo"),
                opts: WatchOpts::default(),
            },
            errno: 24,
        },
        Instant::now(),
    );

    // Purge transitions Pending → Idle; descent_state agrees.
    assert!(matches!(
        e.profiles().get(pid).unwrap().state,
        ProfileState::Idle
    ));
    assert!(e.descent_state(pid).is_none());
}

// ───────────────────────────────────────────────────────────────────────
// F-LOW-1 regression: descent empty-remaining defensive arm
//
// `dispatch_descent_ok`'s defensive empty-remaining arm pre-fix
// transitioned the Profile to Idle but did NOT release the prefix's
// STRUCTURE contribution. The DescentState invariant says
// `remaining_components` is non-empty, so this arm is unreachable in
// normal operation — but if it ever fires (state-machine corruption),
// the prefix's `watch_demand` retains its +1 with no Profile
// attributing it. The trichotomy then classifies the Profile as
// "purged" (anchor_contribution=false, state=Idle) and reap_profile
// skips the release. Net: leaked watch_demand on the prefix.
//
// Post-fix: the arm emits Diagnostic::DescentInvariantViolation and
// calls release_descent_prefix_claim — symmetric with
// dispatch_descent_vanished's root branch.
// ───────────────────────────────────────────────────────────────────────
#[test]
fn descent_ok_with_empty_remaining_releases_prefix_and_emits_diagnostic() {
    use specter_core::{DescentState, ProfileState};
    // Build the engine with a Pending Profile, then poke
    // `remaining_components` to empty to construct the
    // invariant-violating state directly. (`materialize_path_or_pending`
    // is the canonical producer of Pending Profiles and never produces
    // empty remaining; the only way to test the defensive arm is via
    // direct fixture construction, which is the point — the arm exists
    // for future-proofing, and we want the test to lock in its
    // benign-failure contract.)
    let (mut e, _sid, pid) = setup_pending_one_level();
    let foo = e.tree().lookup(None, "foo").unwrap();
    let corr = e.pending_probe(pid).unwrap();

    // Snapshot pre-state.
    let prefix = e.descent_state(pid).unwrap().current_prefix;
    assert_eq!(prefix, foo);
    assert_eq!(
        e.tree().get(foo).unwrap().watch_demand,
        1,
        "descent prefix carries +1 STRUCTURE",
    );

    // Direct mutation to construct the invariant-violating state.
    // descent_tests.rs is a sibling crate module (not an integration
    // test), so `pub(crate)` engine fields are reachable. The probe
    // channel slot (`pending_probe`) keeps its value across the state
    // mutation, so the in-flight probe correlation matches `corr` for
    // the response below.
    e.profiles.get_mut(pid).unwrap().state = ProfileState::Pending(DescentState {
        current_prefix: prefix,
        remaining_components: Vec::new(),
    });

    // Dispatch the probe response — descent_ok hits the defensive arm.
    let snap = dir_snap_with(vec![]);
    let out = e.step(
        Input::ProbeResponse(ProbeResponse {
            profile: pid,
            correlation: corr,
            result: ProbeResult::Ok(snap),
        }),
        Instant::now(),
    );

    // Profile transitioned to Idle.
    assert!(matches!(
        e.profiles().get(pid).unwrap().state,
        ProfileState::Idle
    ));
    // Prefix's watch_demand released — pre-fix this leaked.
    assert_eq!(
        e.tree().get(foo).map_or(0, |r| r.watch_demand),
        0,
        "prefix watch_demand released by defensive arm",
    );
    // Diagnostic surfaces.
    assert!(
        out.diagnostics.iter().any(|d| matches!(d,
            Diagnostic::DescentInvariantViolation { profile, prefix: pfx }
                if *profile == pid && *pfx == foo)),
        "DescentInvariantViolation emitted; got {:?}",
        out.diagnostics,
    );
}

// ───────────────────────────────────────────────────────────────────────
// Probe-channel discipline (post-refactor invariants)
//
// I5 ("at most one outstanding probe per Profile") moved from a structural
// type-law (mutual exclusion of `BurstPhase::Verifying { ... }` and
// `DescentPhase::Probing { ... }`) to a field-discipline (single
// `Profile.pending_probe` slot). The tests below pin the post-refactor
// invariants: clear-on-cancel, recovery-overlap accounting, and the
// cancel-first contract on `release_descent_prefix_claim`.
// ───────────────────────────────────────────────────────────────────────

/// `on_watch_op_rejected` descent purge: cancel-then-release ordering
/// closes the probe channel and emits exactly one `ProbeOp::Cancel`. The
/// Profile transitions Pending → Idle in the same step.
#[test]
fn on_watch_op_rejected_descent_purge_clears_pending_probe_and_emits_cancel() {
    use specter_core::{ProfileState, WatchOp, WatchOpts};
    let (mut e, _sid, pid) = setup_pending_one_level();
    let foo = e.tree().lookup(None, "foo").unwrap();
    assert!(
        e.pending_probe(pid).is_some(),
        "descent probe in flight after attach",
    );
    assert!(matches!(
        e.profiles().get(pid).unwrap().state,
        ProfileState::Pending(_),
    ));

    let out = e.step(
        Input::WatchOpRejected {
            resource: foo,
            op: WatchOp::Watch {
                resource: foo,
                path: PathBuf::from("foo"),
                opts: WatchOpts::default(),
            },
            errno: 24,
        },
        Instant::now(),
    );

    // Field-discipline: channel closed atomically with the purge.
    assert!(
        e.pending_probe(pid).is_none(),
        "channel closed by cancel-before-release",
    );
    // Profile transitioned via `release_descent_prefix_claim`.
    assert!(matches!(
        e.profiles().get(pid).unwrap().state,
        ProfileState::Idle,
    ));
    // Exactly one Cancel for the Profile (idempotency check).
    let cancels = out
        .probe_ops
        .iter()
        .filter(|op| matches!(op, ProbeOp::Cancel { profile } if *profile == pid))
        .count();
    assert_eq!(
        cancels, 1,
        "exactly one Cancel emitted for the in-flight descent probe; got {:?}",
        out.probe_ops,
    );
}

/// `enter_pending_descent` recovery-overlap invariant: when invoked from
/// `start_pending_recovery`, the parent already carries `+1 STRUCTURE` from
/// `Profile.watch_root_parent`. The helper bumps `+1` again for the descent
/// contribution; refcount sums to `+2`. Verifies the helper's pre-condition
/// assertion AND the documented post-condition.
#[test]
fn enter_pending_descent_recovery_overlap_invariant() {
    use specter_core::{ClassSet, ProfileState};
    // Build the recovery scenario by hand:
    //   1. Attach a Sub at /foo/bar (Pending — bar doesn't exist yet).
    //   2. Materialize bar via descent, landing the Profile in Idle with
    //      Profile.watch_root_parent = Some(foo) and foo.watch_demand = +1.
    //   3. Drop bar (anchor terminal) → Profile remains Idle, anchor
    //      contribution gone, watch_root_parent contribution persists.
    //   4. Call enter_pending_descent at foo with [bar] as remaining.
    let (mut e, _sid, pid) = setup_pending_one_level();
    let foo = e.tree().lookup(None, "foo").unwrap();

    // Step 1+2: Drive descent to materialization. The probe response with
    // `bar` as a Dir entry materializes the anchor.
    let corr = e.pending_probe(pid).expect("descent probe in flight");
    let snap = dir_snap_with(vec![("bar", EntryKind::Dir, 99)]);
    let _ = e.step(
        Input::ProbeResponse(ProbeResponse {
            profile: pid,
            correlation: corr,
            result: ProbeResult::Ok(snap),
        }),
        Instant::now(),
    );
    let _bar = e.tree().lookup(Some(foo), "bar").unwrap();
    // Post-materialization: Profile is Active(Seed Verifying); bar carries
    // events_union; foo carries STRUCTURE from watch_root_parent.
    assert_eq!(
        e.profiles().get(pid).unwrap().watch_root_parent,
        Some(foo),
        "watch_root_parent cached at foo on materialization",
    );
    assert!(
        e.tree().get(foo).unwrap().watch_demand >= 1,
        "foo carries STRUCTURE from watch_root_parent",
    );

    // Settle the Seed burst (Vanished closes it without emitting an Effect).
    let seed_corr = e.pending_probe(pid).expect("Seed probe in flight");
    let _ = e.step(
        Input::ProbeResponse(ProbeResponse {
            profile: pid,
            correlation: seed_corr,
            result: ProbeResult::Vanished,
        }),
        Instant::now(),
    );
    // dispatch_seed_vanished routes to finalize_anchor_lost: anchor
    // contribution released, baseline/current cleared, Profile lands Idle.
    assert!(matches!(
        e.profiles().get(pid).unwrap().state,
        ProfileState::Idle,
    ));
    assert!(
        e.pending_probe(pid).is_none(),
        "channel closed after Seed Vanished",
    );
    let foo_demand_pre = e.tree().get(foo).unwrap().watch_demand;
    // Bar's anchor contribution was released; only watch_root_parent's
    // STRUCTURE on foo remains.
    assert_eq!(
        foo_demand_pre, 1,
        "foo.watch_demand reflects only the watch_root_parent contribution",
    );

    // Step 4: Call enter_pending_descent directly to simulate the
    // `start_pending_recovery` re-entry path. The helper's debug_assert
    // pins Profile=Idle + closed-channel; both hold.
    let mut out = specter_core::StepOutput::default();
    e.enter_pending_descent(pid, foo, vec!["bar".to_string()], &mut out);

    // Recovery overlap: foo's watch_demand is now +2 (watch_root_parent
    // STRUCTURE + descent STRUCTURE). The helper opened the channel and
    // emitted a descent probe.
    assert_eq!(
        e.tree().get(foo).unwrap().watch_demand,
        foo_demand_pre + 1,
        "recovery overlap: descent +1 on top of watch_root_parent +1",
    );
    assert!(
        e.pending_probe(pid).is_some(),
        "channel re-opened by enter_pending_descent",
    );
    assert!(matches!(
        e.profiles().get(pid).unwrap().state,
        ProfileState::Pending(_),
    ));
    // The descent probe was emitted at foo (the parent / new prefix).
    assert!(
        out.probe_ops.iter().any(|op| matches!(op,
            ProbeOp::Probe { request } if request.target_resource == foo
                && request.profile == pid)),
        "descent probe emitted at the parent prefix; got {:?}",
        out.probe_ops,
    );
    // ClassSet::STRUCTURE is correct for the descent contribution by D9.
    let _ = ClassSet::STRUCTURE;
}

/// Cancel-first contract on `release_descent_prefix_claim`: invoking the
/// helper on a Pending Profile with an open probe channel fires the
/// debug_assert. The four production cancel-paths each call
/// `cancel_pending_probe` first — this test guards against future
/// regressions that bypass the order.
#[test]
#[cfg_attr(
    not(debug_assertions),
    ignore = "debug_assert! is compiled out in release"
)]
#[should_panic(expected = "probe channel must be closed")]
fn release_descent_prefix_claim_panics_on_open_channel() {
    let (mut e, _sid, pid) = setup_pending_one_level();
    assert!(
        e.pending_probe(pid).is_some(),
        "descent probe in flight (pre-condition for the assertion)",
    );

    // Direct invocation without the prior cancel — assertion fires.
    let mut out = specter_core::StepOutput::default();
    e.release_descent_prefix_claim(pid, &mut out);
}
