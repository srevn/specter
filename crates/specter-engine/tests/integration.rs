//! Cross-module integration tests for `specter-engine`.
//!
//! Two suites:
//! - **P3-era primitives**: `covers + StabilityIndex::compute_parent +
//!   StabilityIndex::propagate` against a real `Tree` + `ProfileMap`.
//! - **P4 lifecycle**: full `Idle ↔ Active(Burst)` flows driven through
//!   `Engine::attach_sub` and `Engine::step` against a `MockSensor`-style
//!   harness (assertions read from `StepOutput`).

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
    dead_code
)]

use compact_str::CompactString;
use specter_core::{
    ArgPart, ArgTemplate, BurstIntent, ChildEntry, ClassSet, CommandTemplate, Diagnostic, DirChild,
    DirMeta, DirSnapshot, EffectOutcome, EffectScope, EntryKind, FsEvent, Input, LeafEntry,
    Placeholder, ProbeCorrelation, ProbeOp, ProbeRequest, ProbeResponse, ProbeResult, Profile,
    ProfileMap, ResourceId, ResourceKind, ResourceRole, ScanConfig, StepOutput, Tree, TreeSnapshot,
    WatchOp,
};
use specter_engine::{Engine, StabilityIndex, SubAttachRequest, covers};
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, Instant, UNIX_EPOCH};

const SETTLE: Duration = Duration::from_millis(100);
const MAX_SETTLE: Duration = Duration::from_secs(6);
const NO_EVENTS: ClassSet = ClassSet::EMPTY;

fn cfg_recursive() -> ScanConfig {
    ScanConfig::builder().recursive(true).build()
}

fn mark_dir(tree: &mut Tree, id: ResourceId) {
    tree.get_mut(id).unwrap().kind = ResourceKind::Dir;
}

#[test]
fn engine_default_constructible() {
    let e = Engine::new();
    assert!(e.next_deadline().is_none());
}

#[test]
fn covers_drives_compute_parent() {
    // Three Resources in a chain: root → a → b. A Profile at root with
    // `recursive = false` does NOT cover b (depth > 1, recursive false);
    // a Profile at root with `recursive = true` DOES.

    // Flavor 1: root's Profile is non-recursive; b has no covering parent.
    {
        let mut tree = Tree::new();
        let mut profiles = ProfileMap::new();
        let root = tree.ensure(None, "root", ResourceRole::User);
        let a = tree.ensure(Some(root), "a", ResourceRole::User);
        let b = tree.ensure(Some(a), "b", ResourceRole::User);
        for r in [root, a, b] {
            mark_dir(&mut tree, r);
        }
        let p_root = profiles.attach(
            &mut tree,
            Profile::new(
                root,
                ScanConfig::builder().recursive(false).build(),
                MAX_SETTLE,
                SETTLE,
                NO_EVENTS,
            ),
        );
        let p_b = profiles.attach(
            &mut tree,
            Profile::new(b, cfg_recursive(), MAX_SETTLE, SETTLE, NO_EVENTS),
        );

        assert!(!covers(profiles.get(p_root).unwrap(), b, &tree));
        assert!(StabilityIndex::compute_parent(&tree, &profiles, p_b).is_none());
    }

    // Flavor 2: root's Profile is recursive; b parents to root.
    {
        let mut tree = Tree::new();
        let mut profiles = ProfileMap::new();
        let root = tree.ensure(None, "root", ResourceRole::User);
        let a = tree.ensure(Some(root), "a", ResourceRole::User);
        let b = tree.ensure(Some(a), "b", ResourceRole::User);
        for r in [root, a, b] {
            mark_dir(&mut tree, r);
        }
        let p_root = profiles.attach(
            &mut tree,
            Profile::new(root, cfg_recursive(), MAX_SETTLE, SETTLE, NO_EVENTS),
        );
        let p_b = profiles.attach(
            &mut tree,
            Profile::new(b, cfg_recursive(), MAX_SETTLE, SETTLE, NO_EVENTS),
        );

        assert!(covers(profiles.get(p_root).unwrap(), b, &tree));
        assert_eq!(
            StabilityIndex::compute_parent(&tree, &profiles, p_b),
            Some(p_root),
        );
    }
}

#[test]
fn compute_parent_then_propagate_round_trip() {
    let mut tree = Tree::new();
    let mut profiles = ProfileMap::new();
    let root = tree.ensure(None, "root", ResourceRole::User);
    let mid = tree.ensure(Some(root), "mid", ResourceRole::User);
    let leaf = tree.ensure(Some(mid), "leaf", ResourceRole::User);
    for r in [root, mid, leaf] {
        mark_dir(&mut tree, r);
    }
    let p_root = profiles.attach(
        &mut tree,
        Profile::new(root, cfg_recursive(), MAX_SETTLE, SETTLE, NO_EVENTS),
    );
    let p_mid = profiles.attach(
        &mut tree,
        Profile::new(mid, cfg_recursive(), MAX_SETTLE, SETTLE, NO_EVENTS),
    );
    let p_leaf = profiles.attach(
        &mut tree,
        Profile::new(leaf, cfg_recursive(), MAX_SETTLE, SETTLE, NO_EVENTS),
    );

    let mut idx = StabilityIndex::new();
    if let Some(parent) = StabilityIndex::compute_parent(&tree, &profiles, p_leaf) {
        idx.set_parent(p_leaf, parent);
    }
    if let Some(parent) = StabilityIndex::compute_parent(&tree, &profiles, p_mid) {
        idx.set_parent(p_mid, parent);
    }
    assert_eq!(idx.parent_of(p_leaf), Some(p_mid));
    assert_eq!(idx.parent_of(p_mid), Some(p_root));

    let _hit = idx.propagate(&mut profiles, p_leaf, 1);
    assert_eq!(profiles.get(p_mid).unwrap().dirty_descendants, 1);
    assert_eq!(profiles.get(p_root).unwrap().dirty_descendants, 1);

    let _hit = idx.propagate(&mut profiles, p_leaf, -1);
    assert_eq!(profiles.get(p_mid).unwrap().dirty_descendants, 0);
    assert_eq!(profiles.get(p_root).unwrap().dirty_descendants, 0);
}

#[test]
fn covers_handles_pattern_with_dir_bypass_in_engine_context() {
    let mut tree = Tree::new();
    let mut profiles = ProfileMap::new();
    let root = tree.ensure(None, "root", ResourceRole::User);
    let src = tree.ensure(Some(root), "src", ResourceRole::User);
    let lib_rs = tree.ensure(Some(src), "lib.rs", ResourceRole::User);
    let lib_c = tree.ensure(Some(src), "lib.c", ResourceRole::User);
    mark_dir(&mut tree, root);
    mark_dir(&mut tree, src);
    tree.get_mut(lib_rs).unwrap().kind = ResourceKind::File;
    tree.get_mut(lib_c).unwrap().kind = ResourceKind::File;

    let p = profiles.attach(
        &mut tree,
        Profile::new(
            root,
            ScanConfig::builder()
                .recursive(true)
                .pattern(specter_core::GlobPattern::compile("*.rs").unwrap())
                .build(),
            MAX_SETTLE,
            SETTLE,
            NO_EVENTS,
        ),
    );
    let profile = profiles.get(p).unwrap();

    assert!(covers(profile, src, &tree), "Dir bypasses pattern");
    assert!(covers(profile, lib_rs, &tree), "matching File covered");
    assert!(
        !covers(profile, lib_c, &tree),
        "non-matching File uncovered"
    );
}

// ---------- P4 single-Profile lifecycle scenarios ----------

/// Pluck the correlation from the Probe (if any) in a `StepOutput`.
fn first_probe_correlation(out: &StepOutput) -> Option<ProbeCorrelation> {
    out.probe_ops.iter().find_map(|op| match op {
        ProbeOp::Probe {
            request: ProbeRequest { correlation, .. },
        } => Some(*correlation),
        ProbeOp::Cancel { .. } => None,
    })
}

fn empty_command() -> CommandTemplate {
    CommandTemplate::new([ArgTemplate::new([ArgPart::literal("/bin/true")])])
}

fn diff_aware_command() -> CommandTemplate {
    CommandTemplate::new([ArgTemplate::new([
        ArgPart::literal("fmt"),
        ArgPart::Placeholder(Placeholder::Created),
    ])])
}

/// V5-native helper: build a `TreeSnapshot::Dir` from a list of
/// `(name, kind, inode)` triples. Multi-segment names (e.g. "sub/foo.rs")
/// are *not* supported — tests in this file use leaf-name segments only.
fn dir_snap(root: ResourceId, children: Vec<(&str, EntryKind, u64)>) -> TreeSnapshot {
    let mut map: BTreeMap<CompactString, ChildEntry> = BTreeMap::new();
    for (name, kind, inode) in children {
        debug_assert!(
            !name.contains('/'),
            "dir_snap takes single-component children; nested paths must be built explicitly",
        );
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

/// Walk through a Standard burst — drains settle timers, injects probe
/// responses with `snap` until the burst stabilizes and emits an Effect.
/// Returns the `StepOutput` containing the Effect.
///
/// `t0` is the moment the `FsEvent` fired. The walker advances time by
/// large strides each iteration so backoff catches up.
fn drive_standard_burst_to_stable(
    e: &mut Engine,
    pid: specter_core::ProfileId,
    snap: TreeSnapshot,
    t0: Instant,
) -> StepOutput {
    let mut t = t0;
    for _ in 0..8 {
        t += SETTLE * 4;
        let correlation = drain_to_probe_correlation(e, t);
        if let Some(c) = correlation {
            let out = e.step(
                Input::ProbeResponse(ProbeResponse {
                    profile: pid,
                    correlation: c,
                    result: ProbeResult::Ok(snap.clone()),
                }),
                t,
            );
            if !out.effects.is_empty() {
                return out;
            }
        }
    }
    panic!("Standard burst failed to stabilize within drive iterations");
}

/// Drain timers and return the most recent probe's correlation, if any
/// fired in the process.
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
    // The whole V4 spine: attach_sub → Seed → Idle → FsEvent → Standard →
    // Effect → EffectComplete → Seed → Idle. Each transition observable
    // in the StepOutputs.
    let mut e = Engine::new();
    let r = e_anchor(&mut e, "src");
    let now = Instant::now();
    let req = SubAttachRequest {
        name: "build".into(),
        resource: r,
        path: None,
        config: cfg_recursive(),
        max_settle: MAX_SETTLE,
        settle: SETTLE,
        command: empty_command(),
        scope: EffectScope::SubtreeRoot,
        events: NO_EVENTS,
    };
    let (sid, attach_out) = e.attach_sub(req, now);

    // attach_sub emits Watch + Suppress (anchor) + Probe (Seed). No Effect.
    assert!(
        attach_out
            .watch_ops
            .iter()
            .any(|op| matches!(op, WatchOp::Watch { .. }))
    );
    assert!(
        attach_out
            .watch_ops
            .iter()
            .any(|op| matches!(op, WatchOp::Suppress { .. }))
    );
    let seed_correlation =
        first_probe_correlation(&attach_out).expect("Seed Probe fires immediately");
    assert!(attach_out.effects.is_empty());

    // Seed Ok → baseline = current = empty snapshot; → Idle; Unsuppress.
    let snap_seed = dir_snap(r, vec![]);
    let seed_resp = e.step(
        Input::ProbeResponse(ProbeResponse {
            profile: pid_of(&e, sid),
            correlation: seed_correlation,
            result: ProbeResult::Ok(snap_seed.clone()),
        }),
        now + Duration::from_millis(1),
    );
    assert!(seed_resp.effects.is_empty(), "Seed never emits Effects");
    assert!(
        seed_resp
            .watch_ops
            .iter()
            .any(|op| matches!(op, WatchOp::Unsuppress { .. }))
    );

    // FsEvent on anchor → Standard Settling. Suppress emitted.
    let t1 = now + Duration::from_millis(10);
    let fs_out = e.step(
        Input::FsEvent {
            resource: r,
            event: FsEvent::Modified,
        },
        t1,
    );
    assert!(
        fs_out
            .watch_ops
            .iter()
            .any(|op| matches!(op, WatchOp::Suppress { .. }))
    );
    assert!(fs_out.effects.is_empty());

    // Drive the Standard burst to a stable verdict (probing against the
    // empty snapshot; the burst stabilizes when current matches the
    // response). The walker advances time and injects probe responses.
    let pid = pid_of(&e, sid);
    let stable_out = drive_standard_burst_to_stable(&mut e, pid, snap_seed.clone(), t1);
    assert_eq!(stable_out.effects.len(), 1);
    assert!(!stable_out.effects[0].forced);

    // EffectComplete::Ok → Engine starts the next Seed burst (V4 fix).
    let post_effect = e.step(
        Input::EffectComplete {
            sub: sid,
            key: stable_out.effects[0].key.clone(),
            result: EffectOutcome::Ok,
        },
        t1 + SETTLE * 16,
    );
    let next_seed_correlation =
        first_probe_correlation(&post_effect).expect("post-Effect Seed Probe");

    // Seed Ok with the same snapshot → Idle, baseline advances to post-
    // Effect state.
    let _final_out = e.step(
        Input::ProbeResponse(ProbeResponse {
            profile: pid,
            correlation: next_seed_correlation,
            result: ProbeResult::Ok(snap_seed),
        }),
        t1 + SETTLE * 16 + Duration::from_millis(1),
    );
}

#[test]
fn vanished_during_seed_clears_baseline_and_diagnoses() {
    let mut e = Engine::new();
    let r = e_anchor(&mut e, "log.txt");
    e.tree_mut().get_mut(r).unwrap().kind = ResourceKind::File;
    let req = SubAttachRequest {
        name: "fmt".into(),
        resource: r,
        path: None,
        config: ScanConfig::builder().build(),
        max_settle: MAX_SETTLE,
        settle: SETTLE,
        command: empty_command(),
        scope: EffectScope::SubtreeRoot,
        events: NO_EVENTS,
    };
    let (sid, out) = e.attach_sub(req, Instant::now());
    let correlation = first_probe_correlation(&out).expect("Seed probe");
    let pid = pid_of(&e, sid);

    let resp_out = e.step(
        Input::ProbeResponse(ProbeResponse {
            profile: pid,
            correlation,
            result: ProbeResult::Vanished,
        }),
        Instant::now(),
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
    let r = e_anchor(&mut e, "src");
    let now = Instant::now();
    let req = SubAttachRequest {
        name: "build".into(),
        resource: r,
        path: None,
        config: cfg_recursive(),
        max_settle: MAX_SETTLE,
        settle: SETTLE,
        command: empty_command(),
        scope: EffectScope::SubtreeRoot,
        events: NO_EVENTS,
    };
    let (sid, attach_out) = e.attach_sub(req, now);
    let pid = pid_of(&e, sid);
    let stale_correlation = first_probe_correlation(&attach_out).expect("Seed probe correlation");

    // Inject FsEvent → transitions to Settling (preserving Seed intent),
    // emits Cancel.
    let _evt_out = e.step(
        Input::FsEvent {
            resource: r,
            event: FsEvent::Modified,
        },
        now,
    );

    // Late ProbeResponse with the now-stale correlation arrives.
    let late_resp = e.step(
        Input::ProbeResponse(ProbeResponse {
            profile: pid,
            correlation: stale_correlation,
            result: ProbeResult::Ok(dir_snap(r, vec![])),
        }),
        now + Duration::from_millis(1),
    );
    assert!(
        late_resp
            .diagnostics
            .iter()
            .any(|d| matches!(d, Diagnostic::StaleProbeResponse { .. }))
    );
    // No baseline change; Profile still Active.
}

#[test]
fn seed_burst_descendants_watched_via_first_probe() {
    let mut e = Engine::new();
    let r = e_anchor(&mut e, "src");
    let req = SubAttachRequest {
        name: "build".into(),
        resource: r,
        path: None,
        config: cfg_recursive(),
        max_settle: MAX_SETTLE,
        settle: SETTLE,
        command: empty_command(),
        scope: EffectScope::SubtreeRoot,
        events: NO_EVENTS,
    };
    let (sid, attach_out) = e.attach_sub(req, Instant::now());
    let pid = pid_of(&e, sid);
    let correlation = first_probe_correlation(&attach_out).unwrap();

    let snap = dir_snap(
        r,
        vec![("foo.rs", EntryKind::File, 1), ("bar", EntryKind::Dir, 2)],
    );
    let resp_out = e.step(
        Input::ProbeResponse(ProbeResponse {
            profile: pid,
            correlation,
            result: ProbeResult::Ok(snap),
        }),
        Instant::now(),
    );
    let watches = resp_out
        .watch_ops
        .iter()
        .filter(|op| matches!(op, WatchOp::Watch { .. }))
        .count();
    // Files don't get Watch ops; only the Dir descendant
    // contributes. The File still materializes as a Resource (for
    // PerStableFile DedupKey support), no FD.
    assert_eq!(watches, 1, "one Watch for Dir descendant only");
}

#[test]
fn force_fire_emits_effect_with_forced_true() {
    let mut e = Engine::new();
    let r = e_anchor(&mut e, "src");
    let now = Instant::now();
    let req = SubAttachRequest {
        name: "build".into(),
        resource: r,
        path: None,
        config: cfg_recursive(),
        max_settle: MAX_SETTLE,
        settle: SETTLE,
        command: empty_command(),
        scope: EffectScope::SubtreeRoot,
        events: NO_EVENTS,
    };
    let (sid, attach_out) = e.attach_sub(req, now);
    let pid = pid_of(&e, sid);
    let seed_corr = first_probe_correlation(&attach_out).unwrap();

    // Complete the Seed burst with empty baseline.
    e.step(
        Input::ProbeResponse(ProbeResponse {
            profile: pid,
            correlation: seed_corr,
            result: ProbeResult::Ok(dir_snap(r, vec![])),
        }),
        now + Duration::from_millis(1),
    );

    // FsEvent → Standard Settling.
    let t1 = now + Duration::from_millis(10);
    e.step(
        Input::FsEvent {
            resource: r,
            event: FsEvent::Modified,
        },
        t1,
    );

    // Advance past max_settle so burst_deadline fires.
    let deadline_t = t1 + MAX_SETTLE + Duration::from_millis(1);
    let probe_corr = drain_to_probe_correlation(&mut e, deadline_t);

    if let Some(corr) = probe_corr {
        // Inject a not-stable response — different snapshot.
        let snap = dir_snap(r, vec![("x", EntryKind::File, 99)]);
        let out = e.step(
            Input::ProbeResponse(ProbeResponse {
                profile: pid,
                correlation: corr,
                result: ProbeResult::Ok(snap),
            }),
            deadline_t,
        );
        assert_eq!(out.effects.len(), 1);
        assert!(
            out.effects[0].forced,
            "force-fired Effect carries forced=true"
        );
    } else {
        panic!("burst_deadline did not produce a probe");
    }
}

#[test]
fn step_output_is_sorted() {
    // Build a multi-Watch scenario (descendants reconciled on first probe)
    // and confirm StepOutput.watch_ops is sorted by ResourceId.
    let mut e = Engine::new();
    let r = e_anchor(&mut e, "root");
    let req = SubAttachRequest {
        name: "build".into(),
        resource: r,
        path: None,
        config: cfg_recursive(),
        max_settle: MAX_SETTLE,
        settle: SETTLE,
        command: empty_command(),
        scope: EffectScope::SubtreeRoot,
        events: NO_EVENTS,
    };
    let (sid, attach_out) = e.attach_sub(req, Instant::now());
    let pid = pid_of(&e, sid);
    let correlation = first_probe_correlation(&attach_out).unwrap();
    let leaves: Vec<(String, EntryKind, u64)> = (0..5)
        .map(|i| (format!("file-{i}"), EntryKind::File, 100 + i))
        .collect();
    let snap = dir_snap(
        r,
        leaves
            .iter()
            .map(|(s, k, i)| (s.as_str(), *k, *i))
            .collect(),
    );
    let out = e.step(
        Input::ProbeResponse(ProbeResponse {
            profile: pid,
            correlation,
            result: ProbeResult::Ok(snap),
        }),
        Instant::now(),
    );
    let resources: Vec<ResourceId> = out
        .watch_ops
        .iter()
        .map(|op| match op {
            WatchOp::Watch { resource, .. } => *resource,
            WatchOp::Unwatch { resource } => *resource,
            WatchOp::Suppress { resource } => *resource,
            WatchOp::Unsuppress { resource } => *resource,
        })
        .collect();
    let mut sorted = resources.clone();
    sorted.sort();
    assert_eq!(resources, sorted, "watch_ops sorted by ResourceId");
}

// ---------- helpers ----------

fn e_anchor(e: &mut Engine, name: &str) -> ResourceId {
    let r = e.tree_mut().ensure(None, name, ResourceRole::User);
    e.tree_mut().get_mut(r).unwrap().kind = ResourceKind::Dir;
    r
}

fn pid_of(e: &Engine, sid: specter_core::SubId) -> specter_core::ProfileId {
    e.subs().get(sid).expect("sub exists").profile
}
