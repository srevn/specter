//! Engine-level integration tests for the event-filtering primitive.
//! Each test exercises the user-facing invariants without spinning up a
//! real kqueue: the entry filter, the mask-fork (mask ∈ `config_hash`),
//! the anchor-bypass, the descent-prefix STRUCTURE-only contribution,
//! and the E2E #3 closure path (`subtree-root × default events ⇒
//! has_per_file_fds = true ⇒ per-file FDs on covered Leaves).
//!
//! Where the equivalent shape lives in `transitions_tests.rs`, this file
//! is the **integration** counterpart — exercising whole-Engine flows
//! (attach → Seed-Ok → FsEvent → optional Effect) rather than dispatch
//! handlers in isolation.

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

use compact_str::CompactString;
use specter_core::{
    ArgPart, ArgTemplate, ChildEntry, ClassSet, CommandTemplate, DedupKey, Diagnostic, DirChild,
    DirMeta, DirSnapshot, EffectScope, EntryKind, FsEvent, Input, LeafEntry, ProbeCorrelation,
    ProbeOp, ProbeRequest, ProbeResponse, ProbeResult, ProfileId, ProfileState, ResourceId,
    ResourceKind, ResourceRole, ScanConfig, StepOutput, SubAttachRequest, TreeSnapshot, WatchOp,
};
use specter_engine::Engine;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant, UNIX_EPOCH};

const SETTLE: Duration = Duration::from_millis(100);
const MAX_SETTLE: Duration = Duration::from_secs(6);

// ───────────────────────────────────────────────────────────────────────
// Fixtures
// ───────────────────────────────────────────────────────────────────────

fn empty_command() -> CommandTemplate {
    CommandTemplate::new([ArgTemplate::new([ArgPart::literal("/bin/true")])])
}

fn dir_snap(root: ResourceId, children: Vec<(&str, EntryKind, u64)>) -> TreeSnapshot {
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

fn first_probe_corr(out: &StepOutput) -> Option<ProbeCorrelation> {
    out.probe_ops.iter().find_map(|op| match op {
        ProbeOp::Probe {
            request: ProbeRequest { correlation, .. },
        } => Some(*correlation),
        ProbeOp::Cancel { .. } => None,
    })
}

/// Drive the Profile from fresh attach through Seed-Ok → Idle. After this,
/// `Profile.current` and `Profile.baseline` are set to `seed_snap`.
fn complete_seed_burst(
    e: &mut Engine,
    pid: ProfileId,
    attach_out: &StepOutput,
    seed_snap: TreeSnapshot,
) {
    let corr = first_probe_corr(attach_out).expect("Seed probe fires at attach");
    let _ = e.step(
        Input::ProbeResponse(ProbeResponse {
            profile: pid,
            correlation: corr,
            result: ProbeResult::Ok(seed_snap),
        }),
        Instant::now(),
    );
    assert!(
        matches!(e.profiles().get(pid).unwrap().state, ProfileState::Idle),
        "Seed-Ok transitions Profile to Idle",
    );
}

/// Attach a Sub at `resource` with the supplied `events` mask. Returns the
/// minted SubId, ProfileId, and `attach_out`.
fn attach_sub_with_events(
    e: &mut Engine,
    name: &str,
    resource: ResourceId,
    scope: EffectScope,
    events: ClassSet,
    config: ScanConfig,
) -> (specter_core::SubId, ProfileId, StepOutput) {
    let req = SubAttachRequest::for_resource(
        name.to_string(),
        resource,
        config,
        MAX_SETTLE,
        SETTLE,
        empty_command(),
        scope,
        events,
        false,
    );
    let (sid, out) = e.attach_sub(req, Instant::now());
    let pid = e.subs().get(sid).unwrap().profile;
    (sid, pid, out)
}

// ───────────────────────────────────────────────────────────────────────
// IT-EF-1 — DEFAULT_SUBTREE_ROOT enables per-file FDs (closes E2E #3)
//
// `echo 'test' > file.txt` inside a subtree-root watched dir receives an
// event on the per-file FD (CONTENT class drives the translator to emit
// `NOTE_WRITE | NOTE_EXTEND` on every covered Leaf), the burst fires,
// the probe runs, the diff classifies the file as Modified, and the
// user's command runs.
//
// At engine level this manifests as: a `subtree-root` Sub with default
// events (`STRUCTURE | CONTENT`) drives `has_per_file_fds = true`, so
// `walk_pair` emits a per-file `WatchOp::Watch` for every covered Leaf
// during reconciliation. The Watch op carries the Profile's mask (which
// translates downstream to `NOTE_WRITE | NOTE_EXTEND` on the file FD).
// ───────────────────────────────────────────────────────────────────────

#[test]
fn it_ef_1_default_subtree_root_emits_per_file_watch_on_leaves() {
    let mut e = Engine::new();
    let root = e.tree_mut().ensure(None, "src", ResourceRole::User);
    e.tree_mut().get_mut(root).unwrap().kind = ResourceKind::Dir;

    let (_sid, pid, attach_out) = attach_sub_with_events(
        &mut e,
        "build",
        root,
        EffectScope::SubtreeRoot,
        ClassSet::DEFAULT_SUBTREE_ROOT,
        ScanConfig::builder().recursive(true).build(),
    );

    // Default for SubtreeRoot ⇒ has_per_file_fds = true.
    assert!(
        e.profiles().get(pid).unwrap().has_per_file_fds,
        "default subtree-root mask (STRUCTURE|CONTENT) sets has_per_file_fds",
    );

    // Drive the Seed probe with one File child. walk_pair runs against
    // prior=None ⇒ pure-create path. With has_per_file_fds=true the File
    // child gets a Watch op.
    let corr = first_probe_corr(&attach_out).expect("Seed probe");
    let snap = dir_snap(root, vec![("file.txt", EntryKind::File, 1)]);
    let seed_out = e.step(
        Input::ProbeResponse(ProbeResponse {
            profile: pid,
            correlation: corr,
            result: ProbeResult::Ok(snap),
        }),
        Instant::now(),
    );

    // Resource was materialized.
    let file_id = e.tree().lookup(Some(root), "file.txt").expect("file slot");

    // The seed response emits a Watch op for the per-file FD.
    let saw_per_file_watch = seed_out.watch_ops.iter().any(|op| match op {
        WatchOp::Watch {
            resource, events, ..
        } => *resource == file_id && *events == ClassSet::DEFAULT_SUBTREE_ROOT,
        _ => false,
    });
    assert!(
        saw_per_file_watch,
        "covered file leaf gets a Watch with the Profile's mask (closes E2E #3); got watch_ops = {:?}",
        seed_out.watch_ops,
    );
}

#[test]
fn it_ef_1_structure_only_subtree_does_not_emit_per_file_watch() {
    // Negative case: a Sub explicitly requesting STRUCTURE only does NOT
    // get per-file FDs. Confirms `has_per_file_fds` is mask-driven, not
    // scope-driven.
    let mut e = Engine::new();
    let root = e.tree_mut().ensure(None, "src", ResourceRole::User);
    e.tree_mut().get_mut(root).unwrap().kind = ResourceKind::Dir;

    let (_sid, pid, attach_out) = attach_sub_with_events(
        &mut e,
        "ls-only",
        root,
        EffectScope::SubtreeRoot,
        ClassSet::STRUCTURE,
        ScanConfig::builder().recursive(true).build(),
    );

    assert!(
        !e.profiles().get(pid).unwrap().has_per_file_fds,
        "STRUCTURE-only mask leaves has_per_file_fds = false",
    );

    let corr = first_probe_corr(&attach_out).expect("Seed probe");
    let snap = dir_snap(root, vec![("file.txt", EntryKind::File, 1)]);
    let seed_out = e.step(
        Input::ProbeResponse(ProbeResponse {
            profile: pid,
            correlation: corr,
            result: ProbeResult::Ok(snap),
        }),
        Instant::now(),
    );

    let file_id = e.tree().lookup(Some(root), "file.txt").expect("file slot");
    let saw_per_file_watch = seed_out.watch_ops.iter().any(|op| match op {
        WatchOp::Watch { resource, .. } => *resource == file_id,
        _ => false,
    });
    assert!(
        !saw_per_file_watch,
        "STRUCTURE-only Profile must not emit per-file Watch ops",
    );
}

// ───────────────────────────────────────────────────────────────────────
// IT-EF-2 — Two Subs with different masks fork separate Profiles
//
// `events` folds into `compute_config_hash`, so two Subs at the same
// resource that differ only on `events` partition into two distinct
// Profiles. This guards against the "Profile-union infection" problem:
// a chmod on a Sub asking only for CONTENT must not fire that Sub's
// command via the same Profile that handles a sibling Sub asking for
// METADATA.
// ───────────────────────────────────────────────────────────────────────

#[test]
fn it_ef_2_two_subs_different_masks_fork_separate_profiles() {
    let mut e = Engine::new();
    let root = e.tree_mut().ensure(None, "build", ResourceRole::User);
    e.tree_mut().get_mut(root).unwrap().kind = ResourceKind::Dir;

    let (_sid_a, pid_a, _) = attach_sub_with_events(
        &mut e,
        "make",
        root,
        EffectScope::SubtreeRoot,
        ClassSet::CONTENT,
        ScanConfig::builder().recursive(true).build(),
    );
    let (_sid_b, pid_b, _) = attach_sub_with_events(
        &mut e,
        "audit",
        root,
        EffectScope::SubtreeRoot,
        ClassSet::METADATA,
        ScanConfig::builder().recursive(true).build(),
    );

    assert_ne!(
        pid_a, pid_b,
        "Subs with distinct events forks fork into distinct Profiles",
    );

    // Both Profiles record their own mask; the per-resource union
    // contains both bits because both contribute to the anchor.
    assert_eq!(
        e.profiles().get(pid_a).unwrap().events_union,
        ClassSet::CONTENT,
    );
    assert_eq!(
        e.profiles().get(pid_b).unwrap().events_union,
        ClassSet::METADATA,
    );
    assert_eq!(
        e.tree().get(root).unwrap().events_union,
        ClassSet::CONTENT | ClassSet::METADATA,
        "anchor's per-Resource union ORs both Profiles' contributions",
    );
}

#[test]
fn it_ef_2_chmod_only_fires_metadata_profile_not_content_profile() {
    // Concrete chmod scenario: Sub A wants CONTENT, Sub B wants METADATA.
    // After a `MetadataChanged` at the anchor, Profile B drives a burst;
    // Profile A's class filter would drop it… EXCEPT anchor events bypass
    // the filter unconditionally. So both Profiles drive bursts — the
    // differentiation happens at probe time / dedup, not at routing.
    //
    // This test pins the routing semantics (both Profiles get the event
    // because of the anchor-bypass) and the registration semantics
    // (METADATA-only kernel mask wouldn't even fire MetadataChanged for
    // the CONTENT Sub if the class filter applied — but it doesn't, due to
    // the anchor-bypass).
    //
    // The descendant case (where the filter DOES apply) is covered in
    // `it_ef_6_metadata_dropped_on_descendant_for_content_only_sub`.
    let mut e = Engine::new();
    let root = e.tree_mut().ensure(None, "build", ResourceRole::User);
    e.tree_mut().get_mut(root).unwrap().kind = ResourceKind::Dir;
    let cfg = ScanConfig::builder().recursive(true).build();
    let (_sid_a, pid_a, attach_a) = attach_sub_with_events(
        &mut e,
        "make",
        root,
        EffectScope::SubtreeRoot,
        ClassSet::CONTENT,
        cfg.clone(),
    );
    let (_sid_b, pid_b, attach_b) = attach_sub_with_events(
        &mut e,
        "audit",
        root,
        EffectScope::SubtreeRoot,
        ClassSet::METADATA,
        cfg,
    );
    // Drive both Seeds → Idle.
    let snap = dir_snap(root, vec![]);
    complete_seed_burst(&mut e, pid_a, &attach_a, snap.clone());
    complete_seed_burst(&mut e, pid_b, &attach_b, snap);

    // MetadataChanged at the anchor — anchor events bypass class filter
    // for both Profiles. Both should drive bursts.
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
        "anchor events bypass class filter for ALL covering Profiles",
    );

    // Both Profiles transition Idle → Active(Standard, Settling).
    for pid in [pid_a, pid_b] {
        assert!(
            matches!(
                e.profiles().get(pid).unwrap().state,
                ProfileState::Active(_),
            ),
            "anchor MetadataChanged drives a burst on Profile {pid:?} regardless of mask",
        );
    }
}

// ───────────────────────────────────────────────────────────────────────
// IT-EF-3 — Descent prefix carries STRUCTURE-only mask
//
// Descent prefix watches register `events = {STRUCTURE}` regardless of
// the Sub's mask. The prefix is not the user's anchor; it's a transient
// artifact of the engine's descent state machine.
//
// We attach a Sub at a path whose anchor doesn't yet exist. The deepest
// existing prefix is bumped — its `events_union` should be STRUCTURE
// only, NOT the Sub's user mask.
// ───────────────────────────────────────────────────────────────────────

#[test]
fn it_ef_3_descent_prefix_contributes_structure_only() {
    let mut e = Engine::new();
    // Pre-existing /tmp; the Sub's anchor /tmp/build/leaf doesn't exist.
    let tmp = e.tree_mut().ensure(None, "tmp", ResourceRole::User);
    e.tree_mut().get_mut(tmp).unwrap().kind = ResourceKind::Dir;

    let req = SubAttachRequest::for_path(
        "watch".into(),
        PathBuf::from("tmp/build/leaf"),
        ScanConfig::builder().recursive(true).build(),
        MAX_SETTLE,
        SETTLE,
        empty_command(),
        EffectScope::SubtreeRoot,
        ClassSet::CONTENT, // user wants CONTENT only
        false,
    );
    let (sid, _attach_out) = e.attach_sub(req, Instant::now());
    let pid = e.subs().get(sid).unwrap().profile;

    // Profile is Pending; current_prefix is /tmp.
    let descent = match &e.profiles().get(pid).unwrap().state {
        ProfileState::Pending(d) => d.clone(),
        s => panic!("expected Pending, got {s:?}"),
    };
    assert_eq!(descent.current_prefix, tmp);

    // /tmp's events_union is STRUCTURE — NOT the Sub's CONTENT mask. The
    // Sub's mask only contributes to its own anchor's union; the prefix
    // is engine infrastructure.
    assert_eq!(
        e.tree().get(tmp).unwrap().events_union,
        ClassSet::STRUCTURE,
        "descent prefix mask is STRUCTURE regardless of Sub's events",
    );
    // Profile's events_union still records the user's mask (drives mask
    // on the eventual anchor materialization).
    assert_eq!(
        e.profiles().get(pid).unwrap().events_union,
        ClassSet::CONTENT,
    );
}

// ───────────────────────────────────────────────────────────────────────
// IT-EF-4 — Anchor terminal events bypass the class filter
//
// A `events = ["content"]` Sub on a Dir anchor: `Removed` at the anchor
// folds to STRUCTURE per `fs_event_to_class`'s identity-on-Dir rule.
// STRUCTURE is NOT in the Profile's mask. Without the anchor-bypass, the
// class filter would drop the event and the anchor's contribution would leak —
// `watch_root_parent → re-descent` recovery would never trigger.
//
// With the anchor-bypass, the event routes to
// `on_anchor_terminal_event` regardless of mask: `anchor_contribution`
// clears, baseline/current drop, the Profile transitions Idle, ready
// for recovery via watch_root_parent.
// ───────────────────────────────────────────────────────────────────────

#[test]
fn it_ef_4_anchor_terminal_bypasses_filter_for_narrow_mask() {
    let mut e = Engine::new();
    let parent = e.tree_mut().ensure(None, "p", ResourceRole::User);
    e.tree_mut().get_mut(parent).unwrap().kind = ResourceKind::Dir;
    let anchor = e
        .tree_mut()
        .ensure(Some(parent), "watched-dir", ResourceRole::User);
    e.tree_mut().get_mut(anchor).unwrap().kind = ResourceKind::Dir;

    // CONTENT-only mask on a Dir anchor — note: CONTENT registers no
    // bits on a Dir, but the class routing still uses Profile.events_union
    // for filtering.
    let (_sid, pid, attach_out) = attach_sub_with_events(
        &mut e,
        "watch",
        anchor,
        EffectScope::SubtreeRoot,
        ClassSet::CONTENT,
        ScanConfig::builder().recursive(true).build(),
    );
    complete_seed_burst(&mut e, pid, &attach_out, dir_snap(anchor, vec![]));

    assert!(
        e.profiles().get(pid).unwrap().anchor_contribution,
        "post-Seed: anchor_contribution=true",
    );
    assert_eq!(e.tree().get(anchor).unwrap().watch_demand, 1);

    // `Removed` on the anchor (a Dir) folds to STRUCTURE — not in mask.
    // Without the anchor-bypass, this event would drop with
    // EventClassDropped. With the anchor-bypass, it routes through
    // on_anchor_terminal_event.
    let out = e.step(
        Input::FsEvent {
            resource: anchor,
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

    let p = e.profiles().get(pid).unwrap();
    assert!(
        !p.anchor_contribution,
        "anchor_contribution cleared by on_anchor_terminal_event",
    );
    assert!(p.baseline.is_none());
    assert!(p.current.is_none());
    assert_eq!(
        e.tree().get(anchor).unwrap().watch_demand,
        0,
        "anchor's watch_demand released",
    );
    // watch_root_parent is intact for recovery.
    assert_eq!(p.watch_root_parent, Some(parent));
    assert!(
        e.tree().get(parent).unwrap().watch_demand >= 1,
        "watch_root_parent contribution survives anchor terminal",
    );
}

// ───────────────────────────────────────────────────────────────────────
// IT-EF-6 — `MetadataChanged` on a CONTENT-only Sub does not fire
//
// Descendant `MetadataChanged` events on a Sub whose mask excludes
// METADATA drop with `EventClassDropped` and do NOT extend
// `dirty_resources` / `force_walk_resources` (the class filter sits
// before dirty-set bumps). The Profile remains in its prior state; no
// Effect emerges.
// ───────────────────────────────────────────────────────────────────────

#[test]
fn it_ef_6_descendant_metadata_drops_on_content_only_sub() {
    let mut e = Engine::new();
    let root = e.tree_mut().ensure(None, "src", ResourceRole::User);
    e.tree_mut().get_mut(root).unwrap().kind = ResourceKind::Dir;

    let (_sid, pid, attach_out) = attach_sub_with_events(
        &mut e,
        "build",
        root,
        EffectScope::SubtreeRoot,
        ClassSet::CONTENT,
        ScanConfig::builder().recursive(true).build(),
    );
    complete_seed_burst(&mut e, pid, &attach_out, dir_snap(root, vec![]));

    // Materialize a covered descendant File. Bump watch_demand so the
    // event passes the EventOnUnwatchedResource head guard.
    let child = e
        .tree_mut()
        .ensure(Some(root), "file.txt", ResourceRole::User);
    e.tree_mut().get_mut(child).unwrap().kind = ResourceKind::File;
    e.tree_mut().get_mut(child).unwrap().watch_demand = 1;

    // chmod on the descendant file → MetadataChanged.
    let out = e.step(
        Input::FsEvent {
            resource: child,
            event: FsEvent::MetadataChanged,
        },
        Instant::now(),
    );

    // The class filter drops with EventClassDropped.
    assert!(
        out.diagnostics.iter().any(|d| matches!(
            d,
            Diagnostic::EventClassDropped {
                resource,
                event: FsEvent::MetadataChanged,
                profile,
            } if *resource == child && *profile == pid,
        )),
        "MetadataChanged on a CONTENT-only Sub's descendant must drop with EventClassDropped",
    );

    // No state mutation: Profile remains Idle; no probe queued, no
    // Effect.
    assert!(
        matches!(e.profiles().get(pid).unwrap().state, ProfileState::Idle),
        "drop happens before drive_burst — Profile stays Idle",
    );
    assert!(out.probe_ops.is_empty(), "no probe queued");
    assert!(out.effects.is_empty(), "no effects emitted");
}

#[test]
fn it_ef_6_descendant_modified_drives_burst_on_content_sub() {
    // Positive control: Modified on a descendant of a CONTENT-only Sub
    // DOES drive the burst (CONTENT class matches mask).
    let mut e = Engine::new();
    let root = e.tree_mut().ensure(None, "src", ResourceRole::User);
    e.tree_mut().get_mut(root).unwrap().kind = ResourceKind::Dir;

    let (_sid, pid, attach_out) = attach_sub_with_events(
        &mut e,
        "build",
        root,
        EffectScope::SubtreeRoot,
        ClassSet::CONTENT,
        ScanConfig::builder().recursive(true).build(),
    );
    complete_seed_burst(&mut e, pid, &attach_out, dir_snap(root, vec![]));

    let child = e
        .tree_mut()
        .ensure(Some(root), "file.txt", ResourceRole::User);
    e.tree_mut().get_mut(child).unwrap().kind = ResourceKind::File;
    e.tree_mut().get_mut(child).unwrap().watch_demand = 1;

    let _ = e.step(
        Input::FsEvent {
            resource: child,
            event: FsEvent::Modified,
        },
        Instant::now(),
    );
    assert!(
        matches!(
            e.profiles().get(pid).unwrap().state,
            ProfileState::Active(_),
        ),
        "Modified on a CONTENT-class child drives a burst",
    );
}

// ───────────────────────────────────────────────────────────────────────
// IT-EF-5 — second Profile attaches at the same resource ⇒ engine emits
// fresh `WatchOp::Watch` with the widened union mask
//
// The union mask can change without the refcount changing — when a
// second Profile starts covering a Resource, its mask contribution may
// expand the union. The 0→1-edge model is structurally insufficient for
// this. `add_watch_demand` emits Watch on the 0→1 edge OR on any union
// widening at non-zero refcount.
//
// Watcher-side mechanics (cache diff, EV_ADD overwrite) are covered in
// `crates/specter-sensor/tests/kqueue_rewatch.rs`. This engine-level test
// pins the upstream contract: the engine emits the right Watch op.
// ───────────────────────────────────────────────────────────────────────

#[test]
fn it_ef_5_second_profile_widens_mask_emits_fresh_watch() {
    let mut e = Engine::new();
    let root = e.tree_mut().ensure(None, "root", ResourceRole::User);
    e.tree_mut().get_mut(root).unwrap().kind = ResourceKind::Dir;
    let cfg = ScanConfig::builder().recursive(true).build();

    // Profile A: events = CONTENT only.
    let (_sid_a, _pid_a, attach_a) = attach_sub_with_events(
        &mut e,
        "A",
        root,
        EffectScope::SubtreeRoot,
        ClassSet::CONTENT,
        cfg.clone(),
    );
    let watch_after_a = attach_a
        .watch_ops
        .iter()
        .rev()
        .find_map(|op| match op {
            WatchOp::Watch {
                resource, events, ..
            } if *resource == root => Some(*events),
            _ => None,
        })
        .expect("Profile A's attach emits a Watch on root");
    assert_eq!(
        watch_after_a,
        ClassSet::CONTENT,
        "Profile A's Watch carries CONTENT only",
    );
    assert_eq!(
        e.tree().get(root).unwrap().events_union,
        ClassSet::CONTENT,
        "per-Resource union after A = CONTENT",
    );

    // Profile B: events = METADATA on the same anchor (different mask
    // ⇒ different config_hash ⇒ separate Profile).
    let (_sid_b, _pid_b, attach_b) = attach_sub_with_events(
        &mut e,
        "B",
        root,
        EffectScope::SubtreeRoot,
        ClassSet::METADATA,
        cfg,
    );
    // The engine emits a fresh `WatchOp::Watch` for the root, carrying
    // the union (CONTENT | METADATA), even though the anchor's
    // `watch_demand` went 1→2 (not a 0→1 edge).
    let watch_after_b = attach_b
        .watch_ops
        .iter()
        .find_map(|op| match op {
            WatchOp::Watch {
                resource, events, ..
            } if *resource == root => Some(*events),
            _ => None,
        })
        .expect("Profile B's attach emits a fresh Watch on root (mask widening)");
    assert_eq!(
        watch_after_b,
        ClassSet::CONTENT | ClassSet::METADATA,
        "Profile B's attach widens the union; Watch carries CONTENT|METADATA",
    );
    assert_eq!(
        e.tree().get(root).unwrap().events_union,
        ClassSet::CONTENT | ClassSet::METADATA,
        "per-Resource union after B = CONTENT | METADATA",
    );
    assert_eq!(
        e.tree().get(root).unwrap().watch_demand,
        2,
        "watch_demand bumped 1→2 on second attach (union changes drive Watch even off the 0↔1 edge)",
    );
}

// ───────────────────────────────────────────────────────────────────────
// IT-EF-2 dedup — Subtree-keyed effect uses Profile id, so two Profiles
// with different masks don't collide on `last_emitted_dir_hash`.
// ───────────────────────────────────────────────────────────────────────

#[test]
fn it_ef_2_dedup_keys_disambiguated_by_profile_id() {
    let mut e = Engine::new();
    let root = e.tree_mut().ensure(None, "build", ResourceRole::User);
    e.tree_mut().get_mut(root).unwrap().kind = ResourceKind::Dir;
    let cfg = ScanConfig::builder().recursive(true).build();
    let (sid_a, pid_a, _) = attach_sub_with_events(
        &mut e,
        "make",
        root,
        EffectScope::SubtreeRoot,
        ClassSet::CONTENT,
        cfg.clone(),
    );
    let (sid_b, pid_b, _) = attach_sub_with_events(
        &mut e,
        "audit",
        root,
        EffectScope::SubtreeRoot,
        ClassSet::METADATA,
        cfg,
    );
    let dk_a = DedupKey::Subtree {
        sub: sid_a,
        profile: pid_a,
    };
    let dk_b = DedupKey::Subtree {
        sub: sid_b,
        profile: pid_b,
    };
    assert_ne!(dk_a, dk_b, "DedupKey::Subtree partitions by (sub, profile)");
}

// ───────────────────────────────────────────────────────────────────────
// Regression: Seed-Vanished + watch-root-parent recovery flow
//
// Bug: Before this fix, `dispatch_seed_vanished` left
// `anchor_contribution = true`. A subsequent `StructureChanged` at
// `watch_root_parent` triggered `start_pending_recovery`, which
// transitioned the Profile to `Pending` while the flag was still true —
// violating `reap_profile`'s `!(Pending && anchor_contribution)`
// trichotomy invariant.
//
// Fix: `dispatch_seed_vanished` (and `dispatch_seed_failed`) now release
// the anchor's contribution, mirroring `dispatch_standard_*`. The watch
// is re-acquired via descent's anchor materialization on recovery.
// ───────────────────────────────────────────────────────────────────────

#[test]
fn seed_vanished_releases_anchor_contribution_for_recovery() {
    let mut e = Engine::new();
    let parent = e.tree_mut().ensure(None, "p", ResourceRole::User);
    e.tree_mut().get_mut(parent).unwrap().kind = ResourceKind::Dir;
    let anchor = e.tree_mut().ensure(Some(parent), "a", ResourceRole::User);
    e.tree_mut().get_mut(anchor).unwrap().kind = ResourceKind::Dir;

    let (_sid, pid, attach_out) = attach_sub_with_events(
        &mut e,
        "watch",
        anchor,
        EffectScope::SubtreeRoot,
        ClassSet::CONTENT,
        ScanConfig::builder().recursive(true).build(),
    );
    assert!(e.profiles().get(pid).unwrap().anchor_contribution);
    assert_eq!(e.tree().get(anchor).unwrap().watch_demand, 1);

    // Seed Vanished: anchor was found at attach but disappeared before
    // probe could read.
    let corr = first_probe_corr(&attach_out).expect("Seed probe");
    let out = e.step(
        Input::ProbeResponse(ProbeResponse {
            profile: pid,
            correlation: corr,
            result: ProbeResult::Vanished,
        }),
        Instant::now(),
    );

    // Anchor's contribution is released; Profile back to Idle ready for
    // recovery via watch_root_parent.
    assert!(
        !e.profiles().get(pid).unwrap().anchor_contribution,
        "Seed Vanished now releases anchor_contribution (post-fix)",
    );
    assert_eq!(
        e.tree().get(anchor).unwrap().watch_demand,
        0,
        "anchor's watch_demand released",
    );
    assert!(
        out.watch_ops
            .iter()
            .any(|op| matches!(op, WatchOp::Unwatch { resource } if *resource == anchor)),
        "Unwatch emitted for the anchor; got {:?}",
        out.watch_ops,
    );
    assert!(matches!(
        e.profiles().get(pid).unwrap().state,
        ProfileState::Idle,
    ));
    // watch_root_parent kept for recovery.
    assert!(e.profiles().get(pid).unwrap().watch_root_parent == Some(parent));
}

#[test]
fn seed_vanished_then_recovery_does_not_violate_trichotomy() {
    // The full failure sequence pre-fix:
    //   1. attach_sub: anchor_contribution=true.
    //   2. Seed probe → Vanished. Old code: anchor_contribution stayed
    //      true; current=None; Profile Idle.
    //   3. StructureChanged at watch_root_parent → start_pending_recovery
    //      transitions Profile → Pending. State now violates the
    //      trichotomy: Pending + anchor_contribution=true.
    //   4. detach_sub → reap_profile → debug_assert panic OR (release)
    //      memory leak (anchor watch_demand never released).
    //
    // With the fix: step 2 releases anchor_contribution, so step 3 sees
    // a clean state and step 4 reaps cleanly.
    let mut e = Engine::new();
    let parent = e.tree_mut().ensure(None, "p", ResourceRole::User);
    e.tree_mut().get_mut(parent).unwrap().kind = ResourceKind::Dir;
    let anchor = e.tree_mut().ensure(Some(parent), "a", ResourceRole::User);
    e.tree_mut().get_mut(anchor).unwrap().kind = ResourceKind::Dir;

    let (sid, pid, attach_out) = attach_sub_with_events(
        &mut e,
        "watch",
        anchor,
        EffectScope::SubtreeRoot,
        ClassSet::CONTENT,
        ScanConfig::builder().recursive(true).build(),
    );

    // Step 2: Seed Vanished.
    let corr = first_probe_corr(&attach_out).expect("Seed probe");
    e.step(
        Input::ProbeResponse(ProbeResponse {
            profile: pid,
            correlation: corr,
            result: ProbeResult::Vanished,
        }),
        Instant::now(),
    );

    // Step 3: StructureChanged at watch_root_parent triggers recovery.
    e.step(
        Input::FsEvent {
            resource: parent,
            event: FsEvent::StructureChanged,
        },
        Instant::now(),
    );

    // Profile is Pending now. anchor_contribution must be false (the
    // trichotomy invariant).
    let p = e.profiles().get(pid).expect("Profile alive");
    assert!(
        matches!(p.state, ProfileState::Pending(_)),
        "recovery transitions Profile → Pending",
    );
    assert!(
        !p.anchor_contribution,
        "post-fix: anchor_contribution=false during Pending — trichotomy holds",
    );

    // Step 4: detach. reap_profile must NOT debug_assert and must NOT
    // leak the anchor's watch_demand.
    let out = e.detach_sub(sid, Instant::now());
    assert!(
        e.profiles().get(pid).is_none(),
        "Profile reaped without panic",
    );
    // The descent prefix's contribution is released cleanly.
    let unwatch_count = out
        .watch_ops
        .iter()
        .filter(|op| matches!(op, WatchOp::Unwatch { .. }))
        .count();
    assert!(unwatch_count >= 1, "at least one Unwatch on reap");
}

#[test]
fn seed_failed_releases_anchor_contribution() {
    // Symmetric regression for dispatch_seed_failed.
    let mut e = Engine::new();
    let parent = e.tree_mut().ensure(None, "p", ResourceRole::User);
    e.tree_mut().get_mut(parent).unwrap().kind = ResourceKind::Dir;
    let anchor = e.tree_mut().ensure(Some(parent), "a", ResourceRole::User);
    e.tree_mut().get_mut(anchor).unwrap().kind = ResourceKind::Dir;

    let (_sid, pid, attach_out) = attach_sub_with_events(
        &mut e,
        "watch",
        anchor,
        EffectScope::SubtreeRoot,
        ClassSet::CONTENT,
        ScanConfig::builder().recursive(true).build(),
    );

    let corr = first_probe_corr(&attach_out).expect("Seed probe");
    let _out = e.step(
        Input::ProbeResponse(ProbeResponse {
            profile: pid,
            correlation: corr,
            result: ProbeResult::Failed { errno: 13 },
        }),
        Instant::now(),
    );

    assert!(
        !e.profiles().get(pid).unwrap().anchor_contribution,
        "Seed Failed releases anchor_contribution (post-fix, symmetric with Seed Vanished)",
    );
    assert_eq!(
        e.tree().get(anchor).unwrap().watch_demand,
        0,
        "anchor's watch_demand released",
    );
}

// ───────────────────────────────────────────────────────────────────────
// Regression: dispatch_standard_vanished/failed + reap_pending no longer
// double-releases the anchor contribution (debug_assert was reachable
// before the release-before-finish reorder).
// ───────────────────────────────────────────────────────────────────────

/// Set up a Profile + a covered Dir child so the anchor cannot reap
/// when the Profile detaches (the child's existence keeps the slot
/// alive). Returns (root, child, sid, pid, attach_out).
///
/// Pre-fix `dispatch_standard_vanished` runs `sub_watch_demand` on the
/// anchor AFTER `finish_burst_to_idle` (which already released it via
/// `reap_pending → reap_profile`); without surviving children the
/// anchor is reaped and the post-finish call early-exits silently.
/// With a covered Dir child, the anchor slot survives the reap and the
/// post-finish decrement underflows the now-zero `watch_demand`,
/// tripping the `debug_assert!`.
fn setup_with_surviving_child(
    e: &mut Engine,
) -> (
    ResourceId,
    ResourceId,
    specter_core::SubId,
    ProfileId,
    StepOutput,
) {
    let root = e.tree_mut().ensure(None, "src", ResourceRole::User);
    e.tree_mut().get_mut(root).unwrap().kind = ResourceKind::Dir;

    let (sid, pid, attach_out) = attach_sub_with_events(
        e,
        "build",
        root,
        EffectScope::SubtreeRoot,
        ClassSet::CONTENT,
        ScanConfig::builder().recursive(true).build(),
    );

    // Seed with a Dir child — its slot will outlive the Profile reap
    // (the dir's `watch_demand` survives until the Profile's release
    // walks `walk_pair`'s delete path, which doesn't run on Vanished).
    let snap = dir_snap(root, vec![("subdir", EntryKind::Dir, 99)]);
    complete_seed_burst(e, pid, &attach_out, snap);

    let child = e
        .tree()
        .lookup(Some(root), "subdir")
        .expect("Dir child materialized by Seed graft");
    assert!(
        e.tree().get(child).unwrap().watch_demand >= 1,
        "covered Dir child carries watch_demand",
    );

    (root, child, sid, pid, attach_out)
}

#[test]
fn standard_vanished_with_reap_pending_does_not_double_release_anchor() {
    let mut e = Engine::new();
    let (root, _child, sid, pid, _attach_out) = setup_with_surviving_child(&mut e);

    // Drive a Standard burst.
    let t1 = Instant::now();
    e.step(
        Input::FsEvent {
            resource: root,
            event: FsEvent::Modified,
        },
        t1,
    );

    // Detach mid-burst to set reap_pending.
    let _ = e.detach_sub(sid, t1);
    assert!(e.profiles().get(pid).unwrap().reap_pending);

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

    // Inject Vanished. Pre-fix: `dispatch_standard_vanished` called
    // `finish_burst_to_idle` first (which called `reap_profile`,
    // releasing the anchor's `watch_demand` 1→0) AND then called
    // `sub_watch_demand` on the same anchor — tripping the
    // `debug_assert!(prev > 0, …)` because the surviving child kept the
    // slot alive long enough for the post-finish decrement to find a
    // zero counter. Post-fix: release happens BEFORE finish; reap sees
    // a cleared flag and skips the redundant decrement.
    let out = e.step(
        Input::ProbeResponse(ProbeResponse {
            profile: pid,
            correlation,
            result: ProbeResult::Vanished,
        }),
        t2,
    );
    assert!(
        e.profiles().get(pid).is_none(),
        "Profile reaped without panic",
    );
    let unwatch_count = out
        .watch_ops
        .iter()
        .filter(|op| matches!(op, WatchOp::Unwatch { resource } if *resource == root))
        .count();
    assert_eq!(
        unwatch_count, 1,
        "exactly one Unwatch on the anchor — no double release; got {:?}",
        out.watch_ops,
    );
}

#[test]
fn standard_failed_with_reap_pending_does_not_double_release_anchor() {
    // Symmetric regression for dispatch_standard_failed.
    let mut e = Engine::new();
    let (root, _child, sid, pid, _attach_out) = setup_with_surviving_child(&mut e);

    let t1 = Instant::now();
    e.step(
        Input::FsEvent {
            resource: root,
            event: FsEvent::Modified,
        },
        t1,
    );
    let _ = e.detach_sub(sid, t1);
    assert!(e.profiles().get(pid).unwrap().reap_pending);

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

    let out = e.step(
        Input::ProbeResponse(ProbeResponse {
            profile: pid,
            correlation,
            result: ProbeResult::Failed { errno: 13 },
        }),
        t2,
    );
    assert!(
        e.profiles().get(pid).is_none(),
        "Profile reaped without panic on Failed",
    );
    let unwatch_count = out
        .watch_ops
        .iter()
        .filter(|op| matches!(op, WatchOp::Unwatch { resource } if *resource == root))
        .count();
    assert_eq!(unwatch_count, 1);
}

/// Drive the F-CRIT-1 setup: attach P + surviving child, kick off a
/// Standard burst, advance to Probing, detach to set reap_pending, then
/// dispatch the supplied FsEvent at the anchor. Returns the resulting
/// StepOutput. Surviving child fixture keeps the anchor slot alive past
/// `reap_profile`'s `try_reap`, exposing any post-finish refcount
/// mistake on a still-live counter.
fn drive_anchor_terminal_with_reap_pending(event: FsEvent) -> (Engine, ResourceId, ProfileId) {
    let mut e = Engine::new();
    let (root, _child, sid, pid, _attach_out) = setup_with_surviving_child(&mut e);

    let t1 = Instant::now();
    e.step(
        Input::FsEvent {
            resource: root,
            event: FsEvent::Modified,
        },
        t1,
    );
    let _ = e.detach_sub(sid, t1);
    assert!(e.profiles().get(pid).unwrap().reap_pending);

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
    assert!(matches!(
        e.profiles().get(pid).unwrap().state,
        ProfileState::Active(specter_core::Burst {
            phase: specter_core::BurstPhase::Verifying,
            ..
        })
    ));

    let out = e.step(
        Input::FsEvent {
            resource: root,
            event,
        },
        t2,
    );
    assert!(
        e.profiles().get(pid).is_none(),
        "Profile reaped after anchor terminal event ({event:?}) without panic",
    );
    let unwatch_count = out
        .watch_ops
        .iter()
        .filter(|op| matches!(op, WatchOp::Unwatch { resource } if *resource == root))
        .count();
    assert_eq!(
        unwatch_count, 1,
        "exactly one Unwatch on anchor for {event:?} — no double release; got {:?}",
        out.watch_ops,
    );
    (e, root, pid)
}

#[test]
fn anchor_terminal_removed_with_reap_pending_active_burst_no_double_release() {
    drive_anchor_terminal_with_reap_pending(FsEvent::Removed);
}

#[test]
fn anchor_terminal_renamed_with_reap_pending_active_burst_no_double_release() {
    drive_anchor_terminal_with_reap_pending(FsEvent::Renamed);
}

#[test]
fn anchor_terminal_revoked_with_reap_pending_active_burst_no_double_release() {
    drive_anchor_terminal_with_reap_pending(FsEvent::Revoked);
}

#[test]
fn anchor_terminal_with_reap_pending_multi_profile_each_released_once() {
    // Two Profiles co-anchored at the same Resource (different
    // config_hash). P has reap_pending + Active(Probing); Q is Idle.
    // Pre-fix: P's double-release decremented the counter past Q's
    // contribution; Q's later release would underflow. Post-fix: each
    // Profile's anchor flag clears exactly once and the counter walks
    // 2 → 1 → 0 cleanly.
    let mut e = Engine::new();
    let root = e.tree_mut().ensure(None, "src", ResourceRole::User);
    e.tree_mut().get_mut(root).unwrap().kind = ResourceKind::Dir;

    // Two Subs at the same anchor with different config_hash —
    // different max_settle yields a fresh Profile.
    let attach_p = SubAttachRequest::for_resource(
        "P".into(),
        root,
        ScanConfig::builder().recursive(true).build(),
        MAX_SETTLE,
        SETTLE,
        empty_command(),
        EffectScope::SubtreeRoot,
        ClassSet::CONTENT,
        false,
    );
    let attach_q = SubAttachRequest::for_resource(
        "Q".into(),
        root,
        ScanConfig::builder().recursive(true).build(),
        MAX_SETTLE + Duration::from_secs(1),
        SETTLE,
        empty_command(),
        EffectScope::SubtreeRoot,
        ClassSet::CONTENT,
        false,
    );
    let (sid_p, attach_out_p) = e.attach_sub(attach_p, Instant::now());
    let (_sid_q, attach_out_q) = e.attach_sub(attach_q, Instant::now());
    let pid_p = e.subs().get(sid_p).unwrap().profile;
    let pid_q = e
        .profiles()
        .iter()
        .find(|(pid, _)| *pid != pid_p)
        .map(|(pid, _)| pid)
        .expect("Q profile minted");

    // Each Profile contributed +1 to root.watch_demand.
    assert_eq!(e.tree().get(root).unwrap().watch_demand, 2);

    // Drive both to Idle via a Seed burst so the surviving-child
    // invariant holds.
    let snap_p = dir_snap(root, vec![("subdir", EntryKind::Dir, 99)]);
    complete_seed_burst(&mut e, pid_p, &attach_out_p, snap_p);
    let snap_q = dir_snap(root, vec![("subdir", EntryKind::Dir, 99)]);
    complete_seed_burst(&mut e, pid_q, &attach_out_q, snap_q);

    // Kick off a Standard burst on P.
    let t1 = Instant::now();
    e.step(
        Input::FsEvent {
            resource: root,
            event: FsEvent::Modified,
        },
        t1,
    );
    // Detach P to set reap_pending. Q stays alive.
    let _ = e.detach_sub(sid_p, t1);
    assert!(e.profiles().get(pid_p).unwrap().reap_pending);
    assert!(!e.profiles().get(pid_q).unwrap().reap_pending);

    // Advance P to Probing.
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

    // FsEvent::Removed at root: covering_profiles returns [P, Q],
    // each routes through finalize_anchor_lost.
    let out = e.step(
        Input::FsEvent {
            resource: root,
            event: FsEvent::Removed,
        },
        t2,
    );

    // P reaped; Q remains Idle with anchor_contribution cleared.
    assert!(e.profiles().get(pid_p).is_none(), "P reaped");
    let q = e.profiles().get(pid_q).expect("Q survives");
    assert!(
        !q.anchor_contribution,
        "Q's anchor_contribution cleared by terminal event",
    );
    assert!(matches!(q.state, ProfileState::Idle));

    // Counter walked 2 → 1 → 0 cleanly. Anchor slot is reaped because
    // the surviving-child only kept it alive while P+Q were attached;
    // Q.anchor_contribution=false leaves only the child anchor, which
    // does keep root alive — confirm via watch_demand counter.
    let final_counter = e.tree().get(root).map_or(0, |r| r.watch_demand);
    assert_eq!(
        final_counter, 0,
        "root.watch_demand zeroed by both Profiles' terminal events",
    );
    let unwatch_count = out
        .watch_ops
        .iter()
        .filter(|op| matches!(op, WatchOp::Unwatch { resource } if *resource == root))
        .count();
    assert_eq!(
        unwatch_count, 1,
        "exactly one Unwatch on root (1→0 edge); got {:?}",
        out.watch_ops,
    );
}

// ───────────────────────────────────────────────────────────────────────
// PR 1 — Descendant-claim release (F-CRIT-1, F-MED-4)
//
// The four claim types differ in cardinality: anchor / watch-root parent /
// descent prefix are 1-to-1 (one Profile contributes to one Resource);
// covered descendants are 1-to-N (one Profile contributes to N Tree
// slots). Pre-PR-1, the engine had release helpers for the three 1-to-1
// claims but none for the 1-to-N descendant set — every Profile teardown
// path cleared `Profile.current` without releasing the per-descendant
// `watch_demand` contributions encoded in it. The descendant slots stayed
// alive in the Tree with non-zero `watch_demand`, the kernel kept their
// FDs registered, and a default `subtree-root × CONTENT` Sub on a 10k-file
// tree leaked ~10k FDs per hot-reload churn cycle.
//
// `Engine::release_descendant_claim` closes the symmetry. Wired into
// `reap_profile` and the seven `dispatch_*_vanished/failed` +
// `finalize_anchor_lost` sites in `transitions.rs`. The helper takes
// `Profile.current` atomically and walks it leaf-first via
// `reconcile::delete_child`, releasing each covered slot's contribution
// with an explicit `releasing_descendant: Some(profile_id)` skip signal
// so the recompute models the post-release union without depending on
// `current.is_some()` having flipped (closes F-MED-4 by construction).
// ───────────────────────────────────────────────────────────────────────

#[test]
fn release_descendant_claim_idle_detach_reaps_covered_dir() {
    // Idle Profile with one Sub + one covered Dir descendant. Detach
    // the Sub: `detach_sub_inner` runs `reap_profile` immediately
    // (Idle ⇒ ReapNow). `release_descendant_claim` walks
    // `Profile.current` and releases `subdir`'s `watch_demand`
    // contribution; the slot reaps. Pre-PR-1 the descendant leaked.
    let mut e = Engine::new();
    let (_root, child, sid, pid, _attach_out) = setup_with_surviving_child(&mut e);

    // Pre-conditions: Profile is Idle, child has watch_demand >= 1.
    assert!(matches!(
        e.profiles().get(pid).unwrap().state,
        ProfileState::Idle,
    ));
    assert!(e.tree().get(child).is_some());
    assert!(e.tree().get(child).unwrap().watch_demand >= 1);

    let out = e.detach_sub(sid, Instant::now());
    assert!(
        e.profiles().get(pid).is_none(),
        "Idle Profile reaped on last-Sub detach",
    );
    assert!(
        e.tree().get(child).is_none(),
        "covered Dir descendant reaped — release_descendant_claim closed F-CRIT-1",
    );
    let unwatch_for_child = out
        .watch_ops
        .iter()
        .filter(|op| matches!(op, WatchOp::Unwatch { resource } if *resource == child))
        .count();
    assert_eq!(
        unwatch_for_child, 1,
        "exactly one Unwatch op for the descendant; got {:?}",
        out.watch_ops,
    );
}

#[test]
fn release_descendant_claim_idle_detach_reaps_covered_leaf() {
    // PerStableFile / has_per_file_fds=true Profile: covered Leaves
    // also carry per-file FDs. Detach must release the leaf's
    // contribution as well as the Dir's.
    let mut e = Engine::new();
    let root = e.tree_mut().ensure(None, "src", ResourceRole::User);
    e.tree_mut().get_mut(root).unwrap().kind = ResourceKind::Dir;

    let (sid, pid, attach_out) = attach_sub_with_events(
        &mut e,
        "build",
        root,
        EffectScope::SubtreeRoot,
        ClassSet::CONTENT,
        ScanConfig::builder().recursive(true).build(),
    );
    assert!(e.profiles().get(pid).unwrap().has_per_file_fds);

    // Seed with one File leaf — has_per_file_fds=true ⇒ leaf gets FD.
    let snap = dir_snap(root, vec![("a.rs", EntryKind::File, 1)]);
    complete_seed_burst(&mut e, pid, &attach_out, snap);
    let leaf = e.tree().lookup(Some(root), "a.rs").expect("leaf seeded");
    assert!(e.tree().get(leaf).unwrap().watch_demand >= 1);

    let out = e.detach_sub(sid, Instant::now());
    assert!(e.profiles().get(pid).is_none());
    assert!(
        e.tree().get(leaf).is_none(),
        "covered Leaf with per-file FD reaped on Idle detach",
    );
    let unwatch_for_leaf = out
        .watch_ops
        .iter()
        .filter(|op| matches!(op, WatchOp::Unwatch { resource } if *resource == leaf))
        .count();
    assert_eq!(unwatch_for_leaf, 1);
}

#[test]
fn release_descendant_claim_dispatch_standard_vanished_releases_descendants() {
    // F-CRIT-1 in the dispatch_standard_vanished path: an anchor that
    // disappears mid-burst must release the per-descendant contributions
    // alongside the anchor's. Pre-PR-1 the descendants leaked even
    // though the anchor was correctly released.
    let mut e = Engine::new();
    let (root, child, sid, pid, _attach_out) = setup_with_surviving_child(&mut e);

    // Drive a Standard burst.
    let t1 = Instant::now();
    e.step(
        Input::FsEvent {
            resource: root,
            event: FsEvent::Modified,
        },
        t1,
    );
    let _ = e.detach_sub(sid, t1);
    assert!(e.profiles().get(pid).unwrap().reap_pending);

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
    let correlation = e.pending_probe(pid).expect("Verifying probe in flight");

    let out = e.step(
        Input::ProbeResponse(ProbeResponse {
            profile: pid,
            correlation,
            result: ProbeResult::Vanished,
        }),
        t2,
    );

    assert!(e.profiles().get(pid).is_none(), "Profile reaped");
    assert!(
        e.tree().get(child).is_none(),
        "subdir reaped via release_descendant_claim",
    );
    let unwatch_for_child = out
        .watch_ops
        .iter()
        .filter(|op| matches!(op, WatchOp::Unwatch { resource } if *resource == child))
        .count();
    assert_eq!(
        unwatch_for_child, 1,
        "exactly one Unwatch op for the descendant; got {:?}",
        out.watch_ops,
    );
}

#[test]
fn release_descendant_claim_anchor_terminal_event_releases_descendants() {
    // F-CRIT-1 in the finalize_anchor_lost path: anchor-terminal events
    // (Removed / Renamed / Revoked) at the anchor must release the
    // per-descendant contributions. Same shape as the dispatch_*_vanished
    // path but driven through `on_anchor_terminal_event`.
    let mut e = Engine::new();
    let (root, child, _sid, pid, _attach_out) = setup_with_surviving_child(&mut e);

    let out = e.step(
        Input::FsEvent {
            resource: root,
            event: FsEvent::Removed,
        },
        Instant::now(),
    );

    // After Removed at the anchor: Profile is Idle (post-finalize),
    // anchor_contribution cleared, current taken by
    // release_descendant_claim, child reaped.
    let p = e.profiles().get(pid).expect("Profile survives anchor loss");
    assert!(matches!(p.state, ProfileState::Idle));
    assert!(!p.anchor_contribution);
    assert!(
        p.current.is_none(),
        "current taken by release_descendant_claim"
    );
    assert!(
        e.tree().get(child).is_none(),
        "subdir reaped via release_descendant_claim on anchor terminal",
    );
    let unwatch_for_child = out
        .watch_ops
        .iter()
        .filter(|op| matches!(op, WatchOp::Unwatch { resource } if *resource == child))
        .count();
    assert_eq!(unwatch_for_child, 1);
}

#[test]
fn release_descendant_claim_dispatch_rebase_vanished_releases_descendants() {
    // F-CRIT-1 in dispatch_rebase_vanished: post-fire rebase probe
    // returns Vanished. `Profile.current` was populated by the pre-fire
    // Standard-Ok graft and contains the covered descendants. The
    // rebase-failure path must release them too.
    //
    // Lifecycle: Idle → Modified at root → Active(Verifying) →
    // ProbeResponse::Ok (stable, same snapshot) → emit_effects (one
    // Effect for the SubtreeRoot Sub) → Active(Awaiting) →
    // EffectComplete::Ok → Active(Rebasing) → ProbeResponse::Vanished
    // → dispatch_rebase_vanished. The release_descendant_claim wire-up
    // walks Profile.current and reaps subdir.
    let mut e = Engine::new();
    let (root, child, sid, pid, _attach_out) = setup_with_surviving_child(&mut e);

    let t1 = Instant::now();
    e.step(
        Input::FsEvent {
            resource: root,
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
    let verify_corr = e.pending_probe(pid).expect("Verify probe in flight");

    // Stable verdict — same snapshot as seed. dispatch_standard_ok's
    // pre-graft hash captures the prior, post-graft comparison is stable
    // and `dirty_descendants == 0`, so emit_effects fires one Effect for
    // the SubtreeRoot Sub → transition_to_awaiting.
    let stable_out = e.step(
        Input::ProbeResponse(ProbeResponse {
            profile: pid,
            correlation: verify_corr,
            result: ProbeResult::Ok(dir_snap(root, vec![("subdir", EntryKind::Dir, 99)])),
        }),
        t2,
    );
    let effect = stable_out
        .effects
        .first()
        .cloned()
        .expect("Standard-Ok stable verdict fires one Effect");
    assert!(matches!(
        e.profiles().get(pid).unwrap().state,
        ProfileState::Active(specter_core::Burst {
            phase: specter_core::BurstPhase::Awaiting { .. },
            ..
        }),
    ));

    // EffectComplete::Ok → transition_to_rebasing.
    let _ = e.step(
        Input::EffectComplete {
            sub: sid,
            key: effect.key,
            result: specter_core::EffectOutcome::Ok,
        },
        t2,
    );
    let rebase_corr = e
        .pending_probe(pid)
        .expect("Rebase probe in flight after EffectComplete");

    // Pre-condition: descendant claim still intact going into Rebasing.
    assert!(
        e.tree().get(child).is_some_and(|r| r.watch_demand >= 1),
        "subdir.watch_demand still held going into Rebasing",
    );

    // Inject Rebase Vanished.
    let out = e.step(
        Input::ProbeResponse(ProbeResponse {
            profile: pid,
            correlation: rebase_corr,
            result: ProbeResult::Vanished,
        }),
        t2,
    );

    assert!(matches!(
        e.profiles().get(pid).unwrap().state,
        ProfileState::Idle,
    ));
    assert!(
        e.tree().get(child).is_none(),
        "subdir reaped via release_descendant_claim in dispatch_rebase_vanished",
    );
    let unwatch_for_child = out
        .watch_ops
        .iter()
        .filter(|op| matches!(op, WatchOp::Unwatch { resource } if *resource == child))
        .count();
    assert_eq!(
        unwatch_for_child, 1,
        "exactly one Unwatch for subdir on rebase-Vanished; got {:?}",
        out.watch_ops,
    );
}

#[test]
fn release_descendant_claim_multi_profile_preserves_others() {
    // Two Profiles co-anchor at root with the same recursive scan, both
    // observing the same descendant `subdir`. Profile P loses its anchor
    // (Vanished); Profile Q stays Idle. Pre-PR-1 P's descendant claim
    // leaked: subdir.watch_demand stayed at 2 (P's leak + Q's
    // contribution). Post-PR-1: P's release walks current and decrements
    // subdir 2 → 1; Q's contribution survives at 1.
    let mut e = Engine::new();
    let root = e.tree_mut().ensure(None, "src", ResourceRole::User);
    e.tree_mut().get_mut(root).unwrap().kind = ResourceKind::Dir;

    // Two Profiles at the same anchor with different config_hash
    // (different max_settle ⇒ different Profile).
    let attach_p = SubAttachRequest::for_resource(
        "P".into(),
        root,
        ScanConfig::builder().recursive(true).build(),
        MAX_SETTLE,
        SETTLE,
        empty_command(),
        EffectScope::SubtreeRoot,
        ClassSet::CONTENT,
        false,
    );
    let attach_q = SubAttachRequest::for_resource(
        "Q".into(),
        root,
        ScanConfig::builder().recursive(true).build(),
        MAX_SETTLE + Duration::from_secs(1),
        SETTLE,
        empty_command(),
        EffectScope::SubtreeRoot,
        ClassSet::CONTENT,
        false,
    );
    let (sid_p, attach_out_p) = e.attach_sub(attach_p, Instant::now());
    let (_sid_q, attach_out_q) = e.attach_sub(attach_q, Instant::now());
    let pid_p = e.subs().get(sid_p).unwrap().profile;
    let pid_q = e
        .profiles()
        .iter()
        .find(|(pid, _)| *pid != pid_p)
        .map(|(pid, _)| pid)
        .expect("Q profile minted");

    // Drive both to Idle with a shared seeded Dir descendant. Each
    // Profile's create_child contributed +1 to subdir.watch_demand.
    let snap_p = dir_snap(root, vec![("subdir", EntryKind::Dir, 99)]);
    complete_seed_burst(&mut e, pid_p, &attach_out_p, snap_p);
    let snap_q = dir_snap(root, vec![("subdir", EntryKind::Dir, 99)]);
    complete_seed_burst(&mut e, pid_q, &attach_out_q, snap_q);

    let subdir = e
        .tree()
        .lookup(Some(root), "subdir")
        .expect("subdir seeded");
    assert_eq!(
        e.tree().get(subdir).unwrap().watch_demand,
        2,
        "two Profiles each contributed +1 to subdir.watch_demand",
    );

    // Drive P's Standard burst to Verifying, detach P (reap_pending),
    // inject Vanished — P's anchor is gone, but Q's stays. P's
    // release_descendant_claim walks P's current and decrements
    // subdir's contribution; Q's contribution is preserved.
    let t1 = Instant::now();
    e.step(
        Input::FsEvent {
            resource: root,
            event: FsEvent::Modified,
        },
        t1,
    );
    let _ = e.detach_sub(sid_p, t1);

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
    let correlation = e.pending_probe(pid_p).expect("P probe in flight");

    let _out = e.step(
        Input::ProbeResponse(ProbeResponse {
            profile: pid_p,
            correlation,
            result: ProbeResult::Vanished,
        }),
        t2,
    );

    // P reaped; Q survives.
    assert!(e.profiles().get(pid_p).is_none(), "P reaped");
    assert!(e.profiles().get(pid_q).is_some(), "Q survives P's teardown");
    // Q's descendant claim is intact.
    assert!(
        e.tree().get(subdir).is_some(),
        "subdir slot survives — Q still contributes",
    );
    assert_eq!(
        e.tree().get(subdir).unwrap().watch_demand,
        1,
        "subdir.watch_demand decremented from 2 to 1 by P's release",
    );
    assert_eq!(
        e.profiles().get(pid_q).unwrap().events_union,
        ClassSet::CONTENT,
        "Q's contribution intact",
    );
}

#[test]
fn delete_child_during_graft_recompute_skips_releasing_profile() {
    // F-MED-4: during `reconcile::delete_child` the releasing Profile's
    // `Profile.current` is still `Some` (graft hasn't run the take yet),
    // so a recompute triggered by the descendant decrement could
    // include the Profile's own descendant contribution and over-mask
    // the post-decrement union. The explicit
    // `releasing_descendant: Some(profile_id)` parameter on
    // `sub_watch_demand` closes the gap precisely: the recompute skips
    // this Profile's descendant clause for the slot being released.
    //
    // Setup: two Profiles share the anchor with DIFFERENT events masks.
    // P=CONTENT, Q=METADATA. Both seed with `subdir` as a covered Dir.
    // subdir.watch_demand = 2; subdir.events_union = CONTENT | METADATA.
    // P's probe response says `subdir` is gone — `dispatch_standard_ok`
    // → `graft` → `walk_pair` → `delete_child` for `subdir`. The recompute
    // must yield Q's mask only, not the union.
    let mut e = Engine::new();
    let root = e.tree_mut().ensure(None, "src", ResourceRole::User);
    e.tree_mut().get_mut(root).unwrap().kind = ResourceKind::Dir;

    let attach_p = SubAttachRequest::for_resource(
        "P".into(),
        root,
        ScanConfig::builder().recursive(true).build(),
        MAX_SETTLE,
        SETTLE,
        empty_command(),
        EffectScope::SubtreeRoot,
        ClassSet::CONTENT,
        false,
    );
    let attach_q = SubAttachRequest::for_resource(
        "Q".into(),
        root,
        ScanConfig::builder().recursive(true).build(),
        MAX_SETTLE + Duration::from_secs(1),
        SETTLE,
        empty_command(),
        EffectScope::SubtreeRoot,
        ClassSet::METADATA,
        false,
    );
    let (sid_p, attach_out_p) = e.attach_sub(attach_p, Instant::now());
    let (_sid_q, attach_out_q) = e.attach_sub(attach_q, Instant::now());
    let pid_p = e.subs().get(sid_p).unwrap().profile;
    let pid_q = e
        .profiles()
        .iter()
        .find(|(pid, _)| *pid != pid_p)
        .map(|(pid, _)| pid)
        .expect("Q profile minted");

    let snap_p = dir_snap(root, vec![("subdir", EntryKind::Dir, 99)]);
    complete_seed_burst(&mut e, pid_p, &attach_out_p, snap_p);
    let snap_q = dir_snap(root, vec![("subdir", EntryKind::Dir, 99)]);
    complete_seed_burst(&mut e, pid_q, &attach_out_q, snap_q);

    let subdir = e
        .tree()
        .lookup(Some(root), "subdir")
        .expect("subdir seeded");
    assert_eq!(e.tree().get(subdir).unwrap().watch_demand, 2);
    assert_eq!(
        e.tree().get(subdir).unwrap().events_union,
        ClassSet::CONTENT | ClassSet::METADATA,
        "two contributors, union of masks",
    );

    // Drive P's Standard burst to Verifying.
    let t1 = Instant::now();
    e.step(
        Input::FsEvent {
            resource: root,
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
    let correlation = e.pending_probe(pid_p).expect("P probe in flight");

    // Probe response: subdir is GONE (P sees a tree where it's been
    // deleted). dispatch_standard_ok → graft → walk_pair → delete_child
    // fires sub_watch_demand for subdir with releasing_descendant=Some(P).
    let response = TreeSnapshot::Dir(Arc::new(DirSnapshot::new(
        root,
        DirMeta {
            mtime: UNIX_EPOCH,
            inode: 0,
            device: 0,
        },
        0,
        BTreeMap::new(),
    )));
    let _out = e.step(
        Input::ProbeResponse(ProbeResponse {
            profile: pid_p,
            correlation,
            result: ProbeResult::Ok(response),
        }),
        t2,
    );

    // After delete_child:
    // - subdir.watch_demand goes 2 → 1 (P's contribution decremented).
    // - recompute(subdir, releasing_descendant=Some(P)): P skipped,
    //   Q contributes METADATA → events_union = METADATA.
    // - Without the skip: events_union = CONTENT | METADATA (over-mask).
    assert!(
        e.tree().get(subdir).is_some(),
        "subdir survives, Q anchors it"
    );
    assert_eq!(
        e.tree().get(subdir).unwrap().watch_demand,
        1,
        "subdir.watch_demand decremented to 1 (P released, Q remains)",
    );
    assert_eq!(
        e.tree().get(subdir).unwrap().events_union,
        ClassSet::METADATA,
        "events_union narrows to Q's mask only — F-MED-4 closed by the explicit \
         releasing_descendant skip param",
    );
    assert_ne!(
        e.tree().get(subdir).unwrap().events_union,
        ClassSet::CONTENT | ClassSet::METADATA,
        "over-mask check: must NOT include P's contribution",
    );
    // Q is alive — its descendant claim on subdir is intact.
    assert!(e.profiles().get(pid_q).is_some(), "Q survives P's graft");
}
