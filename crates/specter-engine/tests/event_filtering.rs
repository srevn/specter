//! Engine-level integration tests for the event-filtering primitive. Each test exercises the
//! user-facing invariants without spinning up a real kqueue: the entry filter, the mask-fork (mask
//! ∈ `config_hash`), the anchor-bypass, the descent-prefix STRUCTURE-only contribution, and the E2E
//! #3 closure path (`subtree-root × default events ⇒ has_per_file_fds = true ⇒ per-file FDs on
//! covered Leaves).
//!
//! Where the equivalent shape lives in `transitions_tests.rs`, this file is the **integration**
//! counterpart — exercising whole-Engine flows (attach → Seed-Ok → FsEvent → optional Effect)
//! rather than dispatch handlers in isolation.

use specter_core::testkit::{dir_snap, proven};
use specter_core::{
    AnchorClaim, BurstFinish, ClassSet, DedupKey, Diagnostic, DirMeta, DirSnapshot, EntryKind,
    FsEvent, FsIdentity, Input, ProbeFailure, ProbeOutcome, ProbeResponse, ProfileId, ProfileState,
    ResourceId, ResourceKind, ResourceRole, ScanConfig, SubAttachAnchor, WatchOp,
};
use specter_engine::Engine;
use specter_engine::testkit::{
    anchor_dir, assert_seed_verifying, attach, attach_returning, complete_effect_to_rebasing,
    drain_due, pre_place_dir, seed_to_idle, verify,
};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant, UNIX_EPOCH};

const SETTLE: Duration = Duration::from_millis(100);
const MAX_SETTLE: Duration = Duration::from_secs(6);

// ───────────────────────────────────────────────────────────────────────
// IT-EF-1 — DEFAULT_SUBTREE_ROOT enables per-file FDs (closes E2E #3)
//
// `echo 'test' > file.txt` inside a subtree-root watched dir receives an event on the per-file FD
// (CONTENT class drives the translator to emit `NOTE_WRITE | NOTE_EXTEND` on every covered Leaf), the
// burst fires, the probe runs, the diff classifies the file as Modified, and the user's command runs.
//
// At engine level this manifests as: a `subtree-root` Sub with default events (`STRUCTURE |
// CONTENT`) drives `has_per_file_fds = true`, so `graft`'s `apply_diff_to_tree` emits a per-file
// `WatchOp::Watch` for every covered Leaf during reconciliation. The Watch op carries the Profile's
// mask (which translates downstream to `NOTE_WRITE | NOTE_EXTEND` on the file FD).
// ───────────────────────────────────────────────────────────────────────

#[test]
fn it_ef_1_default_subtree_root_emits_per_file_watch_on_leaves() {
    let mut e = Engine::new();
    let root = anchor_dir(&mut e, "src");

    let t0 = Instant::now();
    let (_sid, pid, _attach_out) = attach_returning(
        &mut e,
        "build",
        SubAttachAnchor::Resource(root),
        ScanConfig::builder().recursive(true).build(),
        ClassSet::DEFAULT_SUBTREE_ROOT,
        MAX_SETTLE,
        t0,
    );

    // Default for SubtreeRoot ⇒ has_per_file_fds = true.
    assert!(
        e.profiles().get(pid).unwrap().has_per_file_fds(),
        "default subtree-root mask (STRUCTURE|CONTENT) sets has_per_file_fds",
    );

    // The Seed probe fires one settle window after attach, not at attach. Drive the first Seed
    // probe with one File child. graft runs against prior=None ⇒ `Diff::all_created` pure-create
    // path. The first sample is `Retry` by construction (no prior certified hash) so it does not
    // pin, but `apply_snapshot` still grafts: with has_per_file_fds=true the File child gets a
    // Watch op on this very (first) response.
    let (corr, at) = assert_seed_verifying(&mut e, pid, t0);
    let snap = dir_snap(&[("file.txt", EntryKind::File, 1)]);
    let seed_out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid,
            correlation: corr,
            outcome: proven(snap),
        }),
        at,
    );

    // Resource was materialized.
    let file_id = e.tree().lookup(Some(root), "file.txt").expect("file slot");

    // The seed response emits a Watch op for the per-file FD.
    let saw_per_file_watch = seed_out.watch_ops.iter().any(|op| match op {
        WatchOp::Watch {
            resource, events, ..
        } => *resource == file_id && *events == ClassSet::DEFAULT_SUBTREE_ROOT,
        WatchOp::Unwatch { .. } => false,
    });
    assert!(
        saw_per_file_watch,
        "covered file leaf gets a Watch with the Profile's mask (closes E2E #3); got watch_ops = {:?}",
        seed_out.watch_ops,
    );
}

#[test]
fn it_ef_1_structure_only_subtree_does_not_emit_per_file_watch() {
    // Negative case: a Sub explicitly requesting STRUCTURE only does NOT get per-file FDs. Confirms
    // `has_per_file_fds` is mask-driven, not scope-driven.
    let mut e = Engine::new();
    let root = anchor_dir(&mut e, "src");

    let t0 = Instant::now();
    let (_sid, pid, _attach_out) = attach_returning(
        &mut e,
        "ls-only",
        SubAttachAnchor::Resource(root),
        ScanConfig::builder().recursive(true).build(),
        ClassSet::STRUCTURE,
        MAX_SETTLE,
        t0,
    );

    assert!(
        !e.profiles().get(pid).unwrap().has_per_file_fds(),
        "STRUCTURE-only mask leaves has_per_file_fds = false",
    );

    // The first Seed probe fires one settle window after attach.
    let (corr, at) = assert_seed_verifying(&mut e, pid, t0);
    let snap = dir_snap(&[("file.txt", EntryKind::File, 1)]);
    let seed_out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid,
            correlation: corr,
            outcome: proven(snap),
        }),
        at,
    );

    let file_id = e.tree().lookup(Some(root), "file.txt").expect("file slot");
    let saw_per_file_watch = seed_out.watch_ops.iter().any(|op| match op {
        WatchOp::Watch { resource, .. } => *resource == file_id,
        WatchOp::Unwatch { .. } => false,
    });
    assert!(
        !saw_per_file_watch,
        "STRUCTURE-only Profile must not emit per-file Watch ops",
    );
}

// ───────────────────────────────────────────────────────────────────────
// IT-EF-2 — Two Subs with different masks fork separate Profiles
//
// `events` folds into `ProfileIdentity::config_hash`, so two Subs at the same resource that differ
// only on `events` partition into two distinct Profiles. This guards against the "Profile-union
// infection" problem: a chmod on a Sub asking only for CONTENT must not fire that Sub's command via
// the same Profile that handles a sibling Sub asking for METADATA.
// ───────────────────────────────────────────────────────────────────────

#[test]
fn it_ef_2_two_subs_different_masks_fork_separate_profiles() {
    let mut e = Engine::new();
    let root = anchor_dir(&mut e, "build");

    let (_sid_a, pid_a) = attach(
        &mut e,
        "make",
        SubAttachAnchor::Resource(root),
        ScanConfig::builder().recursive(true).build(),
        ClassSet::CONTENT,
        MAX_SETTLE,
        Instant::now(),
    );
    let (_sid_b, pid_b) = attach(
        &mut e,
        "audit",
        SubAttachAnchor::Resource(root),
        ScanConfig::builder().recursive(true).build(),
        ClassSet::METADATA,
        MAX_SETTLE,
        Instant::now(),
    );

    assert_ne!(
        pid_a, pid_b,
        "Subs with distinct events forks fork into distinct Profiles",
    );

    // Both Profiles record their own mask; the per-resource union contains both bits because both
    // contribute to the anchor.
    assert_eq!(e.profiles().get(pid_a).unwrap().events(), ClassSet::CONTENT,);
    assert_eq!(
        e.profiles().get(pid_b).unwrap().events(),
        ClassSet::METADATA,
    );
    assert_eq!(
        e.tree().get(root).unwrap().events_union(),
        ClassSet::CONTENT | ClassSet::METADATA,
        "anchor's per-Resource union ORs both Profiles' contributions",
    );
    let _ = e.cancel_all_in_flight_probes();
}

// ───────────────────────────────────────────────────────────────────────
// IT-EF-2b — Profile.events() is invariant across multi-Sub churn
//
// Two Subs at the same resource with identical (config, max_settle, events) share ONE Profile. The
// Profile's mask is fixed at construction and survives both a sibling join (the join does not
// re-derive it) and a sibling detach (the Profile lives while ≥1 Sub remains). Pins that the
// per-Profile mask is invariant under Sub churn.
// ───────────────────────────────────────────────────────────────────────

#[test]
fn it_ef_2b_profile_events_invariant_across_sub_attach_detach() {
    let mut e = Engine::new();
    let root = anchor_dir(&mut e, "src");

    let mask = ClassSet::CONTENT | ClassSet::METADATA;
    let cfg = || ScanConfig::builder().recursive(true).build();

    let (sid_a, pid_a) = attach(
        &mut e,
        "a",
        SubAttachAnchor::Resource(root),
        cfg(),
        mask,
        MAX_SETTLE,
        Instant::now(),
    );
    assert_eq!(
        e.profiles().get(pid_a).unwrap().events(),
        mask,
        "fresh Profile records the attaching Sub's mask",
    );

    // A sibling with identical identity joins the SAME Profile.
    let (sid_b, pid_b) = attach(
        &mut e,
        "b",
        SubAttachAnchor::Resource(root),
        cfg(),
        mask,
        MAX_SETTLE,
        Instant::now(),
    );
    assert_eq!(pid_a, pid_b, "identical identity ⇒ one shared Profile");
    assert_eq!(
        e.profiles().get(pid_a).unwrap().events(),
        mask,
        "the join does not re-derive events()",
    );

    // Detaching one sibling leaves the Profile alive (b still attached) with its mask unchanged.
    let _ = e.step(Input::DetachSub(sid_a), Instant::now());
    let p = e
        .profiles()
        .get(pid_a)
        .expect("Profile lives while ≥1 Sub remains");
    assert_eq!(
        p.events(),
        mask,
        "events() is invariant across a sibling detach",
    );

    // The surviving Sub still belongs to that same Profile.
    assert_eq!(e.subs().get(sid_b).unwrap().profile(), pid_a);
    let _ = e.cancel_all_in_flight_probes();
}

#[test]
fn it_ef_2_chmod_only_fires_metadata_profile_not_content_profile() {
    // Concrete chmod scenario: Sub A wants CONTENT, Sub B wants METADATA. After a `MetadataChanged`
    // at the anchor, Profile B drives a burst; Profile A's class filter would drop it… EXCEPT
    // anchor events bypass the filter unconditionally. So both Profiles drive bursts — the
    // differentiation happens at probe time / dedup, not at routing.
    //
    // This test pins the routing semantics (both Profiles get the event because of the anchor-bypass)
    // and the registration semantics (METADATA-only kernel mask wouldn't even fire MetadataChanged
    // for the CONTENT Sub if the class filter applied — but it doesn't, due to the anchor-bypass).
    //
    // The descendant case (where the filter DOES apply) is covered in
    // `it_ef_6_descendant_metadata_drops_on_content_only_sub`.
    let mut e = Engine::new();
    let root = anchor_dir(&mut e, "build");
    let cfg = ScanConfig::builder().recursive(true).build();
    let t0_a = Instant::now();
    let (_sid_a, pid_a, _) = attach_returning(
        &mut e,
        "make",
        SubAttachAnchor::Resource(root),
        cfg.clone(),
        ClassSet::CONTENT,
        MAX_SETTLE,
        t0_a,
    );
    let t0_b = Instant::now();
    let (_sid_b, pid_b, _) = attach_returning(
        &mut e,
        "audit",
        SubAttachAnchor::Resource(root),
        cfg,
        ClassSet::METADATA,
        MAX_SETTLE,
        t0_b,
    );
    // Drive both Seeds → Idle.
    let snap = dir_snap(&[]);
    let _ = seed_to_idle(&mut e, pid_a, &snap, t0_a);
    let _ = seed_to_idle(&mut e, pid_b, &snap, t0_b);

    // MetadataChanged at the anchor — anchor events bypass class filter for both Profiles. Both
    // should drive bursts.
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
                e.profiles().get(pid).unwrap().state(),
                ProfileState::Active(_, _),
            ),
            "anchor MetadataChanged drives a burst on Profile {pid:?} regardless of mask",
        );
    }
}

// ───────────────────────────────────────────────────────────────────────
// IT-EF-3 — Descent prefix carries STRUCTURE-only mask
//
// Descent prefix watches register `events = {STRUCTURE}` regardless of the Sub's mask. The prefix
// is not the user's anchor; it's a transient artifact of the engine's descent state machine.
//
// We attach a Sub at a path whose anchor doesn't yet exist. The deepest existing prefix is bumped —
// its `events_union` should be STRUCTURE only, NOT the Sub's user mask.
// ───────────────────────────────────────────────────────────────────────

#[test]
fn it_ef_3_descent_prefix_contributes_structure_only() {
    let mut e = Engine::new();
    // Pre-existing /tmp; the Sub's anchor /tmp/build/leaf doesn't exist.
    let tmp = pre_place_dir(&mut e, &["tmp"]);

    let (_sid, pid, _attach_out) = attach_returning(
        &mut e,
        "watch",
        SubAttachAnchor::Path(PathBuf::from("/tmp/build/leaf")),
        ScanConfig::builder().recursive(true).build(),
        ClassSet::CONTENT, // user wants CONTENT only
        MAX_SETTLE,
        Instant::now(),
    );

    // Profile is Pending; current_prefix is /tmp. Extract the `Copy` ResourceId directly — the
    // descent's probe slot is linear and must not be cloned (nor borrowed-cloned to a no-op).
    let current_prefix = match e.profiles().get(pid).unwrap().state() {
        ProfileState::Pending(d) => d.current_prefix(),
        s => panic!("expected Pending, got {s:?}"),
    };
    assert_eq!(current_prefix, tmp);

    // /tmp's events_union is STRUCTURE — NOT the Sub's CONTENT mask. The Sub's mask only
    // contributes to its own anchor's union; the prefix is engine infrastructure.
    assert_eq!(
        e.tree().get(tmp).unwrap().events_union(),
        ClassSet::STRUCTURE,
        "descent prefix mask is STRUCTURE regardless of Sub's events",
    );
    // Profile's events_union still records the user's mask (drives mask on the eventual anchor
    // materialization).
    assert_eq!(e.profiles().get(pid).unwrap().events(), ClassSet::CONTENT,);
    let _ = e.cancel_all_in_flight_probes();
}

// ───────────────────────────────────────────────────────────────────────
// IT-EF-4 — Anchor terminal events bypass the class filter
//
// A `events = ["content"]` Sub on a Dir anchor: `Removed` at the anchor folds to STRUCTURE per
// `fs_event_to_class`'s identity-on-Dir rule. STRUCTURE is NOT in the Profile's mask. Without the
// anchor-bypass, the class filter would drop the event and the anchor's contribution would leak —
// the loss would never be observed and recovery would never start.
//
// With the anchor-bypass, the event routes to `on_anchor_terminal_event` regardless of mask:
// `anchor_claim` clears to None, baseline/current drop, and the Profile re-enters descent at
// watch_root_parent in the same step.
// ───────────────────────────────────────────────────────────────────────

#[test]
fn it_ef_4_anchor_terminal_bypasses_filter_for_narrow_mask() {
    let mut e = Engine::new();
    let parent = anchor_dir(&mut e, "p");
    let anchor = e
        .tree_mut()
        .ensure_child(parent, "watched-dir", ResourceRole::User)
        .expect("test live parent");
    e.tree_mut().set_kind(anchor, ResourceKind::Dir);

    // CONTENT-only mask on a Dir anchor — note: CONTENT registers no bits on a Dir, but the class
    // routing still uses Profile.events for filtering.
    let t0 = Instant::now();
    let (_sid, pid, _) = attach_returning(
        &mut e,
        "watch",
        SubAttachAnchor::Resource(anchor),
        ScanConfig::builder().recursive(true).build(),
        ClassSet::CONTENT,
        MAX_SETTLE,
        t0,
    );
    let _ = seed_to_idle(&mut e, pid, &dir_snap(&[]), t0);

    assert_eq!(
        e.profiles().get(pid).unwrap().anchor_claim(),
        AnchorClaim::Held,
        "post-Seed: anchor_claim = Held",
    );
    assert_eq!(e.tree().get(anchor).unwrap().watch_demand(), 1);

    // `Removed` on the anchor (a Dir) folds to STRUCTURE — not in mask. Without the anchor-bypass,
    // this event would drop with EventClassDropped. With the anchor-bypass, it routes through
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
    assert_eq!(
        p.anchor_claim(),
        AnchorClaim::None,
        "anchor_claim cleared by on_anchor_terminal_event",
    );
    assert!(p.baseline().is_none());
    assert!(p.current().is_none());
    assert_eq!(
        e.tree().get(anchor).unwrap().watch_demand(),
        0,
        "anchor's watch_demand released",
    );
    // watch_root_parent is intact, and the loss step re-entered descent against it.
    assert_eq!(p.watch_root_parent(), Some(parent));
    assert!(
        matches!(p.state(), ProfileState::Pending(_)),
        "observed loss re-enters descent in the loss step itself",
    );
    assert!(
        e.tree().get(parent).unwrap().watch_demand() >= 1,
        "watch_root_parent contribution survives anchor terminal",
    );
    let _ = e.cancel_all_in_flight_probes();
}

// ───────────────────────────────────────────────────────────────────────
// IT-EF-6 — `MetadataChanged` on a CONTENT-only Sub does not fire
//
// Descendant `MetadataChanged` events on a Sub whose mask excludes METADATA drop with
// `EventClassDropped` and do NOT extend the pre-fire or post-fire burst's `dirty` provenance (the
// class filter sits before dirty-provenance notes). The Profile remains in its prior state; no
// Effect emerges.
// ───────────────────────────────────────────────────────────────────────

#[test]
fn it_ef_6_descendant_metadata_drops_on_content_only_sub() {
    let mut e = Engine::new();
    let root = anchor_dir(&mut e, "src");

    let t0 = Instant::now();
    let (_sid, pid, _) = attach_returning(
        &mut e,
        "build",
        SubAttachAnchor::Resource(root),
        ScanConfig::builder().recursive(true).build(),
        ClassSet::CONTENT,
        MAX_SETTLE,
        t0,
    );
    let _ = seed_to_idle(&mut e, pid, &dir_snap(&[]), t0);

    // Materialize a covered descendant File. Bump watch_demand so the event passes the
    // EventOnUnwatchedResource head guard.
    let child = e
        .tree_mut()
        .ensure_child(root, "file.txt", ResourceRole::User)
        .expect("test live parent");
    e.tree_mut().set_kind(child, ResourceKind::File);
    e.tree_mut().get_mut(child).unwrap().insert_contribution(
        specter_core::ContribKey::ProfileDescendant(pid),
        ClassSet::CONTENT,
    );

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

    // No state mutation: Profile remains Idle; no probe queued, no Effect.
    assert!(
        matches!(e.profiles().get(pid).unwrap().state(), ProfileState::Idle),
        "drop happens before drive_burst — Profile stays Idle",
    );
    assert!(out.probe_ops().is_empty(), "no probe queued");
    assert!(out.effects().is_empty(), "no effects emitted");
}

#[test]
fn it_ef_6_descendant_content_changed_drives_burst_on_content_sub() {
    // Positive control: ContentChanged on a descendant of a CONTENT-only Sub DOES drive the burst
    // (CONTENT class matches mask).
    let mut e = Engine::new();
    let root = anchor_dir(&mut e, "src");

    let t0 = Instant::now();
    let (_sid, pid, _) = attach_returning(
        &mut e,
        "build",
        SubAttachAnchor::Resource(root),
        ScanConfig::builder().recursive(true).build(),
        ClassSet::CONTENT,
        MAX_SETTLE,
        t0,
    );
    let _ = seed_to_idle(&mut e, pid, &dir_snap(&[]), t0);

    let child = e
        .tree_mut()
        .ensure_child(root, "file.txt", ResourceRole::User)
        .expect("test live parent");
    e.tree_mut().set_kind(child, ResourceKind::File);
    e.tree_mut().get_mut(child).unwrap().insert_contribution(
        specter_core::ContribKey::ProfileDescendant(pid),
        ClassSet::CONTENT,
    );

    let _ = e.step(
        Input::FsEvent {
            resource: child,
            event: FsEvent::ContentChanged,
        },
        Instant::now(),
    );
    assert!(
        matches!(
            e.profiles().get(pid).unwrap().state(),
            ProfileState::Active(_, _),
        ),
        "ContentChanged on a CONTENT-class child drives a burst",
    );
}

// ───────────────────────────────────────────────────────────────────────
// IT-EF-5 — second Profile attaches at the same resource ⇒ engine emits fresh `WatchOp::Watch` with
// the widened union mask
//
// The union mask can change without the refcount changing — when a second Profile starts covering a
// Resource, its mask contribution may expand the union. The empty→non-empty edge alone is
// structurally insufficient. `add_watch` emits Watch on that existence edge OR on any union
// widening while the contributions map remains non-empty.
//
// Watcher-side mechanics (cache diff, EV_ADD overwrite) are covered in
// `crates/specter-sensor/tests/kqueue_rewatch.rs`. This engine-level test pins the upstream
// contract: the engine emits the right Watch op.
// ───────────────────────────────────────────────────────────────────────

#[test]
fn it_ef_5_second_profile_widens_mask_emits_fresh_watch() {
    let mut e = Engine::new();
    let root = anchor_dir(&mut e, "root");
    let cfg = ScanConfig::builder().recursive(true).build();

    // Profile A: events = CONTENT only.
    let (_sid_a, _pid_a, attach_a) = attach_returning(
        &mut e,
        "A",
        SubAttachAnchor::Resource(root),
        cfg.clone(),
        ClassSet::CONTENT,
        MAX_SETTLE,
        Instant::now(),
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
        e.tree().get(root).unwrap().events_union(),
        ClassSet::CONTENT,
        "per-Resource union after A = CONTENT",
    );

    // Profile B: events = METADATA on the same anchor (different mask ⇒ different config_hash ⇒
    // separate Profile).
    let (_sid_b, _pid_b, attach_b) = attach_returning(
        &mut e,
        "B",
        SubAttachAnchor::Resource(root),
        cfg,
        ClassSet::METADATA,
        MAX_SETTLE,
        Instant::now(),
    );
    // The engine emits a fresh `WatchOp::Watch` for the root, carrying the union (CONTENT |
    // METADATA), even though the anchor's `watch_demand` went 1→2 (not a 0→1 edge).
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
        e.tree().get(root).unwrap().events_union(),
        ClassSet::CONTENT | ClassSet::METADATA,
        "per-Resource union after B = CONTENT | METADATA",
    );
    assert_eq!(
        e.tree().get(root).unwrap().watch_demand(),
        2,
        "watch_demand bumped 1→2 on second attach (union changes drive Watch even off the 0↔1 edge)",
    );
    let _ = e.cancel_all_in_flight_probes();
}

// ───────────────────────────────────────────────────────────────────────
// IT-EF-2 dedup — the actuator's `DedupKey::Subtree` carries the Profile id, so two Profiles with
// different masks get distinct coalescing keys. (The fire-history does not need to: it is now
// per-Sub `Sub.has_fired`, with the owning Profile implicit, so two Profiles' Subs are inherently
// distinct slots.)
// ───────────────────────────────────────────────────────────────────────

#[test]
fn it_ef_2_dedup_keys_disambiguated_by_profile_id() {
    let mut e = Engine::new();
    let root = anchor_dir(&mut e, "build");
    let cfg = ScanConfig::builder().recursive(true).build();
    let (sid_a, pid_a) = attach(
        &mut e,
        "make",
        SubAttachAnchor::Resource(root),
        cfg.clone(),
        ClassSet::CONTENT,
        MAX_SETTLE,
        Instant::now(),
    );
    let (sid_b, pid_b) = attach(
        &mut e,
        "audit",
        SubAttachAnchor::Resource(root),
        cfg,
        ClassSet::METADATA,
        MAX_SETTLE,
        Instant::now(),
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
    let _ = e.cancel_all_in_flight_probes();
}

// ───────────────────────────────────────────────────────────────────────
// Seed-Vanished releases the anchor claim before descent re-entry
//
// `dispatch_seed_vanished` releases the anchor's contribution (mirroring `dispatch_standard_*`)
// inside `finalize_anchor_lost_and_descend`, *before* the same step re-enters `Pending` — a
// still-Held claim would violate `reap_profile`'s `!(Pending && Held)` trichotomy invariant at
// the descent flip. The watch is re-acquired via descent's anchor materialization on recovery.
// ───────────────────────────────────────────────────────────────────────

#[test]
fn seed_vanished_releases_anchor_claim_for_recovery() {
    let mut e = Engine::new();
    let parent = anchor_dir(&mut e, "p");
    let anchor = e
        .tree_mut()
        .ensure_child(parent, "a", ResourceRole::User)
        .expect("test live parent");
    e.tree_mut().set_kind(anchor, ResourceKind::Dir);

    let t0 = Instant::now();
    let (_sid, pid, _) = attach_returning(
        &mut e,
        "watch",
        SubAttachAnchor::Resource(anchor),
        ScanConfig::builder().recursive(true).build(),
        ClassSet::CONTENT,
        MAX_SETTLE,
        t0,
    );
    assert_eq!(
        e.profiles().get(pid).unwrap().anchor_claim(),
        AnchorClaim::Held,
    );
    assert_eq!(e.tree().get(anchor).unwrap().watch_demand(), 1);

    // Seed Vanished: anchor was found at attach but disappeared before the (settle-deferred) probe
    // could read. Drive the first Seed probe out, then answer Vanished — terminal on the first
    // response (the failure helper runs immediately).
    let (corr, at) = assert_seed_verifying(&mut e, pid, t0);
    let out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid,
            correlation: corr,
            outcome: ProbeOutcome::Vanished,
        }),
        at,
    );

    // Anchor's contribution is released, and the loss step re-entered descent at the parent —
    // the trichotomy invariant is exercised by the `Pending` flip in the same step.
    assert_eq!(
        e.profiles().get(pid).unwrap().anchor_claim(),
        AnchorClaim::None,
        "Seed Vanished releases anchor_claim",
    );
    assert_eq!(
        e.tree().get(anchor).unwrap().watch_demand(),
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
        e.profiles().get(pid).unwrap().state(),
        ProfileState::Pending(_),
    ));
    // watch_root_parent kept; it is the descent prefix now.
    assert!(e.profiles().get(pid).unwrap().watch_root_parent() == Some(parent));
    let _ = e.cancel_all_in_flight_probes();
}

#[test]
fn seed_vanished_then_recovery_does_not_violate_trichotomy() {
    // Step 1: attach_sub: anchor_claim = Held
    let mut e = Engine::new();
    let parent = anchor_dir(&mut e, "p");
    let anchor = e
        .tree_mut()
        .ensure_child(parent, "a", ResourceRole::User)
        .expect("test live parent");
    e.tree_mut().set_kind(anchor, ResourceKind::Dir);

    let t0 = Instant::now();
    let (sid, pid, _) = attach_returning(
        &mut e,
        "watch",
        SubAttachAnchor::Resource(anchor),
        ScanConfig::builder().recursive(true).build(),
        ClassSet::CONTENT,
        MAX_SETTLE,
        t0,
    );

    // Step 2: Seed Vanished (the probe fires one settle window after attach; Vanished is terminal
    // on the first response).
    let (corr, at) = assert_seed_verifying(&mut e, pid, t0);
    e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid,
            correlation: corr,
            outcome: ProbeOutcome::Vanished,
        }),
        at,
    );

    // Step 3: StructureChanged at watch_root_parent triggers recovery.
    e.step(
        Input::FsEvent {
            resource: parent,
            event: FsEvent::StructureChanged,
        },
        Instant::now(),
    );

    // Profile is Pending now. anchor_claim must be None (the trichotomy invariant).
    let p = e.profiles().get(pid).expect("Profile alive");
    assert!(
        matches!(p.state(), ProfileState::Pending(_)),
        "recovery transitions Profile → Pending",
    );
    assert_eq!(
        p.anchor_claim(),
        AnchorClaim::None,
        "post-fix: anchor_claim = None during Pending — trichotomy holds",
    );

    // Step 4: detach. reap_profile must NOT debug_assert and must NOT leak the anchor's watch_demand.
    let out = e.step(Input::DetachSub(sid), Instant::now());
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
fn seed_failed_releases_anchor_claim() {
    // Symmetric regression for dispatch_seed_failed.
    let mut e = Engine::new();
    let parent = anchor_dir(&mut e, "p");
    let anchor = e
        .tree_mut()
        .ensure_child(parent, "a", ResourceRole::User)
        .expect("test live parent");
    e.tree_mut().set_kind(anchor, ResourceKind::Dir);

    let t0 = Instant::now();
    let (_sid, pid, _) = attach_returning(
        &mut e,
        "watch",
        SubAttachAnchor::Resource(anchor),
        ScanConfig::builder().recursive(true).build(),
        ClassSet::CONTENT,
        MAX_SETTLE,
        t0,
    );

    // The Seed probe fires one settle window after attach. Failed is terminal on the first response.
    let (corr, at) = assert_seed_verifying(&mut e, pid, t0);
    let _out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid,
            correlation: corr,
            outcome: ProbeOutcome::Failed(ProbeFailure::Anchor { errno: 13 }),
        }),
        at,
    );

    assert_eq!(
        e.profiles().get(pid).unwrap().anchor_claim(),
        AnchorClaim::None,
        "Seed Failed releases anchor_claim (post-fix, symmetric with Seed Vanished)",
    );
    assert_eq!(
        e.tree().get(anchor).unwrap().watch_demand(),
        0,
        "anchor's watch_demand released",
    );
}

// ───────────────────────────────────────────────────────────────────────
// Regression: dispatch_standard_vanished/failed + reap_pending must not double-release the anchor
// contribution. The release-before-finish ordering keeps the debug_assert unreachable.
// ───────────────────────────────────────────────────────────────────────

/// Set up a Profile + a covered Dir child so the anchor cannot reap when the Profile detaches (the
/// child's existence keeps the slot alive). Returns (root, child, sid, pid). The Seed burst is
/// driven to a pinned `Idle` via the quiescence proof ([`seed_to_idle`]).
fn setup_with_surviving_child(
    e: &mut Engine,
) -> (ResourceId, ResourceId, specter_core::SubId, ProfileId) {
    let root = anchor_dir(e, "src");

    let t0 = Instant::now();
    let (sid, pid, _) = attach_returning(
        e,
        "build",
        SubAttachAnchor::Resource(root),
        ScanConfig::builder().recursive(true).build(),
        ClassSet::CONTENT,
        MAX_SETTLE,
        t0,
    );

    // Seed with a Dir child — its slot will outlive the Profile reap (the dir's `watch_demand`
    // survives until the Profile's release walks `apply_diff_to_tree`'s Phase-1 delete pass, which
    // graft does not reach on Vanished — the failure helper releases via `release_descendant_claim`
    // instead).
    let snap = dir_snap(&[("subdir", EntryKind::Dir, 99)]);
    let _ = seed_to_idle(e, pid, &snap, t0);

    let child = e
        .tree()
        .lookup(Some(root), "subdir")
        .expect("Dir child materialized by Seed graft");
    assert!(
        e.tree().get(child).unwrap().watch_demand() >= 1,
        "covered Dir child carries watch_demand",
    );

    (root, child, sid, pid)
}

#[test]
fn standard_vanished_with_reap_pending_does_not_double_release_anchor() {
    let mut e = Engine::new();
    let (root, _child, sid, pid) = setup_with_surviving_child(&mut e);

    // Drive a Standard burst.
    let t1 = Instant::now();
    e.step(
        Input::FsEvent {
            resource: root,
            event: FsEvent::ContentChanged,
        },
        t1,
    );

    // Detach mid-burst to set reap_pending.
    let _ = e.step(Input::DetachSub(sid), Instant::now());
    assert!(matches!(
        e.profiles().get(pid).unwrap().state().burst_finish(),
        Some(BurstFinish::Reap)
    ));

    // Drain the settle timer to advance to Probing.
    let t2 = t1 + SETTLE * 2;
    drain_due(&mut e, t2);
    let correlation = e.pending_probe_for(pid).expect("Verifying probe in flight");

    // Inject Vanished
    let out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid,
            correlation,
            outcome: ProbeOutcome::Vanished,
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
    let (root, _child, sid, pid) = setup_with_surviving_child(&mut e);

    let t1 = Instant::now();
    e.step(
        Input::FsEvent {
            resource: root,
            event: FsEvent::ContentChanged,
        },
        t1,
    );
    let _ = e.step(Input::DetachSub(sid), Instant::now());
    assert!(matches!(
        e.profiles().get(pid).unwrap().state().burst_finish(),
        Some(BurstFinish::Reap)
    ));

    let t2 = t1 + SETTLE * 2;
    drain_due(&mut e, t2);
    let correlation = e.pending_probe_for(pid).expect("Verifying probe in flight");

    let out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid,
            correlation,
            outcome: ProbeOutcome::Failed(ProbeFailure::Anchor { errno: 13 }),
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

/// Drive the anchor-terminal-with-reap-pending setup: attach P + surviving child, kick off a Standard
/// burst, advance to Probing, detach to set reap_pending, then dispatch the supplied FsEvent at the
/// anchor. Returns the resulting StepOutput. Surviving child fixture keeps the anchor slot alive past
/// `reap_profile`'s `try_reap`, exposing any post-finish refcount mistake on a still-live counter.
fn drive_anchor_terminal_with_reap_pending(event: FsEvent) -> (Engine, ResourceId, ProfileId) {
    let mut e = Engine::new();
    let (root, _child, sid, pid) = setup_with_surviving_child(&mut e);

    let t1 = Instant::now();
    e.step(
        Input::FsEvent {
            resource: root,
            event: FsEvent::ContentChanged,
        },
        t1,
    );
    let _ = e.step(Input::DetachSub(sid), Instant::now());
    assert!(matches!(
        e.profiles().get(pid).unwrap().state().burst_finish(),
        Some(BurstFinish::Reap)
    ));

    let t2 = t1 + SETTLE * 2;
    drain_due(&mut e, t2);
    assert!(matches!(
        e.profiles().get(pid).unwrap().state(),
        ProfileState::Active(
            specter_core::ActiveBurst::PreFire(specter_core::PreFireBurst {
                phase: specter_core::PreFirePhase::Verifying { .. },
                ..
            }),
            _,
        )
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
    // Two Profiles co-anchored at the same Resource (different config_hash). P has reap_pending +
    // Active(Probing); Q is Idle.
    let mut e = Engine::new();
    let root = anchor_dir(&mut e, "src");

    // Two Subs at the same anchor with different config_hash — different max_settle yields a fresh
    // Profile.
    let t0_p = Instant::now();
    let (sid_p, pid_p, _) = attach_returning(
        &mut e,
        "P",
        SubAttachAnchor::Resource(root),
        ScanConfig::builder().recursive(true).build(),
        ClassSet::CONTENT,
        MAX_SETTLE,
        t0_p,
    );
    let t0_q = Instant::now();
    let (_sid_q, _pid_q, _) = attach_returning(
        &mut e,
        "Q",
        SubAttachAnchor::Resource(root),
        ScanConfig::builder().recursive(true).build(),
        ClassSet::CONTENT,
        MAX_SETTLE + Duration::from_secs(1),
        t0_q,
    );
    let pid_q = e
        .profiles()
        .iter()
        .find(|(pid, _)| *pid != pid_p)
        .map(|(pid, _)| pid)
        .expect("Q profile minted");

    // Each Profile contributed +1 to root.watch_demand().
    assert_eq!(e.tree().get(root).unwrap().watch_demand(), 2);

    // Drive both to Idle via a Seed burst so the surviving-child invariant holds.
    let snap_p = dir_snap(&[("subdir", EntryKind::Dir, 99)]);
    let _ = seed_to_idle(&mut e, pid_p, &snap_p, t0_p);
    let snap_q = dir_snap(&[("subdir", EntryKind::Dir, 99)]);
    let _ = seed_to_idle(&mut e, pid_q, &snap_q, t0_q);

    // Kick off a Standard burst on P.
    let t1 = Instant::now();
    e.step(
        Input::FsEvent {
            resource: root,
            event: FsEvent::ContentChanged,
        },
        t1,
    );
    // Detach P to set reap_pending. Q stays alive.
    let _ = e.step(Input::DetachSub(sid_p), Instant::now());
    assert!(matches!(
        e.profiles().get(pid_p).unwrap().state().burst_finish(),
        Some(BurstFinish::Reap)
    ));
    assert!(!matches!(
        e.profiles().get(pid_q).unwrap().state().burst_finish(),
        Some(BurstFinish::Reap)
    ));

    // Advance P to Probing.
    let t2 = t1 + SETTLE * 2;
    drain_due(&mut e, t2);

    // FsEvent::Removed at root: covering_profiles returns [P, Q], each routes through
    // finalize_anchor_lost.
    let out = e.step(
        Input::FsEvent {
            resource: root,
            event: FsEvent::Removed,
        },
        t2,
    );

    // P reaped; Q remains Idle with anchor_claim cleared.
    assert!(e.profiles().get(pid_p).is_none(), "P reaped");
    let q = e.profiles().get(pid_q).expect("Q survives");
    assert_eq!(
        q.anchor_claim(),
        AnchorClaim::None,
        "Q's anchor_claim cleared by terminal event",
    );
    assert!(matches!(q.state(), ProfileState::Idle));

    // Counter walked 2 → 1 → 0 cleanly. Anchor slot is reaped because the surviving-child only kept
    // it alive while P+Q were attached; Q's anchor_claim = None leaves only the child anchor, which
    // does keep root alive — confirm via watch_demand counter.
    let final_counter = e
        .tree()
        .get(root)
        .map_or(0, specter_core::Resource::watch_demand);
    assert_eq!(
        final_counter, 0,
        "root.watch_demand() zeroed by both Profiles' terminal events",
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
// Descendant-claim release
//
// The four claim types differ in cardinality: anchor / watch-root parent / descent prefix are
// 1-to-1 (one Profile contributes to one Resource); covered descendants are 1-to-N (one Profile
// contributes to N Tree slots). The 1-to-N descendant set needs its own release helper: clearing
// `Profile.current` on a teardown path without releasing the per-descendant `watch_demand`
// contributions encoded in it would leave the descendant slots alive in the Tree with non-zero
// `watch_demand`, the kernel keeping their FDs registered — a default `subtree-root × CONTENT` Sub
// on a 10k-file tree would leak ~10k FDs per hot-reload churn cycle.
//
// `Engine::release_descendant_claim` closes the symmetry. Wired into `reap_profile` and the seven
// `dispatch_*_vanished/failed` + `finalize_anchor_lost` sites in `transitions.rs`. The helper takes
// `Profile.current` atomically, then applies a wholesale-deletion `Diff::all_deleted` over it via
// `apply_diff_to_tree`, releasing each covered slot's [`ContribKey::ProfileDescendant`]
// contribution by explicit key. Removal by explicit key is unambiguous regardless of
// `Profile.current`'s visibility to a concurrent graft.
// ───────────────────────────────────────────────────────────────────────

#[test]
fn release_descendant_claim_idle_detach_reaps_covered_dir() {
    // Idle Profile with one Sub + one covered Dir descendant. Detach the Sub: `detach_sub_inner`
    // runs `reap_profile` immediately (Idle ⇒ ReapNow). `release_descendant_claim` walks
    // `Profile.current` and releases `subdir`'s `watch_demand` contribution; the slot reaps, so the
    // descendant does not leak.
    let mut e = Engine::new();
    let (_root, child, sid, pid) = setup_with_surviving_child(&mut e);

    // Pre-conditions: Profile is Idle, child has watch_demand >= 1.
    assert!(matches!(
        e.profiles().get(pid).unwrap().state(),
        ProfileState::Idle,
    ));
    assert!(e.tree().get(child).is_some());
    assert!(e.tree().get(child).unwrap().watch_demand() >= 1);

    let out = e.step(Input::DetachSub(sid), Instant::now());
    assert!(
        e.profiles().get(pid).is_none(),
        "Idle Profile reaped on last-Sub detach",
    );
    assert!(
        e.tree().get(child).is_none(),
        "covered Dir descendant reaped — release_descendant_claim released its slot",
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
    // PerStableFile / has_per_file_fds=true Profile: covered Leaves also carry per-file FDs. Detach
    // must release the leaf's contribution as well as the Dir's.
    let mut e = Engine::new();
    let root = anchor_dir(&mut e, "src");

    let t0 = Instant::now();
    let (sid, pid, _) = attach_returning(
        &mut e,
        "build",
        SubAttachAnchor::Resource(root),
        ScanConfig::builder().recursive(true).build(),
        ClassSet::CONTENT,
        MAX_SETTLE,
        t0,
    );
    assert!(e.profiles().get(pid).unwrap().has_per_file_fds());

    // Seed with one File leaf — has_per_file_fds=true ⇒ leaf gets FD.
    let snap = dir_snap(&[("a.rs", EntryKind::File, 1)]);
    let _ = seed_to_idle(&mut e, pid, &snap, t0);
    let leaf = e.tree().lookup(Some(root), "a.rs").expect("leaf seeded");
    assert!(e.tree().get(leaf).unwrap().watch_demand() >= 1);

    let out = e.step(Input::DetachSub(sid), Instant::now());
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
    // dispatch_standard_vanished path: an anchor that disappears mid-burst must release the
    // per-descendant contributions alongside the anchor's — the descendants must not leak when the
    // anchor is released.
    let mut e = Engine::new();
    let (root, child, sid, pid) = setup_with_surviving_child(&mut e);

    // Drive a Standard burst.
    let t1 = Instant::now();
    e.step(
        Input::FsEvent {
            resource: root,
            event: FsEvent::ContentChanged,
        },
        t1,
    );
    let _ = e.step(Input::DetachSub(sid), Instant::now());
    assert!(matches!(
        e.profiles().get(pid).unwrap().state().burst_finish(),
        Some(BurstFinish::Reap)
    ));

    // Drain the settle timer to advance to Verifying.
    let t2 = t1 + SETTLE * 2;
    drain_due(&mut e, t2);
    let correlation = e.pending_probe_for(pid).expect("Verifying probe in flight");

    let out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid,
            correlation,
            outcome: ProbeOutcome::Vanished,
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
    // finalize_anchor_lost path: anchor-terminal events (Removed / Renamed / Revoked) at the anchor
    // must release the per-descendant contributions. Same shape as the dispatch_*_vanished path but
    // driven through `on_anchor_terminal_event`.
    let mut e = Engine::new();
    let (root, child, _sid, pid) = setup_with_surviving_child(&mut e);

    let out = e.step(
        Input::FsEvent {
            resource: root,
            event: FsEvent::Removed,
        },
        Instant::now(),
    );

    // After Removed at the anchor: Profile is Idle (post-finalize), anchor_claim cleared to None,
    // current taken by release_descendant_claim, child reaped.
    let p = e.profiles().get(pid).expect("Profile survives anchor loss");
    assert!(matches!(p.state(), ProfileState::Idle));
    assert_eq!(p.anchor_claim(), AnchorClaim::None);
    assert!(
        p.current().is_none(),
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
    // dispatch_rebase_vanished: post-fire rebase probe returns Vanished. `Profile.current` was
    // populated by the pre-fire `dispatch_quiescence_ok` `StandardFire` graft (`apply_snapshot`)
    // and contains the covered descendants. The rebase-failure path must release them too.
    //
    // Lifecycle: Idle → ContentChanged at root → Active(Verifying) → ProbeResponse::Ok (stable,
    // same snapshot) → emit_effects (one Effect for the SubtreeRoot Sub) → Active(Awaiting) →
    // EffectComplete::Ok → Active(Rebasing) directly (probe-first; the WholeSubtree rebase probe is
    // minted in that step) → ProbeResponse::Vanished → dispatch_rebase_vanished. The
    // release_descendant_claim wire-up walks Profile.current and reaps subdir.
    let mut e = Engine::new();
    let (root, child, sid, pid) = setup_with_surviving_child(&mut e);

    let t1 = Instant::now();
    e.step(
        Input::FsEvent {
            resource: root,
            event: FsEvent::ContentChanged,
        },
        t1,
    );

    let t2 = t1 + SETTLE * 2;
    drain_due(&mut e, t2);

    // The single Authoritative verify response folds to `Stable`. No covered descendant is in an
    // Active Standard burst, so emit_effects fires one Effect for the SubtreeRoot Sub →
    // transition_to_awaiting.
    let subdir_snap = dir_snap(&[("subdir", EntryKind::Dir, 99)]);
    let v = verify(&mut e, pid, &subdir_snap, t2);
    let effect = v
        .out
        .effects()
        .first()
        .cloned()
        .expect("Standard-Ok stable verdict fires one Effect");
    assert!(matches!(
        e.profiles().get(pid).unwrap().state(),
        ProfileState::Active(
            specter_core::ActiveBurst::PostFire(specter_core::PostFireBurst {
                phase: specter_core::PostFirePhase::Awaiting { .. },
                ..
            }),
            _,
        ),
    ));

    // EffectComplete::Ok drives Awaiting → Rebasing directly (probe-first): the WholeSubtree rebase
    // probe is minted in this very step, with no first Settling window. Read its correlation
    // straight off the in-flight probe for the Vanished response below.
    let _ = complete_effect_to_rebasing(&mut e, sid, effect.key(), t2);
    let rebase_corr = e
        .pending_probe_for(pid)
        .expect("EffectComplete drove Awaiting → Rebasing with the rebase probe in flight");

    // Pre-condition: descendant claim still intact going into Rebasing.
    assert!(
        e.tree().get(child).is_some_and(|r| r.watch_demand() >= 1),
        "subdir.watch_demand() still held going into Rebasing",
    );

    // Inject Rebase Vanished.
    let out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid,
            correlation: rebase_corr,
            outcome: ProbeOutcome::Vanished,
        }),
        t2 + SETTLE,
    );

    assert!(matches!(
        e.profiles().get(pid).unwrap().state(),
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
    // Two Profiles co-anchor at root with the same recursive scan, both observing the same
    // descendant `subdir`. Profile P loses its anchor (Vanished); Profile Q stays Idle. P's release
    // walks current and decrements subdir 2 → 1 — no leak — while Q's contribution survives at 1.
    let mut e = Engine::new();
    let root = anchor_dir(&mut e, "src");

    // Two Profiles at the same anchor with different config_hash (different max_settle ⇒ different
    // Profile).
    let t0_p = Instant::now();
    let (sid_p, pid_p, _) = attach_returning(
        &mut e,
        "P",
        SubAttachAnchor::Resource(root),
        ScanConfig::builder().recursive(true).build(),
        ClassSet::CONTENT,
        MAX_SETTLE,
        t0_p,
    );
    let t0_q = Instant::now();
    let (_sid_q, _pid_q, _) = attach_returning(
        &mut e,
        "Q",
        SubAttachAnchor::Resource(root),
        ScanConfig::builder().recursive(true).build(),
        ClassSet::CONTENT,
        MAX_SETTLE + Duration::from_secs(1),
        t0_q,
    );
    let pid_q = e
        .profiles()
        .iter()
        .find(|(pid, _)| *pid != pid_p)
        .map(|(pid, _)| pid)
        .expect("Q profile minted");

    // Drive both to Idle with a shared seeded Dir descendant. Each Profile's create_child
    // contributed +1 to subdir.watch_demand().
    let snap_p = dir_snap(&[("subdir", EntryKind::Dir, 99)]);
    let _ = seed_to_idle(&mut e, pid_p, &snap_p, t0_p);
    let snap_q = dir_snap(&[("subdir", EntryKind::Dir, 99)]);
    let _ = seed_to_idle(&mut e, pid_q, &snap_q, t0_q);

    let subdir = e
        .tree()
        .lookup(Some(root), "subdir")
        .expect("subdir seeded");
    assert_eq!(
        e.tree().get(subdir).unwrap().watch_demand(),
        2,
        "two Profiles each contributed +1 to subdir.watch_demand()",
    );

    // Drive P's Standard burst to Verifying, detach P (reap_pending), inject Vanished — P's anchor
    // is gone, but Q's stays. P's release_descendant_claim walks P's current and decrements
    // subdir's contribution; Q's contribution is preserved.
    let t1 = Instant::now();
    e.step(
        Input::FsEvent {
            resource: root,
            event: FsEvent::ContentChanged,
        },
        t1,
    );
    let _ = e.step(Input::DetachSub(sid_p), Instant::now());

    let t2 = t1 + SETTLE * 2;
    drain_due(&mut e, t2);
    let correlation = e.pending_probe_for(pid_p).expect("P probe in flight");

    let _out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid_p,
            correlation,
            outcome: ProbeOutcome::Vanished,
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
        e.tree().get(subdir).unwrap().watch_demand(),
        1,
        "subdir.watch_demand() decremented from 2 to 1 by P's release",
    );
    assert_eq!(
        e.profiles().get(pid_q).unwrap().events(),
        ClassSet::CONTENT,
        "Q's contribution intact",
    );
    let _ = e.cancel_all_in_flight_probes();
}

#[test]
fn delete_child_during_graft_recompute_skips_releasing_profile() {
    // During graft's `apply_diff_to_tree` delete pass the releasing Profile's `Profile.current` is
    // still `Some` (graft hasn't run the take yet), so the post-decrement union must not pick up
    // the Profile's own descendant contribution. Removal by explicit
    // [`ContribKey::ProfileDescendant(profile_id)`] key keeps it independent of `Profile.current`'s
    // visibility during the apply. This test pins the post-decrement union to the remaining
    // contributors' mask.
    //
    // Setup: two Profiles share the anchor with DIFFERENT events masks. P=CONTENT, Q=METADATA. Both
    // seed with `subdir` as a covered Dir. subdir.watch_demand = 2; subdir.events_union = CONTENT |
    // METADATA. P's probe response says `subdir` is gone — `dispatch_quiescence_ok` → `graft` →
    // `apply_diff_to_tree` Phase-1 delete for `subdir`. The recompute must yield Q's mask only, not
    // the union.
    let mut e = Engine::new();
    let root = anchor_dir(&mut e, "src");

    let t0_p = Instant::now();
    let (_sid_p, pid_p, _) = attach_returning(
        &mut e,
        "P",
        SubAttachAnchor::Resource(root),
        ScanConfig::builder().recursive(true).build(),
        ClassSet::CONTENT,
        MAX_SETTLE,
        t0_p,
    );
    let t0_q = Instant::now();
    let (_sid_q, _pid_q, _) = attach_returning(
        &mut e,
        "Q",
        SubAttachAnchor::Resource(root),
        ScanConfig::builder().recursive(true).build(),
        ClassSet::METADATA,
        MAX_SETTLE + Duration::from_secs(1),
        t0_q,
    );
    let pid_q = e
        .profiles()
        .iter()
        .find(|(pid, _)| *pid != pid_p)
        .map(|(pid, _)| pid)
        .expect("Q profile minted");

    let snap_p = dir_snap(&[("subdir", EntryKind::Dir, 99)]);
    let _ = seed_to_idle(&mut e, pid_p, &snap_p, t0_p);
    let snap_q = dir_snap(&[("subdir", EntryKind::Dir, 99)]);
    let _ = seed_to_idle(&mut e, pid_q, &snap_q, t0_q);

    let subdir = e
        .tree()
        .lookup(Some(root), "subdir")
        .expect("subdir seeded");
    assert_eq!(e.tree().get(subdir).unwrap().watch_demand(), 2);
    assert_eq!(
        e.tree().get(subdir).unwrap().events_union(),
        ClassSet::CONTENT | ClassSet::METADATA,
        "two contributors, union of masks",
    );

    // Drive P's Standard burst to Verifying.
    let t1 = Instant::now();
    e.step(
        Input::FsEvent {
            resource: root,
            event: FsEvent::ContentChanged,
        },
        t1,
    );
    let t2 = t1 + SETTLE * 2;
    drain_due(&mut e, t2);
    let correlation = e.pending_probe_for(pid_p).expect("P probe in flight");

    // Probe response: subdir is GONE (P sees a tree where it's been deleted).
    // dispatch_quiescence_ok → graft → apply_diff_to_tree Phase-1 fires `sub_watch_then_try_reap`
    // for subdir keyed by `ContribKey::ProfileDescendant(pid_p)`.
    let response = Arc::new(DirSnapshot::new(
        DirMeta::synthetic(UNIX_EPOCH, FsIdentity::synthetic(0, 0)),
        0,
        BTreeMap::new(),
    ));
    let _out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid_p,
            correlation,
            outcome: proven(response),
        }),
        t2,
    );

    // After the graft delete pass:
    // - subdir.watch_demand() goes 2 → 1 (P's ProfileDescendant entry removed).
    // - Remaining contributor Q's mask defines events_union = METADATA.
    // - Without the contribution-map move (lazy-derivation era): the recompute could have
    //   over-masked at CONTENT|METADATA.
    assert!(
        e.tree().get(subdir).is_some(),
        "subdir survives, Q anchors it"
    );
    assert_eq!(
        e.tree().get(subdir).unwrap().watch_demand(),
        1,
        "subdir.watch_demand() decremented to 1 (P released, Q remains)",
    );
    assert_eq!(
        e.tree().get(subdir).unwrap().events_union(),
        ClassSet::METADATA,
        "events_union narrows to Q's mask only — the per-Resource \
         contributions map removes P's contribution by explicit key",
    );
    assert_ne!(
        e.tree().get(subdir).unwrap().events_union(),
        ClassSet::CONTENT | ClassSet::METADATA,
        "over-mask check: must NOT include P's contribution",
    );
    // Q is alive — its descendant claim on subdir is intact.
    assert!(e.profiles().get(pid_q).is_some(), "Q survives P's graft");
    let _ = e.cancel_all_in_flight_probes();
}
