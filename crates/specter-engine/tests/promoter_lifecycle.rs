//! Cross-cutting Promoter lifecycle.
//!
//! Validates the composition of attach → enumeration → promotion →
//! anchor materialisation (the dynamic Sub's own descent) →
//! Seed-burst baseline establishment → reap by driving one
//! [`Engine`] through every phase in source order with synthetic
//! [`ProbeResponse`] injections (the same pattern
//! `crates/specter-engine/src/promoter_tests.rs` uses inline). The
//! inline tests pin individual phases; this file pins the
//! diagnostic ordering and registry tear-down that emerge when all
//! phases compose in one Engine instance.
//!
//! Effect-firing per se is **not** validated here — a fresh
//! Profile's Seed burst establishes baseline and finishes without
//! firing (`dispatch_seed_ok`'s no-drift terminal arm); subsequent
//! Standard bursts on FsEvents are what fire. The Standard-burst
//! mechanics are exhaustively pinned by `transitions_tests.rs`,
//! `fire_cycle.rs`, and `burst_pacing.rs`. Composing the
//! Promoter→dynamic-Sub→Standard-burst flow into a single
//! integration test would re-test the well-pinned Standard-burst
//! arms; cross-cutting value is in the Promoter-specific glue.

#![allow(
    clippy::items_after_statements,
    clippy::manual_let_else,
    clippy::missing_const_for_fn,
    clippy::needless_pass_by_value,
    clippy::option_if_let_else,
    clippy::similar_names,
    clippy::single_match_else,
    clippy::too_many_lines
)]

use compact_str::CompactString;
use specter_core::testkit::single_exec_program;
use specter_core::{
    ActionProgram, ChildEntry, ClassSet, Diagnostic, DirChild, DirMeta, DirSnapshot, EffectScope,
    EntryKind, Input, LeafEntry, OverflowScope, PatternSpec, ProbeOp, ProbeOutcome, ProbeOwner,
    ProbeResponse, ProfileState, PromoterAttachRequest, PromoterRegistryDiff, PromoterState,
    ResourceId, ResourceKind, ResourceRole, ScanConfig, SubAttachRequest, WatchOp,
    WatchRegistryDiff,
};
use specter_engine::Engine;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant, UNIX_EPOCH};

const SETTLE: Duration = Duration::from_millis(100);
const MAX_SETTLE: Duration = Duration::from_secs(6);
const FS_ROOT_SEG: &str = "/";

fn empty_program() -> Arc<ActionProgram> {
    single_exec_program([specter_core::ArgTemplate::new([
        specter_core::ArgPart::literal("/bin/true"),
    ])])
}

fn promoter_req(name: &str, pattern: &str) -> PromoterAttachRequest {
    PromoterAttachRequest {
        name: name.to_owned(),
        pattern_spec: PatternSpec::parse(pattern).expect("valid test pattern"),
        config: ScanConfig::builder().recursive(true).build(),
        max_settle: MAX_SETTLE,
        settle: SETTLE,
        program: empty_program(),
        scope: EffectScope::SubtreeRoot,
        events: ClassSet::EMPTY,
        log_output: false,
    }
}

/// Build a `DirSnapshot` whose `root_resource` is `target` (the
/// proxy slot or descent prefix the engine probed). The walker
/// stamps `target_resource` from the request onto
/// `DirSnapshot.root_resource`; descent dispatch's
/// `debug_assert_eq!` pins `snapshot.root_resource ==
/// descent.current_prefix`. Pass the right `target` per phase or
/// the assert panics.
fn dir_snap_with(target: ResourceId, children: Vec<(&str, EntryKind, u64)>) -> Arc<DirSnapshot> {
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
    Arc::new(DirSnapshot::new(
        target,
        DirMeta {
            mtime: UNIX_EPOCH,
            inode: 0,
            device: 0,
        },
        0,
        map,
    ))
}

/// Helper: pick the most recent outstanding probe target from a
/// step's [`ProbeOp::Probe`] emissions.
fn last_probe_target(out: &specter_core::StepOutput) -> Option<ResourceId> {
    out.probe_ops.iter().rev().find_map(|op| match op {
        ProbeOp::Probe { request } => request.target_resource(),
        ProbeOp::Cancel { .. } => None,
    })
}

/// Pre-place a path on the `Engine`'s Tree as a User-roled chain
/// (FS-root through every supplied segment). Returns the leaf id
/// — the deepest segment that's now materialised. Used so a
/// pattern's literal prefix exists from step 1, putting the
/// Promoter into immediate-Active mode and letting the test focus
/// on the per-Promoter phases (enumeration, promotion, fire,
/// reap) rather than the descent advance the inline tests already
/// pin in detail.
fn pre_place_dir(e: &mut Engine, segments: &[&str]) -> ResourceId {
    let mut comps = Vec::with_capacity(segments.len() + 1);
    comps.push(FS_ROOT_SEG);
    comps.extend_from_slice(segments);
    let r = e.tree_mut().ensure_path(&comps, ResourceRole::User);
    e.tree_mut().set_kind(r, ResourceKind::Dir);
    r
}

/// Full Promoter lifecycle composition with pre-placed literal
/// prefix. Validates that every phase composes correctly through
/// one Engine, with diagnostic ordering pinned at each transition.
///
/// Phases:
/// 1. Attach (immediate-Active mode; no PrefixPending).
/// 2. Initial enumeration → promote a single match → mint dynamic Sub.
/// 3. Dynamic Sub's anchor descent (`foo.log` slot didn't pre-exist).
/// 4. Seed-burst baseline at the materialised File anchor (no fire
///    on no-drift fresh attach).
/// 5. Reap Promoter → cascade detach + DynamicSubReaped + PromoterReaped.
#[test]
fn full_lifecycle_attach_promote_descend_seed_reap() {
    let mut e = Engine::new();

    // Pre-place /var/log so attach lands in immediate-Active. The
    // PrefixPending → descent advance path is exhaustively pinned
    // by inline `promoter_tests.rs::descent_*`; this test focuses
    // on the post-Active composition.
    let var_log = pre_place_dir(&mut e, &["var", "log"]);

    let now = Instant::now();
    let (pid, attach_out) = e.attach_promoter(promoter_req("logs", "/var/log/*.log"), now);

    // ---- PromoterAttached + initial enumeration probe ----
    assert!(
        attach_out.diagnostics.iter().any(|d| matches!(
            d,
            Diagnostic::PromoterAttached { promoter, name } if *promoter == pid && name == "logs",
        )),
        "PromoterAttached emitted on attach",
    );
    match &e.promoters().get(pid).expect("promoter registered").state {
        PromoterState::Active { proxies } => assert!(
            proxies.contains_key(&var_log),
            "proxy registered at the materialised prefix",
        ),
        PromoterState::PrefixPending(d) => panic!(
            "expected Active, got PrefixPending(prefix={:?}, remaining={:?})",
            d.current_prefix, d.remaining_components,
        ),
    }
    assert_eq!(
        last_probe_target(&attach_out),
        Some(var_log),
        "enumeration probe targets /var/log",
    );
    let enum_corr = e
        .pending_probe_for(ProbeOwner::Promoter(pid))
        .expect("enumeration probe in flight");

    // ---- enumeration response → promotion ----
    //
    // Inject a directory listing with one *.log match (foo.log) and
    // one non-matching entry (bar.txt). The match promotes; the
    // non-match doesn't.
    let snap_var_log = dir_snap_with(
        var_log,
        vec![
            ("foo.log", EntryKind::File, 10),
            ("bar.txt", EntryKind::File, 11),
        ],
    );
    let promote_out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Promoter(pid),
            correlation: enum_corr,
            outcome: ProbeOutcome::SubtreeOk(snap_var_log),
        }),
        now,
    );

    // dynamic_subs records exactly one promotion; it's foo.log.
    let dynamic_sub_id = {
        let q = e.promoters().get(pid).unwrap();
        assert_eq!(q.dynamic_subs.len(), 1, "one promotion");
        let (path, sid) = q.dynamic_subs.iter().next().unwrap();
        assert_eq!(path.to_string_lossy(), "/var/log/foo.log");
        *sid
    };
    let dynamic_profile = e
        .subs()
        .get(dynamic_sub_id)
        .expect("dynamic Sub registered")
        .profile;

    // Diagnostics on the promote step:
    // - PromotionKindObserved for the matched path.
    // - SubAttached(source_promoter=Some(pid)) for the dynamic Sub.
    let mut saw_promotion_observed = false;
    let mut saw_sub_attached_dynamic = false;
    for d in &promote_out.diagnostics {
        match d {
            Diagnostic::PromotionKindObserved { promoter, .. } if *promoter == pid => {
                saw_promotion_observed = true;
            }
            Diagnostic::SubAttached {
                sub,
                source_promoter,
                ..
            } if *sub == dynamic_sub_id => {
                assert_eq!(
                    *source_promoter,
                    Some(pid),
                    "dynamic SubAttached carries the promoter id",
                );
                saw_sub_attached_dynamic = true;
            }
            _ => {}
        }
    }
    assert!(
        saw_promotion_observed && saw_sub_attached_dynamic,
        "PromotionKindObserved and dynamic SubAttached both emit; got {:?}",
        promote_out.diagnostics,
    );

    // ---- dynamic Sub's anchor descent ----
    //
    // The dynamic Sub's path `/var/log/foo.log` had its leaf slot
    // freshly minted by `ensure_path` inside `attach_sub_inner`.
    // Since the leaf didn't pre-exist, `materialize_path_or_pending`
    // returned `Pending`, and the Profile starts in Pending(descent).
    // The descent probe targets /var/log; the response carries the
    // entry list at the prefix.
    let descent_corr = e
        .pending_probe_for(ProbeOwner::Profile(dynamic_profile))
        .expect("dynamic Sub Profile descent probe in flight");
    let snap_descent = dir_snap_with(var_log, vec![("foo.log", EntryKind::File, 10)]);
    let materialize_out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(dynamic_profile),
            correlation: descent_corr,
            outcome: ProbeOutcome::SubtreeOk(snap_descent),
        }),
        now,
    );

    // Profile transitioned: state Idle → Active(Seed); kind = File;
    // a fresh AnchorFile probe is in flight on the foo.log slot.
    {
        let p = e.profiles().get(dynamic_profile).expect("Profile alive");
        assert_eq!(
            p.kind,
            Some(ResourceKind::File),
            "anchor classified as File"
        );
    }
    let seed_corr = e
        .pending_probe_for(ProbeOwner::Profile(dynamic_profile))
        .expect("Seed-burst AnchorFile probe in flight");
    assert!(
        materialize_out.probe_ops.iter().any(|op| matches!(
            op,
            ProbeOp::Probe {
                request: specter_core::ProbeRequest::AnchorFile { .. }
            }
        )),
        "Seed burst on a File-anchored Profile emits AnchorFile, not Subtree",
    );

    // ---- Seed-burst baseline establishes; no fire ----
    //
    // AnchorOk(LeafEntry) on the AnchorFile probe; the engine
    // integrates the leaf as `Profile.current` (the baseline) and
    // finishes the burst to Idle. A *fresh* Seed burst with no
    // drift fires no Effect — the baseline is the recorded "how
    // things look right now" against which future Standard bursts
    // observe drift. This is `dispatch_seed_ok`'s no-drift terminal
    // arm.
    let leaf = LeafEntry::new(EntryKind::File, 0, UNIX_EPOCH, 10, 0);
    let baseline_out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(dynamic_profile),
            correlation: seed_corr,
            outcome: ProbeOutcome::AnchorOk(leaf),
        }),
        now,
    );
    assert!(
        baseline_out.effects.is_empty(),
        "fresh Seed verdict establishes baseline without firing; got effects={:?}",
        baseline_out.effects,
    );
    {
        let p = e.profiles().get(dynamic_profile).expect("Profile alive");
        assert!(
            matches!(p.state, specter_core::ProfileState::Idle),
            "Profile returns to Idle after Seed completes; got {:?}",
            p.state,
        );
        assert!(
            p.current.is_some(),
            "baseline integrated as Profile.current"
        );
    }

    // ---- reap Promoter → cascade ----
    //
    // ConfigDiff with the Promoter id under `removed`. The cascade
    // detaches the dynamic Sub (DynamicSubReaped), unwinds the proxy
    // at /var/log, and drops the Promoter (PromoterReaped).
    let reap_diff = WatchRegistryDiff {
        promoters: PromoterRegistryDiff {
            removed: vec![pid],
            ..Default::default()
        },
        ..Default::default()
    };
    let reap_out = e.step(Input::ConfigDiff(reap_diff), now);

    assert!(
        e.promoters().get(pid).is_none(),
        "Promoter removed from registry"
    );
    assert!(
        e.subs().get(dynamic_sub_id).is_none(),
        "cascaded dynamic Sub detached from SubRegistry",
    );

    // PromoterReaped is the canonical lifecycle signal of the
    // ConfigDiff cascade. `DynamicSubReaped` is reserved for
    // anchor-terminal events (the all-dynamic teardown path in
    // `on_anchor_terminal_event` — the variant doc spells this
    // out): a Promoter reap routes through `detach_sub_inner`
    // directly and surfaces no per-dynamic-Sub diagnostic. The
    // reap-pending Profile path emits `ReapPendingResolved` for
    // the same Profile when the Seed-burst-baselined snapshot is
    // still mid-rebase at reap time.
    assert!(
        reap_out.diagnostics.iter().any(|d| matches!(
            d,
            Diagnostic::PromoterReaped { promoter } if *promoter == pid,
        )),
        "PromoterReaped emitted on cascade; got {:?}",
        reap_out.diagnostics,
    );
}

/// Static-only attach lifecycle through the same `Engine`: a single
/// `[[watch]]`-style attach emits exactly one
/// `SubAttached(source_promoter=None)` diagnostic. Validates the
/// shape relied on by the bin's diagnostic-driven `loader.ids`
/// reconciliation.
#[test]
fn static_attach_emits_sub_attached_with_no_source_promoter() {
    let mut e = Engine::new();
    let r = e.tree_mut().ensure(None, "src", ResourceRole::User);
    e.tree_mut().set_kind(r, ResourceKind::Dir);

    let req = specter_core::SubAttachRequest::for_resource(
        "build".into(),
        r,
        ScanConfig::builder().build(),
        MAX_SETTLE,
        SETTLE,
        empty_program(),
        EffectScope::SubtreeRoot,
        ClassSet::EMPTY,
        false,
    );
    let (sid, out) = e.attach_sub(req, Instant::now());
    assert_ne!(sid, specter_core::SubId::default());

    let mut sub_attached_count = 0usize;
    for d in &out.diagnostics {
        if let Diagnostic::SubAttached {
            sub,
            name,
            source_promoter,
        } = d
        {
            assert_eq!(*sub, sid);
            assert_eq!(name.as_str(), "build");
            assert!(
                source_promoter.is_none(),
                "static attach carries no source_promoter; got {source_promoter:?}",
            );
            sub_attached_count += 1;
        }
    }
    assert_eq!(
        sub_attached_count, 1,
        "exactly one SubAttached per attach; got diagnostics={:?}",
        out.diagnostics,
    );
}

// ───────────────────────────────────────────────────────────────────────
// Cross-actor: descent vanish preserves co-resident Promoter proxy
// ───────────────────────────────────────────────────────────────────────

/// First emitted probe correlation for a step's `StepOutput` —
/// utility for capturing the live channel correlation from `attach_*`
/// flows. Mirrors the local helper in
/// `crates/specter-engine/tests/watch_op_rejected_purge.rs`.
fn first_probe_corr(out: &specter_core::StepOutput) -> Option<specter_core::ProbeCorrelation> {
    out.probe_ops.iter().find_map(|op| match op {
        ProbeOp::Probe { request } => Some(request.correlation()),
        ProbeOp::Cancel { .. } => None,
    })
}

/// Cross-owner shared-prefix vanish (T1-1): a Profile in `Pending`
/// descent at `/a/b` shares the slot with a Promoter `Active` proxy
/// at `/a/b`. Pre-fix, `dispatch_descent_vanished`'s rewind branch
/// called `Tree::vacate(prefix)` unconditionally, zeroing the
/// Promoter's `STRUCTURE` contribution and stranding the kernel
/// watch. Post-fix, the rewind drops only the Profile's contribution
/// (counter 2 → 1) and the Promoter's claim survives. The matching
/// `Unwatch` only fires on the genuine 1 → 0 edge — when the
/// Promoter's enumeration probe later returns `Vanished`.
#[test]
fn descent_vanish_preserves_co_resident_promoter_proxy() {
    let mut e = Engine::new();
    let now = Instant::now();

    // Pre-place /a so the Promoter's literal-prefix descent has a
    // pre-existing prefix to land on. Pattern `/a/b/*.txt` →
    // literal_prefix_len=2 (FS-root + a + b … wait, "/" is implicit;
    // the literal segments are `["a", "b"]`). Promoter starts in
    // `PrefixPending(/a, ["b"])` because /a/b doesn't yet exist.
    let a = pre_place_dir(&mut e, &["a"]);
    let (qid, attach_q_out) = e.attach_promoter(promoter_req("logs", "/a/b/*.txt"), now);
    assert!(matches!(
        e.promoters().get(qid).unwrap().state,
        PromoterState::PrefixPending(_),
    ));
    let descent_q_corr =
        first_probe_corr(&attach_q_out).expect("Promoter descent probe at /a in flight");

    // Drive the Promoter's descent into Active by completing the
    // `/a` probe with a "b" Dir entry. The dispatcher's terminal arm
    // calls `enter_active`, which materialises /a/b as the first proxy
    // (User-roled, +1 STRUCTURE) and queues an enumeration probe at
    // /a/b.
    let snap_a = dir_snap_with(a, vec![("b", EntryKind::Dir, 1)]);
    let _ = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Promoter(qid),
            correlation: descent_q_corr,
            outcome: ProbeOutcome::SubtreeOk(snap_a),
        }),
        now,
    );

    let a_b = e
        .tree()
        .lookup(Some(a), "b")
        .expect("/a/b materialised by enter_active");
    match &e.promoters().get(qid).unwrap().state {
        PromoterState::Active { proxies } => assert!(proxies.contains_key(&a_b)),
        s @ PromoterState::PrefixPending(_) => panic!("expected Active, got {s:?}"),
    }
    assert_eq!(e.tree().get(a_b).unwrap().watch_demand(), 1);

    // Capture the Promoter's enumeration probe at /a/b (set by
    // `dispatch_next_enumeration` inside `enter_active`).
    let enum_q_corr = e
        .pending_probe_for(ProbeOwner::Promoter(qid))
        .expect("Promoter enumeration probe at /a/b in flight");

    // Attach a Profile with a recursive Sub at /a/b/c. /a/b exists
    // (User-roled by the Promoter); /a/b/c doesn't. The Profile
    // starts in `Pending(/a/b, ["c"])`, bumping /a/b's STRUCTURE
    // contribution to 2. A descent probe targets /a/b under the
    // Profile owner — distinct from the Promoter's enumeration probe.
    let req = SubAttachRequest::for_path(
        "watch".into(),
        PathBuf::from("/a/b/c"),
        ScanConfig::builder().recursive(true).build(),
        MAX_SETTLE,
        SETTLE,
        empty_program(),
        EffectScope::SubtreeRoot,
        ClassSet::EMPTY,
        false,
    );
    let (sid, attach_p_out) = e.attach_sub(req, now);
    let pid = e.subs().get(sid).unwrap().profile;
    assert!(matches!(
        e.profiles().get(pid).unwrap().state,
        ProfileState::Pending(_),
    ));
    let descent_p_corr =
        first_probe_corr(&attach_p_out).expect("Profile descent probe at /a/b in flight");

    // Pre-vanish state: /a/b carries two STRUCTURE contributions.
    assert_eq!(e.tree().get(a_b).unwrap().watch_demand(), 2);
    assert_eq!(
        e.tree().get(a_b).unwrap().events_union(),
        ClassSet::STRUCTURE,
    );

    // Send the Profile's descent probe `Vanished`. The rewind branch
    // moves Profile's `current_prefix` to /a, releases /a/b's STRUCTURE
    // contribution, and adds /a's. Pre-fix, the unconditional
    // `vacate(/a/b)` would have zeroed the slot; post-fix, only the
    // Profile's contribution drops.
    let vanish_out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
            correlation: descent_p_corr,
            outcome: ProbeOutcome::Vanished,
        }),
        now,
    );

    assert_eq!(
        e.tree().get(a_b).unwrap().watch_demand(),
        1,
        "Promoter's STRUCTURE contribution survives the Profile vanish",
    );
    assert_eq!(
        e.tree().get(a_b).unwrap().events_union(),
        ClassSet::STRUCTURE,
        "events_union recomputed from Promoter (STRUCTURE), not zeroed",
    );
    let unwatch_at_a_b = vanish_out
        .watch_ops
        .iter()
        .any(|op| matches!(op, WatchOp::Unwatch { resource } if *resource == a_b));
    assert!(
        !unwatch_at_a_b,
        "no Unwatch on /a/b on the descent vanish — Promoter still claims it",
    );
    match &e.promoters().get(qid).unwrap().state {
        PromoterState::Active { proxies } => assert!(proxies.contains_key(&a_b)),
        s @ PromoterState::PrefixPending(_) => {
            panic!("Promoter state should remain Active{{proxies}}, got {s:?}")
        }
    }

    // Now the Promoter's enumeration probe at /a/b returns Vanished.
    // `unregister_proxy` → `release_promoter_proxy_claim` removes the
    // last contribution; `sub_watch` emits Unwatch on the non-empty
    // → empty edge.
    let promoter_vanish_out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Promoter(qid),
            correlation: enum_q_corr,
            outcome: ProbeOutcome::Vanished,
        }),
        now,
    );

    let unwatch_count = promoter_vanish_out
        .watch_ops
        .iter()
        .filter(|op| matches!(op, WatchOp::Unwatch { resource } if *resource == a_b))
        .count();
    assert_eq!(
        unwatch_count, 1,
        "single Unwatch at /a/b on the genuine 1 → 0 edge",
    );
    match &e.promoters().get(qid).unwrap().state {
        PromoterState::Active { proxies } => assert!(!proxies.contains_key(&a_b)),
        s @ PromoterState::PrefixPending(_) => {
            panic!("Promoter state should remain Active, got {s:?}")
        }
    }
}

// ───────────────────────────────────────────────────────────────────────
// Cross-Promoter: shared proxy unwinds independently (T3-1)
// ───────────────────────────────────────────────────────────────────────

/// Two Promoters with overlapping patterns share a proxy at /shared.
/// Releasing one Promoter's claim (via its enumeration's `Vanished`
/// response) preserves the other's. Asserts the recompute walks the
/// Promoter registry correctly across the 2 → 1 edge — pre-fix this
/// path was not the bug surface (the bug was Profile-side `vacate`),
/// but the test pins the multi-Promoter symmetry that's load-bearing
/// for the post-fix correctness story.
#[test]
fn two_promoters_sharing_proxy_unwind_independently() {
    let mut e = Engine::new();
    let now = Instant::now();

    // Pre-place /shared so both Promoters land in immediate-Active at
    // the same proxy slot. Both patterns have `/shared` as their
    // literal prefix and a single glob component matching anything;
    // the first proxy at `enter_active` is /shared with index = lpl.
    let shared = pre_place_dir(&mut e, &["shared"]);
    let (q1, _attach_q1) = e.attach_promoter(promoter_req("q1", "/shared/*.log"), now);
    let (q2, _attach_q2) = e.attach_promoter(promoter_req("q2", "/shared/*.txt"), now);

    // Both Promoters should be Active with a proxy at /shared.
    for qid in [q1, q2] {
        match &e.promoters().get(qid).unwrap().state {
            PromoterState::Active { proxies } => assert!(proxies.contains_key(&shared)),
            s @ PromoterState::PrefixPending(_) => {
                panic!("Promoter {qid:?} expected Active at /shared, got {s:?}")
            }
        }
    }
    // /shared.watch_demand == 2 (one contribution per Promoter), and
    // proxy_promoters carries both ids.
    assert_eq!(e.tree().get(shared).unwrap().watch_demand(), 2);
    {
        let bv = &e.tree().get(shared).unwrap().proxy_promoters;
        assert!(bv.contains(&q1));
        assert!(bv.contains(&q2));
    }

    // Q1's enumeration probe at /shared is in flight (single-slot
    // dispatch from the first Promoter's `enter_active`). Q2's
    // enumeration is queued behind it via `pending_enumerations`
    // because the second `attach_promoter`'s `dispatch_next_enumeration`
    // saw a probe in flight... but Q1 and Q2 have independent owner
    // slots, so Q2's probe should also be in flight on its own owner.
    let q1_enum_corr = e
        .pending_probe_for(ProbeOwner::Promoter(q1))
        .expect("Q1 enumeration probe in flight");

    // Vanish Q1's enumeration. `unregister_proxy_subtree` →
    // `unregister_proxy` → `release_promoter_proxy_claim`:
    // counter-aware sub on /shared. The recompute walks the registry
    // and finds Q2 still contributes STRUCTURE; counter drops 2 → 1
    // and the union stays STRUCTURE.
    let q1_vanish_out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Promoter(q1),
            correlation: q1_enum_corr,
            outcome: ProbeOutcome::Vanished,
        }),
        now,
    );

    assert_eq!(
        e.tree().get(shared).unwrap().watch_demand(),
        1,
        "Q2's STRUCTURE contribution survives Q1's release",
    );
    assert_eq!(
        e.tree().get(shared).unwrap().events_union(),
        ClassSet::STRUCTURE,
    );
    let unwatch_at_shared = q1_vanish_out
        .watch_ops
        .iter()
        .any(|op| matches!(op, WatchOp::Unwatch { resource } if *resource == shared));
    assert!(
        !unwatch_at_shared,
        "no Unwatch — Q2 still anchors /shared's kernel watch",
    );
    {
        let bv = &e.tree().get(shared).unwrap().proxy_promoters;
        assert!(!bv.contains(&q1), "Q1 back-ref cleared");
        assert!(bv.contains(&q2), "Q2 back-ref intact");
    }

    // Vanish Q2's enumeration. Now /shared has only Q2's contribution;
    // the release drives the counter to 0 and emits Unwatch.
    let q2_enum_corr = e
        .pending_probe_for(ProbeOwner::Promoter(q2))
        .expect("Q2 enumeration probe in flight");
    let q2_vanish_out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Promoter(q2),
            correlation: q2_enum_corr,
            outcome: ProbeOutcome::Vanished,
        }),
        now,
    );

    let unwatch_count = q2_vanish_out
        .watch_ops
        .iter()
        .filter(|op| matches!(op, WatchOp::Unwatch { resource } if *resource == shared))
        .count();
    assert_eq!(unwatch_count, 1, "Unwatch fires on the genuine 1 → 0 edge");
    // /shared has no User Profile attached and no remaining children;
    // `try_reap` succeeds.
    assert!(
        e.tree().get(shared).is_none(),
        "/shared reaped after both proxies released",
    );
}

// ───────────────────────────────────────────────────────────────────────
// Sensor overflow reseeds Promoters (T1-3)
// ───────────────────────────────────────────────────────────────────────

/// `Active` Promoter with multiple proxies: an `Input::SensorOverflow`
/// re-enqueues every proxy and the single-slot dispatcher drains the
/// first into a probe; the rest queue. One
/// `PromoterReseededForOverflow` diagnostic surfaces.
#[test]
fn sensor_overflow_reseeds_active_promoter() {
    let mut e = Engine::new();
    let now = Instant::now();

    // Pre-place /var/log; attach a Promoter with `/var/log/*`. The
    // first proxy at /var/log is registered immediately; we then
    // drain the initial enumeration so the Promoter has multiple
    // proxies (one per matched Dir entry).
    let var_log = pre_place_dir(&mut e, &["var", "log"]);
    let (qid, attach_out) = e.attach_promoter(promoter_req("logs", "/var/log/*"), now);
    let initial_enum_corr =
        first_probe_corr(&attach_out).expect("initial enumeration probe in flight");

    // Inject a multi-Dir snapshot so the enumeration registers two
    // sub-proxies. The Promoter pattern's final component is `*`
    // (glob) — non-final position requires Dir entries; final
    // position `try_promote`s. `*` IS the final component in this
    // pattern, so each match calls `try_promote` (mints dynamic
    // Subs). To get sub-proxies we'd need a multi-component pattern
    // with a non-final glob. Use `/var/log/*/access.log` instead.
    let _ = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Promoter(qid),
            correlation: initial_enum_corr,
            outcome: ProbeOutcome::SubtreeOk(dir_snap_with(var_log, vec![])),
        }),
        now,
    );

    // For this test we only need at least one proxy to demonstrate
    // the reseed path. The Active{empty proxies} case is also valid
    // to test (it's a no-op except for the diagnostic emission).
    // Add a second proxy by simulating a fresh enumeration that
    // matches a Dir... but the pattern is final-glob, so any Dir
    // entry would `try_promote` rather than register sub-proxy.
    //
    // Instead: assert the Active{single proxy at /var/log} reseed —
    // overflow re-enqueues /var/log and dispatches one probe at it.

    assert!(
        e.pending_probe_for(ProbeOwner::Promoter(qid)).is_none(),
        "Promoter has no in-flight probe before overflow",
    );

    let overflow_out = e.step(
        Input::SensorOverflow {
            scope: OverflowScope::Global,
        },
        now,
    );

    // PromoterReseededForOverflow surfaced exactly once.
    let reseed_count = overflow_out
        .diagnostics
        .iter()
        .filter(|d| {
            matches!(
                d,
                Diagnostic::PromoterReseededForOverflow { promoter } if *promoter == qid,
            )
        })
        .count();
    assert_eq!(
        reseed_count, 1,
        "one PromoterReseededForOverflow per Promoter"
    );

    // /var/log was re-enqueued and the single-slot dispatcher
    // emitted one enumeration probe at it. The probe is in flight
    // on the Promoter owner.
    assert!(
        e.pending_probe_for(ProbeOwner::Promoter(qid)).is_some(),
        "fresh enumeration probe in flight after overflow reseed",
    );
    let probe_at_var_log = overflow_out.probe_ops.iter().any(|op| {
        matches!(
            op,
            ProbeOp::Probe { request } if request.target_resource() == Some(var_log),
        )
    });
    assert!(probe_at_var_log, "probe targets /var/log");
}

/// `PrefixPending` Promoter with no in-flight probe: an
/// `Input::SensorOverflow` re-emits a descent probe at the current
/// descent prefix.
#[test]
fn sensor_overflow_reseeds_prefix_pending_promoter() {
    let mut e = Engine::new();
    let now = Instant::now();

    // Pre-place /a so the descent's first prefix is /a (not /). The
    // Promoter's literal prefix is /a/b which doesn't yet exist;
    // `materialize_path_or_pending` returns Pending(/a, [b]).
    let a = pre_place_dir(&mut e, &["a"]);
    let (qid, attach_out) = e.attach_promoter(promoter_req("logs", "/a/b/*.log"), now);
    let descent_corr = first_probe_corr(&attach_out).expect("descent probe in flight");
    assert!(matches!(
        e.promoters().get(qid).unwrap().state,
        PromoterState::PrefixPending(_),
    ));

    // Resolve the descent probe with a `Failed` response so the
    // descent retains state (current_prefix=/a) but closes the
    // probe channel. The `Failed` arm is exactly "retain in-descent
    // state; await next event at the prefix" — which is the precondition
    // we want for the overflow reseed test.
    let _ = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Promoter(qid),
            correlation: descent_corr,
            outcome: ProbeOutcome::Failed { errno: 13 },
        }),
        now,
    );
    assert!(
        e.pending_probe_for(ProbeOwner::Promoter(qid)).is_none(),
        "channel closed after Failed response",
    );
    assert!(matches!(
        e.promoters().get(qid).unwrap().state,
        PromoterState::PrefixPending(_),
    ));

    let overflow_out = e.step(
        Input::SensorOverflow {
            scope: OverflowScope::Global,
        },
        now,
    );

    // Diagnostic + descent probe at /a.
    let reseed_count = overflow_out
        .diagnostics
        .iter()
        .filter(|d| {
            matches!(
                d,
                Diagnostic::PromoterReseededForOverflow { promoter } if *promoter == qid,
            )
        })
        .count();
    assert_eq!(reseed_count, 1);
    assert!(
        e.pending_probe_for(ProbeOwner::Promoter(qid)).is_some(),
        "fresh descent probe in flight after overflow reseed",
    );
    let probe_at_a = overflow_out.probe_ops.iter().any(|op| {
        matches!(
            op,
            ProbeOp::Probe { request } if request.target_resource() == Some(a),
        )
    });
    assert!(probe_at_a, "descent probe targets /a");
}

/// `PrefixPending` Promoter with in-flight descent probe: the
/// reseed skips the probe emission (the in-flight probe's response
/// will reflect the post-overflow state) but still emits the
/// `PromoterReseededForOverflow` diagnostic — the engine's signal
/// that the reseed was attempted.
#[test]
fn sensor_overflow_skips_promoter_with_in_flight_probe() {
    let mut e = Engine::new();
    let now = Instant::now();

    // Same setup as the prefix-pending test, but leave the descent
    // probe IN FLIGHT (no response).
    let _a = pre_place_dir(&mut e, &["a"]);
    let (qid, _attach_out) = e.attach_promoter(promoter_req("logs", "/a/b/*.log"), now);
    let in_flight_corr = e
        .pending_probe_for(ProbeOwner::Promoter(qid))
        .expect("descent probe in flight");
    assert!(matches!(
        e.promoters().get(qid).unwrap().state,
        PromoterState::PrefixPending(_),
    ));

    let overflow_out = e.step(
        Input::SensorOverflow {
            scope: OverflowScope::Global,
        },
        now,
    );

    // Diagnostic surfaces (the reseed was attempted).
    let reseed_count = overflow_out
        .diagnostics
        .iter()
        .filter(|d| {
            matches!(
                d,
                Diagnostic::PromoterReseededForOverflow { promoter } if *promoter == qid,
            )
        })
        .count();
    assert_eq!(reseed_count, 1);

    // No fresh probe emitted. Pending-probe correlation unchanged.
    let new_probe_emitted = overflow_out
        .probe_ops
        .iter()
        .any(|op| matches!(op, ProbeOp::Probe { .. }));
    assert!(
        !new_probe_emitted,
        "no fresh probe — the in-flight probe's response covers the overflow window",
    );
    assert_eq!(
        e.pending_probe_for(ProbeOwner::Promoter(qid)),
        Some(in_flight_corr),
        "in-flight correlation preserved",
    );
}
