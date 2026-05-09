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
//!
//! Mirrors plan §19.2 (`promoter_lifecycle.rs`).

#![allow(
    clippy::items_after_statements,
    clippy::manual_let_else,
    clippy::missing_const_for_fn,
    clippy::needless_pass_by_value,
    clippy::option_if_let_else,
    clippy::single_match_else,
    clippy::too_many_lines
)]

use compact_str::CompactString;
use specter_core::{
    ChildEntry, ClassSet, CommandTemplate, Diagnostic, DirChild, DirMeta, DirSnapshot, EffectScope,
    EntryKind, Input, LeafEntry, PatternSpec, ProbeOp, ProbeOutcome, ProbeOwner, ProbeResponse,
    PromoterAttachRequest, PromoterRegistryDiff, PromoterState, ResourceId, ResourceKind,
    ResourceRole, ScanConfig, WatchRegistryDiff,
};
use specter_engine::Engine;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, Instant, UNIX_EPOCH};

const SETTLE: Duration = Duration::from_millis(100);
const MAX_SETTLE: Duration = Duration::from_secs(6);
const FS_ROOT_SEG: &str = "/";

fn empty_command() -> CommandTemplate {
    CommandTemplate::new([specter_core::ArgTemplate::new([
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
        command: empty_command(),
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

    // ---- Phase 1: PromoterAttached + initial enumeration probe ----
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

    // ---- Phase 2: enumeration response → promotion ----
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

    // ---- Phase 3: dynamic Sub's anchor descent ----
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

    // ---- Phase 4: Seed-burst baseline establishes; no fire ----
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

    // ---- Phase 5: reap Promoter → cascade ----
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
        empty_command(),
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
