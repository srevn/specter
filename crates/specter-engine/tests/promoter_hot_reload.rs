//! Cross-cutting Promoter hot-reload via [`Input::ConfigDiff`].
//!
//! Validates the engine's [`WatchRegistryDiff`] composition: each diff carries `subs` (static
//! [[watch]] adds/removes/modifies) and `promoters` (dynamic [[watch]] equivalents) and the engine
//! applies both halves atomically in a single step. The assertion surface is the diagnostic stream
//! — the same stream the bin's diagnostic-driven `loader.ids` / `loader.promoter_ids`
//! reconciliation reads.
//!
//! Inline `transitions_tests.rs` pins the per-half edges (Sub-only vs Promoter-only diffs); this
//! file pins the *mixed* cases — Sub + Promoter add together, modify together, remove together —
//! and the diagnostic-stream contract that the bin depends on.

use compact_str::CompactString;
use specter_core::testkit::empty_program;
use specter_core::{
    ClassSet, Diagnostic, EffectScope, Input, PromoterRegistryDiff, ResourceKind, ResourceRole,
    ScanConfig, SubAttachAnchor, SubAttachRequest, SubParams, SubRegistryDiff, WatchRegistryDiff,
};
use specter_engine::Engine;
use specter_engine::testkit::{attach_promoter, promoter_req};
use std::time::{Duration, Instant};

const SETTLE: Duration = Duration::from_millis(100);
const MAX_SETTLE: Duration = Duration::from_secs(6);

fn sub_req_at_root(name: &str, e: &mut Engine) -> SubAttachRequest {
    let r = e.tree_mut().ensure_root("src", ResourceRole::User);
    e.tree_mut().set_kind(r, ResourceKind::Dir);
    SubAttachRequest::for_anchor(
        name.into(),
        SubAttachAnchor::Resource(r),
        ScanConfig::builder().recursive(true).build(),
        MAX_SETTLE,
        SETTLE,
        empty_program(),
        EffectScope::SubtreeRoot,
        ClassSet::EMPTY,
        false,
    )
}

/// `subs.added` + `promoters.added` in the same diff: both attach in a single step, both lifecycle
/// diagnostics emit, and both registries reflect the addition.
///
/// The order of emission follows `on_config_diff`'s internal sequencing (Sub side first, Promoter
/// side second), which the engine's inline test
/// `transitions_tests.rs::config_diff_sub_side_runs_before_promoter_side` pins. Here we validate
/// the *cross-stream* invariant that one diff produces both `SubAttached` and `PromoterAttached` in
/// the same `StepOutput.diagnostics`.
#[test]
fn mixed_add_diff_emits_both_lifecycle_diagnostics() {
    let mut e = Engine::new();
    let sub_req = sub_req_at_root("build", &mut e);

    let diff = WatchRegistryDiff {
        subs: SubRegistryDiff {
            added: vec![sub_req],
            ..Default::default()
        },
        promoters: PromoterRegistryDiff {
            added: vec![promoter_req("logs", "/var/log/*.log")],
            ..Default::default()
        },
    };
    let out = e.step(Input::ConfigDiff(diff), Instant::now());

    let mut saw_sub_attached_static = false;
    let mut saw_promoter_attached = false;
    for d in &out.diagnostics {
        match d {
            Diagnostic::SubAttached {
                name,
                source_promoter: None,
                ..
            } if name == "build" => saw_sub_attached_static = true,
            Diagnostic::PromoterAttached { name, .. } if name == "logs" => {
                saw_promoter_attached = true;
            }
            _ => {}
        }
    }
    assert!(
        saw_sub_attached_static,
        "static SubAttached emitted for `build`; got {:?}",
        out.diagnostics,
    );
    assert!(
        saw_promoter_attached,
        "PromoterAttached emitted for `logs`; got {:?}",
        out.diagnostics,
    );
    let _ = e.cancel_all_in_flight_probes();
}

/// `subs.modified_params` + `promoters.modified` in one diff exercise the two arms' contrasting
/// shapes: the Sub side rebinds the live Sub in place (preserving [`SubId`]) while the Promoter
/// side runs wholesale reap-then-attach (minting a fresh [`PromoterId`]). The Promoter pair is
/// ordered: `PromoterReaped` precedes the fresh `PromoterAttached` so the bin's reconciliation
/// discipline observes the correct end state.
// Codebase-standard SubId/PromoterId names; the old↔new × Sub↔Promoter parallelism is this test's
// subject, not accidental similarity.
#[allow(clippy::similar_names)]
#[test]
fn mixed_modify_diff_rebinds_sub_and_reaps_then_attaches_promoter() {
    let mut e = Engine::new();
    let sub_req = sub_req_at_root("build", &mut e);

    // Initial attach: register one Sub and one Promoter.
    let now = Instant::now();
    let attach_out_a = e.step(Input::AttachSub(sub_req.clone()), now);
    let old_sid =
        specter_core::testkit::first_attached_sub(&attach_out_a).expect("attach_sub succeeded");
    let old_pid = attach_promoter(&mut e, "logs", "/var/log/*.log", now);

    // Same anchor, same identity (the params bucket discriminator) — the Sub falls into
    // `modified_params` and the engine rebinds in place. The Promoter changes its pattern source
    // (identity in the operator sense) and so goes through the wholesale reap+attach.
    let modify_sub_req = SubAttachRequest {
        anchor: sub_req.anchor,
        identity: sub_req.identity,
        params: SubParams {
            name: "build".into(),
            ..sub_req.params
        },
    };
    let modify_promoter_req = promoter_req("logs", "/var/log/*.json");

    let diff = WatchRegistryDiff {
        subs: SubRegistryDiff {
            modified_params: vec![modify_sub_req],
            ..Default::default()
        },
        promoters: PromoterRegistryDiff {
            modified: vec![modify_promoter_req],
            ..Default::default()
        },
    };
    let out = e.step(Input::ConfigDiff(diff), now);

    // Promoter: PromoterReaped(old_pid) precedes PromoterAttached(name=logs). Sub: SubRebound for
    // the live SubId, no SubAttached (rebind preserves the id).
    let mut saw_promoter_reaped = false;
    let mut saw_promoter_attached_after = false;
    let mut saw_sub_rebound = false;
    for d in &out.diagnostics {
        match d {
            Diagnostic::PromoterReaped { promoter } if *promoter == old_pid => {
                saw_promoter_reaped = true;
            }
            Diagnostic::PromoterAttached { name, .. } if name == "logs" => {
                assert!(
                    saw_promoter_reaped,
                    "PromoterAttached for `logs` after reap; got {:?}",
                    out.diagnostics,
                );
                saw_promoter_attached_after = true;
            }
            Diagnostic::SubRebound { sub } if *sub == old_sid => saw_sub_rebound = true,
            Diagnostic::SubAttached { sub, .. } if *sub == old_sid => panic!(
                "modified_params must rebind in place, not re-attach (got SubAttached for old_sid)",
            ),
            _ => {}
        }
    }
    assert!(saw_promoter_reaped, "PromoterReaped emitted on modify");
    assert!(
        saw_promoter_attached_after,
        "PromoterAttached emitted on modify"
    );
    assert!(
        saw_sub_rebound,
        "SubRebound emitted on modified_params (in-place rebind)",
    );

    // Registry state: Sub preserves its id (rebind); Promoter mints a fresh id (wholesale modify).
    let new_sid = e.subs().find_by_name("build").expect("build still live");
    assert_eq!(
        new_sid, old_sid,
        "modified_params rebind preserves the SubId",
    );
    let new_pid = e
        .promoters()
        .find_by_name("logs")
        .expect("logs re-registered");
    assert_ne!(new_pid, old_pid, "Promoter modify mints a fresh id");
    let _ = e.cancel_all_in_flight_probes();
}

/// `subs.removed` + `promoters.removed` together: the diff's authoritative removal lists drive the
/// engine through `detach_sub_inner` and `reap_promoter_inner` respectively, and the diagnostic
/// stream surfaces `PromoterReaped` for the Promoter side. The Sub side does not emit a per-detach
/// diagnostic; the diff's `removed` list is the source of truth the bin uses.
#[test]
fn mixed_remove_diff_emits_promoter_reaped_only() {
    let mut e = Engine::new();
    let sub_req = sub_req_at_root("build", &mut e);
    let now = Instant::now();
    let attach_out = e.step(Input::AttachSub(sub_req), now);
    let sid = specter_core::testkit::first_attached_sub(&attach_out).expect("attach_sub succeeded");
    let pid = attach_promoter(&mut e, "logs", "/var/log/*.log", now);

    let diff = WatchRegistryDiff {
        subs: SubRegistryDiff {
            removed: vec![CompactString::from("build")],
            ..Default::default()
        },
        promoters: PromoterRegistryDiff {
            removed: vec![CompactString::from("logs")],
            ..Default::default()
        },
    };
    let out = e.step(Input::ConfigDiff(diff), now);

    assert!(e.subs().get(sid).is_none(), "Sub removed");
    assert!(e.promoters().get(pid).is_none(), "Promoter reaped");

    let saw_promoter_reaped = out
        .diagnostics
        .iter()
        .any(|d| matches!(d, Diagnostic::PromoterReaped { promoter } if *promoter == pid));
    assert!(saw_promoter_reaped, "PromoterReaped emitted on remove");

    // No SubAttached / PromoterAttached on a remove-only diff.
    let saw_any_attach = out.diagnostics.iter().any(|d| {
        matches!(
            d,
            Diagnostic::SubAttached { .. } | Diagnostic::PromoterAttached { .. }
        )
    });
    assert!(
        !saw_any_attach,
        "no attach diagnostics on a remove-only diff"
    );
    let _ = e.cancel_all_in_flight_probes();
}

/// Static→dynamic migration via path edit: same name, but the path crossed `is_dynamic`. The diff
/// layer (in `specter-config`) emits `subs.removed + promoters.added`; the engine applies both in
/// one step. The diagnostic stream pairs the silent Sub removal (no per-detach variant) with a
/// `PromoterAttached` for the new Promoter — exactly the surface the bin's reconciliation uses to
/// swap the entry between `loader.ids` and `loader.promoter_ids` in one pass.
#[test]
fn static_to_dynamic_migration_diff_swaps_via_diagnostic_stream() {
    let mut e = Engine::new();
    let sub_req = sub_req_at_root("foo", &mut e);
    let now = Instant::now();
    let attach_out = e.step(Input::AttachSub(sub_req), now);
    let sid = specter_core::testkit::first_attached_sub(&attach_out).expect("attach_sub succeeded");

    let diff = WatchRegistryDiff {
        subs: SubRegistryDiff {
            removed: vec![CompactString::from("foo")],
            ..Default::default()
        },
        promoters: PromoterRegistryDiff {
            added: vec![promoter_req("foo", "/var/log/*.log")],
            ..Default::default()
        },
    };
    let out = e.step(Input::ConfigDiff(diff), now);

    assert!(e.subs().get(sid).is_none(), "static `foo` removed");
    let new_pid = e
        .promoters()
        .find_by_name("foo")
        .expect("dynamic `foo` registered");
    let saw_promoter_attached = out.diagnostics.iter().any(|d| matches!(
        d,
        Diagnostic::PromoterAttached { promoter, name } if *promoter == new_pid && name == "foo",
    ));
    assert!(
        saw_promoter_attached,
        "PromoterAttached emitted with the migrated name; got {:?}",
        out.diagnostics,
    );
    let _ = e.cancel_all_in_flight_probes();
}

/// Reverse direction: dynamic→static migration. The diff emits `promoters.removed + subs.added`;
/// the engine reaps the Promoter and attaches a static Sub at the same name. The diagnostic stream
/// pairs `PromoterReaped` with a static `SubAttached(source_promoter=None)`.
#[test]
fn dynamic_to_static_migration_diff_swaps_via_diagnostic_stream() {
    let mut e = Engine::new();
    let now = Instant::now();
    let pid = attach_promoter(&mut e, "foo", "/var/log/*.log", now);
    let static_req = sub_req_at_root("foo", &mut e);

    let diff = WatchRegistryDiff {
        subs: SubRegistryDiff {
            added: vec![static_req],
            ..Default::default()
        },
        promoters: PromoterRegistryDiff {
            removed: vec![CompactString::from("foo")],
            ..Default::default()
        },
    };
    let out = e.step(Input::ConfigDiff(diff), now);

    assert!(e.promoters().get(pid).is_none(), "dynamic `foo` reaped");
    let new_sid = e
        .subs()
        .find_by_name("foo")
        .expect("static `foo` registered");

    let mut saw_promoter_reaped = false;
    let mut saw_static_sub_attached = false;
    for d in &out.diagnostics {
        match d {
            Diagnostic::PromoterReaped { promoter } if *promoter == pid => {
                saw_promoter_reaped = true;
            }
            Diagnostic::SubAttached {
                sub,
                name,
                source_promoter: None,
            } if *sub == new_sid && name == "foo" => saw_static_sub_attached = true,
            _ => {}
        }
    }
    assert!(
        saw_promoter_reaped,
        "PromoterReaped emitted; got {:?}",
        out.diagnostics
    );
    assert!(
        saw_static_sub_attached,
        "static SubAttached emitted; got {:?}",
        out.diagnostics,
    );
    let _ = e.cancel_all_in_flight_probes();
}
