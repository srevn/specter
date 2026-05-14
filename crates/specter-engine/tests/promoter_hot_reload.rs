//! Cross-cutting Promoter hot-reload via [`Input::ConfigDiff`].
//!
//! Validates the engine's [`WatchRegistryDiff`] composition: each
//! diff carries `subs` (static [[watch]] adds/removes/modifies)
//! and `promoters` (dynamic [[watch]] equivalents) and the engine
//! applies both halves atomically in a single step. The assertion
//! surface is the diagnostic stream — the same stream the bin's
//! diagnostic-driven `loader.ids` / `loader.promoter_ids`
//! reconciliation reads.
//!
//! Inline `transitions_tests.rs` pins the per-half edges
//! (Sub-only vs Promoter-only diffs); this file pins the *mixed*
//! cases — Sub + Promoter add together, modify together, remove
//! together — and the diagnostic-stream contract that the bin
//! depends on.

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

use specter_core::testkit::single_exec_program;
use specter_core::{
    ActionProgram, ClassSet, Diagnostic, EffectScope, Input, PatternSpec, PromoterAttachRequest,
    PromoterRegistryDiff, ResourceKind, ResourceRole, ScanConfig, SubAttachRequest,
    SubRegistryDiff, WatchRegistryDiff,
};
use specter_engine::Engine;
use std::sync::Arc;
use std::time::{Duration, Instant};

const SETTLE: Duration = Duration::from_millis(100);
const MAX_SETTLE: Duration = Duration::from_secs(6);

fn empty_program() -> Arc<ActionProgram> {
    single_exec_program([specter_core::ArgTemplate::new([
        specter_core::ArgPart::literal("/bin/true"),
    ])])
}

fn sub_req_at_root(name: &str, e: &mut Engine) -> SubAttachRequest {
    let r = e.tree_mut().ensure_root("src", ResourceRole::User);
    e.tree_mut().set_kind(r, ResourceKind::Dir);
    SubAttachRequest::for_resource(
        name.to_owned(),
        r,
        ScanConfig::builder().recursive(true).build(),
        MAX_SETTLE,
        SETTLE,
        empty_program(),
        EffectScope::SubtreeRoot,
        ClassSet::EMPTY,
        false,
    )
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

/// `subs.added` + `promoters.added` in the same diff: both attach
/// in a single step, both lifecycle diagnostics emit, and both
/// registries reflect the addition.
///
/// The order of emission follows `on_config_diff`'s internal
/// sequencing (Sub side first, Promoter side second), which the
/// engine's inline test
/// `transitions_tests.rs::config_diff_sub_side_runs_before_promoter_side`
/// pins. Here we validate the *cross-stream* invariant that one
/// diff produces both `SubAttached` and `PromoterAttached` in the
/// same `StepOutput.diagnostics`.
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
}

/// `subs.modified` + `promoters.modified` together: each modify is
/// reap-then-attach. The `SubAttached` for the modified static Sub
/// and the `PromoterAttached` for the modified Promoter both emit;
/// each is preceded by the corresponding reap (no `SubDetached`
/// variant exists — the absence is the contract — but
/// `PromoterReaped` precedes the fresh `PromoterAttached`).
///
/// Pins the bin's reconciliation discipline: a modified entry
/// surfaces as one `*Reaped` for the old + one `*Attached` for
/// the new, in that order; the bin's name-keyed map overwrite
/// produces the correct end state.
#[test]
fn mixed_modify_diff_emits_reap_then_attach_for_both_streams() {
    let mut e = Engine::new();
    let sub_req = sub_req_at_root("build", &mut e);

    // Initial attach: register one Sub and one Promoter.
    let now = Instant::now();
    let attach_out_a = e.step(Input::AttachSub(sub_req.clone()), now);
    let old_sid =
        specter_core::testkit::first_attached_sub(&attach_out_a).expect("attach_sub succeeded");
    let attach_out_b = e.step(
        Input::AttachPromoter(promoter_req("logs", "/var/log/*.log")),
        now,
    );
    let old_pid = specter_core::testkit::first_attached_promoter(&attach_out_b)
        .expect("attach_promoter succeeded");

    // Build a modify-diff: same names; both entries with a
    // structurally-distinct request. The static Sub keeps its
    // resource but with a different command (the test rig's
    // `empty_program()` — for symmetry; the registry minted a
    // fresh id regardless on modify). The Promoter changes its
    // pattern (pattern source changed).
    let modify_sub_req = SubAttachRequest {
        name: "build".to_owned(),
        ..sub_req
    };
    let modify_promoter_req = promoter_req("logs", "/var/log/*.json");

    let diff = WatchRegistryDiff {
        subs: SubRegistryDiff {
            modified: vec![(old_sid, modify_sub_req)],
            ..Default::default()
        },
        promoters: PromoterRegistryDiff {
            modified: vec![(old_pid, modify_promoter_req)],
            ..Default::default()
        },
    };
    let out = e.step(Input::ConfigDiff(diff), now);

    // PromoterReaped(old_pid) precedes PromoterAttached(name=logs)
    // for the modified Promoter. SubAttached(name=build) appears
    // for the modified Sub. (No SubDetached / SubReaped variant —
    // the diff's `removed`/`modified` lists are the authoritative
    // source for what disappeared.)
    let mut saw_promoter_reaped = false;
    let mut saw_promoter_attached_after = false;
    let mut saw_sub_attached_for_build = false;
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
            Diagnostic::SubAttached {
                name,
                source_promoter: None,
                ..
            } if name == "build" => saw_sub_attached_for_build = true,
            _ => {}
        }
    }
    assert!(saw_promoter_reaped, "PromoterReaped emitted on modify");
    assert!(
        saw_promoter_attached_after,
        "PromoterAttached emitted on modify"
    );
    assert!(
        saw_sub_attached_for_build,
        "SubAttached emitted on Sub modify"
    );

    // Registry state: fresh ids minted, old ids gone.
    let new_sid = e.subs().find_by_name("build").expect("build re-registered");
    assert_ne!(new_sid, old_sid, "Sub modify mints a fresh id");
    let new_pid = e
        .promoters()
        .find_by_name("logs")
        .expect("logs re-registered");
    assert_ne!(new_pid, old_pid, "Promoter modify mints a fresh id");
}

/// `subs.removed` + `promoters.removed` together: the diff's
/// authoritative removal lists drive the engine through
/// `detach_sub_inner` and `reap_promoter_inner` respectively, and
/// the diagnostic stream surfaces `PromoterReaped` for the
/// Promoter side. The Sub side does not emit a per-detach
/// diagnostic; the diff's `removed` list is the source of truth
/// the bin uses.
#[test]
fn mixed_remove_diff_emits_promoter_reaped_only() {
    let mut e = Engine::new();
    let sub_req = sub_req_at_root("build", &mut e);
    let now = Instant::now();
    let attach_out = e.step(Input::AttachSub(sub_req), now);
    let sid = specter_core::testkit::first_attached_sub(&attach_out).expect("attach_sub succeeded");
    let attach_out = e.step(
        Input::AttachPromoter(promoter_req("logs", "/var/log/*.log")),
        now,
    );
    let pid = specter_core::testkit::first_attached_promoter(&attach_out)
        .expect("attach_promoter succeeded");

    let diff = WatchRegistryDiff {
        subs: SubRegistryDiff {
            removed: vec![sid],
            ..Default::default()
        },
        promoters: PromoterRegistryDiff {
            removed: vec![pid],
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
}

/// Static→dynamic migration via path edit: same name, but the path
/// crossed `is_dynamic`. The diff layer (in `specter-config`) emits
/// `subs.removed + promoters.added`; the engine applies both in
/// one step. The diagnostic stream pairs the silent Sub removal
/// (no per-detach variant) with a `PromoterAttached` for the new
/// Promoter — exactly the surface the bin's reconciliation uses
/// to swap the entry between `loader.ids` and
/// `loader.promoter_ids` in one pass.
#[test]
fn static_to_dynamic_migration_diff_swaps_via_diagnostic_stream() {
    let mut e = Engine::new();
    let sub_req = sub_req_at_root("foo", &mut e);
    let now = Instant::now();
    let attach_out = e.step(Input::AttachSub(sub_req), now);
    let sid = specter_core::testkit::first_attached_sub(&attach_out).expect("attach_sub succeeded");

    let diff = WatchRegistryDiff {
        subs: SubRegistryDiff {
            removed: vec![sid],
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
}

/// Reverse direction: dynamic→static migration. The diff emits
/// `promoters.removed + subs.added`; the engine reaps the
/// Promoter and attaches a static Sub at the same name. The
/// diagnostic stream pairs `PromoterReaped` with a static
/// `SubAttached(source_promoter=None)`.
#[test]
fn dynamic_to_static_migration_diff_swaps_via_diagnostic_stream() {
    let mut e = Engine::new();
    let now = Instant::now();
    let attach_out = e.step(
        Input::AttachPromoter(promoter_req("foo", "/var/log/*.log")),
        now,
    );
    let pid = specter_core::testkit::first_attached_promoter(&attach_out)
        .expect("attach_promoter succeeded");
    let static_req = sub_req_at_root("foo", &mut e);

    let diff = WatchRegistryDiff {
        subs: SubRegistryDiff {
            added: vec![static_req],
            ..Default::default()
        },
        promoters: PromoterRegistryDiff {
            removed: vec![pid],
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
}
