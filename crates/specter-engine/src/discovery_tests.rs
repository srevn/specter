//! Unit pins for the discovery reconcile building blocks: the pure terminus collector, the
//! template ⟺ `MatchChain` attach boundary, and the reconcile's non-Dir-anchor totality arm. The
//! end-to-end reconcile lifecycle lives in `tests/discovery_lifecycle.rs`.

use super::{ChainTerminus, collect_chain_termini};
use crate::Engine;
use crate::testkit::{MAX_SETTLE, SETTLE};
use crate::testkit::{attach_discovery, discovery_subs_of, mint_template, pre_place_dir};
use compact_str::CompactString;
use specter_core::testkit::{covered, dir_snap, dir_snap_nested, empty_program, leaf, uncovered};
use specter_core::{
    ClassSet, EffectScope, EntryKind, Input, PatternSpec, ProfileIdentity, ScanConfig, StepOutput,
    SubAttachAnchor, SubAttachRequest, SubParams,
};
use std::sync::Arc;
use std::time::Instant;

fn terminus(segments: &[&str], kind: EntryKind) -> ChainTerminus {
    ChainTerminus {
        segments: segments.iter().map(|s| CompactString::new(s)).collect(),
        kind,
    }
}

/// td = 1: every root entry is a terminus, whatever its kind — Dir, File, and Symlink all mint
/// (the Promoter-parity rule: `EntryKind → ResourceKind` folds non-dirs to `File` downstream, but
/// the collector reports the snapshot's own kind). Order is the `BTreeMap`'s lexicographic walk.
#[test]
fn termini_at_depth_one_collect_every_entry_kind_in_lexicographic_order() {
    let root = dir_snap(&[
        ("c", EntryKind::Symlink, 3),
        ("a", EntryKind::Dir, 1),
        ("b.log", EntryKind::File, 2),
    ]);
    assert_eq!(
        collect_chain_termini(&root, 1),
        vec![
            terminus(&["a"], EntryKind::Dir),
            terminus(&["b.log"], EntryKind::File),
            terminus(&["c"], EntryKind::Symlink),
        ],
    );
}

/// td = 3: the collector descends `Covered` chain dirs only, and a terminus-level `Covered` dir
/// (a shape the walker never emits — `descends_into` refuses at td) still collects as a Dir
/// terminus rather than being descended past the chain bound.
#[test]
fn termini_at_depth_three_walk_covered_chains_to_the_bound() {
    let root = dir_snap_nested(&[(
        "x",
        covered(dir_snap_nested(&[(
            "y",
            covered(dir_snap_nested(&[
                ("log", uncovered(10)),
                ("w", covered(dir_snap(&[("deep", EntryKind::File, 99)]))),
                ("z.txt", leaf(EntryKind::File, 11)),
            ])),
        )])),
    )]);
    assert_eq!(
        collect_chain_termini(&root, 3),
        vec![
            terminus(&["x", "y", "log"], EntryKind::Dir),
            terminus(&["x", "y", "w"], EntryKind::Dir),
            terminus(&["x", "y", "z.txt"], EntryKind::File),
        ],
    );
}

/// Adversarial snapshot: a `Leaf` and an `Uncovered` Dir strictly above the terminus depth are
/// skipped (totality, not policy — the pruned walk never emits them); only the `Covered` chain's
/// entries at the bound collect. An empty root collects nothing.
#[test]
fn entries_above_the_terminus_that_cannot_recurse_are_skipped() {
    let root = dir_snap_nested(&[
        ("early.txt", leaf(EntryKind::File, 1)),
        ("sealed", uncovered(2)),
        ("chain", covered(dir_snap_nested(&[("log", uncovered(3))]))),
    ]);
    assert_eq!(
        collect_chain_termini(&root, 2),
        vec![terminus(&["chain", "log"], EntryKind::Dir)],
    );
    assert!(collect_chain_termini(&dir_snap(&[]), 1).is_empty());
}

/// The ⟺ attach boundary, template direction: a template on a non-chain Profile is
/// unconstructable — its Profile would classify a firing consequence it can never use.
#[test]
#[should_panic(expected = "SubParams::template ⟺ ScanConfig::MatchChain")]
fn template_on_non_chain_profile_is_unconstructable() {
    let mut e = Engine::new();
    let srv = pre_place_dir(&mut e, &["srv"]);
    let _ = e.step(
        Input::AttachSub(SubAttachRequest::from_parts(
            SubAttachAnchor::Resource(srv),
            ProfileIdentity {
                config: ScanConfig::builder().build(),
                max_settle: MAX_SETTLE,
                events: ClassSet::STRUCTURE,
            },
            SubParams {
                name: "disc".into(),
                program: empty_program(),
                scope: EffectScope::SubtreeRoot,
                settle: SETTLE,
                log_output: false,
                source_promoter: None,
                template: Some(mint_template()),
                source_discovery: None,
            },
        )),
        Instant::now(),
    );
}

/// The ⟺ attach boundary, shape direction: a plain Sub on a chain Profile is unconstructable — it
/// could never react (a chain Profile mints attachments, never Effects). This same assert fires
/// transitively on a chain-shaped *template*: its mint is a template-less Sub on a chain Profile.
#[test]
#[should_panic(expected = "SubParams::template ⟺ ScanConfig::MatchChain")]
fn plain_sub_on_chain_profile_is_unconstructable() {
    let mut e = Engine::new();
    let srv = pre_place_dir(&mut e, &["srv"]);
    let _ = e.step(
        Input::AttachSub(SubAttachRequest::from_parts(
            SubAttachAnchor::Resource(srv),
            ProfileIdentity {
                config: ScanConfig::MatchChain(Arc::new(
                    PatternSpec::parse("/srv/*").expect("valid pattern"),
                )),
                max_settle: MAX_SETTLE,
                events: ClassSet::STRUCTURE,
            },
            SubParams {
                name: "plain".into(),
                program: empty_program(),
                scope: EffectScope::SubtreeRoot,
                settle: SETTLE,
                log_output: false,
                source_promoter: None,
                template: None,
                source_discovery: None,
            },
        )),
        Instant::now(),
    );
}

/// The reconcile's non-Dir-anchor totality arm: with no Dir `current` (the cold probe hasn't
/// answered yet — the same shape as an anchor replaced by a file), reconcile walks no termini and
/// mints nothing; the recovery machinery owns whatever replaced the anchor.
#[test]
fn reconcile_without_dir_current_mints_nothing() {
    let mut e = Engine::new();
    let srv = pre_place_dir(&mut e, &["srv"]);
    let now = Instant::now();
    let (sid, pid) = attach_discovery(
        &mut e,
        "disc",
        SubAttachAnchor::Resource(srv),
        "/srv/*",
        mint_template(),
        now,
    );

    let mut out = StepOutput::default();
    e.reconcile_matches(pid, now, &mut out);
    assert!(
        out.diagnostics.is_empty(),
        "no termini ⇒ no mints, no narration; got {:?}",
        out.diagnostics,
    );
    assert!(discovery_subs_of(&e, sid).is_empty(), "nothing minted");
    let _ = e.cancel_all_in_flight_probes();
}
