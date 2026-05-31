//! Oracle (a): the driver's same-tick recency coalescing is
//! *outcome-equivalent* to processing every queued event.
//!
//! `EngineDriver::drain_sensor` collapses a within-tick run of a
//! same-`(ResourceId, FsEvent)` recency hint to its first occurrence,
//! flushing the dedup horizon at every barrier (identity FsEvent or
//! any non-FsEvent). This test mirrors that exact rule, then feeds the
//! FULL list and the RULE-COALESCED list into two engines built
//! identically, drives both through one real Standard burst lifecycle
//! (settle → pre-fire verify), and asserts they reach the same
//! observable outcome: same probe target (the LCA of the accumulated
//! dirty set — the quantity that would diverge if a distinct
//! `(resource, event)` were wrongly dropped, an F1-key regression, or
//! a barrier failed to flush), same terminal `ProfileState`, same
//! settled baseline.
//!
//! Diagnostics and exact `StepOutput` multiplicity are deliberately
//! *not* compared: a coalesced-dropped recency hint legitimately
//! elides only duplicate diagnostics and idempotent re-work, and the
//! promoter-proxy collapse is lossy-hint-equivalent, not
//! `StepOutput`-identical — so equality is asserted on the engine's
//! decision, never on the byte-shape of its narration.

use proptest::prelude::*;
use specter_core::testkit::dir_snap;
use specter_core::{
    ClassSet, EntryKind, FsEvent, Input, ProfileStateDiscriminant, ResourceId, ScanConfig,
    SubAttachAnchor,
};
use specter_engine::Engine;
use specter_engine::testkit::{
    anchor_dir, attach_returning, last_probe_path, seed_to_idle, verify,
};
use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

const SETTLE: Duration = Duration::from_millis(100);
const MAX_SETTLE: Duration = Duration::from_secs(6);

/// One element of a within-tick input sequence. The recency variants
/// are exactly the class `drain_sensor` coalesces; `Barrier` is its
/// horizon-flush case (an identity FsEvent — `is_recency()` is false).
#[derive(Clone, Copy, Debug)]
enum Ev {
    /// `ContentChanged` at resource index 0/1/2 ⇒ `root` / `a` / `b`.
    ContentChanged(u8),
    /// `StructureChanged` at the `root` Dir anchor.
    StructRoot,
    /// Identity FsEvent at an unwatched resource: a true barrier
    /// (non-recency) that the engine resolves to a pure no-op, so the
    /// only thing it changes is the dedup horizon.
    Barrier,
}

fn arb_seq() -> impl Strategy<Value = Vec<Ev>> {
    let elem = prop_oneof![
        (0u8..3).prop_map(Ev::ContentChanged),
        Just(Ev::StructRoot),
        Just(Ev::Barrier),
    ];
    prop::collection::vec(elem, 1..14)
}

/// Resolve an `Ev` to its `Input` against the fixture's resources.
fn input_of(ev: Ev, root: ResourceId, a: ResourceId, b: ResourceId) -> Input {
    match ev {
        Ev::ContentChanged(0) => Input::FsEvent {
            resource: root,
            event: FsEvent::ContentChanged,
        },
        Ev::ContentChanged(1) => Input::FsEvent {
            resource: a,
            event: FsEvent::ContentChanged,
        },
        Ev::ContentChanged(_) => Input::FsEvent {
            resource: b,
            event: FsEvent::ContentChanged,
        },
        Ev::StructRoot => Input::FsEvent {
            resource: root,
            event: FsEvent::StructureChanged,
        },
        Ev::Barrier => Input::FsEvent {
            resource: ResourceId::default(),
            event: FsEvent::Removed,
        },
    }
}

/// The production collapse rule, mirrored from
/// `EngineDriver::drain_sensor`: keep the first occurrence of each
/// `(resource, event)` recency hint; drop later same-key duplicates;
/// every barrier clears the horizon and is itself always kept.
fn coalesce(inputs: &[Input]) -> Vec<Input> {
    let mut seen: BTreeSet<(ResourceId, FsEvent)> = BTreeSet::new();
    let mut out = Vec::with_capacity(inputs.len());
    for inp in inputs {
        match inp {
            Input::FsEvent { resource, event } if event.is_recency() => {
                if seen.insert((*resource, *event)) {
                    out.push(inp.clone());
                }
            }
            other => {
                seen.clear();
                out.push(other.clone());
            }
        }
    }
    out
}

/// A fresh engine with a recursive subtree-root Sub on Dir `src`,
/// seeded to a pinned `Idle` baseline of `{a, b}` covered Files.
/// Returns the engine, profile, the three live resources, the
/// baseline snapshot, and the post-seed instant.
fn seeded() -> (
    Engine,
    specter_core::ProfileId,
    ResourceId,
    ResourceId,
    ResourceId,
    Arc<specter_core::DirSnapshot>,
    Instant,
) {
    let mut e = Engine::new();
    let root = anchor_dir(&mut e, "src");
    let t0 = Instant::now();
    let (_sid, pid, _out) = attach_returning(
        &mut e,
        "build",
        SubAttachAnchor::Resource(root),
        ScanConfig::builder().recursive(true).build(),
        ClassSet::DEFAULT_SUBTREE_ROOT,
        MAX_SETTLE,
        t0,
    );
    let snap = dir_snap(&[("a", EntryKind::File, 1), ("b", EntryKind::File, 1)]);
    let seed_done = seed_to_idle(&mut e, pid, &snap, t0);
    let a = e.tree().lookup(Some(root), "a").expect("covered child a");
    let b = e.tree().lookup(Some(root), "b").expect("covered child b");
    (e, pid, root, a, b, snap, seed_done)
}

/// Step every input at one shared `now` (one tick), force the
/// Batching settle, capture the emitted probe's target, then run the
/// pre-fire verify to a terminal `Idle`. Returns the observable outcome:
/// `(probe target, terminal state discriminant, settled baseline
/// hash)`.
fn drive(inputs: &[Input]) -> (Option<PathBuf>, ProfileStateDiscriminant, Option<u128>) {
    let (mut e, pid, _root, _a, _b, snap, seed_done) = seeded();
    let now = seed_done + Duration::from_millis(1);
    for inp in inputs {
        e.step(inp.clone(), now);
    }

    // Force the settle, keeping the StepOutput that carries the probe
    // (drain_due discards outputs). All recency inputs drive Batching;
    // its settle expiry emits the Verifying probe whose target is the
    // LCA of the accumulated dirty set.
    let at = now + SETTLE * 4;
    let mut probe_target = None;
    while let Some(en) = e.pop_expired(at) {
        let out = e.step(
            Input::TimerExpired {
                profile: en.profile,
                kind: en.kind,
                id: en.id,
            },
            at,
        );
        if let Some(p) = last_probe_path(&out) {
            probe_target = Some(p);
        }
    }

    // The probe is in flight (≥1 recency input ⇒ a burst started).
    // Settle-spaced hash-equal samples ⇒ Stable; the Sub already
    // fired during seed ⇒ B1 dedup ⇒ finish to Idle, baseline intact.
    let _n2 = verify(&mut e, pid, &snap, at);

    let p = e.profiles().get(pid).expect("profile alive");
    (probe_target, p.state().discriminant(), p.settled_hash())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Coalesced ≡ full on every observable the lossy-hint contract
    /// must preserve, over random within-tick interleavings.
    #[test]
    fn coalesced_drain_matches_full_drain(seq in arb_seq()) {
        // Resolve against a throwaway fixture purely to map `Ev`s to
        // `Input`s with stable resource ids (every `seeded()` builds
        // the same tree, so the ids match the ones `drive` resolves).
        let (_e, _pid, root, a, b, _snap, _t) = seeded();
        let full: Vec<Input> = seq.iter().map(|ev| input_of(*ev, root, a, b)).collect();

        // At least one recency hint ⇒ a burst actually starts (else
        // both lists are barriers-only: trivially equivalent, but no
        // probe to compare — out of this property's scope).
        prop_assume!(full.iter().any(|i| matches!(
            i, Input::FsEvent { event, .. } if event.is_recency()
        )));

        let coalesced = coalesce(&full);
        // The collapse must be a strict prefix-preserving reduction.
        prop_assert!(coalesced.len() <= full.len());

        let full_outcome = drive(&full);
        let coalesced_outcome = drive(&coalesced);
        prop_assert_eq!(
            coalesced_outcome,
            full_outcome,
            "coalesced drain diverged from full drain (probe target / \
             terminal state / baseline)",
        );
    }
}
