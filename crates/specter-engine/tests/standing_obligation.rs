//! Standing-obligation lifecycle integration. Evidence a non-consuming verdict terminal parks
//! (`Abandon` — the bounded ceiling hit on a persistently unreadable chain) survives the burst
//! that witnessed it and rides the next burst's proof obligation, so the formerly-dirty region
//! cannot mtime-skip against the stale baseline once the obstruction heals — the
//! loud-once-then-silent-forever miss is closed. Recovery rides the next in-mask signal, never a
//! poll: the park preserves the terminal's stop-burning-probes intent.

use compact_str::CompactString;
use specter_core::testkit::{dir_snap, empty_program, first_attached_sub, proven};
use specter_core::{
    ClassSet, Diagnostic, EffectScope, EntryKind, FsEvent, Input, ProbeOp, ProbeOutcome,
    ProbeRequest, ProbeResponse, ProfileState, ProofAuthority, ProofObligation, ResourceKind,
    ResourceRole, ScanConfig, SubAttachAnchor, SubAttachRequest,
};
use specter_engine::Engine;
use specter_engine::testkit::{anchor_dir, drain_due, pid_of, seed_to_idle};
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

const SETTLE: Duration = Duration::from_millis(100);
const MAX_SETTLE: Duration = Duration::from_secs(6);

#[test]
fn abandoned_burst_evidence_rides_next_bursts_obligation_and_fires() {
    // A Standard burst opens at file `a`; every verify refuses on its chain (`Undischarged`, a
    // chmod-000-class obstruction); the burst deadline forces and the forced refusal folds
    // `Abandon` — diagnostic, no commit, and the burst's captured paths park into the standing
    // obligation. A later, unrelated in-mask event at file `b` opens a fresh burst: the drained
    // obligation widens its probe's `Chains` over the parked `a` chain, and the fresh
    // Authoritative read fires the originally-witnessed change.
    let mut e = Engine::new();
    let r = anchor_dir(&mut e, "src");
    let file_a = e
        .tree_mut()
        .ensure_child(r, "a", ResourceRole::User)
        .expect("test live parent");
    e.tree_mut().set_kind(file_a, ResourceKind::File);
    let file_b = e
        .tree_mut()
        .ensure_child(r, "b", ResourceRole::User)
        .expect("test live parent");
    e.tree_mut().set_kind(file_b, ResourceKind::File);

    // CONTENT mask: events-complete for a `Subtree` shape, so a single Authoritative sample
    // closes the verdict floor and the heal-side probe ships a `Chains` obligation (the parked
    // chain is visible on the wire, not folded into a `WholeSubtree`).
    let now = Instant::now();
    let snap = dir_snap(&[("a", EntryKind::File, 1), ("b", EntryKind::File, 2)]);
    let req = SubAttachRequest::for_anchor(
        CompactString::new("test"),
        SubAttachAnchor::Resource(r),
        ScanConfig::builder().recursive(true).build(),
        MAX_SETTLE,
        SETTLE,
        empty_program(),
        EffectScope::SubtreeRoot,
        ClassSet::CONTENT,
        false,
    );
    let out = e.step(Input::AttachSub(req), now);
    let sid = first_attached_sub(&out).expect("attach succeeded");
    let pid = pid_of(&e, sid);
    let seed_done = seed_to_idle(&mut e, pid, &snap, now);

    // The witnessed change: an in-mask event at `a` opens the Standard burst.
    let t0 = seed_done + Duration::from_millis(10);
    let _ = e.step(
        Input::FsEvent {
            resource: file_a,
            event: FsEvent::ContentChanged,
        },
        t0,
    );

    let a_path: Arc<Path> = Arc::clone(e.tree().get(file_a).unwrap().path());
    let undischarged = || ProbeOutcome::SubtreeProven {
        snapshot: Arc::clone(&snap),
        authority: ProofAuthority::Undischarged {
            first_unread: Arc::clone(&a_path),
        },
    };

    // First (unforced) refusal: Retry → re-Batch for another window.
    let t1 = t0 + SETTLE * 2;
    let _ = drain_due(&mut e, t1);
    let corr = e.pending_probe_for(pid).expect("Verifying probe in flight");
    let _ = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid,
            correlation: corr,
            outcome: undischarged(),
        }),
        t1,
    );

    // Past the burst deadline: the settle expiry re-enters Verifying, the deadline expiry forces
    // the in-flight verify, and the forced refusal folds `Abandon` — loud terminal, evidence
    // parked, Profile rests Idle.
    let t2 = t0 + MAX_SETTLE + Duration::from_secs(1);
    let _ = drain_due(&mut e, t2);
    let corr = e
        .pending_probe_for(pid)
        .expect("deadline-forced verify in flight");
    let abandon_out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid,
            correlation: corr,
            outcome: undischarged(),
        }),
        t2,
    );
    assert!(
        abandon_out.diagnostics.iter().any(|d| matches!(
            d,
            Diagnostic::QuiescenceCeilingUnreadable { profile, .. } if *profile == pid,
        )),
        "the forced refusal abandons loudly; got {:?}",
        abandon_out.diagnostics,
    );
    assert!(
        matches!(e.profiles().get(pid).unwrap().state(), ProfileState::Idle),
        "Abandon finishes the burst to Idle",
    );

    // The obstruction heals silently. A later, unrelated in-mask event at `b` opens a fresh
    // burst; the constructor drains the parked set into its provenance.
    let t3 = t2 + Duration::from_secs(5);
    let _ = e.step(
        Input::FsEvent {
            resource: file_b,
            event: FsEvent::ContentChanged,
        },
        t3,
    );

    // The fresh burst's verify must obligate over the parked chain: dirty = {a (parked),
    // b (trigger)}, so the emitted probe's `Chains` covers `/src/a` — the walker cannot
    // mtime-skip the formerly-unreadable region.
    let t4 = t3 + SETTLE * 2;
    let mut probe_req: Option<ProbeRequest> = None;
    while let Some(entry) = e.pop_expired(t4) {
        let s = e.step(
            Input::TimerExpired {
                profile: entry.profile,
                kind: entry.kind,
                id: entry.id,
            },
            t4,
        );
        for op in s.probe_ops().iter() {
            if let ProbeOp::Probe { request } = op {
                probe_req = Some(request.clone());
            }
        }
    }
    let corr = match probe_req.expect("settle expiry drives the fresh burst's Verifying") {
        ProbeRequest::Subtree {
            correlation,
            obligation,
            ..
        } => {
            match &obligation {
                ProofObligation::Chains(chains) => {
                    assert!(
                        chains.iter().any(|p| p == &a_path),
                        "the parked chain rides the next burst's obligation; got {chains:?}",
                    );
                }
                ProofObligation::WholeSubtree => {
                    panic!("events-complete Standard burst ships Chains, not WholeSubtree")
                }
            }
            correlation
        }
        other => panic!("Dir-anchored Standard verify emits a Subtree probe; got {other:?}"),
    };

    // The fresh Authoritative read observes the change the abandoned burst witnessed (a's
    // identity differs from the stale baseline) — the originally-witnessed change fires.
    let healed = dir_snap(&[("a", EntryKind::File, 9), ("b", EntryKind::File, 2)]);
    let fire_out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid,
            correlation: corr,
            outcome: proven(healed),
        }),
        t4,
    );
    assert!(
        !fire_out.effects().is_empty(),
        "the originally-witnessed change fires once the obstruction heals",
    );
}
