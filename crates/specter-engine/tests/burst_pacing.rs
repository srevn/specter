//! Regression suite for the settle-timer pacing contract.
//!
//! These cases capture the failure mode that motivated splitting
//! `transition_to_settling` into `event_drives_batching` and
//! `unstable_response_drives_batching`: a dense storm of `FsEvent`s used
//! to inflate the settle backoff curve, eventually pegging the timer at
//! `burst_deadline` and forcing the Effect to fire `forced = true`. The
//! current contract is "every event re-arms `now + settle`," so the
//! burst converges naturally once the storm stops.

#![allow(
    clippy::manual_let_else,
    clippy::match_wildcard_for_single_variants,
    clippy::missing_const_for_fn,
    clippy::needless_pass_by_value,
    clippy::too_many_lines
)]

use compact_str::CompactString;
use specter_core::testkit::single_exec_program;
use specter_core::{
    ActionProgram, ArgPart, ArgTemplate, ChildEntry, ClassSet, DirMeta, DirSnapshot, EffectScope,
    FsEvent, FsIdentity, Input, ProbeCorrelation, ProbeOp, ProbeOutcome, ProbeOwner, ProbeResponse,
    ProfileIdentity, ProofAuthority, ResourceKind, ResourceRole, ScanConfig, StepOutput,
    SubAttachAnchor, SubAttachRequest, SubParams,
};
use specter_engine::Engine;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, Instant, UNIX_EPOCH};

const SETTLE: Duration = Duration::from_millis(100);
const MAX_SETTLE: Duration = Duration::from_secs(6);

fn empty_program() -> Arc<ActionProgram> {
    single_exec_program([ArgTemplate::new([ArgPart::literal("/bin/true")])])
}

fn empty_dir_snap() -> Arc<DirSnapshot> {
    Arc::new(DirSnapshot::new(
        DirMeta::synthetic(UNIX_EPOCH, FsIdentity::synthetic(0, 0)),
        0,
        BTreeMap::<CompactString, ChildEntry>::new(),
    ))
}

fn first_probe_correlation(out: &StepOutput) -> Option<ProbeCorrelation> {
    out.probe_ops().iter().find_map(|op| match op {
        ProbeOp::Probe { request } => Some(request.correlation()),
        ProbeOp::Cancel { .. } => None,
    })
}

/// Drive `e`'s Batching-first Seed burst to Idle and return the
/// instant the Seed settled at (callers rebase later timelines past
/// it to keep instants strictly monotonic).
///
/// A Seed runs the same N=2 settle-spaced quiescence proof as
/// a Standard burst: `start_seed_burst` lands in
/// `Active(PreFire(Batching))` and emits **no** probe at attach. Each
/// of the two settle windows (`t0+SETTLE`, `t0+SETTLE*2`) expires the
/// `Settle` timer (`Batching → Verifying`), emits one Seed probe, and
/// is answered hash-equal: the first sample is `Unstable` by
/// construction (`certified` starts `None`) and re-batches;
/// the second is hash-equal ⇒ `Stable` ⇒ `seed_pin_body` commits,
/// rebases the baseline, and finishes the burst to `Idle`. Both
/// responses share `snap`, so the proof converges cleanly within
/// `max_settle` (never forced).
fn complete_seed_burst(
    e: &mut Engine,
    pid: specter_core::ProfileId,
    snap: Arc<DirSnapshot>,
    t0: Instant,
) -> Instant {
    let mut settled_at = t0;
    for at in [t0 + SETTLE, t0 + SETTLE * 2] {
        while let Some(entry) = e.pop_expired(at) {
            let _ = e.step(
                Input::TimerExpired {
                    profile: entry.profile,
                    kind: entry.kind,
                    id: entry.id,
                },
                at,
            );
        }
        let c = e
            .pending_probe_for(ProbeOwner::Profile(pid))
            .expect("Seed Verifying probe after settle expiry");
        let _ = e.step(
            Input::ProbeResponse(ProbeResponse {
                owner: ProbeOwner::Profile(pid),
                correlation: c,
                outcome: ProbeOutcome::SubtreeProven {
                    snapshot: Arc::clone(&snap),
                    authority: ProofAuthority::Authoritative,
                },
            }),
            at,
        );
        settled_at = at;
    }
    assert!(
        matches!(
            e.profiles().get(pid).expect("Profile alive").state(),
            specter_core::ProfileState::Idle
        ),
        "Seed burst returns to Idle after the N=2 quiescence proof",
    );
    settled_at
}

#[test]
fn dense_event_storm_converges_naturally_below_burst_deadline() {
    let mut e = Engine::new();
    let r = e.tree_mut().ensure_root("src", ResourceRole::User);
    e.tree_mut().set_kind(r, ResourceKind::Dir);

    let now = Instant::now();
    let req = SubAttachRequest {
        anchor: SubAttachAnchor::Resource(r),
        identity: ProfileIdentity {
            config: ScanConfig::builder().recursive(true).build(),
            max_settle: MAX_SETTLE,
            events: ClassSet::CONTENT,
        },
        params: SubParams {
            name: "build".into(),
            program: empty_program(),
            scope: EffectScope::SubtreeRoot,
            settle: SETTLE,
            log_output: false,
            source_promoter: None,
        },
    };
    let attach_out = e.step(Input::AttachSub(req), now);
    let sid = specter_core::testkit::first_attached_sub(&attach_out).expect("attach_sub succeeded");
    let pid = e.subs().get(sid).expect("sub").profile;
    assert!(
        first_probe_correlation(&attach_out).is_none(),
        "Batching-first Seed emits no probe at attach (no probe at attach)",
    );
    let snap = empty_dir_snap();
    // Drive the N=2 Batching-first Seed burst to Idle; the Standard
    // burst under test starts strictly after the Seed's two settle
    // windows (`now + SETTLE*2`) to keep instants monotonic.
    let seed_settled = complete_seed_burst(&mut e, pid, snap.clone(), now);

    // Storm: 8 modify events at 100 ms intervals, rebased past the
    // Seed-establishment window so the Profile is Idle when it begins.
    let storm_start = seed_settled + Duration::from_millis(10);
    let storm_step = Duration::from_millis(100);
    let storm_count = 8;
    for k in 0..storm_count {
        let t = storm_start + storm_step * k;
        let _ = e.step(
            Input::FsEvent {
                resource: r,
                event: FsEvent::Modified,
            },
            t,
        );
    }
    let last_event = storm_start + storm_step * (storm_count - 1);

    // Each event re-armed the settle timer to `last_event + settle`.
    // Drain timers from there onward; the next probe must fire close to
    // `last_event + SETTLE` (well below `burst_deadline = now + MAX_SETTLE`).
    let probe_emit = last_event + SETTLE;
    let probe_correlation = loop {
        let entry = match e.pop_expired(probe_emit) {
            Some(entry) => entry,
            None => panic!(
                "settle timer did not fire within `last_event + settle`; \
                 the conflation regressed: events should re-arm the debounce."
            ),
        };
        let out = e.step(
            Input::TimerExpired {
                profile: entry.profile,
                kind: entry.kind,
                id: entry.id,
            },
            probe_emit,
        );
        if let Some(c) = first_probe_correlation(&out) {
            break c;
        }
    };

    // N=2 quiescence: the prime sample (prior == None ⇒ Unstable)
    // re-arms the debounce at `prime_t + SETTLE`; it must not fire.
    let prime_t = probe_emit + Duration::from_millis(1);
    let primed = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
            correlation: probe_correlation,
            outcome: ProbeOutcome::SubtreeProven {
                snapshot: snap.clone(),
                authority: ProofAuthority::Authoritative,
            },
        }),
        prime_t,
    );
    assert!(
        primed.effects().is_empty(),
        "prime sample (prior == None ⇒ Unstable) must not fire",
    );

    // Drain the re-armed debounce to the confirm Verifying probe. The
    // hash-equal confirm sample is the Stable verdict. The Effect is
    // NOT forced — the burst converged naturally over the N=2
    // quiescence (prime + confirm settle cycles), nowhere near the
    // exponential-backoff regression or the forced burst_deadline.
    let confirm_emit = prime_t + SETTLE;
    let confirm_correlation = loop {
        let entry = match e.pop_expired(confirm_emit) {
            Some(entry) => entry,
            None => panic!(
                "re-armed settle timer did not fire within `prime + settle`; \
                 the N=2 quiescence re-batch regressed."
            ),
        };
        let out = e.step(
            Input::TimerExpired {
                profile: entry.profile,
                kind: entry.kind,
                id: entry.id,
            },
            confirm_emit,
        );
        if let Some(c) = first_probe_correlation(&out) {
            break c;
        }
    };
    let resp_t = confirm_emit + Duration::from_millis(1);
    let stable_out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
            correlation: confirm_correlation,
            outcome: ProbeOutcome::SubtreeProven {
                snapshot: snap,
                authority: ProofAuthority::Authoritative,
            },
        }),
        resp_t,
    );
    assert_eq!(
        stable_out.effects().len(),
        1,
        "quiescence-confirmed stable verdict fires Effect",
    );
    assert!(
        !stable_out.effects()[0].forced,
        "post-fix burst converges before burst_deadline; \
         `forced = true` would mean the regression is back",
    );

    // The Effect must arrive well below the Standard burst's
    // `burst_deadline`. The Seed runs first (its own burst,
    // `now..now+SETTLE*2`); the Standard burst under test starts at
    // its first storm event (`storm_start`), so its deadline is
    // `storm_start + MAX_SETTLE`. Natural N=2 convergence costs two
    // settle cycles (prime + confirm); the bound tracks
    // `last_event + 2·SETTLE` with a 0.5×SETTLE slop margin, still
    // far below the forced `burst_deadline`.
    let burst_deadline = storm_start + MAX_SETTLE;
    let upper_bound = last_event + SETTLE * 2 + SETTLE / 2 + Duration::from_millis(2);
    assert!(
        resp_t < upper_bound,
        "Effect fired at {:?} relative to the Standard burst start; \
         expected near `last_event + 2·settle` ({:?}), well below \
         `burst_deadline` ({:?})",
        resp_t.duration_since(storm_start),
        (last_event + SETTLE * 2).duration_since(storm_start),
        burst_deadline.duration_since(storm_start),
    );
}

#[test]
fn sustained_unstable_response_storm_paces_at_settle() {
    // Verify the second leg of the conflation fix: the probe-unstable
    // path schedules the next attempt at `now + settle`, not at the
    // exponential-backoff curve. We reproduce a sustained-instability
    // burst: every probe response shows a different snapshot, no events
    // arrive in between. Each cycle's next-probe deadline must equal
    // `last_response + settle`, regardless of how many cycles preceded.
    let mut e = Engine::new();
    let r = e.tree_mut().ensure_root("src", ResourceRole::User);
    e.tree_mut().set_kind(r, ResourceKind::Dir);

    let now = Instant::now();
    let req = SubAttachRequest {
        anchor: SubAttachAnchor::Resource(r),
        identity: ProfileIdentity {
            config: ScanConfig::builder().recursive(true).build(),
            max_settle: MAX_SETTLE,
            events: ClassSet::CONTENT,
        },
        params: SubParams {
            name: "build".into(),
            program: empty_program(),
            scope: EffectScope::SubtreeRoot,
            settle: SETTLE,
            log_output: false,
            source_promoter: None,
        },
    };
    let attach_out = e.step(Input::AttachSub(req), now);
    let sid = specter_core::testkit::first_attached_sub(&attach_out).expect("attach_sub succeeded");
    let pid = e.subs().get(sid).expect("sub").profile;
    assert!(
        first_probe_correlation(&attach_out).is_none(),
        "Batching-first Seed emits no probe at attach (no probe at attach)",
    );
    // Drive the N=2 Batching-first Seed burst to Idle; the Standard
    // burst under test starts strictly after the Seed's two settle
    // windows to keep instants monotonic.
    let seed_settled = complete_seed_burst(&mut e, pid, empty_dir_snap(), now);

    // Kick off a Standard burst with one event, rebased past the
    // Seed-establishment window.
    let t_event = seed_settled + Duration::from_millis(10);
    let _ = e.step(
        Input::FsEvent {
            resource: r,
            event: FsEvent::Modified,
        },
        t_event,
    );

    // Three consecutive unstable probe responses. After each, the next
    // probe should fire at `last_response + SETTLE`, not amplified.
    let mut response_at = t_event + SETTLE;
    for cycle in 0u32..3 {
        // Drain settle timer → Verifying.
        let probe_correlation = loop {
            let entry = e.pop_expired(response_at).expect("settle timer pending");
            let out = e.step(
                Input::TimerExpired {
                    profile: entry.profile,
                    kind: entry.kind,
                    id: entry.id,
                },
                response_at,
            );
            if let Some(c) = first_probe_correlation(&out) {
                break c;
            }
        };

        // Reply with a fresh snapshot whose hash differs from `current`.
        // We construct a snapshot with a unique inode so the hash diverges.
        let mut entries = BTreeMap::<CompactString, ChildEntry>::new();
        entries.insert(
            CompactString::new("file"),
            ChildEntry::Leaf(specter_core::LeafEntry::synthetic(
                specter_core::EntryKind::File,
                u64::from(cycle),
                UNIX_EPOCH,
                FsIdentity::synthetic(u64::from(cycle) + 1, 0),
            )),
        );
        let unstable_snap = Arc::new(DirSnapshot::new(
            DirMeta::synthetic(UNIX_EPOCH, FsIdentity::synthetic(0, 0)),
            0,
            entries,
        ));
        let _ = e.step(
            Input::ProbeResponse(ProbeResponse {
                owner: ProbeOwner::Profile(pid),
                correlation: probe_correlation,
                outcome: ProbeOutcome::SubtreeProven {
                    snapshot: unstable_snap,
                    authority: ProofAuthority::Authoritative,
                },
            }),
            response_at,
        );

        let probe_due = response_at + SETTLE;
        assert!(
            e.next_deadline().is_some_and(|d| d <= probe_due),
            "after unstable response cycle {cycle}, next deadline must be at most \
             last_response + settle (no exponential amplification)",
        );

        response_at = probe_due;
    }
}
