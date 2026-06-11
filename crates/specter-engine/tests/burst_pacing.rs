//! Regression suite for the settle-timer pacing contract.
//!
//! These cases capture the failure mode that motivated splitting the pre-fire re-Batch into
//! `event_drives_batching` (driven by an `FsEvent`) and `retry_drives_batching` (driven by a
//! `QuiescenceVerdict::Retry` verify response): a dense storm of `FsEvent`s used to inflate the
//! settle backoff curve, eventually pegging the timer at `burst_deadline` and forcing the Effect to
//! fire `forced = true`. The current contract is "every event re-arms `now + settle`," so the burst
//! converges naturally once the storm stops.

use compact_str::CompactString;
use specter_core::testkit::{dir_snap, empty_program};
use specter_core::{
    ChildEntry, ClassSet, DirMeta, DirSnapshot, EffectScope, FsEvent, FsIdentity, Input,
    ProbeOutcome, ProbeResponse, ProfileIdentity, ProofAuthority, ResourceKind, ResourceRole,
    ScanConfig, SubAttachAnchor, SubAttachRequest, SubParams,
};
use specter_engine::Engine;
use specter_engine::testkit::{MAX_SETTLE, SETTLE, first_probe_correlation, seed_to_idle};
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, Instant, UNIX_EPOCH};

#[test]
fn dense_event_storm_converges_naturally_below_burst_deadline() {
    let mut e = Engine::new();
    let r = e.tree_mut().ensure_root("src", ResourceRole::User);
    e.tree_mut().set_kind(r, ResourceKind::Dir);

    let now = Instant::now();
    let req = SubAttachRequest {
        anchor: SubAttachAnchor::Resource(r),
        identity: ProfileIdentity::new(
            ScanConfig::builder().recursive(true).build(),
            MAX_SETTLE,
            ClassSet::CONTENT,
        ),
        params: SubParams {
            name: "build".into(),
            program: empty_program(),
            scope: EffectScope::SubtreeRoot,
            settle: SETTLE,
            log_output: false,
            template: None,
            source_discovery: None,
        },
    };
    let attach_out = e.step(Input::AttachSub(req), now);
    let sid = specter_core::testkit::first_attached_sub(&attach_out).expect("attach_sub succeeded");
    let pid = e.subs().get(sid).expect("sub").profile();
    assert!(
        first_probe_correlation(&attach_out).is_some(),
        "cold-arm Seed: probe emitted at burst construction",
    );
    let snap = dir_snap(&[]);
    // Drive the cold-arm Seed burst to Idle; the Standard burst under test starts strictly after
    // the Seed's settle window to keep instants monotonic.
    let seed_settled = seed_to_idle(&mut e, pid, &snap, now);

    // Storm: 8 modify events at 100 ms intervals, rebased past the Seed-establishment window so the
    // Profile is Idle when it begins.
    let storm_start = seed_settled + Duration::from_millis(10);
    let storm_step = Duration::from_millis(100);
    let storm_count = 8;
    for k in 0..storm_count {
        let t = storm_start + storm_step * k;
        let _ = e.step(
            Input::FsEvent {
                resource: r,
                event: FsEvent::ContentChanged,
            },
            t,
        );
    }
    let last_event = storm_start + storm_step * (storm_count - 1);

    // Each event re-armed the settle timer to `last_event + settle`. Drain timers from there
    // onward; the next probe must fire close to `last_event + SETTLE` (well below `burst_deadline =
    // now + MAX_SETTLE`).
    let probe_emit = last_event + SETTLE;
    let probe_correlation = loop {
        let Some(entry) = e.pop_expired(probe_emit) else {
            panic!(
                "settle timer did not fire within `last_event + settle`; \
                 the conflation regressed: events should re-arm the debounce."
            )
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

    // The verify response folds to `Stable(StableReason::Natural)` on the first sample — single
    // dispatch fires the Effect.
    let resp_t = probe_emit + Duration::from_millis(1);
    let stable_out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid,
            correlation: probe_correlation,
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
        "Authoritative stable verdict fires one Effect",
    );
    assert!(
        !stable_out.effects()[0].forced,
        "post-fix burst converges before burst_deadline; \
         `forced = true` would mean the regression is back",
    );

    // The Effect must arrive well below the Standard burst's `burst_deadline`. The Seed runs first
    // (its own burst, single settle window); the Standard burst under test starts at its first
    // storm event (`storm_start`), so its deadline is `storm_start + MAX_SETTLE`. Natural
    // convergence costs one settle cycle; the bound tracks `last_event + SETTLE` with a 0.5×SETTLE
    // slop margin, well below the forced `burst_deadline`.
    let burst_deadline = storm_start + MAX_SETTLE;
    let upper_bound = last_event + SETTLE + SETTLE / 2 + Duration::from_millis(2);
    assert!(
        resp_t < upper_bound,
        "Effect fired at {:?} relative to the Standard burst start; \
         expected near `last_event + settle` ({:?}), well below \
         `burst_deadline` ({:?})",
        resp_t.duration_since(storm_start),
        (last_event + SETTLE).duration_since(storm_start),
        burst_deadline.duration_since(storm_start),
    );
}

#[test]
fn sustained_undischarged_response_storm_paces_at_settle() {
    // Verify the second leg of the conflation fix: the `QuiescenceVerdict::Retry` path (here driven
    // by walker-refused `Undischarged !terminal` authority responses) routes via
    // `retry_drives_batching`, which schedules the next attempt at `now + settle`, not at the
    // exponential-backoff curve. We reproduce a sustained-undischarged burst: every probe response
    // refuses the obligation; no events arrive in between. Each cycle's next-probe deadline must
    // equal `last_response + settle`, regardless of how many cycles preceded.
    let mut e = Engine::new();
    let r = e.tree_mut().ensure_root("src", ResourceRole::User);
    e.tree_mut().set_kind(r, ResourceKind::Dir);

    let now = Instant::now();
    let req = SubAttachRequest {
        anchor: SubAttachAnchor::Resource(r),
        identity: ProfileIdentity::new(
            ScanConfig::builder().recursive(true).build(),
            MAX_SETTLE,
            ClassSet::CONTENT,
        ),
        params: SubParams {
            name: "build".into(),
            program: empty_program(),
            scope: EffectScope::SubtreeRoot,
            settle: SETTLE,
            log_output: false,
            template: None,
            source_discovery: None,
        },
    };
    let attach_out = e.step(Input::AttachSub(req), now);
    let sid = specter_core::testkit::first_attached_sub(&attach_out).expect("attach_sub succeeded");
    let pid = e.subs().get(sid).expect("sub").profile();
    assert!(
        first_probe_correlation(&attach_out).is_some(),
        "cold-arm Seed: probe emitted at burst construction",
    );
    // Drive the cold-arm Seed burst to Idle; the Standard burst under test starts strictly after
    // the Seed's settle window to keep instants monotonic.
    let seed_settled = seed_to_idle(&mut e, pid, &dir_snap(&[]), now);

    // Kick off a Standard burst with one event, rebased past the Seed-establishment window.
    let t_event = seed_settled + Duration::from_millis(10);
    let _ = e.step(
        Input::FsEvent {
            resource: r,
            event: FsEvent::ContentChanged,
        },
        t_event,
    );

    // Three consecutive `ProofAuthority::Undischarged` !terminal probe responses (each folding to
    // `QuiescenceVerdict::Retry`). After each, the next probe should fire at `last_response +
    // SETTLE`, not amplified.
    let mut response_at = t_event + SETTLE;
    let unread: std::sync::Arc<std::path::Path> =
        std::sync::Arc::from(std::path::Path::new("src/opaque"));
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

        // Reply with an Undischarged !terminal authority. This folds to `QuiescenceVerdict::Retry`,
        // which routes via `retry_drives_batching` — the surviving retry path that schedules the
        // next attempt at `now + settle`.
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
        let degraded_snap = Arc::new(DirSnapshot::new(
            DirMeta::synthetic(UNIX_EPOCH, FsIdentity::synthetic(0, 0)),
            0,
            entries,
        ));
        let _ = e.step(
            Input::ProbeResponse(ProbeResponse {
                owner: pid,
                correlation: probe_correlation,
                outcome: ProbeOutcome::SubtreeProven {
                    snapshot: degraded_snap,
                    authority: ProofAuthority::Undischarged {
                        first_unread: std::sync::Arc::clone(&unread),
                    },
                },
            }),
            response_at,
        );

        let probe_due = response_at + SETTLE;
        assert!(
            e.next_deadline().is_some_and(|d| d <= probe_due),
            "after undischarged response cycle {cycle}, next deadline must be at most \
             last_response + settle (no exponential amplification)",
        );

        response_at = probe_due;
    }
}
