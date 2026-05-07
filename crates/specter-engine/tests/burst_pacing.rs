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
use specter_core::{
    ArgPart, ArgTemplate, ChildEntry, ClassSet, CommandTemplate, DirMeta, DirSnapshot, EffectScope,
    FsEvent, Input, ProbeCorrelation, ProbeOp, ProbeRequest, ProbeResponse, ProbeResult,
    ResourceId, ResourceKind, ResourceRole, ScanConfig, StepOutput, TreeSnapshot,
};
use specter_engine::{Engine, SubAttachRequest};
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, Instant, UNIX_EPOCH};

const SETTLE: Duration = Duration::from_millis(100);
const MAX_SETTLE: Duration = Duration::from_secs(6);

fn empty_command() -> CommandTemplate {
    CommandTemplate::new([ArgTemplate::new([ArgPart::literal("/bin/true")])])
}

fn empty_dir_snap(root: ResourceId) -> TreeSnapshot {
    TreeSnapshot::Dir(Arc::new(DirSnapshot::new(
        root,
        DirMeta {
            mtime: UNIX_EPOCH,
            inode: 0,
            device: 0,
        },
        0,
        BTreeMap::<CompactString, ChildEntry>::new(),
    )))
}

fn first_probe_correlation(out: &StepOutput) -> Option<ProbeCorrelation> {
    out.probe_ops.iter().find_map(|op| match op {
        ProbeOp::Probe {
            request: ProbeRequest { correlation, .. },
        } => Some(*correlation),
        ProbeOp::Cancel { .. } => None,
    })
}

/// Drive `e`'s Seed burst to Idle: `attach_sub` already fired the Seed
/// probe; respond Ok with the supplied snapshot.
fn complete_seed(
    e: &mut Engine,
    pid: specter_core::ProfileId,
    seed_correlation: ProbeCorrelation,
    snap: TreeSnapshot,
    now: Instant,
) {
    let _ = e.step(
        Input::ProbeResponse(ProbeResponse {
            profile: pid,
            correlation: seed_correlation,
            result: ProbeResult::Ok(snap),
        }),
        now,
    );
}

#[test]
fn dense_event_storm_converges_naturally_below_burst_deadline() {
    let mut e = Engine::new();
    let r = e.tree_mut().ensure(None, "src", ResourceRole::User);
    e.tree_mut().set_kind(r, ResourceKind::Dir);

    let now = Instant::now();
    let req = SubAttachRequest {
        name: "build".into(),
        resource: r,
        path: None,
        config: ScanConfig::builder().recursive(true).build(),
        max_settle: MAX_SETTLE,
        settle: SETTLE,
        command: empty_command(),
        scope: EffectScope::SubtreeRoot,
        events: ClassSet::CONTENT,
        log_output: false,
    };
    let (sid, attach_out) = e.attach_sub(req, now);
    let pid = e.subs().get(sid).expect("sub").profile;
    let seed_correlation =
        first_probe_correlation(&attach_out).expect("Seed probe fires immediately");
    let snap = empty_dir_snap(r);
    complete_seed(&mut e, pid, seed_correlation, snap.clone(), now);

    // Storm: 8 modify events at 100 ms intervals, t0..=t0+700 ms.
    let storm_start = now + Duration::from_millis(10);
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

    // Stable response → Effect, → Idle. Effect is NOT forced (the burst
    // converged naturally;
    let resp_t = probe_emit + Duration::from_millis(1);
    let stable_out = e.step(
        Input::ProbeResponse(ProbeResponse {
            profile: pid,
            correlation: probe_correlation,
            result: ProbeResult::Ok(snap),
        }),
        resp_t,
    );
    assert_eq!(stable_out.effects.len(), 1, "stable verdict fires Effect");
    assert!(
        !stable_out.effects[0].forced,
        "post-fix burst converges before burst_deadline; \
         `forced = true` would mean the regression is back",
    );

    // The Effect must arrive well below `now + MAX_SETTLE`. We give a
    // generous margin (1.5×SETTLE) over the theoretical lower bound to
    // absorb any harness scheduling slop.
    let burst_deadline = now + MAX_SETTLE;
    let upper_bound = last_event + SETTLE + SETTLE / 2 + Duration::from_millis(1);
    assert!(
        resp_t < upper_bound,
        "Effect fired at {:?} relative to burst start; expected near \
         `last_event + settle` ({:?}), well below `burst_deadline` ({:?})",
        resp_t.duration_since(now),
        (last_event + SETTLE).duration_since(now),
        burst_deadline.duration_since(now),
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
    let r = e.tree_mut().ensure(None, "src", ResourceRole::User);
    e.tree_mut().set_kind(r, ResourceKind::Dir);

    let now = Instant::now();
    let req = SubAttachRequest {
        name: "build".into(),
        resource: r,
        path: None,
        config: ScanConfig::builder().recursive(true).build(),
        max_settle: MAX_SETTLE,
        settle: SETTLE,
        command: empty_command(),
        scope: EffectScope::SubtreeRoot,
        events: ClassSet::CONTENT,
        log_output: false,
    };
    let (sid, attach_out) = e.attach_sub(req, now);
    let pid = e.subs().get(sid).expect("sub").profile;
    let seed_correlation =
        first_probe_correlation(&attach_out).expect("Seed probe fires immediately");
    complete_seed(&mut e, pid, seed_correlation, empty_dir_snap(r), now);

    // Kick off a Standard burst with one event.
    let t_event = now + Duration::from_millis(10);
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
            ChildEntry::Leaf(specter_core::LeafEntry::new(
                specter_core::EntryKind::File,
                u64::from(cycle),
                UNIX_EPOCH,
                u64::from(cycle) + 1,
                0,
            )),
        );
        let unstable_snap = TreeSnapshot::Dir(Arc::new(DirSnapshot::new(
            r,
            DirMeta {
                mtime: UNIX_EPOCH,
                inode: 0,
                device: 0,
            },
            0,
            entries,
        )));
        let _ = e.step(
            Input::ProbeResponse(ProbeResponse {
                profile: pid,
                correlation: probe_correlation,
                result: ProbeResult::Ok(unstable_snap),
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
