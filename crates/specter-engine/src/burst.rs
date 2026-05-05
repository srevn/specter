//! Burst lifecycle helpers.
//!
//! Each helper is the **single source** of one transition kind — a phase
//! transition body, a Burst construction, or a return-to-Idle. Centralizing
//! the timer scheduling, refcount edges, and Burst-struct mutations here
//! prevents drift between the transition-row handlers and the
//! post-`EffectComplete` re-probe path.
//!
//! - `start_seed_burst` / `start_standard_burst` — Idle → Active.
//! - `event_drives_batching` (FsEvent during Active) /
//!   `unstable_response_drives_batching` (probe-unstable response) /
//!   `transition_to_verifying` (settle-timer expiry, burst-deadline,
//!   Draining → Verifying reconfirm) /
//!   `transition_to_draining` — Active → Active phase swaps.
//! - `finish_burst_to_idle` — Active → Idle, single point of `-suppress` and
//!   `propagate(-1)`.
//!
//! The two batching helpers exist as a deliberate split rather than one
//! helper with a runtime flag: each caller has **static knowledge** of
//! whether a probe is in flight (only `event_drives_batching` may need to
//! emit `ProbeOp::Cancel`). Encoding that knowledge as helper identity
//! makes a stray Cancel on the just-responded path structurally
//! impossible.
//!
//! Probe emission flows through `probe_channel::emit_probe_op` (single
//! source for `ProbeOp::Probe` construction); `start_seed_burst` and
//! `transition_to_verifying` build a `ProbeEmissionParams::Burst` and
//! route through it.

use crate::Engine;
use crate::refcounts::{add_suppress, sub_suppress};
use smallvec::SmallVec;
use specter_core::{
    Burst, BurstIntent, BurstPhase, Profile, ProfileId, ProfileState, ResourceId, ResourceKind,
    StepOutput, TimerKind, Tree, TreeSnapshot,
};
use std::collections::BTreeSet;
use std::path::PathBuf;
use std::time::Instant;

impl Engine {
    /// Start a Seed burst: no settle wait, immediate Probe.
    /// Caller has verified `Profile.state == Idle`. Constructs the Burst,
    /// schedules `burst_deadline`, mints the probe correlation, emits Probe
    /// (with the Profile's whole `current` as `baseline_subtree` when
    /// post-Effect Seed has one, enabling mtime-skip), and `+suppress` on
    /// the anchor.
    ///
    /// Used in two places: `attach_sub` (fresh Profile baseline) and
    /// `EffectComplete::Ok` while Idle (post-Effect rebase). Same machinery.
    pub(crate) fn start_seed_burst(
        &mut self,
        profile_id: ProfileId,
        now: Instant,
        out: &mut StepOutput,
    ) {
        let Some(p) = self.profiles.get(profile_id) else {
            return;
        };
        debug_assert!(
            matches!(p.state, ProfileState::Idle),
            "start_seed_burst: Profile must be Idle on entry",
        );
        let resource = p.resource;
        let max_settle = p.max_settle;
        // Seed targets the anchor; baseline_subtree is current.subtree_at(anchor)
        // for post-Effect Seeds (gives the walker mtime-skip for noop Effects)
        // and None for fresh-Profile / recovery Seeds (no prior observation).
        let baseline_subtree = p
            .current
            .as_ref()
            .and_then(|s| s.subtree_at(resource, &self.tree));

        let burst_deadline =
            self.timers
                .schedule(now + max_settle, profile_id, TimerKind::BurstDeadline);
        let Some(correlation) = self.mint_probe_correlation(profile_id) else {
            return;
        };

        if let Some(p) = self.profiles.get_mut(profile_id) {
            p.state = ProfileState::Active(Burst {
                burst_deadline,
                phase: BurstPhase::Verifying { correlation },
                intent: BurstIntent::Seed,
                forced: false,
                dirty_resources: BTreeSet::new(),
                force_walk_resources: BTreeSet::new(),
                probe_target: Some(resource),
            });
        }

        add_suppress(&mut self.tree, resource, out);
        self.emit_probe_op(
            profile_id,
            resource,
            correlation,
            crate::probe_channel::ProbeEmissionParams::Burst {
                baseline_subtree,
                force_walk: BTreeSet::new(),
                forced: false,
            },
            out,
        );
    }

    /// Start a Standard burst: schedule settle + `burst_deadline`,
    /// `+suppress`, propagate(+1). No Probe — that fires on `settle_timer`
    /// expiry via `transition_to_verifying`.
    ///
    /// `event_resource` is the `FsEvent`'s source. It seeds both
    /// `dirty_resources` (basis for the next probe's LCA) and
    /// `force_walk_resources` (defeats mtime-skip on event-dirty paths).
    pub(crate) fn start_standard_burst(
        &mut self,
        profile_id: ProfileId,
        event_resource: ResourceId,
        now: Instant,
        out: &mut StepOutput,
    ) {
        let Some(p) = self.profiles.get(profile_id) else {
            return;
        };
        debug_assert!(
            matches!(p.state, ProfileState::Idle),
            "start_standard_burst: Profile must be Idle on entry",
        );
        let resource = p.resource;
        let settle = p.settle;
        let max_settle = p.max_settle;

        let settle_timer = self
            .timers
            .schedule(now + settle, profile_id, TimerKind::Settle);
        let burst_deadline =
            self.timers
                .schedule(now + max_settle, profile_id, TimerKind::BurstDeadline);

        let mut dirty = BTreeSet::new();
        dirty.insert(event_resource);
        let mut force_walk = BTreeSet::new();
        force_walk.insert(event_resource);

        if let Some(p) = self.profiles.get_mut(profile_id) {
            p.state = ProfileState::Active(Burst {
                burst_deadline,
                phase: BurstPhase::Batching { settle_timer },
                intent: BurstIntent::Standard,
                forced: false,
                dirty_resources: dirty,
                force_walk_resources: force_walk,
                probe_target: None,
            });
        }

        add_suppress(&mut self.tree, resource, out);
        let _ = self.stability.propagate(&mut self.profiles, profile_id, 1);
    }

    /// Caller: `drive_burst` Active branch — an `FsEvent` arrived during a
    /// burst. Cancels any in-flight verify (iff the prior phase was
    /// `Verifying`), accumulates the event into `dirty_resources` and
    /// `force_walk_resources`, arms a fresh settle timer, and writes
    /// `phase = Batching { settle_timer }`. `intent`, `forced`, and the
    /// `burst_deadline` are preserved.
    ///
    /// Why this is one of two batching mutators rather than a single
    /// helper with a flag: the caller has static knowledge that the
    /// engine has not just received a probe response. If the prior phase
    /// was `Verifying`, a verify is in flight and we must Cancel it. If
    /// the prior phase was `Batching` or `Draining`, no probe is in
    /// flight. Encoding that as a runtime flag is a category error — the
    /// caller always knows the right answer.
    pub(crate) fn event_drives_batching(
        &mut self,
        profile_id: ProfileId,
        event_resource: ResourceId,
        now: Instant,
        out: &mut StepOutput,
    ) {
        let Some(p) = self.profiles.get(profile_id) else {
            return;
        };
        let ProfileState::Active(_) = &p.state else {
            return;
        };
        let settle = p.settle;

        // Idempotent: emits Cancel iff the probe channel is open
        // (Verifying ⇒ pending_probe = Some(_)). For Batching / Draining
        // entries, no probe is in flight and the helper is a no-op —
        // matching the prior `was_verifying` snapshot's role.
        self.cancel_pending_probe(profile_id, out);

        let settle_timer = self
            .timers
            .schedule(now + settle, profile_id, TimerKind::Settle);

        if let Some(p) = self.profiles.get_mut(profile_id)
            && let ProfileState::Active(burst) = &mut p.state
        {
            burst.dirty_resources.insert(event_resource);
            burst.force_walk_resources.insert(event_resource);
            burst.phase = BurstPhase::Batching { settle_timer };
        }
    }

    /// Caller: `dispatch_standard_ok` not-stable + not-forced — a verify
    /// just responded with an unstable verdict. By construction no probe
    /// is in flight (the caller is in the response handler), so this
    /// helper structurally cannot emit `ProbeOp::Cancel`. Arms a fresh
    /// settle timer and writes `phase = Batching { settle_timer }`.
    ///
    /// `dirty_resources` is intentionally preserved so the next verify
    /// re-targets the same LCA; `force_walk_resources` is already empty
    /// (cleared at the prior `transition_to_verifying`, and no `FsEvent`
    /// can have arrived during `Verifying` without first routing through
    /// `event_drives_batching`).
    pub(crate) fn unstable_response_drives_batching(
        &mut self,
        profile_id: ProfileId,
        now: Instant,
    ) {
        let Some(settle) = self.profiles.get(profile_id).map(|p| p.settle) else {
            return;
        };
        let settle_timer = self
            .timers
            .schedule(now + settle, profile_id, TimerKind::Settle);

        if let Some(p) = self.profiles.get_mut(profile_id)
            && let ProfileState::Active(burst) = &mut p.state
        {
            burst.phase = BurstPhase::Batching { settle_timer };
        }
    }

    /// Phase: `Batching` (or `Draining`) → `Verifying`. Mints a fresh
    /// correlation; emits `ProbeOp::Probe`. The just-fired `settle_timer`
    /// is no longer referenced (lazy invalidation drops the heap entry on
    /// `pop_expired` once the prior `Batching` variant — and with it the
    /// `TimerId` — is overwritten by the assignment below).
    ///
    /// Standard probes target the LCA of the burst's `dirty_resources`,
    /// ship `current.subtree_at(target)` as the walker's mtime-skip
    /// baseline, ship `force_walk_resources` (rendered to paths) so the
    /// walker re-walks paths whose kqueue actually fired since the last
    /// probe, and propagate `Burst.forced` so the walker bypasses
    /// mtime-skip on a force-fire (max-settle deadline elapsed). Seed
    /// probes target the anchor; the Draining → Verifying reconfirm
    /// reuses `Burst.probe_target` (`dirty_resources` is empty by then so
    /// LCA would degenerate to anchor and lose the correct subtree).
    /// `force_walk_resources` is consumed by this emission; new events
    /// accumulate into the cleared set.
    pub(crate) fn transition_to_verifying(&mut self, profile_id: ProfileId, out: &mut StepOutput) {
        let Some(p) = self.profiles.get(profile_id) else {
            return;
        };
        let ProfileState::Active(burst) = &p.state else {
            return;
        };

        let intent = burst.intent;
        let phase = phase_kind(&burst.phase);
        let prior_target = burst.probe_target;
        let dirty_for_lca = burst.dirty_resources.clone();
        let force_set = burst.force_walk_resources.clone();
        let forced = burst.forced;

        // Decide target.
        let target = match (intent, phase) {
            (BurstIntent::Seed, _) => p.resource,
            (BurstIntent::Standard, PhaseKind::Draining) => {
                // Reconfirm probe — re-use the previous target. dirty_resources
                // is empty in Draining, so LCA would degenerate to anchor and
                // lose the correct subtree.
                prior_target.unwrap_or(p.resource)
            }
            (BurstIntent::Standard, _) => lca_target(p, &dirty_for_lca, &self.tree),
        };

        // baseline_subtree (always at `target`, never anchor for Standard).
        let baseline_subtree = p
            .current
            .as_ref()
            .and_then(|s| s.subtree_at(target, &self.tree));
        // force_walk paths (filtered to subtree(target); engine-side close).
        let force_walk_paths = build_force_walk(&force_set, target, &self.tree);

        let Some(correlation) = self.mint_probe_correlation(profile_id) else {
            return;
        };
        if let Some(p) = self.profiles.get_mut(profile_id)
            && let ProfileState::Active(b) = &mut p.state
        {
            b.phase = BurstPhase::Verifying { correlation };
            b.probe_target = Some(target);
            b.force_walk_resources.clear();
        }

        self.emit_probe_op(
            profile_id,
            target,
            correlation,
            crate::probe_channel::ProbeEmissionParams::Burst {
                baseline_subtree,
                force_walk: force_walk_paths,
                forced,
            },
            out,
        );
    }

    /// Phase: `Probing` → `Draining`. Phase swap only — the exit body
    /// (`Draining → Probing` reconfirm) is driven by `finish_burst_to_idle`
    /// when a child Profile's `propagate(-1)` returns this Profile in its
    /// hit-zero list.
    ///
    /// `Draining` is a unit variant: the stable snapshot lives on
    /// `Profile.current` (set by `dispatch_standard_ok` immediately
    /// before this call), so no `Arc<TreeSnapshot>` is duplicated on the
    /// phase variant.
    pub(crate) fn transition_to_draining(&mut self, profile_id: ProfileId) {
        let Some(p) = self.profiles.get_mut(profile_id) else {
            return;
        };
        if let ProfileState::Active(burst) = &mut p.state {
            burst.phase = BurstPhase::Draining;
        }
    }

    /// Active → Idle. Single source of `-suppress` and `propagate(-1)`.
    /// The active burst's timers are not explicitly cancelled — lazy
    /// invalidation in `pop_expired` drops them when they fire.
    /// Idempotent: silent no-op on already-Idle Profiles.
    ///
    /// **Draining-exit driver.** `propagate(-1)` returns ancestors whose
    /// `dirty_descendants` just hit zero AND are in `BurstPhase::Draining`.
    /// The Engine drives each through `transition_to_verifying` in the
    /// same step — the reconfirm probe compares against the Profile's
    /// `current` (set when `dispatch_standard_ok` entered Draining).
    /// Same-step ordering means the `StepOutput` reflects the cascade:
    /// child's burst end → parent reconfirm Probe in one `step` call.
    ///
    /// **Reap-pending.** If the Profile's `reap_pending` flag is set (its
    /// last Sub was detached mid-burst), `Engine::reap_profile` runs in the
    /// same step after `propagate(-1)` to release watch contributions,
    /// parent edges, and Tree slot.
    pub(crate) fn finish_burst_to_idle(&mut self, profile_id: ProfileId, out: &mut StepOutput) {
        let Some(p) = self.profiles.get(profile_id) else {
            return;
        };
        // Tightened from `!matches!(state, Idle)` to `Active(_)`: the
        // burst-end machinery (`sub_suppress`, `propagate(-1)`) is
        // Active-specific. Pending Profiles never bumped the anchor's
        // suppress_count or the ancestor `dirty_descendants`; running
        // this on Pending would underflow `sub_suppress`. The only
        // documented caller-side guard that this defends against is
        // `finalize_anchor_lost` if a future change relaxes its Pending
        // early-return.
        let was_active = matches!(p.state, ProfileState::Active(_));
        let resource = p.resource;
        if !was_active {
            return;
        }

        // Capture the burst's intent before transitioning to Idle. Only
        // Standard bursts call `propagate(+1)` at start (the burst-
        // propagation row), so only Standard bursts should call
        // `propagate(-1)` at end. Seed bursts skip propagation entirely
        // — they never contribute to ancestor `dirty_descendants`.
        let was_standard = matches!(
            &p.state,
            ProfileState::Active(burst) if matches!(burst.intent, BurstIntent::Standard),
        );

        if let Some(p) = self.profiles.get_mut(profile_id) {
            p.state = ProfileState::Idle;
        }

        sub_suppress(&mut self.tree, resource, out);

        if was_standard {
            let hit_zero = self.stability.propagate(&mut self.profiles, profile_id, -1);

            // Draining → Verifying reconfirm for ancestors whose count
            // just hit zero. `transition_to_verifying` mints a fresh
            // correlation and emits Probe; the response routes through
            // `dispatch_standard_ok` as a normal Standard burst.
            for ancestor in hit_zero {
                self.transition_to_verifying(ancestor, out);
            }
        }

        // Reap-pending check. The flag is set by `detach_sub` when the
        // Profile was Active and lost its last Sub; we defer the reap to
        // here so the Profile's burst doesn't fire Effects against a Sub
        // registry that no longer holds the reference.
        let reap_now = self
            .profiles
            .get(profile_id)
            .is_some_and(|p| p.reap_pending);
        if reap_now {
            self.reap_profile(profile_id, out);
        }
    }

}

/// Copy projection of `BurstPhase` for `transition_to_verifying`'s
/// `(intent, phase)` match — neither `Batching`'s `TimerId` nor
/// `Verifying`'s `ProbeCorrelation` is needed at the dispatch site.
/// Mirrors the private decl in `transitions.rs::on_timer_expired`; both
/// kept locally to keep the match-site code inline-readable.
#[derive(Copy, Clone, Eq, PartialEq)]
enum PhaseKind {
    Batching,
    Verifying,
    Draining,
}

const fn phase_kind(p: &BurstPhase) -> PhaseKind {
    match p {
        BurstPhase::Batching { .. } => PhaseKind::Batching,
        BurstPhase::Verifying { .. } => PhaseKind::Verifying,
        BurstPhase::Draining => PhaseKind::Draining,
    }
}

// `TreeSnapshot` reachable for downstream consumers via the burst module
// surface — the lifecycle helpers thread `current.subtree_at` references
// through that type.
const _: fn() = || {
    let _ = std::mem::size_of::<TreeSnapshot>();
};

/// "Lowest covering ancestor of all event-dirty Resources."
/// The single probe target per Standard burst.
///
/// Invariants:
/// - Returns a live `ResourceId` (always — defaults to `profile.resource`).
/// - Result is ALWAYS `ResourceKind::Dir` (Files / Unknown promoted to
///   their parent Dir; probes target Dirs because Files are observed as
///   child entries of their parent).
/// - Result is at-or-above every live entry in `dirty`. Reaped entries
///   are filtered first — a stale `ResourceId` whose slot was vacated
///   mid-burst would yield no parent chain, and the intersection would
///   degenerate.
/// - When `dirty` is empty, returns `profile.resource` (anchor): falls
///   back to a full-walk gracefully.
pub(crate) fn lca_target(
    profile: &Profile,
    dirty: &BTreeSet<ResourceId>,
    tree: &Tree,
) -> ResourceId {
    // 1. Filter stale ResourceIds. A `dirty_resources` entry whose slot
    // was reaped between FsEvent ingestion and probe emission
    // (delete-recreate-different-inode race) yields None on `tree.parent`,
    // narrowing the intersection to nothing.
    let live: SmallVec<[ResourceId; 4]> = dirty
        .iter()
        .copied()
        .filter(|&r| tree.get(r).is_some())
        .collect();

    if live.is_empty() {
        return profile.resource;
    }
    // Anchor in the dirty set ⇒ can't go higher than anchor; trivially LCA.
    if live.contains(&profile.resource) {
        return promote_to_dir(profile.resource, profile, tree);
    }

    // 2. LCA via ancestor-chain intersection. The result is the deepest
    // (max-depth) Resource present in every chain. Empty intersection
    // (rare: cross-anchor dirty mix that should not happen — `on_fs_event`
    // filters by covering Profiles) falls back to anchor.
    let first = live[0];
    let mut chain: BTreeSet<ResourceId> = std::iter::once(first)
        .chain(tree.ancestors(first))
        .collect();
    for &r in &live[1..] {
        let mine: BTreeSet<ResourceId> = std::iter::once(r).chain(tree.ancestors(r)).collect();
        chain = chain.intersection(&mine).copied().collect();
        if chain.is_empty() {
            return profile.resource;
        }
    }
    // Pick the deepest candidate (max ancestor count).
    let lca = chain
        .into_iter()
        .max_by_key(|&r| tree.ancestors(r).count())
        .unwrap_or(profile.resource);

    promote_to_dir(lca, profile, tree)
}

/// Promote a non-Dir candidate to its parent Dir; probes target Dirs.
/// Falls back to `profile.resource` if the chain crosses a reaped slot.
fn promote_to_dir(start: ResourceId, profile: &Profile, tree: &Tree) -> ResourceId {
    let mut current = start;
    loop {
        match tree.get(current).map(|r| r.kind) {
            Some(ResourceKind::Dir) => return current,
            Some(_) => match tree.parent(current) {
                Some(p) => current = p,
                None => return profile.resource,
            },
            None => return profile.resource,
        }
    }
}

/// Build the `force_walk` set the walker consumes. Engine-side closure of
/// `force_walk_resources ∩ subtree(target)` rendered to the walker's
/// path-keyed contract.
///
/// The walker checks `force_walk.iter().any(|p| p.starts_with(current))`
/// at every recursion level; pre-filtering by ancestry of `target` keeps
/// the set minimal — out-of-subtree entries cannot affect the walk and
/// would only inflate the walker's per-dir scan.
pub(crate) fn build_force_walk(
    set: &BTreeSet<ResourceId>,
    target: ResourceId,
    tree: &Tree,
) -> BTreeSet<PathBuf> {
    set.iter()
        .copied()
        .filter(|&r| r_is_at_or_under(r, target, tree))
        .filter_map(|r| tree.path_of(r))
        .collect()
}

/// Returns true iff `r` is `target` or a descendant of `target` (i.e., `r`
/// lies in `target`'s subtree). The walk goes `r → parent(r) → ...` until
/// it hits `target` (true) or runs out of ancestors (false).
fn r_is_at_or_under(r: ResourceId, target: ResourceId, tree: &Tree) -> bool {
    let mut cur = Some(r);
    while let Some(c) = cur {
        if c == target {
            return true;
        }
        cur = tree.parent(c);
    }
    false
}

#[cfg(test)]
mod tests {
    // Tests prioritize readability over the workspace's pedantic style budget.
    #![allow(
        clippy::manual_let_else,
        clippy::match_wildcard_for_single_variants,
        clippy::missing_const_for_fn,
        clippy::needless_pass_by_value,
        clippy::too_many_lines
    )]

    use crate::Engine;
    use specter_core::{
        BurstIntent, BurstPhase, ClassSet, ProbeOp, Profile, ProfileState, ResourceKind,
        ResourceRole, ScanConfig, StepOutput, WatchOp,
    };
    use std::time::{Duration, Instant};

    const SETTLE: Duration = Duration::from_millis(100);
    const MAX_SETTLE: Duration = Duration::from_secs(6);
    const NO_EVENTS: ClassSet = ClassSet::EMPTY;

    /// Build an Engine with a single Profile anchored at `/anchor`. Returns the
    /// Engine + the `ProfileId`.
    fn engine_with_profile() -> (Engine, specter_core::ProfileId) {
        let mut e = Engine::new();
        let r = e.tree.ensure(None, "anchor", ResourceRole::User);
        e.tree.get_mut(r).unwrap().kind = ResourceKind::Dir;
        let pid = e.profiles.attach(
            &mut e.tree,
            Profile::new(
                r,
                ScanConfig::builder().recursive(true).build(),
                MAX_SETTLE,
                SETTLE,
                NO_EVENTS,
            ),
        );
        (e, pid)
    }

    #[test]
    fn start_seed_burst_emits_probe_and_suppress() {
        let (mut e, pid) = engine_with_profile();
        let mut out = StepOutput::default();
        e.start_seed_burst(pid, Instant::now(), &mut out);

        // Profile transitioned to Active(Seed Verifying).
        let p = e.profiles.get(pid).unwrap();
        let burst = match &p.state {
            ProfileState::Active(b) => b,
            _ => panic!("expected Active"),
        };
        assert_eq!(burst.intent, BurstIntent::Seed);
        assert!(matches!(burst.phase, BurstPhase::Verifying { .. }));
        assert!(!burst.forced);

        // Output: one Probe + one Suppress.
        let probes = out
            .probe_ops
            .iter()
            .filter(|op| matches!(op, ProbeOp::Probe { .. }))
            .count();
        assert_eq!(probes, 1);
        let suppresses = out
            .watch_ops
            .iter()
            .filter(|op| matches!(op, WatchOp::Suppress { .. }))
            .count();
        assert_eq!(suppresses, 1);

        // Heap: only burst_deadline (Seed has no settle_timer).
        assert_eq!(e.timers.len(), 1);
    }

    #[test]
    fn start_standard_burst_schedules_two_timers_no_probe() {
        let (mut e, pid) = engine_with_profile();
        let mut out = StepOutput::default();
        e.start_standard_burst(
            pid,
            e.profiles.get(pid).unwrap().resource,
            Instant::now(),
            &mut out,
        );

        let p = e.profiles.get(pid).unwrap();
        let burst = match &p.state {
            ProfileState::Active(b) => b,
            _ => panic!("expected Active"),
        };
        assert_eq!(burst.intent, BurstIntent::Standard);
        assert!(matches!(burst.phase, BurstPhase::Batching { .. }));

        // Heap holds settle_timer + burst_deadline.
        assert_eq!(e.timers.len(), 2);

        // No probe yet (settle_timer fires first).
        assert!(out.probe_ops.is_empty());
        let suppresses = out
            .watch_ops
            .iter()
            .filter(|op| matches!(op, WatchOp::Suppress { .. }))
            .count();
        assert_eq!(suppresses, 1);
    }

    #[test]
    fn transition_to_verifying_mints_correlation_and_emits_probe() {
        let (mut e, pid) = engine_with_profile();
        let mut out = StepOutput::default();
        e.start_standard_burst(
            pid,
            e.profiles.get(pid).unwrap().resource,
            Instant::now(),
            &mut out,
        );
        out.probe_ops.clear();

        e.transition_to_verifying(pid, &mut out);

        let p = e.profiles.get(pid).unwrap();
        let burst = match &p.state {
            ProfileState::Active(b) => b,
            _ => panic!("expected Active"),
        };
        let correlation = match burst.phase {
            BurstPhase::Verifying { correlation } => correlation,
            _ => panic!("expected Verifying phase"),
        };

        // Output: one Probe whose correlation matches.
        let probe_correlation = out.probe_ops.iter().find_map(|op| match op {
            ProbeOp::Probe { request } => Some(request.correlation),
            _ => None,
        });
        assert_eq!(probe_correlation, Some(correlation));
    }

    #[test]
    fn event_during_verifying_emits_cancel_and_resets_batching() {
        // FsEvent during Verifying: Cancel emitted; phase becomes Batching
        // with a fresh settle_timer; intent preserved.
        let (mut e, pid) = engine_with_profile();
        let mut out = StepOutput::default();
        e.start_seed_burst(pid, Instant::now(), &mut out); // Seed → Verifying
        out.probe_ops.clear();
        let r = e.profiles.get(pid).unwrap().resource;

        e.event_drives_batching(pid, r, Instant::now(), &mut out);

        // One Cancel emitted for the in-flight probe.
        let cancel_count = out
            .probe_ops
            .iter()
            .filter(|op| matches!(op, ProbeOp::Cancel { .. }))
            .count();
        assert_eq!(cancel_count, 1);

        let p = e.profiles.get(pid).unwrap();
        let burst = match &p.state {
            ProfileState::Active(b) => b,
            _ => panic!("expected Active"),
        };
        assert!(matches!(burst.phase, BurstPhase::Batching { .. }));
        assert_eq!(
            burst.intent,
            BurstIntent::Seed,
            "intent preserved across Verifying → Batching",
        );
    }

    #[test]
    fn event_during_batching_does_not_emit_cancel() {
        // Already in Batching: a fresh FsEvent reschedules without Cancel.
        let (mut e, pid) = engine_with_profile();
        let mut out = StepOutput::default();
        let r = e.profiles.get(pid).unwrap().resource;
        e.start_standard_burst(pid, r, Instant::now(), &mut out);
        out.probe_ops.clear();

        e.event_drives_batching(pid, r, Instant::now(), &mut out);

        let cancels = out
            .probe_ops
            .iter()
            .filter(|op| matches!(op, ProbeOp::Cancel { .. }))
            .count();
        assert_eq!(cancels, 0);

        // Still Batching; intent preserved.
        let p = e.profiles.get(pid).unwrap();
        let burst = match &p.state {
            ProfileState::Active(b) => b,
            _ => panic!("expected Active"),
        };
        assert!(matches!(burst.phase, BurstPhase::Batching { .. }));
        assert_eq!(burst.intent, BurstIntent::Standard);
    }

    #[test]
    fn unstable_response_does_not_emit_cancel() {
        // Standard burst → Batching → Verifying → simulated unstable
        // response → Batching. The transition emits NO Cancel — the helper
        // structurally refuses to emit one because the verify just
        // responded.
        let (mut e, pid) = engine_with_profile();
        let mut out = StepOutput::default();
        let resource = e.profiles.get(pid).unwrap().resource;
        let now = Instant::now();
        e.start_standard_burst(pid, resource, now, &mut out);
        e.transition_to_verifying(pid, &mut out);
        out.probe_ops.clear();

        e.unstable_response_drives_batching(pid, now);

        assert!(out.probe_ops.is_empty());
        let phase = match &e.profiles.get(pid).unwrap().state {
            ProfileState::Active(burst) => &burst.phase,
            _ => panic!("expected Active"),
        };
        assert!(matches!(phase, BurstPhase::Batching { .. }));
    }

    #[test]
    fn event_storm_during_batching_does_not_amplify_settle() {
        // Fire ten FsEvents at 50 ms intervals during a Standard burst.
        // The settle timer's deadline must equal `last_event + settle`,
        // not be amplified by the prior backoff curve. This is the
        // direct assertion that the conflation has been removed.
        let (mut e, pid) = engine_with_profile();
        let mut out = StepOutput::default();
        let r = e.profiles.get(pid).unwrap().resource;
        let t0 = Instant::now();
        e.start_standard_burst(pid, r, t0, &mut out);

        let mut last_event = t0;
        for k in 1..=10 {
            last_event = t0 + Duration::from_millis(50 * k);
            out.probe_ops.clear();
            e.event_drives_batching(pid, r, last_event, &mut out);
        }

        let phase = match &e.profiles.get(pid).unwrap().state {
            ProfileState::Active(b) => &b.phase,
            _ => panic!("expected Active"),
        };
        let settle_timer = match phase {
            BurstPhase::Batching { settle_timer } => *settle_timer,
            _ => panic!("expected Batching"),
        };
        let deadline = e
            .timers
            .iter()
            .find(|entry| entry.id == settle_timer)
            .map(|entry| entry.deadline)
            .expect("settle timer present in heap");
        assert_eq!(
            deadline,
            last_event + SETTLE,
            "settle timer pinned at last_event + settle, not amplified",
        );
    }

    #[test]
    fn finish_burst_to_idle_emits_unsuppress() {
        let (mut e, pid) = engine_with_profile();
        let mut out = StepOutput::default();
        e.start_seed_burst(pid, Instant::now(), &mut out);
        out.watch_ops.clear();

        e.finish_burst_to_idle(pid, &mut out);

        assert!(matches!(
            e.profiles.get(pid).unwrap().state,
            ProfileState::Idle,
        ));
        let unsuppresses = out
            .watch_ops
            .iter()
            .filter(|op| matches!(op, WatchOp::Unsuppress { .. }))
            .count();
        assert_eq!(unsuppresses, 1);
    }

    #[test]
    fn finish_burst_to_idle_on_idle_is_noop() {
        let (mut e, pid) = engine_with_profile();
        let mut out = StepOutput::default();
        e.finish_burst_to_idle(pid, &mut out);
        assert!(out.watch_ops.is_empty());
        assert!(out.probe_ops.is_empty());
    }

    #[test]
    fn burst_deadline_unchanged_across_phase_transitions() {
        let (mut e, pid) = engine_with_profile();
        let mut out = StepOutput::default();
        e.start_seed_burst(pid, Instant::now(), &mut out);
        let burst_deadline_initial = match &e.profiles.get(pid).unwrap().state {
            ProfileState::Active(b) => b.burst_deadline,
            _ => panic!(),
        };
        let r = e.profiles.get(pid).unwrap().resource;

        e.event_drives_batching(pid, r, Instant::now(), &mut out);
        let burst_deadline_after = match &e.profiles.get(pid).unwrap().state {
            ProfileState::Active(b) => b.burst_deadline,
            _ => panic!(),
        };
        assert_eq!(
            burst_deadline_initial, burst_deadline_after,
            "burst_deadline does not reschedule across Verifying → Batching",
        );
    }

    #[test]
    fn transition_to_draining_swaps_phase_only() {
        let (mut e, pid) = engine_with_profile();
        let mut out = StepOutput::default();
        e.start_seed_burst(pid, Instant::now(), &mut out);

        e.transition_to_draining(pid);

        let p = e.profiles.get(pid).unwrap();
        let burst = match &p.state {
            ProfileState::Active(b) => b,
            _ => panic!(),
        };
        assert!(matches!(burst.phase, BurstPhase::Draining));
        // Intent and forced preserved.
        assert_eq!(burst.intent, BurstIntent::Seed);
    }

    // ---------------------------------------------------------------------------
    // LCA + force_walk + transition_to_verifying
    // ---------------------------------------------------------------------------

    use crate::burst::{build_force_walk, lca_target};
    use std::collections::BTreeSet;

    /// Build a tree-shaped Engine: anchor `/root`, two children `a` and `b`.
    fn engine_with_two_children() -> (
        Engine,
        specter_core::ProfileId,
        specter_core::ResourceId,
        specter_core::ResourceId,
        specter_core::ResourceId,
    ) {
        let mut e = Engine::new();
        let root = e.tree.ensure(None, "root", ResourceRole::User);
        e.tree.get_mut(root).unwrap().kind = ResourceKind::Dir;
        let a = e.tree.ensure(Some(root), "a", ResourceRole::User);
        e.tree.get_mut(a).unwrap().kind = ResourceKind::Dir;
        let b = e.tree.ensure(Some(root), "b", ResourceRole::User);
        e.tree.get_mut(b).unwrap().kind = ResourceKind::Dir;
        let pid = e.profiles.attach(
            &mut e.tree,
            Profile::new(
                root,
                ScanConfig::builder().recursive(true).build(),
                MAX_SETTLE,
                SETTLE,
                NO_EVENTS,
            ),
        );
        (e, pid, root, a, b)
    }

    #[test]
    fn lca_empty_dirty_returns_anchor() {
        let (e, pid, root, _a, _b) = engine_with_two_children();
        let dirty = BTreeSet::new();
        let target = lca_target(e.profiles.get(pid).unwrap(), &dirty, &e.tree);
        assert_eq!(target, root);
    }

    #[test]
    fn lca_two_siblings_returns_parent() {
        let (e, pid, root, a, b) = engine_with_two_children();
        let dirty: BTreeSet<_> = [a, b].iter().copied().collect();
        let target = lca_target(e.profiles.get(pid).unwrap(), &dirty, &e.tree);
        assert_eq!(target, root);
    }

    #[test]
    fn lca_single_dirty_at_anchor_returns_anchor() {
        let (e, pid, root, _a, _b) = engine_with_two_children();
        let dirty: BTreeSet<_> = std::iter::once(root).collect();
        let target = lca_target(e.profiles.get(pid).unwrap(), &dirty, &e.tree);
        assert_eq!(target, root);
    }

    #[test]
    fn lca_single_dirty_deep_returns_self() {
        let (e, pid, _root, a, _b) = engine_with_two_children();
        let dirty: BTreeSet<_> = std::iter::once(a).collect();
        let target = lca_target(e.profiles.get(pid).unwrap(), &dirty, &e.tree);
        assert_eq!(target, a);
    }

    #[test]
    fn lca_filters_stale_resource_ids() {
        let (mut e, pid, root, a, _b) = engine_with_two_children();
        // Reap `a` to make its id stale.
        e.tree.vacate(a);
        e.tree.try_reap(a);
        // Stale id in the set; LCA must filter and return anchor (since the
        // remaining live entry is empty after the filter).
        let dirty: BTreeSet<_> = std::iter::once(a).collect();
        let target = lca_target(e.profiles.get(pid).unwrap(), &dirty, &e.tree);
        assert_eq!(target, root);
    }

    #[test]
    fn build_force_walk_filters_to_subtree_of_target() {
        let (e, _pid, root, a, b) = engine_with_two_children();
        // target = a; only `a` itself qualifies (b is a sibling).
        let set: BTreeSet<_> = [root, a, b].iter().copied().collect();
        let paths = build_force_walk(&set, a, &e.tree);
        let path_a = e.tree.path_of(a).unwrap();
        assert!(paths.contains(&path_a));
        assert!(!paths.contains(&e.tree.path_of(b).unwrap()));
        // root is an ancestor of a (not a descendant), so it's filtered out.
        assert!(!paths.contains(&e.tree.path_of(root).unwrap()));
    }

    #[test]
    fn transition_to_verifying_standard_uses_lca() {
        let (mut e, pid, _root, a, b) = engine_with_two_children();
        let mut out = StepOutput::default();
        let now = Instant::now();
        // Standard burst with two dirty siblings → LCA = root (the anchor).
        e.start_standard_burst(pid, a, now, &mut out);
        // Inject a second dirty resource so LCA computes the sibling parent.
        if let ProfileState::Active(b_burst) = &mut e.profiles.get_mut(pid).unwrap().state {
            b_burst.dirty_resources.insert(b);
            b_burst.force_walk_resources.insert(b);
        }
        let mut probe_out = StepOutput::default();
        e.transition_to_verifying(pid, &mut probe_out);

        let req = probe_out
            .probe_ops
            .iter()
            .find_map(|op| match op {
                ProbeOp::Probe { request } => Some(request),
                ProbeOp::Cancel { .. } => None,
            })
            .expect("Standard probe emitted");
        // a + b's LCA is root (the anchor) because they're siblings under root.
        assert_eq!(req.target_resource, e.profiles.get(pid).unwrap().resource);
        // force_walk has both event-dirty paths.
        assert_eq!(req.force_walk.len(), 2);
    }

    #[test]
    fn transition_to_verifying_clears_force_walk_resources() {
        let (mut e, pid, _root, a, _b) = engine_with_two_children();
        let mut out = StepOutput::default();
        let now = Instant::now();
        e.start_standard_burst(pid, a, now, &mut out);
        e.transition_to_verifying(pid, &mut out);

        // After transition_to_verifying, force_walk_resources should be
        // cleared (consumed by this emission); subsequent events accumulate
        // fresh.
        let burst = match &e.profiles.get(pid).unwrap().state {
            ProfileState::Active(b) => b,
            _ => panic!(),
        };
        assert!(burst.force_walk_resources.is_empty());
        // dirty_resources is preserved (LCA basis spans the whole burst).
        assert!(!burst.dirty_resources.is_empty());
        // probe_target was set to the LCA result.
        assert!(burst.probe_target.is_some());
    }
}
