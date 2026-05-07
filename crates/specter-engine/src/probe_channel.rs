//! Probe channel — engine↔Prober communication primitive.
//!
//! At most one outstanding probe per Profile. The channel is **opened**
//! when the engine emits `ProbeOp::Probe` (after `mint_probe_correlation`
//! writes the correlation to `Profile.pending_probe`). The channel is
//! **closed** when either:
//! - the matching `ProbeResponse` arrives (top of `on_probe_response`
//!   clears the slot before dispatch), or
//! - the engine emits `ProbeOp::Cancel` (`cancel_pending_probe` clears
//!   the slot and emits Cancel atomically).
//!
//! This module is the single source of three disciplines:
//! 1. Counter monotonicity for the probe side — `mint_probe_correlation`
//!    is the only path that bumps `Engine.next_correlation` for a probe
//!    token. The effect side (`next_effect_correlation` in
//!    `transitions.rs`) shares the underlying counter; the typed wrappers
//!    (`ProbeCorrelation` vs `CorrelationId`) keep the spaces disjoint.
//! 2. Channel-state slot — only `mint_probe_correlation` writes
//!    `pending_probe = Some(_)`; only `cancel_pending_probe` and the
//!    `on_probe_response` pre-dispatch clear write `pending_probe = None`.
//! 3. `ProbeOp::Probe` construction — the three typed helpers
//!    (`emit_anchor_probe`, `emit_subtree_probe`, `emit_descent_probe`)
//!    are the only paths that push a `ProbeOp::Probe`. Each helper bakes
//!    the request variant: callers cannot accidentally ship a Subtree
//!    request to a File-anchored Profile, or attach a baseline to a
//!    descent prefix.
//!
//! The data-model slot (`Profile.pending_probe`) lives on the Profile;
//! this module owns the discipline that mutates it.

use crate::Engine;
use specter_core::{
    DirSnapshot, ProbeCorrelation, ProbeOp, ProbeRequest, ProfileId, ResourceId, ScanConfig,
    StepOutput,
};
use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::Arc;

impl Engine {
    /// Open the probe channel: bump the engine-monotonic correlation
    /// counter, mint a fresh `ProbeCorrelation`, and write it to
    /// `Profile.pending_probe`.
    ///
    /// Returns `None` only on stale `pid` (defense-in-depth; production
    /// paths look up the Profile within the same `&mut self` window
    /// immediately before this call). On success the slot is `Some(c)`
    /// and the returned `ProbeCorrelation` matches.
    ///
    /// **I5 enforcement.** A `debug_assert!` fires on double-open. Release
    /// builds silently overwrite — benign because the now-orphaned
    /// outstanding probe's response will fail the
    /// `pending_probe == Some(received)` check at the top of
    /// `on_probe_response` and emit `StaleProbeResponse`. The assertion
    /// is the early-warning signal in CI/dev.
    #[must_use]
    pub(crate) fn mint_probe_correlation(&mut self, pid: ProfileId) -> Option<ProbeCorrelation> {
        let p = self.profiles.get_mut(pid)?;
        debug_assert!(
            p.pending_probe.is_none(),
            "I5 violated: minting probe correlation while channel is open \
             (existing = {:?}, profile = {pid:?})",
            p.pending_probe,
        );
        // The borrow on `p` ends here (NLL); `&mut self.next_correlation`
        // and the re-borrow of `self.profiles` below are disjoint.
        debug_assert!(
            self.next_correlation < u64::MAX,
            "Engine.next_correlation saturated at u64::MAX; subsequent probe \
             correlations would collide with effect correlations and break \
             stale-response detection",
        );
        self.next_correlation = self.next_correlation.saturating_add(1);
        let correlation = ProbeCorrelation(self.next_correlation);
        if let Some(p) = self.profiles.get_mut(pid) {
            p.pending_probe = Some(correlation);
        }
        Some(correlation)
    }

    /// Close the probe channel and emit `ProbeOp::Cancel` iff the channel
    /// was open. Idempotent — silently no-ops on a closed channel or stale
    /// `pid`.
    ///
    /// Sole caller surface for the four cancel-emission paths:
    /// `event_drives_batching`, `finalize_anchor_lost`,
    /// `on_watch_op_rejected` descent purge, `reap_profile`.
    pub(crate) fn cancel_pending_probe(&mut self, pid: ProfileId, out: &mut StepOutput) {
        if let Some(p) = self.profiles.get_mut(pid)
            && p.pending_probe.take().is_some()
        {
            out.probe_ops.push(ProbeOp::Cancel { profile: pid });
        }
    }

    /// Read accessor for `Profile.pending_probe`. Returns `None` for a
    /// stale `pid` or a closed channel. Mirrors the read API previously
    /// served by `DescentState::probe_correlation()` and the
    /// `BurstPhase::Verifying { correlation }` destructure — both deleted
    /// in favour of this single Profile-level slot. `pub` so integration
    /// tests in the engine crate's `tests/` directory can query the
    /// channel state via the engine's public surface.
    #[must_use]
    pub fn pending_probe(&self, pid: ProfileId) -> Option<ProbeCorrelation> {
        self.profiles.get(pid)?.pending_probe
    }

    /// Emit `ProbeRequest::AnchorFile`. The walker runs a single `lstat`
    /// against `target_path` and returns `ProbeOutcome::AnchorOk` (or
    /// `Vanished` / `Failed`).
    ///
    /// `correlation` must already be on `Profile.pending_probe` (the
    /// caller minted it via `mint_probe_correlation` immediately prior).
    /// `target_path` is captured at the call site so this helper avoids
    /// borrowing `&self.tree` during the emit.
    ///
    /// Associated function (no `self`): the helper is a thin variant
    /// constructor with no Engine-state dependency. Callers reach it as
    /// `Self::emit_anchor_probe(...)`.
    pub(crate) fn emit_anchor_probe(
        profile_id: ProfileId,
        correlation: ProbeCorrelation,
        target_path: PathBuf,
        out: &mut StepOutput,
    ) {
        out.probe_ops.push(ProbeOp::Probe {
            request: ProbeRequest::AnchorFile {
                profile: profile_id,
                correlation,
                target_path,
            },
        });
    }

    /// Emit `ProbeRequest::Subtree`. Recursive Dir walk honouring the
    /// Profile's `ScanConfig`; walker returns
    /// `ProbeOutcome::SubtreeOk(Arc<DirSnapshot>)` rooted at
    /// `target_resource`.
    ///
    /// `scan_config` and `captured_with` come from the Profile — the
    /// caller already holds a `&Profile` borrow at every call site (to
    /// read `kind`, `current`, `resource`, etc.) and threads
    /// `(p.config.clone(), p.config_hash)` through here. The helper does
    /// not re-borrow `self` to look them up, which would also force the
    /// helper to take `&self` for an otherwise stateless construction.
    ///
    /// Associated function (no `self`): same rationale as
    /// [`Self::emit_anchor_probe`].
    pub(crate) fn emit_subtree_probe(
        profile_id: ProfileId,
        correlation: ProbeCorrelation,
        target_resource: ResourceId,
        target_path: PathBuf,
        scan_config: ScanConfig,
        captured_with: u64,
        baseline_subtree: Option<Arc<DirSnapshot>>,
        force_walk: BTreeSet<PathBuf>,
        forced: bool,
        out: &mut StepOutput,
    ) {
        out.probe_ops.push(ProbeOp::Probe {
            request: ProbeRequest::Subtree {
                profile: profile_id,
                correlation,
                target_resource,
                target_path,
                scan_config,
                captured_with,
                baseline_subtree,
                force_walk,
                forced,
            },
        });
    }

    /// Emit `ProbeRequest::Descent`. Single-level enumeration of the
    /// prefix; walker hardcodes the override config (`recursive=false`,
    /// `hidden=true`, no exclude/pattern, no `max_depth`) — the
    /// Profile's user-facing filters would mask the very segment descent
    /// is searching for.
    ///
    /// Walker still returns `ProbeOutcome::SubtreeOk(arc)` carrying the
    /// prefix's direct children; descent dispatch reads
    /// `arc.entries.get(name)` and discards the snapshot.
    ///
    /// Associated function (no `self`): same rationale as
    /// [`Self::emit_anchor_probe`].
    pub(crate) fn emit_descent_probe(
        profile_id: ProfileId,
        correlation: ProbeCorrelation,
        target_path: PathBuf,
        out: &mut StepOutput,
    ) {
        out.probe_ops.push(ProbeOp::Probe {
            request: ProbeRequest::Descent {
                profile: profile_id,
                correlation,
                target_path,
            },
        });
    }
}

#[cfg(test)]
mod tests {
    use crate::Engine;
    use specter_core::{ClassSet, Profile, ResourceRole, ScanConfig, StepOutput};
    use std::time::Duration;

    const SETTLE: Duration = Duration::from_millis(100);
    const MAX_SETTLE: Duration = Duration::from_secs(6);

    /// Attach a fresh `Idle` Profile at a synthetic anchor, returning the
    /// engine and the new `ProfileId`. The Profile carries no Subs and no
    /// claims — purely a vehicle for exercising the probe-channel slot in
    /// isolation.
    fn fresh_engine_with_idle_profile() -> (Engine, specter_core::ProfileId) {
        let mut e = Engine::new();
        let r = e.tree.ensure(None, "anchor", ResourceRole::User);
        let pid = e.profiles.attach(
            &mut e.tree,
            Profile::new(
                r,
                ScanConfig::builder().build(),
                MAX_SETTLE,
                SETTLE,
                ClassSet::EMPTY,
            ),
        );
        (e, pid)
    }

    /// Double-open is a state-machine bug (I5 violation). Debug builds fire
    /// the assertion in `mint_probe_correlation`; release builds silently
    /// overwrite (the now-orphaned probe's response staless against the
    /// new correlation).
    #[test]
    #[cfg_attr(
        not(debug_assertions),
        ignore = "debug_assert! is compiled out in release"
    )]
    #[should_panic(expected = "I5 violated")]
    fn mint_probe_correlation_panics_on_double_open() {
        let (mut e, pid) = fresh_engine_with_idle_profile();
        let _ = e.mint_probe_correlation(pid).expect("first mint succeeds");
        let _ = e.mint_probe_correlation(pid); // panics: I5 violated
    }

    /// Closed channel + cancel = no-op. The helper's idempotence is
    /// load-bearing — `event_drives_batching` invokes it on every event,
    /// regardless of whether a probe is in flight.
    #[test]
    fn cancel_pending_probe_idempotent_on_closed_channel() {
        let (mut e, pid) = fresh_engine_with_idle_profile();
        assert!(e.pending_probe(pid).is_none(), "channel starts closed");
        let mut out = StepOutput::default();
        e.cancel_pending_probe(pid, &mut out);
        assert!(
            out.probe_ops.is_empty(),
            "no Cancel emitted on closed channel",
        );
        assert!(e.pending_probe(pid).is_none(), "channel remains closed");
    }

    /// Open channel + cancel = single Cancel emission + slot cleared. Pairs
    /// with the idempotence test above to fully spec the helper's contract.
    #[test]
    fn cancel_pending_probe_emits_and_clears_on_open_channel() {
        use specter_core::ProbeOp;
        let (mut e, pid) = fresh_engine_with_idle_profile();
        let corr = e.mint_probe_correlation(pid).expect("mint succeeds");
        assert_eq!(e.pending_probe(pid), Some(corr));

        let mut out = StepOutput::default();
        e.cancel_pending_probe(pid, &mut out);

        assert_eq!(out.probe_ops.len(), 1, "exactly one Cancel emitted");
        assert!(
            matches!(out.probe_ops[0], ProbeOp::Cancel { profile } if profile == pid),
            "Cancel targets the same Profile",
        );
        assert!(e.pending_probe(pid).is_none(), "channel closed post-cancel");
    }

    /// Cancel is per-Profile: closing one Profile's channel doesn't touch
    /// another's. Cross-Profile concurrency is necessary for descent fan-out
    /// at `on_descent_event` (multiple Pending Profiles awaiting siblings
    /// under one prefix).
    #[test]
    fn cancel_pending_probe_is_per_profile() {
        let mut e = Engine::new();
        let r1 = e.tree.ensure(None, "a", ResourceRole::User);
        let r2 = e.tree.ensure(None, "b", ResourceRole::User);
        let cfg = ScanConfig::builder().build();
        let pid1 = e.profiles.attach(
            &mut e.tree,
            Profile::new(r1, cfg.clone(), MAX_SETTLE, SETTLE, ClassSet::EMPTY),
        );
        let pid2 = e.profiles.attach(
            &mut e.tree,
            Profile::new(r2, cfg, MAX_SETTLE, SETTLE, ClassSet::EMPTY),
        );
        let c1 = e.mint_probe_correlation(pid1).unwrap();
        let c2 = e.mint_probe_correlation(pid2).unwrap();

        let mut out = StepOutput::default();
        e.cancel_pending_probe(pid1, &mut out);

        assert!(e.pending_probe(pid1).is_none());
        assert_eq!(
            e.pending_probe(pid2),
            Some(c2),
            "pid2's channel untouched by pid1's cancel",
        );
        assert_ne!(c1, c2, "correlations are distinct across Profiles");
    }

    /// Stale `pid` (post-detach) — both helpers no-op without panic.
    /// `cancel_pending_probe` defends against late `WatchOpRejected`
    /// purges arriving after `reap_profile` already detached.
    #[test]
    fn helpers_noop_on_stale_pid() {
        let (mut e, pid) = fresh_engine_with_idle_profile();
        let _ = e.profiles.detach(&mut e.tree, pid);

        assert!(
            e.mint_probe_correlation(pid).is_none(),
            "mint returns None for stale pid",
        );

        let mut out = StepOutput::default();
        e.cancel_pending_probe(pid, &mut out);
        assert!(out.probe_ops.is_empty(), "cancel no-ops on stale pid");
        assert!(e.pending_probe(pid).is_none(), "accessor returns None");
    }
}
