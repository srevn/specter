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
//! 3. `ProbeOp::Probe` construction — `emit_probe_op` is the only path
//!    that pushes a `ProbeOp::Probe`.
//!
//! The data-model slot (`Profile.pending_probe`) lives on the Profile;
//! this module owns the discipline that mutates it.

use crate::Engine;
use specter_core::{
    DirSnapshot, ProbeCorrelation, ProbeKind, ProbeOp, ProbeRequest, ProfileId, ResourceId,
    ResourceKind, ScanConfig, StepOutput,
};
use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::Arc;

/// Parameters that vary between Burst-driven and Descent-driven probe
/// emissions. The two cases are **disjoint** — descent never ships a
/// baseline, force_walk, or forced flag — so the type encodes them as
/// distinct enum variants. `ProbeEmissionParams::Descent` is unit; no
/// future contributor can construct a "descent probe with a baseline"
/// because the type forbids it.
///
/// Burst-side fields whose values come from the Profile (`scan_config`,
/// `captured_with`) are NOT carried here — `emit_probe_op` reads them
/// from the Profile at emission time, eliminating the parameter-passing
/// redundancy a struct-shape would have introduced.
pub(crate) enum ProbeEmissionParams {
    /// Burst-driven probe: ships baseline (mtime-skip basis), force_walk
    /// (kqueue-driven dirty paths), and the `forced` bit (max-settle
    /// override). Used by `start_seed_burst` and `transition_to_verifying`.
    Burst {
        baseline_subtree: Option<Arc<DirSnapshot>>,
        force_walk: BTreeSet<PathBuf>,
        forced: bool,
    },
    /// Descent-driven probe: minimal single-level enumeration overriding
    /// the Profile's `ScanConfig`. No baseline (the descent has no prior
    /// observation at any prefix); no force_walk (the prefix is the
    /// target); not forced. Used by `attach_sub_inner` Pending branch,
    /// `start_pending_recovery`, `dispatch_descent_ok` advance branch,
    /// `dispatch_descent_vanished` rewind branch, and `on_descent_event`.
    Descent,
}

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

    /// Push a `ProbeOp::Probe` onto `out.probe_ops`. The single source of
    /// `ProbeOp::Probe` construction; both Burst and Descent emission
    /// route through here.
    ///
    /// `correlation` must already be on `Profile.pending_probe` (the
    /// caller minted it via `mint_probe_correlation` immediately prior).
    /// `params` carries the disjoint per-arm fields.
    ///
    /// Resolves the probe kind from the target's `ResourceKind`
    /// (`Unknown` defaults to `Directory` — the more permissive choice;
    /// the Sensor returns `Vanished` on kind mismatch, which the Engine
    /// then handles as Removed).
    pub(crate) fn emit_probe_op(
        &self,
        profile_id: ProfileId,
        target_resource: ResourceId,
        correlation: ProbeCorrelation,
        params: ProbeEmissionParams,
        out: &mut StepOutput,
    ) {
        let Some(p) = self.profiles.get(profile_id) else {
            return;
        };
        let kind = match self.tree.get(target_resource).map(|r| r.kind) {
            Some(ResourceKind::File) => ProbeKind::File,
            _ => ProbeKind::Directory,
        };
        let target_path = self.tree.path_of(target_resource).unwrap_or_default();

        let (scan_config, baseline_subtree, force_walk, forced) = match params {
            ProbeEmissionParams::Burst {
                baseline_subtree,
                force_walk,
                forced,
            } => (p.config.clone(), baseline_subtree, force_walk, forced),
            // Descent probes only need the immediate children of the
            // prefix to search for the next segment — a recursive walk
            // is wasted I/O, and the user's pattern would filter out the
            // very file we're looking for. Override to a minimal
            // single-level enumeration: `recursive = false` (no descent
            // into children), `pattern = None` (don't filter — we're
            // looking for any segment by name), `exclude = []` (don't
            // hide ancestors of the anchor), `hidden = true` (don't skip
            // dot-prefixed ancestors), `max_depth = None` (irrelevant
            // when `recursive=false`). The Seed burst that follows
            // anchor materialization uses the Profile's real config.
            ProbeEmissionParams::Descent => (
                ScanConfig::builder()
                    .recursive(false)
                    .hidden(true)
                    .max_depth(None)
                    .build(),
                None,
                BTreeSet::new(),
                false,
            ),
        };

        out.probe_ops.push(ProbeOp::Probe {
            request: ProbeRequest {
                profile: profile_id,
                correlation,
                kind,
                target_resource,
                target_path,
                scan_config,
                // Profile identity is stable across Burst and Descent
                // emission. For descent the override `scan_config` differs
                // from the Profile's real one, but the walker stamps
                // `captured_with` onto the synthesised `DirSnapshot` and
                // the descent shim discards the snapshot before any
                // consumer reads it.
                captured_with: p.config_hash,
                baseline_subtree,
                force_walk,
                forced,
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
