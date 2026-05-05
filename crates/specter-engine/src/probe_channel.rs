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

    /// Read accessor for `Profile.pending_probe`. Returns `None` for stale
    /// `pid` or a closed channel.
    #[must_use]
    pub(crate) fn pending_probe(&self, pid: ProfileId) -> Option<ProbeCorrelation> {
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
