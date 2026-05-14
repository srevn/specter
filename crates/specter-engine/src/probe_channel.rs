//! Probe channel — engine↔Prober communication primitive.
//!
//! At most one outstanding probe per [`ProbeOwner`]. The channel is
//! **opened** when the engine emits `ProbeOp::Probe` (after
//! `mint_owner_correlation` writes the correlation to the owner's slot).
//! The channel is **closed** when either:
//! - the matching `ProbeResponse` arrives (top of `on_probe_response`
//!   clears the slot before dispatch), or
//! - the engine emits `ProbeOp::Cancel` (`cancel_owner_probe` clears
//!   the slot and emits Cancel atomically).
//!
//! This module is the single source of three disciplines:
//! 1. Counter monotonicity for the probe side — `mint_owner_correlation`
//!    is the only path that drives `Engine.probe_correlations` forward.
//!    The counter is phantom-typed
//!    ([`crate::counter::MonotonicCounter<ProbeCorrelation>`]), so a
//!    misrouted token cannot numerically collide with an effect-side
//!    [`specter_core::CorrelationId`] — the type system rejects
//!    cross-counter wiring before the values exist.
//! 2. Channel-state slot — only `mint_owner_correlation` writes
//!    `pending_probe = Some(_)`; only `cancel_owner_probe` and the
//!    `on_probe_response` pre-dispatch clear write `pending_probe = None`.
//! 3. `ProbeOp::Probe` construction — the three typed helpers
//!    (`emit_anchor_probe`, `emit_subtree_probe`, `emit_descent_probe`)
//!    are the only paths that push a `ProbeOp::Probe`. Each helper bakes
//!    the request variant: callers cannot accidentally ship a Subtree
//!    request to a File-anchored Profile, or attach a baseline to a
//!    descent prefix.
//!
//! Per-owner slot lookup goes through [`Engine::pending_slot_mut`] /
//! [`Engine::pending_slot`] — a single match on [`ProbeOwner`] keeps the
//! `mint`/`cancel`/read trio symmetric. The match is exhaustive over
//! every owner kind today; extending the enum requires adding one arm
//! here and one in the dispatcher (`on_probe_response`).

use crate::Engine;
use specter_core::{
    DirSnapshot, ProbeCorrelation, ProbeOp, ProbeOwner, ProbeRequest, ResourceId, ScanConfig,
    StepOutput,
};
use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::Arc;

impl Engine {
    /// Mutable accessor for the per-owner pending-probe slot. Returns
    /// `None` for a stale owner (slotmap key whose entity has been
    /// reaped); otherwise threads through to the owner's
    /// `pending_probe: Option<ProbeCorrelation>` field.
    ///
    /// This is the single source of truth that
    /// `mint_owner_correlation` / `cancel_owner_probe` /
    /// `pending_slot` route through — adding a new
    /// [`ProbeOwner`] variant means extending exactly one match here
    /// (and one in `on_probe_response`).
    fn pending_slot_mut(&mut self, owner: ProbeOwner) -> Option<&mut Option<ProbeCorrelation>> {
        match owner {
            ProbeOwner::Profile(pid) => self.profiles.get_mut(pid).map(|p| &mut p.pending_probe),
            ProbeOwner::Promoter(pid) => self.promoters.get_mut(pid).map(|q| &mut q.pending_probe),
        }
    }

    /// Read counterpart of [`Self::pending_slot_mut`]. Returns `None`
    /// for a stale owner or a closed channel.
    fn pending_slot(&self, owner: ProbeOwner) -> Option<ProbeCorrelation> {
        match owner {
            ProbeOwner::Profile(pid) => self.profiles.get(pid).and_then(|p| p.pending_probe),
            ProbeOwner::Promoter(pid) => self.promoters.get(pid).and_then(|q| q.pending_probe),
        }
    }

    /// Open the probe channel: bump the engine-monotonic correlation
    /// counter, mint a fresh `ProbeCorrelation`, and write it to the
    /// owner's pending-probe slot.
    ///
    /// Returns `None` only on a stale owner (defense-in-depth;
    /// production paths look up the owner within the same `&mut self`
    /// window immediately before this call). On success the slot is
    /// `Some(c)` and the returned `ProbeCorrelation` matches.
    ///
    /// **I5 enforcement.** A `debug_assert!` fires on double-open.
    /// Release builds silently overwrite — benign because the
    /// now-orphaned outstanding probe's response will fail the
    /// `pending_probe == Some(received)` check at the top of
    /// `on_probe_response` and emit `StaleProbeResponse`. The assertion
    /// is the early-warning signal in CI/dev.
    ///
    /// **Saturation.** Counter saturation panics unconditionally via
    /// [`crate::counter::MonotonicCounter::next`]; release builds are
    /// not exempt because silent wrap would re-issue an
    /// already-outstanding correlation and break stale-response
    /// detection.
    #[must_use]
    pub(crate) fn mint_owner_correlation(&mut self, owner: ProbeOwner) -> Option<ProbeCorrelation> {
        // I5 staleness check. Read-only; the &mut downgrade window for
        // the slot write happens below.
        let existing = self.pending_slot(owner);
        debug_assert!(
            existing.is_none(),
            "I5 violated: minting probe correlation while channel is open \
             (existing = {existing:?}, owner = {owner:?})",
        );
        // Stale owner — defensive bail before consuming a counter tick.
        self.pending_slot_mut(owner)?;

        let correlation = self.probe_correlations.next();
        if let Some(slot) = self.pending_slot_mut(owner) {
            *slot = Some(correlation);
        }
        Some(correlation)
    }

    /// Close the probe channel and emit `ProbeOp::Cancel` iff the channel
    /// was open. Idempotent — silently no-ops on a closed channel or stale
    /// owner.
    ///
    /// Sole caller surface for the Profile-side cancel-emission paths:
    /// `event_drives_batching`, `finalize_anchor_lost`,
    /// `on_watch_op_rejected` descent purge, `reap_profile`. Promoter
    /// callers route through [`Engine::reap_promoter_inner`].
    ///
    /// **Per-owner sibling-state cleanup.** Promoter owners carry a
    /// second slot (`pending_enumeration_target`) that pairs with
    /// `pending_probe` for enumeration probes; the lockstep is owned
    /// by [`specter_core::Promoter::close_probe_channel`] (the canonical
    /// "close both fields together" entry point on the Promoter type).
    /// Profile owners have no equivalent (descent target lives on
    /// `Profile.state` directly), so this helper short-circuits to a
    /// plain take on the slot.
    pub(crate) fn cancel_owner_probe(&mut self, owner: ProbeOwner, out: &mut StepOutput) {
        let was_open = match owner {
            ProbeOwner::Profile(pid) => self
                .profiles
                .get_mut(pid)
                .is_some_and(|p| p.pending_probe.take().is_some()),
            ProbeOwner::Promoter(pid) => self.promoters.get_mut(pid).is_some_and(|q| {
                let was_open = q.pending_probe.is_some();
                q.close_probe_channel();
                was_open
            }),
        };
        if was_open {
            out.probe_ops.push(ProbeOp::Cancel { owner });
        }
    }

    /// Read accessor for the owner's pending-probe slot. Returns `None`
    /// for a stale owner or a closed channel. `pub` so integration
    /// tests in the engine crate's `tests/` directory can query the
    /// channel state via the engine's public surface.
    #[must_use]
    pub fn pending_probe_for(&self, owner: ProbeOwner) -> Option<ProbeCorrelation> {
        self.pending_slot(owner)
    }

    /// Emit `ProbeRequest::AnchorFile`. The walker runs a single `lstat`
    /// against `target_path` and returns `ProbeOutcome::AnchorOk` (or
    /// `Vanished` / `Failed`).
    ///
    /// `correlation` must already be on the owner's pending-probe slot
    /// (the caller minted it via `mint_owner_correlation` immediately
    /// prior). `target_path` is captured at the call site so this
    /// helper avoids borrowing `&self.tree` during the emit.
    ///
    /// Associated function (no `self`): the helper is a thin variant
    /// constructor with no Engine-state dependency. Callers reach it as
    /// `Self::emit_anchor_probe(...)`.
    pub(crate) fn emit_anchor_probe(
        owner: ProbeOwner,
        correlation: ProbeCorrelation,
        target_path: PathBuf,
        out: &mut StepOutput,
    ) {
        out.probe_ops.push(ProbeOp::Probe {
            request: ProbeRequest::AnchorFile {
                owner,
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
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn emit_subtree_probe(
        owner: ProbeOwner,
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
                owner,
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
    /// `arc.entries.get(name)` and (for Profile descent) discards the
    /// snapshot.
    ///
    /// `target_resource` is the prefix the engine is enumerating. The
    /// walker stamps it onto `DirSnapshot.root_resource` so consumers
    /// reading the snapshot directly can identify the prefix without
    /// consulting per-state engine fields.
    ///
    /// Associated function (no `self`): same rationale as
    /// [`Self::emit_anchor_probe`].
    pub(crate) fn emit_descent_probe(
        owner: ProbeOwner,
        correlation: ProbeCorrelation,
        target_resource: ResourceId,
        target_path: PathBuf,
        out: &mut StepOutput,
    ) {
        out.probe_ops.push(ProbeOp::Probe {
            request: ProbeRequest::Descent {
                owner,
                correlation,
                target_resource,
                target_path,
            },
        });
    }
}

#[cfg(test)]
mod tests {
    use crate::Engine;
    use specter_core::{
        ClassSet, ProbeOp, ProbeOwner, Profile, ResourceRole, ScanConfig, StepOutput,
    };
    use std::time::Duration;

    const SETTLE: Duration = Duration::from_millis(100);
    const MAX_SETTLE: Duration = Duration::from_secs(6);

    /// Attach a fresh `Idle` Profile at a synthetic anchor, returning the
    /// engine and the new [`ProbeOwner`]. The Profile carries no Subs and
    /// no claims — purely a vehicle for exercising the probe-channel slot
    /// in isolation.
    fn fresh_engine_with_idle_profile() -> (Engine, ProbeOwner) {
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
        (e, ProbeOwner::Profile(pid))
    }

    /// Double-open is a state-machine bug (I5 violation). Debug builds fire
    /// the assertion in `mint_owner_correlation`; release builds silently
    /// overwrite (the now-orphaned probe's response staless against the
    /// new correlation).
    #[test]
    #[cfg_attr(
        not(debug_assertions),
        ignore = "debug_assert! is compiled out in release"
    )]
    #[should_panic(expected = "I5 violated")]
    fn mint_owner_correlation_panics_on_double_open() {
        let (mut e, owner) = fresh_engine_with_idle_profile();
        let _ = e
            .mint_owner_correlation(owner)
            .expect("first mint succeeds");
        let _ = e.mint_owner_correlation(owner); // panics: I5 violated
    }

    /// Counter saturation — release-runnable. Distinct from the
    /// `debug_assert!`-gated I5 test above: the underlying
    /// [`crate::counter::MonotonicCounter`] uses an unconditional
    /// `assert!`, so the panic survives the release profile. Pairs with
    /// the `MonotonicCounter` unit tests in `counter.rs`; this site test
    /// proves the engine routes through the counter at the `mint`
    /// boundary rather than reimplementing the bump.
    #[test]
    #[should_panic(expected = "MonotonicCounter")]
    fn mint_owner_correlation_panics_on_counter_saturation() {
        let (mut e, owner) = fresh_engine_with_idle_profile();
        e.probe_correlations.prime(u64::MAX);
        let _ = e.mint_owner_correlation(owner);
    }

    /// Closed channel + cancel = no-op. The helper's idempotence is
    /// load-bearing — `event_drives_batching` invokes it on every event,
    /// regardless of whether a probe is in flight.
    #[test]
    fn cancel_owner_probe_idempotent_on_closed_channel() {
        let (mut e, owner) = fresh_engine_with_idle_profile();
        assert!(
            e.pending_probe_for(owner).is_none(),
            "channel starts closed",
        );
        let mut out = StepOutput::default();
        e.cancel_owner_probe(owner, &mut out);
        assert!(
            out.probe_ops.is_empty(),
            "no Cancel emitted on closed channel",
        );
        assert!(
            e.pending_probe_for(owner).is_none(),
            "channel remains closed",
        );
    }

    /// Open channel + cancel = single Cancel emission + slot cleared. Pairs
    /// with the idempotence test above to fully spec the helper's contract.
    #[test]
    fn cancel_owner_probe_emits_and_clears_on_open_channel() {
        let (mut e, owner) = fresh_engine_with_idle_profile();
        let corr = e.mint_owner_correlation(owner).expect("mint succeeds");
        assert_eq!(e.pending_probe_for(owner), Some(corr));

        let mut out = StepOutput::default();
        e.cancel_owner_probe(owner, &mut out);

        assert_eq!(out.probe_ops.len(), 1, "exactly one Cancel emitted");
        assert!(
            matches!(out.probe_ops[0], ProbeOp::Cancel { owner: o } if o == owner),
            "Cancel targets the same owner",
        );
        assert!(
            e.pending_probe_for(owner).is_none(),
            "channel closed post-cancel",
        );
    }

    /// Cancel is per-owner: closing one owner's channel doesn't touch
    /// another's. Cross-owner concurrency is necessary for descent fan-out
    /// at `on_descent_event` (multiple Pending Profiles awaiting siblings
    /// under one prefix).
    #[test]
    fn cancel_owner_probe_is_per_owner() {
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
        let owner1 = ProbeOwner::Profile(pid1);
        let owner2 = ProbeOwner::Profile(pid2);
        let c1 = e.mint_owner_correlation(owner1).unwrap();
        let c2 = e.mint_owner_correlation(owner2).unwrap();

        let mut out = StepOutput::default();
        e.cancel_owner_probe(owner1, &mut out);

        assert!(e.pending_probe_for(owner1).is_none());
        assert_eq!(
            e.pending_probe_for(owner2),
            Some(c2),
            "owner2's channel untouched by owner1's cancel",
        );
        assert_ne!(c1, c2, "correlations are distinct across owners");
    }

    /// Stale owner (post-detach) — both helpers no-op without panic.
    /// `cancel_owner_probe` defends against late `WatchOpRejected`
    /// purges arriving after `reap_profile` already detached.
    #[test]
    fn helpers_noop_on_stale_owner() {
        let (mut e, owner) = fresh_engine_with_idle_profile();
        match owner {
            ProbeOwner::Profile(pid) => {
                let _ = e.profiles.detach(&mut e.tree, pid);
            }
            ProbeOwner::Promoter(_) => {
                unreachable!("fresh_engine_with_idle_profile only produces Profile owners");
            }
        }

        assert!(
            e.mint_owner_correlation(owner).is_none(),
            "mint returns None for stale owner",
        );

        let mut out = StepOutput::default();
        e.cancel_owner_probe(owner, &mut out);
        assert!(out.probe_ops.is_empty(), "cancel no-ops on stale owner");
        assert!(
            e.pending_probe_for(owner).is_none(),
            "accessor returns None",
        );
    }
}
