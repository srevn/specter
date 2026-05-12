//! Profile-claim release helpers.
//!
//! A Profile holds at most three Resource-side claims, each with a
//! per-Profile bookkeeping field and a per-Resource refcount contribution:
//!
//! 1. **Anchor.** `Profile.anchor_claim == AnchorClaim::Held` ⇒ Profile
//!    contributes `Profile.events_union` to `Profile.resource.watch_demand`.
//! 2. **Watch-root parent.** `Profile.watch_root_parent = Some(parent)` ⇒
//!    Profile contributes `STRUCTURE` to `parent.watch_demand`.
//! 3. **Descent prefix.** `Profile.state = Pending(d)` ⇒ Profile contributes
//!    `STRUCTURE` to `d.current_prefix.watch_demand`.
//!
//! The two sides must stay synchronised: the per-Resource `events_union`
//! recompute in [`crate::refcounts::sub_watch_demand`] reads the Profile
//! fields to attribute contributions, so the field MUST be cleared BEFORE
//! `sub_watch_demand` for the recompute to model the post-release union.
//! This module is the single source of truth for that discipline.
//!
//! Each helper is:
//! - **Idempotent.** Flag-already-cleared ⇒ no-op. Safe to call from any
//!   site without first checking the claim's presence.
//! - **Safe in any counter state.** Post-clamp slots
//!   (`clamp_watch_demand_to_zero` from `Input::WatchOpRejected`) and
//!   post-vacate slots both leave `watch_demand == 0`; the helper's
//!   `sub_watch_demand` short-circuits silently in that case (see
//!   [`crate::refcounts`] module rustdoc). The flag-clear runs
//!   regardless.

use crate::Engine;
use crate::reconcile::{delete_child, purge_per_file_fired_subs_for_reaped_slots};
use crate::refcounts::sub_watch_demand;
use specter_core::{AnchorClaim, ProbeOwner, ProfileId, ProfileState, StepOutput, TreeSnapshot};

impl Engine {
    /// Release the Profile's anchor `watch_demand` contribution if held.
    /// Idempotent (flag-false ⇒ no-op). Safe on a post-clamp counter
    /// (`watch_demand == 0` ⇒ `sub_watch_demand` short-circuits without
    /// emission; see the [`crate::refcounts`] module rustdoc).
    ///
    /// Does NOT call `try_reap` on the anchor — the Profile's own
    /// back-reference still anchors the slot. Callers that detach the
    /// Profile (e.g., `reap_profile`) try-reap the anchor afterwards.
    pub(crate) fn release_anchor_claim(&mut self, pid: ProfileId, out: &mut StepOutput) {
        let Some(p) = self.profiles.get(pid) else {
            return;
        };
        let AnchorClaim::Held = p.anchor_claim else {
            return;
        };
        let resource = p.resource;

        if let Some(p) = self.profiles.get_mut(pid) {
            p.anchor_claim = AnchorClaim::None;
        }

        sub_watch_demand(
            &mut self.tree,
            &self.profiles,
            &self.promoters,
            resource,
            None,
            out,
        );
    }

    /// Release the Profile's watch-root parent `watch_demand` contribution
    /// if held. Idempotent; safe in any counter state. Calls `try_reap`
    /// on the parent slot — the parent's `WatchRootParent` role is the
    /// only thing keeping it alive when no User Profile is anchored at
    /// or below it, and that's now stale.
    pub(crate) fn release_watch_root_parent_claim(&mut self, pid: ProfileId, out: &mut StepOutput) {
        let Some(p) = self.profiles.get(pid) else {
            return;
        };
        let Some(parent) = p.watch_root_parent else {
            return;
        };

        if let Some(p) = self.profiles.get_mut(pid) {
            p.watch_root_parent = None;
        }

        sub_watch_demand(
            &mut self.tree,
            &self.profiles,
            &self.promoters,
            parent,
            None,
            out,
        );

        self.tree.try_reap(parent);
    }

    /// Release the Profile's descent prefix `watch_demand` contribution if
    /// `Pending`. Transitions the Profile to `Idle`. Idempotent (non-Pending
    /// ⇒ no-op); safe in any counter state. Calls `try_reap` on the
    /// prefix slot — its `DescentScaffold` role is no longer
    /// load-bearing once no descent claims it.
    ///
    /// **Cancel-first contract.** Callers that may have an in-flight probe
    /// (e.g., `reap_profile`, `on_watch_op_rejected` descent purge) MUST
    /// invoke [`Engine::cancel_pending_probe`] before this helper. The
    /// debug_assert below catches any future regression: in release builds
    /// a missed cancel leaks one `ProbeOp::Cancel` emission, and the
    /// prober's eventual response is dropped as `StaleProbeResponse` —
    /// benign degradation, but worth surfacing loudly in dev / CI.
    pub(crate) fn release_descent_prefix_claim(&mut self, pid: ProfileId, out: &mut StepOutput) {
        let Some(prefix) = self
            .descent_state(ProbeOwner::Profile(pid))
            .map(|d| d.current_prefix)
        else {
            return;
        };

        debug_assert!(
            self.profiles
                .get(pid)
                .is_some_and(|p| p.pending_probe.is_none()),
            "release_descent_prefix_claim: probe channel must be closed before release; \
             caller must invoke cancel_pending_probe (or take the response-dispatch path) \
             first to avoid losing the Cancel emission (profile = {pid:?})",
        );

        if let Some(p) = self.profiles.get_mut(pid) {
            p.state = ProfileState::Idle;
        }

        sub_watch_demand(
            &mut self.tree,
            &self.profiles,
            &self.promoters,
            prefix,
            None,
            out,
        );

        self.tree.try_reap(prefix);
    }

    /// Release every per-descendant `watch_demand` contribution this
    /// Profile holds — the fourth member of the claim quartet, completing
    /// the symmetry with the three single-resource helpers above.
    ///
    /// **Take-and-walk.** Atomically takes `Profile.current` (sets to
    /// `None`), then walks the taken snapshot in reverse-lex order
    /// calling [`reconcile::delete_child`] on each top-level entry. The
    /// helper recurses leaf-before-parent and releases the per-slot
    /// `watch_demand` contribution with an explicit
    /// `releasing_descendant: Some(profile_id)` signal, so the recompute
    /// (multi-contributor case) skips this Profile's own descendant
    /// contribution even though `current` was still observable mid-walk
    /// (closes F-MED-4 by construction).
    ///
    /// **Idempotent.** `current.is_none()` ⇒ no-op. A second invocation
    /// in the same step finds `None` after the first call's `take`.
    /// Pending Profiles (no `current` by invariant) and File-anchored
    /// Profiles (`TreeSnapshot::File`, no descendants) short-circuit on
    /// the dispatch.
    ///
    /// **Safe in any counter state.** [`reconcile::delete_child`] calls
    /// `sub_watch_demand` unconditionally; the helper short-circuits
    /// silently on a zero counter (post-clamp slots, or slots a prior
    /// sub-walk in this take-and-walk pass already drained — see the
    /// [`crate::refcounts`] module rustdoc).
    ///
    /// **Per-file dedup hygiene.** The walk reaps covered Leaves; their
    /// `ResourceId`s may key entries in OTHER Profiles' (or this one's)
    /// `fired_subs` set. Mirror [`graft`]'s post-walk purge across the
    /// whole registry to drop the now-stale entries. Cross-Profile
    /// sharing makes the registry-wide scan necessary — a per-Profile
    /// purge would miss entries other Profiles wrote against the same
    /// descendant slot.
    ///
    /// **Sole call sites.** [`Engine::reap_profile`] and the seven
    /// `dispatch_*_vanished/failed` + `finalize_anchor_lost` sites in
    /// `transitions.rs`. Closes F-CRIT-1 by completing the four-claim
    /// release symmetry — every prior teardown path released the three
    /// 1-to-1 claims (anchor / watch-root parent / descent prefix) but
    /// left the 1-to-N descendant claims encoded in `Profile.current`
    /// stranded in the Tree.
    pub(crate) fn release_descendant_claim(&mut self, pid: ProfileId, out: &mut StepOutput) {
        // Take the snapshot atomically. Idempotent: subsequent calls
        // find `None` and short-circuit without further work.
        let taken = self.profiles.get_mut(pid).and_then(|p| p.current.take());
        let Some(snapshot) = taken else {
            return;
        };

        // File-anchored Profiles hold no descendant claims (a Leaf has
        // no descendants). The Dir arm is the only contributor.
        let TreeSnapshot::Dir(arc) = snapshot else {
            return;
        };

        // Dispatch the walk under a co-existing immutable borrow of the
        // Profile (for `delete_child`'s `&Profile` arg) and the
        // ProfileMap (for the recompute path's registry walk).
        // `&mut self.tree` is a disjoint-field borrow.
        {
            let Some(profile) = self.profiles.get(pid) else {
                return;
            };
            let anchor = profile.resource;
            // Reverse-lex per level — `delete_child` handles its own
            // internal reverse-lex within Dir children; this loop
            // covers the top-level entries of the snapshot. Together
            // they yield strict leaf-before-parent reap order, so
            // `try_reap` sees a vacated child set when it processes
            // each parent.
            for (name, child) in arc.entries.iter().rev() {
                delete_child(
                    &mut self.tree,
                    &self.profiles,
                    &self.promoters,
                    profile,
                    pid,
                    anchor,
                    name.as_str(),
                    child,
                    out,
                );
            }
        }

        // Cross-Profile dedup hygiene: covered Leaves reaped above may
        // appear in any Profile's `fired_subs` set keyed at their
        // `ResourceId`. Mirrors graft's post-walk_pair purge.
        purge_per_file_fired_subs_for_reaped_slots(&mut self.profiles, &self.tree);
    }

    /// Discard every anchor-derived state when the anchor is lost or
    /// kernel-rejected. The Profile transitions to "Idle without
    /// anchor": no claim, no snapshot, no cached kind. Recovery flows
    /// exclusively through [`specter_core::Profile::watch_root_parent`]'s
    /// next `StructureChanged` → `Engine::start_pending_recovery` →
    /// descent → `Engine::dispatch_descent_ok` anchor branch (which
    /// re-classifies `kind` from the parent's directory listing).
    ///
    /// **Cleared.**
    /// - `Profile.current` — taken by [`Engine::release_descendant_claim`].
    /// - `Profile.baseline = None`.
    /// - `Profile.kind = None`. The anchor's on-disk shape may have
    ///   changed across the lost-recovered cycle; the cache must not
    ///   misroute the next Seed burst's probe-shape dispatch. With
    ///   `kind = None`, `start_seed_burst` falls through to its
    ///   `Subtree` arm — a kind-mismatched `Vanished` then routes
    ///   through the normal descent-recovery path in either direction
    ///   (`Some(File)` against a now-Dir slot is the path that would
    ///   otherwise misroute as `AnchorFile` and waste a round-trip).
    /// - `Profile.anchor_claim = AnchorClaim::None` — via
    ///   [`Engine::release_anchor_claim`].
    ///
    /// **Preserved — by design.**
    /// - `Profile.watch_root_parent` — the recovery channel. Releasing
    ///   it here would close auto-recovery on anchor reappearance;
    ///   only `reap_profile` and `on_watch_op_rejected`'s parent purge
    ///   clear it.
    /// - `Profile.fired_subs` — fire history survives anchor loss.
    ///   The post-recovery Seed-Ok consults
    ///   [`Engine::seed_drift_observed`] to decide whether to re-fire;
    ///   the SeedDrift filter narrows to the Subtree subset of
    ///   `fired_subs`. Clearing here would silently fail to re-fire
    ///   emitted-once Effects on every recovery.
    /// - All other fields (`parent_profile`, `events_union`,
    ///   `has_per_file_fds`, `config*`, `resource`, `reap_pending`,
    ///   `settle*`).
    ///
    /// **Captured then later cleared on recovery.**
    /// - `Profile.last_settled_hash_at_loss` — set from
    ///   `baseline.hash()` immediately before this helper clears
    ///   `baseline` (via [`specter_core::Profile::capture_witness_at_loss`]).
    ///   The witness substitutes for the now-cleared `baseline.hash()`
    ///   in the next Seed-Ok's drift verdict; both branches of
    ///   `dispatch_seed_ok` and `dispatch_rebase_ok` call
    ///   [`specter_core::Profile::rebase_baseline`], which clears it on
    ///   consume. The cross-field invariant
    ///   `baseline.is_some() ⇒ last_settled_hash_at_loss.is_none()`
    ///   holds at every step boundary outside this helper's lifetime.
    ///
    /// **Pre-condition.** The probe channel must already be closed.
    /// Callers either took the response-dispatch path (which closes
    /// the channel before any dispatch arm runs, see
    /// `on_probe_response`) or invoked [`Engine::cancel_pending_probe`]
    /// first (`finalize_anchor_lost`'s pattern). The helper does not
    /// call `cancel_pending_probe` itself — matches the
    /// `release_*_claim` cancel-first contract.
    ///
    /// **Idempotence.** Each step short-circuits on already-cleared
    /// state: `release_descendant_claim` finds `current.is_none()` and
    /// returns; `baseline = None` and `kind = None` are no-ops against
    /// already-`None` fields; `release_anchor_claim` sees
    /// `AnchorClaim::None` and short-circuits.
    ///
    /// **Safe in any counter state.** Inherits from
    /// [`Engine::release_anchor_claim`]'s post-clamp tolerance —
    /// `sub_watch_demand` short-circuits silently on a zero counter
    /// (`clamp_watch_demand_to_zero` from `Input::WatchOpRejected` is
    /// the dominant source of this state).
    ///
    /// **Snapshot-shape invariant.**
    /// [`specter_core::Profile::kind`]'s rustdoc pins
    /// `current = Some(File) ⇒ kind == Some(File)` and
    /// `current = Some(Dir) ⇒ kind == Some(Dir)`. The helper preserves
    /// the invariant by clearing both atomically: `current` is taken
    /// in step 1, `kind = None` in step 2, before any reader can
    /// observe an intermediate state (the helper runs synchronously
    /// inside one `Engine::step` under `&mut self`).
    ///
    /// **Sole call sites.** The seven `dispatch_*_vanished/failed` +
    /// `finalize_anchor_lost` sites in `transitions.rs`. **Not** called
    /// by [`Engine::reap_profile`] — the reap path performs the same
    /// two release calls inline rather than via this helper. "Profile
    /// dies" has no next Seed burst, so the `kind` and `baseline`
    /// writes would be wasted on a struct about to drop; see
    /// `reap_profile`'s rustdoc for the asymmetry rationale.
    pub(crate) fn discard_anchor_state(&mut self, pid: ProfileId, out: &mut StepOutput) {
        // Order:
        //   1. release_descendant_claim runs first — it `take()`s
        //      `current`. The descendant walk and its per-slot
        //      recompute need the snapshot, and downstream recomputes
        //      (including release_anchor_claim's `events_union` walk)
        //      must see the post-take world with this Profile's
        //      descendant contributions already gone.
        //   2. Field clears (`baseline`, `kind`) are pure
        //      Profile-state writes; no Tree-side recompute reads
        //      them.
        //   3. release_anchor_claim runs last so its recompute walks
        //      a fully-cleared Profile.
        self.release_descendant_claim(pid, out);

        if let Some(p) = self.profiles.get_mut(pid) {
            p.capture_witness_at_loss();
            p.baseline = None;
            // `current` is already None — release_descendant_claim took it.
            p.kind = None;
        }

        self.release_anchor_claim(pid, out);
    }
}

#[cfg(test)]
#[path = "claims_tests.rs"]
mod tests;
