//! Profile-claim release helpers.
//!
//! A Profile holds at most three Resource-side claims, each with a
//! per-Profile bookkeeping field and a per-Resource refcount contribution:
//!
//! 1. **Anchor.** `Profile.anchor_contribution = true` ⇒ Profile contributes
//!    `Profile.events_union` to `Profile.resource.watch_demand`.
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
//! - **Counter-aware.** `tree.get(r).watch_demand == 0` ⇒ skip
//!   `sub_watch_demand` (the counter has already been zeroed by an
//!   out-of-band path, e.g., `clamp_watch_demand_to_zero` from
//!   `Input::WatchOpRejected`). Only the Profile flag clears.
//!
//! The counter-existence check is load-bearing: without it, calling a
//! helper post-clamp would underflow `sub_watch_demand`'s
//! `debug_assert!(prev > 0)`. With it, the helper is safe in any state.

use crate::Engine;
use crate::refcounts::sub_watch_demand;
use specter_core::{ClassSet, ProfileId, ProfileState, StepOutput};

impl Engine {
    /// Release the Profile's anchor `watch_demand` contribution if held.
    /// Idempotent (flag-false ⇒ no-op). Counter-aware (counter==0 ⇒ flag
    /// clears only, no underflow).
    ///
    /// Does NOT call `try_reap` on the anchor — the Profile's own
    /// back-reference still anchors the slot. Callers that detach the
    /// Profile (e.g., `reap_profile`) try-reap the anchor afterwards.
    pub(crate) fn release_anchor_claim(&mut self, pid: ProfileId, out: &mut StepOutput) {
        let Some(p) = self.profiles.get(pid) else {
            return;
        };
        if !p.anchor_contribution {
            return;
        }
        let resource = p.resource;
        let mask = p.events_union;

        if let Some(p) = self.profiles.get_mut(pid) {
            p.anchor_contribution = false;
        }

        if self.tree.get(resource).is_some_and(|r| r.watch_demand > 0) {
            sub_watch_demand(&mut self.tree, &self.profiles, resource, mask, out);
        }
    }

    /// Release the Profile's watch-root parent `watch_demand` contribution
    /// if held. Idempotent. Counter-aware. Calls `try_reap` on the parent
    /// slot — the parent's `WatchRootParent` role is the only thing
    /// keeping it alive when no User Profile is anchored at or below it,
    /// and that's now stale.
    pub(crate) fn release_watch_root_parent_claim(
        &mut self,
        pid: ProfileId,
        out: &mut StepOutput,
    ) {
        let Some(p) = self.profiles.get(pid) else {
            return;
        };
        let Some(parent) = p.watch_root_parent else {
            return;
        };

        if let Some(p) = self.profiles.get_mut(pid) {
            p.watch_root_parent = None;
        }

        if self.tree.get(parent).is_some_and(|r| r.watch_demand > 0) {
            sub_watch_demand(&mut self.tree, &self.profiles, parent, ClassSet::STRUCTURE, out);
        }

        self.tree.try_reap(parent);
    }

    /// Release the Profile's descent prefix `watch_demand` contribution if
    /// `Pending`. Transitions the Profile to `Idle`. Idempotent (non-Pending
    /// ⇒ no-op). Counter-aware. Calls `try_reap` on the prefix slot — its
    /// `DescentScaffold` role is no longer load-bearing once no descent
    /// claims it.
    ///
    /// Note: the descent's `probe_correlation`, if any, is dropped along
    /// with the state transition. Callers that need to cancel an in-flight
    /// probe (e.g., `reap_profile`, `on_watch_op_rejected`) MUST emit
    /// `ProbeOp::Cancel` before calling this helper.
    pub(crate) fn release_descent_prefix_claim(&mut self, pid: ProfileId, out: &mut StepOutput) {
        let Some(prefix) = self.descent_state(pid).map(|d| d.current_prefix) else {
            return;
        };

        if let Some(p) = self.profiles.get_mut(pid) {
            p.state = ProfileState::Idle;
        }

        if self.tree.get(prefix).is_some_and(|r| r.watch_demand > 0) {
            sub_watch_demand(&mut self.tree, &self.profiles, prefix, ClassSet::STRUCTURE, out);
        }

        self.tree.try_reap(prefix);
    }
}
