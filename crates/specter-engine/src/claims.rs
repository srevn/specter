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
use crate::reconcile::{delete_child, purge_per_file_dedup_for_reaped_slots};
use crate::refcounts::sub_watch_demand;
use specter_core::{ClassSet, ProfileId, ProfileState, StepOutput, TreeSnapshot};

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
            sub_watch_demand(&mut self.tree, &self.profiles, resource, mask, None, out);
        }
    }

    /// Release the Profile's watch-root parent `watch_demand` contribution
    /// if held. Idempotent. Counter-aware. Calls `try_reap` on the parent
    /// slot — the parent's `WatchRootParent` role is the only thing
    /// keeping it alive when no User Profile is anchored at or below it,
    /// and that's now stale.
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

        if self.tree.get(parent).is_some_and(|r| r.watch_demand > 0) {
            sub_watch_demand(
                &mut self.tree,
                &self.profiles,
                parent,
                ClassSet::STRUCTURE,
                None,
                out,
            );
        }

        self.tree.try_reap(parent);
    }

    /// Release the Profile's descent prefix `watch_demand` contribution if
    /// `Pending`. Transitions the Profile to `Idle`. Idempotent (non-Pending
    /// ⇒ no-op). Counter-aware. Calls `try_reap` on the prefix slot — its
    /// `DescentScaffold` role is no longer load-bearing once no descent
    /// claims it.
    ///
    /// **Cancel-first contract.** Callers that may have an in-flight probe
    /// (e.g., `reap_profile`, `on_watch_op_rejected` descent purge) MUST
    /// invoke [`Engine::cancel_pending_probe`] before this helper. The
    /// debug_assert below catches any future regression: in release builds
    /// a missed cancel leaks one `ProbeOp::Cancel` emission, and the
    /// prober's eventual response is dropped as `StaleProbeResponse` —
    /// benign degradation, but worth surfacing loudly in dev / CI.
    pub(crate) fn release_descent_prefix_claim(&mut self, pid: ProfileId, out: &mut StepOutput) {
        let Some(prefix) = self.descent_state(pid).map(|d| d.current_prefix) else {
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

        if self.tree.get(prefix).is_some_and(|r| r.watch_demand > 0) {
            sub_watch_demand(
                &mut self.tree,
                &self.profiles,
                prefix,
                ClassSet::STRUCTURE,
                None,
                out,
            );
        }

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
    /// **Counter-aware.** [`reconcile::delete_child`] gates each
    /// `sub_watch_demand` on `tree.get(r).watch_demand > 0`, so
    /// post-clamp slots (zero counter) skip without underflow.
    ///
    /// **Per-file dedup hygiene.** The walk reaps covered Leaves; their
    /// `ResourceId`s may key entries in OTHER Profiles' (or this one's)
    /// `last_emitted_dir_hash` map. Mirror [`graft`]'s post-walk purge
    /// across the whole registry to drop the now-stale entries.
    /// Cross-Profile sharing makes the registry-wide scan necessary —
    /// a per-Profile purge would miss entries other Profiles wrote
    /// against the same descendant slot.
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
        let taken = self
            .profiles
            .get_mut(pid)
            .and_then(|p| p.current.take());
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
        // appear in any Profile's `last_emitted_dir_hash` keyed at
        // their `ResourceId`. Mirrors graft's post-walk_pair purge.
        purge_per_file_dedup_for_reaped_slots(&mut self.profiles, &self.tree);
    }
}
