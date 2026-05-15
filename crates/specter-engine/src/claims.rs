//! Profile-claim release helpers.
//!
//! A Profile holds at most four Resource-side claims, each keyed by a
//! distinct [`ContribKey`] variant in the per-Resource contributions
//! map (`specter-core/resource.rs`):
//!
//! 1. **Anchor.** `Profile.anchor_claim == AnchorClaim::Held` ⇒ the
//!    Profile contributes [`ContribKey::ProfileAnchor`] at
//!    `Profile.resource` with mask `Profile.events_union`.
//! 2. **Watch-root parent.** `Profile.watch_root_parent =
//!    Some(parent)` ⇒ the Profile contributes
//!    [`ContribKey::ProfileParent`] at `parent` with mask `STRUCTURE`.
//! 3. **Descent prefix.** `Profile.state = Pending(d)` ⇒ the Profile
//!    contributes [`ContribKey::ProfileDescent`] at
//!    `d.current_prefix` with mask `STRUCTURE`.
//! 4. **Covered descendants.** Maintained per-slot inside `walk_pair` /
//!    `release_descendant_claim`; each contribution is keyed by
//!    [`ContribKey::ProfileDescendant`].
//!
//! The contribution map is the source of truth for refcounting;
//! removal is by key, not by registry walk. The per-Profile state
//! field (the matching flag from list above) can be cleared in either
//! order relative to `sub_watch`. This module clears the flag *first*
//! for consistency with the pre-existing call ordering and so that
//! subsequent helpers reading owner state see the post-release shape.
//!
//! Each helper is:
//! - **Idempotent.** Flag-already-cleared ⇒ no-op. Safe to call from any
//!   site without first checking the claim's presence.
//! - **Safe in any post-vacate state.** [`crate::refcounts::sub_watch`]
//!   silently skips an absent key — reachable after
//!   [`specter_core::Tree::vacate`] cleared the map.

use crate::Engine;
use crate::reconcile::{apply_diff_to_tree, purge_per_file_fired_subs_for_resources};
use crate::refcounts::{sub_watch, sub_watch_then_try_reap};
use specter_core::{
    AnchorClaim, ContribKey, DescentState, Diff, ProbeOwner, ProfileId, ProfileState, StepOutput,
    TreeSnapshot,
};

impl Engine {
    /// Release the Profile's anchor contribution if held. Idempotent
    /// (flag-false ⇒ no-op). Safe on a post-vacate slot —
    /// [`crate::refcounts::sub_watch`] silently skips an absent key
    /// (see the [`crate::refcounts`] module rustdoc).
    ///
    /// Does NOT call `try_reap` on the anchor — the Profile's own
    /// back-reference still anchors the slot. Callers that detach the
    /// Profile (e.g., `reap_profile`) try-reap the anchor afterwards.
    pub(crate) fn release_anchor_claim(&mut self, pid: ProfileId, out: &mut StepOutput) {
        let Some(p) = self.profiles.get(pid) else {
            return;
        };
        let AnchorClaim::Held = p.anchor_claim() else {
            return;
        };
        let resource = p.resource;

        if let Some(p) = self.profiles.get_mut(pid) {
            p.release_anchor_claim_now();
        }

        sub_watch(
            &mut self.tree,
            resource,
            ContribKey::ProfileAnchor(pid),
            out,
        );
    }

    /// Release the Profile's watch-root parent contribution if held.
    /// Idempotent; safe in any post-vacate state. Calls `try_reap` on
    /// the parent slot — with this Profile's [`ContribKey::ProfileParent`]
    /// just removed, the slot reaps unless some other claim still
    /// holds it (a sibling child, another Profile parented here, a
    /// Promoter proxy / prefix). The reap is a no-op at the call moment
    /// when [`Engine::reap_profile`] runs this helper before the
    /// anchor's own `try_reap` — the anchor is still a child of the
    /// parent — but the cascading `try_reap` performed by [`Tree::try_reap`]
    /// on the eventual anchor reap walks back up and frees the parent
    /// in that same step.
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

        sub_watch_then_try_reap(&mut self.tree, parent, ContribKey::ProfileParent(pid), out);
    }

    /// Release the Profile's descent prefix `watch_demand` contribution if
    /// `Pending`. Transitions the Profile to `Idle`. Idempotent (non-Pending
    /// ⇒ no-op); safe in any counter state. Calls `try_reap` on the
    /// prefix slot — with this Profile's
    /// [`ContribKey::ProfileDescent`] just removed, the slot reaps
    /// unless something else still claims it (most often a child slot
    /// in the descent chain toward the anchor, or another descent's
    /// contribution at the shared prefix). The prefix's role tag
    /// (`DescentScaffold` from initial `ensure_path`, or `User` /
    /// `WatchRootParent` if a peer Profile previously promoted it) is
    /// metadata; it does not affect this reap.
    ///
    /// **Cancel-first contract.** Callers that may have an in-flight probe
    /// (e.g., `reap_profile`, `on_watch_op_rejected` descent purge) MUST
    /// invoke [`Engine::cancel_owner_probe`] before this helper. The
    /// debug_assert below catches any future regression: in release builds
    /// a missed cancel leaks one `ProbeOp::Cancel` emission, and the
    /// prober's eventual response is dropped as `StaleProbeResponse` —
    /// benign degradation, but worth surfacing loudly in dev / CI.
    pub(crate) fn release_descent_prefix_claim(&mut self, pid: ProfileId, out: &mut StepOutput) {
        let Some(prefix) = self
            .descent_state(ProbeOwner::Profile(pid))
            .map(DescentState::current_prefix)
        else {
            return;
        };

        debug_assert!(
            self.probe_channel
                .correlation_for(ProbeOwner::Profile(pid))
                .is_none(),
            "release_descent_prefix_claim: probe channel must be closed before release; \
             caller must invoke cancel_owner_probe (or take the response-dispatch path) \
             first to avoid losing the Cancel emission (profile = {pid:?})",
        );

        if let Some(p) = self.profiles.get_mut(pid) {
            p.transition_state(ProfileState::Idle);
        }

        sub_watch_then_try_reap(&mut self.tree, prefix, ContribKey::ProfileDescent(pid), out);
    }

    /// Release every per-descendant contribution this Profile holds —
    /// the fourth member of the claim quartet, completing the
    /// symmetry with the three single-resource helpers above.
    ///
    /// **Take-and-apply.** Atomically takes `Profile.current` (sets to
    /// `None`), synthesises a wholesale-deletion [`Diff`] over the
    /// taken snapshot via [`Diff::all_deleted`], and feeds it to
    /// [`crate::reconcile::apply_diff_to_tree`] (which releases each
    /// slot's [`ContribKey::ProfileDescendant`] contribution by
    /// explicit key, then vacates and reaps any slot left with no
    /// remaining anchors).
    ///
    /// **Idempotent.** `current.is_none()` ⇒ no-op. A second invocation
    /// in the same step finds `None` after the first call's `take`.
    /// Pending Profiles (no `current` by invariant) and File-anchored
    /// Profiles (`TreeSnapshot::File`, no descendants) short-circuit on
    /// the dispatch.
    ///
    /// **Safe in any post-vacate state.**
    /// [`crate::reconcile::apply_diff_to_tree`] calls
    /// [`crate::refcounts::sub_watch`] unconditionally; the helper
    /// silently skips absent keys (post-vacate slots, or slots a prior
    /// sub-walk in this take-and-apply pass already drained — see the
    /// [`crate::refcounts`] module rustdoc).
    ///
    /// **Per-file dedup hygiene.** The Diff-driven pass reaps covered
    /// Leaves; their `ResourceId`s may key entries in OTHER Profiles'
    /// (or this one's) `fired_subs` set. Mirror [`graft`](crate::reconcile::graft)'s
    /// post-apply purge via the scoped
    /// [`crate::reconcile::purge_per_file_fired_subs_for_resources`].
    /// Cross-Profile sharing means the loop iterates every Profile;
    /// the membership check is scoped to the reaped set.
    ///
    /// **Sole call sites.** [`Engine::reap_profile`] and the seven
    /// `dispatch_*_vanished/failed` + `finalize_anchor_lost` sites in
    /// `transitions.rs`. Completes the four-claim release symmetry:
    /// the three 1-to-1 claims (anchor / watch-root parent / descent
    /// prefix) plus the 1-to-N descendant claims encoded in
    /// `Profile.current`.
    pub(crate) fn release_descendant_claim(&mut self, pid: ProfileId, out: &mut StepOutput) {
        // Take the snapshot atomically. Idempotent: subsequent calls
        // find `None` and short-circuit without further work.
        let taken = self
            .profiles
            .get_mut(pid)
            .and_then(specter_core::Profile::take_current);
        let Some(snapshot) = taken else {
            return;
        };

        // File-anchored Profiles hold no descendant claims (a Leaf has
        // no descendants). The Dir arm is the only contributor.
        let TreeSnapshot::Dir(arc) = snapshot else {
            return;
        };

        // Synthesise the wholesale-deletion Diff outside the Profile
        // borrow scope — `Diff::all_deleted` reads only the snapshot
        // and is `&self` on the Diff side.
        let diff = Diff::all_deleted(&arc);

        // Apply the Diff under a co-existing immutable borrow of the
        // Profile (for `apply_diff_to_tree`'s `&Profile` arg).
        // `&mut self.tree` is a disjoint-field borrow.
        let reaped = {
            let Some(profile) = self.profiles.get(pid) else {
                return;
            };
            let anchor = profile.resource;
            apply_diff_to_tree(&diff, profile, pid, anchor, &mut self.tree, out)
        };

        // Cross-Profile dedup hygiene: covered Leaves reaped above may
        // appear in any Profile's `fired_subs` set keyed at their
        // `ResourceId`. Scoped to the small reaped set, but iterates
        // every Profile to handle cross-Profile sharing.
        if !reaped.is_empty() {
            purge_per_file_fired_subs_for_resources(&mut self.profiles, &reaped);
        }
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
    /// - `Profile.current` — taken by [`Engine::release_descendant_claim`]
    ///   before this helper runs; the atomic clear below is a no-op
    ///   backstop over the already-`None` field.
    /// - `(Profile.baseline, Profile.current, Profile.kind)` — nulled
    ///   atomically by
    ///   [`specter_core::Profile::clear_anchor_classification`] (survival
    ///   witness captured first; see below). `kind` resets because the
    ///   anchor's on-disk shape may have changed across the
    ///   lost-recovered cycle; the cache must not misroute the next Seed
    ///   burst's probe-shape dispatch. With `kind = None`,
    ///   `start_seed_burst` falls through to its `Subtree` arm — a
    ///   kind-mismatched `Vanished` then routes through the normal
    ///   descent-recovery path in either direction (`Some(File)` against
    ///   a now-Dir slot is the path that would otherwise misroute as
    ///   `AnchorFile` and waste a round-trip).
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
    ///   `has_per_file_fds`, `config*`, `resource`, `settle*`). The
    ///   prior `reap_pending: bool` field is gone; the deferred-reap
    ///   directive now rides on `ProfileState::Active`'s payload via
    ///   [`specter_core::BurstFinish`], so its preservation across
    ///   recovery is part of `state`'s preservation (the helper does
    ///   not write `state`).
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
    /// `on_probe_response`) or invoked [`Engine::cancel_owner_probe`]
    /// first (`finalize_anchor_lost`'s pattern). The helper does not
    /// call `cancel_owner_probe` itself — matches the
    /// `release_*_claim` cancel-first contract.
    ///
    /// **Idempotence.** Each step short-circuits on already-cleared
    /// state: `release_descendant_claim` finds `current.is_none()` and
    /// returns; `baseline = None` and `kind = None` are no-ops against
    /// already-`None` fields; `release_anchor_claim` sees
    /// `AnchorClaim::None` and short-circuits.
    ///
    /// **Safe in any post-vacate state.** Inherits from
    /// [`Engine::release_anchor_claim`]'s tolerance —
    /// [`crate::refcounts::sub_watch`] silently skips an absent key
    /// ([`specter_core::Tree::vacate`] from `Input::WatchOpRejected`
    /// is the dominant source of this state).
    ///
    /// **Snapshot-shape invariant.**
    /// [`specter_core::Profile::kind`]'s rustdoc pins
    /// `current = Some(File) ⇒ kind == Some(File)` and
    /// `current = Some(Dir) ⇒ kind == Some(Dir)`. The invariant holds
    /// because the `(baseline, current, kind)` triple is cleared
    /// atomically by [`specter_core::Profile::clear_anchor_classification`]
    /// in step 2 (after `current`'s take in step 1), before any reader
    /// can observe an intermediate state (the helper runs synchronously
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
        //   2. clear_anchor_classification atomically captures the
        //      survival witness then nulls the `(baseline, current,
        //      kind)` triple — pure Profile-state writes; no Tree-side
        //      recompute reads them. `current` is already `None` (step
        //      1 took it); the helper's write is a no-op backstop.
        //   3. release_anchor_claim runs last so its recompute walks
        //      a fully-cleared Profile.
        self.release_descendant_claim(pid, out);

        if let Some(p) = self.profiles.get_mut(pid) {
            p.clear_anchor_classification();
        }

        self.release_anchor_claim(pid, out);
    }
}

#[cfg(test)]
#[path = "claims_tests.rs"]
mod tests;
