//! Profile-claim release helpers.
//!
//! A Profile holds at most four Resource-side claims, each keyed by a distinct [`ContribKey`]
//! variant in the per-Resource contributions map (`specter-core/resource.rs`):
//!
//! 1. **Anchor.** `Profile.anchor_claim == AnchorClaim::Held` ⇒ the Profile contributes
//!    [`ContribKey::ProfileAnchor`] at `Profile.resource` with mask `Profile.events`.
//! 2. **Watch-root parent.** `Profile.watch_root_parent = Some(parent)` ⇒ the Profile contributes
//!    [`ContribKey::ProfileParent`] at `parent` with mask `STRUCTURE`.
//! 3. **Descent prefix.** `Profile.state = Pending(d)` ⇒ the Profile contributes
//!    [`ContribKey::ProfileDescent`] at `d.current_prefix` with mask `STRUCTURE`.
//! 4. **Covered descendants.** Maintained per-slot inside [`crate::reconcile::apply_diff_to_tree`]
//!    (reached via `graft` on a probe response, or `release_descendant_claim` on teardown); each
//!    contribution is keyed by [`ContribKey::ProfileDescendant`].
//!
//! The contribution map is the source of truth for refcounting; removal is by key, not by registry
//! walk. The per-Profile state field (the matching flag from list above) can be cleared in either
//! order relative to `sub_watch`. This module clears the flag *first* for consistency with the
//! pre-existing call ordering and so that subsequent helpers reading owner state see the
//! post-release shape.
//!
//! Each helper is:
//! - **Idempotent.** Flag-already-cleared ⇒ no-op. Safe to call from any site without first
//!   checking the claim's presence.
//! - **Safe in any post-vacate state.** [`crate::refcounts::sub_watch`] silently skips an absent
//!   key — reachable after [`specter_core::Tree::vacate`] cleared the map.

use crate::Engine;
use crate::reconcile::apply_diff_to_tree;
use crate::refcounts::{sub_watch, sub_watch_then_try_reap};
use crate::transitions::ParkNarration;
use specter_core::{
    AnchorClaim, ContribKey, DescentState, Diff, ProfileId, StepOutput, TreeSnapshot,
};

impl Engine {
    /// Release the Profile's anchor contribution if held. Idempotent (flag-false ⇒ no-op). Safe on
    /// a post-vacate slot — [`crate::refcounts::sub_watch`] silently skips an absent key (see the
    /// [`crate::refcounts`] module rustdoc).
    ///
    /// Does NOT call `try_reap` on the anchor — the Profile's own back-reference still anchors the
    /// slot. Callers that detach the Profile (e.g., `reap_profile`) try-reap the anchor afterwards.
    pub(crate) fn release_anchor_claim(&mut self, pid: ProfileId, out: &mut StepOutput) {
        let Some(p) = self.profiles.get(pid) else {
            return;
        };
        let AnchorClaim::Held = p.anchor_claim() else {
            return;
        };
        let resource = p.resource();

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

    /// Release the Profile's watch-root parent contribution if held. Idempotent; safe in any
    /// post-vacate state. Calls `try_reap` on the parent slot — with this Profile's
    /// [`ContribKey::ProfileParent`] just removed, the slot reaps unless some other claim still
    /// holds it (a sibling child, another Profile parented here). The reap is a no-op at the call
    /// moment when [`Engine::reap_profile`] runs this helper before the anchor's own `try_reap` —
    /// the anchor is still a child of the parent — but the cascading `try_reap` performed by
    /// [`specter_core::Tree::try_reap`] on the eventual anchor reap walks back up and frees the
    /// parent in that same step.
    pub(crate) fn release_watch_root_parent_claim(&mut self, pid: ProfileId, out: &mut StepOutput) {
        // `take_watch_root_parent` reads and clears the cached id in one move, so the
        // read-then-null pair collapses to a single `get_mut` (was a `get` for the presence check,
        // then a `get_mut` to null it).
        let Some(parent) = self
            .profiles
            .get_mut(pid)
            .and_then(specter_core::Profile::take_watch_root_parent)
        else {
            return;
        };

        sub_watch_then_try_reap(&mut self.tree, parent, ContribKey::ProfileParent(pid), out);
    }

    /// Release the Profile's descent prefix `watch_demand` contribution if `Pending`. Parks the
    /// Profile through [`Engine::park_profile`] — a torn-down descent is an anchorless terminal,
    /// and the park's typed state keeps every `Idle` consumer (overflow re-Seed, attach join, burst
    /// routing) from mistaking it for a healthy rest; `narration` distinguishes the operational
    /// teardowns (watch rejection, root-vanish abandon, walker-contract abandon — each narrating
    /// one [`specter_core::Diagnostic::ProfileParked`] inside the helper, so the three callers
    /// cannot drift apart) from `reap_profile`'s same-step teardown (silent — the reap narrates
    /// instead). Idempotent (non-Pending ⇒ no-op); safe in any counter state. Calls `try_reap` on
    /// the prefix slot — with this Profile's [`ContribKey::ProfileDescent`] just removed, the slot
    /// reaps unless something else still claims it (most often a child slot in the descent chain
    /// toward the anchor, or another descent's contribution at the shared prefix). The prefix's
    /// role tag (`DescentScaffold` from initial `ensure_path`, or `User` / `WatchRootParent` if a
    /// peer Profile previously promoted it) is metadata; it does not affect this reap.
    ///
    /// **Cancel-first contract.** Callers that may have an in-flight probe (e.g., `reap_profile`,
    /// `on_watch_op_rejected` descent purge) MUST invoke [`Engine::cancel_owner_probe`] before this
    /// helper. `ProbeSlot`'s Drop tripwire enforces this structurally: the park transition below
    /// drops the prior `Pending(DescentState)`, and an armed descent slot reaching that drop panics
    /// in every build — its orphaned correlation would otherwise stale-detect its own response. The
    /// discard *is* the enforcement site; no local witness is needed.
    pub(crate) fn release_descent_prefix_claim(
        &mut self,
        pid: ProfileId,
        narration: ParkNarration,
        out: &mut StepOutput,
    ) {
        let Some(prefix) = self.descent_state(pid).map(DescentState::current_prefix) else {
            return;
        };

        // The cancel-first contract is enforced here: the park's transition drops the prior
        // `Pending(DescentState)`; an armed descent slot trips `ProbeSlot`'s Drop tripwire.
        self.park_profile(pid, narration, out);

        sub_watch_then_try_reap(&mut self.tree, prefix, ContribKey::ProfileDescent(pid), out);
    }

    /// Release every per-descendant contribution this Profile holds — the fourth member of the
    /// claim quartet, completing the symmetry with the three single-resource helpers above.
    ///
    /// **Take-and-apply.** Atomically takes `Profile.current` (sets to `None`), synthesises a
    /// wholesale-deletion [`Diff`] over the taken snapshot via [`Diff::all_deleted`], and feeds it
    /// to [`crate::reconcile::apply_diff_to_tree`] (which releases each slot's
    /// [`ContribKey::ProfileDescendant`] contribution by explicit key, then vacates and reaps any
    /// slot left with no remaining anchors).
    ///
    /// **Idempotent.** `current.is_none()` ⇒ no-op. A second invocation in the same step finds
    /// `None` after the first call's `take`. Pending Profiles (no `current` by invariant) and
    /// File-anchored Profiles (`TreeSnapshot::File`, no descendants) short-circuit on the dispatch.
    ///
    /// **Safe in any post-vacate state.** [`crate::reconcile::apply_diff_to_tree`] calls
    /// [`crate::refcounts::sub_watch`] unconditionally; the helper silently skips absent keys
    /// (post-vacate slots, or slots a prior sub-walk in this take-and-apply pass already drained —
    /// see the [`crate::refcounts`] module rustdoc).
    ///
    /// **Sole call sites.** [`Engine::reap_profile`] and [`Engine::discard_anchor_state`] (reached
    /// through the `finalize_anchor_lost` coordinator in `transitions.rs`). Completes the
    /// four-claim release symmetry: the three 1-to-1 claims (anchor / watch-root parent / descent
    /// prefix) plus the 1-to-N descendant claims encoded in `Profile.current`.
    pub(crate) fn release_descendant_claim(&mut self, pid: ProfileId, out: &mut StepOutput) {
        // Take the snapshot atomically. Idempotent: subsequent calls find `None` and short-circuit
        // without further work.
        let taken = self
            .profiles
            .get_mut(pid)
            .and_then(specter_core::Profile::take_current);
        let Some(snapshot) = taken else {
            return;
        };

        // File-anchored Profiles hold no descendant claims (a Leaf has no descendants); the Dir arm
        // is the only contributor. A File anchor's loss→recovery survival witness is captured by
        // `discard_anchor_state` (via `clear_anchor_classification`), not here — this helper only
        // releases the 1-to-N Dir claims.
        let TreeSnapshot::Dir(arc) = snapshot else {
            return;
        };

        // Synthesise the wholesale-deletion Diff outside the Profile borrow scope —
        // `Diff::all_deleted` reads only the snapshot and is `&self` on the Diff side.
        let diff = Diff::all_deleted(&arc);

        // Apply the Diff under a scoped immutable Profile borrow (for `apply_diff_to_tree`'s
        // `&Profile` arg); `&mut self.tree` and `&mut self.coverage_scratch` are disjoint-field
        // borrows. Purely side-effecting — no reaped-slot return: per-Sub fire history dies with
        // its Sub, so a reaped leaf has nothing to purge by `ResourceId`.
        {
            let Some(profile) = self.profiles.get(pid) else {
                return;
            };
            let anchor = profile.resource();
            apply_diff_to_tree(
                &diff,
                profile,
                pid,
                anchor,
                &mut self.tree,
                out,
                &mut self.coverage_scratch,
            );
        }
    }

    /// Discard every anchor-derived state when the anchor is lost or kernel-rejected: no claim, no
    /// snapshot, no cached kind. Recovery is a descent at
    /// [`specter_core::Profile::watch_root_parent`] ending in `Engine::dispatch_descent_ok`'s
    /// anchor branch (which re-classifies `kind` from the parent's directory listing) — entered
    /// immediately by the observed-loss coordinator's descend wrapper
    /// (`Engine::finalize_anchor_lost_and_descend`), or, for the probe-`Failed` / watch-rejection
    /// terminals that park (`Engine::finalize_anchor_lost_and_park`), by a later recovery trigger →
    /// `Engine::start_pending_recovery`.
    ///
    /// **Cleared.**
    /// - The anchor classification (kind ⊕ live snapshot ⊕ settled baseline) collapses to
    ///   `Unclassified` via [`specter_core::Profile::clear_anchor_classification`], which captures
    ///   the survival witness in the same move (see below). [`Engine::release_descendant_claim`]
    ///   has already `take()`d the live `current` before this helper runs, so the collapse only has
    ///   the kind discriminant and settled reference left to reset. The kind must reset because the
    ///   anchor's on-disk shape may have changed across the lost→recovered cycle and a stale
    ///   discriminant would misroute the next Seed burst's probe-shape dispatch: `Unclassified`
    ///   makes `start_seed_burst` fall through to its `Subtree` arm, and a kind-mismatched
    ///   `Vanished` then routes through the normal descent-recovery path in either direction
    ///   (`Some(File)` against a now-Dir slot is the case that would otherwise misroute as
    ///   `AnchorFile` and waste a round-trip).
    /// - `Profile.anchor_claim = AnchorClaim::None` — via [`Engine::release_anchor_claim`].
    ///
    /// **Preserved — by design.**
    /// - `Profile.watch_root_parent` — the recovery channel. Releasing it here would close
    ///   auto-recovery on anchor reappearance; only `reap_profile` and `on_watch_op_rejected`'s
    ///   parent purge clear it.
    /// - The per-Sub fire history ([`specter_core::Sub::has_fired`]) — not a Profile field at all,
    ///   so it is outside this helper's reach by construction: it lives on the Subs in the
    ///   registry, which anchor loss does not touch, so it survives the loss→recovery window for
    ///   free. The post-recovery Seed-Ok consults [`Engine::seed_drift_observed`] (which reads that
    ///   per-Sub state) to decide whether to re-fire; were it cleared, emitted-once Effects would
    ///   silently fail to re-fire on every recovery.
    /// - All other fields (`events`, `has_per_file_fds`, `config*`, `resource`, `settle*`). The
    ///   deferred-reap directive rides on `ProfileState::Active`'s payload via
    ///   [`specter_core::BurstFinish`], so its preservation across recovery is part of `state`'s
    ///   preservation (the helper does not write `state`).
    ///
    /// **Captured here, consumed on recovery.**
    /// - The survival witness — `clear_anchor_classification` derives it from the settled reference's
    ///   hash and stores it in the collapsed `Unclassified` arm, substituting for the dropped
    ///   baseline in the next Seed-Ok's drift verdict ([`Engine::seed_drift_observed`] reads it via
    ///   [`specter_core::Profile::settled_hash`]). `dispatch_rebase_ok` and the Seed-Ok recovery pin
    ///   only (the `EmitMode::SeedDrift` seal in `fire_and_settle`, or the silent `SilentPin` arm of
    ///   `fire_or_seal`, reached from the [`specter_core::QuiescenceVerdict::Stable`] Seed verdicts —
    ///   both `Natural` and `Forced`) call [`specter_core::Profile::rebase_baseline`], which consumes
    ///   it (the `Witness → Snapshot` move); the Seed [`specter_core::QuiescenceVerdict::Retry`] arm
    ///   grafts (or skips) without rebasing, so the witness outlives an unbounded re-batch loop and
    ///   is consumed only at the eventual pin. A live baseline and a survival witness are mutually
    ///   exclusive *by construction* in the anchor sum — the old `baseline.is_some() ⇒ …is_none()`
    ///   rule is a type property now, not a step-boundary invariant.
    ///
    /// **Pre-condition.** The owner's probe slot must already be disarmed. The sole caller,
    /// `finalize_anchor_lost`, invokes [`Engine::cancel_owner_probe`] first (a no-op on the
    /// response-dispatch routes, whose slot `on_probe_response` already disarmed). The helper does
    /// not call `cancel_owner_probe` itself — matches the `release_*_claim` cancel-first contract.
    ///
    /// **Idempotence.** Each step short-circuits on already-cleared state: `release_descendant_claim`
    /// finds `current.is_none()` and returns; `clear_anchor_classification` on an
    /// already-`Unclassified` anchor preserves the carried witness rather than overwriting it;
    /// `release_anchor_claim` sees `AnchorClaim::None` and short-circuits.
    ///
    /// **Safe in any post-vacate state.** Inherits from [`Engine::release_anchor_claim`]'s tolerance
    /// — [`crate::refcounts::sub_watch`] silently skips an absent key ([`specter_core::Tree::vacate`]
    /// from `Input::WatchOpRejected` is the dominant source of this state).
    ///
    /// **Snapshot-shape coherence is structural.** The anchor sum's discriminant *is* the kind, so
    /// `current = Some(K) ⇒ kind == Some(K)` cannot be violated by any representable value — there
    /// is no separate kind/baseline/current triple to keep in agreement.
    /// [`specter_core::Profile::clear_anchor_classification`] (step 2) collapses the classification
    /// to `Unclassified` in one move; it runs synchronously inside one `Engine::step` under `&mut
    /// self`, so no reader observes an intermediate.
    ///
    /// **Sole call site.** `finalize_anchor_lost` in `transitions.rs` — the anchor-loss coordinator
    /// every observed-loss route (anchor-terminal event, the six probe vanished/failed dispatches,
    /// the kind-mismatch certifier arm, the watch-rejection purge) funnels through. **Not** called
    /// by [`Engine::reap_profile`] — the reap path performs the same two release calls inline
    /// rather than via this helper. "Profile dies" has no next Seed burst, so resetting the
    /// classification would be wasted on a struct about to drop; see `reap_profile`'s rustdoc for
    /// the asymmetry rationale.
    ///
    /// **No carrier-count bookkeeping.** `Profile::is_nonsteady` is a pure state predicate
    /// (`Pending ∨ Parked`); this coordinator writes no `state`, so the anchor clears below cannot
    /// move the count — the caller's eventual park / descent entry records its own edge through the
    /// `ProfileMap` chokepoints.
    pub(crate) fn discard_anchor_state(&mut self, pid: ProfileId, out: &mut StepOutput) {
        // Order:
        //   1. release_descendant_claim runs first — it `take()`s `current`. The descendant walk
        //      and its per-slot recompute need the snapshot, and downstream recomputes (including
        //      release_anchor_claim's `events_union` walk) must see the post-take world with this
        //      Profile's descendant contributions already gone.
        //   2. clear_anchor_classification collapses File/Dir → Unclassified, atomically capturing
        //      the survival witness from the settled reference. Step 1's take_current left
        //      `current` None but `settled` intact, so the witness is still available — pure
        //      Profile-state writes, no Tree-side recompute reads them.
        //   3. release_anchor_claim runs last so its recompute walks a fully-cleared Profile.
        self.release_descendant_claim(pid, out);

        if let Some(p) = self.profiles.get_mut(pid) {
            p.clear_anchor_classification();
        }

        self.release_anchor_claim(pid, out);

        // Coordinator-exit coherence tripwire, symmetric with `Profile::materialize_anchor`'s. The
        // classification collapse above is structural, but a future regression that reordered these
        // steps or left the Profile classified / still holding the anchor claim while `Pending`
        // would trip here at the write site rather than latently at the next dispatch or reap.
        if let Some(p) = self.profiles.get(pid) {
            p.debug_assert_anchor_coherent();
        }
    }
}

#[cfg(test)]
mod tests {
    //! Unit tests for `Engine::discard_anchor_state` — pins the helper's contract: which Profile
    //! fields are cleared, which are preserved, idempotence, post-vacate safety, and invariance of
    //! the lifetime-fixed fields (`events_union`, `has_per_file_fds`).
    //!
    //! Goes hand-in-hand with the per-site `dispatch_*_clears_profile_kind` assertions in
    //! `transitions_tests.rs`, which exercise the helper through each production call site.

    #![allow(
        clippy::items_after_statements,
        clippy::manual_let_else,
        clippy::missing_const_for_fn,
        clippy::needless_pass_by_value,
        clippy::too_many_lines
    )]

    use crate::Engine;
    use compact_str::CompactString;
    use specter_core::testkit::single_exec_program;
    use specter_core::{
        ActionProgram, AnchorClaim, ArgPart, ArgTemplate, ChildEntry, ClassSet, DirChild, DirMeta,
        DirSnapshot, EffectScope, EntryKind, FsIdentity, Input, LeafEntry, ProbeOutcome,
        ProbeResponse, ProfileId, ProofAuthority, ResourceId, ResourceKind, ResourceRole,
        ScanConfig, StepOutput, SubAttachAnchor, SubAttachRequest, SubId, WatchOp,
    };
    use std::collections::BTreeMap;
    use std::sync::Arc;
    use std::time::{Duration, Instant, UNIX_EPOCH};

    const SETTLE: Duration = Duration::from_millis(100);
    const MAX_SETTLE: Duration = Duration::from_secs(6);

    fn empty_program() -> Arc<ActionProgram> {
        single_exec_program([ArgTemplate::new([ArgPart::literal("/bin/true")])])
    }

    fn dir_snap(children: Vec<(&str, EntryKind, u64)>) -> Arc<DirSnapshot> {
        let mut map: BTreeMap<CompactString, ChildEntry> = BTreeMap::new();
        for (name, kind, inode) in children {
            let child = match kind {
                EntryKind::Dir => {
                    ChildEntry::Dir(DirChild::Uncovered(FsIdentity::synthetic(inode, 0)))
                }
                _ => ChildEntry::Leaf(LeafEntry::synthetic(
                    kind,
                    0,
                    UNIX_EPOCH,
                    FsIdentity::synthetic(inode, 0),
                )),
            };
            map.insert(CompactString::new(name), child);
        }
        Arc::new(DirSnapshot::new(
            DirMeta::synthetic(UNIX_EPOCH, FsIdentity::synthetic(0, 0)),
            0,
            map,
        ))
    }

    /// Drive a fresh-attach cold Seed burst from `Active(PreFire(Verifying))` through its
    /// quiescence verdict to pinned `Idle`, committing `snap` as `current` + `baseline`.
    ///
    /// The cold-arm Seed burst pins on the first `Authoritative` sample: a cold-Seed `SilentPin`
    /// consequence does not owe quiescence proof, so the witness is
    /// [`QuiescenceWitness::EventsReliable`] and the fold folds to `Stable(StableReason::Natural)`;
    /// dispatch reaches `SilentPin` (no fired Subs, no drift) and finishes to Idle. The cold-arm
    /// Verifying-first contract puts the probe in flight at burst construction, so this helper
    /// answers it directly — no settle expiry step.
    fn drive_fresh_seed_to_idle(
        e: &mut Engine,
        pid: ProfileId,
        snap: Arc<DirSnapshot>,
        t0: Instant,
    ) {
        let corr = e
            .pending_probe_for(pid)
            .expect("cold-arm Seed Verifying probe in flight at burst construction");
        e.step(
            Input::ProbeResponse(ProbeResponse {
                owner: pid,
                correlation: corr,
                outcome: ProbeOutcome::SubtreeProven {
                    snapshot: Arc::clone(&snap),
                    authority: ProofAuthority::Authoritative,
                },
            }),
            t0 + SETTLE,
        );
        assert!(
            matches!(
                e.profiles().get(pid).unwrap().state(),
                specter_core::ProfileState::Idle
            ),
            "two settle-spaced equal Seed samples pin the baseline → Idle",
        );
    }

    /// Build an Engine + a Profile materialised at `root`. Returns the `(SubId, ProfileId,
    /// anchor_id, parent_id)` tuple. The anchor sits under a parent slot so `watch_root_parent` is
    /// set; both are Dir; `events = ClassSet::EMPTY` keeps `has_per_file_fds = false`.
    fn engine_with_materialised_profile(
        events: ClassSet,
    ) -> (Engine, SubId, ProfileId, ResourceId, ResourceId) {
        let mut e = Engine::new();
        let parent = e.tree_mut().ensure_root("var", ResourceRole::User);
        e.tree_mut().set_kind(parent, ResourceKind::Dir);
        let anchor = e
            .tree_mut()
            .ensure_child(parent, "log", ResourceRole::User)
            .expect("test live parent");
        e.tree_mut().set_kind(anchor, ResourceKind::Dir);

        let req = SubAttachRequest::for_anchor(
            "watch".into(),
            SubAttachAnchor::Resource(anchor),
            ScanConfig::builder().recursive(true).build(),
            MAX_SETTLE,
            SETTLE,
            empty_program(),
            EffectScope::SubtreeRoot,
            events,
            false,
        );
        let t0 = Instant::now();
        let attach_out = e.step(Input::AttachSub(req), t0);
        let sid =
            specter_core::testkit::first_attached_sub(&attach_out).expect("attach_sub succeeded");
        let pid = e.subs().get(sid).unwrap().profile();

        // Drive the cold-arm Seed through its quiescence proof so `current` and `baseline` pin to
        // the empty-dir observation.
        drive_fresh_seed_to_idle(&mut e, pid, dir_snap(vec![]), t0);

        (e, sid, pid, anchor, parent)
    }

    #[test]
    fn discard_anchor_state_clears_kind_baseline_current_anchor_claim() {
        let (mut e, _sid, pid, _anchor, _parent) =
            engine_with_materialised_profile(ClassSet::EMPTY);

        // Pre-condition.
        {
            let p = e.profiles().get(pid).expect("Profile lives");
            assert_eq!(p.kind(), Some(ResourceKind::Dir));
            assert!(p.baseline().is_some());
            assert!(p.current().is_some());
            assert_eq!(p.anchor_claim(), AnchorClaim::Held);
        }

        let mut out = StepOutput::default();
        e.discard_anchor_state(pid, &mut out);

        let p = e.profiles().get(pid).expect("Profile lives");
        assert!(p.kind().is_none(), "kind cleared");
        assert!(p.baseline().is_none(), "baseline cleared");
        assert!(p.current().is_none(), "current taken by descendant release");
        assert_eq!(p.anchor_claim(), AnchorClaim::None, "anchor claim released");
    }

    #[test]
    fn discard_anchor_state_preserves_watch_root_parent() {
        let (mut e, _sid, pid, _anchor, parent) = engine_with_materialised_profile(ClassSet::EMPTY);
        assert_eq!(
            e.profiles().get(pid).unwrap().watch_root_parent(),
            Some(parent),
        );

        let mut out = StepOutput::default();
        e.discard_anchor_state(pid, &mut out);

        assert_eq!(
            e.profiles().get(pid).unwrap().watch_root_parent(),
            Some(parent),
            "recovery channel preserved across anchor loss",
        );
        // Parent's watch_demand still carries this Profile's STRUCTURE contribution — the recompute
        // walks covering Profiles, finds this one still claims the parent, and keeps the union.
        assert!(
            e.tree().get(parent).is_some_and(|r| r.watch_demand() >= 1),
            "parent watch_demand preserved",
        );
    }

    #[test]
    fn discard_anchor_state_carries_settled_hash_through_loss() {
        let (mut e, _sid, pid, _anchor, _parent) =
            engine_with_materialised_profile(ClassSet::EMPTY);

        let pre_loss_hash = e
            .profiles()
            .get(pid)
            .and_then(|p| p.baseline().map(|s| s.hash()))
            .expect("fixture must produce baseline");
        // Active mode: the settled reference *is* the live baseline — a separate survival witness
        // alongside a held baseline is not representable in the anchor sum.
        assert_eq!(
            e.profiles().get(pid).unwrap().settled_hash(),
            Some(pre_loss_hash),
            "active mode: settled reference is the live baseline hash",
        );

        let mut out = StepOutput::default();
        e.discard_anchor_state(pid, &mut out);

        let p = e.profiles().get(pid).unwrap();
        assert!(p.baseline().is_none(), "discard cleared the baseline");
        assert_eq!(
            p.settled_hash(),
            Some(pre_loss_hash),
            "the survival witness carries the pre-loss baseline hash through \
             the loss window so post-recovery drift still has a reference",
        );
    }

    #[test]
    fn discard_anchor_state_preserves_fired_subs() {
        // Negative-space contract: anchor loss does not clear fire history. The history is now
        // per-Sub (`Sub.has_fired`) and `discard_anchor_state` operates on the Profile only, so
        // survival across the loss window is structural — but the property still matters:
        // post-recovery drift must re-fire emitted-once Effects.
        let (mut e, sid, pid, _anchor, _parent) = engine_with_materialised_profile(ClassSet::EMPTY);
        e.subs.mark_fired(sid);
        assert!(
            e.subs.get(sid).is_some_and(specter_core::Sub::has_fired),
            "precondition: fire recorded",
        );

        let mut out = StepOutput::default();
        e.discard_anchor_state(pid, &mut out);

        assert!(
            e.subs.get(sid).is_some_and(specter_core::Sub::has_fired),
            "fire history survives anchor loss",
        );
    }

    #[test]
    fn discard_anchor_state_idempotent_preserves_witness() {
        let (mut e, _sid, pid, _anchor, _parent) =
            engine_with_materialised_profile(ClassSet::EMPTY);

        let mut out = StepOutput::default();
        e.discard_anchor_state(pid, &mut out);
        let witness_after_first = e.profiles().get(pid).unwrap().settled_hash();
        assert!(
            witness_after_first.is_some(),
            "first discard captures the survival witness",
        );

        let mut out2 = StepOutput::default();
        e.discard_anchor_state(pid, &mut out2);

        assert_eq!(
            e.profiles().get(pid).unwrap().settled_hash(),
            witness_after_first,
            "second discard against an already-Unclassified anchor preserves \
             the prior witness rather than overwriting it with None",
        );
    }

    #[test]
    fn discard_anchor_state_idempotent() {
        let (mut e, _sid, pid, _anchor, _parent) =
            engine_with_materialised_profile(ClassSet::EMPTY);
        let mut out = StepOutput::default();
        e.discard_anchor_state(pid, &mut out);

        let snap_after_first = {
            let p = e.profiles().get(pid).expect("Profile lives");
            (
                p.kind(),
                p.baseline().is_some(),
                p.current().is_some(),
                p.anchor_claim(),
            )
        };

        let mut out2 = StepOutput::default();
        e.discard_anchor_state(pid, &mut out2);

        let snap_after_second = {
            let p = e.profiles().get(pid).expect("Profile lives");
            (
                p.kind(),
                p.baseline().is_some(),
                p.current().is_some(),
                p.anchor_claim(),
            )
        };

        assert_eq!(
            snap_after_first, snap_after_second,
            "second invocation observes the same Profile state",
        );
        assert!(
            out2.watch_ops.is_empty() && out2.probe_ops().is_empty(),
            "second invocation emits no ops; got watch_ops={:?} probe_ops={:?}",
            out2.watch_ops,
            out2.probe_ops(),
        );
    }

    #[test]
    fn discard_anchor_state_safe_after_vacate() {
        // anchor contributions were cleared (e.g., by WatchOpRejected → `Tree::vacate`);
        // `release_anchor_claim`'s `sub_watch` must silently skip the absent
        // `ContribKey::ProfileAnchor(pid)` key and skip emitting a second Unwatch (vacate already
        // emitted one).
        let (mut e, _sid, pid, anchor, _parent) = engine_with_materialised_profile(ClassSet::EMPTY);

        // Capture the pre-vacate counter to make sure vacate actually fires.
        assert!(e.tree().get(anchor).is_some_and(|r| r.watch_demand() > 0));

        let mut vacate_out = StepOutput::default();
        e.tree_mut().vacate(anchor, &mut vacate_out);
        assert_eq!(
            e.tree()
                .get(anchor)
                .map_or(0, specter_core::Resource::watch_demand),
            0,
            "vacate zeroed the counter",
        );

        let mut out = StepOutput::default();
        e.discard_anchor_state(pid, &mut out);
        // Anchor's edge already fired during vacate; the helper must not emit a second Unwatch on
        // the post-vacate counter.
        assert!(
            !out.watch_ops
                .iter()
                .any(|op| matches!(op, WatchOp::Unwatch { resource } if *resource == anchor)),
            "no stray Unwatch on post-vacate anchor; got {:?}",
            out.watch_ops,
        );
        // Profile state still cleared correctly.
        let p = e.profiles().get(pid).expect("Profile lives");
        assert_eq!(p.anchor_claim(), AnchorClaim::None);
        assert!(p.kind().is_none());
    }

    #[test]
    fn discard_anchor_state_no_op_on_already_lost_profile() {
        let (mut e, _sid, pid, _anchor, _parent) =
            engine_with_materialised_profile(ClassSet::EMPTY);
        let mut first_out = StepOutput::default();
        e.discard_anchor_state(pid, &mut first_out);

        // Second call against a fully-cleared Profile — no ops, no diagnostics.
        let mut out = StepOutput::default();
        e.discard_anchor_state(pid, &mut out);

        assert!(
            out.watch_ops.is_empty(),
            "no watch ops; got {:?}",
            out.watch_ops
        );
        assert!(
            out.probe_ops().is_empty(),
            "no probe ops; got {:?}",
            out.probe_ops()
        );
        assert!(
            out.diagnostics.is_empty(),
            "no diagnostics; got {:?}",
            out.diagnostics,
        );
        assert!(out.effects().is_empty());
    }

    #[test]
    fn discard_anchor_state_preserves_events_union_and_per_file_fds() {
        // events_union and has_per_file_fds are invariant for the Profile's lifetime under the
        // events-folds-into-config_hash discipline; the helper must not touch them.
        let (mut e, _sid, pid, _anchor, _parent) =
            engine_with_materialised_profile(ClassSet::CONTENT);
        let (events_before, fds_before) = {
            let p = e.profiles().get(pid).expect("Profile lives");
            (p.events(), p.has_per_file_fds())
        };
        assert_eq!(events_before, ClassSet::CONTENT);
        assert!(fds_before, "CONTENT events ⇒ per-file FDs enabled");

        let mut out = StepOutput::default();
        e.discard_anchor_state(pid, &mut out);

        let p = e.profiles().get(pid).expect("Profile lives");
        assert_eq!(p.events(), events_before, "events_union invariant");
        assert_eq!(
            p.has_per_file_fds(),
            fds_before,
            "has_per_file_fds invariant"
        );
    }

    #[test]
    fn discard_anchor_state_walks_descendants_and_releases_their_demand() {
        // Materialise a Profile with a Dir child; verify the per-descendant contribution is
        // released by the helper.
        let mut e = Engine::new();
        let anchor = e.tree_mut().ensure_root("src", ResourceRole::User);
        e.tree_mut().set_kind(anchor, ResourceKind::Dir);

        let req = SubAttachRequest::for_anchor(
            "watch".into(),
            SubAttachAnchor::Resource(anchor),
            ScanConfig::builder().recursive(true).build(),
            MAX_SETTLE,
            SETTLE,
            empty_program(),
            EffectScope::SubtreeRoot,
            ClassSet::EMPTY,
            false,
        );
        let t0 = Instant::now();
        let attach_out = e.step(Input::AttachSub(req), t0);
        let sid =
            specter_core::testkit::first_attached_sub(&attach_out).expect("attach_sub succeeded");
        let pid = e.subs().get(sid).unwrap().profile();
        drive_fresh_seed_to_idle(
            &mut e,
            pid,
            dir_snap(vec![("nested", EntryKind::Dir, 1)]),
            t0,
        );

        // Confirm the child slot is materialised + watched.
        let nested_id = e.tree().lookup(Some(anchor), "nested").expect("child slot");
        assert!(
            e.tree()
                .get(nested_id)
                .is_some_and(|r| r.watch_demand() >= 1),
            "child watch_demand bumped by graft",
        );

        let mut out = StepOutput::default();
        e.discard_anchor_state(pid, &mut out);

        // Child's contribution from this Profile released; the slot may even have been reaped if no
        // other claimers remain. Either way, its watch_demand drops to 0.
        let child_demand = e
            .tree()
            .get(nested_id)
            .map_or(0, specter_core::Resource::watch_demand);
        assert_eq!(
            child_demand, 0,
            "descendant contribution released after discard_anchor_state",
        );
        let _ = sid;
    }

    /// Anchor-loss mid-burst with a dirty descendant: the abnormal-end path through `Tree::vacate`
    /// cleanly reaps the descendant slot with the kernel-watch protocol balanced. `vacate` is a
    /// single-protocol (`Unwatch`-only) terminus, so no suppress-precondition can be violated. This
    /// pins the *positive* invariant — exactly one `Unwatch(b)` closes the descendant's watch via
    /// the terminus, the slot reaps, and the Profile reverts to anchor-loss state.
    ///
    /// Lifecycle reproduced:
    /// 1. Profile P at `/a` (Dir), STRUCTURE-only, with materialised descendant `/a/b` (Dir) —
    ///    `b.watch_demand == 1`.
    /// 2. `FsEvent` at `/a` ⇒ `start_standard_burst` ⇒ `Active(PreFire(Batching))`.
    /// 3. `FsEvent` at `/a/b` mid-Batching ⇒ `event_drives_batching` tracks `b`'s path in the
    ///    burst's `dirty` provenance.
    /// 4. `WatchOpRejected` on the anchor ⇒ `on_watch_op_rejected` ⇒ `finalize_anchor_lost(P)` ⇒
    ///    `discard_anchor_state(P)` ⇒ `release_descendant_claim(P)` walks the snapshot ⇒
    ///    `delete_child(b)` ⇒ `sub_watch_then_try_reap(b)`: the last contribution drains (emits
    ///    `WatchOp::Unwatch { resource: b }`) then `try_reap` removes the slot — `vacate`'s
    ///    `Unwatch` branch is dormant there (the map is already empty by `has_anchors`' contract).
    #[test]
    fn release_descendant_claim_clean_reaps_dirty_descendant_via_vacate() {
        // Materialise P at /a with Dir descendant /a/b. STRUCTURE-only ⇒ `has_per_file_fds =
        // false`, so the descendant clause's Dir branch is the contribution this exercises.
        let mut e = Engine::new();
        let anchor = e.tree_mut().ensure_root("a", ResourceRole::User);
        e.tree_mut().set_kind(anchor, ResourceKind::Dir);

        let req = SubAttachRequest::for_anchor(
            "watch".into(),
            SubAttachAnchor::Resource(anchor),
            ScanConfig::builder().recursive(true).build(),
            MAX_SETTLE,
            SETTLE,
            empty_program(),
            EffectScope::SubtreeRoot,
            ClassSet::STRUCTURE,
            false,
        );
        let t0 = Instant::now();
        let attach_out = e.step(Input::AttachSub(req), t0);
        let sid =
            specter_core::testkit::first_attached_sub(&attach_out).expect("attach_sub succeeded");
        let pid = e.subs().get(sid).unwrap().profile();

        // Seed-Ok response materialises descendant /a/b as a Dir.
        drive_fresh_seed_to_idle(&mut e, pid, dir_snap(vec![("b", EntryKind::Dir, 7)]), t0);
        let b_id = e.tree().lookup(Some(anchor), "b").expect("b materialised");
        assert_eq!(
            e.tree().get(b_id).unwrap().watch_demand(),
            1,
            "descendant b carries P's STRUCTURE contribution",
        );

        // FsEvent at the anchor opens a Standard burst (Idle → Active).
        e.step(
            Input::FsEvent {
                resource: anchor,
                event: specter_core::FsEvent::StructureChanged,
            },
            Instant::now(),
        );

        // FsEvent at the descendant mid-Batching: `event_drives_batching` tracks `b` in the burst's
        // dirty / force-walk accumulator. This is the per-event state the deleted global suppress
        // filter used to poison for a co-resident Profile; assert it concretely.
        e.step(
            Input::FsEvent {
                resource: b_id,
                event: specter_core::FsEvent::StructureChanged,
            },
            Instant::now(),
        );
        {
            let p = e.profiles().get(pid).expect("Profile lives");
            let pre = match p.state() {
                specter_core::ProfileState::Active(specter_core::ActiveBurst::PreFire(pre), _) => {
                    pre
                }
                other => panic!("expected Active(PreFire) mid-burst, got {other:?}"),
            };
            assert!(
                matches!(pre.phase, specter_core::PreFirePhase::Batching { .. }),
                "descendant event keeps the burst Batching",
            );
            let b_path = e.tree().path_of(b_id).expect("b path resolves");
            assert!(
                pre.dirty.chains().contains(&b_path),
                "event_drives_batching tracked b's path in dirty (the obligation basis)",
            );
        }

        // WatchOpRejected on the anchor: the abnormal-end path through finalize_anchor_lost →
        // discard_anchor_state → release_descendant_claim → delete_child(b) → vacate. The
        // single-protocol terminus makes the old suppress-precondition dev-panic unconstructable;
        // the test reaching its asserts is itself the no-panic witness.
        let purge_out = e.step(
            Input::WatchOpRejected {
                resource: anchor,
                failure: specter_core::WatchFailure::Pressure { errno: 24 },
            },
            Instant::now(),
        );

        // Clean reap: the single-contributor descendant slot is gone.
        assert!(
            e.tree().get(b_id).is_none(),
            "descendant b reaped after delete_child + try_reap",
        );

        // The kernel-watch protocol stays balanced through the single-protocol vacate terminus:
        // exactly one Unwatch(b), no other op references the reaped descendant.
        let unwatch_b = purge_out
            .watch_ops
            .iter()
            .filter(|op| matches!(op, WatchOp::Unwatch { resource } if *resource == b_id))
            .count();
        assert_eq!(
            unwatch_b, 1,
            "exactly one Unwatch(b) closes b's watch via the vacate terminus; \
             got {:?}",
            purge_out.watch_ops,
        );

        // Profile reverts to anchor-loss state: anchor_claim cleared, baseline / kind cleared,
        // watch_root_parent preserved (the recovery channel — but the anchor is a root in this
        // fixture, so `watch_root_parent` is None throughout).
        let p = e.profiles().get(pid).expect("Profile lives");
        assert_eq!(p.anchor_claim(), AnchorClaim::None);
        assert!(p.kind().is_none());
        assert!(p.baseline().is_none());
        assert!(p.current().is_none());

        // The purge's loss wrapper finished the burst and parked the anchorless Profile.
        assert!(matches!(p.state(), specter_core::ProfileState::Parked));
    }

    /// `release_anchor_claim` flips a materialised `Held` claim to `None` and is idempotent — a
    /// second call is a no-op on both the claim and the Tree (the early-return guard on
    /// `anchor_claim`), so it emits no further watch op. Isolates the helper that
    /// `discard_anchor_state` composes; pins the materialise ↔ release symmetry directly.
    #[test]
    fn release_anchor_claim_is_symmetric_and_idempotent() {
        let (mut e, _sid, pid, _anchor, _parent) =
            engine_with_materialised_profile(ClassSet::EMPTY);

        assert_eq!(
            e.profiles().get(pid).unwrap().anchor_claim(),
            AnchorClaim::Held,
            "fixture materialised the anchor → claim Held",
        );

        let mut out = StepOutput::default();
        e.release_anchor_claim(pid, &mut out);
        assert_eq!(
            e.profiles().get(pid).unwrap().anchor_claim(),
            AnchorClaim::None,
            "release flips Held → None (symmetric with materialise)",
        );

        let mut out2 = StepOutput::default();
        e.release_anchor_claim(pid, &mut out2);
        assert_eq!(
            e.profiles().get(pid).unwrap().anchor_claim(),
            AnchorClaim::None,
            "second release is idempotent — stays None",
        );
        assert!(
            out2.watch_ops.is_empty(),
            "idempotent release emits no further Tree watch op: {:?}",
            out2.watch_ops,
        );
    }
}
