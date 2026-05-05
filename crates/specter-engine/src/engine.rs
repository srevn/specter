//! `Engine` — pure, deterministic, total.
//!
//! The engine owns the data model (`Tree`, `ProfileMap`, `SubRegistry`),
//! the timer wheel, and the stability index; `step` consumes one [`Input`]
//! at a time and emits a sorted [`StepOutput`]. State-machine bodies live
//! in sibling modules:
//! - `burst.rs` — Idle ↔ Active phase transitions.
//! - `transitions.rs` — per-input handlers (`on_fs_event`, etc.).
//! - `reconcile.rs` — newly-discovered descendants.
//! - `refcounts.rs` — `watch_demand` / `suppress_count` edges.
//!
//! `step` is the single dispatch point; each `Input` variant routes to the
//! corresponding `on_*` handler. `attach_sub` is the engine's public
//! Sub-attachment API.

use crate::refcounts::add_watch_demand;
use crate::stability::StabilityIndex;
use crate::timer::TimerHeap;
use specter_core::{
    BurstPhase, ClassSet, DedupKey, DescentState, Diagnostic, Effect, Input, ProbeCorrelation,
    ProbeOp, Profile, ProfileId, ProfileMap, ProfileState, ResourceId, StepOutput, Sub,
    SubAttachRequest, SubId, SubRegistry, TimerId, Tree, WatchOp, compute_config_hash,
};
use std::path::Component;
use std::time::{Duration, Instant};

/// Synthetic segment representing the filesystem root `/`. Lives in the
/// Tree as a single `DescentScaffold`-roled root that absolute-path
/// attaches share — every absolute attach decomposes to `[FS_ROOT_SEG,
/// ...real segments]` so descents have a guaranteed-existing starting
/// ancestor. `Tree::path_of` reconstructs an absolute path from this
/// segment because `PathBuf::push` resets to absolute when given `"/"`.
/// Verified by `tree::tests::path_of_handles_absolute_root_segment`.
pub(crate) const FS_ROOT_SEG: &str = "/";

/// `pub(crate)` field visibility lets sibling modules read/write engine
/// state directly. External consumers go through the public methods.
///
/// Per-Profile pending-path descent state lives inline on
/// `ProfileState::Pending(DescentState)`. Read through
/// [`Engine::descent_state`] / [`Engine::descent_state_mut`];
/// fan-out queries use [`Engine::descents_at_prefix`].
#[derive(Debug, Default)]
pub struct Engine {
    pub(crate) tree: Tree,
    pub(crate) profiles: ProfileMap,
    pub(crate) subs: SubRegistry,
    pub(crate) timers: TimerHeap,
    pub(crate) stability: StabilityIndex,
    pub(crate) next_correlation: u64,
}

impl Engine {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The Profile's descent state, if it is `Pending`. Returns `None`
    /// for `Idle` and `Active` Profiles. Sole reader API for the
    /// `ProfileState::Pending(DescentState)` payload outside the
    /// routing/trichotomy match sites.
    #[must_use]
    pub(crate) fn descent_state(&self, pid: ProfileId) -> Option<&DescentState> {
        match &self.profiles.get(pid)?.state {
            ProfileState::Pending(d) => Some(d),
            _ => None,
        }
    }

    /// Mutable counterpart to [`Engine::descent_state`].
    pub(crate) fn descent_state_mut(&mut self, pid: ProfileId) -> Option<&mut DescentState> {
        match &mut self.profiles.get_mut(pid)?.state {
            ProfileState::Pending(d) => Some(d),
            _ => None,
        }
    }

    /// Profiles whose Pending descent has `current_prefix == resource`.
    /// Sole consumer is `on_fs_event`'s descent fan-out (multiple
    /// Profiles may share one prefix — e.g., two Subs anchored at sibling
    /// children await the shared parent's materialization).
    ///
    /// O(profiles). `SmallVec<[ProfileId; 2]>` matches the typical case
    /// (0-1 descents per prefix; 2 covers the "shared scaffold" case).
    pub(crate) fn descents_at_prefix(
        &self,
        resource: ResourceId,
    ) -> smallvec::SmallVec<[ProfileId; 2]> {
        let mut out: smallvec::SmallVec<[ProfileId; 2]> = smallvec::SmallVec::new();
        for (pid, p) in self.profiles.iter() {
            if let ProfileState::Pending(d) = &p.state
                && d.current_prefix == resource
            {
                out.push(pid);
            }
        }
        out
    }

    /// Pure, deterministic, total. Consumes one [`Input`], emits a sorted
    /// [`StepOutput`]. Each variant routes to the corresponding
    /// `on_*` handler (`transitions.rs`).
    ///
    /// `match_same_arms` is permitted: explicit arms document the routing
    /// even when bodies are uniform; the trailing wildcard absorbs
    /// `non_exhaustive` v2+ variants.
    #[allow(clippy::match_same_arms)]
    pub fn step(&mut self, input: Input, now: Instant) -> StepOutput {
        let mut out = StepOutput::default();
        match input {
            Input::FsEvent { resource, event } => {
                self.on_fs_event(resource, event, now, &mut out);
            }
            Input::ProbeResponse(resp) => {
                self.on_probe_response(resp, now, &mut out);
            }
            Input::TimerExpired(id) => {
                self.on_timer_expired(id, now, &mut out);
            }
            Input::EffectComplete { sub, key, result } => {
                self.on_effect_complete(sub, key, &result, now, &mut out);
            }
            Input::WatchOpRejected {
                resource,
                op,
                errno,
            } => {
                self.on_watch_op_rejected(resource, op, errno, now, &mut out);
            }
            Input::ConfigDiff(diff) => {
                self.on_config_diff(diff, now, &mut out);
            }
            // `Input` is `non_exhaustive` in `core`; downstream pattern
            // matches require a wildcard. New variants land alongside
            // their handlers.
            _ => {}
        }
        self.sort_step_output(&mut out);
        out
    }

    /// Attach a Sub to an existing Resource (`req.resource`). Reuses an
    /// existing Profile when `(resource, config_hash)` matches; otherwise
    /// creates a fresh Profile, emits `WatchOp::Watch` on its anchor, and
    /// starts a `Burst { intent: Seed, phase: Probing }` to establish the
    /// initial baseline.
    ///
    /// Returns the minted [`SubId`] and a sorted [`StepOutput`].
    ///
    /// # Panics
    /// Panics if `req.resource` is stale (no live Tree slot). The Engine
    /// must construct the Resource before attaching a Sub to it.
    pub fn attach_sub(&mut self, req: SubAttachRequest, now: Instant) -> (SubId, StepOutput) {
        let mut out = StepOutput::default();
        let sub_id = self.attach_sub_inner(req, now, &mut out);
        self.sort_step_output(&mut out);
        (sub_id, out)
    }

    /// Inner attach used by `on_config_diff` to compose multiple
    /// detach/attach operations into a single (sorted) `StepOutput`.
    /// Returns the minted [`SubId`] (or `SubId::default()` if anchor
    /// resolution fails — only the path-based path can fail; the engine
    /// emits a Diagnostic in that case).
    pub(crate) fn attach_sub_inner(
        &mut self,
        req: SubAttachRequest,
        now: Instant,
        out: &mut StepOutput,
    ) -> SubId {
        // Resolve anchor. Path-based attach materializes scaffolds;
        // resource-based attach trusts the caller's id.
        let (anchor, pending_components) = match req.path.as_ref() {
            Some(path) => {
                let Some(comps) = decompose_attach_path(path, out) else {
                    return SubId::default();
                };
                let materialize = self.materialize_path_or_pending(&comps);
                match materialize {
                    crate::descent::MaterializeResult::Materialized(id) => (id, None),
                    crate::descent::MaterializeResult::Pending {
                        anchor,
                        prefix,
                        remaining,
                    } => (anchor, Some((prefix, remaining))),
                }
            }
            None => (req.resource, None),
        };

        // If the anchor was a DescentScaffold and now becomes the
        // anchor of a User Profile, promote its role. Pending case is
        // handled at materialization time inside descent dispatch.
        if pending_components.is_none()
            && let Some(res) = self.tree.get(anchor)
            && matches!(res.role, specter_core::ResourceRole::DescentScaffold)
        {
            self.tree.set_role(anchor, specter_core::ResourceRole::User);
        }

        let cfg_hash = compute_config_hash(&req.config, req.max_settle, req.events);
        let profile_id = if let Some(pid) = self.profiles.find(anchor, cfg_hash) {
            pid
        } else {
            let p = Profile::new(
                anchor,
                req.config.clone(),
                req.max_settle,
                req.settle,
                req.events,
            );
            self.profiles.attach(&mut self.tree, p)
        };

        let is_fresh_profile = self
            .profiles
            .get(profile_id)
            .is_some_and(|p| p.sub_refcount == 0);

        // Insert the Sub.
        let sub_id = self.subs.insert(|sid| {
            Sub::new(
                sid,
                req.name,
                profile_id,
                req.command,
                req.scope,
                req.settle,
                req.max_settle,
                req.events,
            )
        });

        if let Some(p) = self.profiles.get_mut(profile_id) {
            p.sub_refcount = p.sub_refcount.saturating_add(1);
        }

        if !is_fresh_profile {
            // Existing Profile: recompute settle. `min(existing,
            // new_sub.settle)` is the cheapest correct update — the Sub
            // joins an existing burst lifecycle (or shares the Idle
            // baseline). No fresh Watch / Probe / parent-edge work.
            //
            // Under D3 the events mask folds into `config_hash`, so a
            // Sub joining an existing Profile shares its mask by
            // construction — `events_union` and `has_per_file_fds` are
            // invariant for the Profile's lifetime. No retroactive
            // per-leaf `watch_demand` bump is needed (the prior B2
            // recompute machinery is removed).
            if let Some(p) = self.profiles.get_mut(profile_id)
                && req.settle < p.settle
            {
                p.settle = req.settle;
            }
            return sub_id;
        }

        // ===== Fresh Profile path =====

        // Capture the Profile's mask before any &mut borrows. Used as the
        // anchor's contribution (immediate-Seed path); the descent prefix
        // path uses `STRUCTURE` per D9 instead.
        let events_union = self
            .profiles
            .get(profile_id)
            .map_or(ClassSet::EMPTY, |p| p.events_union);

        if let Some((prefix, remaining)) = pending_components {
            // Pending descent. Bump the prefix's watch_demand with a
            // `STRUCTURE` contribution (D9 — the prefix is infrastructure;
            // it always wants to see directory-entry changes regardless
            // of the Sub's user mask). Profile.state stays Idle while
            // the descent runs; the anchor materializes via
            // `dispatch_descent_ok`'s anchor branch, which then sets up
            // the watch-root-parent contribution and starts the Seed
            // burst. Setting watch_root_parent here would bump
            // watch_demand on a `DescentScaffold` slot that doesn't
            // exist on disk yet, generating a `WatchOpRejected` from
            // the Sensor.
            self.compute_and_set_parent_edge(profile_id);
            self.recompute_dependent_parent_edges(profile_id);

            add_watch_demand(&mut self.tree, prefix, ClassSet::STRUCTURE, out);

            let correlation = self.next_probe_correlation();
            if let Some(p) = self.profiles.get_mut(profile_id) {
                p.state = ProfileState::Pending(DescentState {
                    current_prefix: prefix,
                    remaining_components: remaining,
                    probe_correlation: Some(correlation),
                });
            }
            self.emit_descent_probe(profile_id, prefix, correlation, out);
        } else {
            // Immediate-Seed path. Anchor exists; bump its watch_demand
            // with the Profile's events_union (the user-declared mask),
            // set up the watch-root parent (STRUCTURE per D9), compute
            // parent edges, start the Seed burst.
            add_watch_demand(&mut self.tree, anchor, events_union, out);
            if let Some(p) = self.profiles.get_mut(profile_id) {
                p.anchor_contribution = true;
            }
            self.set_watch_root_parent(profile_id, anchor, out);
            self.compute_and_set_parent_edge(profile_id);
            self.recompute_dependent_parent_edges(profile_id);

            self.start_seed_burst(profile_id, now, out);
        }

        sub_id
    }

    /// Set up the Profile's watch-root parent contribution. For each
    /// User-role Profile P, the Engine ensures `P.resource.parent` (if
    /// it exists) carries a `+1` `watch_demand` contribution from P. The
    /// parent's role is promoted to `WatchRootParent` only if it was
    /// previously a bare `DescentScaffold`; `User` parents stay `User`
    /// (never demote User).
    ///
    /// Caches the parent id on `Profile.watch_root_parent` so
    /// `reap_profile` can release the contribution without re-deriving.
    /// `None` if the anchor has no parent in the Tree (a root anchor) —
    /// root rename detection is then unavailable.
    ///
    /// Sole call sites: `attach_sub_inner` (immediate-Seed path, where
    /// the anchor exists on disk and so does its parent) and
    /// `descent::dispatch_descent_ok` (anchor materialization).
    pub(crate) fn set_watch_root_parent(
        &mut self,
        profile_id: ProfileId,
        anchor: ResourceId,
        out: &mut StepOutput,
    ) {
        let Some(parent_id) = self.tree.parent(anchor) else {
            return;
        };

        // Idempotent: "Watch root deletion" recovery re-enters descent
        // on a Profile whose `watch_root_parent` field was set at the
        // original materialization and never cleared on
        // `on_anchor_terminal_event`. When recovery's descent advances
        // back to anchor materialization it would otherwise call this
        // helper again, double-bumping the parent's `watch_demand` for
        // the same Profile. Skip the bump if the cache already points
        // at the same parent id.
        let already_set = self
            .profiles
            .get(profile_id)
            .is_some_and(|p| p.watch_root_parent == Some(parent_id));
        if already_set {
            return;
        }

        // Promote role: DescentScaffold → WatchRootParent. User and
        // existing WatchRootParent stay as they are.
        if let Some(parent) = self.tree.get(parent_id)
            && matches!(parent.role, specter_core::ResourceRole::DescentScaffold)
        {
            self.tree
                .set_role(parent_id, specter_core::ResourceRole::WatchRootParent);
        }

        // D9 — the watch-root parent is engine infrastructure (used to
        // detect anchor reappearance after a `rm -rf` of the anchor).
        // Contribution is `STRUCTURE` regardless of the Sub's user mask.
        // The corresponding bookkeeping flag is `Profile.watch_root_parent
        // == Some(parent_id)`, written below; the recompute path reads
        // that flag to attribute this contribution back to the Profile.
        add_watch_demand(&mut self.tree, parent_id, ClassSet::STRUCTURE, out);

        if let Some(p) = self.profiles.get_mut(profile_id) {
            p.watch_root_parent = Some(parent_id);
        }
    }

    /// Compute and store the parent edge for a fresh Profile. `compute_parent`
    /// walks Resource ancestors; the smallest covering [`ProfileId`] wins
    /// by deterministic tie-break.
    fn compute_and_set_parent_edge(&mut self, profile_id: ProfileId) {
        if let Some(parent) = StabilityIndex::compute_parent(&self.tree, &self.profiles, profile_id)
        {
            self.stability.set_parent(profile_id, parent);
        }
    }

    /// After adding a fresh Profile, recompute parent edges of every
    /// other Profile that the new one might interpose. O(profiles²)
    /// worst-case; acceptable for v1's small configs.
    fn recompute_dependent_parent_edges(&mut self, new_profile: ProfileId) {
        let candidates: Vec<ProfileId> = self
            .profiles
            .iter()
            .filter(|(pid, _)| *pid != new_profile)
            .map(|(pid, _)| pid)
            .collect();
        self.stability
            .recompute_parent_edges_for_subset(&self.tree, &self.profiles, candidates);
    }

    /// Detach a Sub by id.
    ///
    /// Decrements `Profile.sub_refcount`; recomputes `Profile.settle =
    /// min(remaining_subs.settles)`. If the count reaches zero:
    /// - **Idle Profile:** reap immediately. Release anchor `watch_demand`
    ///   (1→0 emits Unwatch), release `watch_root_parent` contribution,
    ///   clear parent edge, recompute parent edges of dependents, and
    ///   `try_reap` the anchor Resource.
    /// - **Active Profile:** set `Profile.reap_pending = true`. The active
    ///   burst runs to completion; on `finish_burst_to_idle`, the Engine
    ///   skips Effect emission (`emit_effects` checks `reap_pending`) and
    ///   reaps the Profile in the same step as the Probing → Idle
    ///   transition.
    ///
    /// If the count remains > 0, the Profile stays alive; only
    /// `Profile.settle` is recomputed.
    ///
    /// Idempotent on stale `SubId` (Diagnostic + drop). Returns the sorted
    /// `StepOutput` of any ops emitted.
    pub fn detach_sub(&mut self, sub: SubId, now: Instant) -> StepOutput {
        let mut out = StepOutput::default();
        self.detach_sub_inner(sub, now, &mut out);
        self.sort_step_output(&mut out);
        out
    }

    /// Inner detach used by `on_config_diff` to compose multiple
    /// detach/attach operations into a single (sorted) `StepOutput`.
    pub(crate) fn detach_sub_inner(&mut self, sub: SubId, _now: Instant, out: &mut StepOutput) {
        let profile_id = match self.subs.remove(sub) {
            Some(s) => s.profile,
            None => {
                out.diagnostics
                    .push(Diagnostic::EffectCompleteForUnknownSub { sub });
                return;
            }
        };

        let Some(p) = self.profiles.get_mut(profile_id) else {
            return;
        };
        p.sub_refcount = p.sub_refcount.saturating_sub(1);
        let new_refcount = p.sub_refcount;

        // Bundle B1: purge `last_emitted_dir_hash` entries keyed by the
        // detached Sub. Their suppress-targets are stale; leaving them
        // would let a *different* Sub on the same Profile (sharing a
        // DedupKey by accident — only PerFile keys collide on
        // `(sub, resource)`, so this is mostly a hygienic guard) inherit
        // a stale fingerprint. The full reap path below drops the whole
        // map alongside the Profile, so we only run this purge on the
        // refcount-still-positive branch.
        if new_refcount > 0 {
            if let Some(p) = self.profiles.get_mut(profile_id) {
                p.last_emitted_dir_hash.retain(|k, _| match k {
                    DedupKey::Subtree { sub: s, .. } | DedupKey::PerFile { sub: s, .. } => {
                        *s != sub
                    }
                });
            }
            // Recompute Profile.settle = min(remaining_subs.settles).
            //
            // Under D3 every Sub on a Profile shares the same `events`
            // mask (events folds into `config_hash`); detaching one Sub
            // cannot flip `Profile.has_per_file_fds` or
            // `Profile.events_union`. The prior B2 retroactive-bump
            // machinery is therefore unreachable and has been removed —
            // a future v2 predicate-layer change would re-introduce it
            // shaped around per-Sub contributions, not the broken
            // recompute-per-attach pattern.
            self.recompute_profile_settle(profile_id);
            return;
        }

        // new_refcount == 0: reap immediately for Idle / Pending
        // Profiles, defer for Active Profiles. Pending Profiles reap
        // synchronously — they have no burst whose `finish_burst_to_idle`
        // would resolve a deferred reap, so they use the same path as
        // Idle ones.
        let lifecycle = self.profiles.get(profile_id).map(|p| match &p.state {
            ProfileState::Idle | ProfileState::Pending(_) => DetachLifecycle::ReapNow,
            ProfileState::Active(_) => DetachLifecycle::DeferToBurstEnd,
            // `non_exhaustive` ProfileState: any future variant is
            // treated as ReapNow (no burst to drive the deferred path).
            _ => DetachLifecycle::ReapNow,
        });
        match lifecycle {
            Some(DetachLifecycle::ReapNow) => {
                self.reap_profile(profile_id, out);
            }
            Some(DetachLifecycle::DeferToBurstEnd) => {
                if let Some(p) = self.profiles.get_mut(profile_id) {
                    p.reap_pending = true;
                }
            }
            None => {}
        }
    }

    /// Reap a Profile: release every contribution it holds (anchor watch,
    /// watch-root parent watch, descent prefix watch), clear its parent
    /// edge, recompute parent edges of any dependents, detach from
    /// `ProfileMap`, try-reap the anchor Resource, and emit a
    /// `ReapPendingResolved` Diagnostic.
    ///
    /// **Trichotomy.** A Profile holds at most one of {anchor contribution,
    /// descent prefix contribution} at any time:
    ///
    ///   - **Materialized** (immediate-Seed bumped anchor, or descent
    ///     advanced through anchor materialization): the Profile owns +1
    ///     on `anchor.watch_demand` and +1 on
    ///     `watch_root_parent.watch_demand` (when the anchor has a
    ///     parent). No `ProfileState::Pending(_)` payload.
    ///   - **Pending**: the Profile owns +1 on
    ///     `descent.current_prefix.watch_demand` and nothing else
    ///     (`watch_root_parent` is set only at materialization;
    ///     `anchor_contribution` only flips at the same site).
    ///   - **Purged** (post-WatchOpRejected purge): the Profile owns no
    ///     contributions; the clamp atomically released them and the
    ///     associated bookkeeping was cleaned up by the purge fan-out.
    ///
    /// Each release helper is idempotent and counter-aware, so calling
    /// all three in any order yields the correct net effect for any
    /// trichotomy state.
    ///
    /// Sole call sites: `detach_sub_inner` (Idle / Pending Profile,
    /// immediate reap) and `finish_burst_to_idle` (deferred reap when
    /// `reap_pending` was set mid-burst).
    pub(crate) fn reap_profile(&mut self, profile_id: ProfileId, out: &mut StepOutput) {
        let Some(p) = self.profiles.get(profile_id) else {
            return;
        };
        let anchor = p.resource;

        // Trichotomy invariant: Pending and anchor_contribution are
        // mutually exclusive. Descent flips Pending → Idle and bumps the
        // anchor atomically in `dispatch_descent_ok`'s anchor branch.
        debug_assert!(
            !(matches!(p.state, ProfileState::Pending(_)) && p.anchor_contribution),
            "reap_profile: Pending + anchor_contribution must be mutually exclusive",
        );

        // Cancel any in-flight descent probe BEFORE the descent-prefix
        // helper transitions the Profile to Idle (which drops the
        // correlation). Without this, the prober ships a ProbeResponse
        // for a now-detached Profile, the engine drops it as
        // StaleProbeResponse — wasted prober capacity and I/O.
        // Mirrors `on_watch_op_rejected`'s descent-purge pattern.
        if let Some(d) = self.descent_state(profile_id)
            && d.probe_correlation.is_some()
        {
            out.probe_ops.push(ProbeOp::Cancel {
                profile: profile_id,
            });
        }

        // Release every claim this Profile may hold. Helpers are
        // idempotent — no-op when the corresponding flag is unset (or
        // counter is zero, post-clamp).
        self.release_descent_prefix_claim(profile_id, out);
        self.release_anchor_claim(profile_id, out);
        self.release_watch_root_parent_claim(profile_id, out);

        // Detach the Profile from the registry.
        let _ = self.profiles.detach(&mut self.tree, profile_id);

        // Clear and recompute parent edges.
        self.stability.clear_parent(profile_id);
        self.stability.recompute_parent_edges_for_dependents(
            &self.tree,
            &self.profiles,
            profile_id,
        );

        // Try to reap the anchor's slot. No-op if it still has children,
        // other Profiles, or an infrastructure role.
        self.tree.try_reap(anchor);

        out.diagnostics.push(Diagnostic::ReapPendingResolved {
            profile: profile_id,
        });
    }

    /// Recompute `Profile.settle = min(remaining_subs.settles)` after a
    /// Sub addition or removal. O(subs-on-profile), bounded — typically
    /// 1–2 in v1 because `max_settle` already partitions Profiles.
    pub(crate) fn recompute_profile_settle(&mut self, profile_id: ProfileId) {
        let new_min: Option<Duration> = self
            .subs
            .at(profile_id)
            .iter()
            .filter_map(|sid| self.subs.get(*sid))
            .map(|s| s.settle)
            .min();
        if let (Some(s), Some(p)) = (new_min, self.profiles.get_mut(profile_id)) {
            p.settle = s;
        }
    }

    /// Read-only view of the Engine's `Tree`.
    ///
    /// The bin uses this to inspect Resource topology; tests use it for
    /// setup verification.
    #[must_use]
    pub const fn tree(&self) -> &Tree {
        &self.tree
    }

    /// Mutable access for path-to-`ResourceId` materialization.
    ///
    /// The bin uses this at startup to walk a config's `path` strings into
    /// the Tree before calling `attach_sub`. Use the dedicated refcount
    /// helpers to modify `watch_demand` / `suppress_count` — direct
    /// mutation breaks the 0↔1 edge invariant.
    pub const fn tree_mut(&mut self) -> &mut Tree {
        &mut self.tree
    }

    /// Read-only view of the `ProfileMap`.
    ///
    /// For inspection only; state-machine mutations route through `step`
    /// and `attach_sub`.
    #[must_use]
    pub const fn profiles(&self) -> &ProfileMap {
        &self.profiles
    }

    /// Read-only view of the `SubRegistry`.
    #[must_use]
    pub const fn subs(&self) -> &SubRegistry {
        &self.subs
    }

    /// Earliest pending timer deadline, or `None` if no timers are armed.
    #[must_use]
    pub fn next_deadline(&self) -> Option<Instant> {
        self.timers.peek_top().map(|e| e.deadline)
    }

    /// Pop the earliest expired-and-still-referenced timer. Stale entries
    /// (cancelled because the Profile's burst was reset) are silently
    /// dropped.
    pub fn pop_expired(&mut self, now: Instant) -> Option<TimerId> {
        loop {
            let top = self.timers.peek_top()?;
            if top.deadline > now {
                return None;
            }
            let entry = self
                .timers
                .pop_top()
                .expect("peek_top returned Some; pop_top must too");
            if Self::is_timer_referenced(&self.profiles, entry.profile, entry.id) {
                return Some(entry.id);
            }
            // Stale — silently drop, continue draining.
        }
    }

    /// Mint a fresh `ProbeCorrelation` token. Engine-monotonic; saturating
    /// at `u64::MAX` (unreachable in any realistic deployment).
    pub(crate) const fn next_probe_correlation(&mut self) -> ProbeCorrelation {
        self.next_correlation = self.next_correlation.saturating_add(1);
        ProbeCorrelation(self.next_correlation)
    }

    /// Whether `id` is referenced by `profile`'s active burst — `pop_expired`
    /// uses this to filter stale heap heads. Only `Active` Profiles
    /// schedule timers; `Idle` and `Pending` Profiles never do.
    #[allow(clippy::match_same_arms)]
    fn is_timer_referenced(profiles: &ProfileMap, profile: ProfileId, id: TimerId) -> bool {
        let Some(p) = profiles.get(profile) else {
            return false;
        };
        match &p.state {
            ProfileState::Idle => false,
            // Descent is event-driven (no settle timer, no burst
            // deadline). The arm is structurally redundant with the
            // wildcard but documents that Pending intentionally has no
            // timers.
            ProfileState::Pending(_) => false,
            ProfileState::Active(burst) => {
                burst.settle_timer == Some(id) || burst.burst_deadline == id
            }
            // `non_exhaustive` ProfileState: future variants conservatively
            // treat the timer as unreferenced (drains it).
            _ => false,
        }
    }

    /// Sort: `watch_ops` by `ResourceId`; `probe_ops` by `ProfileId`;
    /// `effects` by `(SubId, ResourceId)`. `Subtree`-keyed effects look up
    /// the Profile's anchor at sort time, so the method takes `&self`.
    /// `diagnostics` follow insertion order — they aren't part of the
    /// user-visible sort guarantee.
    pub(crate) fn sort_step_output(&self, out: &mut StepOutput) {
        out.watch_ops.sort_by_key(Self::watch_op_key);
        out.probe_ops.sort_by_key(Self::probe_op_key);
        out.effects.sort_by_key(|e| self.effect_sort_key(e));
    }

    fn effect_sort_key(&self, e: &Effect) -> (SubId, ResourceId) {
        match &e.key {
            DedupKey::PerFile { sub, resource } => (*sub, *resource),
            DedupKey::Subtree { sub, profile } => {
                let resource = self
                    .profiles
                    .get(*profile)
                    .map_or_else(ResourceId::default, |p| p.resource);
                (*sub, resource)
            }
        }
    }

    pub(crate) const fn watch_op_key(op: &WatchOp) -> ResourceId {
        match op {
            WatchOp::Watch { resource, .. }
            | WatchOp::Unwatch { resource }
            | WatchOp::Suppress { resource }
            | WatchOp::Unsuppress { resource } => *resource,
        }
    }

    pub(crate) const fn probe_op_key(op: &ProbeOp) -> ProfileId {
        match op {
            ProbeOp::Probe { request } => request.profile,
            ProbeOp::Cancel { profile } => *profile,
        }
    }
}

// `BurstPhase` is reachable from this module via the `is_timer_referenced`
// path — re-export-friendly type usage.
const _: fn() = || {
    let _ = BurstPhase::Settling;
};

/// Local lifecycle classifier for `detach_sub_inner`. Three
/// outcomes when a Profile loses its last Sub:
/// - `ReapNow`: Profile is `Idle` or `Pending`. Neither holds a burst
///   that would resolve a deferred reap; `reap_profile` runs
///   immediately, releasing the descent prefix (Pending) or the anchor
///   contribution (Idle / materialized).
/// - `DeferToBurstEnd`: Profile is `Active`. Set `reap_pending = true`;
///   `finish_burst_to_idle` runs `reap_profile` once the burst finishes
///   so the in-flight burst doesn't fire Effects against a stale Sub
///   registry.
enum DetachLifecycle {
    ReapNow,
    DeferToBurstEnd,
}

/// Decompose an attach path into Tree segments. `RootDir` becomes the
/// synthetic [`FS_ROOT_SEG`] so absolute attaches share one root in the
/// Tree (`Tree::path_of` reconstructs an absolute path because
/// `PathBuf::push("/")` resets to absolute).
///
/// Returns `None` and emits [`Diagnostic::AttachPathInvalid`] on:
/// - empty paths (no real components);
/// - relative components `.` / `..` (config validation should canonicalize
///   before attach — defense-in-depth);
/// - `Component::Prefix` (Windows-only; unreachable on Unix v1).
///
/// Non-UTF-8 segments are skipped silently — Tree keys are `&str`-interned
/// and the engine can't represent them. This matches the prior filter
/// behavior.
fn decompose_attach_path<'a>(
    path: &'a std::path::Path,
    out: &mut StepOutput,
) -> Option<Vec<&'a str>> {
    let mut comps: Vec<&str> = Vec::with_capacity(path.components().count());
    for c in path.components() {
        match c {
            Component::RootDir => comps.push(FS_ROOT_SEG),
            Component::Normal(s) => match s.to_str() {
                Some(name) if !name.is_empty() => comps.push(name),
                _ => {}
            },
            Component::CurDir | Component::ParentDir => {
                out.diagnostics.push(Diagnostic::AttachPathInvalid {
                    hint: "non-canonical attach path (`.`/`..`); canonicalize before attach",
                });
                return None;
            }
            Component::Prefix(_) => {
                out.diagnostics.push(Diagnostic::AttachPathInvalid {
                    hint: "Windows path prefix not supported on Unix v1",
                });
                return None;
            }
        }
    }
    if comps.is_empty() {
        out.diagnostics.push(Diagnostic::AttachPathInvalid {
            hint: "empty attach path",
        });
        return None;
    }
    Some(comps)
}

#[cfg(test)]
mod tests {
    use super::*;
    use slotmap::KeyData;
    use specter_core::{
        DedupKey, EffectOutcome, FsEvent, Input, ProbeCorrelation, ProbeKind, ProbeOp,
        ProbeRequest, ProbeResponse, ProbeResult, ProfileId, ResourceId, ScanConfig, StepOutput,
        SubId, SubRegistryDiff, TimerId, WatchOp, WatchOpts,
    };
    use std::path::PathBuf;
    use std::time::{Duration, Instant};

    // Compile-time `Send + Sync` check on `Engine`. The bin loop parks
    // `Engine` on its own thread; `Send + Sync` is load-bearing for that.
    const _: fn() = || {
        const fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Engine>();
    };

    fn rid(n: u64) -> ResourceId {
        ResourceId::from(KeyData::from_ffi(n))
    }

    fn pidn(n: u64) -> ProfileId {
        ProfileId::from(KeyData::from_ffi(n))
    }

    #[test]
    fn step_fs_event_for_unwatched_resource_diagnoses() {
        let mut e = Engine::new();
        let out = e.step(
            Input::FsEvent {
                resource: ResourceId::default(),
                event: FsEvent::Modified,
            },
            Instant::now(),
        );
        let has_diag = out
            .diagnostics
            .iter()
            .any(|d| matches!(d, specter_core::Diagnostic::EventOnUnwatchedResource { .. }));
        assert!(has_diag);
        assert!(out.watch_ops.is_empty());
        assert!(out.probe_ops.is_empty());
        assert!(out.effects.is_empty());
    }

    #[test]
    fn step_probe_response_unknown_profile_diagnoses() {
        let mut e = Engine::new();
        let resp = ProbeResponse {
            profile: ProfileId::default(),
            correlation: ProbeCorrelation(0),
            result: ProbeResult::Vanished,
        };
        let out = e.step(Input::ProbeResponse(resp), Instant::now());
        let has_diag = out
            .diagnostics
            .iter()
            .any(|d| matches!(d, specter_core::Diagnostic::StaleProbeResponse { .. }));
        assert!(has_diag);
    }

    #[test]
    fn step_timer_expired_stale_id_diagnoses() {
        let mut e = Engine::new();
        let out = e.step(Input::TimerExpired(TimerId::default()), Instant::now());
        let has_diag = out
            .diagnostics
            .iter()
            .any(|d| matches!(d, specter_core::Diagnostic::StaleTimer { .. }));
        assert!(has_diag);
    }

    #[test]
    fn step_effect_complete_unknown_sub_diagnoses() {
        let mut e = Engine::new();
        let out = e.step(
            Input::EffectComplete {
                sub: SubId::default(),
                key: DedupKey::default(),
                result: EffectOutcome::Ok,
            },
            Instant::now(),
        );
        let has_diag = out.diagnostics.iter().any(|d| {
            matches!(
                d,
                specter_core::Diagnostic::EffectCompleteForUnknownSub { .. }
            )
        });
        assert!(has_diag);
    }

    #[test]
    fn step_watch_op_rejected_emits_diagnostic_for_stale_resource() {
        // Stale ResourceId or already-Unwatched resource yields a
        // Diagnostic + no other ops.
        let mut e = Engine::new();
        let op = WatchOp::Unwatch {
            resource: ResourceId::default(),
        };
        let out = e.step(
            Input::WatchOpRejected {
                resource: ResourceId::default(),
                op,
                errno: 24,
            },
            Instant::now(),
        );
        assert!(out.watch_ops.is_empty());
        assert!(out.probe_ops.is_empty());
        assert!(out.effects.is_empty());
        let has_diag = out.diagnostics.iter().any(|d| {
            matches!(
                d,
                specter_core::Diagnostic::WatchOpRejected { errno: 24, .. }
            )
        });
        assert!(has_diag);
    }

    #[test]
    fn step_config_diff_with_empty_diff_is_noop() {
        let mut e = Engine::new();
        let out = e.step(
            Input::ConfigDiff(SubRegistryDiff::default()),
            Instant::now(),
        );
        assert!(out.watch_ops.is_empty());
        assert!(out.probe_ops.is_empty());
        assert!(out.effects.is_empty());
        assert!(out.diagnostics.is_empty());
    }

    #[test]
    fn next_deadline_is_none_when_no_timers() {
        let e = Engine::new();
        assert!(e.next_deadline().is_none());
    }

    #[test]
    fn next_deadline_returns_top_after_schedule() {
        let mut e = Engine::new();
        let now = Instant::now();
        let when = now + Duration::from_millis(100);
        e.timers.schedule(when, ProfileId::default());
        assert_eq!(e.next_deadline(), Some(when));
    }

    #[test]
    fn pop_expired_returns_none_when_top_in_future() {
        let mut e = Engine::new();
        let now = Instant::now();
        let when = now + Duration::from_secs(10);
        e.timers.schedule(when, ProfileId::default());
        assert_eq!(e.pop_expired(now), None);
        assert_eq!(e.timers.len(), 1, "future-dated entries are not drained");
    }

    #[test]
    fn pop_expired_drains_stale_entries_silently() {
        // Schedule timers for null/unknown Profiles (no Active state holds
        // them). The validating drain consumes every stale entry, but returns
        // None — there's nothing live to fire.
        let mut e = Engine::new();
        let now = Instant::now();
        let past = now
            .checked_sub(Duration::from_millis(1))
            .expect("test clock has room for sub-millisecond rewind");
        e.timers.schedule(past, ProfileId::default());
        e.timers.schedule(past, ProfileId::default());
        e.timers.schedule(past, ProfileId::default());

        assert_eq!(e.pop_expired(now), None);
        assert_eq!(e.timers.len(), 0, "stale heads were drained");
    }

    #[test]
    fn pop_expired_stops_at_first_future_entry() {
        // Mix of expired-stale and future-dated. The drain consumes the stale
        // expired heads, then returns None when peeking a future-dated entry.
        let mut e = Engine::new();
        let now = Instant::now();
        let past = now
            .checked_sub(Duration::from_millis(1))
            .expect("test clock has room for sub-millisecond rewind");
        e.timers.schedule(past, ProfileId::default());
        e.timers
            .schedule(now + Duration::from_secs(10), ProfileId::default());

        assert_eq!(e.pop_expired(now), None);
        assert_eq!(e.timers.len(), 1, "future entry remains");
        assert!(e.next_deadline().unwrap() > now);
    }

    #[test]
    fn next_probe_correlation_is_monotonic() {
        let mut e = Engine::new();
        let a = e.next_probe_correlation();
        let b = e.next_probe_correlation();
        let c = e.next_probe_correlation();
        assert!(a < b);
        assert!(b < c);
        assert_eq!(a, ProbeCorrelation(1));
        assert_eq!(b, ProbeCorrelation(2));
        assert_eq!(c, ProbeCorrelation(3));
    }

    #[test]
    fn sort_step_output_orders_watch_ops_by_resource_id() {
        let r1 = rid(1);
        let r2 = rid(2);
        let r3 = rid(3);
        let mut out = StepOutput::default();
        out.watch_ops.push(WatchOp::Suppress { resource: r3 });
        out.watch_ops.push(WatchOp::Watch {
            resource: r1,
            path: PathBuf::from("/x"),
            opts: WatchOpts::default(),
        });
        out.watch_ops.push(WatchOp::Unwatch { resource: r2 });

        let e = Engine::new();
        e.sort_step_output(&mut out);

        let resources: Vec<ResourceId> = out.watch_ops.iter().map(Engine::watch_op_key).collect();
        assert_eq!(resources, vec![r1, r2, r3]);
    }

    #[test]
    fn sort_step_output_orders_probe_ops_by_profile_id() {
        let p1 = pidn(1);
        let p2 = pidn(2);
        let mut out = StepOutput::default();
        out.probe_ops.push(ProbeOp::Cancel { profile: p2 });
        out.probe_ops.push(ProbeOp::Probe {
            request: ProbeRequest {
                profile: p1,
                correlation: ProbeCorrelation(7),
                kind: ProbeKind::File,
                target_resource: ResourceId::default(),
                target_path: PathBuf::from("/y"),
                scan_config: ScanConfig::builder().build(),
                captured_with: 0,
                baseline_subtree: None,
                force_walk: std::collections::BTreeSet::new(),
                forced: false,
            },
        });

        let e = Engine::new();
        e.sort_step_output(&mut out);

        let profiles: Vec<ProfileId> = out.probe_ops.iter().map(Engine::probe_op_key).collect();
        assert_eq!(profiles, vec![p1, p2]);
    }

    #[test]
    fn engine_default_constructible_has_empty_state() {
        let e = Engine::new();
        assert!(e.tree.is_empty());
        assert!(e.profiles.is_empty());
        assert!(e.subs.is_empty());
        assert!(e.timers.is_empty());
        assert!(e.next_deadline().is_none());
        assert_eq!(e.next_correlation, 0);
        assert!(e.stability.parent_of(ProfileId::default()).is_none());
    }

    // ===== decompose_attach_path =====
    //
    // Path decomposition is the seam between user-supplied `PathBuf` (from
    // the bin's TOML loader) and the Tree's `&str` segment world. The fix
    // preserves `Component::RootDir` as the synthetic [`FS_ROOT_SEG`] so
    // `Tree::path_of` reconstructs an absolute path on the way back out.

    #[test]
    fn decompose_absolute_path_preserves_root_marker() {
        let mut out = StepOutput::default();
        let comps = decompose_attach_path(std::path::Path::new("/tmp"), &mut out)
            .expect("absolute path decomposes");
        assert_eq!(comps, vec![FS_ROOT_SEG, "tmp"]);
        assert!(out.diagnostics.is_empty());
    }

    #[test]
    fn decompose_absolute_deep_path_preserves_each_segment() {
        let mut out = StepOutput::default();
        let comps = decompose_attach_path(std::path::Path::new("/var/log/myapp"), &mut out)
            .expect("absolute deep path decomposes");
        assert_eq!(comps, vec![FS_ROOT_SEG, "var", "log", "myapp"]);
    }

    #[test]
    fn decompose_relative_path_skips_root_marker() {
        let mut out = StepOutput::default();
        let comps = decompose_attach_path(std::path::Path::new("foo/bar"), &mut out)
            .expect("relative path decomposes");
        assert_eq!(comps, vec!["foo", "bar"]);
        assert!(out.diagnostics.is_empty());
    }

    #[test]
    fn decompose_empty_path_emits_diagnostic() {
        let mut out = StepOutput::default();
        let result = decompose_attach_path(std::path::Path::new(""), &mut out);
        assert!(result.is_none());
        assert!(out.diagnostics.iter().any(|d| matches!(
            d,
            specter_core::Diagnostic::AttachPathInvalid { hint } if hint.contains("empty"),
        )));
    }

    #[test]
    fn decompose_path_with_curdir_is_rejected() {
        // Rust's `Path::components()` normalizes embedded `/./` away, but
        // preserves a leading `./`. Use `./foo` to actually exercise the
        // `Component::CurDir` arm.
        let mut out = StepOutput::default();
        let result = decompose_attach_path(std::path::Path::new("./foo"), &mut out);
        assert!(result.is_none());
        assert!(out.diagnostics.iter().any(|d| matches!(
            d,
            specter_core::Diagnostic::AttachPathInvalid { hint } if hint.contains("non-canonical"),
        )));
    }

    #[test]
    fn decompose_path_with_parentdir_is_rejected() {
        let mut out = StepOutput::default();
        let result = decompose_attach_path(std::path::Path::new("/var/../log"), &mut out);
        assert!(result.is_none());
        assert!(out.diagnostics.iter().any(|d| matches!(
            d,
            specter_core::Diagnostic::AttachPathInvalid { hint } if hint.contains("non-canonical"),
        )));
    }

    #[test]
    fn decompose_root_only_path_is_single_segment() {
        let mut out = StepOutput::default();
        let comps = decompose_attach_path(std::path::Path::new("/"), &mut out)
            .expect("root-only path decomposes");
        assert_eq!(comps, vec![FS_ROOT_SEG]);
    }

    // ===== set_watch_root_parent idempotence =====

    #[test]
    fn set_watch_root_parent_idempotent_on_recovery_path() {
        // "Watch root deletion" recovery re-enters descent on a Profile
        // whose `watch_root_parent` was set at the original materialization.
        // When recovery's descent advances back to anchor materialization,
        // set_watch_root_parent must not double-bump the parent's
        // watch_demand.
        let mut e = Engine::new();
        let parent = e.tree.ensure(None, "p", specter_core::ResourceRole::User);
        let anchor = e
            .tree
            .ensure(Some(parent), "a", specter_core::ResourceRole::User);
        let profile = specter_core::Profile::new(
            anchor,
            ScanConfig::builder().build(),
            Duration::from_secs(1),
            Duration::from_millis(50),
            specter_core::ClassSet::EMPTY,
        );
        let pid = e.profiles.attach(&mut e.tree, profile);

        // First call: bumps parent's watch_demand and caches it on Profile.
        let mut out = StepOutput::default();
        e.set_watch_root_parent(pid, anchor, &mut out);
        let after_first = e.tree.get(parent).unwrap().watch_demand;
        assert_eq!(after_first, 1, "first call bumps parent watch_demand");
        assert_eq!(e.profiles.get(pid).unwrap().watch_root_parent, Some(parent));

        // Second call with the same anchor: must be a no-op (no bump).
        let mut out2 = StepOutput::default();
        e.set_watch_root_parent(pid, anchor, &mut out2);
        let after_second = e.tree.get(parent).unwrap().watch_demand;
        assert_eq!(after_second, 1, "second call does NOT double-bump");
        assert!(
            out2.watch_ops.is_empty(),
            "no Watch op emitted on second call"
        );
    }
}
