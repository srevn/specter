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
use crate::timer::{TimerEntry, TimerHeap};
use specter_core::{
    AnchorClaim, BurstPhase, ClassSet, DedupKey, DescentState, Diagnostic, Input, ProbeOwner,
    Profile, ProfileId, ProfileMap, ProfileState, PromoterRegistry, PromoterState, ResourceId,
    StepOutput, Sub, SubAttachRequest, SubId, SubRegistry, TimerId, TimerKind, Tree,
    compute_config_hash,
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
/// Per-owner descent state lives inline on the owner's state enum
/// (`ProfileState::Pending(DescentState)` for Profiles,
/// `PromoterState::PrefixPending(DescentState)` for Promoters). Read through
/// the owner-polymorphic `Engine::descent_state` / `Engine::descent_state_mut`
/// (both `pub(crate)`); per-event fan-out lives next to its sole consumer
/// (`Engine::classify_event_carriers` in `transitions.rs`).
#[derive(Debug, Default)]
pub struct Engine {
    pub(crate) tree: Tree,
    pub(crate) profiles: ProfileMap,
    pub(crate) subs: SubRegistry,
    /// Engine-resident dynamic-watch sources. Empty during Phase 4 (the
    /// data shapes ship before any lifecycle code references them);
    /// `recompute_resource_events` accepts a borrow of this field so the
    /// signature is stable as Phase 5+ wires in actual Promoter
    /// contributions to per-Resource `events_union`.
    pub(crate) promoters: PromoterRegistry,
    pub(crate) timers: TimerHeap,
    pub(crate) next_correlation: u64,
}

impl Engine {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Owner-polymorphic descent state accessor. Returns the `DescentState`
    /// payload of the owner's "in-descent" state variant
    /// (`ProfileState::Pending` for Profiles, `PromoterState::PrefixPending`
    /// for Promoters). Returns `None` for owners not currently descending,
    /// stale ids, or any other state.
    ///
    /// Sole reader API for the descent-state payload outside the routing
    /// match sites in `on_*_probe_response`. The exhaustive `ProbeOwner`
    /// match enforces that adding a new owner kind requires extending
    /// this accessor — the same discipline `pending_slot` enforces in
    /// `probe_channel.rs`.
    #[must_use]
    pub(crate) fn descent_state(&self, owner: ProbeOwner) -> Option<&DescentState> {
        match owner {
            ProbeOwner::Profile(pid) => match &self.profiles.get(pid)?.state {
                ProfileState::Pending(d) => Some(d),
                ProfileState::Idle | ProfileState::Active(_) => None,
            },
            ProbeOwner::Promoter(pid) => match &self.promoters.get(pid)?.state {
                PromoterState::PrefixPending(d) => Some(d),
                PromoterState::Active { .. } => None,
            },
        }
    }

    /// Mutable counterpart to [`Engine::descent_state`].
    pub(crate) fn descent_state_mut(&mut self, owner: ProbeOwner) -> Option<&mut DescentState> {
        match owner {
            ProbeOwner::Profile(pid) => match &mut self.profiles.get_mut(pid)?.state {
                ProfileState::Pending(d) => Some(d),
                ProfileState::Idle | ProfileState::Active(_) => None,
            },
            ProbeOwner::Promoter(pid) => match &mut self.promoters.get_mut(pid)?.state {
                PromoterState::PrefixPending(d) => Some(d),
                PromoterState::Active { .. } => None,
            },
        }
    }

    /// Pure, deterministic, total. Consumes one [`Input`], emits a sorted
    /// [`StepOutput`]. Each variant routes to the corresponding
    /// `on_*` handler (`transitions.rs`). Exhaustive — adding a variant
    /// to [`Input`] is a compile error here until a handler lands.
    pub fn step(&mut self, input: Input, now: Instant) -> StepOutput {
        let mut out = StepOutput::default();
        match input {
            Input::FsEvent { resource, event } => {
                self.on_fs_event(resource, event, now, &mut out);
            }
            Input::ProbeResponse(resp) => {
                self.on_probe_response(resp, now, &mut out);
            }
            Input::TimerExpired { profile, kind, id } => {
                self.on_timer_expired(profile, kind, id, now, &mut out);
            }
            Input::EffectComplete { sub, key, result } => {
                self.on_effect_complete(sub, &key, &result, now, &mut out);
            }
            Input::WatchOpRejected {
                resource,
                op,
                failure,
            } => {
                self.on_watch_op_rejected(resource, op, failure, &mut out);
            }
            Input::ConfigDiff(diff) => {
                self.on_config_diff(diff, now, &mut out);
            }
            Input::SensorOverflow { scope } => {
                self.on_sensor_overflow(scope, now, &mut out);
            }
        }
        out.sort_for_emission();
        out
    }

    /// Attach a Sub to an existing Resource (`req.resource`) or to a
    /// path that the engine materialises (`req.path`). Reuses an
    /// existing Profile when `(resource, config_hash)` matches;
    /// otherwise creates a fresh Profile, emits `WatchOp::Watch` on its
    /// anchor, and starts a `Burst { intent: Seed, phase: Verifying }`
    /// to establish the initial baseline.
    ///
    /// **Zombie revival.** When the matched Profile is in deferred-reap
    /// state (`reap_pending == true`, set by `detach_sub_inner` when the
    /// last Sub detached during an Active burst), the attach revives it:
    /// the flag clears, [`Diagnostic::ReapPendingCancelled`] emits, and
    /// the cleanup the deferred detach skipped
    /// (`recompute_profile_settle`, dead-id `fired_subs` purge) runs. The
    /// in-flight burst continues to completion under the new Sub set.
    ///
    /// Returns the minted [`SubId`] and a sorted [`StepOutput`]. On a
    /// path-rejection short-circuit (see invariants below), returns
    /// `SubId::default()` plus a sorted [`StepOutput`] carrying a
    /// [`Diagnostic::AttachPathInvalid`] and no other ops.
    ///
    /// # Production invariants (path-based attach)
    ///
    /// 1. **Absolute paths only.** `req.path` must be absolute and
    ///    UTF-8. The internal `decompose_attach_path` is the canonical
    ///    gate; it rejects non-absolute paths, non-UTF-8 segments,
    ///    `.` / `..` components, Windows path prefixes, and empty
    ///    segments. The bin layer's `canonicalize_lenient` already
    ///    enforces absolute paths for TOML-loaded configs, but
    ///    hot-reload `ConfigDiff::added` constructs `SubAttachRequest`
    ///    from a different path; the gate keeps the engine's contract
    ///    independent of every caller.
    /// 2. **Single FS-root.** Every absolute attach decomposes to
    ///    `[FS_ROOT_SEG, ...]`; `materialize_path_or_pending` lazily
    ///    bootstraps a synthetic `/` slot (role
    ///    `ResourceRole::DescentScaffold`) before the pre-existence
    ///    walk so every Profile's rewind chain terminates at this
    ///    shared slot. The FS-root invariant is documented here rather
    ///    than enforced at the Tree type level — unit tests for
    ///    lower-level Tree functions (`coverage`, `refcounts`) still
    ///    construct multi-root trees outside of `attach_sub`.
    /// 3. **`Tree::path_of` reconstructs absolute paths.**
    ///    `PathBuf::push("/")` resets the buffer to absolute, so the
    ///    Sensor's `WatchOp::Watch { path }` always carries an absolute
    ///    path for any Profile registered via `attach_sub`.
    ///
    /// # Panics
    /// Panics if `req.resource` is stale (no live Tree slot) on the
    /// resource-based attach path. The Engine must construct the
    /// Resource before attaching a Sub to it.
    pub fn attach_sub(&mut self, req: SubAttachRequest, now: Instant) -> (SubId, StepOutput) {
        let mut out = StepOutput::default();
        let sub_id = self.attach_sub_inner(req, now, &mut out);
        out.sort_for_emission();
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
        // Find-or-create. The branch decision is the structural source of
        // truth for "is this Profile newly minted in this attach call?" —
        // every derived predicate (`sub_refcount == 0`,
        // `anchor_claim == None`) is ambiguous against a zombie Profile
        // awaiting deferred reap (`reap_pending == true`, `anchor_claim`
        // still `Held`). Capturing the decision here makes the
        // fresh-Profile branch structurally unreachable for any existing
        // Profile.
        let (profile_id, is_fresh_profile) = if let Some(pid) = self.profiles.find(anchor, cfg_hash)
        {
            (pid, false)
        } else {
            let p = Profile::new(
                anchor,
                req.config.clone(),
                req.max_settle,
                req.settle,
                req.events,
            );
            let pid = self.profiles.attach(&mut self.tree, p);
            // Cache the anchor's classified kind on the Profile. `None` for
            // a `DescentScaffold` anchor (Pending path; descent's
            // materialisation branch writes the field) or a freshly
            // `ensure`'d-but-unprobed slot (the first Seed-Ok fallback in
            // `dispatch_seed_ok` writes the field). Existing Profiles —
            // `find` branch above — already carry the field from their
            // own first-classify moment; refreshing here would either
            // no-op or trample the canonical first observation.
            let anchor_kind = self.tree.get(anchor).and_then(specter_core::Resource::kind);
            if let Some(p) = self.profiles.get_mut(pid) {
                p.kind = anchor_kind;
            }
            (pid, true)
        };

        // Insert the Sub. `source_promoter` is `None` for static
        // (operator-declared) attaches and `Some(promoter_id)` for
        // dynamic attaches synthesised by a Promoter's `try_promote`
        // (Phase 5+); the request carries the stamp.
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
                req.log_output,
                req.source_promoter,
            )
        });

        if let Some(p) = self.profiles.get_mut(profile_id) {
            p.sub_refcount = p.sub_refcount.saturating_add(1);
        }

        if !is_fresh_profile {
            // Existing-Profile branch. Two semantic sub-cases:
            //
            //   (a) Normal join: the Profile holds live Subs; `Profile.settle`
            //       already aggregates min over them. O(1) min-update.
            //   (b) Zombie revival: the Profile lost its last Sub during an
            //       Active burst, and `detach_sub_inner` set
            //       `reap_pending = true` to defer reap to
            //       `finish_burst_to_idle`. The deferred-reap branch
            //       deliberately skipped the cleanup the refcount>0 path
            //       performs (`fired_subs` purge,
            //       `recompute_profile_settle`). The fresh attach revives
            //       the Profile; we now run that cleanup symmetrically and
            //       clear `reap_pending` so the burst doesn't reap a
            //       Profile that now holds a live Sub.
            //
            // The events mask folds into `config_hash`, so a Sub joining
            // an existing Profile shares its mask by construction —
            // `events_union` and `has_per_file_fds` are invariant for the
            // Profile's lifetime. No retroactive per-leaf `watch_demand`
            // bump is needed.
            let was_zombie = self
                .profiles
                .get(profile_id)
                .is_some_and(|p| p.reap_pending);
            if was_zombie {
                if let Some(p) = self.profiles.get_mut(profile_id) {
                    p.reap_pending = false;
                }
                out.diagnostics.push(Diagnostic::ReapPendingCancelled {
                    profile: profile_id,
                });
                // Recompute over the live Sub set (just the attaching Sub
                // on first revival; further attaches in the same step
                // take the normal-join arm because `reap_pending` is
                // already cleared).
                self.recompute_profile_settle(profile_id);
                // Drop dead-id `fired_subs` entries the deferred-reap
                // detach skipped. Functionally inert (`emit_effects`
                // iterates live SubIds), but a memory hygiene call.
                self.purge_dead_fired_subs(profile_id);
            } else if let Some(p) = self.profiles.get_mut(profile_id)
                && req.settle < p.settle
            {
                p.settle = req.settle;
            }
            return sub_id;
        }

        // ===== Fresh Profile path =====

        // Capture the Profile's mask before any &mut borrows. Used as the
        // anchor's contribution (immediate-Seed path); the descent prefix
        // path uses `STRUCTURE` instead.
        let events_union = self
            .profiles
            .get(profile_id)
            .map_or(ClassSet::EMPTY, |p| p.events_union);

        if let Some((prefix, remaining)) = pending_components {
            // Pending descent. Profile.state stays Idle while the descent
            // runs; the anchor materializes via `dispatch_descent_ok`'s
            // anchor branch, which then sets up the watch-root-parent
            // contribution and starts the Seed burst. Setting
            // watch_root_parent here would bump watch_demand on a
            // `DescentScaffold` slot that doesn't exist on disk yet,
            // generating a `WatchOpRejected` from the Sensor.
            //
            // Parent-edge work runs ahead of `enter_pending_descent` — the
            // helper deliberately omits it (the recovery path's call site
            // doesn't need it) and the Idle → Pending refcount sequence
            // (add_watch_demand → mint → state → emit) lives inside the
            // helper.
            self.compute_and_set_parent_edge(profile_id);
            self.recompute_dependent_parent_edges(profile_id);
            self.enter_pending_descent(profile_id, prefix, remaining, out);
        } else {
            // Immediate-Seed path. Anchor exists; bump its watch_demand
            // with the Profile's events_union (the user-declared mask),
            // set up the watch-root parent (STRUCTURE), compute parent
            // edges, start the Seed burst.
            add_watch_demand(&mut self.tree, anchor, events_union, out);
            if let Some(p) = self.profiles.get_mut(profile_id) {
                p.anchor_claim = AnchorClaim::Held;
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

        // The watch-root parent is engine infrastructure (used to detect
        // anchor reappearance after a `rm -rf` of the anchor).
        // Contribution is `STRUCTURE` regardless of the Sub's user mask.
        // The corresponding bookkeeping flag is `Profile.watch_root_parent
        // == Some(parent_id)`, written below; the recompute path reads
        // that flag to attribute this contribution back to the Profile.
        add_watch_demand(&mut self.tree, parent_id, ClassSet::STRUCTURE, out);

        if let Some(p) = self.profiles.get_mut(profile_id) {
            p.watch_root_parent = Some(parent_id);
        }
    }

    /// Compute and store the parent edge for a fresh Profile via the
    /// `coverage::nearest_covering_ancestor` derivation: walk Resource
    /// ancestors of the Profile's anchor; the smallest covering
    /// [`ProfileId`] wins by deterministic tie-break. Routes through
    /// `stability::write_parent_edge` so the self-parent
    /// `debug_assert_ne!` lives at a single source.
    fn compute_and_set_parent_edge(&mut self, profile_id: ProfileId) {
        let parent =
            crate::coverage::nearest_covering_ancestor(&self.tree, &self.profiles, profile_id);
        crate::stability::write_parent_edge(&mut self.profiles, profile_id, parent);
    }

    /// After adding a fresh Profile, recompute parent edges of every
    /// other Profile that the new one might interpose. Narrowed to
    /// strict descendants of the new anchor: a Profile P' can only
    /// re-parent to the new Profile if its anchor is a strict
    /// descendant of `new_profile.resource` (the new Profile is a
    /// covering ancestor of P' and may interpose between P' and its
    /// previous parent). Profiles at the same anchor (different
    /// `config_hash`), at sibling subtrees, or at ancestor positions
    /// cannot be affected by the new attach.
    ///
    /// O(N × depth) — N profiles each tested by ancestor-walk against
    /// the new anchor. The pre-narrowing form was O(N²) at the
    /// `compute_parent` level; this filter is the critical-path
    /// reduction.
    fn recompute_dependent_parent_edges(&mut self, new_profile: ProfileId) {
        let Some(new_anchor) = self.profiles.get(new_profile).map(|p| p.resource) else {
            return;
        };
        let candidates: Vec<ProfileId> = self
            .profiles
            .iter()
            .filter(|(pid, _)| *pid != new_profile)
            .filter_map(|(pid, p)| {
                self.tree
                    .ancestors(p.resource)
                    .any(|a| a == new_anchor)
                    .then_some(pid)
            })
            .collect();
        crate::stability::recompute_parent_edges_for_subset(
            &self.tree,
            &mut self.profiles,
            candidates,
        );
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
    ///   reaps the Profile in the same step as the Active → Idle
    ///   transition (any pre-fire phase converges through
    ///   `finish_burst_to_idle`).
    ///
    /// If the count remains > 0, the Profile stays alive; only
    /// `Profile.settle` is recomputed.
    ///
    /// Idempotent on stale `SubId` (Diagnostic + drop). Returns the sorted
    /// `StepOutput` of any ops emitted.
    pub fn detach_sub(&mut self, sub: SubId, now: Instant) -> StepOutput {
        let mut out = StepOutput::default();
        self.detach_sub_inner(sub, now, &mut out);
        out.sort_for_emission();
        out
    }

    /// Inner detach used by `on_config_diff` to compose multiple
    /// detach/attach operations into a single (sorted) `StepOutput`.
    pub(crate) fn detach_sub_inner(&mut self, sub: SubId, _now: Instant, out: &mut StepOutput) {
        let profile_id = match self.subs.remove(sub) {
            Some(s) => s.profile,
            None => {
                out.diagnostics.push(Diagnostic::DetachUnknownSub { sub });
                return;
            }
        };

        let Some(p) = self.profiles.get_mut(profile_id) else {
            return;
        };
        p.sub_refcount = p.sub_refcount.saturating_sub(1);
        let new_refcount = p.sub_refcount;

        // Purge `fired_subs` entries keyed by the detached Sub. The fire
        // history must drop with the Sub: a future drift verdict on the
        // Profile must not re-fire an Effect for a Sub the user has
        // detached. The full reap path below drops the whole set
        // alongside the Profile, so this targeted purge runs only on the
        // refcount-still-positive branch.
        if new_refcount > 0 {
            if let Some(p) = self.profiles.get_mut(profile_id) {
                p.fired_subs.retain(|k| match k {
                    DedupKey::Subtree { sub: s, .. } | DedupKey::PerFile { sub: s, .. } => {
                        *s != sub
                    }
                });
            }
            // Recompute Profile.settle = min(remaining_subs.settles).
            //
            // Every Sub on a Profile shares the same `events` mask
            // (events folds into `config_hash`); detaching one Sub
            // cannot flip `Profile.has_per_file_fds` or
            // `Profile.events_union`.
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
        });
        match lifecycle {
            Some(DetachLifecycle::ReapNow) => {
                self.reap_profile(profile_id, out);
            }
            Some(DetachLifecycle::DeferToBurstEnd) => {
                // `fired_subs` purge and `recompute_profile_settle` are
                // deliberately skipped — the Profile is about to drop on
                // burst end, so the cleanup would be wasted. A revival
                // via fresh `attach_sub_inner` (zombie-revival branch)
                // un-defers the reap and runs the cleanup symmetrically.
                if let Some(p) = self.profiles.get_mut(profile_id) {
                    p.reap_pending = true;
                }
            }
            None => {}
        }
    }

    /// Reap a Profile: release every contribution it holds (anchor watch,
    /// watch-root parent watch, descent prefix watch, per-descendant
    /// watches), clear its parent edge, recompute parent edges of any
    /// dependents, detach from `ProfileMap`, try-reap the anchor Resource,
    /// and emit a `ReapPendingResolved` Diagnostic.
    ///
    /// **Quartet.** A Profile may hold up to four kinds of contribution
    /// to per-Resource `watch_demand`:
    ///
    ///   - **Anchor** (1-to-1): `Profile.resource.watch_demand` carries
    ///     `+1` from this Profile while
    ///     `Profile.anchor_claim == AnchorClaim::Held`.
    ///   - **Watch-root parent** (1-to-1): `Profile.watch_root_parent`'s
    ///     resource carries `+1` `STRUCTURE` for anchor-reappearance
    ///     detection.
    ///   - **Descent prefix** (1-to-1): the deepest existing prefix on
    ///     a Pending Profile's path carries `+1` `STRUCTURE`.
    ///   - **Per-descendant** (1-to-N): every covered Tree slot in
    ///     `Profile.current` carries `+1` (Dir always; Leaf under
    ///     `has_per_file_fds`). The 1-to-N source-of-truth is the
    ///     snapshot itself, not a per-Profile flag.
    ///
    /// **Trichotomy invariant** (preserved from prior shape, now within
    /// the quartet). Anchor and descent-prefix are mutually exclusive at
    /// any moment: either the Profile is `Pending` (descent prefix only)
    /// or materialized (anchor + descendants + watch-root parent). The
    /// clamp recovery path (`Input::WatchOpRejected`) leaves the Profile
    /// with no contributions; the purge fan-out cleans up the
    /// bookkeeping. Each release helper is idempotent and counter-aware,
    /// so the call order is "all four, in any sequence" — none of them
    /// fault if their corresponding bookkeeping is already cleared.
    ///
    /// **Note on `discard_anchor_state` overlap.** This helper performs
    /// `release_descendant_claim` + `release_anchor_claim` inline
    /// rather than via [`Engine::discard_anchor_state`]. The two
    /// helpers differ in purpose:
    ///
    /// - `discard_anchor_state` exists for the "anchor lost, Profile
    ///   lives" case — the seven `dispatch_*_vanished/failed` +
    ///   `finalize_anchor_lost` sites. Its `kind = None` and
    ///   `baseline = None` writes prepare the Profile for the next
    ///   Seed burst's probe-shape dispatch, and it deliberately
    ///   preserves `watch_root_parent` (the recovery channel).
    /// - `reap_profile` is "Profile dies entirely." There is no next
    ///   Seed burst — the Profile detaches on the line below the four
    ///   release helpers — so the `kind` and `baseline` writes that
    ///   `discard_anchor_state` would perform are wasted on a struct
    ///   about to drop. Reap also releases `watch_root_parent`, which
    ///   `discard_anchor_state` deliberately preserves.
    ///
    /// The structural overlap (both call `release_descendant_claim +
    /// release_anchor_claim`) is intentional; the field clears and
    /// `watch_root_parent` release are deliberately partitioned across
    /// the two helpers.
    ///
    /// Sole call sites: `detach_sub_inner` (Idle / Pending Profile,
    /// immediate reap) and `finish_burst_to_idle` (deferred reap when
    /// `reap_pending` was set mid-burst).
    pub(crate) fn reap_profile(&mut self, profile_id: ProfileId, out: &mut StepOutput) {
        let Some(p) = self.profiles.get(profile_id) else {
            return;
        };
        let anchor = p.resource;

        // Trichotomy invariant: Pending and AnchorClaim::Held are mutually
        // exclusive. Descent flips Pending → Idle and bumps the anchor
        // atomically in `dispatch_descent_ok`'s anchor branch.
        debug_assert!(
            !matches!(
                (&p.state, p.anchor_claim),
                (ProfileState::Pending(_), AnchorClaim::Held),
            ),
            "reap_profile: Pending + AnchorClaim::Held must be mutually exclusive",
        );

        // Close the probe channel BEFORE the descent-prefix helper
        // transitions the Profile to Idle. Idempotent: emits Cancel
        // iff a probe was in flight (Pending with a descent probe in
        // flight for this call path; Active+Verifying never reaches
        // `reap_profile`'s entry — `finish_burst_to_idle` runs
        // `reap_profile` only after the burst response cleared the
        // channel). Mirrors `on_watch_op_rejected`'s descent-purge
        // pattern.
        self.cancel_owner_probe(ProbeOwner::Profile(profile_id), out);

        // Release every claim this Profile may hold. Helpers are
        // idempotent — no-op when the corresponding flag / snapshot is
        // unset (or counter is zero, post-clamp). Order is by claim
        // cardinality: 1-to-1 prefixed claims first, then the 1-to-N
        // descendant walk, then the 1-to-1 anchor and parent. The
        // descendant walk relies on `Profile.current` being intact, so
        // it must run before any helper that clears the snapshot — but
        // all helpers in this quartet leave `current` alone except the
        // descendant helper itself (which `take()`s it).
        self.release_descent_prefix_claim(profile_id, out);
        self.release_descendant_claim(profile_id, out);
        self.release_anchor_claim(profile_id, out);
        self.release_watch_root_parent_claim(profile_id, out);

        // Detach the Profile from the registry. The Profile's
        // `parent_profile` field dies with the struct — no separate
        // clear step is needed. Dependents whose `parent_profile`
        // still points at the now-removed slot are rewritten by
        // `recompute_parent_edges_for_dependents` below.
        let _ = self.profiles.detach(&mut self.tree, profile_id);

        crate::stability::recompute_parent_edges_for_dependents(
            &self.tree,
            &mut self.profiles,
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

    /// Drop `Profile.fired_subs` entries keyed by `SubId`s no longer in
    /// the registry. Called from `attach_sub_inner`'s zombie-revival
    /// branch — `detach_sub_inner`'s deferred-reap path skips per-Sub
    /// purges (the Profile was about to drop), but a revival un-defers
    /// the reap and the leftover keys become stale-id residue.
    /// O(fired_subs × live_subs); both terms are small in v1.
    fn purge_dead_fired_subs(&mut self, profile_id: ProfileId) {
        let live: Vec<SubId> = self.subs.at(profile_id).to_vec();
        if let Some(p) = self.profiles.get_mut(profile_id) {
            p.fired_subs.retain(|k| match k {
                DedupKey::Subtree { sub, .. } | DedupKey::PerFile { sub, .. } => live.contains(sub),
            });
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
    /// dropped. The returned [`TimerEntry`] carries the owning profile,
    /// kind, and id together — the bin forwards it to
    /// [`Input::TimerExpired`] without any rediscovery.
    pub fn pop_expired(&mut self, now: Instant) -> Option<TimerEntry> {
        loop {
            let top = self.timers.peek_top()?;
            if top.deadline > now {
                return None;
            }
            let entry = self
                .timers
                .pop_top()
                .expect("peek_top returned Some; pop_top must too");
            if Self::is_timer_referenced(&self.profiles, entry.profile, entry.kind, entry.id) {
                return Some(entry);
            }
            // Stale — silently drop, continue draining.
        }
    }

    /// Whether `id` is the live timer for `profile`'s `kind` slot —
    /// `pop_expired` uses this to filter stale heap heads, and
    /// `on_timer_expired` re-runs it as defense-in-depth for direct
    /// `step(Input::TimerExpired)` callers (tests, fuzzers).
    ///
    /// `kind` narrows the comparison:
    /// - `Settle` checks the `BurstPhase::Batching { settle_timer }`
    ///   slot only — no settle timer exists in any other phase.
    /// - `BurstDeadline` checks the Burst-level field, but only while
    ///   the burst is in a pre-fire phase (`Batching` / `Verifying` /
    ///   `Draining`). Once the burst transitions to `Awaiting` the
    ///   deadline is moot (we have already fired); a stale fire is
    ///   dropped silently here so `handle_burst_deadline` is never
    ///   reached from a post-fire phase.
    /// - `AwaitGateDeadline` checks the `BurstPhase::Awaiting
    ///   { gate_deadline }` slot only. Late fires from `Rebasing` or
    ///   beyond (where the phase has already advanced) are dropped.
    ///
    /// Only `Active` Profiles schedule timers; `Idle` and `Pending`
    /// Profiles own none of these slots.
    pub(crate) fn is_timer_referenced(
        profiles: &ProfileMap,
        profile: ProfileId,
        kind: TimerKind,
        id: TimerId,
    ) -> bool {
        let Some(p) = profiles.get(profile) else {
            return false;
        };
        let ProfileState::Active(burst) = &p.state else {
            return false;
        };
        match kind {
            TimerKind::Settle => matches!(
                &burst.phase,
                BurstPhase::Batching { settle_timer } if *settle_timer == id,
            ),
            TimerKind::BurstDeadline => {
                burst.burst_deadline == id
                    && matches!(
                        &burst.phase,
                        BurstPhase::Batching { .. } | BurstPhase::Verifying | BurstPhase::Draining,
                    )
            }
            TimerKind::AwaitGateDeadline => matches!(
                &burst.phase,
                BurstPhase::Awaiting { gate_deadline, .. } if *gate_deadline == id,
            ),
        }
    }
}

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

/// Decompose an absolute, UTF-8 attach path into Tree segments, with
/// `RootDir` mapped to the synthetic [`FS_ROOT_SEG`] so the engine has a
/// single shared root for every attach. `Tree::path_of` reconstructs an
/// absolute path from this segment because `PathBuf::push("/")` resets
/// to absolute.
///
/// Returns `None` and emits [`Diagnostic::AttachPathInvalid`] for:
/// - non-absolute paths (engine requires fully-qualified paths; the bin
///   layer's `canonicalize_lenient` enforces this for TOML-loaded
///   configs, but hot-reload `ConfigDiff::added` and test-only attaches
///   can bypass the bin's discipline — the gate keeps the engine's
///   contract independent of every caller);
/// - paths with non-UTF-8 bytes (the Tree's segment store is
///   `&str`-keyed; the engine cannot represent non-UTF-8 segments);
/// - relative components `.` / `..` (caller must canonicalize before
///   attach);
/// - Windows path prefixes (Unix v1 only);
/// - empty path segments (defense-in-depth against hand-constructed
///   `PathBuf`s — `PathBuf` itself normalises double-slashes).
///
/// **Post-condition.** On `Some(comps)`, `comps[0] == FS_ROOT_SEG` and
/// every `comps[i]` is a non-empty UTF-8 string. `materialize_path_or_pending`
/// relies on this to skip the FS-root pre-existence check and bootstrap
/// the slot unconditionally.
pub(crate) fn decompose_attach_path<'a>(
    path: &'a std::path::Path,
    out: &mut StepOutput,
) -> Option<Vec<&'a str>> {
    if !path.is_absolute() {
        out.diagnostics.push(Diagnostic::AttachPathInvalid {
            path: path.to_path_buf(),
            hint: "path must be absolute (engine requires fully-qualified paths)",
        });
        return None;
    }

    // Single upfront UTF-8 check on the whole path. On Unix, `Path::to_str`
    // returns `Some` iff every byte is valid UTF-8; a `Some` result means
    // every `Component::Normal`'s byte-slice is also UTF-8. The loop body's
    // `s.to_str().expect(...)` is sound under this precondition.
    if path.to_str().is_none() {
        out.diagnostics.push(Diagnostic::AttachPathInvalid {
            path: path.to_path_buf(),
            hint: "non-UTF-8 path segment (engine requires UTF-8)",
        });
        return None;
    }

    let mut comps: Vec<&str> = Vec::with_capacity(path.components().count());
    for c in path.components() {
        match c {
            Component::RootDir => comps.push(FS_ROOT_SEG),
            Component::Normal(s) => {
                let name = s.to_str().expect("path UTF-8 verified above");
                if name.is_empty() {
                    out.diagnostics.push(Diagnostic::AttachPathInvalid {
                        path: path.to_path_buf(),
                        hint: "empty path segment",
                    });
                    return None;
                }
                comps.push(name);
            }
            Component::CurDir | Component::ParentDir => {
                out.diagnostics.push(Diagnostic::AttachPathInvalid {
                    path: path.to_path_buf(),
                    hint: "non-canonical attach path (`.`/`..`); canonicalize before attach",
                });
                return None;
            }
            Component::Prefix(_) => {
                out.diagnostics.push(Diagnostic::AttachPathInvalid {
                    path: path.to_path_buf(),
                    hint: "Windows path prefix not supported on Unix v1",
                });
                return None;
            }
        }
    }

    // The `is_absolute()` guard above guarantees `Component::RootDir` was
    // emitted, which puts `FS_ROOT_SEG` at `comps[0]`. Defense-in-depth
    // assertion against future regressions or hand-constructed paths
    // that confuse the components iterator.
    debug_assert!(
        !comps.is_empty() && comps[0] == FS_ROOT_SEG,
        "decompose_attach_path post-condition: absolute path → comps[0] == FS_ROOT_SEG",
    );

    Some(comps)
}

#[cfg(test)]
mod tests {
    use super::*;
    use specter_core::{
        DedupKey, EffectOutcome, FsEvent, Input, ProbeCorrelation, ProbeOutcome, ProbeResponse,
        ProfileId, ResourceId, ScanConfig, StepOutput, SubId, TimerId, TimerKind, WatchOp,
        WatchRegistryDiff,
    };
    use std::time::{Duration, Instant};

    // Compile-time `Send + Sync` check on `Engine`. The bin loop parks
    // `Engine` on its own thread; `Send + Sync` is load-bearing for that.
    const _: fn() = || {
        const fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Engine>();
    };

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
            owner: ProbeOwner::Profile(ProfileId::default()),
            correlation: ProbeCorrelation(0),
            outcome: ProbeOutcome::Vanished,
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
        let out = e.step(
            Input::TimerExpired {
                profile: ProfileId::default(),
                kind: TimerKind::Settle,
                id: TimerId::default(),
            },
            Instant::now(),
        );
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
                failure: specter_core::WatchFailure::Pressure { errno: 24 },
            },
            Instant::now(),
        );
        assert!(out.watch_ops.is_empty());
        assert!(out.probe_ops.is_empty());
        assert!(out.effects.is_empty());
        let has_diag = out.diagnostics.iter().any(|d| {
            matches!(
                d,
                specter_core::Diagnostic::WatchOpRejected {
                    failure: specter_core::WatchFailure::Pressure { errno: 24 },
                    ..
                }
            )
        });
        assert!(has_diag);
    }

    #[test]
    fn step_config_diff_with_empty_diff_is_noop() {
        let mut e = Engine::new();
        let out = e.step(
            Input::ConfigDiff(WatchRegistryDiff::default()),
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
        e.timers
            .schedule(when, ProfileId::default(), TimerKind::Settle);
        assert_eq!(e.next_deadline(), Some(when));
    }

    #[test]
    fn pop_expired_returns_none_when_top_in_future() {
        let mut e = Engine::new();
        let now = Instant::now();
        let when = now + Duration::from_secs(10);
        e.timers
            .schedule(when, ProfileId::default(), TimerKind::Settle);
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
        e.timers
            .schedule(past, ProfileId::default(), TimerKind::Settle);
        e.timers
            .schedule(past, ProfileId::default(), TimerKind::Settle);
        e.timers
            .schedule(past, ProfileId::default(), TimerKind::Settle);

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
        e.timers
            .schedule(past, ProfileId::default(), TimerKind::Settle);
        e.timers.schedule(
            now + Duration::from_secs(10),
            ProfileId::default(),
            TimerKind::Settle,
        );

        assert_eq!(e.pop_expired(now), None);
        assert_eq!(e.timers.len(), 1, "future entry remains");
        assert!(e.next_deadline().unwrap() > now);
    }

    #[test]
    fn mint_probe_correlation_is_monotonic_per_profile() {
        // Three Profiles, each minted once: the shared
        // `Engine.next_correlation` counter advances monotonically
        // across mints regardless of which Profile owns each open
        // channel. Pinning to discrete Profiles avoids the I5
        // double-open assertion (one open channel per Profile).
        let mut e = Engine::new();
        let r1 = e.tree.ensure(None, "x", specter_core::ResourceRole::User);
        let r2 = e.tree.ensure(None, "y", specter_core::ResourceRole::User);
        let r3 = e.tree.ensure(None, "z", specter_core::ResourceRole::User);
        let cfg = ScanConfig::builder().build();
        let pid1 = e.profiles.attach(
            &mut e.tree,
            specter_core::Profile::new(
                r1,
                cfg.clone(),
                Duration::from_secs(6),
                Duration::from_millis(50),
                specter_core::ClassSet::EMPTY,
            ),
        );
        let pid2 = e.profiles.attach(
            &mut e.tree,
            specter_core::Profile::new(
                r2,
                cfg.clone(),
                Duration::from_secs(6),
                Duration::from_millis(50),
                specter_core::ClassSet::EMPTY,
            ),
        );
        let pid3 = e.profiles.attach(
            &mut e.tree,
            specter_core::Profile::new(
                r3,
                cfg,
                Duration::from_secs(6),
                Duration::from_millis(50),
                specter_core::ClassSet::EMPTY,
            ),
        );

        let a = e
            .mint_owner_correlation(ProbeOwner::Profile(pid1))
            .expect("pid1 is live");
        let b = e
            .mint_owner_correlation(ProbeOwner::Profile(pid2))
            .expect("pid2 is live");
        let c = e
            .mint_owner_correlation(ProbeOwner::Profile(pid3))
            .expect("pid3 is live");
        assert!(a < b);
        assert!(b < c);
        assert_eq!(a, ProbeCorrelation(1));
        assert_eq!(b, ProbeCorrelation(2));
        assert_eq!(c, ProbeCorrelation(3));

        // Slots populated symmetrically.
        assert_eq!(e.pending_probe_for(ProbeOwner::Profile(pid1)), Some(a));
        assert_eq!(e.pending_probe_for(ProbeOwner::Profile(pid2)), Some(b));
        assert_eq!(e.pending_probe_for(ProbeOwner::Profile(pid3)), Some(c));
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
    fn decompose_empty_path_rejected_as_non_absolute() {
        // An empty `Path` is non-absolute on Unix; the gate's `is_absolute`
        // check fires before any component-level work, so the diagnostic's
        // hint is "absolute" rather than "empty". The empty-segment hint
        // is reserved for the `Component::Normal` branch's defense-in-depth
        // check (hand-constructed paths with empty components).
        let mut out = StepOutput::default();
        let result = decompose_attach_path(std::path::Path::new(""), &mut out);
        assert!(result.is_none());
        assert!(out.diagnostics.iter().any(|d| matches!(
            d,
            specter_core::Diagnostic::AttachPathInvalid { path, hint }
                if path == std::path::Path::new("") && hint.contains("absolute"),
        )));
    }

    // Note: `Component::CurDir` is structurally unreachable for absolute
    // paths — `Path::components()` normalises `./` away when it appears
    // after `RootDir`, and the `is_absolute()` gate rejects leading-`./`
    // paths before the loop runs. The `CurDir | ParentDir` match arm
    // remains as defense-in-depth, exercised in production only via
    // `ParentDir` (covered by the next test).

    #[test]
    fn decompose_path_with_parentdir_is_rejected() {
        let mut out = StepOutput::default();
        let result = decompose_attach_path(std::path::Path::new("/var/../log"), &mut out);
        assert!(result.is_none());
        assert!(out.diagnostics.iter().any(|d| matches!(
            d,
            specter_core::Diagnostic::AttachPathInvalid { path, hint }
                if path == std::path::Path::new("/var/../log") && hint.contains("non-canonical"),
        )));
    }

    #[test]
    fn decompose_root_only_path_is_single_segment() {
        let mut out = StepOutput::default();
        let comps = decompose_attach_path(std::path::Path::new("/"), &mut out)
            .expect("root-only path decomposes");
        assert_eq!(comps, vec![FS_ROOT_SEG]);
    }

    // ===== Boundary rejection tests =====
    //
    // Three rejection categories pin at the decomposition seam:
    // non-absolute paths, non-UTF-8 segments, and non-canonical
    // components. The tests below cover the categories not already
    // exercised above (`parentdir_is_rejected` covers the `..` case).

    #[test]
    fn decompose_relative_multi_segment_path_emits_diagnostic() {
        let mut out = StepOutput::default();
        let result = decompose_attach_path(std::path::Path::new("foo/bar"), &mut out);
        assert!(result.is_none());
        assert!(out.diagnostics.iter().any(|d| matches!(
            d,
            specter_core::Diagnostic::AttachPathInvalid { path, hint }
                if path == std::path::Path::new("foo/bar") && hint.contains("absolute"),
        )));
    }

    #[test]
    fn decompose_relative_single_segment_path_emits_diagnostic() {
        // The single-segment case is the one the dropped `None` branch
        // of `materialize_path_or_pending` used to handle as a degenerate
        // `prefix == anchor` fixture. Post-Group-D the gate rejects it
        // outright.
        let mut out = StepOutput::default();
        let result = decompose_attach_path(std::path::Path::new("foo"), &mut out);
        assert!(result.is_none());
        assert!(out.diagnostics.iter().any(|d| matches!(
            d,
            specter_core::Diagnostic::AttachPathInvalid { path, hint }
                if path == std::path::Path::new("foo") && hint.contains("absolute"),
        )));
    }

    #[cfg(unix)]
    #[test]
    fn decompose_path_with_non_utf8_segment_emits_diagnostic() {
        // Non-UTF-8 segments sneak in via `canonicalize` resolving through
        // a symlink whose target component holds non-UTF-8 bytes. The
        // pre-fix decomposer silently dropped these, attaching the engine
        // to the wrong directory; the gate now rejects with an explicit
        // diagnostic.
        use std::ffi::OsStr;
        use std::os::unix::ffi::OsStrExt;
        use std::path::PathBuf;

        let bad_seg = OsStr::from_bytes(&[0xFF, 0xFE]);
        let mut path = PathBuf::from("/foo");
        path.push(bad_seg);
        path.push("bar");

        let mut out = StepOutput::default();
        let result = decompose_attach_path(&path, &mut out);
        assert!(result.is_none());
        assert!(out.diagnostics.iter().any(|d| matches!(
            d,
            specter_core::Diagnostic::AttachPathInvalid { hint, .. }
                if hint.contains("non-UTF-8"),
        )));
    }

    #[test]
    fn attach_path_invalid_carries_offending_path() {
        let mut e = Engine::new();
        let bad = std::path::PathBuf::from("./relative/with/dot");
        let req = SubAttachRequest::for_path(
            "bad".to_string(),
            bad.clone(),
            ScanConfig::builder().recursive(false).build(),
            Duration::from_millis(100),
            Duration::from_millis(50),
            specter_core::CommandTemplate::new(std::iter::empty()),
            specter_core::EffectScope::SubtreeRoot,
            specter_core::ClassSet::default(),
            false,
        );
        let (_sub, out) = e.attach_sub(req, Instant::now());

        let saw = out.diagnostics.iter().any(|d| {
            matches!(
                d,
                specter_core::Diagnostic::AttachPathInvalid { path, .. } if path == &bad,
            )
        });
        assert!(saw, "AttachPathInvalid must carry the offending path");
    }

    /// End-to-end gate enforcement: a relative-path attach request rolls
    /// up no `SubId`, no Tree slots, and no Profile — only the diagnostic
    /// surfaces. Pins the contract that `attach_sub_inner`'s
    /// `decompose_attach_path` short-circuit is total: rejection
    /// produces `SubId::default()` and zero side-effects on engine state.
    #[test]
    fn attach_with_relative_path_emits_diagnostic_and_no_state() {
        let mut e = Engine::new();
        let pre_tree_len = e.tree.len();
        let pre_profile_count = e.profiles.len();

        let bad = std::path::PathBuf::from("relative/path");
        let req = SubAttachRequest::for_path(
            "rel".to_string(),
            bad.clone(),
            ScanConfig::builder().recursive(false).build(),
            Duration::from_millis(100),
            Duration::from_millis(50),
            specter_core::CommandTemplate::new(std::iter::empty()),
            specter_core::EffectScope::SubtreeRoot,
            specter_core::ClassSet::default(),
            false,
        );
        let (sid, out) = e.attach_sub(req, Instant::now());

        assert_eq!(sid, SubId::default(), "rejected attach mints no SubId");
        assert_eq!(e.tree.len(), pre_tree_len, "no Tree slots created");
        assert_eq!(e.profiles.len(), pre_profile_count, "no Profile attached");
        assert!(e.subs.is_empty(), "no Sub recorded in registry");
        assert!(out.watch_ops.is_empty(), "no watch ops emitted");
        assert!(out.probe_ops.is_empty(), "no probe ops emitted");
        assert!(out.effects.is_empty(), "no effects emitted");
        assert!(out.diagnostics.iter().any(|d| matches!(
            d,
            specter_core::Diagnostic::AttachPathInvalid { path, hint }
                if path == &bad && hint.contains("absolute"),
        )));
    }

    /// End-to-end counterpart for non-UTF-8 paths. The test fabricates a
    /// path with a non-UTF-8 segment via `OsStr::from_bytes` (Unix-only)
    /// and confirms the same total-rejection contract: no SubId, no Tree
    /// slots, no Profile.
    #[cfg(unix)]
    #[test]
    fn attach_with_non_utf8_path_emits_diagnostic_and_no_state() {
        use std::ffi::OsStr;
        use std::os::unix::ffi::OsStrExt;
        use std::path::PathBuf;

        let bad_seg = OsStr::from_bytes(&[0xFF, 0xFE]);
        let mut path = PathBuf::from("/foo");
        path.push(bad_seg);

        let mut e = Engine::new();
        let pre_tree_len = e.tree.len();
        let pre_profile_count = e.profiles.len();

        let req = SubAttachRequest::for_path(
            "bad".to_string(),
            path.clone(),
            ScanConfig::builder().recursive(false).build(),
            Duration::from_millis(100),
            Duration::from_millis(50),
            specter_core::CommandTemplate::new(std::iter::empty()),
            specter_core::EffectScope::SubtreeRoot,
            specter_core::ClassSet::default(),
            false,
        );
        let (sid, out) = e.attach_sub(req, Instant::now());

        assert_eq!(sid, SubId::default(), "rejected attach mints no SubId");
        assert_eq!(e.tree.len(), pre_tree_len, "no Tree slots created");
        assert_eq!(e.profiles.len(), pre_profile_count, "no Profile attached");
        assert!(e.subs.is_empty(), "no Sub recorded in registry");
        assert!(out.watch_ops.is_empty(), "no watch ops emitted");
        assert!(out.probe_ops.is_empty(), "no probe ops emitted");
        assert!(out.effects.is_empty(), "no effects emitted");
        assert!(out.diagnostics.iter().any(|d| matches!(
            d,
            specter_core::Diagnostic::AttachPathInvalid { path: p, hint }
                if p == &path && hint.contains("non-UTF-8"),
        )));
    }

    #[test]
    fn detach_unknown_sub_emits_dedicated_diagnostic() {
        let mut e = Engine::new();
        let bogus = SubId::default();
        let out = e.detach_sub(bogus, Instant::now());

        let saw_dedicated = out.diagnostics.iter().any(|d| {
            matches!(
                d,
                specter_core::Diagnostic::DetachUnknownSub { sub } if *sub == bogus,
            )
        });
        let saw_wrong = out.diagnostics.iter().any(|d| {
            matches!(
                d,
                specter_core::Diagnostic::EffectCompleteForUnknownSub { .. },
            )
        });
        assert!(saw_dedicated, "detach miss must emit DetachUnknownSub");
        assert!(
            !saw_wrong,
            "detach miss must NOT emit EffectCompleteForUnknownSub",
        );
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

    // ===== Zombie revival =====

    fn revival_attach_req(anchor: ResourceId, name: &str, settle: Duration) -> SubAttachRequest {
        SubAttachRequest::for_resource(
            name.into(),
            anchor,
            ScanConfig::builder().build(),
            Duration::from_secs(6),
            settle,
            specter_core::CommandTemplate::new(std::iter::empty()),
            specter_core::EffectScope::SubtreeRoot,
            ClassSet::EMPTY,
            false,
        )
    }

    #[test]
    fn attach_revives_reap_pending_profile() {
        // Detach A's Sub mid-Active to set `reap_pending = true`, then
        // re-attach B at the same `(anchor, config_hash)`. The revival
        // path must:
        //   - reuse A's Profile (same ProfileId),
        //   - leave the anchor's watch_demand at 1 (no double-bump),
        //   - emit no spurious Watch op for the anchor,
        //   - clear `reap_pending`,
        //   - keep `anchor_claim` Held,
        //   - recompute `Profile.settle` to B's settle (NOT min-update —
        //     A is gone, B is the only live Sub),
        //   - drop A's stale-id `fired_subs` entry,
        //   - emit `Diagnostic::ReapPendingCancelled`.
        let mut e = Engine::new();
        let r = e
            .tree
            .ensure(None, "anchor", specter_core::ResourceRole::User);
        e.tree.set_kind(r, specter_core::ResourceKind::Dir);
        let now = Instant::now();

        let (sid_a, _) = e.attach_sub(revival_attach_req(r, "A", Duration::from_millis(50)), now);
        let pid = e.subs().get(sid_a).unwrap().profile;
        let watch_demand_after_attach = e.tree.get(r).unwrap().watch_demand;
        assert_eq!(watch_demand_after_attach, 1, "anchor watch_demand from A");

        // Pre-populate a `fired_subs` entry keyed by A so we can verify
        // the revival's dead-id purge.
        e.profiles
            .get_mut(pid)
            .unwrap()
            .fired_subs
            .insert(DedupKey::Subtree {
                sub: sid_a,
                profile: pid,
            });

        // Detach A. Profile is Active → reap_pending=true; anchor watch
        // unchanged; `fired_subs` survives the deferred-reap branch.
        let _ = e.detach_sub(sid_a, now);
        assert!(e.profiles().get(pid).unwrap().reap_pending);
        assert_eq!(e.tree.get(r).unwrap().watch_demand, 1);
        assert_eq!(e.profiles().get(pid).unwrap().fired_subs.len(), 1);

        // Revive with B (settle=200ms; deliberately larger than A's stale
        // 50ms so the min-update would be visibly wrong).
        let (sid_b, attach_out) =
            e.attach_sub(revival_attach_req(r, "B", Duration::from_millis(200)), now);
        let pid_b = e.subs().get(sid_b).unwrap().profile;

        assert_eq!(pid_b, pid, "B reuses A's Profile");
        assert_eq!(
            e.tree.get(r).unwrap().watch_demand,
            1,
            "anchor watch_demand unchanged on revival (no double-bump)",
        );
        let p = e.profiles().get(pid).unwrap();
        assert!(!p.reap_pending, "reap_pending cleared");
        assert_eq!(p.anchor_claim, AnchorClaim::Held, "anchor_claim stays Held");
        assert_eq!(
            p.settle,
            Duration::from_millis(200),
            "settle recomputed to B's (only live Sub) — min-update would yield 50ms",
        );
        assert!(
            !p.fired_subs.iter().any(|k| matches!(
                k,
                DedupKey::Subtree { sub, .. } | DedupKey::PerFile { sub, .. } if *sub == sid_a,
            )),
            "A's dead-id fired_subs entry purged on revival",
        );
        let anchor_watch_ops: Vec<_> = attach_out
            .watch_ops
            .iter()
            .filter(|op| matches!(op, WatchOp::Watch { resource, .. } if *resource == r))
            .collect();
        assert!(
            anchor_watch_ops.is_empty(),
            "no spurious Watch op for the anchor on revival; got {anchor_watch_ops:?}",
        );
        assert!(
            attach_out.diagnostics.iter().any(|d| matches!(
                d,
                Diagnostic::ReapPendingCancelled { profile } if *profile == pid,
            )),
            "ReapPendingCancelled emitted on revival",
        );
    }

    #[test]
    fn finish_burst_to_idle_does_not_reap_revived_profile() {
        // After revival, the in-flight burst's lifecycle continues under
        // the new Sub set. When the probe responds and the burst ends,
        // `finish_burst_to_idle` must NOT call `reap_profile` (the
        // revival cleared `reap_pending`).
        let mut e = Engine::new();
        let r = e
            .tree
            .ensure(None, "anchor", specter_core::ResourceRole::User);
        e.tree.set_kind(r, specter_core::ResourceKind::Dir);
        let now = Instant::now();

        let (sid_a, attach_out) =
            e.attach_sub(revival_attach_req(r, "A", Duration::from_millis(50)), now);
        let pid = e.subs().get(sid_a).unwrap().profile;
        let seed_corr = attach_out
            .probe_ops
            .iter()
            .find_map(|op| match op {
                specter_core::ProbeOp::Probe { request } => Some(request.correlation()),
                specter_core::ProbeOp::Cancel { .. } => None,
            })
            .expect("attach emitted Probe");

        e.detach_sub(sid_a, now);
        let (sid_b, _) = e.attach_sub(revival_attach_req(r, "B", Duration::from_millis(50)), now);

        // Drive the in-flight Seed-Verifying burst to a terminal Vanished.
        // `dispatch_seed_vanished → finalize_anchor_lost → finish_burst_to_idle`
        // would reap if `reap_pending` were still set; the revival cleared
        // it, so the Profile transitions to Idle (anchor lost) and stays.
        let out = e.step(
            Input::ProbeResponse(ProbeResponse {
                owner: ProbeOwner::Profile(pid),
                correlation: seed_corr,
                outcome: ProbeOutcome::Vanished,
            }),
            now,
        );

        assert!(
            e.profiles().get(pid).is_some(),
            "Profile alive (revival pre-empted reap)",
        );
        assert!(e.subs().get(sid_b).is_some(), "B still attached");
        assert!(
            !out.diagnostics.iter().any(|d| matches!(
                d,
                Diagnostic::ReapPendingResolved { profile } if *profile == pid,
            )),
            "ReapPendingResolved must NOT emit for a revived Profile",
        );
    }
}
