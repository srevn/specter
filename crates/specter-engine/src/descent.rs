//! Pending-path descent.
//!
//! Pending descent runs **outside** the Burst lifecycle. A Profile whose
//! anchor doesn't yet exist on the filesystem lives in
//! `ProfileState::Pending(DescentState)`. The descent emits its own
//! probes through the engine's `ProbeChannel` (keyed by
//! `ProbeOwner::Profile(_)` with `OpenKind::ProfileDescent`), advances
//! one path component per probe response, and ends by materializing
//! the anchor — at which point the Profile transitions Pending → Idle
//! and immediately Idle → Active(Seed) to establish its baseline.
//!
//! **Why a parallel state machine?** Burst semantics don't fit:
//! - Probe target ≠ `Profile.resource` during descent (probes go to the
//!   deepest existing prefix, not the anchor).
//! - There's no Effect to fire — the Profile has no baseline yet.
//! - The settle timer (carried inside `PreFirePhase::Batching`) and
//!   `burst_deadline` are stability concerns; descent
//!   is event-driven (a `StructureChanged` at the prefix triggers a
//!   fresh probe with no settle wait).
//! - I5 stays intact: at most one outstanding probe per Profile.
//!   `Pending` and `Active` are mutually exclusive `ProfileState`
//!   variants (the compiler proves it); within `Pending`, an in-flight
//!   probe is signalled by an open entry in the engine's `ProbeChannel`
//!   under `ProbeOwner::Profile(_)` — same channel used by `Active`.
//!
//! **Lifecycle.**
//! 1. `Engine::attach_sub` with a path-based request walks the path; if
//!    any non-leaf segment was freshly created (`role =
//!    DescentScaffold`), the Profile transitions Idle → Pending. The
//!    deepest existing ancestor is `current_prefix`; the remaining path
//!    components await materialization.
//! 2. The engine bumps `current_prefix.watch_demand` and emits a
//!    `ProbeOp::Probe` at the prefix.
//! 3. `dispatch_descent_probe` consumes the response:
//!    - `Ok(snap)`: look for the next remaining component as a
//!      single-level child. Found and is the anchor → materialize
//!      (`set_role` to `User`, set kind, bump anchor's `watch_demand`,
//!      drop the prefix's, transition Pending → Idle, start a Seed
//!      burst). Found but not the anchor → advance descent one segment.
//!      Not found → await the next event.
//!    - `Vanished`: the prefix itself is gone. Sub the prefix's
//!      contribution; vacate; rewind to the next-existing ancestor;
//!      emit a fresh probe.
//!    - `Failed { errno }`: retain Pending state; emit Diagnostic; await
//!      next event.
//! 4. `on_descent_event` triggers a fresh probe (no settle) on
//!    `StructureChanged` at `current_prefix`. I5: drops the event if a
//!    probe is already in flight (channel open for this owner).

use crate::probe_channel::OpenKind;
use crate::refcounts::{add_watch, sub_watch, sub_watch_then_try_reap};
use compact_str::CompactString;
use specter_core::{
    ClassSet, ContribKey, DescentRemaining, DescentState, Diagnostic, DirSnapshot, EntryKind,
    FS_ROOT_SEGMENT, ProbeOwner, ProfileId, ProfileState, ResourceId, ResourceKind, ResourceRole,
    StepOutput, TreePath,
};
use std::time::Instant;

/// Result of `Engine::materialize_path_or_pending`. Either the entire
/// path resolved to a live Tree slot (the anchor exists; proceed with
/// the normal P4 Seed-burst flow) or the deepest existing prefix is an
/// ancestor (descent registers; remaining components are tracked).
pub(crate) enum MaterializeResult {
    /// All segments existed; the leaf is `User`-rooted.
    Materialized(ResourceId),
    /// Descent is needed. The leaf `ResourceId` is the anchor's
    /// (currently `DescentScaffold`-roled) slot; the engine registers
    /// `DescentState` keyed by the Profile's id once it's been minted.
    /// `remaining` is non-empty by [`DescentRemaining`]'s type
    /// invariant — `materialize_path_or_pending` reaches this variant
    /// only when `prefix_idx + 1 < components.len()`, guaranteeing
    /// `from_vec` succeeds.
    Pending {
        anchor: ResourceId,
        prefix: ResourceId,
        remaining: DescentRemaining,
    },
}

impl crate::Engine {
    /// Walk a validated [`TreePath`] into the Tree. The leaf is created
    /// with `ResourceRole::User`; non-leaf components are
    /// `ResourceRole::DescentScaffold` if freshly created (the existing
    /// `ensure_root` / `ensure_child` preserve existing roles, so an already-
    /// User parent stays User).
    ///
    /// Returns `Materialized` iff every segment was already a live Tree
    /// slot AND the leaf's role is `User` after the walk (i.e., no
    /// scaffolding was created). Otherwise returns `Pending` with the
    /// deepest existing ancestor as `prefix` and the remaining components
    /// as the descent path.
    ///
    /// "Deepest existing ancestor" is determined by Tree-side
    /// pre-existence: each component is `lookup`'d before the walk;
    /// the deepest `i` for which `lookup(path.segments()[..=i])`
    /// succeeded before the materialising `ensure_path` call is the
    /// prefix index. The FS-root bootstrap guarantees `i >= 0` for every
    /// absolute attach. Role plays no part in this decision — a slot
    /// that existed before the walk may be a `User` peer anchor, a
    /// `WatchRootParent` of some other Profile, or a `DescentScaffold`
    /// retained from an earlier Pending Profile's descent chain; any
    /// of those count as "pre-existing".
    ///
    /// **Pre-conditions are now type-enforced.** [`TreePath`]'s type
    /// invariants (non-empty; `segments()[0] == FS_ROOT_SEGMENT`) make
    /// the prior `debug_assert!` and release-mode degradation branch
    /// structurally impossible.
    pub(crate) fn materialize_path_or_pending(&mut self, path: &TreePath) -> MaterializeResult {
        // Borrow segments as `&[&str]` once for the Tree-side helpers
        // (`lookup`, `ensure_path`, `resolve_components`) which all key
        // on `&str`. One small allocation bounded by path depth.
        let components: Vec<&str> = path.segments().iter().map(CompactString::as_str).collect();

        // FS-root bootstrap. Unconditional: [`TreePath`]'s invariant
        // guarantees `components[0] == FS_ROOT_SEGMENT`, and
        // `ensure_root` is idempotent (returns the existing slot if a
        // root at `/` already exists). The role is `DescentScaffold`
        // on first creation; if a prior `User` attach at `/` already
        // promoted the slot, the preserve-existing-role contract
        // leaves it alone. Bootstrapping
        // unconditionally guarantees every Profile's rewind chain
        // terminates at this `/` slot — the kernel always `lstat`s `/`
        // successfully on Unix, so a `Vanished` response from a `/` probe
        // is impossible, making cascading parent destruction
        // (`rm -rf /a/b/c/d`) recoverable: the descent stays Pending at
        // `/` waiting for the cascade's bottom segment to reappear.
        self.tree
            .ensure_root(FS_ROOT_SEGMENT, ResourceRole::DescentScaffold);

        // Snapshot which segments existed BEFORE the walk so we can tell
        // freshly-scaffolded segments from already-existing ones. The
        // bootstrap above guarantees `components[0]` (FS-root) always
        // pre-exists.
        let mut pre_existed: Vec<bool> = Vec::with_capacity(components.len());
        let mut cur_lookup: Option<ResourceId> = None;
        for comp in &components {
            let id = self.tree.lookup(cur_lookup, comp);
            pre_existed.push(id.is_some());
            cur_lookup = id;
        }
        debug_assert!(
            pre_existed[0],
            "materialize_path_or_pending: FS-root bootstrap must make components[0] pre-exist",
        );

        // Now do the walk. `ensure_path` creates non-leaf as
        // `DescentScaffold`, leaf as `User`.
        let anchor = self
            .tree
            .ensure_path(&components, ResourceRole::User)
            .expect("TreePath::segments() is non-empty by type invariant");

        // Walk forward to find the deepest pre-existing prefix. The
        // bootstrap guarantees `pre_existed[0] == true`, so `prefix_idx`
        // is always at least `0` — no `Option<usize>` trichotomy is
        // needed.
        let mut prefix_idx: usize = 0;
        for (i, &existed) in pre_existed.iter().enumerate() {
            if existed {
                prefix_idx = i;
            } else {
                break;
            }
        }

        if prefix_idx + 1 == components.len() {
            // Whole path pre-existed. P4 immediate-Seed path.
            MaterializeResult::Materialized(anchor)
        } else {
            // Segments [0..=prefix_idx] pre-existed; [prefix_idx+1..] are
            // scaffolds. `ensure_path` above created every segment, so
            // `resolve_components` on any prefix is guaranteed to
            // succeed — convert from the prior `unwrap_or(anchor)` (which
            // masked an invariant violation) to `expect` with an explicit
            // contract message.
            let prefix = self
                .resolve_components(&components[..=prefix_idx])
                .expect("ensure_path created every component; prefix slice must resolve");
            // Reuse the already-validated `CompactString` segments from
            // [`TreePath`] rather than re-allocating from `&str`. Bounded
            // by path depth and lifts straight into [`DescentRemaining`].
            let remaining_vec: Vec<CompactString> = path.segments()[prefix_idx + 1..].to_vec();
            // `prefix_idx + 1 < components.len()` is structurally
            // guaranteed by the outer `if`, so `from_vec` always
            // succeeds here; `expect` documents the contract and gives
            // a precise panic message if a future refactor weakens the
            // outer guard.
            let remaining = DescentRemaining::from_vec(remaining_vec).expect(
                "materialize_path_or_pending: Pending branch with empty remaining is \
                 structurally impossible (prefix_idx + 1 < components.len())",
            );
            MaterializeResult::Pending {
                anchor,
                prefix,
                remaining,
            }
        }
    }

    /// Resolve a sequence of path components to its leaf `ResourceId`
    /// without mutating the Tree. Returns `None` if any segment doesn't
    /// resolve.
    pub(crate) fn resolve_components(&self, components: &[&str]) -> Option<ResourceId> {
        let mut cur: Option<ResourceId> = None;
        for comp in components {
            cur = Some(self.tree.lookup(cur, comp)?);
        }
        cur
    }

    /// Enter `ProfileState::Pending` against `prefix` with `remaining`
    /// path components (single-component segments, anchor last). Mints
    /// the probe correlation, flips the Profile to `Pending`, bumps the
    /// prefix's `STRUCTURE` `watch_demand` contribution, and emits the
    /// descent probe — the four-step Idle → Pending entry sequence as
    /// a single helper.
    ///
    /// **Ordering: mint → state-flip → add_watch → emit.** Symmetric with
    /// [`Self::materialize_profile_anchor`]'s state-before-refcount
    /// pattern. The mint runs *first* so a precondition violation
    /// (correlation slot already busy) leaves the engine with no side
    /// effects: no leaked `+1 STRUCTURE` contribution at the prefix, no
    /// state flip, no probe emission. State-flip *then* refcount keeps
    /// the contribution attribution coherent with the Profile's claim
    /// shape at the moment of the refcount edge.
    ///
    /// **Pre-condition.** Profile must be `Idle` with a closed probe
    /// channel. The debug_assert below catches any caller passing a
    /// non-Idle Profile or one with an open channel entry.
    ///
    /// **Caller responsibility.** Parent-edge work
    /// ([`Engine::install_parent_edges_for`]) is NOT done here — the
    /// fresh-attach path needs it on first entry (called from
    /// `bootstrap_pending`); the recovery path doesn't (the parent
    /// edges already exist on the recovering Profile). Keeping the
    /// helper minimal preserves that contract.
    ///
    /// **Recovery-overlap invariant.** When called from `start_pending_recovery`,
    /// the Profile already holds a `+1 STRUCTURE` contribution on the
    /// parent via `Profile.watch_root_parent` (set at the original anchor
    /// materialization, never cleared on `on_anchor_terminal_event`). This
    /// helper bumps `+1 STRUCTURE` again on the same resource, giving `+2`.
    /// At re-materialization the descent contribution is subbed and the
    /// `watch_root_parent` contribution persists — `set_watch_root_parent`
    /// is idempotent on the recovery path (`engine.rs::set_watch_root_parent`
    /// short-circuits when the cache already points at the same parent).
    pub(crate) fn enter_pending_descent(
        &mut self,
        profile_id: ProfileId,
        prefix: ResourceId,
        remaining: DescentRemaining,
        out: &mut StepOutput,
    ) {
        let owner = ProbeOwner::Profile(profile_id);
        debug_assert!(
            self.profiles
                .get(profile_id)
                .is_some_and(|p| matches!(p.state(), ProfileState::Idle))
                && self.probe_channel.correlation_for(owner).is_none(),
            "enter_pending_descent: Profile must be Idle with closed probe channel; \
             caller must invoke cancel_owner_probe (or take the response-dispatch path) \
             and release prior state before re-entering descent (profile = {profile_id:?})",
        );

        // Step 1: open the channel. Probe-channel opens panic on I5
        // breach (double-open); a stale Profile here would be a
        // programming error caught upstream.
        let correlation = self.probe_channel.open(owner, OpenKind::ProfileDescent);

        // Step 2: state-flip Idle → Pending. Done before the refcount
        // edge so any reader between this point and step 3 sees the
        // Profile's claim shape that the contribution will attribute
        // to (matches `materialize_profile_anchor`'s sequencing).
        if let Some(p) = self.profiles.get_mut(profile_id) {
            p.transition_state(ProfileState::Pending(DescentState::new(prefix, remaining)));
        }

        // Step 3: install the prefix's STRUCTURE contribution.
        add_watch(
            &mut self.tree,
            prefix,
            ContribKey::ProfileDescent(profile_id),
            ClassSet::STRUCTURE,
            out,
        );

        // Step 4: emit the descent probe at the prefix.
        let target_path = self.tree.path_of(prefix).unwrap_or_default();
        Self::emit_descent_probe(owner, correlation, target_path, out);
    }

    /// Dispatch a successful descent response. The walker honoured the
    /// `Descent` request shape and returned a single-level
    /// `Arc<DirSnapshot>` for the prefix; this routine looks up the next
    /// remaining segment by name and either advances descent one level,
    /// materializes the descent target (Profile anchor or Promoter active
    /// proxy), or awaits the next event.
    ///
    /// **Owner-polymorphic.** The dispatch body is shared between
    /// `ProbeOwner::Profile` (Profile pending-path descent ending in
    /// anchor materialisation) and `ProbeOwner::Promoter` (Promoter
    /// literal-prefix descent ending in `enter_active`). The two diverge
    /// only at the terminal arm and the per-owner diagnostic / cleanup
    /// for the (structurally unreachable) empty-remaining invariant
    /// breach. All other branches — walker-stamp guard, segment lookup,
    /// snapshot-not-present early-return, slot materialisation, and the
    /// non-terminal advance — are identical and operate via the
    /// owner-polymorphic descent state accessors.
    ///
    /// **Caller (`on_*_probe_response`).** The probe channel was
    /// closed (via `ProbeChannel::close_if`) before dispatch; this
    /// function may re-open it via `ProbeChannel::open` in the
    /// advance branch.
    pub(crate) fn dispatch_descent_ok(
        &mut self,
        owner: ProbeOwner,
        snapshot: &DirSnapshot,
        now: Instant,
        out: &mut StepOutput,
    ) {
        // Sample the head segment + arity from descent state, then drop
        // the borrow. We clone only the head (cheap when CompactString
        // stays inline); the tail mutation runs in place via
        // `descent_state_mut` later, no whole-vec rebuild.
        let Some(descent) = self.descent_state(owner) else {
            return;
        };
        let prefix = descent.current_prefix();

        // The walker echoes `(owner, correlation)` verbatim — the
        // probe channel's match in `on_*_probe_response` already
        // enforces request/response pairing, so any divergence would
        // surface as `StaleProbeResponse`, not reach this point. The
        // snapshot itself carries pure content; engine identity stays
        // engine-side (here, `descent.current_prefix()`).
        // `DescentRemaining` is non-empty by type invariant — the prior
        // defensive empty-arm + `descent_invariant_diagnostic` /
        // `release_owner_descent_prefix` recovery is no longer
        // reachable (and the corresponding `Diagnostic` variants have
        // been retired).
        let next_segment = descent.remaining_components().head().clone();
        let is_terminal = descent.remaining_components().is_terminal();

        // Descent probes ship `recursive=false`, so the response is a
        // single-level Dir snapshot — look up the next segment by name in
        // the BTreeMap directly.
        let entry_kind = match snapshot.entries.get(next_segment.as_str()) {
            Some(child) => child.kind(),
            None => {
                // Next segment not yet present; await next event. v1
                // descent doesn't mtime-skip, so no need to retain the
                // snapshot — the next probe will get a fresh
                // `lstat`-walked DirSnapshot anyway.
                return;
            }
        };

        // Materialize the next segment as a Tree slot. Look it up first;
        // if absent, ensure as DescentScaffold (the terminal arms may
        // promote to User via `set_role`).
        let new_resource = match self.tree.lookup(Some(prefix), &next_segment) {
            Some(r) => r,
            None => self
                .tree
                .ensure_child(prefix, &next_segment, ResourceRole::DescentScaffold)
                .expect(
                    "descent prefix held alive by ProfileDescent / PromoterPrefix contribution",
                ),
        };
        self.tree
            .set_kind(new_resource, kind_from_entry(entry_kind));

        if is_terminal {
            // Per-owner terminal action. Profile materialises its anchor
            // and starts a Seed burst; Promoter enters Active and queues
            // its first proxy enumeration. Both helpers release the
            // prefix's STRUCTURE contribution and install the new slot's
            // contribution as part of their own state-flip sequence.
            match owner {
                ProbeOwner::Profile(pid) => {
                    self.materialize_profile_anchor(
                        pid,
                        prefix,
                        new_resource,
                        entry_kind,
                        now,
                        out,
                    );
                }
                ProbeOwner::Promoter(pid) => {
                    let lpl = self
                        .promoters
                        .get(pid)
                        .map_or(0, |q| q.pattern.literal_prefix_len());
                    self.enter_active(pid, Some(prefix), new_resource, lpl, now, out);
                }
            }
        } else {
            self.advance_descent(owner, prefix, new_resource, out);
        }
    }

    /// Advance descent one literal segment. Shared body between the
    /// Profile and Promoter dispatch — the only divergence is which
    /// state's `DescentState` payload gets mutated, which the
    /// owner-polymorphic `descent_state_mut` already routes.
    ///
    /// Sequence:
    /// 1. Mint a fresh probe correlation for `owner` (opens channel).
    /// 2. Mutate descent state in place: advance `current_prefix` to
    ///    the new resource and drop the consumed head segment from
    ///    `remaining_components`.
    /// 3. Release the old prefix's STRUCTURE contribution; install the
    ///    new prefix's.
    /// 4. Emit the fresh descent probe at the new prefix.
    ///
    /// The old prefix stays alive because the freshly-advanced
    /// `new_prefix` is still its `children` entry — the structural
    /// anchor holds the slot across `sub_watch`. No `try_reap` here:
    /// the routine release helper would see a non-empty `children` map
    /// and short-circuit anyway, so we skip the call. (Role is metadata
    /// throughout — its tag stays `DescentScaffold` from the initial
    /// `ensure_child` but does not affect retention.)
    fn advance_descent(
        &mut self,
        owner: ProbeOwner,
        old_prefix: ResourceId,
        new_prefix: ResourceId,
        out: &mut StepOutput,
    ) {
        let correlation = self.probe_channel.open(owner, descent_open_kind(owner));
        if let Some(d) = self.descent_state_mut(owner) {
            d.advance_to(new_prefix);
            // Non-terminal by caller contract — `dispatch_descent_ok`
            // routes terminal descents through anchor materialization
            // before reaching `advance_descent`. The debug_assert
            // inside `DescentRemaining::advance` pins this for
            // regression detection.
            d.remaining_components_mut().advance();
        }

        let key = descent_key(owner);
        sub_watch(&mut self.tree, old_prefix, key, out);
        add_watch(&mut self.tree, new_prefix, key, ClassSet::STRUCTURE, out);

        let target_path = self.tree.path_of(new_prefix).unwrap_or_default();
        Self::emit_descent_probe(owner, correlation, target_path, out);
    }

    /// Owner-polymorphic descent-prefix release. Routes to the per-owner
    /// claim helper, both of which read the prefix from descent state
    /// (Profile: `Pending(d).current_prefix()`; Promoter:
    /// `PrefixPending(d).current_prefix()`).
    ///
    /// Per-owner cleanup (parallel shape):
    ///
    /// - **Profile.** Delegates to [`Self::release_descent_prefix_claim`]:
    ///   transitions `Pending → Idle`, releases the prefix's
    ///   [`specter_core::ContribKey::ProfileDescent`] contribution,
    ///   and `try_reap`s the prefix slot.
    /// - **Promoter.** Delegates to
    ///   [`Self::release_promoter_descent_prefix_claim`]: transitions
    ///   `PrefixPending → Active{empty}` for owner bookkeeping,
    ///   removes the
    ///   [`specter_core::ContribKey::PromoterPrefix`] contribution
    ///   by key, and `try_reap`s.
    ///
    /// Three call sites:
    /// - [`Self::dispatch_descent_ok`]'s structurally-unreachable
    ///   empty-remaining arm.
    /// - [`Self::dispatch_descent_vanished`]'s no-rewind-target arm
    ///   (FS-root vanish, structurally unreachable on Unix).
    /// - [`Self::on_watch_op_rejected`]'s descent-prefix purge loops
    ///   (Profile and Promoter sides).
    ///
    /// Tolerant of any post-vacate state — `sub_watch` silently
    /// skips an absent key, so the state-flip alone is the observable
    /// cleanup in those degenerate paths.
    fn release_owner_descent_prefix(&mut self, owner: ProbeOwner, out: &mut StepOutput) {
        match owner {
            ProbeOwner::Profile(pid) => self.release_descent_prefix_claim(pid, out),
            ProbeOwner::Promoter(pid) => self.release_promoter_descent_prefix_claim(pid, out),
        }
    }

    /// Promote `new_resource` to the Profile's anchor slot. Sole call site
    /// is [`Self::dispatch_descent_ok`]'s terminal arm — the descent has
    /// just resolved its last remaining segment and the Profile is about
    /// to leave `Pending` for `Idle → Active(Seed)`.
    ///
    /// Sequence (load-bearing):
    /// 1. Flip the slot's role to `User` (was `DescentScaffold` from the
    ///    descent walk).
    /// 2. Capture `Profile.events` for the anchor's contribution.
    /// 3. Transition the Profile **before** any refcount op via
    ///    [`specter_core::Profile::materialize_anchor`] — atomic
    ///    `Pending → Idle`, claim install, kind pin. The recompute
    ///    (multi-contributor case) reads `Profile.state` and
    ///    `Profile.anchor_claim` to attribute contributions; the
    ///    post-flip world has the prefix's STRUCTURE source gone (state
    ///    no longer Pending) and the anchor's mask source owed.
    /// 4. Sub the prefix's STRUCTURE; add the anchor's mask.
    /// 5. Install the watch-root-parent contribution (deferred from
    ///    `attach_sub_inner` because the parent didn't exist on disk
    ///    when the Profile attached).
    /// 6. Start the Seed burst.
    fn materialize_profile_anchor(
        &mut self,
        profile_id: ProfileId,
        prefix: ResourceId,
        new_resource: ResourceId,
        entry_kind: EntryKind,
        now: Instant,
        out: &mut StepOutput,
    ) {
        self.tree.set_role(new_resource, ResourceRole::User);

        let events_union = self
            .profiles
            .get(profile_id)
            .map_or(ClassSet::EMPTY, specter_core::Profile::events);

        let anchor_kind = kind_from_entry(entry_kind);
        if let Some(p) = self.profiles.get_mut(profile_id) {
            p.materialize_anchor(anchor_kind);
        }

        // Profile.resource was assigned to the anchor's slot at attach
        // time; the materialised slot's id should match by construction.
        debug_assert!(
            self.profiles
                .get(profile_id)
                .is_some_and(|p| p.resource == new_resource),
            "descent anchor materialization: Profile.resource diverges from descent anchor",
        );

        sub_watch(
            &mut self.tree,
            prefix,
            ContribKey::ProfileDescent(profile_id),
            out,
        );
        add_watch(
            &mut self.tree,
            new_resource,
            ContribKey::ProfileAnchor(profile_id),
            events_union,
            out,
        );

        self.set_watch_root_parent(profile_id, new_resource, out);
        self.start_seed_burst(profile_id, now, out);
    }

    /// **Rewind chain depth.** A `Vanished` response on a rewound prefix
    /// triggers a further rewind via the same path. The chain depth is
    /// bounded by the tree-distance from the original prefix to its
    /// ultimate ancestor — at most one rewind cycle per ancestor level.
    /// Each rewind step **adds** a `+1 STRUCTURE` `watch_demand` on the
    /// new prefix; in production the chain auto-extends watches up the
    /// ancestor chain until it reaches a still-present ancestor, whose
    /// probe returns `Ok` and routes to `dispatch_descent_ok`'s
    /// "next segment not yet present; await next event" branch.
    ///
    /// **Branch reachability post-bootstrap.** With the unconditional
    /// FS-root bootstrap in [`Self::materialize_path_or_pending`], every
    /// descent's rewind chain terminates at the FS-root slot `/`. The
    /// kernel always `lstat`s `/` successfully on Unix, so a `Vanished`
    /// response from a `/` probe is impossible — meaning the `None` arm
    /// below is structurally unreachable in production. A cascade like
    /// `rm -rf /a/b/c/d` with anchor at `/d` rewinds through `/c`, `/b`,
    /// `/a`, `/` and terminates on `/`'s `Ok` rather than reaching the
    /// arm; the descent stays Pending at `/` waiting for the cascade's
    /// bottom segment to reappear, which makes cascading parent
    /// destruction auto-recoverable. The arm is retained as
    /// defense-in-depth against kernel anomalies (e.g., a chrooted
    /// environment where `/` is somehow inaccessible) and to keep the
    /// recursion well-typed; tests must construct the state directly to
    /// exercise it.
    ///
    /// For an `N`-level cascade with a Profile anchored at the leaf,
    /// the engine emits up to `N` rewind cycles per Pending Profile
    /// (one Watch + one descent probe per cycle). Acceptable in v1.
    /// Owner-polymorphic Vanished response handler. Rewinds descent to
    /// the next-existing ancestor of `prefix`. Mirrors Profile and
    /// Promoter descents through the same body — the only divergence is
    /// the per-owner diagnostic and, in the structurally-unreachable
    /// "no rewind target" arm, the per-owner state-flip out of descent.
    ///
    /// **Bounded chain depth.** Each rewind step adds a `+1 STRUCTURE`
    /// `watch_demand` on the new prefix; the chain auto-extends watches
    /// up the ancestor chain until it reaches a still-present ancestor
    /// (whose probe returns `Ok` and routes to `dispatch_descent_ok`'s
    /// "next segment not yet present; await next event" branch). With
    /// FS-root bootstrap (`materialize_path_or_pending`'s unconditional
    /// ensure), every owner's rewind chain terminates at the FS-root
    /// slot `/` — the kernel always lstats `/` successfully on Unix, so
    /// `Vanished` from `/` is impossible in production. The `None` arm
    /// is retained as defense-in-depth and to keep the recursion
    /// well-typed; tests must construct the state directly to exercise
    /// it.
    pub(crate) fn dispatch_descent_vanished(
        &mut self,
        owner: ProbeOwner,
        _now: Instant,
        out: &mut StepOutput,
    ) {
        let Some(descent) = self.descent_state(owner) else {
            return;
        };
        let prefix = descent.current_prefix();

        out.diagnostics
            .push(descent_vanished_diagnostic(owner, prefix));

        // Capture the parent + the prefix's segment name BEFORE we mutate
        // descent state. `parent` selects the rewind branch; `prefix_name`
        // becomes the new first remaining component.
        let parent = self.tree.parent(prefix);
        let prefix_name = self.tree.name(prefix).map(CompactString::from);

        match parent {
            Some(parent_id) => {
                // Rewind. The vanished prefix's segment becomes the
                // *first* remaining component (we're descending into it
                // from the parent again).
                //
                // In-place mutation: prepend onto the existing
                // `remaining_components` rather than cloning + rebuilding
                // a fresh DescentState — saves both the whole-vec clone
                // and the per-element CompactString clone.
                let correlation = self.probe_channel.open(owner, descent_open_kind(owner));
                if let Some(d) = self.descent_state_mut(owner) {
                    d.advance_to(parent_id);
                    if let Some(name) = prefix_name {
                        d.remaining_components_mut().prepend(name);
                    }
                }

                let key = descent_key(owner);
                sub_watch_then_try_reap(&mut self.tree, prefix, key, out);

                add_watch(&mut self.tree, parent_id, key, ClassSet::STRUCTURE, out);

                let target_path = self.tree.path_of(parent_id).unwrap_or_default();
                Self::emit_descent_probe(owner, correlation, target_path, out);
            }
            None => {
                // Root prefix vanished — no rewind target. Delegate to
                // the per-owner release helper (state-flip terminal +
                // counter-aware sub + try_reap), matching the
                // empty-remaining arm in `dispatch_descent_ok`. The
                // helper's preconditions hold here: the probe channel
                // was closed by `on_*_probe_response` before dispatch
                // (cancel-first contract) and descent state is
                // unflipped at entry.
                //
                // The owner is left stuck without a usable descent path —
                // operator recovery is required (Profile: stuck Idle;
                // Promoter: stuck Active{empty}).
                self.release_owner_descent_prefix(owner, out);
            }
        }
    }

    /// Owner-polymorphic Failed response handler. Both Profile and
    /// Promoter descents retain in-descent state and emit a per-owner
    /// diagnostic; the next event at the prefix re-triggers via
    /// [`Self::on_descent_event`].
    pub(crate) fn dispatch_descent_failed(
        &self,
        owner: ProbeOwner,
        errno: i32,
        out: &mut StepOutput,
    ) {
        let prefix = match self.descent_state(owner) {
            Some(d) => d.current_prefix(),
            None => return,
        };
        out.diagnostics
            .push(descent_failed_diagnostic(owner, prefix, errno));
        // Retain in-descent state; await next event at the prefix.
    }

    /// Owner-polymorphic handler for `FsEvent` arriving at a descent's
    /// `current_prefix`. Triggers a fresh probe (no settle wait —
    /// descent is event-driven). I5: drops the event if a probe is
    /// already in flight (the in-flight probe will pick up the change
    /// in its response). The "in flight" signal is the open channel
    /// entry for this owner.
    pub(crate) fn on_descent_event(
        &mut self,
        owner: ProbeOwner,
        _now: Instant,
        out: &mut StepOutput,
    ) {
        if self.probe_channel.correlation_for(owner).is_some() {
            return;
        }
        let prefix = match self.descent_state(owner) {
            Some(d) => d.current_prefix(),
            None => return,
        };

        let correlation = self.probe_channel.open(owner, descent_open_kind(owner));
        let target_path = self.tree.path_of(prefix).unwrap_or_default();
        Self::emit_descent_probe(owner, correlation, target_path, out);
    }
}

pub(crate) const fn kind_from_entry(k: EntryKind) -> ResourceKind {
    match k {
        EntryKind::File | EntryKind::Symlink | EntryKind::Other => ResourceKind::File,
        EntryKind::Dir => ResourceKind::Dir,
    }
}

/// Owner-polymorphic [`ContribKey`] for the descent-prefix
/// contribution. Profiles in `Pending` claim
/// [`ContribKey::ProfileDescent`]; Promoters in `PrefixPending` claim
/// [`ContribKey::PromoterPrefix`]. Sole site that fans out the two
/// arms — `advance_descent` and `dispatch_descent_vanished` both use
/// the same key for sub-then-add on the descent state's prefix slot.
const fn descent_key(owner: ProbeOwner) -> ContribKey {
    match owner {
        ProbeOwner::Profile(pid) => ContribKey::ProfileDescent(pid),
        ProbeOwner::Promoter(qid) => ContribKey::PromoterPrefix(qid),
    }
}

/// Owner-polymorphic [`OpenKind`] for descent probes. Profile descents
/// open with [`OpenKind::ProfileDescent`]; Promoter descents with
/// [`OpenKind::PromoterDescent`]. Pairs with [`descent_key`]: both
/// fan-outs are 1-to-1 with the owner discriminant and mirror the
/// shape of `ProbeOwner` itself.
const fn descent_open_kind(owner: ProbeOwner) -> OpenKind {
    match owner {
        ProbeOwner::Profile(_) => OpenKind::ProfileDescent,
        ProbeOwner::Promoter(_) => OpenKind::PromoterDescent,
    }
}

/// Per-owner diagnostic emitted when a descent probe returns
/// `Vanished`. Profile ships [`Diagnostic::PendingPathProbeVanished`];
/// Promoter ships [`Diagnostic::PromoterDescentVanished`]. Sole caller
/// is [`Engine::dispatch_descent_vanished`].
const fn descent_vanished_diagnostic(owner: ProbeOwner, prefix: ResourceId) -> Diagnostic {
    match owner {
        ProbeOwner::Profile(profile) => Diagnostic::PendingPathProbeVanished { profile, prefix },
        ProbeOwner::Promoter(promoter) => Diagnostic::PromoterDescentVanished { promoter, prefix },
    }
}

/// Per-owner diagnostic emitted when a descent probe returns
/// `Failed { errno }`. Profile ships [`Diagnostic::PendingPathProbeFailed`];
/// Promoter ships [`Diagnostic::PromoterDescentFailed`]. Sole caller is
/// [`Engine::dispatch_descent_failed`].
const fn descent_failed_diagnostic(
    owner: ProbeOwner,
    prefix: ResourceId,
    errno: i32,
) -> Diagnostic {
    match owner {
        ProbeOwner::Profile(profile) => Diagnostic::PendingPathProbeFailed {
            profile,
            prefix,
            errno,
        },
        ProbeOwner::Promoter(promoter) => Diagnostic::PromoterDescentFailed {
            promoter,
            prefix,
            errno,
        },
    }
}

#[cfg(test)]
#[path = "descent_tests.rs"]
mod tests;
