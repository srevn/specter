//! Pending-path descent.
//!
//! Pending descent runs **outside** the Burst lifecycle. A Profile whose anchor doesn't yet exist
//! on the filesystem lives in `ProfileState::Pending(DescentState)`. The descent emits its own
//! probes correlated by the `ProbeSlot` on `DescentState` (armed at every mint, disarmed when the
//! response is consumed), advances one path component per probe response, and ends by materializing
//! the anchor — at which point the Profile transitions Pending → Idle and immediately Idle →
//! Active(Seed) to establish its baseline.
//!
//! **Why a parallel state machine?** Burst semantics don't fit:
//! - Probe target ≠ `Profile.resource` during descent (probes go to the deepest existing prefix,
//!   not the anchor).
//! - There's no Effect to fire — the Profile has no baseline yet.
//! - The settle timer (carried inside `PreFirePhase::Batching`) and `burst_deadline` are stability
//!   concerns; descent is event-driven (a `StructureChanged` at the prefix triggers a fresh probe
//!   with no settle wait).
//! - I5 stays intact: at most one outstanding probe per Profile. `Pending` and `Active` are
//!   mutually exclusive `ProfileState` variants (the compiler proves it); within `Pending`, an
//!   in-flight probe is exactly an armed `DescentState.probe` slot — one slot per descent, so two
//!   simultaneous descent probes are unconstructable.
//!
//! **Lifecycle.**
//! 1. `Engine::attach_sub` with a path-based request walks the path; if any non-leaf segment was
//!    freshly created (`role = DescentScaffold`), the Profile transitions Idle → Pending. The deepest
//!    existing ancestor is `current_prefix`; the remaining path components await materialization.
//! 2. The engine bumps `current_prefix.watch_demand` and emits a `ProbeOp::Probe` at the prefix.
//! 3. `dispatch_descent_probe` consumes the response:
//!    - `Ok(snap)`: look for the next remaining component as a single-level child. Found and is the
//!      anchor → materialize (promote to `User`, set kind, bump anchor's `watch_demand`, drop the
//!      prefix's, transition Pending → Idle, start a Seed burst). Found but not the anchor →
//!      advance descent one segment. Not found → await the next event.
//!    - `Vanished`: the prefix itself is gone. Sub the prefix's contribution; vacate; rewind to the
//!      next-existing ancestor; emit a fresh probe.
//!    - `Failed { errno }`: retain Pending state; emit Diagnostic; await next event.
//! 4. `on_descent_event` triggers a fresh probe (no settle) on `StructureChanged` at
//!    `current_prefix`. I5: drops the event if a probe is already in flight (the descent slot is
//!    armed).

use crate::probe::DescentOutcome;
use crate::refcounts::{add_watch, sub_watch, sub_watch_then_try_reap};
use compact_str::CompactString;
use specter_core::{
    ClassSet, ContribKey, DescentRemaining, DescentState, Diagnostic, DirSnapshot, EntryKind,
    FS_ROOT_SEGMENT, ProbeFailure, ProbeSlot, ProfileId, ProfileState, ResourceId, ResourceKind,
    ResourceRole, StepOutput, TreePath,
};
use std::time::Instant;

/// Result of `Engine::materialize_path_or_pending`. Either the entire path resolved to a live Tree
/// slot (the anchor exists; proceed with the normal P4 Seed-burst flow) or the deepest existing
/// prefix is an ancestor (descent registers; remaining components are tracked).
pub(crate) enum MaterializeResult {
    /// All segments existed; the leaf is `User`-rooted.
    Materialized(ResourceId),
    /// Descent is needed. The leaf `ResourceId` is the anchor's (currently `DescentScaffold`-roled)
    /// slot; the engine registers `DescentState` keyed by the Profile's id once it's been minted.
    /// `remaining` is non-empty by [`DescentRemaining`]'s type invariant —
    /// `materialize_path_or_pending` reaches this variant only when `prefix_idx + 1 <
    /// components.len()`, guaranteeing `from_vec` succeeds.
    Pending {
        anchor: ResourceId,
        prefix: ResourceId,
        remaining: DescentRemaining,
    },
}

impl crate::Engine {
    /// Walk a validated [`TreePath`] into the Tree. The leaf is created with `ResourceRole::User`;
    /// non-leaf components are `ResourceRole::DescentScaffold` if freshly created (the existing
    /// `ensure_root` / `ensure_child` preserve existing roles, so an already- User parent stays
    /// User).
    ///
    /// Returns `Materialized` iff every segment was already a live Tree slot AND the leaf's role is
    /// `User` after the walk (i.e., no scaffolding was created). Otherwise returns `Pending` with
    /// the deepest existing ancestor as `prefix` and the remaining components as the descent path.
    ///
    /// "Deepest existing ancestor" is determined by Tree-side pre-existence: each component is
    /// `lookup`'d before the walk; the deepest `i` for which `lookup(path.segments()[..=i])`
    /// succeeded before the materialising `ensure_path` call is the prefix index. The FS-root
    /// bootstrap guarantees `i >= 0` for every absolute attach. Role plays no part in this decision
    /// — a slot that existed before the walk may be a `User` peer anchor, a `WatchRootParent` of
    /// some other Profile, or a `DescentScaffold` retained from an earlier Pending Profile's
    /// descent chain; any of those count as "pre-existing".
    ///
    /// **Pre-conditions are now type-enforced.** [`TreePath`]'s type invariants (non-empty;
    /// `segments()[0] == FS_ROOT_SEGMENT`) make the prior `debug_assert!` and release-mode
    /// degradation branch structurally impossible.
    pub(crate) fn materialize_path_or_pending(&mut self, path: &TreePath) -> MaterializeResult {
        // Borrow segments as `&[&str]` once for the Tree-side helpers (`lookup`, `ensure_path`,
        // `resolve_components`) which all key on `&str`. One small allocation bounded by path depth.
        let components: Vec<&str> = path.segments().iter().map(CompactString::as_str).collect();

        // FS-root bootstrap. Unconditional: [`TreePath`]'s invariant guarantees `components[0] ==
        // FS_ROOT_SEGMENT`, and `ensure_root` is idempotent (returns the existing slot if a root at
        // `/` already exists). The role is `DescentScaffold` on first creation; if a prior `User`
        // attach at `/` already promoted the slot, the preserve-existing-role contract leaves it
        // alone. Bootstrapping unconditionally guarantees every Profile's rewind chain terminates
        // at this `/` slot — the kernel always `lstat`s `/` successfully on Unix, so a `Vanished`
        // response from a `/` probe is impossible, making cascading parent destruction (`rm -rf
        // /a/b/c/d`) recoverable: the descent stays Pending at `/` waiting for the cascade's bottom
        // segment to reappear.
        self.tree
            .ensure_root(FS_ROOT_SEGMENT, ResourceRole::DescentScaffold);

        // Snapshot which segments existed BEFORE the walk so we can tell freshly-scaffolded
        // segments from already-existing ones. The bootstrap above guarantees `components[0]`
        // (FS-root) always pre-exists.
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

        // Now do the walk. `ensure_path` creates non-leaf as `DescentScaffold`, leaf as `User`.
        let anchor = self
            .tree
            .ensure_path(&components, ResourceRole::User)
            .expect("TreePath::segments() is non-empty by type invariant");

        // Walk forward to find the deepest pre-existing prefix. The bootstrap guarantees
        // `pre_existed[0] == true`, so `prefix_idx` is always at least `0` — no `Option<usize>`
        // trichotomy is needed.
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
            // Segments [0..=prefix_idx] pre-existed; [prefix_idx+1..] are scaffolds. `ensure_path`
            // above created every segment, so `resolve_components` on any prefix is guaranteed to
            // succeed — convert from the prior `unwrap_or(anchor)` (which masked an invariant
            // violation) to `expect` with an explicit contract message.
            let prefix = self
                .resolve_components(&components[..=prefix_idx])
                .expect("ensure_path created every component; prefix slice must resolve");
            // Reuse the already-validated `CompactString` segments from [`TreePath`] rather than
            // re-allocating from `&str`. Bounded by path depth and lifts straight into
            // [`DescentRemaining`].
            let remaining_vec: Vec<CompactString> = path.segments()[prefix_idx + 1..].to_vec();
            // `prefix_idx + 1 < components.len()` is structurally guaranteed by the outer `if`, so
            // `from_vec` always succeeds here; `expect` documents the contract and gives a precise
            // panic message if a future refactor weakens the outer guard.
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

    /// Resolve a sequence of path components to its leaf `ResourceId` without mutating the Tree.
    /// Returns `None` if any segment doesn't resolve.
    pub(crate) fn resolve_components(&self, components: &[&str]) -> Option<ResourceId> {
        let mut cur: Option<ResourceId> = None;
        for comp in components {
            cur = Some(self.tree.lookup(cur, comp)?);
        }
        cur
    }

    /// Enter `ProfileState::Pending` against `prefix` with `remaining` path components
    /// (single-component segments, anchor last). Mints the probe correlation, flips the Profile to
    /// `Pending`, bumps the prefix's `STRUCTURE` `watch_demand` contribution, and emits the descent
    /// probe — the four-step Idle → Pending entry sequence as a single helper.
    ///
    /// **Ordering: mint → state-flip → add_watch → emit.** Symmetric with
    /// [`Self::materialize_profile_anchor`]'s state-before-refcount pattern. The mint runs *first*
    /// so the `Pending` state is constructed with its probe slot already armed — phase-without-
    /// correlation cannot exist. State-flip *then* refcount keeps the contribution attribution
    /// coherent with the Profile's claim shape at the moment of the refcount edge.
    ///
    /// **Pre-condition.** Profile must be `Idle`. The debug_assert below catches any caller passing
    /// a non-Idle Profile. ("No in-flight probe" is implied, not separately asserted: `Idle`
    /// carries no `DescentState`, so an idle Profile structurally has no probe slot.)
    ///
    /// **Recovery-overlap invariant.** When called from `start_pending_recovery`, the Profile already
    /// holds a `+1 STRUCTURE` contribution on the parent via `Profile.watch_root_parent` (set at the
    /// original anchor materialization, never cleared on `on_anchor_terminal_event`). This helper
    /// bumps `+1 STRUCTURE` again on the same resource, giving `+2`. At re-materialization the
    /// descent contribution is subbed and the `watch_root_parent` contribution persists —
    /// `set_watch_root_parent` is idempotent on the recovery path (`engine.rs::set_watch_root_parent`
    /// short-circuits when the cache already points at the same parent).
    pub(crate) fn enter_pending_descent(
        &mut self,
        profile_id: ProfileId,
        prefix: ResourceId,
        remaining: DescentRemaining,
        out: &mut StepOutput,
    ) {
        debug_assert!(
            self.profiles
                .get(profile_id)
                .is_some_and(|p| matches!(p.state(), ProfileState::Idle)),
            "enter_pending_descent: Profile must be Idle before re-entering \
             descent; caller must release prior state first. ('No in-flight \
             probe' is not a separate condition — `Idle` carries no \
             `DescentState`, so an idle Profile structurally has no probe \
             slot.) (profile = {profile_id:?})",
        );

        // Step 1: mint the correlation. Runs first so the Pending state below is constructed with
        // its slot already armed — no window where the phase exists without a correlation.
        let correlation = self.mint_probe_correlation();

        // Step 2: state-flip Idle → Pending, constructed armed. Done before the refcount edge so
        // any reader between this point and step 3 sees the Profile's claim shape that the
        // contribution will attribute to (matches `materialize_profile_anchor`'s sequencing). Loud
        // arm — the entry `debug_assert` proved the Profile live + Idle, so `get_mut` resolving
        // `None` is a state-machine breach, not a benign race; a silent skip would leave the slot
        // un-constructed while the emit below still fires (no probe, no diagnostic — a wedge).
        if self.profiles.get(profile_id).is_none() {
            unreachable!(
                "enter_pending_descent: Profile {profile_id:?} vanished \
                 between the Idle precondition and the construct-armed \
                 Pending transition"
            );
        }
        // Liveness is proven above, so the wrapper's internal `get_mut` resolves `Some`
        // (synchronous, no intervening mutation) — the construct-armed `ProbeSlot` is only built
        // for a live Profile, never stranded into a `None`-path drop.
        self.profiles.transition_state(
            profile_id,
            ProfileState::Pending(DescentState::new(
                prefix,
                remaining,
                ProbeSlot::armed(correlation, ()),
            )),
        );

        // Step 3: install the prefix's STRUCTURE contribution.
        add_watch(
            &mut self.tree,
            prefix,
            ContribKey::ProfileDescent(profile_id),
            ClassSet::STRUCTURE,
            out,
        );

        // Step 4: the choke reads the correlation back off the Pending descent slot and resolves
        // the prefix target off state.
        self.emit_owner_probe(profile_id, out);
    }

    /// Fan a typed descent response out to its terminal helper — symmetric with
    /// [`Self::dispatch_burst_outcome`], total over the three [`DescentOutcome`] variants. The
    /// illegal `AnchorOk` / `SubtreeProven` shapes were already rejected by the
    /// `DescentOutcome::try_from` parse at the demux seam, so they never reach here.
    pub(crate) fn dispatch_descent(
        &mut self,
        owner: ProfileId,
        outcome: DescentOutcome,
        now: Instant,
        out: &mut StepOutput,
    ) {
        match outcome {
            DescentOutcome::DirEnumerated(snapshot) => {
                self.dispatch_descent_ok(owner, &snapshot, now, out);
            }
            DescentOutcome::Vanished => self.dispatch_descent_vanished(owner, now, out),
            DescentOutcome::Failed(failure) => self.dispatch_descent_failed(owner, failure, out),
        }
    }

    /// Recover a descent from a walker-contract violation — a `Descent` probe whose payload
    /// resolved to an `AnchorOk` / `SubtreeProven` proof the route cannot accept (descent never
    /// queries an anchor's `lstat` shape or a subtree proof). The typed [`DescentOutcome`] parse
    /// rejected the payload at the demux seam; this **abandons** the descent prefix.
    ///
    /// `debug_assert!` in dev/CI (a production walker never emits this shape), then in release
    /// emits [`Diagnostic::WalkerContractViolated`] and routes through
    /// [`Self::release_descent_prefix_claim`] — the abandon terminal (the same release path the
    /// root-prefix `dispatch_descent_vanished` branch uses), **not** `dispatch_descent_vanished`
    /// itself: that rewinds to the parent and re-arms a fresh probe, which against a
    /// persistently-buggy walker is a tight re-probe loop. Abandoning leaves the Profile
    /// operator-recoverable (stuck Idle) and self-healing on a fresh descent. The probe slot was
    /// disarmed by `take_owner_probe` before dispatch and the descent state is unflipped at entry,
    /// so the release helper's preconditions hold.
    pub(crate) fn walker_contract_violated_descent(
        &mut self,
        owner: ProfileId,
        out: &mut StepOutput,
    ) {
        debug_assert!(
            false,
            "walker contract violated: a Descent probe received a non-enumeration \
             outcome (AnchorOk | SubtreeProven) — descent never queries an anchor \
             shape (owner = {owner:?})",
        );
        out.diagnostics
            .push(Diagnostic::WalkerContractViolated { owner });
        self.release_descent_prefix_claim(owner, out);
    }

    /// Dispatch a successful descent response. The walker honoured the `Descent` request shape and
    /// returned a single-level `Arc<DirSnapshot>` for the prefix; this routine looks up the next
    /// remaining segment by name and either advances descent one level, materializes the anchor, or
    /// awaits the next event.
    ///
    /// **Caller (`on_probe_response`).** The descent probe slot was disarmed (consume-once) before
    /// dispatch; the advance / rewind branches re-arm it with a freshly-minted correlation.
    pub(crate) fn dispatch_descent_ok(
        &mut self,
        owner: ProfileId,
        snapshot: &DirSnapshot,
        now: Instant,
        out: &mut StepOutput,
    ) {
        // Sample the head segment + arity from descent state, then drop the borrow. We clone only
        // the head (cheap when CompactString stays inline); the tail mutation runs in place via
        // `descent_state_mut` later, no whole-vec rebuild.
        let Some(descent) = self.descent_state(owner) else {
            return;
        };
        let prefix = descent.current_prefix();

        // The walker echoes `(owner, correlation)` verbatim — the gate match in `on_probe_response`
        // already enforces request/response pairing, so any divergence would surface as
        // `StaleProbeResponse`, not reach this point. The snapshot itself carries pure content;
        // engine identity stays engine-side (here, `descent.current_prefix()`). `DescentRemaining`
        // is non-empty by type invariant, so there is no defensive empty-arm recovery path and no
        // corresponding `Diagnostic` variants.
        let next_segment = descent.remaining_components().head().clone();
        let is_terminal = descent.remaining_components().is_terminal();

        // Descent probes walk the single-level `ScanConfig::Descent` shape, so the response is a
        // one-level Dir snapshot — look up the next segment by name in the BTreeMap directly.
        let entry_kind = match snapshot.entries().get(next_segment.as_str()) {
            Some(child) => child.kind(),
            None => {
                // Next segment not yet present; await next event. v1 descent doesn't mtime-skip, so
                // no need to retain the snapshot — the next probe will get a fresh `lstat`-walked
                // DirSnapshot anyway.
                return;
            }
        };

        // Materialize the next segment as a Tree slot. Look it up first; if absent, ensure as
        // DescentScaffold (the terminal arms may promote it to User via `promote_scaffold`).
        let new_resource = match self.tree.lookup(Some(prefix), &next_segment) {
            Some(r) => r,
            None => self
                .tree
                .ensure_child(prefix, &next_segment, ResourceRole::DescentScaffold)
                .expect("descent prefix held alive by the ProfileDescent contribution"),
        };
        self.tree.set_kind(new_resource, entry_kind.into());

        if is_terminal {
            // Terminal: materialise the anchor and start the Seed burst. The helper releases the
            // prefix's STRUCTURE contribution and installs the anchor's as part of its own
            // state-flip sequence.
            self.materialize_profile_anchor(owner, prefix, new_resource, entry_kind, now, out);
        } else {
            self.advance_descent(owner, prefix, new_resource, out);
        }
    }

    /// Advance descent one literal segment.
    ///
    /// Sequence:
    /// 1. Mint a fresh probe correlation for `owner`.
    /// 2. Mutate descent state in place: advance `current_prefix` to the new resource, drop the
    ///    consumed head segment from `remaining_components`, and re-arm the probe slot with the
    ///    fresh correlation (the response handler disarmed it before routing here, so `arm`'s
    ///    empty-slot precondition holds).
    /// 3. Release the old prefix's STRUCTURE contribution; install the new prefix's.
    /// 4. Emit the fresh descent probe at the new prefix.
    ///
    /// The old prefix stays alive because the freshly-advanced `new_prefix` is still its `children`
    /// entry — the structural anchor holds the slot across `sub_watch`. No `try_reap` here: the
    /// routine release helper would see a non-empty `children` map and short-circuit anyway, so we
    /// skip the call. (Role is metadata throughout — its tag stays `DescentScaffold` from the
    /// initial `ensure_child` but does not affect retention.)
    fn advance_descent(
        &mut self,
        owner: ProfileId,
        old_prefix: ResourceId,
        new_prefix: ResourceId,
        out: &mut StepOutput,
    ) {
        let correlation = self.mint_probe_correlation();
        // Loud arm — `dispatch_descent_ok` reaches here only for an owner the response gate proved
        // in descent (slot disarmed before routing), so `descent_state_mut` resolving `None` is a
        // state-machine breach. Silent skip ⇒ no re-arm, no probe, no diagnostic (a wedge); loud ⇒
        // crash.
        let Some(d) = self.descent_state_mut(owner) else {
            unreachable!(
                "advance_descent: owner {owner:?} not in descent after \
                 dispatch_descent_ok proved it"
            );
        };
        d.advance_to(new_prefix);
        // Non-terminal by caller contract — `dispatch_descent_ok` routes terminal descents through
        // anchor materialization before reaching `advance_descent`. The debug_assert inside
        // `DescentRemaining::advance` pins this for regression detection.
        d.remaining_components_mut().advance();
        d.arm_probe(correlation);

        let key = ContribKey::ProfileDescent(owner);
        sub_watch(&mut self.tree, old_prefix, key, out);
        add_watch(&mut self.tree, new_prefix, key, ClassSet::STRUCTURE, out);

        // The choke reads the correlation back off the descent slot and resolves the
        // (just-advanced) prefix target off state.
        self.emit_owner_probe(owner, out);
    }

    /// Promote `new_resource` to the Profile's anchor slot. Sole call site is
    /// [`Self::dispatch_descent_ok`]'s terminal arm — the descent has just resolved its last
    /// remaining segment and the Profile is about to leave `Pending` for `Idle → Active(Seed)`.
    ///
    /// Sequence (load-bearing):
    /// 1. Promote the slot's role to `User` via [`specter_core::Tree::promote_scaffold`] — a no-op
    ///    if a co-resident peer already gave the slot a real role (`WatchRootParent` / `User`), so
    ///    materialization never clobbers a peer's claim.
    /// 2. Capture `Profile.events` for the anchor's contribution.
    /// 3. Transition the Profile **before** any refcount op via
    ///    [`specter_core::Profile::materialize_anchor`] — atomic `Pending → Idle`, claim install,
    ///    kind pin. The recompute (multi-contributor case) reads `Profile.state` and
    ///    `Profile.anchor_claim` to attribute contributions; the post-flip world has the prefix's
    ///    STRUCTURE source gone (state no longer Pending) and the anchor's mask source owed.
    /// 4. Sub the prefix's STRUCTURE; add the anchor's mask.
    /// 5. Install the watch-root-parent contribution (deferred from `attach_sub_inner` because the
    ///    parent didn't exist on disk when the Profile attached).
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
        // `new_resource` is either a freshly-ensured DescentScaffold or a peer's pre-existing slot
        // (the caller's lookup hit). `promote_scaffold` flips only a still-scaffold slot and no-ops
        // on a real role, so materialization never clobbers a co-resident peer's WatchRootParent /
        // User.
        self.tree.promote_scaffold(new_resource, ResourceRole::User);

        let events_union = self
            .profiles
            .get(profile_id)
            .map_or(ClassSet::EMPTY, specter_core::Profile::events);

        let anchor_kind = ResourceKind::from(entry_kind);
        if let Some(p) = self.profiles.get_mut(profile_id) {
            p.materialize_anchor(anchor_kind);
        }

        // Profile.resource was assigned to the anchor's slot at attach time; the materialised
        // slot's id should match by construction.
        debug_assert!(
            self.profiles
                .get(profile_id)
                .is_some_and(|p| p.resource() == new_resource),
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

        self.set_watch_root_parent(profile_id, out);
        self.start_seed_burst(profile_id, None, now, out);
    }

    /// **Rewind chain depth.** A `Vanished` response on a rewound prefix triggers a further rewind
    /// via the same path. The chain depth is bounded by the tree-distance from the original prefix
    /// to its ultimate ancestor — at most one rewind cycle per ancestor level. Each rewind step
    /// **adds** a `+1 STRUCTURE` `watch_demand` on the new prefix; in production the chain
    /// auto-extends watches up the ancestor chain until it reaches a still-present ancestor, whose
    /// probe returns `Ok` and routes to `dispatch_descent_ok`'s "next segment not yet present;
    /// await next event" branch.
    ///
    /// **Branch reachability post-bootstrap.** With the unconditional FS-root bootstrap in
    /// [`Self::materialize_path_or_pending`], every descent's rewind chain terminates at the FS-root
    /// slot `/`. The kernel always `lstat`s `/` successfully on Unix, so a `Vanished` response from a
    /// `/` probe is impossible — meaning the `None` arm below is structurally unreachable in
    /// production. A cascade like `rm -rf /a/b/c/d` with anchor at `/d` rewinds through `/c`, `/b`,
    /// `/a`, `/` and terminates on `/`'s `Ok` rather than reaching the arm; the descent stays Pending
    /// at `/` waiting for the cascade's bottom segment to reappear, which makes cascading parent
    /// destruction auto-recoverable. The arm is retained as defense-in-depth against kernel anomalies
    /// (e.g., a chrooted environment where `/` is somehow inaccessible) and to keep the recursion
    /// well-typed; tests must construct the state directly to exercise it.
    ///
    /// For an `N`-level cascade with a Profile anchored at the leaf, the engine emits up to `N`
    /// rewind cycles per Pending Profile (one Watch + one descent probe per cycle). Acceptable in
    /// v1. Rewinds descent to the next-existing ancestor of `prefix`.
    pub(crate) fn dispatch_descent_vanished(
        &mut self,
        owner: ProfileId,
        _now: Instant,
        out: &mut StepOutput,
    ) {
        let Some(descent) = self.descent_state(owner) else {
            return;
        };
        let prefix = descent.current_prefix();

        out.diagnostics.push(Diagnostic::PendingPathProbeVanished {
            profile: owner,
            prefix,
        });

        // Capture the parent + the prefix's segment name BEFORE we mutate descent state. `parent`
        // selects the rewind branch; `prefix_name` becomes the new first remaining component.
        let parent = self.tree.parent(prefix);
        let prefix_name = self.tree.name(prefix).map(CompactString::from);

        match parent {
            Some(parent_id) => {
                // Rewind. The vanished prefix's segment becomes the *first* remaining component
                // (we're descending into it from the parent again).
                //
                // In-place mutation: prepend onto the existing `remaining_components` rather than
                // cloning + rebuilding a fresh DescentState — saves both the whole-vec clone and
                // the per-element CompactString clone.
                let correlation = self.mint_probe_correlation();
                // Loud arm — `dispatch_descent_vanished` already resolved `descent_state(owner)` at
                // entry, so this `_mut` re-projection is structurally `Some`; a `None` is a
                // state-machine breach, not a benign race. (The inner `prefix_name` `if let` is a
                // genuine `Option` — the root prefix has no segment name — and stays a conditional.)
                let Some(d) = self.descent_state_mut(owner) else {
                    unreachable!(
                        "dispatch_descent_vanished: owner {owner:?} not in \
                         descent after the entry resolution proved it"
                    );
                };
                d.advance_to(parent_id);
                if let Some(name) = prefix_name {
                    d.remaining_components_mut().prepend(name);
                }
                d.arm_probe(correlation);

                let key = ContribKey::ProfileDescent(owner);
                sub_watch_then_try_reap(&mut self.tree, prefix, key, out);

                add_watch(&mut self.tree, parent_id, key, ClassSet::STRUCTURE, out);

                // The choke reads the correlation back off the descent slot and resolves the
                // (rewound) parent target off state.
                self.emit_owner_probe(owner, out);
            }
            None => {
                // Root prefix vanished — no rewind target. Delegate to the release helper
                // (state-flip terminal + counter-aware sub + try_reap). Its preconditions hold
                // here: the descent probe slot was disarmed by `on_probe_response` before dispatch
                // (cancel-first contract) and descent state is unflipped at entry.
                //
                // The Profile is left stuck Idle without a usable descent path — operator recovery
                // is required.
                self.release_descent_prefix_claim(owner, out);
            }
        }
    }

    /// Failed response handler. The descent retains in-descent state and emits a diagnostic; the
    /// next event at the prefix re-triggers via [`Self::on_descent_event`].
    pub(crate) fn dispatch_descent_failed(
        &self,
        owner: ProfileId,
        failure: ProbeFailure,
        out: &mut StepOutput,
    ) {
        let prefix = match self.descent_state(owner) {
            Some(d) => d.current_prefix(),
            None => return,
        };
        out.diagnostics.push(Diagnostic::PendingPathProbeFailed {
            profile: owner,
            prefix,
            failure,
        });
        // Retain in-descent state; await next event at the prefix.
    }

    /// Handler for `FsEvent` arriving at a descent's `current_prefix`. Triggers a fresh probe (no
    /// settle wait — descent is event-driven). I5: drops the event if a probe is already in flight
    /// (the in-flight probe will pick up the change in its response). The "in flight" signal is an
    /// armed descent probe slot for this Profile.
    ///
    /// Returns `true` iff it re-armed the descent slot and emitted a fresh probe; `false` when a
    /// gate skipped (probe already in flight, or the Profile is no longer descending). The overflow
    /// reseed path keys its diagnostic on this — the gates here are the single source of "did a
    /// reseed happen", so an external re-check could never drift from them. The `FsEvent` dispatch
    /// loop discards the value (a skipped descent event needs no narration).
    pub(crate) fn on_descent_event(
        &mut self,
        owner: ProfileId,
        _now: Instant,
        out: &mut StepOutput,
    ) -> bool {
        if self.pending_probe_for(owner).is_some() {
            return false;
        }
        // Liveness gate: an `FsEvent` for an owner no longer descending is a benign post-transition
        // race — nothing to re-probe. The choke reads `current_prefix` back off the descent slot at
        // emit time.
        if self.descent_state(owner).is_none() {
            return false;
        }

        let correlation = self.mint_probe_correlation();
        // Loud arm — the `descent_state` gate just above proved the owner in descent and (no
        // in-flight probe ⇒) nothing mutated it, so this `_mut` re-projection is structurally `Some`.
        let Some(d) = self.descent_state_mut(owner) else {
            unreachable!(
                "on_descent_event: owner {owner:?} left descent between \
                 the liveness gate and the re-arm"
            );
        };
        d.arm_probe(correlation);
        // The choke reads the correlation back off the descent slot and resolves the prefix target
        // off state.
        self.emit_owner_probe(owner, out);
        true
    }
}

#[cfg(test)]
#[path = "descent_tests.rs"]
mod tests;
