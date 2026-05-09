//! Pending-path descent.
//!
//! Pending descent runs **outside** the Burst lifecycle. A Profile whose
//! anchor doesn't yet exist on the filesystem lives in
//! `ProfileState::Pending(DescentState)`. The descent emits its own probes
//! (the correlation lives on `Profile.pending_probe`, the per-Profile
//! probe-channel slot), advances one path component per probe response,
//! and ends by materializing the anchor — at which point the Profile
//! transitions Pending → Idle and immediately Idle → Active(Seed) to
//! establish its baseline.
//!
//! **Why a parallel state machine?** Burst semantics don't fit:
//! - Probe target ≠ `Profile.resource` during descent (probes go to the
//!   deepest existing prefix, not the anchor).
//! - There's no Effect to fire — the Profile has no baseline yet.
//! - The settle timer (carried inside `BurstPhase::Batching`) and
//!   `burst_deadline` are stability concerns; descent
//!   is event-driven (a `StructureChanged` at the prefix triggers a
//!   fresh probe with no settle wait).
//! - I5 stays intact: at most one outstanding probe per Profile.
//!   `Pending` and `Active` are mutually exclusive `ProfileState`
//!   variants (the compiler proves it); within `Pending`, an in-flight
//!   probe is signalled by `Profile.pending_probe = Some(_)` — the same
//!   discipline as `Active(Burst { phase: Verifying })`.
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
//!    probe is already in flight (`Profile.pending_probe.is_some()`).

use crate::refcounts::{add_watch_demand, sub_watch_demand};
use compact_str::CompactString;
use specter_core::{
    AnchorClaim, ClassSet, DescentState, Diagnostic, DirSnapshot, EntryKind, ProbeOwner, ProfileId,
    ProfileState, ResourceId, ResourceKind, ResourceRole, StepOutput,
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
    Pending {
        anchor: ResourceId,
        prefix: ResourceId,
        remaining: Vec<CompactString>,
    },
}

impl crate::Engine {
    /// Walk `components` into the Tree. The leaf is created with
    /// `ResourceRole::User`; non-leaf components are
    /// `ResourceRole::DescentScaffold` if freshly created (the existing
    /// `Tree::ensure` contract preserves existing roles, so an already-
    /// User parent stays User).
    ///
    /// Returns `Materialized` iff every segment was already a live Tree
    /// slot AND the leaf's role is `User` after the walk (i.e., no
    /// scaffolding was created). Otherwise returns `Pending` with the
    /// deepest existing ancestor as `prefix` and the remaining components
    /// as the descent path.
    ///
    /// "Deepest existing ancestor" is determined by walking up from the
    /// leaf and asking the `ResourceRole`: the first ancestor whose role
    /// is **not** `DescentScaffold` is the prefix. (A `User` or
    /// `WatchRootParent` ancestor is fine — the watch state is owned by
    /// some other Profile in the case of `User`, or by the engine's
    /// infrastructure in the case of `WatchRootParent`.)
    ///
    /// **Pre-condition.** `components` is non-empty and `components[0] ==
    /// FS_ROOT_SEG`. [`crate::engine::decompose_attach_path`] is the
    /// canonical producer and enforces both invariants by gate; the
    /// `debug_assert!` below pins the contract for any future
    /// hand-constructed caller (test fixtures, fuzzers).
    pub(crate) fn materialize_path_or_pending(&mut self, components: &[&str]) -> MaterializeResult {
        debug_assert!(
            !components.is_empty() && components[0] == crate::engine::FS_ROOT_SEG,
            "materialize_path_or_pending pre-condition: components must be non-empty and \
             components[0] == FS_ROOT_SEG (decompose_attach_path is the canonical producer)",
        );
        // Release-mode degradation for the empty case: the Engine's
        // caller (`attach_sub_inner`) maps `Materialized(default)` to a
        // no-op return path. Keep this short-circuit so a future caller
        // misuse can't index out-of-bounds on `components[0]` below.
        if components.is_empty() {
            return MaterializeResult::Materialized(ResourceId::default());
        }

        // FS-root bootstrap. Unconditional: the pre-condition guarantees
        // `components[0] == FS_ROOT_SEG`, and `Tree::ensure` is idempotent
        // (returns the existing slot if `(parent=None, segment="/")`
        // already maps to one). The role is `DescentScaffold` on first
        // creation; if a prior `User` attach at `/` already promoted the
        // slot, `ensure`'s preserve-existing-role contract leaves it
        // alone. Bootstrapping unconditionally guarantees every Profile's
        // rewind chain terminates at this `/` slot — the kernel always
        // `lstat`s `/` successfully on Unix, so a `Vanished` response
        // from a `/` probe is impossible, making cascading parent
        // destruction (`rm -rf /a/b/c/d`) recoverable: the descent stays
        // Pending at `/` waiting for the cascade's bottom segment to
        // reappear.
        self.tree.ensure(
            None,
            crate::engine::FS_ROOT_SEG,
            ResourceRole::DescentScaffold,
        );

        // Snapshot which segments existed BEFORE the walk so we can tell
        // freshly-scaffolded segments from already-existing ones. The
        // bootstrap above guarantees `components[0]` (FS-root) always
        // pre-exists.
        let mut pre_existed: Vec<bool> = Vec::with_capacity(components.len());
        let mut cur_lookup: Option<ResourceId> = None;
        for comp in components {
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
        let anchor = self.tree.ensure_path(components, ResourceRole::User);

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
            let remaining: Vec<CompactString> = components[prefix_idx + 1..]
                .iter()
                .map(|&s| CompactString::from(s))
                .collect();
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
    /// path components (single-component segments, anchor last). Bumps the
    /// prefix's `STRUCTURE` `watch_demand` contribution, opens the
    /// probe channel, writes the descent state, and emits the descent
    /// probe — the four-step Idle → Pending entry sequence as a single
    /// helper.
    ///
    /// **Pre-condition.** Profile must be `Idle` with a closed probe
    /// channel. The debug_assert below catches any caller passing a
    /// non-Idle Profile or one with `pending_probe.is_some()`.
    ///
    /// **Caller responsibility.** Parent-edge work (`compute_and_set_parent_edge`,
    /// `recompute_dependent_parent_edges`) is NOT done here — the fresh-attach
    /// path needs it on first entry; the recovery path doesn't (the parent
    /// edges already exist on the recovering Profile). Keeping the helper
    /// minimal preserves that contract.
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
        remaining: Vec<CompactString>,
        out: &mut StepOutput,
    ) {
        debug_assert!(
            self.profiles.get(profile_id).is_some_and(|p| {
                matches!(p.state, ProfileState::Idle) && p.pending_probe.is_none()
            }),
            "enter_pending_descent: Profile must be Idle with closed probe channel; \
             caller must invoke cancel_owner_probe (or take the response-dispatch path) \
             and release prior state before re-entering descent (profile = {profile_id:?})",
        );

        add_watch_demand(&mut self.tree, prefix, ClassSet::STRUCTURE, out);

        let owner = ProbeOwner::Profile(profile_id);
        let Some(correlation) = self.mint_owner_correlation(owner) else {
            return;
        };

        if let Some(p) = self.profiles.get_mut(profile_id) {
            p.state = ProfileState::Pending(DescentState {
                current_prefix: prefix,
                remaining_components: remaining,
            });
        }

        let target_path = self.tree.path_of(prefix).unwrap_or_default();
        Self::emit_descent_probe(owner, correlation, prefix, target_path, out);
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
    /// **Caller (`on_*_probe_response`).** The probe channel was closed
    /// before dispatch; this function may re-open it via
    /// `mint_owner_correlation` in the advance branch.
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
        let prefix = descent.current_prefix;

        // Defense-in-depth: the walker stamps `DirSnapshot.root_resource`
        // with the `target_resource` we placed on the `Descent` request,
        // which the engine sets to `descent.current_prefix` at every
        // `emit_descent_probe` site. Divergence signals a walker bug or
        // a wire-side regression.
        debug_assert_eq!(
            snapshot.root_resource, prefix,
            "walker stamp diverges from emitted target_resource (owner = {owner:?})",
        );
        let Some(next_segment) = descent.remaining_components.first().cloned() else {
            // The DescentState invariant (core/profile.rs) says
            // `remaining_components` is non-empty: the descent target is
            // the last component, and descent leaves the in-descent state
            // on materialization rather than emptying the vec. If we
            // ever reach this arm, it's a state-machine bug. Take the
            // conservative recovery path: surface the breach via a
            // per-owner Diagnostic and release the prefix claim
            // symmetrically (clears descent state AND releases the +1
            // watch_demand contribution, matching
            // `dispatch_descent_vanished`'s root branch). Without the
            // release, the prefix's counter would leak.
            out.diagnostics
                .push(descent_invariant_diagnostic(owner, prefix));
            self.release_owner_descent_prefix(owner, out);
            return;
        };
        let is_terminal = descent.remaining_components.len() == 1;

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
                .ensure(Some(prefix), &next_segment, ResourceRole::DescentScaffold),
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
    ///    `remaining_components`. State-flip BEFORE `sub_watch_demand`
    ///    so the recompute attributes the contribution to the new
    ///    prefix, not the old one.
    /// 3. Release the old prefix's STRUCTURE contribution; install the
    ///    new prefix's.
    /// 4. Emit the fresh descent probe at the new prefix.
    ///
    /// The old prefix retains its `DescentScaffold` role (set on its
    /// own `Tree::ensure` at descent's start) — the role survives
    /// `sub_watch_demand`; reaping is deferred to a future state
    /// transition. No `try_reap` here.
    fn advance_descent(
        &mut self,
        owner: ProbeOwner,
        old_prefix: ResourceId,
        new_prefix: ResourceId,
        out: &mut StepOutput,
    ) {
        let Some(correlation) = self.mint_owner_correlation(owner) else {
            return;
        };
        if let Some(d) = self.descent_state_mut(owner) {
            d.current_prefix = new_prefix;
            d.remaining_components.remove(0);
        }

        sub_watch_demand(
            &mut self.tree,
            &self.profiles,
            &self.promoters,
            old_prefix,
            ClassSet::STRUCTURE,
            None,
            out,
        );
        add_watch_demand(&mut self.tree, new_prefix, ClassSet::STRUCTURE, out);

        let target_path = self.tree.path_of(new_prefix).unwrap_or_default();
        Self::emit_descent_probe(owner, correlation, new_prefix, target_path, out);
    }

    /// Owner-polymorphic descent-prefix release. Routes to the per-owner
    /// claim helper, both of which read the prefix from descent state
    /// (Profile: `Pending(d).current_prefix`; Promoter:
    /// `PrefixPending(d).current_prefix`).
    ///
    /// Per-owner cleanup (parallel shape):
    ///
    /// - **Profile.** Delegates to [`Self::release_descent_prefix_claim`]:
    ///   transitions `Pending → Idle`, releases the prefix's STRUCTURE
    ///   contribution (counter-aware), and `try_reap`s the prefix slot.
    /// - **Promoter.** Delegates to
    ///   [`Self::release_promoter_descent_prefix_claim`]: transitions
    ///   `PrefixPending → Active{empty}` BEFORE `sub_watch_demand` so the
    ///   recompute drops the 5a attribution; counter-aware sub on the
    ///   prefix's STRUCTURE; `try_reap`.
    ///
    /// Three call sites:
    /// - [`Self::dispatch_descent_ok`]'s structurally-unreachable
    ///   empty-remaining arm.
    /// - [`Self::dispatch_descent_vanished`]'s no-rewind-target arm
    ///   (FS-root vanish, structurally unreachable on Unix).
    /// - [`Self::on_watch_op_rejected`]'s descent-prefix purge loops
    ///   (Profile and Promoter sides).
    ///
    /// Counter-awareness mirrors the per-owner helper discipline: a
    /// prior `clamp_watch_demand_to_zero` may have left the slot at 0,
    /// in which case the state-flip alone is the observable cleanup.
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
    /// 2. Capture `Profile.events_union` for the anchor's contribution.
    /// 3. Transition the Profile **before** any refcount op:
    ///    `anchor_claim = Held`, `state = Idle`, `kind = Some(anchor_kind)`.
    ///    The recompute (multi-contributor case) reads `Profile.state` and
    ///    `Profile.anchor_claim` to attribute contributions; the post-flip
    ///    world has the prefix's STRUCTURE source gone (state no longer
    ///    Pending) and the anchor's mask source owed.
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
            .map_or(ClassSet::EMPTY, |p| p.events_union);

        let anchor_kind = kind_from_entry(entry_kind);
        if let Some(p) = self.profiles.get_mut(profile_id) {
            p.anchor_claim = AnchorClaim::Held;
            p.state = ProfileState::Idle;
            p.kind = Some(anchor_kind);
        }

        // Profile.resource was assigned to the anchor's slot at attach
        // time; the materialised slot's id should match by construction.
        debug_assert!(
            self.profiles
                .get(profile_id)
                .is_some_and(|p| p.resource == new_resource),
            "descent anchor materialization: Profile.resource diverges from descent anchor",
        );

        sub_watch_demand(
            &mut self.tree,
            &self.profiles,
            &self.promoters,
            prefix,
            ClassSet::STRUCTURE,
            None,
            out,
        );
        add_watch_demand(&mut self.tree, new_resource, events_union, out);

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
        let prefix = descent.current_prefix;

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
                // Update descent state BEFORE sub_watch_demand so the
                // recompute attributes this owner's STRUCTURE
                // contribution to the new prefix (parent_id), not the
                // vanished one.
                //
                // In-place mutation: prepend onto the existing
                // `remaining_components` rather than cloning + rebuilding
                // a fresh DescentState — saves both the whole-vec clone
                // and the per-element CompactString clone.
                let Some(correlation) = self.mint_owner_correlation(owner) else {
                    return;
                };
                if let Some(d) = self.descent_state_mut(owner) {
                    d.current_prefix = parent_id;
                    if let Some(name) = prefix_name {
                        d.remaining_components.insert(0, name);
                    }
                }

                sub_watch_demand(
                    &mut self.tree,
                    &self.profiles,
                    &self.promoters,
                    prefix,
                    ClassSet::STRUCTURE,
                    None,
                    out,
                );
                // No `vacate` — `sub_watch_demand` cleared the union iff
                // this owner was the last contributor; remaining
                // contributors (co-resident Profile / Promoter claims)
                // keep theirs. `try_reap` removes the slot iff
                // `has_anchors()` returns false.
                self.tree.try_reap(prefix);

                add_watch_demand(&mut self.tree, parent_id, ClassSet::STRUCTURE, out);

                let target_path = self.tree.path_of(parent_id).unwrap_or_default();
                Self::emit_descent_probe(owner, correlation, parent_id, target_path, out);
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
            Some(d) => d.current_prefix,
            None => return,
        };
        out.diagnostics
            .push(descent_failed_diagnostic(owner, prefix, errno));
        // Retain in-descent state; await next event at the prefix.
    }

    /// Owner-polymorphic Handle an `FsEvent` arriving at a descent's
    /// `current_prefix`. Triggers a fresh probe (no settle wait —
    /// descent is event-driven). I5: drops the event if a probe is
    /// already in flight (the in-flight probe will pick up the change
    /// in its response). The "in flight" signal is the per-owner
    /// probe-channel slot, read via [`Engine::pending_probe_for`].
    pub(crate) fn on_descent_event(
        &mut self,
        owner: ProbeOwner,
        _now: Instant,
        out: &mut StepOutput,
    ) {
        if self.pending_probe_for(owner).is_some() {
            return;
        }
        let prefix = match self.descent_state(owner) {
            Some(d) => d.current_prefix,
            None => return,
        };

        let Some(correlation) = self.mint_owner_correlation(owner) else {
            return;
        };
        let target_path = self.tree.path_of(prefix).unwrap_or_default();
        Self::emit_descent_probe(owner, correlation, prefix, target_path, out);
    }
}

pub(crate) const fn kind_from_entry(k: EntryKind) -> ResourceKind {
    match k {
        EntryKind::File | EntryKind::Symlink | EntryKind::Other => ResourceKind::File,
        EntryKind::Dir => ResourceKind::Dir,
    }
}

/// Per-owner diagnostic emitted when the descent dispatcher observes
/// `DescentState.remaining_components.is_empty()` — a state-machine
/// invariant breach. Profile and Promoter ship distinct variants
/// (`DescentInvariantViolation` vs `PromoterDescentInvariantViolation`)
/// so operator logs disambiguate the source without parsing the carried
/// id type.
const fn descent_invariant_diagnostic(owner: ProbeOwner, prefix: ResourceId) -> Diagnostic {
    match owner {
        ProbeOwner::Profile(profile) => Diagnostic::DescentInvariantViolation { profile, prefix },
        ProbeOwner::Promoter(promoter) => {
            Diagnostic::PromoterDescentInvariantViolation { promoter, prefix }
        }
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
