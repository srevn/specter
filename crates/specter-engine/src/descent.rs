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
use specter_core::{
    ClassSet, DescentState, Diagnostic, EntryKind, ProfileId, ProfileState, ResourceId,
    ResourceKind, ResourceRole, StepOutput, TreeSnapshot,
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
        remaining: Vec<String>,
    },
}

/// Variant-free echo of `ProbeResult`. Used internally by
/// `dispatch_descent_probe` to route on the variant without re-cloning
/// at the dispatch boundary.
pub(crate) enum ProbeResultArm {
    Ok(TreeSnapshot),
    Vanished,
    Failed { errno: i32 },
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
    pub(crate) fn materialize_path_or_pending(&mut self, components: &[&str]) -> MaterializeResult {
        if components.is_empty() {
            return MaterializeResult::Materialized(ResourceId::default());
        }

        // FS-root bootstrap. Absolute attaches start with the synthetic
        // [`crate::engine::FS_ROOT_SEG`] (`"/"`) — the filesystem root
        // always exists on Unix, so we lazy-ensure a Tree slot for it
        // before pre-existence sampling. The pre-existence walk's first
        // step then sees the root as live, anchoring `prefix_idx` at
        // `Some(0)` for any absolute path. Non-absolute (test-only)
        // attaches skip this and may legitimately produce `prefix_idx =
        // None` if the Tree is empty (handled below).
        if components[0] == crate::engine::FS_ROOT_SEG {
            self.tree.ensure(
                None,
                crate::engine::FS_ROOT_SEG,
                ResourceRole::DescentScaffold,
            );
        }

        // Snapshot which segments existed BEFORE the walk so we can tell
        // freshly-scaffolded segments from already-existing ones. After
        // the bootstrap above, an absolute attach's first segment always
        // pre-exists.
        let mut pre_existed: Vec<bool> = Vec::with_capacity(components.len());
        let mut cur_lookup: Option<ResourceId> = None;
        for comp in components {
            let id = self.tree.lookup(cur_lookup, comp);
            pre_existed.push(id.is_some());
            cur_lookup = id;
        }

        // Now do the walk. `ensure_path` creates non-leaf as
        // `DescentScaffold`, leaf as `User`.
        let anchor = self.tree.ensure_path(components, ResourceRole::User);

        // Walk forward to find the deepest pre-existing prefix. If every
        // segment pre-existed, descent isn't needed.
        let mut prefix_idx: Option<usize> = None;
        for (i, &existed) in pre_existed.iter().enumerate() {
            if existed {
                prefix_idx = Some(i);
            } else {
                break;
            }
        }

        match prefix_idx {
            Some(i) if i + 1 == components.len() => {
                // Whole path pre-existed. P4 path.
                MaterializeResult::Materialized(anchor)
            }
            Some(i) => {
                // Segments [0..=i] pre-existed; [i+1..] are scaffolds.
                let prefix = self.resolve_components(&components[..=i]);
                let remaining: Vec<String> = components[i + 1..]
                    .iter()
                    .map(|s| (*s).to_string())
                    .collect();
                MaterializeResult::Pending {
                    anchor,
                    prefix: prefix.unwrap_or(anchor),
                    remaining,
                }
            }
            None => {
                // Reachable only for **relative**-path attaches against
                // an empty Tree (test fixtures use this path). Absolute
                // attaches always have at least the bootstrapped FS-root
                // pre-existing, so `prefix_idx >= Some(0)`.
                //
                // Degenerate semantics: the root segment is treated as
                // both the "anchor scaffold" and the descent prefix. The
                // first probe at the root will return `Vanished` and the
                // rewind is a no-op (root has no parent, see
                // `dispatch_descent_vanished`'s `None` branch).
                let remaining: Vec<String> = components.iter().map(|s| (*s).to_string()).collect();
                let root = self.resolve_components(&components[..1]).unwrap_or(anchor);
                MaterializeResult::Pending {
                    anchor,
                    prefix: root,
                    remaining,
                }
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
        remaining: Vec<String>,
        out: &mut StepOutput,
    ) {
        debug_assert!(
            self.profiles.get(profile_id).is_some_and(|p| {
                matches!(p.state, ProfileState::Idle) && p.pending_probe.is_none()
            }),
            "enter_pending_descent: Profile must be Idle with closed probe channel; \
             caller must invoke cancel_pending_probe (or take the response-dispatch path) \
             and release prior state before re-entering descent (profile = {profile_id:?})",
        );

        add_watch_demand(&mut self.tree, prefix, ClassSet::STRUCTURE, out);

        let Some(correlation) = self.mint_probe_correlation(profile_id) else {
            return;
        };

        if let Some(p) = self.profiles.get_mut(profile_id) {
            p.state = ProfileState::Pending(DescentState {
                current_prefix: prefix,
                remaining_components: remaining,
            });
        }

        self.emit_probe_op(
            profile_id,
            prefix,
            correlation,
            crate::probe_channel::ProbeEmissionParams::Descent,
            out,
        );
    }

    /// Dispatch a `ProbeResponse` for a Profile whose state is
    /// `ProfileState::Pending`. `on_probe_response` matches on `&p.state`
    /// and routes here if the Pending arm wins. The probe channel
    /// (`Profile.pending_probe`) was closed at the top of
    /// `on_probe_response`; this function may re-open it via
    /// `mint_probe_correlation` in the advance / rewind paths.
    pub(crate) fn dispatch_descent_probe(
        &mut self,
        profile_id: ProfileId,
        result: ProbeResultArm,
        now: Instant,
        out: &mut StepOutput,
    ) {
        match result {
            ProbeResultArm::Ok(snapshot) => {
                self.dispatch_descent_ok(profile_id, snapshot, now, out);
            }
            ProbeResultArm::Vanished => {
                self.dispatch_descent_vanished(profile_id, now, out);
            }
            ProbeResultArm::Failed { errno } => {
                self.dispatch_descent_failed(profile_id, errno, out);
            }
        }
    }

    // `snapshot: TreeSnapshot` is taken by-value to mirror the
    // `dispatch_*_ok` family in `transitions.rs`; clippy notes we only
    // borrow it. Allow it — keeping the signature uniform across descent
    // and standard dispatch is the higher-order concern.
    #[allow(clippy::needless_pass_by_value)]
    fn dispatch_descent_ok(
        &mut self,
        profile_id: ProfileId,
        snapshot: TreeSnapshot,
        now: Instant,
        out: &mut StepOutput,
    ) {
        // Read what we need from descent state, then drop the borrow.
        let Some(descent) = self.descent_state(profile_id) else {
            return;
        };
        let prefix = descent.current_prefix;
        let remaining = descent.remaining_components.clone();

        if remaining.is_empty() {
            // The DescentState invariant (core/profile.rs) says
            // `remaining_components` is non-empty: the anchor itself is
            // the last component, and descent transitions Pending → Idle
            // on materialization rather than emptying the vec. If we
            // ever reach this arm, it's a state-machine bug. Take the
            // conservative recovery path: surface the breach via a
            // Diagnostic and release the prefix claim symmetrically
            // (clears state to Idle AND releases the +1 watch_demand
            // contribution, matching `dispatch_descent_vanished`'s
            // root branch). Without the release, the prefix's counter
            // would leak.
            out.diagnostics.push(Diagnostic::DescentInvariantViolation {
                profile: profile_id,
                prefix,
            });
            self.release_descent_prefix_claim(profile_id, out);
            return;
        }

        let next_segment = remaining[0].clone();
        let is_anchor = remaining.len() == 1;

        // Descent probes ship `recursive=false`, so the response is a
        // single-level Dir snapshot — look up the next segment by name in
        // the BTreeMap directly.
        let entry_kind = match &snapshot {
            TreeSnapshot::Dir(arc) => match arc.entries.get(next_segment.as_str()) {
                Some(child) => child.kind(),
                None => {
                    // Next segment not yet present; await next event. v1
                    // descent doesn't mtime-skip, so no need to retain the
                    // snapshot — the next probe will get a fresh
                    // `lstat`-walked DirSnapshot anyway.
                    return;
                }
            },
            TreeSnapshot::File(_) => {
                // Descent probed a Dir; got a File. Treat as Vanished —
                // kind mismatch on the prefix.
                self.dispatch_descent_vanished(profile_id, now, out);
                return;
            }
        };

        // Materialize the next segment as a Tree slot. Look it up first;
        // if absent, ensure as DescentScaffold (the role flips below).
        let new_resource = match self.tree.lookup(Some(prefix), &next_segment) {
            Some(r) => r,
            None => self
                .tree
                .ensure(Some(prefix), &next_segment, ResourceRole::DescentScaffold),
        };
        self.tree.set_kind(new_resource, kind_from_entry(entry_kind));

        if is_anchor {
            // Materialize: flip role to User; swap watch_demand from
            // prefix to anchor; set up the watch-root-parent
            // contribution (deferred from `attach_sub_inner` because the
            // anchor's parent didn't exist on disk during pending);
            // transition Pending → Idle; start Seed burst.
            //
            // Sub the prefix's STRUCTURE contribution BEFORE clearing
            // the descent state's Pending status: the recompute (multi-
            // contributor case) reads `Profile.state == Pending(d) &&
            // d.current_prefix == prefix` to attribute this Profile's
            // STRUCTURE contribution. Cleanest sequencing: transition
            // state to Idle FIRST, then sub_watch_demand sees a clean
            // post-release world. The anchor materialization writes are
            // also moved up so the Profile is in a consistent state for
            // the upcoming refcount ops on the anchor.
            self.tree.set_role(new_resource, ResourceRole::User);

            // Capture the Profile's user mask now; used as the anchor's
            // contribution. The Profile's events_union is invariant, so
            // this is a one-time read.
            let events_union = self
                .profiles
                .get(profile_id)
                .map_or(ClassSet::EMPTY, |p| p.events_union);

            // Transition Pending → Idle and set anchor_contribution = true
            // BEFORE the refcount ops so the recompute sees the post-
            // transition contribution attribution: the prefix's STRUCTURE
            // contribution is gone (state no longer Pending), the anchor's
            // mask contribution is owed (anchor_contribution = true).
            if let Some(p) = self.profiles.get_mut(profile_id) {
                p.anchor_contribution = true;
                p.state = ProfileState::Idle;
            }

            // Profile.resource was assigned to the anchor's slot at
            // attach_sub time; the anchor's id should match.
            debug_assert!(
                self.profiles
                    .get(profile_id)
                    .is_some_and(|p| p.resource == new_resource),
                "descent anchor materialization: Profile.resource diverges from descent anchor",
            );

            sub_watch_demand(
                &mut self.tree,
                &self.profiles,
                prefix,
                ClassSet::STRUCTURE,
                None,
                out,
            );
            add_watch_demand(&mut self.tree, new_resource, events_union, out);

            // Watch-root-parent contribution. The anchor's parent now exists
            // on disk; install the contribution and cache the parent id.
            self.set_watch_root_parent(profile_id, new_resource, out);

            // Start the Seed burst against the anchor.
            self.start_seed_burst(profile_id, now, out);
        } else {
            // Advance one level. Sub the old prefix's contribution; add
            // the new prefix's. Refresh the descent state and emit a
            // fresh probe.
            //
            // Update descent state BEFORE sub_watch_demand so the recompute
            // (multi-contributor case) attributes the prefix's STRUCTURE
            // contribution to the new prefix, not the old one.
            let Some(correlation) = self.mint_probe_correlation(profile_id) else {
                return;
            };
            let new_remaining: Vec<String> = remaining[1..].to_vec();
            if let Some(d) = self.descent_state_mut(profile_id) {
                d.current_prefix = new_resource;
                d.remaining_components = new_remaining;
            }

            sub_watch_demand(
                &mut self.tree,
                &self.profiles,
                prefix,
                ClassSet::STRUCTURE,
                None,
                out,
            );
            add_watch_demand(&mut self.tree, new_resource, ClassSet::STRUCTURE, out);

            self.emit_probe_op(
                profile_id,
                new_resource,
                correlation,
                crate::probe_channel::ProbeEmissionParams::Descent,
                out,
            );
        }
    }

    fn dispatch_descent_vanished(
        &mut self,
        profile_id: ProfileId,
        _now: Instant,
        out: &mut StepOutput,
    ) {
        let Some(descent) = self.descent_state(profile_id) else {
            return;
        };
        let prefix = descent.current_prefix;

        out.diagnostics.push(Diagnostic::PendingPathProbeVanished {
            profile: profile_id,
            prefix,
        });

        // Capture the prior remaining-components and the prefix's
        // segment name BEFORE we mutate descent state. The vanished
        // prefix's segment becomes the first remaining component on
        // rewind.
        let parent = self.tree.parent(prefix);
        let prior_remaining: Vec<String> = self
            .descent_state(profile_id)
            .map(|s| s.remaining_components.clone())
            .unwrap_or_default();
        let prefix_name = self.tree.name(prefix).map(str::to_string);

        match parent {
            Some(parent_id) => {
                // Rewind. The vanished prefix's segment becomes the
                // *first* remaining component (we're descending into it
                // from the parent again).
                //
                // Update descent state BEFORE sub_watch_demand so the
                // recompute attributes this Profile's STRUCTURE
                // contribution to the new prefix (parent_id), not the
                // vanished one.
                let mut new_remaining = prior_remaining;
                if let Some(name) = prefix_name {
                    new_remaining.insert(0, name);
                }

                let Some(correlation) = self.mint_probe_correlation(profile_id) else {
                    return;
                };
                if let Some(p) = self.profiles.get_mut(profile_id) {
                    p.state = ProfileState::Pending(DescentState {
                        current_prefix: parent_id,
                        remaining_components: new_remaining,
                    });
                }

                sub_watch_demand(
                    &mut self.tree,
                    &self.profiles,
                    prefix,
                    ClassSet::STRUCTURE,
                    None,
                    out,
                );
                self.tree.vacate(prefix);
                self.tree.try_reap(prefix);

                add_watch_demand(&mut self.tree, parent_id, ClassSet::STRUCTURE, out);

                self.emit_probe_op(
                    profile_id,
                    parent_id,
                    correlation,
                    crate::probe_channel::ProbeEmissionParams::Descent,
                    out,
                );
            }
            None => {
                // Root prefix vanished — no rewind target. Clear the
                // descent state to Idle BEFORE sub_watch_demand so the
                // recompute correctly excludes this Profile's
                // contribution. The Profile is now stuck Idle without
                // an anchor — operator recovery is required.
                if let Some(p) = self.profiles.get_mut(profile_id) {
                    p.state = ProfileState::Idle;
                }
                sub_watch_demand(
                    &mut self.tree,
                    &self.profiles,
                    prefix,
                    ClassSet::STRUCTURE,
                    None,
                    out,
                );
                self.tree.vacate(prefix);
                self.tree.try_reap(prefix);
            }
        }
    }

    fn dispatch_descent_failed(&self, profile_id: ProfileId, errno: i32, out: &mut StepOutput) {
        let prefix = match self.descent_state(profile_id) {
            Some(d) => d.current_prefix,
            None => return,
        };
        out.diagnostics.push(Diagnostic::PendingPathProbeFailed {
            profile: profile_id,
            prefix,
            errno,
        });
        // Retain pending state; await next event at the prefix.
    }

    /// Handle an `FsEvent` arriving at a descent's `current_prefix`.
    /// Triggers a fresh probe (no settle wait — descent is event-driven).
    /// I5: drops the event if a probe is already in flight (the in-flight
    /// probe will pick up the change in its response). The "in flight"
    /// signal is the per-Profile probe-channel slot
    /// ([`crate::Engine::pending_probe`] reads `Profile.pending_probe`).
    pub(crate) fn on_descent_event(
        &mut self,
        profile_id: ProfileId,
        _now: Instant,
        out: &mut StepOutput,
    ) {
        if self.pending_probe(profile_id).is_some() {
            return;
        }
        let prefix = match self.descent_state(profile_id) {
            Some(d) => d.current_prefix,
            None => return,
        };

        let Some(correlation) = self.mint_probe_correlation(profile_id) else {
            return;
        };
        self.emit_probe_op(
            profile_id,
            prefix,
            correlation,
            crate::probe_channel::ProbeEmissionParams::Descent,
            out,
        );
    }
}

const fn kind_from_entry(k: EntryKind) -> ResourceKind {
    match k {
        EntryKind::File | EntryKind::Symlink | EntryKind::Other => ResourceKind::File,
        EntryKind::Dir => ResourceKind::Dir,
    }
}

#[cfg(test)]
#[path = "descent_tests.rs"]
mod tests;
