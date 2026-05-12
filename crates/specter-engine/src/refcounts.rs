//! Refcount-edge helpers for the per-Resource contributions map and
//! `Resource.suppress_count`.
//!
//! Two refcounts, decoupled:
//! - **Contributions map** (`Resource.contributions`) gates FD lifetime:
//!   a Resource is Watched iff the map is non-empty. The map is a
//!   `BTreeMap<ContribKey, ClassSet>`: each key identifies a single
//!   contributor (Profile anchor / parent / descent / descendant, or
//!   Promoter prefix / proxy); the value is that contributor's
//!   `ClassSet` mask. The per-Resource events union is the OR fold over
//!   the map's values.
//! - **`suppress_count`** gates event delivery — silenced iff `> 0`.
//!
//! Each primitive emits `WatchOp` ops as follows:
//! - [`add_watch`]: `Watch` on the empty → non-empty edge OR on any
//!   union change at non-empty.
//! - [`sub_watch`]: `Unwatch` on the non-empty → empty edge; `Watch`
//!   on any union change at non-empty.
//! - [`add_suppress`] / [`sub_suppress`]: `Suppress` / `Unsuppress` on
//!   the 0↔1 edge only — suppression is binary and orthogonal to the
//!   events mask.
//!
//! **Two release paths.**
//! - [`sub_watch_then_try_reap`] is the **routine per-key release** —
//!   drop one contributor at `(r, key)` and free the slot iff this
//!   release zeroed `Resource::has_anchors()`. Co-resident
//!   contributions at other keys keep the slot watched.
//! - [`crate::Tree::vacate`] is the **protocol terminus** — clear the
//!   whole contributions map atomically and zero `suppress_count`,
//!   emitting both closing ops in one step. Used when every
//!   contribution at the slot is being abandoned at once (kernel-watch
//!   rejection, or `reconcile::delete_child` on a fully-drained slot).
//!
//! **Idempotent absent-key sub.** Calling [`sub_watch`] for a key that
//! is not in the map is a silent no-op. This makes the helper safe to
//! invoke against post-vacate slots ([`crate::Tree::vacate`] cleared
//! the map) and slots drained by a prior sub-walk in the same step
//! (e.g., [`Engine::release_descendant_claim`]'s take-and-walk pass).
//!
//! **Source of truth.** Contribution attribution is **data**: each
//! caller passes the explicit [`ContribKey`] for the role it owns.
//! There is no walk-the-registry recompute; the union is the OR fold
//! over the map's current values, computed lazily by
//! [`specter_core::Resource::events_union`]. Adding a new contributor
//! kind is a [`ContribKey`] variant + its sole call site, with no
//! engine-wide propagation.
//!
//! Stale `ResourceId`: the lookup short-circuits with no mutation and
//! no op emission. The Engine maintains "non-empty contributions ⇒
//! live slot" by attaching contributions only at live Resources, so a
//! stale id here means a logic bug elsewhere; the silent return is
//! defence-in-depth.

use specter_core::{ClassSet, ContribKey, ResourceId, StepOutput, Tree, WatchOp};

/// Install or update the contribution at `(r, key)` with `mask`,
/// emitting `WatchOp::Watch` on the existence edge or when the
/// per-Resource union widens (or otherwise changes).
///
/// **No registry walk.** Signature is purely Resource-local —
/// `(&mut Tree, ResourceId, ContribKey, ClassSet, &mut StepOutput)`.
///
/// **Idempotent.** Re-inserting the same `(key, mask)` is a no-op
/// (the map already contains it; no union change; no emission).
/// Re-inserting `key` with a *different* `mask` overwrites and emits
/// `Watch` iff the union changes.
///
/// `mask == EMPTY` is legitimate (e.g., a defensive call from a
/// fixture that hasn't wired its mask yet); the Sensor degrades to
/// identity-floor-only registration on the resulting `WatchOp::Watch`.
///
/// The `WatchOp`'s `path` is resolved at emission via [`Tree::path_of`];
/// if path resolution fails (the slot exists but a segment doesn't
/// resolve — unreachable for live slots), the op carries
/// `PathBuf::new()` and the Sensor reports `WatchOpRejected` on
/// attempt.
pub fn add_watch(
    tree: &mut Tree,
    r: ResourceId,
    key: ContribKey,
    mask: ClassSet,
    out: &mut StepOutput,
) {
    let Some(res) = tree.get_mut(r) else {
        return;
    };
    let was_empty = res.contributions.is_empty();
    let prev_union = res.events_union();
    res.contributions.insert(key, mask);
    let new_union = res.events_union();
    let kind = res.kind_raw();

    let emit = was_empty || new_union != prev_union;
    if emit {
        // Reborrow `tree` for `path_of` once the `res` borrow ends
        // (the line above is the last use).
        let path = tree.path_of(r).unwrap_or_default();
        out.watch_ops.push(WatchOp::Watch {
            resource: r,
            path,
            kind,
            events: new_union,
        });
    }
}

/// Remove the contribution at `(r, key)`. Emits `WatchOp::Unwatch` on
/// the non-empty → empty edge; emits a fresh `WatchOp::Watch` when
/// the per-Resource union changes but contributions remain.
///
/// **No registry walk.** Removal is by key; no Profile / Promoter
/// state is read.
///
/// **No release-of-state contract.** The caller's bookkeeping
/// (`Profile.anchor_claim`, `Profile.watch_root_parent`,
/// `Profile.state`, `Promoter.state`, etc.) can be cleared in either
/// order relative to this call — the contribution map is the source
/// of truth for refcounting, independent of owner state.
///
/// **Idempotent.** Absent key ⇒ silent no-op. Reachable post-vacate
/// ([`crate::Tree::vacate`] cleared the map) or post-prior-sub-walk
/// (a sister helper drained this slot earlier in the same step).
pub fn sub_watch(tree: &mut Tree, r: ResourceId, key: ContribKey, out: &mut StepOutput) {
    let Some(res) = tree.get_mut(r) else {
        return;
    };
    let prev_union = res.events_union();
    if res.contributions.remove(&key).is_none() {
        return;
    }

    if res.contributions.is_empty() {
        out.watch_ops.push(WatchOp::Unwatch { resource: r });
        return;
    }

    let new_union = res.events_union();
    if new_union != prev_union {
        let kind = res.kind_raw();
        let path = tree.path_of(r).unwrap_or_default();
        out.watch_ops.push(WatchOp::Watch {
            resource: r,
            path,
            kind,
            events: new_union,
        });
    }
}

/// `+suppress_count` on `r`. Emits `WatchOp::Suppress` on the 0→1 edge.
///
/// Suppression is orthogonal to the events mask — it gates kernel event
/// *delivery* on an existing FD, not registration. The mask is unaffected.
pub fn add_suppress(tree: &mut Tree, r: ResourceId, out: &mut StepOutput) {
    let Some(res) = tree.get_mut(r) else {
        return;
    };
    let prev = res.suppress_count;
    res.suppress_count = prev.saturating_add(1);
    if prev == 0 {
        out.watch_ops.push(WatchOp::Suppress { resource: r });
    }
}

/// `-suppress_count` on `r`. Emits `WatchOp::Unsuppress` on the 1→0
/// edge. Safe in any counter state, including `prev == 0` —
/// [`crate::Tree::vacate`] (the protocol terminus) can legitimately
/// zero `suppress_count` mid-burst (via either of its two production
/// callers: `reconcile::delete_child` on a drained slot, or
/// `on_watch_op_rejected` on kernel-watch failure), so the eventual
/// symmetric `sub_suppress` from `finish_burst_to_idle`'s drain enters
/// here on a zero counter and short-circuits without emission.
pub fn sub_suppress(tree: &mut Tree, r: ResourceId, out: &mut StepOutput) {
    let Some(res) = tree.get_mut(r) else {
        return;
    };
    let prev = res.suppress_count;
    if prev == 0 {
        return;
    }
    res.suppress_count = prev - 1;
    if prev == 1 {
        out.watch_ops.push(WatchOp::Unsuppress { resource: r });
    }
}

/// Remove the contribution at `(r, key)` and try-reap the slot.
/// Returns `true` iff the slot was reaped.
///
/// Composes the two halves of the **routine per-key release**:
/// [`sub_watch`] drains the contribution from
/// `Resource.contributions` (idempotent on absent key — safe against
/// post-vacate slots and slots a prior sub-walk in this step already
/// drained); [`Tree::try_reap`] then removes the slot iff
/// `Resource::has_anchors() == false`.
///
/// **Anchors and contributions.** [`specter_core::Resource::has_anchors`]
/// reads four retention sources: `children`, `profiles` back-ref,
/// `proxy_promoters`, and the contributions map itself. The map is
/// canonical for "this slot holds a live kernel-watch claim";
/// removing the last contribution at a slot zeroes that source, and
/// the slot reaps iff none of the three back-ref vectors still
/// reaches into it. The role tag (`User` / `WatchRootParent` /
/// `DescentScaffold`) is metadata and never gates retention.
///
/// In every production call site, at least one *other* claim is
/// structurally certain to keep the slot alive across this release —
/// for the descent-prefix release, the prefix's child chain toward the
/// anchor; for the watch-root parent release, the anchor child slot;
/// for the proxy release, the contribution's own removal is gated by
/// `proxy_promoters` having been cleared by the caller first.
/// [`specter_core::Tree::try_reap`] cascades upward when its own
/// reap orphans a parent, so a single release helper at a leaf-most
/// slot transparently frees the whole prefix chain it owned.
///
/// **Distinct from [`Tree::vacate`].** Vacate is the protocol
/// terminus: it clears the entire contribution map AND zeros
/// `suppress_count` in one atomic step, emitting both closing ops
/// together. Use vacate when every contribution at the slot is being
/// abandoned at once (kernel-watch rejection routed through
/// `on_watch_op_rejected`, or `reconcile::delete_child` on a
/// fully-drained slot). Use this helper for the **routine per-key
/// release path** where one contributor releases its single key.
///
/// Caller discipline:
/// - Owner-side bookkeeping (state flag, snapshot field, etc.) is
///   the caller's responsibility — this helper only mutates the
///   contributions map and the slot lifecycle.
/// - Cancel-first preconditions (probe-channel closed) are the
///   caller's responsibility — see the `debug_assert!`s on the
///   higher-level `release_*_claim` helpers in [`crate::claims`] and
///   [`crate::promoter_claims`].
pub fn sub_watch_then_try_reap(
    tree: &mut Tree,
    r: ResourceId,
    key: ContribKey,
    out: &mut StepOutput,
) -> bool {
    sub_watch(tree, r, key, out);
    tree.try_reap(r)
}

#[cfg(test)]
mod tests {
    use super::{add_suppress, add_watch, sub_suppress, sub_watch, sub_watch_then_try_reap};
    use specter_core::{ClassSet, ContribKey, ProfileId, ResourceRole, StepOutput, Tree, WatchOp};

    fn fresh() -> (Tree, specter_core::ResourceId) {
        let mut tree = Tree::new();
        let r = tree.ensure(None, "anchor", ResourceRole::User);
        (tree, r)
    }

    /// Last `WatchOp::Watch` emitted, for asserting on its `events`.
    fn last_watch_events(out: &StepOutput) -> Option<ClassSet> {
        out.watch_ops.iter().rev().find_map(|op| match op {
            WatchOp::Watch { events, .. } => Some(*events),
            _ => None,
        })
    }

    #[test]
    fn add_watch_zero_to_one_emits_watch_with_contribution() {
        let (mut tree, r) = fresh();
        let mut out = StepOutput::default();
        let key = ContribKey::ProfileAnchor(ProfileId::default());
        add_watch(&mut tree, r, key, ClassSet::CONTENT, &mut out);
        assert_eq!(tree.get(r).unwrap().watch_demand(), 1);
        assert_eq!(tree.get(r).unwrap().events_union(), ClassSet::CONTENT);
        assert_eq!(out.watch_ops.len(), 1);
        match &out.watch_ops[0] {
            WatchOp::Watch {
                resource, events, ..
            } => {
                assert_eq!(*resource, r);
                assert_eq!(*events, ClassSet::CONTENT);
            }
            op => panic!("expected Watch, got {op:?}"),
        }
    }

    #[test]
    fn add_watch_distinct_keys_widen_union_and_emit_watch() {
        // Two distinct keys at the same resource ⇒ refcount 2, union
        // is the OR of the two masks. Each `add_watch` past the
        // empty edge emits a `Watch` iff the union widens.
        let (mut tree, r) = fresh();
        let mut out = StepOutput::default();
        let key_a = ContribKey::ProfileAnchor(ProfileId::default());
        let key_b = ContribKey::ProfileParent(ProfileId::default());

        add_watch(&mut tree, r, key_a, ClassSet::CONTENT, &mut out);
        out.watch_ops.clear();

        // Same key, same mask ⇒ no-op (map already has it; no union change).
        add_watch(&mut tree, r, key_a, ClassSet::CONTENT, &mut out);
        assert_eq!(tree.get(r).unwrap().watch_demand(), 1);
        assert!(
            out.watch_ops.is_empty(),
            "no Watch emitted when (key, mask) idempotent",
        );

        // Distinct key, distinct mask ⇒ widens union ⇒ emit.
        add_watch(&mut tree, r, key_b, ClassSet::METADATA, &mut out);
        assert_eq!(tree.get(r).unwrap().watch_demand(), 2);
        assert_eq!(
            tree.get(r).unwrap().events_union(),
            ClassSet::CONTENT | ClassSet::METADATA,
        );
        assert_eq!(out.watch_ops.len(), 1);
        assert_eq!(
            last_watch_events(&out),
            Some(ClassSet::CONTENT | ClassSet::METADATA),
        );
    }

    #[test]
    fn add_watch_with_empty_mask_at_zero_emits_identity_floor_watch() {
        // 0→1 with EMPTY mask: still emits Watch (existence edge),
        // but `opts.events == EMPTY` ⇒ Sensor degrades to identity
        // floor.
        let (mut tree, r) = fresh();
        let mut out = StepOutput::default();
        let key = ContribKey::ProfileAnchor(ProfileId::default());
        add_watch(&mut tree, r, key, ClassSet::EMPTY, &mut out);
        assert_eq!(tree.get(r).unwrap().watch_demand(), 1);
        assert_eq!(tree.get(r).unwrap().events_union(), ClassSet::EMPTY);
        assert_eq!(out.watch_ops.len(), 1);
        assert_eq!(last_watch_events(&out), Some(ClassSet::EMPTY));
    }

    #[test]
    fn sub_watch_last_contributor_emits_unwatch() {
        let (mut tree, r) = fresh();
        let mut out = StepOutput::default();
        let key = ContribKey::ProfileAnchor(ProfileId::default());
        add_watch(&mut tree, r, key, ClassSet::CONTENT, &mut out);
        out.watch_ops.clear();

        sub_watch(&mut tree, r, key, &mut out);
        assert_eq!(tree.get(r).unwrap().watch_demand(), 0);
        assert_eq!(tree.get(r).unwrap().events_union(), ClassSet::EMPTY);
        assert_eq!(out.watch_ops.len(), 1);
        assert!(matches!(
            out.watch_ops[0],
            WatchOp::Unwatch { resource } if resource == r,
        ));
    }

    #[test]
    fn sub_watch_with_remaining_contributors_emits_narrowing_watch() {
        // Two distinct contributors with different masks; removing
        // one narrows the union and emits a Watch with the narrower
        // mask.
        let (mut tree, r) = fresh();
        let mut out = StepOutput::default();
        let key_a = ContribKey::ProfileAnchor(ProfileId::default());
        let key_b = ContribKey::ProfileParent(ProfileId::default());

        add_watch(&mut tree, r, key_a, ClassSet::CONTENT, &mut out);
        add_watch(&mut tree, r, key_b, ClassSet::METADATA, &mut out);
        assert_eq!(tree.get(r).unwrap().watch_demand(), 2);
        out.watch_ops.clear();

        sub_watch(&mut tree, r, key_a, &mut out);
        assert_eq!(tree.get(r).unwrap().watch_demand(), 1);
        assert_eq!(tree.get(r).unwrap().events_union(), ClassSet::METADATA);
        assert_eq!(
            last_watch_events(&out),
            Some(ClassSet::METADATA),
            "Watch emitted with the narrowed mask",
        );
        assert!(
            !out.watch_ops
                .iter()
                .any(|op| matches!(op, WatchOp::Unwatch { .. })),
            "no Unwatch — contributions still non-empty",
        );
    }

    #[test]
    fn sub_watch_no_emit_when_union_unchanged() {
        // Two contributors with overlapping masks: removing one
        // leaves the union unchanged, so no Watch op is emitted.
        let (mut tree, r) = fresh();
        let mut out = StepOutput::default();
        let key_a = ContribKey::ProfileAnchor(ProfileId::default());
        let key_b = ContribKey::ProfileParent(ProfileId::default());

        add_watch(&mut tree, r, key_a, ClassSet::CONTENT, &mut out);
        add_watch(&mut tree, r, key_b, ClassSet::CONTENT, &mut out);
        out.watch_ops.clear();

        sub_watch(&mut tree, r, key_a, &mut out);
        assert_eq!(tree.get(r).unwrap().watch_demand(), 1);
        assert_eq!(tree.get(r).unwrap().events_union(), ClassSet::CONTENT);
        assert!(
            out.watch_ops.is_empty(),
            "no Watch emitted when remaining union equals prior",
        );
    }

    #[test]
    fn sub_watch_absent_key_is_silent_noop() {
        // Map missing the key: no underflow, no emission. Reachable
        // post-vacate or after a prior sub-walk drained this slot in
        // the same step.
        let (mut tree, r) = fresh();
        let mut out = StepOutput::default();
        sub_watch(
            &mut tree,
            r,
            ContribKey::ProfileAnchor(ProfileId::default()),
            &mut out,
        );
        assert!(out.watch_ops.is_empty());
        assert_eq!(tree.get(r).unwrap().watch_demand(), 0);
    }

    #[test]
    fn add_watch_stale_resource_is_noop() {
        let mut tree = Tree::new();
        let r = tree.ensure(None, "ghost", ResourceRole::User);
        assert!(tree.try_reap(r));
        let mut out = StepOutput::default();
        add_watch(
            &mut tree,
            r,
            ContribKey::ProfileAnchor(ProfileId::default()),
            ClassSet::STRUCTURE,
            &mut out,
        );
        assert!(out.watch_ops.is_empty());
    }

    #[test]
    fn sub_watch_stale_resource_is_noop() {
        let mut tree = Tree::new();
        let r = tree.ensure(None, "ghost", ResourceRole::User);
        assert!(tree.try_reap(r));
        let mut out = StepOutput::default();
        sub_watch(
            &mut tree,
            r,
            ContribKey::ProfileAnchor(ProfileId::default()),
            &mut out,
        );
        assert!(out.watch_ops.is_empty());
    }

    #[test]
    fn add_suppress_zero_to_one_emits_suppress() {
        let (mut tree, r) = fresh();
        let mut out = StepOutput::default();
        add_suppress(&mut tree, r, &mut out);
        assert_eq!(tree.get(r).unwrap().suppress_count, 1);
        assert_eq!(out.watch_ops.len(), 1);
        assert!(matches!(
            out.watch_ops[0],
            WatchOp::Suppress { resource } if resource == r,
        ));
    }

    #[test]
    fn add_suppress_two_no_extra_emit() {
        let (mut tree, r) = fresh();
        let mut out = StepOutput::default();
        add_suppress(&mut tree, r, &mut out);
        out.watch_ops.clear();
        add_suppress(&mut tree, r, &mut out);
        assert_eq!(tree.get(r).unwrap().suppress_count, 2);
        assert!(out.watch_ops.is_empty());
    }

    #[test]
    fn sub_suppress_one_to_zero_emits_unsuppress() {
        let (mut tree, r) = fresh();
        let mut out = StepOutput::default();
        add_suppress(&mut tree, r, &mut out);
        out.watch_ops.clear();
        sub_suppress(&mut tree, r, &mut out);
        assert_eq!(tree.get(r).unwrap().suppress_count, 0);
        assert_eq!(out.watch_ops.len(), 1);
        assert!(matches!(
            out.watch_ops[0],
            WatchOp::Unsuppress { resource } if resource == r,
        ));
    }

    #[test]
    fn sub_suppress_at_zero_counter_is_silent_noop() {
        // Symmetric to the watch_demand case — `Tree::vacate` can
        // legitimately zero `suppress_count` while emitting the
        // closing `Unsuppress`, and the eventual symmetric drain from
        // `finish_burst_to_idle` then enters here on `prev == 0`.
        let (mut tree, r) = fresh();
        let mut out = StepOutput::default();
        sub_suppress(&mut tree, r, &mut out);
        assert!(out.watch_ops.is_empty());
        assert_eq!(tree.get(r).unwrap().suppress_count, 0);
    }

    #[test]
    fn watch_and_suppress_are_independent() {
        let (mut tree, r) = fresh();
        let mut out = StepOutput::default();
        let key = ContribKey::ProfileAnchor(ProfileId::default());
        add_watch(&mut tree, r, key, ClassSet::CONTENT, &mut out);
        add_suppress(&mut tree, r, &mut out);
        {
            let res = tree.get(r).unwrap();
            assert_eq!(res.watch_demand(), 1);
            assert_eq!(res.suppress_count, 1);
        }
        sub_watch(&mut tree, r, key, &mut out);
        let res = tree.get(r).unwrap();
        assert_eq!(res.watch_demand(), 0);
        // suppress unchanged by the watch decrement.
        assert_eq!(res.suppress_count, 1);
    }

    #[test]
    fn sub_watch_then_try_reap_last_contributor_reaps_slot() {
        // Sole contributor drops: the sub_watch step empties the
        // contributions map, the try_reap step frees the slot.
        // Returns true.
        let (mut tree, r) = fresh();
        let mut out = StepOutput::default();
        let key = ContribKey::ProfileAnchor(ProfileId::default());
        add_watch(&mut tree, r, key, ClassSet::CONTENT, &mut out);
        out.watch_ops.clear();

        let reaped = sub_watch_then_try_reap(&mut tree, r, key, &mut out);

        assert!(reaped, "slot reaped on the empty → reapable edge");
        assert!(tree.get(r).is_none(), "slot is gone");
        assert_eq!(out.watch_ops.len(), 1);
        assert!(matches!(
            out.watch_ops[0],
            WatchOp::Unwatch { resource } if resource == r,
        ));
    }

    #[test]
    fn sub_watch_then_try_reap_role_only_slot_emits_unwatch_and_reaps() {
        // Production parallel to `release_watch_root_parent_claim` and
        // the `release_*_descent_prefix_claim` family: the role tag is
        // metadata, so a slot whose only retention claim was the just-
        // dropped contribution reaps in the same step. `sub_watch`
        // emits `Unwatch` on the empty edge; `try_reap` returns `true`
        // and the slot is gone.
        let mut tree = Tree::new();
        let r = tree.ensure(None, "parent", ResourceRole::WatchRootParent);
        let mut out = StepOutput::default();
        let key = ContribKey::ProfileParent(ProfileId::default());
        add_watch(&mut tree, r, key, ClassSet::STRUCTURE, &mut out);
        out.watch_ops.clear();

        let reaped = sub_watch_then_try_reap(&mut tree, r, key, &mut out);

        assert!(
            reaped,
            "role is metadata; slot reaps once contributions drain"
        );
        assert!(tree.get(r).is_none(), "slot is gone");
        assert_eq!(out.watch_ops.len(), 1);
        assert!(matches!(
            out.watch_ops[0],
            WatchOp::Unwatch { resource } if resource == r,
        ));
    }

    #[test]
    fn sub_watch_then_try_reap_narrows_union_with_coresident_key() {
        // Two distinct contributors at the same slot: dropping one
        // narrows the events union but the surviving co-resident
        // contribution itself anchors the slot. `sub_watch` emits the
        // narrowing `Watch` (not `Unwatch`) since the map stays
        // non-empty; `try_reap` is a no-op because the slot still has
        // a live contribution (one of `has_anchors`'s four retention
        // sources).
        let mut tree = Tree::new();
        let r = tree.ensure(None, "prefix", ResourceRole::DescentScaffold);
        let mut out = StepOutput::default();
        let key_a = ContribKey::ProfileDescent(ProfileId::default());
        let key_b = ContribKey::ProfileParent(ProfileId::default());
        add_watch(&mut tree, r, key_a, ClassSet::CONTENT, &mut out);
        add_watch(&mut tree, r, key_b, ClassSet::METADATA, &mut out);
        out.watch_ops.clear();

        let reaped = sub_watch_then_try_reap(&mut tree, r, key_a, &mut out);

        assert!(!reaped, "co-resident contribution anchors the slot");
        assert_eq!(tree.get(r).unwrap().watch_demand(), 1);
        assert_eq!(
            last_watch_events(&out),
            Some(ClassSet::METADATA),
            "narrowed Watch emitted; no Unwatch",
        );
        assert!(
            !out.watch_ops
                .iter()
                .any(|op| matches!(op, WatchOp::Unwatch { .. })),
        );
    }

    #[test]
    fn sub_watch_then_try_reap_role_only_slot_with_child_survives() {
        // A `WatchRootParent`-roled slot whose retention also includes
        // a sibling child stays alive: dropping the contribution
        // empties the contributions map but `children` still anchors.
        // Mirrors the live `release_watch_root_parent_claim` path,
        // where the anchor child has not yet been reaped at the call
        // moment; the subsequent anchor reap then cascades through and
        // frees the parent in the same step.
        let mut tree = Tree::new();
        let parent = tree.ensure(None, "parent", ResourceRole::WatchRootParent);
        let _anchor = tree.ensure(Some(parent), "anchor", ResourceRole::User);
        let mut out = StepOutput::default();
        let key = ContribKey::ProfileParent(ProfileId::default());
        add_watch(&mut tree, parent, key, ClassSet::STRUCTURE, &mut out);
        out.watch_ops.clear();

        let reaped = sub_watch_then_try_reap(&mut tree, parent, key, &mut out);

        assert!(!reaped, "child anchors the parent slot");
        assert!(tree.get(parent).is_some());
        assert_eq!(out.watch_ops.len(), 1);
        assert!(matches!(
            out.watch_ops[0],
            WatchOp::Unwatch { resource } if resource == parent,
        ));
    }

    #[test]
    fn sub_watch_then_try_reap_absent_key_is_silent_release() {
        // Absent key: sub_watch is a silent no-op (no emission).
        // try_reap still runs and frees the slot iff has_anchors() is
        // false. Reachable post-vacate or after a prior sub-walk in
        // the same step.
        let (mut tree, r) = fresh();
        let mut out = StepOutput::default();
        let reaped = sub_watch_then_try_reap(
            &mut tree,
            r,
            ContribKey::ProfileAnchor(ProfileId::default()),
            &mut out,
        );
        assert!(reaped, "no anchors at all → slot reaps");
        assert!(out.watch_ops.is_empty());
    }
}
