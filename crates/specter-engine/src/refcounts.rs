//! Refcount-edge helpers for `Resource.watch_demand`,
//! `Resource.suppress_count`, and the per-Resource `events_union` mask.
//!
//! Two refcounts, decoupled:
//! - `watch_demand` gates FD lifetime — a Resource is Watched iff `> 0`.
//! - `suppress_count` gates event delivery — silenced iff `> 0`.
//!
//! In addition to the binary refcount, every Resource carries a
//! `events_union: ClassSet` — the OR of every covering Profile's
//! contribution. The union is the carrier for the per-FD
//! kqueue mask: `add_watch_demand` ORs the incoming contribution onto the
//! cached union; `sub_watch_demand` recomputes from scratch by walking
//! covering Profiles in the registry. `WatchOp::Watch` is emitted whenever
//! the union or the existence flips — not just on the 0↔1 edge — so the
//! Sensor's mask cache stays in sync with the engine's view.
//!
//! Each helper emits `WatchOp` ops as follows:
//! - `add_watch_demand`: `Watch` on the 0→1 edge OR on any union change at
//!   non-zero refcount.
//! - `sub_watch_demand`: `Unwatch` on the 1→0 edge; `Watch` on any union
//!   change at refcount > 0.
//! - `add_suppress` / `sub_suppress`: `Suppress` / `Unsuppress` on the 0↔1
//!   edge only — suppression is binary and orthogonal to the mask.
//!
//! Underflows on the watch / suppress counters are debug-asserted; in
//! release the counter clamps at 0 and the edge op is suppressed (the
//! Sensor is already Unwatched / Unsuppressed in that state).
//!
//! Stale `ResourceId`: the lookup short-circuits with no mutation and no op
//! emission. The Engine maintains `watch_demand > 0 ⇒ live slot` (I6) by
//! attaching contributions only at live Resources, so a stale id here means
//! a logic bug elsewhere; the silent return is defense-in-depth.

use crate::coverage::covers;
use specter_core::{
    AnchorClaim, ClassSet, Profile, ProfileId, ProfileMap, ProfileState, Promoter,
    PromoterRegistry, PromoterState, Resource, ResourceId, ResourceKind, StepOutput, Tree, WatchOp,
};

/// `+watch_demand` on `r`, contributing `contribution` to `r.events_union`.
///
/// `add_watch_demand` is **unchanged** by the Promoter generalisation: it
/// only ORs the incoming contribution onto the cached union; no
/// recompute walk is involved. Promoter-side contributions surface
/// through `sub_watch_demand`'s recompute path (the on-decrement
/// recompute walks both registries) and by direct `add_watch_demand`
/// emissions from the Promoter helpers that pass
/// `ClassSet::STRUCTURE` for the proxy / prefix slot they own.
///
/// Emits `WatchOp::Watch` when:
/// - The refcount transitions 0→1 (existence edge), OR
/// - The cached `events_union` widens to include any new bit from
///   `contribution` (mask edge).
///
/// `contribution == EMPTY` is legitimate (e.g., a defensive call from a
/// fixture that hasn't wired its mask yet); the Sensor degrades to
/// identity-floor-only registration on the resulting `WatchOp::Watch`.
///
/// The `WatchOp`'s `path` is resolved at emission via `Tree::path_of`; if
/// path resolution fails (the slot exists but a segment doesn't resolve
/// through the interner — unreachable for live slots), the op carries
/// `PathBuf::new()` and the Sensor reports `WatchOpRejected` on attempt.
pub fn add_watch_demand(
    tree: &mut Tree,
    r: ResourceId,
    contribution: ClassSet,
    out: &mut StepOutput,
) {
    let (prev_refcount, prev_union, new_union) = {
        let Some(res) = tree.get_mut(r) else {
            return;
        };
        let prev_refcount = res.watch_demand;
        let prev_union = res.events_union;
        let new_union = prev_union | contribution;
        res.watch_demand = prev_refcount.saturating_add(1);
        res.events_union = new_union;
        (prev_refcount, prev_union, new_union)
    };

    if prev_refcount == 0 || new_union != prev_union {
        let path = tree.path_of(r).unwrap_or_default();
        // Preserve raw `Unknown` here — the sensor's
        // `matches_or_unknown` verification treats it as the engine's
        // intentional wildcard at fresh-watch time.
        let kind = tree
            .get(r)
            .map_or(ResourceKind::Unknown, Resource::kind_raw);
        out.watch_ops.push(WatchOp::Watch {
            resource: r,
            path,
            kind,
            events: new_union,
        });
    }
}

/// `-watch_demand` on `r`. Emits `WatchOp::Unwatch` on the 1→0 edge; emits a
/// fresh `WatchOp::Watch` when the per-Resource `events_union` shrinks at
/// refcount > 0.
///
/// `contribution` is documentation-only at v1: the recompute walks every
/// covering Profile's contribution from scratch (a value-subtract on the
/// cached union would be unsound — bits owed to the releasing Profile may
/// be owned by another Profile too). The parameter survives in the
/// signature for caller-readability symmetry with `add_watch_demand` and
/// to give v2 per-(Profile, Resource) tracking a natural source-of-truth
/// for the removal.
///
/// `profiles` is the registry the recompute walks. Callers pass
/// `&self.profiles` after the releasing Profile's state-tracking field
/// (`anchor_claim`, `state == Pending(d)`, or `watch_root_parent`) has
/// been cleared, so the recompute models the post-release state.
///
/// `promoters` extends the recompute to the Promoter-side contributions
/// (sources 5a / 5b in [`recompute_resource_events`]). Callers pass
/// `&self.promoters` after the releasing Promoter's state field has
/// been flipped (the analogue of the Profile-side flag-clear); the
/// recompute walks both registries and ORs the union.
///
/// `releasing_descendant` — when `Some(pid)`, the recompute (multi-
/// contributor case) skips `pid`'s **descendant** contribution to `r`.
/// This is the explicit signal for the descendant case where no
/// per-Profile flag exists: `Profile.current.is_some()` is the proxy
/// state, but during `reconcile::delete_child` it's still set while the
/// contribution is being released — the parameter resolves the ambiguity
/// precisely. Anchor / watch-root-parent / descent-prefix releases pass
/// `None`; the corresponding flag-clear by the caller already excludes
/// the releasing Profile's contribution of that kind.
///
/// Underflow → `debug_assert!` panic in dev; in release the counter clamps
/// at 0 and no op is emitted (the Sensor is already in the Unwatched
/// state).
pub fn sub_watch_demand(
    tree: &mut Tree,
    profiles: &ProfileMap,
    promoters: &PromoterRegistry,
    r: ResourceId,
    contribution: ClassSet,
    releasing_descendant: Option<ProfileId>,
    out: &mut StepOutput,
) {
    // Documentation-only at v1; the recompute walks all covering Profiles
    // rather than subtracting bits. Future v2 predicate-based filtering
    // may use this as the per-(Profile, Resource) removal key.
    let _ = contribution;

    let (prev_refcount, prev_union) = {
        let Some(res) = tree.get_mut(r) else {
            return;
        };
        let prev = res.watch_demand;
        debug_assert!(prev > 0, "watch_demand underflow at {r:?}");
        if prev == 0 {
            return;
        }
        res.watch_demand = prev - 1;
        (prev, res.events_union)
    };

    if prev_refcount == 1 {
        // 1→0: clear union and emit Unwatch. No recompute needed — no
        // covering Profile remains.
        if let Some(res) = tree.get_mut(r) {
            res.events_union = ClassSet::EMPTY;
        }
        out.watch_ops.push(WatchOp::Unwatch { resource: r });
        return;
    }

    // refcount > 0 still: recompute the union over remaining covering
    // contributions. The releasing Profile's state-tracking flag must be
    // cleared by the caller before this call; the recompute then yields
    // the correct post-release union.
    let new_union = recompute_resource_events(tree, profiles, promoters, r, releasing_descendant);
    if new_union != prev_union {
        if let Some(res) = tree.get_mut(r) {
            res.events_union = new_union;
        }
        let path = tree.path_of(r).unwrap_or_default();
        // Preserve raw `Unknown` (sensor wildcard); see the rustdoc on
        // `Resource::kind_raw` and the parallel construction in
        // `add_watch_demand`.
        let kind = tree
            .get(r)
            .map_or(ResourceKind::Unknown, Resource::kind_raw);
        out.watch_ops.push(WatchOp::Watch {
            resource: r,
            path,
            kind,
            events: new_union,
        });
    }
}

/// Walk every Profile in `profiles` and every Promoter in `promoters`,
/// ORing each entity's contribution to `resource` into a running union.
///
/// Profile contributions — four sources, each matched to its dedicated
/// per-Profile bookkeeping field:
///
/// 1. **Anchor.** `Profile.anchor_claim == AnchorClaim::Held` AND
///    `Profile.resource == resource` ⇒ contributes `Profile.events_union`.
/// 2. **Watch-root parent.** `Profile.watch_root_parent == Some(resource)`
///    ⇒ contributes `STRUCTURE` (the parent watch exists so the
///    Profile can detect anchor reappearance via the parent's
///    `StructureChanged` event).
/// 3. **Pending-descent prefix.** `Profile.state == Pending(d)` AND
///    `d.current_prefix == resource` ⇒ contributes `STRUCTURE` (the
///    descent prefix watch exists so the engine sees the next path
///    segment materialize).
/// 4. **Covered descendant.** `resource != Profile.resource` AND
///    `Profile.current.is_some()` AND
///    `Some(pid) != releasing_descendant` AND
///    `covers(Profile, resource, tree) == true` ⇒ contributes
///    `Profile.events_union` if the resource is a Dir (always-watched
///    under the reconciler's gating) or if `Profile.has_per_file_fds` is
///    true (per-file FD demand for in-place edit detection).
///
/// The descendant clause has two gates beyond the topology check:
/// - `Profile.current.is_some()` excludes Profiles that hold no descendant
///   claims (Pending-without-Seed, post-Vanished-with-current-cleared,
///   post-`release_descendant_claim`-take). The snapshot itself is the
///   per-Profile bookkeeping for descendant claims; without `current`, the
///   Profile claims no descendants regardless of topology.
/// - `Some(pid) != releasing_descendant` is the explicit skip for the
///   delete-during-graft case (`reconcile::delete_child` runs while
///   `Profile.current` is still `Some`; the gate excludes this Profile's
///   own descendant contribution from the post-decrement recompute).
///
/// Together the two gates ensure the recompute reports the post-release
/// union without depending on `current.is_some()` having flipped (graft
/// hasn't taken `current` yet at `delete_child` time).
///
/// Promoter contributions — two mutually-exclusive sources, each gated
/// on `Promoter.state` exhaustively (the discriminator is the only way
/// the same Promoter could over-contribute to the same resource):
///
/// 5a. **PrefixPending prefix.** `Promoter.state == PrefixPending(d)` AND
///     `d.current_prefix == resource` ⇒ contributes `STRUCTURE`. The
///     descent prefix watch lets the engine see the next literal
///     segment materialize.
/// 5b. **Active proxy.** `Promoter.state == Active { proxies }` AND
///     `proxies.contains_key(&resource)` ⇒ contributes `STRUCTURE`. Each
///     proxy is a directory the engine wants to enumerate on next
///     event.
///
/// Both Promoter sources contribute `STRUCTURE` only — the proxy /
/// prefix watch's purpose is to discover children appearing or
/// disappearing. Proxy events route to the Promoter's enumeration
/// dispatcher, independent of Profile-side bursts.
///
/// `releasing_descendant` is Profile-only (no Promoter analogue): a
/// Promoter holds at most one contribution per resource (5a XOR 5b),
/// and the lifecycle helpers flip `Promoter.state` *before* calling
/// `sub_watch_demand`, mirroring the Profile-side flag-clear discipline.
/// The recompute reads post-flip state directly.
fn recompute_resource_events(
    tree: &Tree,
    profiles: &ProfileMap,
    promoters: &PromoterRegistry,
    resource: ResourceId,
    releasing_descendant: Option<ProfileId>,
) -> ClassSet {
    let mut union = ClassSet::EMPTY;
    for (pid, p) in profiles.iter() {
        union |= profile_contribution_for(p, pid, resource, tree, releasing_descendant);
    }
    for (_qid, q) in promoters.iter() {
        union |= promoter_contribution_for(q, resource);
    }
    union
}

/// Single Promoter's contribution to `resource`'s `events_union`.
///
/// Two mutually-exclusive sources, keyed off `Promoter.state`:
/// - **5a. PrefixPending prefix** (state == PrefixPending && d.current_prefix == r) ⇒ STRUCTURE.
/// - **5b. Active proxy** (state == Active && proxies.contains_key(&r)) ⇒ STRUCTURE.
///
/// The two arms cannot fire simultaneously: state is a sum-type, and a
/// PrefixPending → Active transition is a single state-flip. The only
/// race is during the helper that performs the transition itself, which
/// flips `Promoter.state` *before* calling `sub_watch_demand` against
/// the prior prefix — analogous to the Profile-side flag-clear
/// discipline.
fn promoter_contribution_for(promoter: &Promoter, resource: ResourceId) -> ClassSet {
    match &promoter.state {
        PromoterState::PrefixPending(d) if d.current_prefix == resource => ClassSet::STRUCTURE,
        PromoterState::Active { proxies } if proxies.contains_key(&resource) => ClassSet::STRUCTURE,
        _ => ClassSet::EMPTY,
    }
}

/// Single Profile's contribution to `resource`'s `events_union`. Computes
/// the union of every applicable source per `recompute_resource_events`'s
/// four-source enumeration. Pure read; no mutation.
fn profile_contribution_for(
    profile: &Profile,
    pid: ProfileId,
    resource: ResourceId,
    tree: &Tree,
    releasing_descendant: Option<ProfileId>,
) -> ClassSet {
    let mut union = ClassSet::EMPTY;

    // 1. Anchor — requires the per-Profile `anchor_claim == Held` so
    // anchor terminal events (which clear the claim) immediately stop
    // contributing.
    if matches!(profile.anchor_claim, AnchorClaim::Held) && profile.resource == resource {
        union |= profile.events_union;
    }

    // 2. Watch-root parent — STRUCTURE contribution.
    if profile.watch_root_parent == Some(resource) {
        union |= ClassSet::STRUCTURE;
    }

    // 3. Pending-descent prefix — STRUCTURE contribution.
    if let ProfileState::Pending(d) = &profile.state
        && d.current_prefix == resource
    {
        union |= ClassSet::STRUCTURE;
    }

    // 4. Covered descendant. Two gates beyond the topology check:
    //    * `current.is_some()` — Profiles without a snapshot hold no
    //      descendant claims by definition.
    //    * `Some(pid) != releasing_descendant` — explicit skip for the
    //      delete-during-graft case where `current` is still `Some` while
    //      this Profile's descendant claim is being released.
    let releasing_this = releasing_descendant == Some(pid);
    if !releasing_this
        && profile.current.is_some()
        && profile.resource != resource
        && covers(profile, resource, tree)
    {
        // Per-Resource Dir contribution gates on a *definitely-Dir*
        // classification — unprobed slots fold under the
        // `has_per_file_fds` branch instead, matching the engine's
        // 1-to-N "Dir always; Leaf under has_per_file_fds" contract.
        let is_dir = matches!(
            tree.get(resource).and_then(Resource::kind),
            Some(ResourceKind::Dir),
        );
        if is_dir || profile.has_per_file_fds {
            union |= profile.events_union;
        }
    }

    union
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

/// `-suppress_count` on `r`. Emits `WatchOp::Unsuppress` on the 1→0 edge.
/// Same underflow discipline as `sub_watch_demand`.
pub fn sub_suppress(tree: &mut Tree, r: ResourceId, out: &mut StepOutput) {
    let Some(res) = tree.get_mut(r) else {
        return;
    };
    let prev = res.suppress_count;
    debug_assert!(prev > 0, "suppress_count underflow at {r:?}");
    if prev == 0 {
        return;
    }
    res.suppress_count = prev - 1;
    if prev == 1 {
        out.watch_ops.push(WatchOp::Unsuppress { resource: r });
    }
}

/// Clamp `Resource.watch_demand` (plus `events_union` and `kind`) to
/// zero atomically, dropping every kernel-watch contribution at once.
/// Sole legitimate use: `Input::WatchOpRejected` recovery — the Sensor
/// failed to install the kernel watch, so the Engine has to revert to
/// "this Resource is not watched at all". The matching per-Profile
/// claim cleanup is the caller's responsibility (see
/// `Engine::on_watch_op_rejected`'s fan-out).
///
/// Emits `WatchOp::Unwatch` iff `watch_demand` was previously > 0; the
/// Sensor's idempotence guards repeats. `events_union` is reset to
/// `ClassSet::EMPTY` so the next 0→1 contribution starts the union fresh.
/// `kind` is reset to `Unknown` so the next probe can stamp it from the
/// response.
///
/// **`suppress_count` is deliberately preserved.** Suppression is
/// in-engine bookkeeping for in-flight burst phases; it tracks
/// `start_*_burst` ↔ `finish_burst_to_idle` symmetry on the Profile
/// side, not the kernel-watch existence. Zeroing it would underflow
/// `sub_suppress` when the affected Profile's burst eventually
/// finishes (`finalize_anchor_lost` → `finish_burst_to_idle` →
/// `sub_suppress`). The caller's per-claim fan-out drives the
/// burst-end machinery; suppress decrements come for free from there.
///
/// A stale `ResourceId` (slot already reaped) is a no-op + no emission;
/// the caller emits the corresponding `Diagnostic` at the call site.
pub fn clamp_watch_demand_to_zero(tree: &mut Tree, r: ResourceId, out: &mut StepOutput) {
    let Some(res) = tree.get_mut(r) else {
        return;
    };
    if res.watch_demand == 0 {
        return;
    }
    res.watch_demand = 0;
    res.events_union = ClassSet::EMPTY;
    // The mutable borrow on `res` ends here; `set_kind` reborrows.
    tree.set_kind(r, ResourceKind::Unknown);
    out.watch_ops.push(WatchOp::Unwatch { resource: r });
}

#[cfg(test)]
mod tests {
    use super::{
        add_suppress, add_watch_demand, clamp_watch_demand_to_zero, recompute_resource_events,
        sub_suppress, sub_watch_demand,
    };
    use compact_str::CompactString;
    use specter_core::{
        AnchorClaim, ClassSet, DescentState, Profile, ProfileMap, ProfileState, PromoterRegistry,
        ResourceKind, ResourceRole, ScanConfig, StepOutput, Tree, WatchOp,
    };
    use std::time::Duration;

    const SETTLE: Duration = Duration::from_millis(100);
    const MAX_SETTLE: Duration = Duration::from_secs(6);

    fn fresh() -> (Tree, specter_core::ResourceId) {
        let mut tree = Tree::new();
        let r = tree.ensure(None, "anchor", ResourceRole::User);
        (tree, r)
    }

    fn empty_profiles() -> ProfileMap {
        ProfileMap::new()
    }

    /// Empty promoter registry for `sub_watch_demand` /
    /// `recompute_resource_events` test calls. These unit tests
    /// exercise Profile-only behaviour; Promoter-side recompute
    /// coverage uses non-empty registries elsewhere.
    fn empty_promoters() -> PromoterRegistry {
        PromoterRegistry::new()
    }

    fn cfg() -> ScanConfig {
        ScanConfig::builder().recursive(true).build()
    }

    /// Last `WatchOp::Watch` emitted, for asserting on its `events`.
    fn last_watch_events(out: &StepOutput) -> Option<ClassSet> {
        out.watch_ops.iter().rev().find_map(|op| match op {
            WatchOp::Watch { events, .. } => Some(*events),
            _ => None,
        })
    }

    #[test]
    fn add_watch_demand_zero_to_one_emits_watch_with_contribution() {
        let (mut tree, r) = fresh();
        let mut out = StepOutput::default();
        add_watch_demand(&mut tree, r, ClassSet::CONTENT, &mut out);
        assert_eq!(tree.get(r).unwrap().watch_demand, 1);
        assert_eq!(tree.get(r).unwrap().events_union, ClassSet::CONTENT);
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
    fn add_watch_demand_one_to_two_emits_watch_only_when_union_widens() {
        // 0→1 with CONTENT: emits Watch.
        // 1→2 with CONTENT (already in union): no emit.
        // 2→3 with METADATA (widens union): emits Watch with CONTENT|METADATA.
        let (mut tree, r) = fresh();
        let mut out = StepOutput::default();

        add_watch_demand(&mut tree, r, ClassSet::CONTENT, &mut out);
        out.watch_ops.clear();

        add_watch_demand(&mut tree, r, ClassSet::CONTENT, &mut out);
        assert_eq!(tree.get(r).unwrap().watch_demand, 2);
        assert!(
            out.watch_ops.is_empty(),
            "no Watch emitted when union unchanged at refcount > 0"
        );

        add_watch_demand(&mut tree, r, ClassSet::METADATA, &mut out);
        assert_eq!(tree.get(r).unwrap().watch_demand, 3);
        assert_eq!(
            tree.get(r).unwrap().events_union,
            ClassSet::CONTENT | ClassSet::METADATA,
        );
        assert_eq!(out.watch_ops.len(), 1);
        assert_eq!(
            last_watch_events(&out),
            Some(ClassSet::CONTENT | ClassSet::METADATA),
        );
    }

    #[test]
    fn add_watch_demand_with_empty_contribution_at_zero_emits_identity_floor_watch() {
        // 0→1 with EMPTY contribution: still emits Watch (existence edge),
        // but `opts.events == EMPTY` ⇒ Sensor degrades to identity floor.
        let (mut tree, r) = fresh();
        let mut out = StepOutput::default();
        add_watch_demand(&mut tree, r, ClassSet::EMPTY, &mut out);
        assert_eq!(tree.get(r).unwrap().watch_demand, 1);
        assert_eq!(tree.get(r).unwrap().events_union, ClassSet::EMPTY);
        assert_eq!(out.watch_ops.len(), 1);
        assert_eq!(last_watch_events(&out), Some(ClassSet::EMPTY));
    }

    #[test]
    fn sub_watch_demand_one_to_zero_emits_unwatch_and_clears_union() {
        let (mut tree, r) = fresh();
        let profiles = empty_profiles();
        let promoters = empty_promoters();
        let mut out = StepOutput::default();
        add_watch_demand(&mut tree, r, ClassSet::CONTENT, &mut out);
        out.watch_ops.clear();

        sub_watch_demand(
            &mut tree,
            &profiles,
            &promoters,
            r,
            ClassSet::CONTENT,
            None,
            &mut out,
        );
        assert_eq!(tree.get(r).unwrap().watch_demand, 0);
        assert_eq!(tree.get(r).unwrap().events_union, ClassSet::EMPTY);
        assert_eq!(out.watch_ops.len(), 1);
        assert!(matches!(
            out.watch_ops[0],
            WatchOp::Unwatch { resource } if resource == r,
        ));
    }

    #[test]
    fn sub_watch_demand_above_one_recomputes_union_from_profiles() {
        // Two Profiles cover the anchor; both contribute via
        // `anchor_claim = AnchorClaim::Held`. Sub one → recompute walks the
        // remaining Profile and yields its mask.
        let mut tree = Tree::new();
        let r = tree.ensure(None, "anchor", ResourceRole::User);
        let mut profiles = ProfileMap::new();
        let promoters = empty_promoters();

        // Profile A: events_union = CONTENT
        let p_a = profiles.attach(
            &mut tree,
            Profile::new(r, cfg(), MAX_SETTLE, SETTLE, ClassSet::CONTENT),
        );
        // Profile B: events_union = METADATA — needs different config_hash
        // to attach at the same Resource. Use different max_settle.
        let p_b = profiles.attach(
            &mut tree,
            Profile::new(
                r,
                cfg(),
                MAX_SETTLE + Duration::from_secs(1),
                SETTLE,
                ClassSet::METADATA,
            ),
        );
        // Mark anchor_claim = Held on both (simulates the post-attach state
        // where each Profile has bumped the anchor's watch_demand).
        profiles.get_mut(p_a).unwrap().anchor_claim = AnchorClaim::Held;
        profiles.get_mut(p_b).unwrap().anchor_claim = AnchorClaim::Held;

        let mut out = StepOutput::default();
        add_watch_demand(&mut tree, r, ClassSet::CONTENT, &mut out);
        add_watch_demand(&mut tree, r, ClassSet::METADATA, &mut out);
        assert_eq!(tree.get(r).unwrap().watch_demand, 2);
        assert_eq!(
            tree.get(r).unwrap().events_union,
            ClassSet::CONTENT | ClassSet::METADATA,
        );

        // Simulate Profile A releasing: clear its anchor_claim BEFORE
        // sub_watch_demand so the recompute reflects the post-release
        // state. The recompute should then yield METADATA only.
        profiles.get_mut(p_a).unwrap().anchor_claim = AnchorClaim::None;
        out.watch_ops.clear();
        sub_watch_demand(
            &mut tree,
            &profiles,
            &promoters,
            r,
            ClassSet::CONTENT,
            None,
            &mut out,
        );

        assert_eq!(tree.get(r).unwrap().watch_demand, 1);
        assert_eq!(tree.get(r).unwrap().events_union, ClassSet::METADATA);
        assert_eq!(
            last_watch_events(&out),
            Some(ClassSet::METADATA),
            "Watch emitted with the recomputed (narrower) mask",
        );
        // No Unwatch — refcount is still > 0.
        assert!(
            !out.watch_ops
                .iter()
                .any(|op| matches!(op, WatchOp::Unwatch { .. })),
        );
    }

    #[test]
    fn sub_watch_demand_above_one_no_emit_when_union_unchanged() {
        // Both Profiles contribute the same mask. Releasing one preserves
        // the union (the other still contributes the same bits).
        let mut tree = Tree::new();
        let r = tree.ensure(None, "anchor", ResourceRole::User);
        let mut profiles = ProfileMap::new();
        let promoters = empty_promoters();

        let p_a = profiles.attach(
            &mut tree,
            Profile::new(r, cfg(), MAX_SETTLE, SETTLE, ClassSet::CONTENT),
        );
        let p_b = profiles.attach(
            &mut tree,
            Profile::new(
                r,
                cfg(),
                MAX_SETTLE + Duration::from_secs(1),
                SETTLE,
                ClassSet::CONTENT,
            ),
        );
        profiles.get_mut(p_a).unwrap().anchor_claim = AnchorClaim::Held;
        profiles.get_mut(p_b).unwrap().anchor_claim = AnchorClaim::Held;

        let mut out = StepOutput::default();
        add_watch_demand(&mut tree, r, ClassSet::CONTENT, &mut out);
        add_watch_demand(&mut tree, r, ClassSet::CONTENT, &mut out);
        out.watch_ops.clear();

        profiles.get_mut(p_a).unwrap().anchor_claim = AnchorClaim::None;
        sub_watch_demand(
            &mut tree,
            &profiles,
            &promoters,
            r,
            ClassSet::CONTENT,
            None,
            &mut out,
        );

        assert_eq!(tree.get(r).unwrap().watch_demand, 1);
        assert_eq!(tree.get(r).unwrap().events_union, ClassSet::CONTENT);
        assert!(
            out.watch_ops.is_empty(),
            "no Watch emitted when recomputed union equals prior",
        );
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "watch_demand underflow")]
    fn sub_watch_demand_underflow_panics_in_debug() {
        let (mut tree, r) = fresh();
        let profiles = empty_profiles();
        let promoters = empty_promoters();
        let mut out = StepOutput::default();
        sub_watch_demand(
            &mut tree,
            &profiles,
            &promoters,
            r,
            ClassSet::EMPTY,
            None,
            &mut out,
        );
    }

    #[test]
    fn add_watch_demand_stale_resource_is_noop() {
        let mut tree = Tree::new();
        let r = tree.ensure(None, "ghost", ResourceRole::User);
        assert!(tree.try_reap(r));
        let mut out = StepOutput::default();
        add_watch_demand(&mut tree, r, ClassSet::STRUCTURE, &mut out);
        assert!(out.watch_ops.is_empty());
    }

    #[test]
    fn sub_watch_demand_stale_resource_is_noop() {
        let mut tree = Tree::new();
        let r = tree.ensure(None, "ghost", ResourceRole::User);
        assert!(tree.try_reap(r));
        let profiles = empty_profiles();
        let promoters = empty_promoters();
        let mut out = StepOutput::default();
        sub_watch_demand(
            &mut tree,
            &profiles,
            &promoters,
            r,
            ClassSet::EMPTY,
            None,
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

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "suppress_count underflow")]
    fn sub_suppress_underflow_panics_in_debug() {
        let (mut tree, r) = fresh();
        let mut out = StepOutput::default();
        sub_suppress(&mut tree, r, &mut out);
    }

    #[test]
    fn watch_and_suppress_are_independent() {
        let (mut tree, r) = fresh();
        let profiles = empty_profiles();
        let promoters = empty_promoters();
        let mut out = StepOutput::default();
        add_watch_demand(&mut tree, r, ClassSet::CONTENT, &mut out);
        add_suppress(&mut tree, r, &mut out);
        let res = tree.get(r).unwrap();
        assert_eq!(res.watch_demand, 1);
        assert_eq!(res.suppress_count, 1);
        sub_watch_demand(
            &mut tree,
            &profiles,
            &promoters,
            r,
            ClassSet::CONTENT,
            None,
            &mut out,
        );
        let res = tree.get(r).unwrap();
        assert_eq!(res.watch_demand, 0);
        // suppress unchanged by the watch decrement.
        assert_eq!(res.suppress_count, 1);
    }

    #[test]
    fn clamp_watch_demand_to_zero_emits_unwatch_and_clears_events_union() {
        let (mut tree, r) = fresh();
        let mut out = StepOutput::default();
        add_watch_demand(&mut tree, r, ClassSet::CONTENT, &mut out);
        add_watch_demand(&mut tree, r, ClassSet::METADATA, &mut out);
        add_watch_demand(&mut tree, r, ClassSet::STRUCTURE, &mut out);
        assert_eq!(tree.get(r).unwrap().watch_demand, 3);
        assert_eq!(
            tree.get(r).unwrap().events_union,
            ClassSet::CONTENT | ClassSet::METADATA | ClassSet::STRUCTURE,
        );
        out.watch_ops.clear();

        clamp_watch_demand_to_zero(&mut tree, r, &mut out);

        let res = tree.get(r).unwrap();
        assert_eq!(res.watch_demand, 0);
        assert_eq!(res.suppress_count, 0);
        assert!(res.kind().is_none(), "clamp resets kind to Unknown");
        assert_eq!(res.events_union, ClassSet::EMPTY);
        assert_eq!(out.watch_ops.len(), 1);
        assert!(matches!(
            out.watch_ops[0],
            WatchOp::Unwatch { resource } if resource == r,
        ));
    }

    #[test]
    fn clamp_watch_demand_to_zero_already_zero_is_noop() {
        let (mut tree, r) = fresh();
        let mut out = StepOutput::default();
        clamp_watch_demand_to_zero(&mut tree, r, &mut out);
        assert!(out.watch_ops.is_empty());
        assert_eq!(tree.get(r).unwrap().watch_demand, 0);
        assert_eq!(tree.get(r).unwrap().events_union, ClassSet::EMPTY);
    }

    #[test]
    fn clamp_watch_demand_to_zero_stale_resource_is_noop() {
        let mut tree = Tree::new();
        let r = tree.ensure(None, "ghost", ResourceRole::User);
        assert!(tree.try_reap(r));
        let mut out = StepOutput::default();
        clamp_watch_demand_to_zero(&mut tree, r, &mut out);
        assert!(out.watch_ops.is_empty());
    }

    #[test]
    fn clamp_watch_demand_to_zero_preserves_suppress_count() {
        // Suppression is in-engine bookkeeping for in-flight burst phases;
        // clamp tracks the kernel-watch existence (FD lifetime) only.
        // Zeroing suppress_count would break the start_*_burst ↔
        // finish_burst_to_idle symmetry on the Profile side and
        // underflow sub_suppress when the affected Profile's burst
        // eventually finishes via finalize_anchor_lost.
        let (mut tree, r) = fresh();
        let mut out = StepOutput::default();
        add_watch_demand(&mut tree, r, ClassSet::CONTENT, &mut out);
        add_suppress(&mut tree, r, &mut out);
        add_suppress(&mut tree, r, &mut out);
        assert_eq!(tree.get(r).unwrap().suppress_count, 2);
        out.watch_ops.clear();

        clamp_watch_demand_to_zero(&mut tree, r, &mut out);

        assert_eq!(
            tree.get(r).unwrap().suppress_count,
            2,
            "clamp leaves suppress_count untouched — the Profile's burst-end \
             machinery decrements it symmetrically",
        );
        let unwatch_count = out
            .watch_ops
            .iter()
            .filter(|op| matches!(op, WatchOp::Unwatch { .. }))
            .count();
        assert_eq!(unwatch_count, 1);
        // No Unsuppress emit either — the clamp is silent on suppress.
        let unsuppress_count = out
            .watch_ops
            .iter()
            .filter(|op| matches!(op, WatchOp::Unsuppress { .. }))
            .count();
        assert_eq!(unsuppress_count, 0);
    }

    // ---------------------------------------------------------------------------
    // recompute_resource_events — direct unit tests
    // ---------------------------------------------------------------------------

    #[test]
    fn recompute_with_zero_profiles_yields_empty() {
        let (tree, r) = fresh();
        let profiles = empty_profiles();
        let promoters = empty_promoters();
        assert_eq!(
            recompute_resource_events(&tree, &profiles, &promoters, r, None),
            ClassSet::EMPTY,
        );
    }

    #[test]
    fn recompute_with_single_anchor_profile_yields_its_mask() {
        let mut tree = Tree::new();
        let r = tree.ensure(None, "anchor", ResourceRole::User);
        let mut profiles = ProfileMap::new();
        let promoters = empty_promoters();
        let pid = profiles.attach(
            &mut tree,
            Profile::new(r, cfg(), MAX_SETTLE, SETTLE, ClassSet::CONTENT),
        );
        profiles.get_mut(pid).unwrap().anchor_claim = AnchorClaim::Held;

        assert_eq!(
            recompute_resource_events(&tree, &profiles, &promoters, r, None),
            ClassSet::CONTENT,
        );
    }

    #[test]
    fn recompute_excludes_anchor_when_anchor_claim_none() {
        // The claim is the source of truth; without `Held`, the Profile's
        // anchor mask doesn't contribute.
        let mut tree = Tree::new();
        let r = tree.ensure(None, "anchor", ResourceRole::User);
        let mut profiles = ProfileMap::new();
        let promoters = empty_promoters();
        let pid = profiles.attach(
            &mut tree,
            Profile::new(r, cfg(), MAX_SETTLE, SETTLE, ClassSet::CONTENT),
        );
        // anchor_claim defaults to None.
        assert_eq!(profiles.get(pid).unwrap().anchor_claim, AnchorClaim::None,);

        assert_eq!(
            recompute_resource_events(&tree, &profiles, &promoters, r, None),
            ClassSet::EMPTY,
        );
    }

    #[test]
    fn recompute_or_s_two_anchor_profiles_with_overlapping_classes() {
        let mut tree = Tree::new();
        let r = tree.ensure(None, "anchor", ResourceRole::User);
        let mut profiles = ProfileMap::new();
        let promoters = empty_promoters();
        let p_a = profiles.attach(
            &mut tree,
            Profile::new(r, cfg(), MAX_SETTLE, SETTLE, ClassSet::CONTENT),
        );
        let p_b = profiles.attach(
            &mut tree,
            Profile::new(
                r,
                cfg(),
                MAX_SETTLE + Duration::from_secs(1),
                SETTLE,
                ClassSet::CONTENT | ClassSet::METADATA,
            ),
        );
        profiles.get_mut(p_a).unwrap().anchor_claim = AnchorClaim::Held;
        profiles.get_mut(p_b).unwrap().anchor_claim = AnchorClaim::Held;

        assert_eq!(
            recompute_resource_events(&tree, &profiles, &promoters, r, None),
            ClassSet::CONTENT | ClassSet::METADATA,
        );
    }

    #[test]
    fn recompute_includes_watch_root_parent_as_structure() {
        // A Profile whose `watch_root_parent == resource` contributes
        // STRUCTURE regardless of its own events_union mask.
        let mut tree = Tree::new();
        let parent = tree.ensure(None, "p", ResourceRole::WatchRootParent);
        let anchor = tree.ensure(Some(parent), "a", ResourceRole::User);
        let mut profiles = ProfileMap::new();
        let promoters = empty_promoters();
        let pid = profiles.attach(
            &mut tree,
            Profile::new(anchor, cfg(), MAX_SETTLE, SETTLE, ClassSet::CONTENT),
        );
        profiles.get_mut(pid).unwrap().watch_root_parent = Some(parent);

        // Recomputing on `parent` yields STRUCTURE only — the Profile is
        // contributing to its watch-root parent, not its anchor.
        assert_eq!(
            recompute_resource_events(&tree, &profiles, &promoters, parent, None),
            ClassSet::STRUCTURE,
        );
    }

    #[test]
    fn recompute_includes_descent_prefix_as_structure() {
        let mut tree = Tree::new();
        let prefix = tree.ensure(None, "p", ResourceRole::DescentScaffold);
        let scaffold = tree.ensure(Some(prefix), "anchor", ResourceRole::DescentScaffold);
        let mut profiles = ProfileMap::new();
        let promoters = empty_promoters();
        let pid = profiles.attach(
            &mut tree,
            Profile::new(scaffold, cfg(), MAX_SETTLE, SETTLE, ClassSet::CONTENT),
        );
        profiles.get_mut(pid).unwrap().state = ProfileState::Pending(DescentState {
            current_prefix: prefix,
            remaining_components: vec![CompactString::from("anchor")],
        });

        assert_eq!(
            recompute_resource_events(&tree, &profiles, &promoters, prefix, None),
            ClassSet::STRUCTURE,
        );
    }

    #[test]
    fn recompute_or_s_three_distinct_sources() {
        // Anchor of Profile A + watch-root parent of Profile B + descent
        // prefix of Profile C — all targeting the same resource.
        let mut tree = Tree::new();
        let r = tree.ensure(None, "shared", ResourceRole::User);
        let mut profiles = ProfileMap::new();
        let promoters = empty_promoters();

        // Profile A: anchored at r, mask = CONTENT.
        let p_a = profiles.attach(
            &mut tree,
            Profile::new(r, cfg(), MAX_SETTLE, SETTLE, ClassSet::CONTENT),
        );
        profiles.get_mut(p_a).unwrap().anchor_claim = AnchorClaim::Held;

        // Profile B: anchored elsewhere, watch_root_parent == r.
        let other_b = tree.ensure(Some(r), "child_b", ResourceRole::User);
        let p_b = profiles.attach(
            &mut tree,
            Profile::new(other_b, cfg(), MAX_SETTLE, SETTLE, ClassSet::METADATA),
        );
        profiles.get_mut(p_b).unwrap().watch_root_parent = Some(r);

        // Profile C: pending descent at r.
        let scaffold_c = tree.ensure(Some(r), "scaffold_c", ResourceRole::DescentScaffold);
        let p_c = profiles.attach(
            &mut tree,
            Profile::new(
                scaffold_c,
                cfg(),
                MAX_SETTLE + Duration::from_secs(2),
                SETTLE,
                ClassSet::CONTENT,
            ),
        );
        profiles.get_mut(p_c).unwrap().state = ProfileState::Pending(DescentState {
            current_prefix: r,
            remaining_components: vec![CompactString::from("x")],
        });

        // Anchor of A (CONTENT) | parent-edge of B (STRUCTURE) | descent of C (STRUCTURE)
        // = CONTENT | STRUCTURE.
        assert_eq!(
            recompute_resource_events(&tree, &profiles, &promoters, r, None),
            ClassSet::CONTENT | ClassSet::STRUCTURE,
        );
    }

    /// Stub `Profile.current` to a placeholder `TreeSnapshot::Dir` so the
    /// descendant clause's `current.is_some()` gate is satisfied. Tests in
    /// this module exercise the recompute logic in isolation; production
    /// sets `current` via graft.
    fn stub_current(profiles: &mut ProfileMap, pid: specter_core::ProfileId) {
        use specter_core::{DirMeta, DirSnapshot, TreeSnapshot};
        use std::collections::BTreeMap;
        use std::sync::Arc;
        use std::time::UNIX_EPOCH;
        let snap = DirSnapshot::new(
            specter_core::ResourceId::default(),
            DirMeta {
                mtime: UNIX_EPOCH,
                inode: 0,
                device: 0,
            },
            0,
            BTreeMap::new(),
        );
        profiles.get_mut(pid).unwrap().current = Some(TreeSnapshot::Dir(Arc::new(snap)));
    }

    #[test]
    fn recompute_includes_covered_dir_descendant() {
        // Profile A is anchored at root with recursive=true. A subdirectory
        // is a covered descendant — it should contribute A.events_union
        // when its kind is Dir.
        let mut tree = Tree::new();
        let root = tree.ensure(None, "root", ResourceRole::User);
        tree.set_kind(root, ResourceKind::Dir);
        let sub = tree.ensure(Some(root), "sub", ResourceRole::User);
        tree.set_kind(sub, ResourceKind::Dir);
        let mut profiles = ProfileMap::new();
        let promoters = empty_promoters();
        let pid = profiles.attach(
            &mut tree,
            Profile::new(root, cfg(), MAX_SETTLE, SETTLE, ClassSet::STRUCTURE),
        );
        // Descendant clause requires `current.is_some()` — a Profile with
        // no snapshot holds no descendant claims by definition.
        stub_current(&mut profiles, pid);

        assert_eq!(
            recompute_resource_events(&tree, &profiles, &promoters, sub, None),
            ClassSet::STRUCTURE,
        );
    }

    #[test]
    fn recompute_excludes_descendant_when_current_is_none() {
        // Profile covers `sub` topologically but `current.is_none()` ⇒
        // descendant clause is gated off (the snapshot is the per-Profile
        // bookkeeping for descendant claims).
        let mut tree = Tree::new();
        let root = tree.ensure(None, "root", ResourceRole::User);
        tree.set_kind(root, ResourceKind::Dir);
        let sub = tree.ensure(Some(root), "sub", ResourceRole::User);
        tree.set_kind(sub, ResourceKind::Dir);
        let mut profiles = ProfileMap::new();
        let promoters = empty_promoters();
        let pid = profiles.attach(
            &mut tree,
            Profile::new(root, cfg(), MAX_SETTLE, SETTLE, ClassSet::STRUCTURE),
        );
        assert!(profiles.get(pid).unwrap().current.is_none());

        assert_eq!(
            recompute_resource_events(&tree, &profiles, &promoters, sub, None),
            ClassSet::EMPTY,
        );
    }

    #[test]
    fn recompute_excludes_descendant_when_releasing_descendant_matches() {
        // Profile covers `sub` and has `current = Some` — would normally
        // contribute. The explicit `releasing_descendant: Some(pid)` skip
        // closes F-MED-4: during `delete_child` mid-graft, `current` is
        // still set while the Profile is releasing its claim.
        let mut tree = Tree::new();
        let root = tree.ensure(None, "root", ResourceRole::User);
        tree.set_kind(root, ResourceKind::Dir);
        let sub = tree.ensure(Some(root), "sub", ResourceRole::User);
        tree.set_kind(sub, ResourceKind::Dir);
        let mut profiles = ProfileMap::new();
        let promoters = empty_promoters();
        let pid = profiles.attach(
            &mut tree,
            Profile::new(root, cfg(), MAX_SETTLE, SETTLE, ClassSet::STRUCTURE),
        );
        stub_current(&mut profiles, pid);

        // Without the skip: contributes STRUCTURE.
        assert_eq!(
            recompute_resource_events(&tree, &profiles, &promoters, sub, None),
            ClassSet::STRUCTURE,
        );
        // With the skip: this Profile's descendant contribution is excluded.
        assert_eq!(
            recompute_resource_events(&tree, &profiles, &promoters, sub, Some(pid)),
            ClassSet::EMPTY,
        );
    }

    #[test]
    fn recompute_skips_covered_leaf_when_has_per_file_fds_false() {
        // STRUCTURE-only Profile ⇒ has_per_file_fds = false ⇒ covered
        // leaves do NOT contribute (matches walk_pair gating).
        let mut tree = Tree::new();
        let root = tree.ensure(None, "root", ResourceRole::User);
        tree.set_kind(root, ResourceKind::Dir);
        let leaf = tree.ensure(Some(root), "f.rs", ResourceRole::User);
        tree.set_kind(leaf, ResourceKind::File);
        let mut profiles = ProfileMap::new();
        let promoters = empty_promoters();
        let pid = profiles.attach(
            &mut tree,
            Profile::new(root, cfg(), MAX_SETTLE, SETTLE, ClassSet::STRUCTURE),
        );
        assert!(!profiles.get(pid).unwrap().has_per_file_fds);
        stub_current(&mut profiles, pid);

        assert_eq!(
            recompute_resource_events(&tree, &profiles, &promoters, leaf, None),
            ClassSet::EMPTY,
        );
    }

    #[test]
    fn recompute_includes_covered_leaf_when_has_per_file_fds_true() {
        // CONTENT (or METADATA) ⇒ has_per_file_fds = true ⇒ covered leaves
        // contribute the Profile's events_union.
        let mut tree = Tree::new();
        let root = tree.ensure(None, "root", ResourceRole::User);
        tree.set_kind(root, ResourceKind::Dir);
        let leaf = tree.ensure(Some(root), "f.rs", ResourceRole::User);
        tree.set_kind(leaf, ResourceKind::File);
        let mut profiles = ProfileMap::new();
        let promoters = empty_promoters();
        let pid = profiles.attach(
            &mut tree,
            Profile::new(root, cfg(), MAX_SETTLE, SETTLE, ClassSet::CONTENT),
        );
        stub_current(&mut profiles, pid);

        assert_eq!(
            recompute_resource_events(&tree, &profiles, &promoters, leaf, None),
            ClassSet::CONTENT,
        );
    }
}
