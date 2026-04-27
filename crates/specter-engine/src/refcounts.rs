//! Refcount-edge helpers for `Resource.watch_demand` and
//! `Resource.suppress_count`.
//!
//! Two refcounts, decoupled:
//! - `watch_demand` gates FD lifetime — a Resource is Watched iff `> 0`.
//! - `suppress_count` gates event delivery — silenced iff `> 0`.
//!
//! Each helper emits the corresponding `WatchOp` only on the **0↔1 edge**;
//! repeated increments past 1 (or decrements above 0) are silent. Underflows
//! are debug-asserted; in release the counter clamps at 0 and the edge op is
//! suppressed (the Sensor is already-Unwatched/Unsuppressed in that state).
//!
//! Stale `ResourceId`: the lookup short-circuits with no mutation and no op
//! emission. The Engine maintains `watch_demand > 0 ⇒ live slot` (I6) by
//! attaching contributions only at live Resources, so a stale id here means
//! a logic bug elsewhere; the silent return is defense-in-depth.

use specter_core::{ResourceId, ResourceKind, StepOutput, Tree, WatchOp, WatchOpts};

/// `+watch_demand` on `r`. Emits `WatchOp::Watch` on the 0→1 edge. The
/// `WatchOps`'s `path` is resolved at emission via `Tree::path_of`; if path
/// resolution fails (the slot exists but a segment doesn't resolve through
/// the interner — unreachable for live slots), the op carries `PathBuf::new()`
/// and the Sensor reports `WatchOpRejected` on attempt.
pub fn add_watch_demand(tree: &mut Tree, r: ResourceId, out: &mut StepOutput) {
    let Some(res) = tree.get_mut(r) else {
        return;
    };
    let prev = res.watch_demand;
    res.watch_demand = prev.saturating_add(1);
    if prev == 0 {
        let path = tree.path_of(r).unwrap_or_default();
        out.watch_ops.push(WatchOp::Watch {
            resource: r,
            path,
            opts: WatchOpts::default(),
        });
    }
}

/// `-watch_demand` on `r`. Emits `WatchOp::Unwatch` on the 1→0 edge.
/// Underflow → `debug_assert!` panic in dev; in release the counter clamps
/// at 0 and no op is emitted (the Sensor is already in the Unwatched state).
pub fn sub_watch_demand(tree: &mut Tree, r: ResourceId, out: &mut StepOutput) {
    let Some(res) = tree.get_mut(r) else {
        return;
    };
    let prev = res.watch_demand;
    debug_assert!(prev > 0, "watch_demand underflow at {r:?}");
    if prev == 0 {
        return;
    }
    res.watch_demand = prev - 1;
    if prev == 1 {
        out.watch_ops.push(WatchOp::Unwatch { resource: r });
    }
}

/// `+suppress_count` on `r`. Emits `WatchOp::Suppress` on the 0→1 edge.
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

/// Clamp `Resource.watch_demand` (and `suppress_count`) to 0 atomically,
/// dropping every contribution at once. Sole legitimate use:
/// `Input::WatchOpRejected` recovery — the Sensor failed to install the
/// kernel watch, so the Engine has to revert to "this Resource is not
/// watched at all" and let reconciliation rebuild contributions from each
/// covering Profile on the parent's next `StructureChanged`.
///
/// Emits `WatchOp::Unwatch` iff `watch_demand` was previously > 0; the
/// Sensor's idempotence guards repeats. `kind` is reset to `Unknown` so
/// the next probe can stamp it from the response. A
/// stale `ResourceId` (slot already reaped) is a no-op + no emission;
/// the caller emits the corresponding `Diagnostic` at the call site.
pub fn clamp_watch_demand_to_zero(tree: &mut Tree, r: ResourceId, out: &mut StepOutput) {
    let Some(res) = tree.get_mut(r) else {
        return;
    };
    let prev = res.watch_demand;
    if prev == 0 {
        return;
    }
    res.watch_demand = 0;
    res.suppress_count = 0;
    res.kind = ResourceKind::Unknown;
    out.watch_ops.push(WatchOp::Unwatch { resource: r });
}

#[cfg(test)]
mod tests {
    use super::{
        add_suppress, add_watch_demand, clamp_watch_demand_to_zero, sub_suppress, sub_watch_demand,
    };
    use specter_core::{ResourceKind, ResourceRole, StepOutput, Tree, WatchOp};

    fn fresh() -> (Tree, specter_core::ResourceId) {
        let mut tree = Tree::new();
        let r = tree.ensure(None, "anchor", ResourceRole::User);
        (tree, r)
    }

    #[test]
    fn add_watch_demand_zero_to_one_emits_watch() {
        let (mut tree, r) = fresh();
        let mut out = StepOutput::default();
        add_watch_demand(&mut tree, r, &mut out);
        assert_eq!(tree.get(r).unwrap().watch_demand, 1);
        assert_eq!(out.watch_ops.len(), 1);
        match &out.watch_ops[0] {
            WatchOp::Watch { resource, .. } => assert_eq!(*resource, r),
            op => panic!("expected Watch, got {op:?}"),
        }
    }

    #[test]
    fn add_watch_demand_one_to_two_no_emit() {
        let (mut tree, r) = fresh();
        let mut out = StepOutput::default();
        add_watch_demand(&mut tree, r, &mut out);
        out.watch_ops.clear();
        add_watch_demand(&mut tree, r, &mut out);
        assert_eq!(tree.get(r).unwrap().watch_demand, 2);
        assert!(out.watch_ops.is_empty());
    }

    #[test]
    fn sub_watch_demand_one_to_zero_emits_unwatch() {
        let (mut tree, r) = fresh();
        let mut out = StepOutput::default();
        add_watch_demand(&mut tree, r, &mut out);
        out.watch_ops.clear();
        sub_watch_demand(&mut tree, r, &mut out);
        assert_eq!(tree.get(r).unwrap().watch_demand, 0);
        assert_eq!(out.watch_ops.len(), 1);
        assert!(matches!(
            out.watch_ops[0],
            WatchOp::Unwatch { resource } if resource == r,
        ));
    }

    #[test]
    fn sub_watch_demand_two_to_one_no_emit() {
        let (mut tree, r) = fresh();
        let mut out = StepOutput::default();
        add_watch_demand(&mut tree, r, &mut out);
        add_watch_demand(&mut tree, r, &mut out);
        out.watch_ops.clear();
        sub_watch_demand(&mut tree, r, &mut out);
        assert_eq!(tree.get(r).unwrap().watch_demand, 1);
        assert!(out.watch_ops.is_empty());
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "watch_demand underflow")]
    fn sub_watch_demand_underflow_panics_in_debug() {
        let (mut tree, r) = fresh();
        let mut out = StepOutput::default();
        sub_watch_demand(&mut tree, r, &mut out);
    }

    #[test]
    fn add_watch_demand_stale_resource_is_noop() {
        let mut tree = Tree::new();
        let r = tree.ensure(None, "ghost", ResourceRole::User);
        assert!(tree.try_reap(r));
        let mut out = StepOutput::default();
        add_watch_demand(&mut tree, r, &mut out);
        assert!(out.watch_ops.is_empty());
    }

    #[test]
    fn sub_watch_demand_stale_resource_is_noop() {
        let mut tree = Tree::new();
        let r = tree.ensure(None, "ghost", ResourceRole::User);
        assert!(tree.try_reap(r));
        let mut out = StepOutput::default();
        sub_watch_demand(&mut tree, r, &mut out);
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
        let mut out = StepOutput::default();
        add_watch_demand(&mut tree, r, &mut out);
        add_suppress(&mut tree, r, &mut out);
        let res = tree.get(r).unwrap();
        assert_eq!(res.watch_demand, 1);
        assert_eq!(res.suppress_count, 1);
        sub_watch_demand(&mut tree, r, &mut out);
        let res = tree.get(r).unwrap();
        assert_eq!(res.watch_demand, 0);
        // suppress unchanged by the watch decrement.
        assert_eq!(res.suppress_count, 1);
    }

    #[test]
    fn clamp_watch_demand_to_zero_emits_unwatch_for_loaded_resource() {
        // Clamp drops every contribution at once; emits Unwatch on the
        // previously-non-zero edge.
        let (mut tree, r) = fresh();
        let mut out = StepOutput::default();
        add_watch_demand(&mut tree, r, &mut out);
        add_watch_demand(&mut tree, r, &mut out);
        add_watch_demand(&mut tree, r, &mut out);
        assert_eq!(tree.get(r).unwrap().watch_demand, 3);
        out.watch_ops.clear();

        clamp_watch_demand_to_zero(&mut tree, r, &mut out);

        let res = tree.get(r).unwrap();
        assert_eq!(res.watch_demand, 0);
        assert_eq!(res.suppress_count, 0);
        assert_eq!(res.kind, ResourceKind::Unknown);
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
    fn clamp_watch_demand_to_zero_zeros_suppress_too() {
        // The Sensor has no live FD after the rejection, so suppression
        // is meaningless — both refcounts go to zero.
        let (mut tree, r) = fresh();
        let mut out = StepOutput::default();
        add_watch_demand(&mut tree, r, &mut out);
        add_suppress(&mut tree, r, &mut out);
        add_suppress(&mut tree, r, &mut out);
        assert_eq!(tree.get(r).unwrap().suppress_count, 2);
        out.watch_ops.clear();

        clamp_watch_demand_to_zero(&mut tree, r, &mut out);

        assert_eq!(tree.get(r).unwrap().suppress_count, 0);
        // Only Unwatch is emitted — no Unsuppress (the clamp drops without
        // rebalancing the suppress counter at the Sensor; v1 accepts).
        let unwatch_count = out
            .watch_ops
            .iter()
            .filter(|op| matches!(op, WatchOp::Unwatch { .. }))
            .count();
        assert_eq!(unwatch_count, 1);
    }
}
