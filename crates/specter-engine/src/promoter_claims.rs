//! Promoter-claim release helpers.
//!
//! A Promoter holds at most one Resource-side claim at a time, gated on
//! `PromoterState`:
//!
//! 1. **Descent prefix.** `Promoter.state == PrefixPending(d)` ⇒ Promoter
//!    contributes `STRUCTURE` to `d.current_prefix.watch_demand` (source
//!    5a in [`crate::refcounts::recompute_resource_events`]).
//! 2. **Active proxy.** `Promoter.state == Active { proxies }` ⇒ Promoter
//!    contributes `STRUCTURE` to each `proxies.keys()` slot's
//!    `watch_demand` (source 5b).
//!
//! The two states are mutually exclusive, so a Promoter contributes via 5a
//! XOR 5b at any instant. The state-flip on `PrefixPending → Active`
//! transfers contribution attribution: clear before refcount work, the
//! recompute reads post-flip.
//!
//! This module is the Promoter-side mirror of [`crate::claims`]. The same
//! two-side synchronisation discipline applies: clear the per-Promoter
//! state field BEFORE `sub_watch_demand`, so the recompute models the
//! post-release union. Each helper is:
//!
//! - **Idempotent.** State already in the post-release shape ⇒ no-op. Safe
//!   to call from any site without first checking the claim's presence.
//! - **Safe in any counter state.** `sub_watch_demand` short-circuits
//!   silently on a zero counter (post-clamp slots, or slots reached
//!   through `Tree::vacate`'s protocol-closer); see the
//!   [`crate::refcounts`] module rustdoc. Only the state-flip persists
//!   in those degenerate paths.
//! - **Cancel-first.** A `debug_assert!` enforces the
//!   probe-channel-closed precondition. Callers that may have an
//!   in-flight probe MUST invoke [`Engine::cancel_owner_probe`]
//!   first; `cancel_owner_probe` is a no-op on a closed channel, so
//!   "always cancel before release" is the safe default.

use crate::Engine;
use crate::refcounts::sub_watch_demand;
use specter_core::{PromoterId, PromoterState, ResourceId, StepOutput};
use std::collections::BTreeMap;

impl Engine {
    /// Release the Promoter's literal-prefix `watch_demand` contribution
    /// if `PrefixPending`. Transitions the Promoter to `Active{empty}`.
    /// Idempotent (non-`PrefixPending` ⇒ no-op); safe in any counter
    /// state (a post-clamp `watch_demand == 0` short-circuits inside
    /// `sub_watch_demand`). Calls `try_reap` on the prefix slot — its
    /// `DescentScaffold` role is no longer load-bearing once no descent
    /// claims it.
    ///
    /// **Cancel-first contract.** Callers with a possibly-in-flight
    /// descent probe (e.g., `on_watch_op_rejected`'s descent purge,
    /// `dispatch_descent_vanished`'s no-rewind arm via
    /// `release_owner_descent_prefix`) MUST invoke
    /// [`Engine::cancel_owner_probe`] first; the `debug_assert!` below
    /// catches the regression. The empty-remaining arm in
    /// [`Engine::dispatch_descent_ok`] reaches us with a closed channel
    /// because `on_promoter_probe_response` closes the channel before
    /// dispatch.
    pub(crate) fn release_promoter_descent_prefix_claim(
        &mut self,
        qid: PromoterId,
        out: &mut StepOutput,
    ) {
        let Some(prefix) = self.promoters.get(qid).and_then(|q| match &q.state {
            PromoterState::PrefixPending(d) => Some(d.current_prefix),
            PromoterState::Active { .. } => None,
        }) else {
            return;
        };

        debug_assert!(
            self.promoters
                .get(qid)
                .is_some_and(|q| q.pending_probe.is_none()),
            "release_promoter_descent_prefix_claim: probe channel must be closed before release; \
             caller must invoke cancel_owner_probe (or take the response-dispatch path) \
             first to avoid losing the Cancel emission (promoter = {qid:?})",
        );

        // State flip BEFORE sub. The recompute walks Promoter contributions
        // post-flip — Active{empty} drops the 5a (PrefixPending) attribution
        // to `prefix`, leaving only co-resident contributors (Profile
        // descents, other Promoter proxies).
        if let Some(q) = self.promoters.get_mut(qid) {
            q.state = PromoterState::Active {
                proxies: BTreeMap::new(),
            };
        }

        sub_watch_demand(
            &mut self.tree,
            &self.profiles,
            &self.promoters,
            prefix,
            None,
            out,
        );

        self.tree.try_reap(prefix);
    }

    /// Release the Promoter's `Active` proxy claim at `resource`.
    /// Idempotent (non-`Active` or proxy-not-present ⇒ no-op); safe in
    /// any counter state. Clears the per-Resource back-ref. Calls
    /// `try_reap` on the proxy slot — with the back-ref cleared and
    /// the `User`-roled slot, `has_anchors` returns false for
    /// promoter-only slots; shared slots (Profile descent prefix, other
    /// Promoter proxies) survive.
    ///
    /// **Cancel-first contract for in-flight enumeration probes
    /// targeting this proxy.** Callers MUST cancel the probe first; the
    /// `debug_assert!` below pins the contract. `pending_enumeration_target`
    /// is the engine-side signal that an enumeration probe is targeting
    /// `resource`; if it points at any other proxy of the same Promoter,
    /// the probe is unaffected by our release and stays in flight.
    ///
    /// Sole production callers post-Tier-1: [`Engine::unregister_proxy`]
    /// (which delegates here), [`Engine::on_watch_op_rejected`]'s proxy
    /// purge, and [`Engine::reap_promoter_inner`] (via
    /// `unregister_proxy`).
    pub(crate) fn release_promoter_proxy_claim(
        &mut self,
        qid: PromoterId,
        resource: ResourceId,
        out: &mut StepOutput,
    ) {
        let active_with_proxy = self.promoters.get(qid).is_some_and(|q| match &q.state {
            PromoterState::Active { proxies } => proxies.contains_key(&resource),
            PromoterState::PrefixPending(_) => false,
        });
        if !active_with_proxy {
            return;
        }

        debug_assert!(
            self.promoters
                .get(qid)
                .is_some_and(|q| q.pending_enumeration_target != Some(resource)),
            "release_promoter_proxy_claim: probe channel for this proxy must be closed first; \
             caller must invoke cancel_owner_probe before release \
             (promoter = {qid:?}, proxy = {resource:?})",
        );

        // 1. Clear map + queue entry FIRST. The recompute walk on
        // `sub_watch_demand` below reads `proxies.contains_key(&r)`; we
        // need it post-clear so the walk drops 5b on this resource.
        if let Some(q) = self.promoters.get_mut(qid) {
            if let PromoterState::Active { proxies } = &mut q.state {
                proxies.remove(&resource);
            }
            q.pending_enumerations.remove(&resource);
        }

        // 2. Decrement. `sub_watch_demand` is safe in any counter state —
        // post-clamp paths land on `prev == 0` and the helper short-circuits
        // silently (the Sensor is already Unwatched for this Resource).
        sub_watch_demand(
            &mut self.tree,
            &self.profiles,
            &self.promoters,
            resource,
            None,
            out,
        );

        // 3. Clear back-ref. retain in place to avoid disturbing
        // co-resident Promoters' entries.
        if let Some(res) = self.tree.get_mut(resource) {
            res.proxy_promoters.retain(|id| *id != qid);
        }

        // 4. try_reap. With the back-ref cleared and the `User` role
        // (set by `enter_active`'s `set_role` demotion), `has_anchors`
        // returns false for promoter-only slots — they reap. Slots
        // shared with a Profile descent / anchor or another Promoter's
        // proxy stay.
        self.tree.try_reap(resource);
    }
}
