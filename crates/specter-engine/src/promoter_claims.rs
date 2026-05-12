//! Promoter-claim release helpers.
//!
//! A Promoter holds at most one Resource-side contribution per slot,
//! gated on `PromoterState`:
//!
//! 1. **Descent prefix.** `Promoter.state == PrefixPending(d)` ⇒ the
//!    Promoter contributes [`ContribKey::PromoterPrefix`] with
//!    `STRUCTURE` at `d.current_prefix`.
//! 2. **Active proxy.** `Promoter.state == Active { proxies }` ⇒ the
//!    Promoter contributes [`ContribKey::PromoterProxy`] with
//!    `STRUCTURE` at each `proxies.keys()` slot.
//!
//! The two state arms are mutually exclusive — a Promoter holds the
//! prefix XOR proxy keys at any instant. The state-flip on
//! `PrefixPending → Active` is owner-bookkeeping; the contribution
//! map's source of truth changes via explicit `add_watch` / `sub_watch`
//! calls keyed by [`ContribKey::PromoterPrefix`] /
//! [`ContribKey::PromoterProxy`].
//!
//! This module is the Promoter-side mirror of [`crate::claims`]. Each
//! helper is:
//!
//! - **Idempotent.** State already in the post-release shape ⇒ no-op. Safe
//!   to call from any site without first checking the claim's presence.
//! - **Safe in any post-clamp / post-vacate state.**
//!   [`crate::refcounts::sub_watch`] silently skips an absent key (the
//!   map's [`ContribKey::PromoterPrefix`] /
//!   [`ContribKey::PromoterProxy`] entry has already been cleared by a
//!   prior path).
//! - **Cancel-first.** A `debug_assert!` enforces the
//!   probe-channel-closed precondition. Callers that may have an
//!   in-flight probe MUST invoke [`Engine::cancel_owner_probe`]
//!   first; `cancel_owner_probe` is a no-op on a closed channel, so
//!   "always cancel before release" is the safe default.

use crate::Engine;
use crate::refcounts::sub_watch;
use specter_core::{ContribKey, PromoterId, PromoterState, ResourceId, StepOutput};
use std::collections::BTreeMap;

impl Engine {
    /// Release the Promoter's literal-prefix
    /// [`ContribKey::PromoterPrefix`] contribution if `PrefixPending`.
    /// Transitions the Promoter to `Active{empty}`. Idempotent
    /// (non-`PrefixPending` ⇒ no-op); safe in any post-clamp /
    /// post-vacate state — `sub_watch` silently skips an absent key.
    /// Calls `try_reap` on the prefix slot — its `DescentScaffold`
    /// role is no longer load-bearing once no descent claims it.
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

        // State flip is owner-bookkeeping: the contribution map's
        // [`ContribKey::PromoterPrefix`] key is removed below by
        // explicit key, independent of state.
        if let Some(q) = self.promoters.get_mut(qid) {
            q.state = PromoterState::Active {
                proxies: BTreeMap::new(),
            };
        }

        sub_watch(&mut self.tree, prefix, ContribKey::PromoterPrefix(qid), out);

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

        // 1. Clear map + queue entry. Owner-bookkeeping only; the
        // contribution map's [`ContribKey::PromoterProxy`] key is the
        // refcount source of truth and is removed below.
        if let Some(q) = self.promoters.get_mut(qid) {
            if let PromoterState::Active { proxies } = &mut q.state {
                proxies.remove(&resource);
            }
            q.pending_enumerations.remove(&resource);
        }

        // 2. Decrement by explicit key. `sub_watch` silently skips an
        // absent key — safe against post-clamp / post-vacate slots.
        sub_watch(
            &mut self.tree,
            resource,
            ContribKey::PromoterProxy(qid),
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
