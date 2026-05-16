//! Promoter-claim release helpers.
//!
//! A Promoter holds at most one Resource-side contribution per slot,
//! gated on `PromoterState`:
//!
//! 1. **Descent prefix.** `Promoter.state == PrefixPending(d)` ⇒ the
//!    Promoter contributes [`ContribKey::PromoterPrefix`] with
//!    `STRUCTURE` at `d.current_prefix()`.
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
//! - **Safe in any post-vacate state.**
//!   [`crate::refcounts::sub_watch`] silently skips an absent key (the
//!   map's [`ContribKey::PromoterPrefix`] /
//!   [`ContribKey::PromoterProxy`] entry has already been cleared by a
//!   prior path).
//! - **Cancel-first.** Callers that may have an in-flight probe MUST
//!   invoke [`Engine::cancel_owner_probe`] first (idempotent — a no-op
//!   on an already-disarmed slot, so "always cancel before release" is
//!   the safe default). The two release paths enforce this differently
//!   because they differ structurally:
//!   - **Descent prefix.** `release_promoter_descent_prefix_claim`'s
//!     `PrefixPending → Active{empty}` flip *drops* the prior
//!     `PrefixPending(DescentState)`. An armed descent slot reaching
//!     that drop trips `ProbeSlot`'s Drop tripwire in every build —
//!     the discard *is* the enforcement; no local witness is kept.
//!   - **Active proxy.** `release_promoter_proxy_claim` removes one
//!     proxy but keeps the Promoter `Active`, so the `enumerating`
//!     slot is *not* dropped — the Drop tripwire cannot see this. A
//!     `debug_assert!` is retained there: it guards a distinct
//!     invariant (the in-flight enumeration must not target the proxy
//!     being torn down), which the linear-slot guard does not cover.

use crate::Engine;
use crate::refcounts::sub_watch_then_try_reap;
use specter_core::{ContribKey, ProbeSlot, PromoterId, PromoterState, ResourceId, StepOutput};
use std::collections::BTreeMap;

impl Engine {
    /// Release the Promoter's literal-prefix
    /// [`ContribKey::PromoterPrefix`] contribution if `PrefixPending`.
    /// Transitions the Promoter to `Active{empty}`. Idempotent
    /// (non-`PrefixPending` ⇒ no-op); safe in any post-vacate state
    /// — `sub_watch` silently skips an absent key.
    /// Calls `try_reap` on the prefix slot — with this Promoter's
    /// prefix contribution just removed, the slot reaps unless another
    /// claim still holds it (a child Promoter / Profile slot below the
    /// prefix, a peer descent at the same level, or a sibling-anchored
    /// User Profile that promoted the slot earlier). The role tag is
    /// metadata; it does not affect this reap.
    ///
    /// **Cancel-first contract.** Callers with a possibly-in-flight
    /// descent probe (e.g., `on_watch_op_rejected`'s descent purge,
    /// `dispatch_descent_vanished`'s no-rewind arm via
    /// `release_owner_descent_prefix`) MUST invoke
    /// [`Engine::cancel_owner_probe`] first. `ProbeSlot`'s Drop tripwire
    /// enforces this structurally: the `PrefixPending → Active{empty}`
    /// flip below drops the prior `PrefixPending(DescentState)`, and an
    /// armed descent slot reaching that drop panics in every build. The
    /// empty-remaining arm in [`Engine::dispatch_descent_ok`] reaches us
    /// with the slot already disarmed because `on_promoter_probe_response`
    /// consumes it before dispatch.
    pub(crate) fn release_promoter_descent_prefix_claim(
        &mut self,
        qid: PromoterId,
        out: &mut StepOutput,
    ) {
        let Some(prefix) = self.promoters.get(qid).and_then(|q| match &q.state {
            PromoterState::PrefixPending(d) => Some(d.current_prefix()),
            PromoterState::Active { .. } => None,
        }) else {
            return;
        };

        // State flip is owner-bookkeeping: the contribution map's
        // [`ContribKey::PromoterPrefix`] key is removed below by
        // explicit key, independent of state. The cancel-first contract
        // is enforced here: this flip drops the prior
        // `PrefixPending(DescentState)`; an armed descent slot trips
        // `ProbeSlot`'s Drop tripwire.
        if let Some(q) = self.promoters.get_mut(qid) {
            q.state = PromoterState::Active {
                proxies: BTreeMap::new(),
                enumerating: ProbeSlot::empty(),
            };
        }

        sub_watch_then_try_reap(&mut self.tree, prefix, ContribKey::PromoterPrefix(qid), out);
    }

    /// Release the Promoter's `Active` proxy claim at `resource`.
    /// Idempotent (non-`Active` or proxy-not-present ⇒ no-op); safe in
    /// any counter state. Clears the per-Resource back-ref. Calls
    /// `try_reap` on the proxy slot — with the back-ref cleared and
    /// the proxy contribution released, `has_anchors` returns false for
    /// promoter-only slots; shared slots (Profile descent prefix, other
    /// Promoter proxies, an anchored User Profile, surviving children)
    /// survive. The role tag is metadata; it does not affect this reap.
    ///
    /// **Cancel-first contract for in-flight enumeration probes
    /// targeting this proxy.** Callers MUST cancel the probe first;
    /// the `debug_assert!` below pins the contract by reading the
    /// `Active` enumeration slot's tag via
    /// [`specter_core::PromoterState::enumeration_target`] — a tag
    /// equal to `resource` means an enumeration is in flight for *this*
    /// proxy. An empty slot, or a tag pointing at any sibling proxy of
    /// the same Promoter, means the probe is unaffected by our release
    /// and stays in flight.
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
            PromoterState::Active { proxies, .. } => proxies.contains_key(&resource),
            PromoterState::PrefixPending(_) => false,
        });
        if !active_with_proxy {
            return;
        }

        debug_assert!(
            self.promoters
                .get(qid)
                .and_then(|q| q.state.enumeration_target())
                != Some(resource),
            "release_promoter_proxy_claim: in-flight enumeration targets this proxy; \
             caller must invoke cancel_owner_probe before release \
             (promoter = {qid:?}, proxy = {resource:?})",
        );

        // 1. Clear map + queue entry. Owner-bookkeeping only; the
        // contribution map's [`ContribKey::PromoterProxy`] key is the
        // refcount source of truth and is removed below.
        if let Some(q) = self.promoters.get_mut(qid) {
            if let PromoterState::Active { proxies, .. } = &mut q.state {
                proxies.remove(&resource);
            }
            q.pending_enumerations.remove(&resource);
        }

        // 2. Clear back-ref before the release+reap helper so the
        // helper's `try_reap` sees `has_anchors() == false` for
        // promoter-only slots. `remove_proxy_promoter` leaves
        // co-resident Promoters' entries undisturbed (filter, not
        // clear); reordering to here from after `sub_watch` is safe
        // because `sub_watch` only reads / writes `contributions`.
        if let Some(res) = self.tree.get_mut(resource) {
            res.remove_proxy_promoter(qid);
        }

        // 3. Release the [`ContribKey::PromoterProxy`] contribution
        // and try-reap. With the back-ref cleared (above) and the
        // proxy contribution released here, `has_anchors` returns
        // false for promoter-only slots — they reap (role is metadata
        // and never gates retention). Slots shared with a Profile
        // descent / anchor or another Promoter's proxy stay.
        sub_watch_then_try_reap(
            &mut self.tree,
            resource,
            ContribKey::PromoterProxy(qid),
            out,
        );
    }
}
