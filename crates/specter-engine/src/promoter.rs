//! Promoter lifecycle and dispatch.
//!
//! A `Promoter` (peer to `Profile`) owns a `PatternSpec`, watches the
//! literal-prefix path until it materialises (`PrefixPending`), then
//! enumerates per-pattern proxies (`Active`) and synthesises dynamic Subs
//! for every match against the pattern's variable segments.
//!
//! The state machine is:
//!
//! ```text
//!   Idle (no slot)
//!         │ attach_promoter
//!         ▼
//!   PrefixPending(d) ──── descent ────▶ Active { proxies }
//!                          ▲
//!                          │ rewind on Vanished
//! ```
//!
//! Single-slot probe per Promoter: the descent probe and the proxy
//! enumeration probe share `Promoter.pending_probe`. Concurrent
//! enumerations queue via `pending_enumerations`; `dispatch_next_enumeration`
//! pops one target at a time.
//!
//! Two response dispatchers route on state:
//! - `PrefixPending` → descent. The dispatcher arms route to the
//!   owner-polymorphic [`Engine::dispatch_descent_ok`] /
//!   [`Engine::dispatch_descent_vanished`] /
//!   [`Engine::dispatch_descent_failed`] in `descent.rs`; on completion
//!   of the last literal segment, [`Engine::enter_active`] (the
//!   Promoter-side terminal-arm helper) flips the state and registers
//!   the first proxy.
//! - `Active` → enumeration (`dispatch_promoter_enumeration_*`); each
//!   response either registers sub-proxies (intermediate components),
//!   mints dynamic Subs (final component), or unregisters proxies that
//!   no longer correspond to a directory entry.

use crate::Engine;
use crate::descent::{MaterializeResult, kind_from_entry};
use crate::engine::decompose_attach_path;
use crate::refcounts::{add_watch, sub_watch_then_try_reap};
use compact_str::CompactString;
use specter_core::{
    ClassSet, ContribKey, DescentState, Diagnostic, DirSnapshot, EntryKind, PatternComponent,
    PatternSpec, ProbeOutcome, ProbeOwner, ProbeResponse, Promoter, PromoterAttachRequest,
    PromoterId, PromoterState, ProxyState, ResourceId, ResourceRole, StepOutput, SubAttachRequest,
    SubId,
};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

/// Threshold beyond which the engine emits a one-shot
/// [`Diagnostic::PromoterFanoutThreshold`] for a Promoter. Operator
/// signal that the pattern is matching more targets than typical —
/// likely a too-broad pattern. The latch on
/// `Promoter.warned_at_threshold` suppresses repeats so a steady-state
/// busy Promoter only warns once per lifetime.
pub(crate) const FANOUT_WARNING_THRESHOLD: usize = 1000;

impl Engine {
    /// Attach a Promoter to the engine. Materialises the literal-prefix
    /// path on the Tree (creating scaffolds where the prefix doesn't
    /// yet exist on disk), opens the probe channel, and returns the
    /// minted [`PromoterId`] alongside a sorted [`StepOutput`].
    ///
    /// **Two materialisation paths** branched inside
    /// `attach_promoter_inner`:
    ///
    /// - **Immediate `Active`** — the literal-prefix path resolved to a
    ///   live Tree slot. The Promoter is constructed with empty
    ///   `Active { proxies: {} }`; `enter_active` then registers the
    ///   first proxy at the prefix and queues the initial enumeration.
    /// - **`PrefixPending`** — the literal prefix doesn't yet exist.
    ///   The Promoter is constructed with `PrefixPending(d)`; the
    ///   prefix's STRUCTURE `watch_demand` bumps and a descent probe
    ///   emits at `d.current_prefix`. The descent dispatcher walks the
    ///   prefix segment-by-segment as each materialises, ending in a
    ///   single `enter_active` call when the last literal resolves.
    ///
    /// On a malformed `pattern_spec` (defense-in-depth — `PatternSpec`
    /// parse should have rejected the same shape upstream), the engine
    /// returns `PromoterId::default()` and emits
    /// [`Diagnostic::AttachPathInvalid`] without registering anything.
    pub fn attach_promoter(
        &mut self,
        req: PromoterAttachRequest,
        now: Instant,
    ) -> (PromoterId, StepOutput) {
        let mut out = StepOutput::default();
        let pid = self.attach_promoter_inner(req, now, &mut out);
        out.sort_for_emission();
        (pid, out)
    }

    /// Inner attach used by `on_config_diff` to compose multiple
    /// detach/attach operations into a single (sorted) `StepOutput`.
    ///
    /// Compute-then-insert: the materialisation outcome decides the
    /// initial `PromoterState` shape *before* the registry insert, so
    /// the Promoter is registered with its final state and never
    /// observed in a transient placeholder shape (no `Active{empty}`
    /// stand-in for a `PrefixPending` Promoter that downstream
    /// owner-state readers could see mid-mutation).
    #[allow(clippy::needless_pass_by_value)]
    pub(crate) fn attach_promoter_inner(
        &mut self,
        req: PromoterAttachRequest,
        now: Instant,
        out: &mut StepOutput,
    ) -> PromoterId {
        // 1. Render the literal prefix. components[0..literal_prefix_len]
        // are all Literal post-parse; the loop is a fold.
        let prefix_path = render_literal_prefix(&req.pattern_spec);

        // 2. Decompose. Defense-in-depth: PatternSpec::parse should have
        // rejected anything decompose_attach_path would reject. The
        // None arm emits Diagnostic::AttachPathInvalid; we surface a
        // sentinel id mirroring `attach_sub_inner`.
        let Some(comps) = decompose_attach_path(&prefix_path, out) else {
            return PromoterId::default();
        };

        // 3. Compute materialise BEFORE insert. Pick the final state
        // shape; insert once.
        let materialize = self.materialize_path_or_pending(&comps);

        // 4. Capture the literal_prefix_len; first proxy at materialisation
        // carries this index in `pattern.components`.
        let lpl = req.pattern_spec.literal_prefix_len();

        // 5. Construct the initial state directly. No placeholder.
        let initial_state = match &materialize {
            MaterializeResult::Materialized(_) => PromoterState::Active {
                proxies: BTreeMap::new(),
            },
            MaterializeResult::Pending {
                prefix, remaining, ..
            } => PromoterState::PrefixPending(DescentState {
                current_prefix: *prefix,
                remaining_components: remaining.clone(),
            }),
        };

        // 6. Mint the Promoter with the final state. `insert_with_key`
        // closure embeds the freshly-minted id into the value.
        let promoter_id = self.promoters.insert(|id| Promoter {
            id,
            name: CompactString::from(req.name.as_str()),
            pattern: req.pattern_spec.clone(),
            config: req.config.clone(),
            max_settle: req.max_settle,
            settle: req.settle,
            program: Arc::clone(&req.program),
            scope: req.scope,
            events: req.events,
            log_output: req.log_output,
            state: initial_state,
            pending_probe: None,
            pending_enumeration_target: None,
            pending_enumerations: BTreeSet::new(),
            dynamic_subs: BTreeMap::new(),
            warned_at_threshold: false,
        });

        // 7. Lifecycle diagnostic. Emitted before any operation that
        // could early-return so the bin's `name → PromoterId` map sees
        // the registration deterministically.
        out.diagnostics.push(Diagnostic::PromoterAttached {
            promoter: promoter_id,
            name: CompactString::from(req.name.as_str()),
        });

        // 8. Branch on materialise outcome to set up watches/probes.
        match materialize {
            MaterializeResult::Materialized(prefix_resource) => {
                // Single helper: enter_active demotes the slot's role
                // (DescentScaffold → User) where applicable, registers
                // the first proxy, and dispatches the initial
                // enumeration. No prior watch_demand contribution to
                // release on this path.
                self.enter_active(
                    promoter_id,
                    /* prior_prefix_to_release */ None,
                    /* new_proxy_resource */ prefix_resource,
                    /* pattern_component_index */ lpl,
                    now,
                    out,
                );
            }
            MaterializeResult::Pending { prefix, .. } => {
                // PrefixPending: install the
                // [`ContribKey::PromoterPrefix`] STRUCTURE
                // contribution and emit the descent probe.
                add_watch(
                    &mut self.tree,
                    prefix,
                    ContribKey::PromoterPrefix(promoter_id),
                    ClassSet::STRUCTURE,
                    out,
                );
                let owner = ProbeOwner::Promoter(promoter_id);
                let Some(correlation) = self.mint_owner_correlation(owner) else {
                    return promoter_id;
                };
                let target_path = self.tree.path_of(prefix).unwrap_or_default();
                Self::emit_descent_probe(owner, correlation, prefix, target_path, out);
            }
        }

        promoter_id
    }

    /// Single helper for both Promoter materialisation paths:
    ///
    /// - **immediate-Materialized** (from `attach_promoter_inner`): no
    ///   prior prefix to release; first proxy installed at the prefix
    ///   slot.
    /// - **`PrefixPending → Active`** (from
    ///   [`Engine::dispatch_descent_ok`]'s last-literal arm for the
    ///   Promoter owner): prior prefix's STRUCTURE contribution
    ///   releases; new proxy installed at the freshly-materialised slot.
    ///
    /// Pre-conditions:
    /// - `promoter_id` exists in the registry (caller just inserted it
    ///   or read it back from registry).
    /// - State is either fresh `Active{empty}` (immediate-Materialized
    ///   arm) or `PrefixPending(_)` (descent-completion arm — caller
    ///   must NOT have flipped state before this helper).
    /// - `new_proxy_resource` is a live Tree slot.
    /// - `pattern_component_index == pattern.literal_prefix_len`
    ///   (the first proxy at materialisation time).
    ///
    /// Post-conditions:
    /// - State is `Active { proxies: { new_proxy_resource → ProxyState{lpl} } }`.
    /// - `prior_prefix_to_release` (if Some) had its STRUCTURE
    ///   contribution released (counter -1; recompute walks
    ///   post-state-flip Promoter contribution).
    /// - `new_proxy_resource` carries +1 STRUCTURE watch_demand.
    /// - `pending_enumerations` contains `{new_proxy_resource}`.
    /// - `dispatch_next_enumeration` has been called (probe emitted iff
    ///   no prior probe in flight, which is structurally guaranteed at
    ///   this point — the descent probe just closed, the immediate
    ///   arm has never opened a channel).
    pub(crate) fn enter_active(
        &mut self,
        promoter_id: PromoterId,
        prior_prefix_to_release: Option<ResourceId>,
        new_proxy_resource: ResourceId,
        pattern_component_index: usize,
        now: Instant,
        out: &mut StepOutput,
    ) {
        // [S-8] Promoter-fresh slots demote to `User` role for
        // diagnostic clarity: the proxy is functionally a User
        // anchorage for the dynamic Sub family this Promoter is about
        // to mint. The role is metadata only — retention runs through
        // the `proxy_promoters` back-ref and the proxy contribution,
        // independent of role. The demotion is informational; preserve
        // it so observers (logs, future tracing) read the slot's role
        // as its current functional purpose rather than its descent
        // origin.
        if let Some(res) = self.tree.get(new_proxy_resource)
            && matches!(res.role, ResourceRole::DescentScaffold)
        {
            self.tree.set_role(new_proxy_resource, ResourceRole::User);
        }

        // 1. Flip state to `Active { proxies: empty }` BEFORE any
        // refcount work. Owner-bookkeeping: the contribution-map
        // entry for `prior_prefix_to_release` is keyed by
        // [`ContribKey::PromoterPrefix`] and is removed by explicit
        // key below; the state-flip aligns
        // [`PromoterState`] readers with the post-release shape.
        if let Some(q) = self.promoters.get_mut(promoter_id) {
            q.state = PromoterState::Active {
                proxies: BTreeMap::new(),
            };
        }

        // 2. Release the prior prefix's STRUCTURE contribution if any.
        // The contribution map's key was [`ContribKey::PromoterPrefix`]
        // — the post-state-flip Active{empty} no longer holds this
        // claim, but the contribution map is the source of truth and
        // is removed by key here. `try_reap` is idempotent — the slot
        // survives iff something else still holds it (children,
        // profiles, co-resident contributions, proxy_promoters
        // back-refs); role is metadata and never gates retention.
        if let Some(prior) = prior_prefix_to_release {
            sub_watch_then_try_reap(
                &mut self.tree,
                prior,
                ContribKey::PromoterPrefix(promoter_id),
                out,
            );
        }

        // 3. Register the proxy at new_proxy_resource. `register_proxy`
        // inserts into proxies map, queues enumeration (gated on
        // !already_carries per [H-5]), bumps watch_demand, and sets
        // the back-ref.
        self.register_proxy(
            promoter_id,
            new_proxy_resource,
            pattern_component_index,
            out,
        );

        // 4. Drain initial enumeration (single-slot: no probe in
        // flight here, so this dispatches immediately).
        self.dispatch_next_enumeration(promoter_id, now, out);
    }

    /// Register a proxy at `resource` for `promoter_id`.
    ///
    /// Pre-condition: state is `Active { .. }`. Sole production callers
    /// (`enter_active`, `dispatch_promoter_enumeration_ok`'s forward
    /// pass) guarantee `Active` by construction. The `PrefixPending`
    /// arms below are `unreachable!()` to surface caller bugs loudly
    /// in both dev and release rather than silently no-op.
    ///
    /// [H-5] `pending_enumerations.insert(R)` is gated on
    /// `!already_carries`. Re-registration of a proxy already in the
    /// map is a no-op on both the watch_demand counter and the
    /// enumeration queue — the back-ref is also idempotent via the
    /// `contains` check.
    pub(crate) fn register_proxy(
        &mut self,
        promoter_id: PromoterId,
        resource: ResourceId,
        pattern_component_index: usize,
        out: &mut StepOutput,
    ) {
        let already_carries = self
            .promoters
            .get(promoter_id)
            .is_some_and(|q| match &q.state {
                PromoterState::Active { proxies } => proxies.contains_key(&resource),
                PromoterState::PrefixPending(_) => unreachable!(
                    "register_proxy: state must be Active (promoter = {promoter_id:?}); \
                     enter_active and dispatch_promoter_enumeration_ok are the sole callers \
                     and both ensure Active before invocation"
                ),
            });

        if let Some(q) = self.promoters.get_mut(promoter_id) {
            match &mut q.state {
                PromoterState::Active { proxies } => {
                    proxies.insert(
                        resource,
                        ProxyState {
                            pattern_component_index,
                        },
                    );
                }
                PromoterState::PrefixPending(_) => unreachable!(
                    "register_proxy: state must be Active (promoter = {promoter_id:?})"
                ),
            }
            // Only enqueue when NEWLY registered. Re-registration is
            // structurally idempotent — both on the contribution map
            // (`already_carries` skips `add_watch` below) and on the
            // queue (this gate).
            if !already_carries {
                q.pending_enumerations.insert(resource);
            }
        }

        // Back-ref: keep `Resource.proxy_promoters` in lockstep with
        // `Promoter.proxies`. The contains check handles cross-Promoter
        // sharing on the same slot — typical case is no overlap, so
        // the SmallVec inline cap of 1 absorbs.
        if let Some(r) = self.tree.get_mut(resource)
            && !r.proxy_promoters.contains(&promoter_id)
        {
            r.proxy_promoters.push(promoter_id);
        }

        if !already_carries {
            add_watch(
                &mut self.tree,
                resource,
                ContribKey::PromoterProxy(promoter_id),
                ClassSet::STRUCTURE,
                out,
            );
        }
    }

    /// Unregister a single proxy at `resource` for `promoter_id`. Thin
    /// wrapper over [`Self::release_promoter_proxy_claim`] preserved for
    /// the existing call-site shape.
    ///
    /// Inverse of [`Self::register_proxy`]: clears the proxies map
    /// entry FIRST (I-Promoter-Proxy-Reap), drops the +1 STRUCTURE
    /// contribution (counter-aware), clears the back-ref, and
    /// `try_reap`s the slot. `pending_enumerations.remove(&r)` is also
    /// cleared so a queued enumeration for this proxy doesn't resurrect
    /// after reap.
    ///
    /// With [S-8] (`User` role) and the back-ref cleared, `has_anchors`
    /// returns false for promoter-only slots — they reap. Slots shared
    /// with a Profile descent / anchor or another Promoter's proxy
    /// stay.
    ///
    /// **Cancel-first contract.** Inherited from
    /// [`Self::release_promoter_proxy_claim`]: callers with an
    /// in-flight enumeration probe targeting `resource` MUST invoke
    /// [`Self::cancel_owner_probe`] first. Existing call sites
    /// ([`Self::unregister_proxy_subtree`] from
    /// `dispatch_promoter_enumeration_*` and
    /// [`Self::reap_promoter_inner`]) all reach the helper with the
    /// channel closed.
    pub(crate) fn unregister_proxy(
        &mut self,
        promoter_id: PromoterId,
        resource: ResourceId,
        out: &mut StepOutput,
    ) {
        self.release_promoter_proxy_claim(promoter_id, resource, out);
    }

    /// Unregister `r` and any descendant proxies of this Promoter
    /// rooted at or below `r`. Called from
    /// [`Self::dispatch_promoter_enumeration_ok`]'s reverse pass when
    /// a parent enumeration observes that a previously-registered
    /// proxy's directory is gone.
    ///
    /// Dynamic Subs whose anchor is at or below `r` are NOT cleaned up
    /// here — they reap via their own anchor-terminal events through the
    /// recovery-split path. The decoupling preserves the contract that
    /// only enumeration adds, only anchor-terminal removes.
    pub(crate) fn unregister_proxy_subtree(
        &mut self,
        promoter_id: PromoterId,
        r: ResourceId,
        out: &mut StepOutput,
    ) {
        // Collect the set of proxies to unregister (the proxy at `r`
        // plus any descendant proxies of this Promoter). Snapshot
        // under a read borrow so the per-proxy unregister loop below
        // can take `&mut self`.
        let to_unregister: Vec<ResourceId> = self
            .promoters
            .get(promoter_id)
            .map(|q| match &q.state {
                PromoterState::Active { proxies } => proxies
                    .keys()
                    .copied()
                    .filter(|&p| p == r || self.tree.ancestors(p).any(|a| a == r))
                    .collect(),
                PromoterState::PrefixPending(_) => Vec::new(),
            })
            .unwrap_or_default();

        for proxy in to_unregister {
            self.unregister_proxy(promoter_id, proxy, out);
        }
    }

    /// Promoter-side probe response handler. Closed under the Promoter
    /// owner kind: I5 staleness check, channel close, and dispatch by
    /// state. Sibling owner kinds (Profile) route through their own
    /// `on_*_probe_response` from [`Self::on_probe_response`].
    pub(crate) fn on_promoter_probe_response(
        &mut self,
        promoter_id: PromoterId,
        response: ProbeResponse,
        now: Instant,
        out: &mut StepOutput,
    ) {
        let owner = response.owner;
        let received = response.correlation;

        // I5 staleness check + capture `pending_enumeration_target`
        // under one read borrow. The captured target identifies the
        // proxy a `Vanished` / `Failed` enumeration response refers to
        // — those outcomes carry no payload, and `pending_enumerations`
        // no longer holds the target after `pop_first` consumed it at
        // probe-emit time. `None` while a descent probe is in flight
        // (descent reads target from `DescentState`).
        let (is_live, current_target) = match self.promoters.get(promoter_id) {
            Some(q) => (
                q.pending_probe == Some(received),
                q.pending_enumeration_target,
            ),
            None => (false, None),
        };
        if !is_live {
            out.diagnostics.push(Diagnostic::StaleProbeResponse {
                owner,
                correlation: received,
            });
            return;
        }

        // Close the channel BEFORE dispatching. Dispatch arms may
        // re-open a fresh channel (descent advance, post-enumeration
        // drain); they MUST see a closed channel on entry, otherwise
        // the I5 debug_assert in `mint_owner_correlation` fires.
        // `close_probe_channel` clears `pending_probe` and
        // `pending_enumeration_target` in lockstep — the lockstep
        // contract lives on the Promoter type so two separate writes
        // can't quietly drift.
        if let Some(q) = self.promoters.get_mut(promoter_id) {
            q.close_probe_channel();
        }

        // Route on state. PrefixPending → descent; Active → enumerate.
        let dispatch = self.promoters.get(promoter_id).map(|q| match &q.state {
            PromoterState::PrefixPending(_) => PromoterDispatch::Descent,
            PromoterState::Active { .. } => PromoterDispatch::Enumerate,
        });

        match (dispatch, response.outcome) {
            (Some(PromoterDispatch::Descent), ProbeOutcome::SubtreeOk(arc)) => {
                self.dispatch_descent_ok(ProbeOwner::Promoter(promoter_id), &arc, now, out);
            }
            (Some(PromoterDispatch::Descent), ProbeOutcome::Vanished) => {
                self.dispatch_descent_vanished(ProbeOwner::Promoter(promoter_id), now, out);
            }
            (Some(PromoterDispatch::Descent), ProbeOutcome::Failed { errno }) => {
                self.dispatch_descent_failed(ProbeOwner::Promoter(promoter_id), errno, out);
            }
            (Some(PromoterDispatch::Descent), ProbeOutcome::AnchorOk(_)) => {
                // Walker-contract violation: descent always probes a
                // Dir prefix, walker must return SubtreeOk or
                // Vanished.
                debug_assert!(
                    false,
                    "walker contract violated: Promoter descent received AnchorOk \
                     (promoter = {promoter_id:?})",
                );
                out.diagnostics.push(Diagnostic::StaleProbeResponse {
                    owner,
                    correlation: received,
                });
            }
            (Some(PromoterDispatch::Enumerate), ProbeOutcome::SubtreeOk(arc)) => {
                self.dispatch_promoter_enumeration_ok(promoter_id, &arc, now, out);
            }
            (Some(PromoterDispatch::Enumerate), ProbeOutcome::Vanished) => {
                self.dispatch_promoter_enumeration_vanished(promoter_id, current_target, out);
            }
            (Some(PromoterDispatch::Enumerate), ProbeOutcome::Failed { errno }) => {
                self.dispatch_promoter_enumeration_failed(promoter_id, current_target, errno, out);
            }
            (Some(PromoterDispatch::Enumerate), ProbeOutcome::AnchorOk(_)) => {
                debug_assert!(
                    false,
                    "walker contract violated: Promoter enumeration received AnchorOk \
                     (promoter = {promoter_id:?})",
                );
                out.diagnostics.push(Diagnostic::StaleProbeResponse {
                    owner,
                    correlation: received,
                });
            }
            (None, _) => {
                // Promoter reaped between the probe submit and
                // response. Defense-in-depth — `is_live` above already
                // excludes the post-reap case (registry get fails),
                // but the match needs an arm.
                out.diagnostics.push(Diagnostic::StaleProbeResponse {
                    owner,
                    correlation: received,
                });
            }
        }

        // Drain the next queued enumeration (if any). No-op if a probe
        // is in flight (descent advance reopened the slot) or the
        // queue is empty.
        self.dispatch_next_enumeration(promoter_id, now, out);
    }

    /// Drain one queued enumeration target into a probe. No-op if a
    /// probe is already in flight (single-slot discipline) or the
    /// queue is empty.
    ///
    /// Records the popped target on `Promoter.pending_enumeration_target`
    /// in lockstep with `pending_probe` — `Vanished` / `Failed`
    /// responses carry no payload, so the dispatcher reads this slot
    /// at response time to identify which proxy the response refers
    /// to. The `Ok` arm reads `snapshot.root_resource` per [C-1] and
    /// the slot's value is redundant for it; the field clears
    /// uniformly in `on_promoter_probe_response`.
    pub(crate) fn dispatch_next_enumeration(
        &mut self,
        promoter_id: PromoterId,
        _now: Instant,
        out: &mut StepOutput,
    ) {
        // At most one outstanding probe per Promoter.
        if self
            .pending_probe_for(ProbeOwner::Promoter(promoter_id))
            .is_some()
        {
            return;
        }

        // Pop the next pending enumeration target. `pop_first` is a
        // single-shot fetch+remove from the BTreeSet so the queue
        // stays in lockstep with the in-flight probe.
        let target = self
            .promoters
            .get_mut(promoter_id)
            .and_then(|q| q.pending_enumerations.pop_first());
        let Some(target) = target else {
            return;
        };

        let owner = ProbeOwner::Promoter(promoter_id);
        let Some(correlation) = self.mint_owner_correlation(owner) else {
            return;
        };

        // Lockstep with `pending_probe`: record the in-flight target
        // so `Vanished` / `Failed` responses can identify the proxy.
        if let Some(q) = self.promoters.get_mut(promoter_id) {
            q.pending_enumeration_target = Some(target);
        }

        let target_path = self.tree.path_of(target).unwrap_or_default();
        // [C-1] target_resource carried on the wire so the dispatch
        // arm can identify which proxy this response corresponds to
        // via `snapshot.root_resource`.
        Self::emit_descent_probe(owner, correlation, target, target_path, out);
    }

    /// Successful enumeration response — the walker enumerated one
    /// level of the proxy at `snapshot.root_resource` and returned its
    /// children. Two passes:
    ///
    /// - **Forward pass** (additions): walk `snapshot.entries`; for
    ///   each entry matching the proxy's pattern component, either
    ///   register a sub-proxy (intermediate position) or
    ///   `try_promote` (final position).
    /// - **Reverse pass** (removals): walk this Promoter's proxies
    ///   that are direct children of `target`; for any whose name is
    ///   no longer in the snapshot, `unregister_proxy_subtree` to
    ///   cascade-clean stale state. Dynamic Subs whose anchor is
    ///   under the unregistered subtree reap via their own
    ///   anchor-terminal events.
    pub(crate) fn dispatch_promoter_enumeration_ok(
        &mut self,
        promoter_id: PromoterId,
        snapshot: &DirSnapshot,
        _now: Instant,
        out: &mut StepOutput,
    ) {
        // [C-1] target is the walker's stamp — the proxy id we
        // probed. Reading from the snapshot lets the dispatcher work
        // without a per-Promoter "current enumeration target" slot.
        let target = snapshot.root_resource;

        // Look up this proxy's pattern_component_index.
        let proxy_state = self.promoters.get(promoter_id).and_then(|q| {
            if let PromoterState::Active { proxies } = &q.state {
                proxies.get(&target).copied()
            } else {
                None
            }
        });
        let Some(ProxyState {
            pattern_component_index,
        }) = proxy_state
        else {
            // Proxy was unregistered between submit and response
            // (e.g., a parent enumeration's reverse pass cascaded
            // through this proxy mid-flight — see edge case 18.13).
            // No harmful action; one wasted round-trip.
            return;
        };

        // Snapshot the components vec under a read borrow so the
        // forward pass below can take `&mut self`. Cloning is cheap
        // — pattern.components() is a small Vec.
        let components: Vec<PatternComponent> = self
            .promoters
            .get(promoter_id)
            .map(|q| q.pattern.components().to_vec())
            .unwrap_or_default();
        if pattern_component_index >= components.len() {
            debug_assert!(
                false,
                "proxy pattern_component_index out of range \
                 (promoter = {promoter_id:?}, idx = {pattern_component_index}, \
                  len = {})",
                components.len(),
            );
            return;
        }

        let is_final = pattern_component_index + 1 == components.len();

        // Forward pass: walk snapshot.entries, register matches.
        // We borrow `next_component` from the cloned `components`,
        // which has a lifetime `'_` tied to the `components` local.
        let next_component = &components[pattern_component_index];
        for (name, child) in &snapshot.entries {
            let name_str: &str = name.as_str();
            let matches = match next_component {
                PatternComponent::Literal(s) => name_str == s.as_str(),
                PatternComponent::Glob(g) => g.matches_path(Path::new(name_str)),
            };
            if !matches {
                continue;
            }

            let child_kind = child.kind();
            let child_is_dir = matches!(child_kind, EntryKind::Dir);

            if is_final {
                // Final: try_promote. The synthesised Sub's anchor is
                // the matched path; the Sub joins (or creates) a
                // Profile via `(resource, config_hash)` dedup.
                let promote_path = self
                    .tree
                    .path_of(target)
                    .map(|p| p.join(name_str))
                    .unwrap_or_default();
                self.try_promote(promoter_id, &promote_path, child_kind, out);
            } else {
                // Non-final: only descend into Dir matches. A literal
                // or glob matching a Leaf at a non-final position
                // can't lead anywhere; the engine drops without
                // diagnostic (the user's pattern was malformed for
                // the actual filesystem state).
                if !child_is_dir {
                    continue;
                }

                // [S-8] Use User role for promoter sub-proxy slots.
                // The back-ref (proxy_promoters) is the retention
                // signal; DescentScaffold would leak after
                // unregister.
                let child_resource =
                    self.tree.lookup(Some(target), name_str).unwrap_or_else(|| {
                        self.tree.ensure(Some(target), name_str, ResourceRole::User)
                    });
                self.tree
                    .set_kind(child_resource, kind_from_entry(child_kind));
                self.register_proxy(
                    promoter_id,
                    child_resource,
                    pattern_component_index + 1,
                    out,
                );
            }
        }

        // Reverse pass: unwind proxies whose underlying entry is gone.
        // Scope: direct children of `target` only — deeper proxies
        // cascade through `unregister_proxy_subtree`'s ancestor
        // filter when this list lands on their parent.
        let snapshot_names: BTreeSet<&str> =
            snapshot.entries.keys().map(CompactString::as_str).collect();
        let stale: Vec<ResourceId> = self
            .promoters
            .get(promoter_id)
            .map(|q| match &q.state {
                PromoterState::Active { proxies } => proxies
                    .keys()
                    .copied()
                    .filter(|&r| {
                        self.tree.parent(r) == Some(target)
                            && self
                                .tree
                                .name(r)
                                .is_some_and(|n| !snapshot_names.contains(n))
                    })
                    .collect(),
                PromoterState::PrefixPending(_) => Vec::new(),
            })
            .unwrap_or_default();
        for stale_proxy in stale {
            self.unregister_proxy_subtree(promoter_id, stale_proxy, out);
        }
    }

    /// Vanished response on a proxy enumeration. The proxy directory
    /// at `target` is gone from disk; the engine cascade-cleans the
    /// proxy and any sub-proxies under it via
    /// [`Self::unregister_proxy_subtree`]. Dynamic Subs anchored
    /// inside the unwound subtree are NOT reaped here — they reap via
    /// their own anchor-terminal events through the recovery-split
    /// path, preserving the rule that only anchor-terminal removes
    /// dynamic Subs.
    ///
    /// `target` comes from `Promoter.pending_enumeration_target`
    /// captured at the response handler's read-borrow window. A
    /// kernel-driven cascade (the proxy's parent's enumeration_ok
    /// reverse pass triggered by the parent's `StructureChanged`
    /// event when the proxy is removed) reaches the same end state;
    /// observing `Vanished` directly short-circuits that round-trip.
    ///
    /// On the rare path where `target` is `None` (lockstep invariant
    /// broken — production paths set the slot at probe-emit time),
    /// this is a defensive `debug_assert` + diagnostic-only fallback.
    pub(crate) fn dispatch_promoter_enumeration_vanished(
        &mut self,
        promoter_id: PromoterId,
        target: Option<ResourceId>,
        out: &mut StepOutput,
    ) {
        let Some(target) = target else {
            debug_assert!(
                false,
                "dispatch_promoter_enumeration_vanished: \
                 pending_enumeration_target was None at response time \
                 (promoter = {promoter_id:?})",
            );
            out.diagnostics
                .push(Diagnostic::PromoterEnumerationVanished {
                    promoter: promoter_id,
                    proxy: ResourceId::default(),
                });
            return;
        };
        out.diagnostics
            .push(Diagnostic::PromoterEnumerationVanished {
                promoter: promoter_id,
                proxy: target,
            });
        self.unregister_proxy_subtree(promoter_id, target, out);
    }

    /// Failed response on a proxy enumeration. Retains proxy state;
    /// the next kernel event at the proxy re-triggers enumeration.
    /// Failures here are typically transient (`EACCES`, `EIO`); a
    /// permanent failure leaves the proxy stalled until the
    /// underlying condition clears or the operator restarts.
    ///
    /// `target` is captured in lockstep with the probe correlation —
    /// the `None` arm is defense-in-depth for the lockstep
    /// invariant, mirroring [`Self::dispatch_promoter_enumeration_vanished`].
    #[allow(clippy::unused_self)]
    pub(crate) fn dispatch_promoter_enumeration_failed(
        &self,
        promoter_id: PromoterId,
        target: Option<ResourceId>,
        errno: i32,
        out: &mut StepOutput,
    ) {
        let proxy = target.unwrap_or_else(|| {
            debug_assert!(
                false,
                "dispatch_promoter_enumeration_failed: \
                 pending_enumeration_target was None at response time \
                 (promoter = {promoter_id:?})",
            );
            ResourceId::default()
        });
        out.diagnostics.push(Diagnostic::PromoterEnumerationFailed {
            promoter: promoter_id,
            proxy,
            errno,
        });
    }

    /// Mint a dynamic Sub at `promote_path` for `promoter_id`.
    ///
    /// Enumeration ADDS; only anchor-terminal removes. At most one
    /// dynamic Sub per `(promoter_id, resolved_path)` — the contains
    /// check below gates re-promotion of a path the engine already
    /// minted a Sub for.
    pub(crate) fn try_promote(
        &mut self,
        promoter_id: PromoterId,
        promote_path: &Path,
        observed_kind: EntryKind,
        out: &mut StepOutput,
    ) {
        // Dedup gate: one dynamic Sub per `(promoter_id, resolved_path)`.
        let already_present = self
            .promoters
            .get(promoter_id)
            .is_some_and(|q| q.dynamic_subs.contains_key(promote_path));
        if already_present {
            return;
        }

        // Capture spec fields BEFORE the &mut borrow chain on
        // attach_sub_inner. Cloning the heavy fields once (config /
        // program) is cheaper than re-borrowing the registry across
        // each access. `program` Arc-clones — refcount bump only.
        let Some((promoter_name, config, max_settle, settle, program, scope, events, log_output)) =
            self.promoters.get(promoter_id).map(|q| {
                (
                    q.name.clone(),
                    q.config.clone(),
                    q.max_settle,
                    q.settle,
                    Arc::clone(&q.program),
                    q.scope,
                    q.events,
                    q.log_output,
                )
            })
        else {
            return;
        };

        let synthesized = format!("{promoter_name}@{}", promote_path.display());

        let req = SubAttachRequest::for_dynamic(
            synthesized,
            promote_path.to_path_buf(),
            config,
            max_settle,
            settle,
            program,
            scope,
            events,
            log_output,
            promoter_id,
        );

        let sub_id = self.attach_sub_inner(req, Instant::now(), out);
        if sub_id == SubId::default() {
            // attach_sub_inner emitted AttachPathInvalid. No
            // bookkeeping — the Sub never registered.
            return;
        }

        // Register in dedup map (enumeration ADD).
        if let Some(q) = self.promoters.get_mut(promoter_id) {
            q.dynamic_subs.insert(promote_path.to_path_buf(), sub_id);
        }

        out.diagnostics.push(Diagnostic::PromotionKindObserved {
            promoter: promoter_id,
            path: promote_path.to_path_buf(),
            kind: kind_from_entry(observed_kind),
        });

        // Fanout warning. One-shot per Promoter lifetime.
        let count = self
            .promoters
            .get(promoter_id)
            .map_or(0, |q| q.dynamic_subs.len());
        let warned = self
            .promoters
            .get(promoter_id)
            .is_some_and(|q| q.warned_at_threshold);
        if count > FANOUT_WARNING_THRESHOLD && !warned {
            out.diagnostics.push(Diagnostic::PromoterFanoutThreshold {
                promoter: promoter_id,
                count,
            });
            if let Some(q) = self.promoters.get_mut(promoter_id) {
                q.warned_at_threshold = true;
            }
        }
    }

    /// Promoter-side proxy event handler. Sole entry point from
    /// [`Self::on_fs_event`]'s proxy-fan-out loop.
    ///
    /// Idempotently enqueues `resource` into `pending_enumerations`
    /// and drains one slot. Behaviour:
    /// - If the Promoter is in `Active` and has a proxy at
    ///   `resource` (the back-ref pointed at this Promoter): enqueue
    ///   + dispatch.
    /// - If the proxy was unregistered earlier in the same step (the
    ///   back-ref snapshot was taken pre-mutation): emit
    ///   [`Diagnostic::PromoterProxyStaleEvent`] and drop.
    pub(crate) fn on_promoter_proxy_event(
        &mut self,
        promoter_id: PromoterId,
        resource: ResourceId,
        now: Instant,
        out: &mut StepOutput,
    ) {
        let has_proxy = self
            .promoters
            .get(promoter_id)
            .is_some_and(|q| match &q.state {
                PromoterState::Active { proxies } => proxies.contains_key(&resource),
                PromoterState::PrefixPending(_) => false,
            });
        if !has_proxy {
            out.diagnostics.push(Diagnostic::PromoterProxyStaleEvent {
                promoter: promoter_id,
                resource,
            });
            return;
        }

        // Enqueue. BTreeSet::insert is idempotent — concurrent
        // events at the same proxy collapse to one enumeration.
        if let Some(q) = self.promoters.get_mut(promoter_id) {
            q.pending_enumerations.insert(resource);
        }
        self.dispatch_next_enumeration(promoter_id, now, out);
    }

    /// Notify a Promoter that one of its dynamic Subs has reaped (the
    /// Sub's anchor disappeared, and the all-dynamic teardown branch of
    /// [`Engine::on_anchor_terminal_event`] is unwinding the Profile).
    ///
    /// Removes the `(path → sub_id)` entry from `Promoter.dynamic_subs`
    /// and emits [`Diagnostic::DynamicSubReaped`]. This is one of three
    /// documented mutators of `dynamic_subs` (alongside
    /// [`Self::try_promote`] for inserts and
    /// [`Self::reap_promoter_inner`] for full drains).
    ///
    /// **Stale notification.** A concurrent
    /// [`Self::reap_promoter_inner`] (e.g., reload removing the
    /// Promoter in the same step) may have already cleared the
    /// dynamic_subs map, in which case the lookup-by-`sub_id` returns
    /// `None` and the call is a benign no-op (no diagnostic emitted).
    pub(crate) fn on_dynamic_sub_reaped(
        &mut self,
        promoter_id: PromoterId,
        sub_id: SubId,
        out: &mut StepOutput,
    ) {
        let removed = self.promoters.get_mut(promoter_id).and_then(|q| {
            let path = q
                .dynamic_subs
                .iter()
                .find(|&(_, sid)| *sid == sub_id)
                .map(|(p, _)| p.clone());
            path.and_then(|p| q.dynamic_subs.remove(&p).map(|_| p))
        });
        if let Some(path) = removed {
            out.diagnostics.push(Diagnostic::DynamicSubReaped {
                promoter: promoter_id,
                sub: sub_id,
                path,
            });
        }
    }

    /// Reap a Promoter by id. Cancels any in-flight probe, detaches
    /// every dynamic Sub the Promoter has minted, releases the per-Resource
    /// `watch_demand` contributions (literal-prefix in `PrefixPending`,
    /// every proxy in `Active`), and removes the Promoter from the
    /// registry.
    ///
    /// Public entry point. Sole call site outside tests is the
    /// hot-reload path (`on_config_diff`); the test surface uses it
    /// directly to exercise teardown.
    ///
    /// Stale `pid` is a silent no-op (no diagnostic) — mirrors
    /// `cancel_owner_probe` and `detach_sub_inner`'s defensive
    /// idempotence on stale ids.
    pub fn reap_promoter(&mut self, pid: PromoterId, now: Instant) -> StepOutput {
        let mut out = StepOutput::default();
        self.reap_promoter_inner(pid, now, &mut out);
        out.sort_for_emission();
        out
    }

    /// Inner reap used by `on_config_diff` to compose multiple
    /// detach/attach operations into a single (sorted) [`StepOutput`].
    ///
    /// Sequence:
    /// 1. Cancel any in-flight probe (descent or enumeration) and
    ///    clear `pending_enumeration_target` in lockstep.
    /// 2. Pre-clear `dynamic_subs` so any cascading detach paths see
    ///    an empty map (defense-in-depth — `detach_sub_inner` itself
    ///    doesn't read it). Then iterate the captured Sub ids and
    ///    detach each via [`Engine::detach_sub_inner`]; each detach
    ///    runs the standard deferred-reap-or-immediate-reap branch
    ///    on the corresponding Profile.
    /// 3. State-branch on `Promoter.state`:
    ///    - `PrefixPending`: flip state to `Active{empty}` for owner
    ///      bookkeeping, then release the prefix's
    ///      [`specter_core::ContribKey::PromoterPrefix`] contribution
    ///      and try-reap the slot via [`sub_watch_then_try_reap`].
    ///    - `Active`: snapshot the proxies and call
    ///      [`Self::unregister_proxy`] on each — which removes the
    ///      [`specter_core::ContribKey::PromoterProxy`] contribution,
    ///      clears the back-ref, and try-reaps the slot.
    /// 4. Remove the Promoter from the registry. Emit
    ///    [`Diagnostic::PromoterReaped`].
    pub(crate) fn reap_promoter_inner(
        &mut self,
        promoter_id: PromoterId,
        now: Instant,
        out: &mut StepOutput,
    ) {
        // Stale id: silent no-op. Mirrors `detach_sub_inner`'s
        // defensive shape but without the `DetachUnknownSub`-style
        // diagnostic — there is no `DetachUnknownPromoter` variant in
        // the catalog and the stale path is benign (the bin races a
        // ConfigDiff against an in-flight reap).
        if self.promoters.get(promoter_id).is_none() {
            return;
        }

        // 1. Close the probe channel. `cancel_owner_probe` clears
        // both `pending_probe` and `pending_enumeration_target` for
        // Promoter owners and emits `ProbeOp::Cancel` iff the channel
        // was open.
        self.cancel_owner_probe(ProbeOwner::Promoter(promoter_id), out);
        // Drain `pending_enumerations` for hygiene during the reap
        // window. The BTreeSet drops with the Promoter at step 4, so
        // this is defensive against any future mid-reap reader (and
        // matches the plan's explicit drain).
        if let Some(q) = self.promoters.get_mut(promoter_id) {
            q.pending_enumerations.clear();
        }

        // 2. Detach every dynamic Sub. Pre-clear `dynamic_subs` so
        // cascading paths observe an empty map; then iterate the
        // captured Sub ids and route each through `detach_sub_inner`
        // (which decrements profile refcount and reaps the underlying
        // Profile when the dynamic Sub was its last attachment).
        let sub_ids: Vec<SubId> = self
            .promoters
            .get(promoter_id)
            .map(|q| q.dynamic_subs.values().copied().collect())
            .unwrap_or_default();
        if let Some(q) = self.promoters.get_mut(promoter_id) {
            q.dynamic_subs.clear();
        }
        for sub_id in sub_ids {
            self.detach_sub_inner(sub_id, now, out);
        }

        // 3. State-branch on the per-Resource cleanup.
        let state_kind = self.promoters.get(promoter_id).map(|q| match &q.state {
            PromoterState::PrefixPending(_) => PromoterReapStateKind::PrefixPending,
            PromoterState::Active { .. } => PromoterReapStateKind::Active,
        });
        match state_kind {
            Some(PromoterReapStateKind::PrefixPending) => {
                // Capture the prefix BEFORE flipping state. The
                // contribution map's [`ContribKey::PromoterPrefix`]
                // key is removed below; the state-flip is for owner
                // bookkeeping only.
                let prefix = self.promoters.get(promoter_id).and_then(|q| {
                    if let PromoterState::PrefixPending(d) = &q.state {
                        Some(d.current_prefix)
                    } else {
                        None
                    }
                });
                if let Some(q) = self.promoters.get_mut(promoter_id) {
                    q.state = PromoterState::Active {
                        proxies: BTreeMap::new(),
                    };
                }
                if let Some(prefix) = prefix {
                    sub_watch_then_try_reap(
                        &mut self.tree,
                        prefix,
                        ContribKey::PromoterPrefix(promoter_id),
                        out,
                    );
                }
            }
            Some(PromoterReapStateKind::Active) => {
                // Snapshot the proxy keys; `unregister_proxy` is
                // idempotent and clears its own back-refs,
                // watch_demand, and slot. Order doesn't matter: each
                // proxy's cleanup is self-contained.
                let proxy_list: Vec<ResourceId> = self
                    .promoters
                    .get(promoter_id)
                    .map(|q| match &q.state {
                        PromoterState::Active { proxies } => proxies.keys().copied().collect(),
                        PromoterState::PrefixPending(_) => Vec::new(),
                    })
                    .unwrap_or_default();
                for r in proxy_list {
                    self.unregister_proxy(promoter_id, r, out);
                }
            }
            None => {
                // Promoter vanished mid-reap (a detach_sub_inner
                // cascade reached `reap_promoter_inner` again — which
                // it shouldn't, but defense-in-depth). Skip the
                // per-Resource cleanup.
            }
        }

        // 4. Remove the Promoter from the registry and emit the
        // lifecycle diagnostic.
        if self.promoters.remove(promoter_id).is_some() {
            out.diagnostics.push(Diagnostic::PromoterReaped {
                promoter: promoter_id,
            });
        }
    }
}

/// Build a `PathBuf` from `spec.components()[0..spec.literal_prefix_len()]`.
/// Each component in the prefix is `Literal` post-parse; the loop is
/// a fold. Glob in the literal prefix is a parse-invariant breach
/// (`debug_assert!`).
fn render_literal_prefix(spec: &PatternSpec) -> PathBuf {
    let mut p = PathBuf::new();
    for comp in &spec.components()[0..spec.literal_prefix_len()] {
        match comp {
            PatternComponent::Literal(s) => p.push(s.as_str()),
            PatternComponent::Glob(_) => {
                debug_assert!(false, "glob in literal prefix violates parse invariant");
            }
        }
    }
    p
}

/// State-keyed dispatch tag for [`Engine::on_promoter_probe_response`].
/// Computed once per response from the Promoter's current state, then
/// matched against the response outcome to route into the descent or
/// enumeration arm.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum PromoterDispatch {
    Descent,
    Enumerate,
}

/// State-discriminant projection used by
/// [`Engine::reap_promoter_inner`]. The branch logic mutates
/// `Promoter.state` (PrefixPending → Active{empty} flip) and walks
/// proxies under separate borrows; this projection captures the
/// pre-mutation discriminant so the dispatcher doesn't hold a
/// `&PromoterState` across the mutation window.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum PromoterReapStateKind {
    PrefixPending,
    Active,
}

#[cfg(test)]
#[path = "promoter_tests.rs"]
mod tests;
