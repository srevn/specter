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
//! - `PrefixPending` → descent (`dispatch_promoter_descent_*`); on
//!   completion of the last literal segment, `enter_active` flips the
//!   state and registers the first proxy.
//! - `Active` → enumeration (`dispatch_promoter_enumeration_*`); each
//!   response either registers sub-proxies (intermediate components),
//!   mints dynamic Subs (final component), or unregisters proxies that
//!   no longer correspond to a directory entry.

use crate::Engine;
use crate::descent::{MaterializeResult, kind_from_entry};
use crate::engine::decompose_attach_path;
use crate::refcounts::{add_watch_demand, sub_watch_demand};
use compact_str::CompactString;
use specter_core::{
    ClassSet, DescentState, Diagnostic, DirSnapshot, EntryKind, PatternComponent, PatternSpec,
    ProbeOutcome, ProbeOwner, ProbeResponse, Promoter, PromoterAttachRequest, PromoterId,
    PromoterState, ProxyState, ResourceId, ResourceRole, StepOutput, SubAttachRequest, SubId,
};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
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
    /// [`Self::attach_promoter_inner`]:
    ///
    /// - **Immediate `Active`** — the literal-prefix path resolved to a
    ///   live Tree slot. The Promoter is constructed with empty
    ///   `Active { proxies: {} }`; [`Self::enter_active`] then
    ///   registers the first proxy at the prefix and queues the
    ///   initial enumeration.
    /// - **`PrefixPending`** — the literal prefix doesn't yet exist.
    ///   The Promoter is constructed with `PrefixPending(d)`; the
    ///   prefix's STRUCTURE `watch_demand` bumps and a descent probe
    ///   emits at `d.current_prefix`. The descent dispatcher walks the
    ///   prefix segment-by-segment as each materialises, ending in a
    ///   single [`Self::enter_active`] call when the last literal
    ///   resolves.
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

    /// Inner attach used by `on_config_diff` (Phase 11) to compose
    /// multiple detach/attach operations into a single (sorted)
    /// `StepOutput`.
    ///
    /// Compute-then-insert: the materialisation outcome decides the
    /// initial `PromoterState` shape *before* the registry insert, so
    /// the Promoter is registered with its final state and never
    /// observed in a transient placeholder shape (no `Active{empty}`
    /// stand-in for a `PrefixPending` Promoter that the
    /// recompute_resource_events walk could see mid-mutation).
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
            command: req.command.clone(),
            scope: req.scope,
            events: req.events,
            log_output: req.log_output,
            state: initial_state,
            pending_probe: None,
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
                // PrefixPending: bump the prefix's STRUCTURE
                // contribution (source 5a in
                // recompute_resource_events) and emit the descent
                // probe.
                add_watch_demand(&mut self.tree, prefix, ClassSet::STRUCTURE, out);
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
    ///   `dispatch_promoter_descent_ok`'s last-literal arm): prior
    ///   prefix's STRUCTURE contribution releases; new proxy installed
    ///   at the freshly-materialised slot.
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
        // [S-8] Promoter-fresh slots use User role: the back-ref
        // (`proxy_promoters`) is the retention signal, not
        // DescentScaffold. For a fresh slot the descent created
        // (DescentScaffold role from `Tree::ensure(_, _, DescentScaffold)`),
        // demote to User; for a previously-User slot (shared with a
        // prior Profile / Promoter), `set_role` is a no-op when the
        // role is unchanged.
        if let Some(res) = self.tree.get(new_proxy_resource)
            && matches!(res.role, ResourceRole::DescentScaffold)
        {
            self.tree.set_role(new_proxy_resource, ResourceRole::User);
        }

        // 1. Flip state to `Active { proxies: empty }` BEFORE any
        // refcount work. The state-flip is what tells
        // recompute_resource_events to drop the 5a (PrefixPending)
        // attribution if any — the prior prefix release below sees
        // post-flip Promoter contribution attribution.
        if let Some(q) = self.promoters.get_mut(promoter_id) {
            q.state = PromoterState::Active {
                proxies: BTreeMap::new(),
            };
        }

        // 2. Release the prior prefix's STRUCTURE contribution if any.
        // Recompute walks Promoter (now Active{empty}, no contribution
        // to prior prefix) → counter -1 cleanly. try_reap is
        // idempotent — slot survives iff something else still holds
        // it (children, profiles, role anchors).
        if let Some(prior) = prior_prefix_to_release {
            sub_watch_demand(
                &mut self.tree,
                &self.profiles,
                &self.promoters,
                prior,
                ClassSet::STRUCTURE,
                None,
                out,
            );
            self.tree.try_reap(prior);
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
    /// Pre-condition: state is `Active { .. }` (caller has flipped
    /// from `PrefixPending` if applicable). The debug_assert below
    /// catches caller bugs.
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
                PromoterState::PrefixPending(_) => {
                    debug_assert!(false, "register_proxy: state must be Active");
                    false
                }
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
                PromoterState::PrefixPending(_) => return,
            }
            // [H-5] Only enqueue when NEWLY registered. Re-registration
            // is structurally idempotent — both on the counter
            // (`already_carries` skips add_watch_demand below) and on
            // the queue (this gate).
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
            add_watch_demand(&mut self.tree, resource, ClassSet::STRUCTURE, out);
        }
    }

    /// Unregister a single proxy at `resource` for `promoter_id`.
    ///
    /// Inverse of [`Self::register_proxy`]: clears the proxies map
    /// entry FIRST (I-Promoter-Proxy-Reap), drops the +1 STRUCTURE
    /// contribution, clears the back-ref, and try_reaps the slot.
    /// `pending_enumerations.remove(&r)` is also cleared so a queued
    /// enumeration for this proxy doesn't resurrect after reap.
    ///
    /// With [S-8] (User role) and the back-ref cleared, `has_anchors`
    /// returns false for promoter-only slots — they reap. Slots
    /// shared with a Profile descent / anchor or another Promoter's
    /// proxy stay.
    pub(crate) fn unregister_proxy(
        &mut self,
        promoter_id: PromoterId,
        resource: ResourceId,
        out: &mut StepOutput,
    ) {
        // 1. Clear map + queue entry FIRST. The recompute walk on the
        // sub_watch_demand below reads `proxies.contains_key(&r)`; we
        // need it post-clear so the walk drops 5b on this resource.
        if let Some(q) = self.promoters.get_mut(promoter_id) {
            if let PromoterState::Active { proxies } = &mut q.state {
                proxies.remove(&resource);
            }
            q.pending_enumerations.remove(&resource);
        }

        // 2. Sub the +1 STRUCTURE contribution.
        sub_watch_demand(
            &mut self.tree,
            &self.profiles,
            &self.promoters,
            resource,
            ClassSet::STRUCTURE,
            None,
            out,
        );

        // 3. Clear back-ref. retain in place to avoid disturbing
        // co-resident Promoters' entries. SmallVec::retain hands the
        // closure `&mut T`; deref read is sufficient.
        if let Some(res) = self.tree.get_mut(resource) {
            res.proxy_promoters.retain(|id| *id != promoter_id);
        }

        // 4. try_reap. With [S-8] (User role) + cleared back-ref,
        // has_anchors returns false for promoter-only slots — they
        // reap. Slots shared with a Profile descent / anchor stay.
        self.tree.try_reap(resource);
    }

    /// Unregister `r` and any descendant proxies of this Promoter
    /// rooted at or below `r`. Called from
    /// [`Self::dispatch_promoter_enumeration_ok`]'s reverse pass when
    /// a parent enumeration observes that a previously-registered
    /// proxy's directory is gone.
    ///
    /// Dynamic Subs whose anchor is at or below `r` are NOT cleaned up
    /// here — they reap via their own anchor-terminal events (Phase
    /// 8's recovery-split path). The decoupling preserves the
    /// I-Promoter-4 contract: only enumeration adds, only
    /// anchor-terminal removes.
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

        // I5 staleness check: live iff the slot held the received
        // correlation. Catches stale-id (post-reap), post-cancel
        // arrivals, out-of-order responses across Promoter lifetime.
        let is_live = self
            .promoters
            .get(promoter_id)
            .is_some_and(|q| q.pending_probe == Some(received));
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
        if let Some(q) = self.promoters.get_mut(promoter_id) {
            q.pending_probe = None;
        }

        // Route on state. PrefixPending → descent; Active → enumerate.
        let dispatch = self.promoters.get(promoter_id).map(|q| match &q.state {
            PromoterState::PrefixPending(_) => PromoterDispatch::Descent,
            PromoterState::Active { .. } => PromoterDispatch::Enumerate,
        });

        match (dispatch, response.outcome) {
            (Some(PromoterDispatch::Descent), ProbeOutcome::SubtreeOk(arc)) => {
                self.dispatch_promoter_descent_ok(promoter_id, &arc, now, out);
            }
            (Some(PromoterDispatch::Descent), ProbeOutcome::Vanished) => {
                self.dispatch_promoter_descent_vanished(promoter_id, out);
            }
            (Some(PromoterDispatch::Descent), ProbeOutcome::Failed { errno }) => {
                self.dispatch_promoter_descent_failed(promoter_id, errno, out);
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
                self.dispatch_promoter_enumeration_vanished(promoter_id, out);
            }
            (Some(PromoterDispatch::Enumerate), ProbeOutcome::Failed { errno }) => {
                self.dispatch_promoter_enumeration_failed(promoter_id, errno, out);
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

    /// Successful descent response — the walker enumerated one level
    /// of `current_prefix` and returned a single-level
    /// [`Arc<DirSnapshot>`]. Look up the next remaining literal
    /// segment by name; if found, materialise it as a Tree slot, then
    /// either advance descent one segment or transition to Active via
    /// [`Self::enter_active`].
    pub(crate) fn dispatch_promoter_descent_ok(
        &mut self,
        promoter_id: PromoterId,
        snapshot: &DirSnapshot,
        now: Instant,
        out: &mut StepOutput,
    ) {
        let Some(q) = self.promoters.get(promoter_id) else {
            return;
        };
        let PromoterState::PrefixPending(d) = &q.state else {
            return;
        };
        let prefix = d.current_prefix;
        let lpl = q.pattern.literal_prefix_len();

        // [C-1] Defense-in-depth: walker stamp must match the prefix
        // we requested. Tolerate ResourceId::default() for legacy
        // test fixtures that don't populate root_resource.
        debug_assert!(
            snapshot.root_resource == prefix
                || snapshot.root_resource == ResourceId::default()
                || prefix == ResourceId::default(),
            "walker stamp diverges from emitted target_resource (Promoter descent): \
             snapshot.root_resource = {:?}, descent.current_prefix = {:?}",
            snapshot.root_resource,
            prefix,
        );

        let Some(next_segment) = d.remaining_components.first().cloned() else {
            // Invariant breach: PrefixPending requires non-empty
            // remaining_components (the last literal of
            // `pattern.components[0..lpl]` is the segment that
            // triggers `enter_active`). Fall through to a defensive
            // recovery: emit the breach and rewind state to Active.
            // We can't unwind the prefix watch cleanly without state
            // visibility — the diagnostic is the operator-facing
            // signal.
            out.diagnostics
                .push(Diagnostic::PromoterDescentInvariantViolation {
                    promoter: promoter_id,
                    prefix,
                });
            return;
        };
        let is_last_literal = d.remaining_components.len() == 1;

        // Look up the next segment in the snapshot's children.
        let entry_kind = match snapshot.entries.get(next_segment.as_str()) {
            Some(child) => child.kind(),
            None => return, // Not yet present; await next event.
        };

        // Materialise the next slot. Intermediate descent slots use
        // DescentScaffold; the last-literal proxy slot demotes to User
        // inside `enter_active` per [S-8].
        let new_resource = match self.tree.lookup(Some(prefix), &next_segment) {
            Some(r) => r,
            None => self
                .tree
                .ensure(Some(prefix), &next_segment, ResourceRole::DescentScaffold),
        };
        self.tree
            .set_kind(new_resource, kind_from_entry(entry_kind));

        if is_last_literal {
            // [M-2] Single helper: enter_active releases the prior
            // prefix's STRUCTURE contribution, flips state, registers
            // the first proxy, and dispatches the initial enumeration.
            self.enter_active(
                promoter_id,
                /* prior_prefix_to_release */ Some(prefix),
                /* new_proxy_resource */ new_resource,
                /* pattern_component_index */ lpl,
                now,
                out,
            );
        } else {
            // Advance one literal segment. State stays PrefixPending.
            // Update descent state in place (saves a vec rebuild).
            // Sequencing matches the Profile-side advance: state-flip
            // BEFORE sub_watch_demand so the recompute attributes the
            // STRUCTURE contribution to the new prefix.
            let owner = ProbeOwner::Promoter(promoter_id);
            let Some(correlation) = self.mint_owner_correlation(owner) else {
                return;
            };

            if let Some(q) = self.promoters.get_mut(promoter_id)
                && let PromoterState::PrefixPending(d) = &mut q.state
            {
                d.current_prefix = new_resource;
                d.remaining_components.remove(0);
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
            add_watch_demand(&mut self.tree, new_resource, ClassSet::STRUCTURE, out);
            // The OLD prefix retains DescentScaffold role (set on its
            // own `Tree::ensure` at descent's start) so it survives the
            // sub_watch_demand. No try_reap here.

            let target_path = self.tree.path_of(new_resource).unwrap_or_default();
            Self::emit_descent_probe(owner, correlation, new_resource, target_path, out);
        }
    }

    /// Vanished response on the descent prefix. Rewind to the
    /// next-existing ancestor of `prefix`. Mirrors Profile descent's
    /// rewind path.
    ///
    /// **Bounded chain depth.** The chain auto-extends watches up the
    /// ancestor chain until it reaches a still-present ancestor (whose
    /// probe returns `Ok` and routes to the await-event arm). With
    /// FS-root bootstrap (`materialize_path_or_pending`'s
    /// unconditional ensure), every Promoter's rewind chain
    /// terminates at the FS-root slot `/` — the kernel always lstats
    /// `/` successfully on Unix, so `Vanished` from `/` is impossible
    /// in production.
    pub(crate) fn dispatch_promoter_descent_vanished(
        &mut self,
        promoter_id: PromoterId,
        out: &mut StepOutput,
    ) {
        let Some(q) = self.promoters.get(promoter_id) else {
            return;
        };
        let PromoterState::PrefixPending(d) = &q.state else {
            return;
        };
        let prefix = d.current_prefix;

        out.diagnostics.push(Diagnostic::PromoterDescentVanished {
            promoter: promoter_id,
            prefix,
        });

        let parent = self.tree.parent(prefix);
        let prefix_name = self.tree.name(prefix).map(CompactString::from);

        match parent {
            Some(parent_id) => {
                // Rewind. The vanished prefix's segment becomes the
                // *first* remaining component (we're descending into
                // it from the parent again). State-flip BEFORE
                // sub_watch_demand so the recompute attributes the
                // STRUCTURE contribution to the new prefix.
                let owner = ProbeOwner::Promoter(promoter_id);
                let Some(correlation) = self.mint_owner_correlation(owner) else {
                    return;
                };

                if let Some(q) = self.promoters.get_mut(promoter_id)
                    && let PromoterState::PrefixPending(d) = &mut q.state
                {
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
                self.tree.vacate(prefix);
                self.tree.try_reap(prefix);
                add_watch_demand(&mut self.tree, parent_id, ClassSet::STRUCTURE, out);

                let target_path = self.tree.path_of(parent_id).unwrap_or_default();
                Self::emit_descent_probe(owner, correlation, parent_id, target_path, out);
            }
            None => {
                // Root prefix vanished — no rewind target. Promoter
                // stuck PrefixPending with the prefix's STRUCTURE
                // contribution dropped. Operator recovery is required
                // to re-attach.
                sub_watch_demand(
                    &mut self.tree,
                    &self.profiles,
                    &self.promoters,
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

    /// Failed response on the descent prefix. Retain `PrefixPending`
    /// state; await next event at the prefix.
    pub(crate) fn dispatch_promoter_descent_failed(
        &self,
        promoter_id: PromoterId,
        errno: i32,
        out: &mut StepOutput,
    ) {
        let prefix = match self.promoters.get(promoter_id).map(|q| &q.state) {
            Some(PromoterState::PrefixPending(d)) => d.current_prefix,
            _ => return,
        };
        out.diagnostics.push(Diagnostic::PromoterDescentFailed {
            promoter: promoter_id,
            prefix,
            errno,
        });
        // Retain PrefixPending; await next event.
    }

    /// Drain one queued enumeration target into a probe. No-op if a
    /// probe is already in flight (single-slot discipline) or the
    /// queue is empty.
    pub(crate) fn dispatch_next_enumeration(
        &mut self,
        promoter_id: PromoterId,
        _now: Instant,
        out: &mut StepOutput,
    ) {
        // I-Promoter-1: at most one outstanding probe.
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

    /// Vanished response on a proxy enumeration. Phase 6's stub
    /// retains state and emits a diagnostic; Phase 9's mid-chain
    /// unwind extends this with the proper cascade.
    ///
    /// Why retain: without a per-Promoter "current enumeration
    /// target" slot the engine cannot identify which proxy vanished
    /// from the response alone (`Vanished` carries no payload). The
    /// proxy's parent's enumeration_ok reverse pass — triggered by
    /// the kernel's `StructureChanged` event on the parent when the
    /// proxy directory is removed — is the canonical cascade.
    ///
    /// `&self` is unused in this stub but retained on the method
    /// signature so Phase 9 can extend in place without churning
    /// every dispatch arm in [`Self::on_promoter_probe_response`].
    #[allow(clippy::unused_self)]
    pub(crate) fn dispatch_promoter_enumeration_vanished(
        &self,
        promoter_id: PromoterId,
        out: &mut StepOutput,
    ) {
        out.diagnostics.push(Diagnostic::PromoterDescentVanished {
            promoter: promoter_id,
            prefix: ResourceId::default(),
        });
    }

    /// Failed response on a proxy enumeration. Retain state; await
    /// next event at the proxy.
    ///
    /// `&self` is unused — see [`Self::dispatch_promoter_enumeration_vanished`].
    #[allow(clippy::unused_self)]
    pub(crate) fn dispatch_promoter_enumeration_failed(
        &self,
        promoter_id: PromoterId,
        errno: i32,
        out: &mut StepOutput,
    ) {
        out.diagnostics.push(Diagnostic::PromoterDescentFailed {
            promoter: promoter_id,
            prefix: ResourceId::default(),
            errno,
        });
    }

    /// Mint a dynamic Sub at `promote_path` for `promoter_id`.
    ///
    /// I-Promoter-4: enumeration ADDS; only anchor-terminal removes.
    /// I-Promoter-5: at most one dynamic Sub per
    /// `(promoter_id, resolved_path)` — the contains check below
    /// gates re-promotion of a path the engine already minted a Sub
    /// for.
    pub(crate) fn try_promote(
        &mut self,
        promoter_id: PromoterId,
        promote_path: &Path,
        observed_kind: EntryKind,
        out: &mut StepOutput,
    ) {
        // I-Promoter-5: dedup gate.
        let already_present = self
            .promoters
            .get(promoter_id)
            .is_some_and(|q| q.dynamic_subs.contains_key(promote_path));
        if already_present {
            return;
        }

        // Capture spec fields BEFORE the &mut borrow chain on
        // attach_sub_inner. Cloning the heavy fields once (config /
        // command) is cheaper than re-borrowing the registry across
        // each access.
        let Some((promoter_name, config, max_settle, settle, command, scope, events, log_output)) =
            self.promoters.get(promoter_id).map(|q| {
                (
                    q.name.clone(),
                    q.config.clone(),
                    q.max_settle,
                    q.settle,
                    q.command.clone(),
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
            command,
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

        // Register in dedup map (I-Promoter-4 ADD).
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
                debug_assert!(false, "glob in literal prefix violates parse invariant",);
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

#[cfg(test)]
#[path = "promoter_tests.rs"]
mod tests;
