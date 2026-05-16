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
//! Single-slot probe per Promoter, structural by mutually-exclusive
//! state: the descent probe lives on the `PrefixPending` descent's own
//! [`specter_core::ProbeSlot`]; the proxy enumeration probe lives on
//! the `Active` variant's `enumerating` [`specter_core::ProbeSlot`],
//! tagged with the proxy `ResourceId` it targets. The two states are
//! mutually exclusive, so a Promoter holds exactly one slot.
//! Concurrent enumerations queue via `pending_enumerations`;
//! `dispatch_next_enumeration` pops one target at a time and arms the
//! `Active` slot for it.
//!
//! Response routing splits by state — [`Engine::on_promoter_probe_response`]
//! gates staleness on `pending_probe_for`, snapshots the routing class
//! pre-disarm, disarms the slot once (consume-once), then dispatches:
//! - `PrefixPending` → descent: routes the outcome to the
//!   owner-polymorphic [`Engine::dispatch_descent_ok`] /
//!   [`Engine::dispatch_descent_vanished`] /
//!   [`Engine::dispatch_descent_failed`] in `descent.rs`; on completion
//!   of the last literal segment, [`Engine::enter_active`] (the
//!   Promoter-side terminal-arm helper) flips the state and registers
//!   the first proxy.
//! - `Active` → enumeration (`dispatch_promoter_enumeration_*`): each
//!   response either registers sub-proxies (intermediate components),
//!   mints dynamic Subs (final component), or unregisters proxies that
//!   no longer correspond to a directory entry. The proxy `ResourceId`
//!   comes from the pre-disarm route snapshot (the slot's tag) — the
//!   wire carries paths only, so the slot tag is the single
//!   authoritative source for the dispatch key across every outcome.

use crate::Engine;
use crate::descent::{MaterializeResult, kind_from_entry};
use crate::path::empty_path;
use crate::probe::ProbeRoute;
use crate::refcounts::{add_watch, sub_watch_then_try_reap};
use compact_str::CompactString;
use specter_core::{
    ClassSet, ContribKey, DescentState, Diagnostic, DirSnapshot, EntryKind, PatternComponent,
    PatternSpec, ProbeOutcome, ProbeOwner, ProbeResponse, ProbeSlot, Promoter,
    PromoterAttachRequest, PromoterId, PromoterState, ProxyState, ResourceId, ResourceRole,
    StepOutput, SubAttachAnchor, SubAttachRequest, SubId, SubParams, Tree,
};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
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
    /// yet exist on disk), opens the probe channel, and emits a
    /// [`Diagnostic::PromoterAttached`] carrying the minted
    /// [`PromoterId`].
    ///
    /// Sole public entry is [`crate::Input::AttachPromoter`] via
    /// [`Engine::step`]; the `pub(crate)` inner survives because
    /// [`Engine::on_config_diff`] composes multiple detach/attach
    /// operations into one [`StepOutput`] on hot reload.
    ///
    /// **Two materialisation paths:**
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
    /// emits [`Diagnostic::AttachPathInvalid`] and returns `None`
    /// without minting a [`PromoterId`].
    ///
    /// Compute-then-insert: the materialisation outcome decides the
    /// initial `PromoterState` shape *before* the registry insert, so
    /// the Promoter is registered with its final state and never
    /// observed in a transient placeholder shape (no `Active{empty}`
    /// stand-in for a `PrefixPending` Promoter that downstream
    /// owner-state readers could see mid-mutation).
    pub(crate) fn attach_promoter_inner(
        &mut self,
        req: PromoterAttachRequest,
        now: Instant,
        out: &mut StepOutput,
    ) -> Option<PromoterId> {
        // 1. Render the literal prefix. components[0..literal_prefix_len]
        // are all Literal post-parse; the loop is a fold.
        let prefix_path = render_literal_prefix(&req.pattern_spec);

        // 2. Parse. Defense-in-depth: PatternSpec::parse should have
        // rejected anything Tree::parse_attach_path would reject. On
        // failure, emit Diagnostic::AttachPathInvalid (preserving the
        // operator-facing hint) and return None.
        let parsed = match Tree::parse_attach_path(&prefix_path) {
            Ok(p) => p,
            Err(err) => {
                out.diagnostics.push(Diagnostic::AttachPathInvalid {
                    path: prefix_path,
                    hint: err.hint(),
                });
                return None;
            }
        };

        // 3. Compute materialise BEFORE insert. Pick the final state
        // shape; insert once.
        let materialize = self.materialize_path_or_pending(&parsed);

        // 4. Capture the literal_prefix_len; first proxy at materialisation
        // carries this index in `pattern.components`.
        let lpl = req.pattern_spec.literal_prefix_len();

        // 5. Construct the initial state directly. No placeholder. A
        // pending descent mints its correlation *here* so the slot is
        // constructed already armed — there is no window where the
        // PrefixPending phase exists without its probe correlation.
        let (initial_state, pending_descent) = match &materialize {
            MaterializeResult::Materialized(_) => (
                PromoterState::Active {
                    proxies: BTreeMap::new(),
                    enumerating: ProbeSlot::empty(),
                },
                None,
            ),
            MaterializeResult::Pending {
                prefix, remaining, ..
            } => {
                let correlation = self.mint_probe_correlation();
                (
                    PromoterState::PrefixPending(DescentState::new(
                        *prefix,
                        remaining.clone(),
                        ProbeSlot::armed(correlation, ()),
                    )),
                    Some((*prefix, correlation)),
                )
            }
        };

        // 6. Mint the Promoter with the final state. The slotmap key is
        // the identity authority — no id is embedded. `pattern_spec`
        // and `identity` move in (no clone); `program`'s Arc moves in —
        // the hot dispatcher bumps the refcount per enumeration response
        // to release the registry's read borrow before the forward pass
        // takes `&mut self`. Cloning the `PatternSpec` here (rather than
        // moving) would allocate per `Glob` component on every attach.
        let promoter_id = self.promoters.insert(Promoter {
            name: CompactString::from(req.name.as_str()),
            pattern: Arc::new(req.pattern_spec),
            identity: req.identity,
            program: req.program,
            scope: req.scope,
            settle: req.settle,
            log_output: req.log_output,
            state: initial_state,
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
            MaterializeResult::Pending { .. } => {
                let (prefix, correlation) =
                    pending_descent.expect("Pending materialise ⇒ correlation minted at step 5");
                // PrefixPending: install the
                // [`ContribKey::PromoterPrefix`] STRUCTURE
                // contribution, then emit the descent probe carrying
                // the already-armed slot's correlation.
                add_watch(
                    &mut self.tree,
                    prefix,
                    ContribKey::PromoterPrefix(promoter_id),
                    ClassSet::STRUCTURE,
                    out,
                );
                let owner = ProbeOwner::Promoter(promoter_id);
                let target_path = self.tree.path_of(prefix).unwrap_or_else(empty_path);
                Self::emit_descent_probe(owner, correlation, target_path, out);
            }
        }

        Some(promoter_id)
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
        _now: Instant,
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
        self.tree
            .promote_scaffold(new_proxy_resource, ResourceRole::User);

        // 1. Flip state to `Active { proxies: empty }` BEFORE any
        // refcount work. Owner-bookkeeping: the contribution-map
        // entry for `prior_prefix_to_release` is keyed by
        // [`ContribKey::PromoterPrefix`] and is removed by explicit
        // key below; the state-flip aligns
        // [`PromoterState`] readers with the post-release shape.
        if let Some(q) = self.promoters.get_mut(promoter_id) {
            q.state = PromoterState::Active {
                proxies: BTreeMap::new(),
                enumerating: ProbeSlot::empty(),
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
        self.dispatch_next_enumeration(promoter_id, out);
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
                PromoterState::Active { proxies, .. } => proxies.contains_key(&resource),
                PromoterState::PrefixPending(_) => unreachable!(
                    "register_proxy: state must be Active (promoter = {promoter_id:?}); \
                     enter_active and dispatch_promoter_enumeration_ok are the sole callers \
                     and both ensure Active before invocation"
                ),
            });

        if let Some(q) = self.promoters.get_mut(promoter_id) {
            match &mut q.state {
                PromoterState::Active { proxies, .. } => {
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
    ///
    /// **Cost.** BFS from `r` over the Tree's children, gated on the
    /// per-Resource [`specter_core::Resource::proxy_promoters`]
    /// back-ref. Cost is O(subtree_size) — the prior shape iterated
    /// the Promoter's full `proxies` map and walked
    /// `tree.ancestors(p)` per entry (O(total_proxies × depth)),
    /// scaling with fan-out elsewhere in the tree.
    /// `SmallVec::contains` on the back-ref is constant for the
    /// typical single-Promoter fan-out (inline cap 1).
    pub(crate) fn unregister_proxy_subtree(
        &mut self,
        promoter_id: PromoterId,
        r: ResourceId,
        out: &mut StepOutput,
    ) {
        // BFS-collect every Tree descendant of `r` (inclusive) that
        // carries a back-ref to this Promoter. The lockstep invariant
        // (`Resource.proxy_promoters` ↔ `Promoter.state.proxies`)
        // means the back-ref is the right-side projection of the join
        // and yields exactly the same set as the prior
        // ancestor-filter approach. A stale `r` (already reaped) is
        // a benign no-op: `tree.get` returns None on the first
        // iteration, `children_ids` returns the empty iterator, and
        // the BFS terminates with `to_unregister` empty.
        let mut to_unregister: Vec<ResourceId> = Vec::new();
        let mut queue: VecDeque<ResourceId> = VecDeque::from([r]);
        while let Some(node) = queue.pop_front() {
            let has_back_ref = self
                .tree
                .get(node)
                .is_some_and(|res| res.proxy_promoters.contains(&promoter_id));
            if has_back_ref {
                to_unregister.push(node);
            }
            // Enqueue children. `children_ids` returns an empty
            // iterator for a stale `node`, so the defensive case is
            // structurally absorbed.
            queue.extend(self.tree.children_ids(node));
        }

        // Order is order-independent for correctness:
        // `release_promoter_proxy_claim` (delegate of
        // `unregister_proxy`) is self-contained — back-ref clear,
        // contribution release, then `try_reap`. BFS order keeps
        // cleanup parent-before-child, but ancestor cascades inside
        // `try_reap` reach the same end state from any iteration
        // order (a parent whose proxy still contributes
        // [`specter_core::ContribKey::PromoterProxy`] survives the
        // first reap of its child; the parent's later release closes
        // the cascade).
        for proxy in to_unregister {
            self.unregister_proxy(promoter_id, proxy, out);
        }
    }

    /// Promoter-side probe response handler. Mirrors the Profile-side
    /// shape: one staleness gate, snapshot the routing class *before*
    /// the disarm, consume the slot once, then dispatch.
    ///
    /// Both Promoter probe carriers are state-resident — a
    /// `PrefixPending` descent on its `DescentState` slot, an `Active`
    /// enumeration on `enumerating`. `pending_probe_for(owner)` gates
    /// staleness uniformly; `take_owner_probe` disarms once
    /// (consume-once); a mismatch yields
    /// [`Diagnostic::StaleProbeResponse`] with no state change.
    ///
    /// `probe_route` is snapshotted pre-disarm: the disarm empties the
    /// slot but leaves the state variant intact, so the route — and
    /// the enumeration `target` it carries — stays valid through
    /// dispatch. `Vanished` / `Failed` enumeration responses carry no
    /// wire payload; that pre-disarm route snapshot is the sole
    /// authority for the proxy `ResourceId`.
    ///
    /// A matched correlation routing to a Profile class, or to none,
    /// is unconstructable for a Promoter owner — the loud arm is
    /// regression detection, not a reachable path.
    pub(crate) fn on_promoter_probe_response(
        &mut self,
        promoter_id: PromoterId,
        response: ProbeResponse,
        now: Instant,
        out: &mut StepOutput,
    ) {
        let owner = response.owner;
        let received = response.correlation;

        if self.pending_probe_for(owner) != Some(received) {
            out.diagnostics.push(Diagnostic::StaleProbeResponse {
                owner,
                correlation: received,
            });
            return;
        }

        // Snapshot the route before the disarm: `take_owner_probe`
        // empties the slot but leaves the carrier variant intact, so a
        // route captured here (including the enumeration `target`)
        // stays valid through dispatch.
        let route = self.probe_route(owner);
        let consumed = self.take_owner_probe(owner);
        debug_assert_eq!(
            consumed,
            Some(received),
            "consume-once: promoter slot disarm must yield the gated \
             correlation (promoter = {promoter_id:?})",
        );
        #[cfg(debug_assertions)]
        self.dispatch_ledger.record(owner, received);

        match route {
            Some(ProbeRoute::Descent) => match response.outcome {
                ProbeOutcome::SubtreeOk(arc) => self.dispatch_descent_ok(owner, &arc, now, out),
                ProbeOutcome::Vanished => self.dispatch_descent_vanished(owner, now, out),
                ProbeOutcome::Failed { errno } => self.dispatch_descent_failed(owner, errno, out),
                ProbeOutcome::AnchorOk(_) => {
                    // Descent probes a Dir prefix; the walker returns
                    // `SubtreeOk` / `Vanished`. `AnchorOk` is a
                    // walker-side regression.
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
            },

            Some(ProbeRoute::Enumerating { target }) => match response.outcome {
                ProbeOutcome::SubtreeOk(arc) => {
                    self.dispatch_promoter_enumeration_ok(promoter_id, target, &arc, now, out);
                }
                ProbeOutcome::Vanished => {
                    self.dispatch_promoter_enumeration_vanished(promoter_id, target, out);
                }
                ProbeOutcome::Failed { errno } => {
                    self.dispatch_promoter_enumeration_failed(promoter_id, target, errno, out);
                }
                ProbeOutcome::AnchorOk(_) => {
                    // Enumeration probes a Dir proxy; the walker
                    // returns `SubtreeOk` / `Vanished`. `AnchorOk` is a
                    // walker-side regression.
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
            },

            Some(ProbeRoute::Verifying { .. } | ProbeRoute::Rebasing) | None => {
                // The gated correlation matched but the route is a
                // Profile class or none: unconstructable for a Promoter
                // owner (its correlation lives on exactly one Promoter
                // carrier). The loud arm replaces the old
                // channel-vs-state cross-affinity bail.
                debug_assert!(
                    false,
                    "matched correlation but Promoter state is not a Promoter-probe-bearing \
                     carrier (promoter = {promoter_id:?})",
                );
                out.diagnostics.push(Diagnostic::StaleProbeResponse {
                    owner,
                    correlation: received,
                });
            }
        }

        // Tail drain: a terminal descent flipped to Active and
        // `enter_active` already armed its first enumeration; a
        // non-terminal advance re-armed the descent slot; either way
        // the I5 gate inside `dispatch_next_enumeration` no-ops. An
        // enumeration response left its slot empty, so the next queued
        // target drains here.
        self.dispatch_next_enumeration(promoter_id, out);
    }

    /// Drain one queued enumeration target into a probe. No-op if a
    /// probe is already in flight (single-slot discipline) or the
    /// queue is empty.
    ///
    /// Mints a fresh correlation and arms the `Active` enumeration slot
    /// for the popped `target` in a single state write — the slot is
    /// constructed armed, never armed-after-emit, so there is no window
    /// where the enumeration is in flight without its correlation on
    /// state. The slot's tag is the sole authority for the proxy
    /// [`ResourceId`] across every response outcome (`SubtreeOk` /
    /// `Vanished` / `Failed`); the wire carries only `target_path`.
    /// Consumed once via `take_owner_probe` in
    /// [`Self::on_promoter_probe_response`].
    pub(crate) fn dispatch_next_enumeration(
        &mut self,
        promoter_id: PromoterId,
        out: &mut StepOutput,
    ) {
        let owner = ProbeOwner::Promoter(promoter_id);
        // At most one outstanding probe per Promoter (I5).
        if self.pending_probe_for(owner).is_some() {
            return;
        }

        // Pop the next pending enumeration target. `pop_first` is a
        // single-shot fetch+remove from the BTreeSet so the queue
        // stays in lockstep with the in-flight probe. A non-empty
        // queue implies `Active` — the sole populators (`register_proxy`
        // and the overflow reseed) both require it — so the
        // `arm_enumeration` below never reaches its `PrefixPending`
        // guard.
        let target = self
            .promoters
            .get_mut(promoter_id)
            .and_then(|q| q.pending_enumerations.pop_first());
        let Some(target) = target else {
            return;
        };

        let correlation = self.mint_probe_correlation();
        if let Some(q) = self.promoters.get_mut(promoter_id) {
            q.state.arm_enumeration(correlation, target);
        }

        let target_path = self.tree.path_of(target).unwrap_or_else(empty_path);
        Self::emit_descent_probe(owner, correlation, target_path, out);
    }

    /// Successful enumeration response — the walker enumerated one
    /// level of the proxy at `target` and returned its children. Two
    /// passes:
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
    ///
    /// `target` is the proxy [`ResourceId`] taken from the enumeration
    /// slot's tag (snapshotted pre-disarm in the response handler) —
    /// the wire is path-only, so that tag is the single authoritative
    /// source for the dispatch key.
    pub(crate) fn dispatch_promoter_enumeration_ok(
        &mut self,
        promoter_id: PromoterId,
        target: ResourceId,
        snapshot: &DirSnapshot,
        now: Instant,
        out: &mut StepOutput,
    ) {
        // Look up this proxy's pattern_component_index.
        let proxy_state = self.promoters.get(promoter_id).and_then(|q| {
            if let PromoterState::Active { proxies, .. } = &q.state {
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

        // Bump the Promoter's pattern Arc to release the read borrow on
        // `self.promoters` before the forward pass takes `&mut self` via
        // `try_promote` / `register_proxy`. The refcount bump replaces
        // the prior per-response `components().to_vec()` clone — every
        // `Glob` component cloned three `String`s, so the cost scaled
        // with pattern depth × glob count. The auto-deref `Arc<T> → &T`
        // lets the loop body read components without further ceremony.
        let Some(pattern) = self
            .promoters
            .get(promoter_id)
            .map(|q| Arc::clone(&q.pattern))
        else {
            return;
        };
        let components = pattern.components();
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

        let next_index = pattern_component_index + 1;
        let is_final = next_index == components.len();

        // Loop-invariant for the Glob arm's per-entry join below; bind
        // once. `None` is the defensive target-was-reaped path — the
        // `proxy_state` lookup above already proved the slot live this
        // step, so it is unreachable under normal operation; the
        // `as_deref().map().unwrap_or_default()` chain below degrades to
        // an empty `PathBuf` (rejected downstream by
        // `decompose_attach_path`) rather than panicking.
        let target_path = self.tree.path_of(target);

        // Forward pass: state-keyed dispatch on the next pattern
        // component. The Literal arm matches at most one entry by
        // `BTreeMap` uniqueness — O(log N) `get_key_value` — so the
        // surrounding loop collapses to an `if let`. The Glob arm
        // retains the O(N) scan; each entry must be matcher-tested.
        // The per-match body is duplicated inline rather than
        // extracted into a helper: the two iteration shapes
        // structurally differ (no loop, no `continue` in the Literal
        // arm), and a `(&mut self, ...)` helper would carry nine
        // arguments for ten lines of body.
        let next_component = &components[pattern_component_index];
        match next_component {
            PatternComponent::Literal(lit) => {
                if let Some((name, child)) = snapshot.entries().get_key_value(lit.as_str()) {
                    let name_str: &str = name.as_str();
                    let child_kind = child.kind();
                    if is_final {
                        // Final: resource-anchored try_promote.
                        // Lookup-or-ensure the slot at (target,
                        // name_str), stamp the observed kind (so
                        // Profile.kind cache populates at attach
                        // instead of waiting for the first Seed
                        // probe), then mint. `User` role is what
                        // attach_sub_inner promotes DescentScaffold
                        // to — pre-ensuring with `User` short-circuits
                        // that path.
                        let anchor_resource =
                            self.tree.lookup(Some(target), name_str).unwrap_or_else(|| {
                                self.tree
                                    .ensure_child(target, name_str, ResourceRole::User)
                                    .expect(
                                        "promoter target held alive by anchor / proxy_promoters",
                                    )
                            });
                        self.tree
                            .set_kind(anchor_resource, kind_from_entry(child_kind));
                        let promote_path = target_path
                            .as_deref()
                            .map(|p| p.join(name_str))
                            .unwrap_or_default();
                        self.try_promote(
                            promoter_id,
                            anchor_resource,
                            promote_path,
                            child_kind,
                            now,
                            out,
                        );
                    } else if matches!(child_kind, EntryKind::Dir) {
                        // Non-final: only descend into Dir matches.
                        // A literal matching a Leaf at a non-final
                        // position can't lead anywhere; the engine
                        // drops without diagnostic (the user's
                        // pattern was malformed for the actual
                        // filesystem state).
                        //
                        // [S-8] Use User role for promoter sub-proxy
                        // slots. The back-ref (proxy_promoters) is
                        // the retention signal; DescentScaffold would
                        // leak after unregister.
                        let child_resource =
                            self.tree.lookup(Some(target), name_str).unwrap_or_else(|| {
                                self.tree
                                    .ensure_child(target, name_str, ResourceRole::User)
                                    .expect(
                                        "promoter target held alive by anchor / proxy_promoters",
                                    )
                            });
                        self.tree
                            .set_kind(child_resource, kind_from_entry(child_kind));
                        self.register_proxy(promoter_id, child_resource, next_index, out);
                    }
                }
            }
            PatternComponent::Glob(g) => {
                for (name, child) in snapshot.entries() {
                    let name_str: &str = name.as_str();
                    if !g.matches_path(Path::new(name_str)) {
                        continue;
                    }
                    let child_kind = child.kind();
                    if is_final {
                        let anchor_resource =
                            self.tree.lookup(Some(target), name_str).unwrap_or_else(|| {
                                self.tree
                                    .ensure_child(target, name_str, ResourceRole::User)
                                    .expect(
                                        "promoter target held alive by anchor / proxy_promoters",
                                    )
                            });
                        self.tree
                            .set_kind(anchor_resource, kind_from_entry(child_kind));
                        let promote_path = target_path
                            .as_deref()
                            .map(|p| p.join(name_str))
                            .unwrap_or_default();
                        self.try_promote(
                            promoter_id,
                            anchor_resource,
                            promote_path,
                            child_kind,
                            now,
                            out,
                        );
                    } else if matches!(child_kind, EntryKind::Dir) {
                        let child_resource =
                            self.tree.lookup(Some(target), name_str).unwrap_or_else(|| {
                                self.tree
                                    .ensure_child(target, name_str, ResourceRole::User)
                                    .expect(
                                        "promoter target held alive by anchor / proxy_promoters",
                                    )
                            });
                        self.tree
                            .set_kind(child_resource, kind_from_entry(child_kind));
                        self.register_proxy(promoter_id, child_resource, next_index, out);
                    }
                }
            }
        }

        // Reverse pass: unwind proxies whose underlying entry is gone.
        // Walking the Tree from `target` down (rather than iterating
        // the Promoter's full `proxies` map and filtering by
        // `parent == target`) scales with `target.fanout` instead of
        // `Promoter.total_proxies` — symmetric with the BFS in
        // [`Self::unregister_proxy_subtree`] (both replace
        // "iterate Promoter.proxies" with "walk the Tree's right side
        // of the join via `proxy_promoters` back-ref"). Deeper proxies
        // cascade through the BFS inside `unregister_proxy_subtree`.
        let snapshot_names: BTreeSet<&str> = snapshot
            .entries()
            .keys()
            .map(CompactString::as_str)
            .collect();
        let stale: Vec<ResourceId> = self
            .tree
            .children_ids(target)
            .filter(|&child| {
                let has_back_ref = self
                    .tree
                    .get(child)
                    .is_some_and(|res| res.proxy_promoters.contains(&promoter_id));
                if !has_back_ref {
                    return false;
                }
                self.tree
                    .name(child)
                    .is_some_and(|n| !snapshot_names.contains(n))
            })
            .collect();
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
    /// `target` is the enumeration slot's tag, snapshotted pre-disarm
    /// in the response handler. A kernel-driven cascade (the proxy's
    /// parent's enumeration_ok reverse pass triggered by the parent's
    /// `StructureChanged` event when the proxy is removed) reaches
    /// the same end state; observing `Vanished` directly short-circuits
    /// that round-trip.
    pub(crate) fn dispatch_promoter_enumeration_vanished(
        &mut self,
        promoter_id: PromoterId,
        target: ResourceId,
        out: &mut StepOutput,
    ) {
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
    /// `target` is the enumeration slot's tag, snapshotted pre-disarm
    /// in the response handler — always present, no defensive
    /// fallback needed.
    #[allow(clippy::unused_self)]
    pub(crate) fn dispatch_promoter_enumeration_failed(
        &self,
        promoter_id: PromoterId,
        target: ResourceId,
        errno: i32,
        out: &mut StepOutput,
    ) {
        out.diagnostics.push(Diagnostic::PromoterEnumerationFailed {
            promoter: promoter_id,
            proxy: target,
            errno,
        });
    }

    /// Mint a dynamic Sub at `anchor_resource` for `promoter_id`.
    ///
    /// Enumeration ADDS; only anchor-terminal removes. At most one
    /// dynamic Sub per `(promoter_id, anchor_resource)` — the contains
    /// check below gates re-promotion at an anchor the engine already
    /// minted a Sub for.
    ///
    /// **`now` is load-bearing.** `attach_sub_inner` schedules the
    /// new Profile's `BurstDeadline` at `now + max_settle`. Threading
    /// the step's `now` keeps the dynamic Sub's clock coherent with
    /// the rest of the step; reading the system clock here would let
    /// time advance silently between caller and callee within a single
    /// `step` invocation.
    ///
    /// **Resource-anchored attach.** `anchor_resource` is the live
    /// Tree slot the forward-pass call site looked up or freshly
    /// `ensure_child`'d as [`specter_core::ResourceRole::User`] before
    /// invoking `try_promote`. The request is built via
    /// [`SubAttachRequest::from_parts`] with a
    /// [`SubAttachAnchor::Resource`]; the engine's Resource-arm
    /// liveness check passes on a slot this freshly minted, so
    /// `attach_sub_inner` cannot return `None` — the `.expect` records
    /// that invariant rather than masking a soft early-return.
    ///
    /// **`promote_path` is diagnostic-only.** The caller's
    /// `target_path.join(name_str)` flows through as an owned
    /// `PathBuf` so the [`Diagnostic::PromotionKindObserved`] payload
    /// can move it (last use, no clone). The dedup-map insert keys on
    /// `anchor_resource: Copy` — no consumption-ordering constraint
    /// between the insert and the diagnostic.
    pub(crate) fn try_promote(
        &mut self,
        promoter_id: PromoterId,
        anchor_resource: ResourceId,
        promote_path: PathBuf,
        observed_kind: EntryKind,
        now: Instant,
        out: &mut StepOutput,
    ) {
        // Dedup gate: one dynamic Sub per `(promoter_id, anchor_resource)`.
        let already_present = self
            .promoters
            .get(promoter_id)
            .is_some_and(|q| q.dynamic_subs.contains_key(&anchor_resource));
        if already_present {
            return;
        }

        // Capture spec fields BEFORE the &mut borrow chain on
        // attach_sub_inner. `identity` is cloned once (its `ScanConfig`
        // is the only heap field) and `program` Arc-cloned (refcount
        // bump) — cheaper than re-borrowing the registry per access.
        let Some((promoter_name, identity, settle, program, scope, log_output)) =
            self.promoters.get(promoter_id).map(|q| {
                (
                    q.name.clone(),
                    q.identity.clone(),
                    q.settle,
                    Arc::clone(&q.program),
                    q.scope,
                    q.log_output,
                )
            })
        else {
            return;
        };

        let synthesized = format!("{promoter_name}@{}", promote_path.display());

        // Resource-anchored: `anchor_resource` is the slot the
        // forward-pass just looked up or `ensure_child`'d as `User`, so
        // the engine's Resource-arm liveness check passes and
        // `attach_sub_inner` cannot return `None` here. `identity` is
        // moved straight through — no flat round-trip.
        let req = SubAttachRequest::from_parts(
            SubAttachAnchor::Resource(anchor_resource),
            identity,
            SubParams {
                name: synthesized,
                program,
                scope,
                settle,
                log_output,
                source_promoter: Some(promoter_id),
            },
        );

        let sub_id = self.attach_sub_inner(req, now, out).expect(
            "promoter forward-pass anchored at a freshly ensured live User slot; \
             the engine's Resource-arm liveness check cannot trip",
        );

        // Dedup map. `anchor_resource: Copy` — no consumption
        // constraint against the diagnostic move below.
        if let Some(q) = self.promoters.get_mut(promoter_id) {
            q.dynamic_subs.insert(anchor_resource, sub_id);
        }

        // Diagnostic. Last use of `promote_path` — move into the
        // variant.
        out.diagnostics.push(Diagnostic::PromotionKindObserved {
            promoter: promoter_id,
            path: promote_path,
            kind: kind_from_entry(observed_kind),
        });

        // Fanout warning. One-shot per Promoter lifetime. Single read
        // of the Promoter — within this `&mut self` step nothing else
        // can mutate `dynamic_subs.len()` or `warned_at_threshold`
        // between two reads, so the fused projection makes the
        // no-races contract locally visible.
        let crossed = self.promoters.get(promoter_id).and_then(|q| {
            let count = q.dynamic_subs.len();
            (count > FANOUT_WARNING_THRESHOLD && !q.warned_at_threshold).then_some(count)
        });
        if let Some(count) = crossed {
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
        _now: Instant,
        out: &mut StepOutput,
    ) {
        let has_proxy = self
            .promoters
            .get(promoter_id)
            .is_some_and(|q| match &q.state {
                PromoterState::Active { proxies, .. } => proxies.contains_key(&resource),
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
        self.dispatch_next_enumeration(promoter_id, out);
    }

    /// Notify a Promoter that one of its dynamic Subs has reaped (the
    /// Sub's anchor disappeared, and the all-dynamic teardown branch of
    /// [`Engine::on_anchor_terminal_event`] is unwinding the Profile).
    ///
    /// Removes the `(anchor_resource → sub_id)` entry from
    /// `Promoter.dynamic_subs` and emits
    /// [`Diagnostic::DynamicSubReaped`]. This is one of three documented
    /// mutators of `dynamic_subs` (alongside [`Self::try_promote`] for
    /// inserts and [`Self::reap_promoter_inner`] for full drains).
    ///
    /// `anchor_resource` is the dedup-map key — by construction, the
    /// same `Profile.resource` `try_promote` stamped into the map.
    /// `anchor_path` is the operator-facing diagnostic payload (the
    /// `Diagnostic` variant is path-keyed for log readability).
    ///
    /// The caller ([`Engine::on_anchor_terminal_all_dynamic`]) captures
    /// both once before the per-Sub loop because every dynamic Sub on
    /// a Profile shares the same anchor by the
    /// `(resource, config_hash)` attach dedup; threading them through
    /// avoids re-walking the ancestor chain inside the inner loop.
    ///
    /// **Stale notification.** A concurrent
    /// [`Self::reap_promoter_inner`] (e.g., reload removing the
    /// Promoter in the same step) may have already cleared the
    /// dynamic_subs map, in which case the `remove` returns `None` and
    /// the call is a benign no-op (no diagnostic emitted).
    pub(crate) fn on_dynamic_sub_reaped(
        &mut self,
        promoter_id: PromoterId,
        sub_id: SubId,
        anchor_resource: ResourceId,
        anchor_path: &Path,
        out: &mut StepOutput,
    ) {
        let removed = self
            .promoters
            .get_mut(promoter_id)
            .and_then(|q| q.dynamic_subs.remove(&anchor_resource));
        if let Some(stored_sub_id) = removed {
            debug_assert_eq!(
                stored_sub_id, sub_id,
                "on_dynamic_sub_reaped: dynamic_subs entry's SubId disagrees with caller's \
                 (promoter = {promoter_id:?}, anchor = {anchor_resource:?})",
            );
            out.diagnostics.push(Diagnostic::DynamicSubReaped {
                promoter: promoter_id,
                sub: sub_id,
                path: anchor_path.to_path_buf(),
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
    ///
    /// Time-independent like `detach_sub_inner`: the helper
    /// drives only refcount and registry teardown; bursts running on
    /// Profiles cascaded by promoter reap continue under their
    /// existing schedule.
    pub fn reap_promoter(&mut self, pid: PromoterId) -> StepOutput {
        let mut out = StepOutput::default();
        self.reap_promoter_inner(pid, &mut out);
        out.sort_for_emission();
        out
    }

    /// Inner reap used by `on_config_diff` to compose multiple
    /// detach/attach operations into a single (sorted) [`StepOutput`].
    ///
    /// Sequence:
    /// 1. Cancel any in-flight probe (descent or enumeration). The
    ///    disarm clears the slot's correlation and, for an `Active`
    ///    enumeration, its proxy-target tag along with it.
    /// 2. Pre-clear `dynamic_subs` so any cascading detach paths see
    ///    an empty map (defense-in-depth — `detach_sub_inner` itself
    ///    doesn't read it). Then iterate the captured Sub ids and
    ///    detach each via [`Engine::detach_sub_inner`]; each detach
    ///    runs the standard deferred-reap-or-immediate-reap branch
    ///    on the corresponding Profile.
    /// 3. Release the per-Resource claims by sequencing the two
    ///    idempotent release helpers — `PromoterState` is
    ///    `PrefixPending` XOR `Active`, and each helper is a no-op
    ///    against the wrong arm, so the unconditional sequence
    ///    handles both cases without a discriminant projection:
    ///    - [`Self::release_promoter_descent_prefix_claim`] releases
    ///      the [`specter_core::ContribKey::PromoterPrefix`]
    ///      contribution (PrefixPending arm); side-effects a state
    ///      flip to `Active{empty}` so the proxy iteration that
    ///      follows sees a uniform shape.
    ///    - [`Self::unregister_proxy`] (per snapshot of
    ///      `state.proxies.keys()`) clears the back-ref, drops the
    ///      [`specter_core::ContribKey::PromoterProxy`] contribution,
    ///      and try-reaps the slot.
    /// 4. Remove the Promoter from the registry. Emit
    ///    [`Diagnostic::PromoterReaped`].
    pub(crate) fn reap_promoter_inner(&mut self, promoter_id: PromoterId, out: &mut StepOutput) {
        // Stale id: silent no-op. Mirrors `detach_sub_inner`'s
        // defensive shape but without the `DetachUnknownSub`-style
        // diagnostic — there is no `DetachUnknownPromoter` variant in
        // the catalog and the stale path is benign (the bin races a
        // ConfigDiff against an in-flight reap).
        if self.promoters.get(promoter_id).is_none() {
            return;
        }

        // 1. Consume any in-flight probe. `cancel_owner_probe` disarms
        // the descent slot (PrefixPending) or closes the enumeration
        // channel entry (Active), and emits `ProbeOp::Cancel` iff one
        // was in flight. `pending_enumerations` drains as a side effect
        // of the proxy-release pass below: every queued entry
        // corresponds to a registered proxy, and
        // `release_promoter_proxy_claim` removes its own queue entry
        // inside `unregister_proxy`.
        self.cancel_owner_probe(ProbeOwner::Promoter(promoter_id), out);
        debug_assert!(
            self.pending_probe_for(ProbeOwner::Promoter(promoter_id))
                .is_none(),
            "reap_promoter_inner: probe still in flight for promoter = {promoter_id:?} \
             after cancel_owner_probe; cancel-first contract violated",
        );

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
            self.detach_sub_inner(sub_id, out);
        }

        // 3. Release the per-Resource claims. Both helpers are
        // idempotent against the wrong state arm, so the unconditional
        // sequence handles `PrefixPending` and `Active` uniformly
        // without a discriminant projection.
        //
        // `release_promoter_descent_prefix_claim` no-ops when state is
        // already `Active`; when state is `PrefixPending` it captures
        // `current_prefix`, flips state to `Active{empty}`, and releases
        // the [`ContribKey::PromoterPrefix`] contribution + try-reaps
        // the slot. The cancel-first precondition (channel closed) is
        // satisfied by the cancel above.
        self.release_promoter_descent_prefix_claim(promoter_id, out);

        // Snapshot the proxy keys post-release: state is now
        // `Active` for both input arms — empty for the
        // PrefixPending-input case (zero iterations), populated for the
        // Active-input case. The `PrefixPending` match arm is
        // unreachable at this point and present for defensive shape
        // only. `unregister_proxy` (delegating to
        // `release_promoter_proxy_claim`) clears the back-ref, drops
        // the contribution, removes the queue entry, and try-reaps the
        // slot. Order doesn't matter — each proxy's cleanup is
        // self-contained.
        let proxy_list: Vec<ResourceId> = self
            .promoters
            .get(promoter_id)
            .map(|q| match &q.state {
                PromoterState::Active { proxies, .. } => proxies.keys().copied().collect(),
                PromoterState::PrefixPending(_) => Vec::new(),
            })
            .unwrap_or_default();
        for r in proxy_list {
            self.unregister_proxy(promoter_id, r, out);
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

#[cfg(test)]
#[path = "promoter_tests.rs"]
mod tests;
