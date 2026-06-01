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
//!   PrefixPending(d) ───── descent ─────▶ Active { proxies }
//!         ▲   ▲                                 │
//!         │   │ rewind on Vanished              │ terminus rm -rf
//!         │   └─────────────────────────────────┘ ⇒ proxies: ∅
//!         │ start_promoter_prefix_recovery (parent's next
//!         │   StructureChanged on the preserved prefix_parent edge)
//! ```
//!
//! `PrefixPending → Active` ([`Engine::enter_active`]) is bidirectional:
//! the inverse `Active → PrefixPending`
//! ([`specter_core::Promoter::reenter_prefix_pending`]) is the
//! terminus-loss recovery move, the structural mirror of a Profile's
//! `Pending ↔ Idle` anchor-loss recovery.
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
//! gates on [`Engine::promoter_probe_gate`] (correlation + routing class
//! in one resolution), disarms the slot once (consume-once), then
//! dispatches:
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
use crate::descent::MaterializeResult;
use crate::probe::{DescentOutcome, PromoterProbeRoute, WalkerContractViolation};
use crate::refcounts::{add_watch, sub_watch};
use compact_str::{CompactString, format_compact};
use specter_core::{
    ClassSet, ContribKey, DescentState, DetachReason, Diagnostic, DirSnapshot, EntryKind,
    PatternComponent, PatternSpec, ProbeFailure, ProbeOwner, ProbeResponse, ProbeSlot, Promoter,
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
/// likely a too-broad pattern. `Promoter::latch_fanout_warning`
/// check-and-latches atomically, so a steady-state busy Promoter only
/// warns once per lifetime by construction.
pub(crate) const FANOUT_WARNING_THRESHOLD: usize = 1000;

impl Engine {
    /// Attach a Promoter to the engine. Materialises the literal-prefix
    /// path on the Tree (creating scaffolds where the prefix doesn't
    /// yet exist on disk), arms the Promoter's state-resident probe
    /// slot, and emits a [`Diagnostic::PromoterAttached`] carrying
    /// the minted [`PromoterId`].
    ///
    /// Sole public entry is [`specter_core::Input::AttachPromoter`] via
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
        out: &mut StepOutput,
    ) -> Option<PromoterId> {
        // Structural post-insert token: carried out of step 5's `match`
        // so the post-insert work is type-enforced rather than
        // re-derived from a surviving `materialize` via `expect`.
        // `PendingDescent` carries only `prefix` (the `add_watch`
        // target); the slot is constructed armed inside the inserted
        // Promoter and `emit_owner_probe` reads the correlation back
        // off state. Declared at function top (no
        // items-after-statements).
        enum PostInsert {
            EnterActive { proxy_resource: ResourceId },
            PendingDescent { prefix: ResourceId },
        }

        // `req` stays intact until construction: `Promoter::from_request`
        // is its single consumer (every field moves there once). The
        // pattern spec is borrowed twice pre-insert (prefix render +
        // literal length); the lifecycle diagnostic takes the one
        // irreducible inline `CompactString` clone since the registry
        // owns the original.

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
                    path: Arc::from(prefix_path),
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
        // `materialize` is consumed exactly once: `remaining` moves
        // straight into the linear descent slot (no clone); the
        // post-insert step rides out as the `PostInsert` token above.
        let (initial_state, post_insert) = match materialize {
            MaterializeResult::Materialized(proxy_resource) => (
                PromoterState::Active {
                    proxies: BTreeMap::new(),
                    enumerating: ProbeSlot::empty(),
                },
                PostInsert::EnterActive { proxy_resource },
            ),
            MaterializeResult::Pending {
                prefix, remaining, ..
            } => {
                let correlation = self.mint_probe_correlation();
                (
                    PromoterState::PrefixPending(DescentState::new(
                        prefix,
                        remaining,
                        ProbeSlot::armed(correlation, ()),
                    )),
                    PostInsert::PendingDescent { prefix },
                )
            }
        };

        // 6. Mint the Promoter with the final state. The slotmap key
        // is the identity authority — no id is embedded.
        // `from_request` moves every spec field in (Arc-wrapping the
        // pattern so the hot dispatcher can refcount-bump it per
        // enumeration response) and starts the runtime fields empty;
        // the lone copy is `diag_name` for the narration below.
        let diag_name = req.name.clone();
        let promoter_id = self
            .promoters
            .insert(Promoter::from_request(req, initial_state));

        // 7. Lifecycle diagnostic — pure operator narration, emitted
        // before any operation that could early-return so it is
        // deterministic across the step. Identity resolution uses the
        // engine's `by_name` index, not this stream.
        out.diagnostics.push(Diagnostic::PromoterAttached {
            promoter: promoter_id,
            name: diag_name,
        });

        // 8. Branch on the post-insert token to set up watches/probes.
        match post_insert {
            PostInsert::EnterActive { proxy_resource } => {
                // Single helper: enter_active demotes the slot's role
                // (DescentScaffold → User) where applicable, registers
                // the first proxy, and dispatches the initial
                // enumeration. No prior watch_demand contribution to
                // release on this path.
                self.enter_active(
                    promoter_id,
                    /* prior_prefix_to_release */ None,
                    /* new_proxy_resource */ proxy_resource,
                    /* pattern_component_index */ lpl,
                    out,
                );
            }
            PostInsert::PendingDescent { prefix } => {
                // PrefixPending: install the
                // [`ContribKey::PromoterPrefix`] STRUCTURE contribution,
                // then route through the choke. The descent slot was
                // *constructed armed* inside the inserted Promoter
                // (`promoters.insert` is infallible — no fallible arm
                // guard, so no loud `.expect`; this is the second
                // construct-armed-then-infallible-write launch site,
                // the Promoter twin of `start_seed_burst`), so
                // `emit_owner_probe` reads the correlation and the
                // prefix target straight back off state.
                add_watch(
                    &mut self.tree,
                    prefix,
                    ContribKey::PromoterPrefix(promoter_id),
                    ClassSet::STRUCTURE,
                    out,
                );
                self.emit_owner_probe(ProbeOwner::Promoter(promoter_id), out);
            }
        }

        Some(promoter_id)
    }

    /// Pure pre-check that returns `true` iff a subsequent
    /// [`Self::attach_promoter_inner`] call with `req` would clear its
    /// only fallible boundary — `Tree::parse_attach_path` on the
    /// rendered literal prefix. Mutates nothing; never installs a
    /// scaffold, never mints a `PromoterId`.
    ///
    /// Sole consumer is [`Self::on_config_diff`]'s `promoters.modified`
    /// arm: it runs the validate *before* reaping the old Promoter so
    /// a malformed prefix doesn't tear down a live Promoter for nothing.
    /// The reap + attach pair is then total — validate said yes, so the
    /// attach won't surface a parse error.
    ///
    /// Emits the same diagnostic [`Self::attach_promoter_inner`] would
    /// on the failure path (`AttachPathInvalid`), so the validate-then-act
    /// site never re-emits on its own. The re-parse at attach time (one
    /// `Tree::parse_attach_path` call, O(prefix length), pure) is the cost
    /// of total composition.
    ///
    /// Defense-in-depth: a `PatternSpec` that survived parse
    /// validation cannot fail this check (the parser is strictly
    /// stricter than `Tree::parse_attach_path`), so failure here would
    /// signal an upstream contract breach. Emitting on failure rather
    /// than `unreachable!`'ing keeps the engine total on bad input.
    ///
    /// Associated function — no engine state is read. The Sub-side
    /// counterpart [`Self::validate_sub_attach`] is a `&self` method
    /// because its `Resource` arm reads `self.tree`; this side has only
    /// the rendered-prefix arm, so the honest shape carries no `&self`.
    pub(crate) fn validate_promoter_attach(
        req: &PromoterAttachRequest,
        out: &mut StepOutput,
    ) -> bool {
        let prefix_path = render_literal_prefix(&req.pattern_spec);
        match Tree::parse_attach_path(&prefix_path) {
            Ok(_) => true,
            Err(err) => {
                out.diagnostics.push(Diagnostic::AttachPathInvalid {
                    path: Arc::from(prefix_path),
                    hint: err.hint(),
                });
                false
            }
        }
    }

    /// Single helper for both Promoter materialisation paths —
    /// structurally identical to [`Engine::materialize_profile_anchor`]'s
    /// prefix → parent handoff:
    ///
    /// - **immediate-Materialized** (from `attach_promoter_inner`): no
    ///   prior prefix to release; first proxy installed at the prefix
    ///   slot.
    /// - **`PrefixPending → Active`** (from
    ///   [`Engine::dispatch_descent_ok`]'s last-literal arm for the
    ///   Promoter owner, including the *terminus-loss recovery*
    ///   re-descent): the prior prefix's
    ///   [`ContribKey::PromoterPrefix`] contribution is *handed off*
    ///   (released without `try_reap`) to the preserved parent-edge
    ///   watch, and the new proxy is installed at the
    ///   freshly-materialised terminus (a child of that prior prefix).
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
    /// - `prior_prefix_to_release` (if `Some`) had its
    ///   [`ContribKey::PromoterPrefix`] contribution released by a
    ///   plain `sub_watch` — **no `try_reap`**: the slot is kept alive
    ///   on purpose as the recovery parent edge (and the new terminus
    ///   is its live child anyway). The descent-watch role transfers
    ///   to the parent-edge contribution installed next, exactly as the
    ///   Profile twin trades `ProfileDescent` for `ProfileParent`.
    /// - `tree.parent(new_proxy_resource)` carries `+1 STRUCTURE`
    ///   [`ContribKey::PromoterPrefixParent`] (the preserved recovery
    ///   edge), cached on `Promoter.prefix_parent` — *unless* the
    ///   terminus is `/` (no parent), where the channel is neither
    ///   installable nor needed. Idempotent on the recovery cycle
    ///   (`set_promoter_prefix_parent`'s `already_set` skip).
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
        self.promoters
            .mutate(promoter_id, specter_core::Promoter::enter_active_empty);

        // 2. Hand off the prior prefix's STRUCTURE contribution. Plain
        // `sub_watch` — NOT `sub_watch_then_try_reap`: under the
        // recovery symmetry the prior prefix *is* the terminus's parent
        // and must stay watched as the preserved recovery edge (step 4
        // re-claims it under [`ContribKey::PromoterPrefixParent`]).
        // Try-reaping here would be wrong; it would also be a no-op in
        // practice (the freshly-materialised terminus is still its
        // child), so the explicit `sub_watch` states the intent rather
        // than relying on a coincidental short-circuit. Mirrors
        // `materialize_profile_anchor`'s plain `sub_watch` of
        // `ProfileDescent`.
        if let Some(prior) = prior_prefix_to_release {
            sub_watch(
                &mut self.tree,
                prior,
                ContribKey::PromoterPrefix(promoter_id),
                out,
            );
        }

        // 3. Register the proxy at new_proxy_resource. `register_proxy`
        // inserts into proxies map, queues enumeration (gated on
        // !already_carries per [H-5]), bumps watch_demand, and sets
        // the back-ref. Runs BEFORE step 4 so the terminus has a home
        // in `Promoter.proxies` — step 4's parent-edge helper sources
        // its target via `Promoter::terminus()` rather than threading a
        // parameter the engine seam would otherwise have to trust.
        self.register_proxy(
            promoter_id,
            new_proxy_resource,
            pattern_component_index,
            out,
        );

        // 4. Install the preserved parent-edge recovery contribution —
        // uniform for BOTH paths (descent: `prior == parent(terminus)`;
        // immediate-Materialized: `prior == None`, no descent contrib
        // ever existed — exactly `bootstrap_immediate`'s shape, which
        // calls `set_watch_root_parent` with no prior descent release).
        // Reads the terminus back from `Promoter::terminus()` — the
        // proxy registered at step 3 with `pattern_component_index ==
        // literal_prefix_len` is its own structural address.
        // Idempotent on the terminus-loss recovery cycle: the helper's
        // `already_set` short-circuit keeps the parent's `watch_demand`
        // at exactly `+1` across any number of loss → recovery cycles.
        self.set_promoter_prefix_parent(promoter_id, out);

        // 5. Drain initial enumeration (single-slot: no probe in
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
            .is_some_and(|q| match q.state() {
                PromoterState::Active { proxies, .. } => proxies.contains_key(&resource),
                PromoterState::PrefixPending(_) => unreachable!(
                    "register_proxy: state must be Active (promoter = {promoter_id:?}); \
                     enter_active and dispatch_promoter_enumeration_ok are the sole callers \
                     and both ensure Active before invocation"
                ),
            });

        self.promoters.mutate(promoter_id, |q| {
            q.insert_proxy(resource, pattern_component_index);
            // Only enqueue when NEWLY registered. Re-registration is
            // structurally idempotent — both on the contribution map
            // (`already_carries` skips `add_watch` below) and on the
            // queue (this gate).
            if !already_carries {
                q.enqueue_enumeration(resource);
            }
        });

        // Back-ref: keep `Resource.proxy_promoters` in lockstep with
        // `Promoter.proxies`. `insert_proxy_promoter` absorbs the
        // cross-Promoter / re-registration dedup internally; the
        // SmallVec inline cap of 1 covers the typical no-overlap case.
        if let Some(r) = self.tree.get_mut(resource) {
            r.insert_proxy_promoter(promoter_id);
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
    /// enumeration slot already disarmed.
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
        // (`Resource.proxy_promoters` ↔ the Promoter's `Active` proxies)
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
                .is_some_and(|res| res.proxy_promoters().contains(&promoter_id));
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
    /// shape: one gate yielding correlation + route, consume the slot
    /// once, then dispatch a typed outcome.
    ///
    /// Both Promoter probe carriers are state-resident — a
    /// `PrefixPending` descent on its `DescentState` slot, an `Active`
    /// enumeration on `enumerating`. [`Engine::promoter_probe_gate`]
    /// resolves the owner once for the gated correlation *and* the
    /// routing class; `take_owner_probe` disarms once (consume-once); an
    /// absent gate or a `received` mismatch yields
    /// [`Diagnostic::StaleProbeResponse`] with no state change, and
    /// (load-bearing) returns *before* the tail drain so a stale
    /// response never pumps the enumeration queue.
    ///
    /// The route is captured *with* the gate, before the disarm: the
    /// disarm empties the slot but leaves the state variant intact, so
    /// the route — and the enumeration `target` it carries — stays
    /// valid through dispatch. `Vanished` / `Failed` enumeration
    /// responses carry no wire payload; that pre-disarm route is the
    /// sole authority for the proxy `ResourceId`.
    ///
    /// **Typed decode, uniform with the Profile descent.** Both routes
    /// ride the `Descent` wire, so both parse the outcome through
    /// [`DescentOutcome::try_from`]: a descent advances /
    /// rewinds / fails, an enumeration consumes / drops / retries. An
    /// illegal `AnchorOk` / `SubtreeProven` proof (the walker contracted
    /// to enumerate a directory, not lower an anchor) is a
    /// walker-contract violation routed to the honest
    /// [`Diagnostic::WalkerContractViolated`] recovery — the descent
    /// abandons its prefix, the enumeration drops its proxy — never the
    /// (matched) `StaleProbeResponse`. The match is total over
    /// [`PromoterProbeRoute`]'s two variants: the Profile-only
    /// `Verifying` / `Rebasing` classes are unrepresentable here, so the
    /// owner-split carries no cross-owner arm at all.
    pub(crate) fn on_promoter_probe_response(
        &mut self,
        promoter_id: PromoterId,
        response: ProbeResponse,
        now: Instant,
        out: &mut StepOutput,
    ) {
        let owner = response.owner;
        let received = response.correlation;

        // One resolution yields the gated correlation *and* the routing
        // class (including the enumeration `target`). Captured with the
        // gate — before the disarm — so it stays valid through
        // dispatch. An absent gate or a `received` mismatch is every
        // stale path; the early `return` is load-bearing — it skips
        // the tail drain below so a stale response never pumps the
        // enumeration queue.
        let Some((_, route)) = self
            .promoter_probe_gate(promoter_id)
            .filter(|&(c, _)| c == received)
        else {
            out.diagnostics.push(Diagnostic::StaleProbeResponse {
                owner,
                correlation: received,
            });
            return;
        };
        let consumed = self.take_owner_probe(owner);
        debug_assert_eq!(
            consumed,
            Some(received),
            "consume-once: promoter slot disarm must yield the gated \
             correlation (promoter = {promoter_id:?})",
        );
        #[cfg(debug_assertions)]
        self.dispatch_ledger.record(owner, received);

        // Both Promoter routes ride the `Descent` wire, so both parse the
        // outcome through `DescentOutcome::try_from` — uniform with the
        // Profile descent demux. An illegal `AnchorOk` / `SubtreeProven`
        // proof is a walker-contract violation routed to the
        // route-appropriate honest recovery, never the (matched)
        // `StaleProbeResponse`. The match is total over
        // `PromoterProbeRoute`'s two variants: the Profile-only
        // `Verifying` / `Rebasing` classes are unrepresentable here.
        match route {
            PromoterProbeRoute::Descent => match DescentOutcome::try_from(response.outcome) {
                Ok(descent) => self.dispatch_descent(owner, descent, now, out),
                Err(WalkerContractViolation) => self.walker_contract_violated_descent(owner, out),
            },

            PromoterProbeRoute::Enumerating { target } => {
                match DescentOutcome::try_from(response.outcome) {
                    Ok(DescentOutcome::DirEnumerated(arc)) => {
                        self.dispatch_promoter_enumeration_ok(promoter_id, target, &arc, now, out);
                    }
                    Ok(DescentOutcome::Vanished) => {
                        self.dispatch_promoter_enumeration_vanished(promoter_id, target, out);
                    }
                    Ok(DescentOutcome::Failed(failure)) => {
                        self.dispatch_promoter_enumeration_failed(
                            promoter_id,
                            target,
                            failure,
                            out,
                        );
                    }
                    Err(WalkerContractViolation) => {
                        self.walker_contract_violated_enumeration(promoter_id, target, out);
                    }
                }
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
    /// [`ResourceId`] across every response outcome (`DirEnumerated` /
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
            .and_then(Promoter::pop_enumeration);
        let Some(target) = target else {
            return;
        };

        let correlation = self.mint_probe_correlation();
        // Loud arm — `pop_enumeration` resolved this same Promoter one
        // statement above, so the re-`get_mut` is structurally `Some`;
        // a `None` is a state-machine breach, not a benign race. Silent
        // skip ⇒ no arm, then the choke emits nothing and no diagnostic
        // (a wedge). (`arm_enumeration`'s own `unreachable!` is the
        // inner `PrefixPending` guard; this is the registry-resolution
        // arm.)
        let Some(q) = self.promoters.get_mut(promoter_id) else {
            unreachable!(
                "dispatch_next_enumeration: Promoter {promoter_id:?} \
                 vanished between pop_enumeration and arm_enumeration"
            );
        };
        q.arm_enumeration(correlation, target);

        // The choke reads the correlation and the proxy target back off
        // the `Active` enumeration slot's tag (the path-only wire cannot
        // echo the `ResourceId`).
        self.emit_owner_probe(owner, out);
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
            if let PromoterState::Active { proxies, .. } = q.state() {
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

        // Loop-invariant for the per-entry path join; bind once. `None`
        // is the target-was-reaped path, unreachable here: the
        // `proxy_state` lookup above already proved the slot live this
        // step. If it ever did surface, the empty `PathBuf` is *not*
        // rejected anywhere — `try_promote` attaches via
        // `SubAttachAnchor::Resource` (no path decomposition); the path
        // only feeds the synthesized Sub name and the
        // `PromotionKindObserved` diagnostic. Degrading rather than
        // panicking is the deliberate choice; the live-target invariant
        // makes it moot in practice.
        let target_path = self.tree.path_of(target);

        // Forward pass: state-keyed dispatch on the next pattern
        // component. The Literal arm matches at most one entry by
        // `BTreeMap` uniqueness — O(log N) `get_key_value`, so it
        // collapses to an `if let`; the Glob arm retains the O(N)
        // matcher scan. Both feed the shared
        // `promote_or_descend_candidate` (the per-match body) — the
        // arms now differ only in iteration shape.
        let next_component = &components[pattern_component_index];
        match next_component {
            PatternComponent::Literal(lit) => {
                if let Some((name, child)) = snapshot.entries().get_key_value(lit.as_str()) {
                    self.promote_or_descend_candidate(
                        promoter_id,
                        target,
                        target_path.as_deref(),
                        name.as_str(),
                        child.kind(),
                        is_final,
                        next_index,
                        now,
                        out,
                    );
                }
            }
            PatternComponent::Glob(g) => {
                for (name, child) in snapshot.entries() {
                    if !g.matches_path(Path::new(name.as_str())) {
                        continue;
                    }
                    self.promote_or_descend_candidate(
                        promoter_id,
                        target,
                        target_path.as_deref(),
                        name.as_str(),
                        child.kind(),
                        is_final,
                        next_index,
                        now,
                        out,
                    );
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
                    .is_some_and(|res| res.proxy_promoters().contains(&promoter_id));
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

    /// One enumeration candidate: snapshot entry `name_str` (kind
    /// `child_kind`) under proxy `target` that matched the next pattern
    /// component. Final position ⇒ promote; intermediate `Dir` ⇒
    /// register a sub-proxy; intermediate non-`Dir` ⇒ drop (a
    /// literal/glob matching a leaf mid-pattern leads nowhere; the
    /// pattern was malformed for the actual filesystem state).
    ///
    /// The single body the Literal and Glob arms of
    /// [`Self::dispatch_promoter_enumeration_ok`] share — they differ
    /// only in iteration shape (one `get_key_value`, one filtered
    /// scan), so the per-match logic lives here once. The flat
    /// argument list (the `too_many_arguments` allow) is the
    /// deliberate trade: a context struct for one internal helper with
    /// two call sites would add a lifetime-bound type for no real
    /// encapsulation, and the duplication it removes is the worse
    /// smell.
    ///
    /// **Read-only fast-path.** At the final position a pure
    /// [`Tree::lookup`] + [`Self::promoter_already_promoted`]
    /// short-circuits an already-promoted anchor *before* any
    /// `ensure_child` / `set_kind` / path join. On a stable fan-out
    /// re-enumeration (the common case — a parent's `StructureChanged`
    /// refire) each already-promoted entry then costs one `lookup`
    /// plus one per-Profile gate query and zero mutation. It is
    /// strictly a fast-path: only an existing slot can carry a
    /// Profile, so a missing slot (`lookup` ⇒ `None`) correctly falls
    /// through to the mint path, and `try_promote`'s own derived gate
    /// remains the single mint decision site — the backstop
    /// re-confirms on the freshly `ensure_child`'d slot, which carries
    /// no Profile ⇒ `false`.
    fn promote_or_descend_candidate(
        &mut self,
        promoter_id: PromoterId,
        target: ResourceId,
        target_path: Option<&Path>,
        name_str: &str,
        child_kind: EntryKind,
        is_final: bool,
        next_index: usize,
        now: Instant,
        out: &mut StepOutput,
    ) {
        if is_final {
            // Read-only fast-path: skip an already-promoted anchor
            // before any mutation. `lookup` is reused as the
            // lookup-or-ensure input below — one O(log N) probe, not
            // two.
            let existing = self.tree.lookup(Some(target), name_str);
            if let Some(slot) = existing
                && self.promoter_already_promoted(promoter_id, slot)
            {
                return;
            }

            // Resource-anchored `try_promote`. Lookup-or-ensure the
            // slot at (target, name_str), stamp the observed kind (so
            // `Profile.kind` caches at attach instead of waiting for
            // the first Seed probe), then mint. `User` is the role
            // `attach_sub_inner` would promote a `DescentScaffold`
            // to — pre-ensuring `User` short-circuits that path.
            let anchor_resource = existing.unwrap_or_else(|| {
                self.tree
                    .ensure_child(target, name_str, ResourceRole::User)
                    .expect("promoter target held alive by anchor / proxy_promoters")
            });
            self.tree.set_kind(anchor_resource, child_kind.into());
            let promote_path: Arc<Path> = match target_path {
                Some(p) => Arc::from(p.join(name_str)),
                None => Arc::from(Path::new("")),
            };
            self.try_promote(
                promoter_id,
                anchor_resource,
                promote_path,
                child_kind,
                now,
                out,
            );
        } else if matches!(child_kind, EntryKind::Dir) {
            // Non-final: only descend into `Dir` matches. [S-8] use
            // `User` role for sub-proxy slots — the `proxy_promoters`
            // back-ref is the retention signal; a `DescentScaffold`
            // would leak after unregister.
            let child_resource = self.tree.lookup(Some(target), name_str).unwrap_or_else(|| {
                self.tree
                    .ensure_child(target, name_str, ResourceRole::User)
                    .expect("promoter target held alive by anchor / proxy_promoters")
            });
            self.tree.set_kind(child_resource, child_kind.into());
            self.register_proxy(promoter_id, child_resource, next_index, out);
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
    ///
    /// **Preserve, don't recover here.** When `target` is the terminus
    /// (the unique proxy-tree root), the downward BFS in
    /// [`Self::unregister_proxy_subtree`] empties *every* proxy —
    /// `Active { proxies: ∅ }` — but it is structurally
    /// downward-only, so it cannot reach the ancestor carrying
    /// [`ContribKey::PromoterPrefixParent`]: `Promoter.prefix_parent`
    /// survives untouched. Recovery is **event-gated**, not synchronous:
    /// no terminus discriminant is read here and no recovery is kicked
    /// off. The preserved parent edge re-enters descent on the parent's
    /// *next* `StructureChanged` via `classify_event_carriers` →
    /// `start_promoter_prefix_recovery` — the structural mirror of a
    /// Profile's anchor-loss → `watch_root_parent` recovery.
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
        failure: ProbeFailure,
        out: &mut StepOutput,
    ) {
        out.diagnostics.push(Diagnostic::PromoterEnumerationFailed {
            promoter: promoter_id,
            proxy: target,
            failure,
        });
    }

    /// Recover a proxy enumeration from a walker-contract violation — an
    /// enumeration (a `ProbeRequest::Descent` on the path-only wire) whose
    /// payload resolved to an `AnchorOk` / `SubtreeProven` proof the route
    /// cannot accept (an enumeration queries a directory listing, never an
    /// anchor's `lstat` shape or a subtree proof). The typed
    /// [`DescentOutcome`] parse rejected the payload at the demux seam;
    /// this **drops the proxy subtree**.
    ///
    /// `debug_assert!` in dev/CI (a production walker never emits this
    /// shape), then in release emits [`Diagnostic::WalkerContractViolated`]
    /// and routes through [`Self::unregister_proxy_subtree`] — a proxy the
    /// walker cannot enumerate is functionally vanished from the promoter's
    /// view, so this reuses the same teardown the `Vanished` arm uses.
    /// Loop-safe: the drop re-probes nothing, and the response handler's
    /// tail drain advances the next queued target. This is the enumeration
    /// analog of the descent abandon ([`Self::walker_contract_violated_descent`]):
    /// an enumeration has no less-resolved state to rewind to, so giving up
    /// releases the proxy. Self-healing — a fresh `[[watch]]` glob match
    /// re-registers it. The enumeration slot was disarmed by
    /// `take_owner_probe` before dispatch.
    fn walker_contract_violated_enumeration(
        &mut self,
        promoter_id: PromoterId,
        target: ResourceId,
        out: &mut StepOutput,
    ) {
        debug_assert!(
            false,
            "walker contract violated: a Promoter enumeration (a Descent-wire probe) \
             received a non-enumeration outcome (AnchorOk | SubtreeProven) — an \
             enumeration queries a directory listing, never an anchor shape \
             (promoter = {promoter_id:?})",
        );
        out.diagnostics.push(Diagnostic::WalkerContractViolated {
            owner: ProbeOwner::Promoter(promoter_id),
        });
        self.unregister_proxy_subtree(promoter_id, target, out);
    }

    /// Whether `promoter_id` already has a live dynamic Sub anchored
    /// at `anchor` — the promotion dedup gate, derived from
    /// `SubRegistry` truth (the single source since `dynamic_subs` was
    /// deleted; nothing to mirror, nothing to drift).
    ///
    /// Resolves the same `(resource, config_hash)` partition
    /// `find_or_create_profile` keys on: a dynamic Sub for this
    /// `(promoter, anchor)` pair, if one exists, lives on the Profile
    /// at `(anchor, promoter.identity.config_hash())` tagged
    /// `source_promoter == Some(promoter_id)`. A stale `promoter_id`,
    /// or an anchor with no matching Profile, ⇒ not promoted.
    ///
    /// Cost is O(Subs on that one Profile) — `subs.at` is the
    /// per-Profile slice, not a registry-wide scan; a Profile carries
    /// a single-digit Sub count in practice.
    fn promoter_already_promoted(&self, promoter_id: PromoterId, anchor: ResourceId) -> bool {
        let Some(cfg) = self
            .promoters
            .get(promoter_id)
            .map(|q| q.identity.config_hash())
        else {
            return false;
        };
        let Some(profile) = self.profiles.find(anchor, cfg) else {
            return false;
        };
        self.subs.at(profile).iter().any(|&sid| {
            self.subs
                .get(sid)
                .is_some_and(|s| s.source_promoter == Some(promoter_id))
        })
    }

    /// Mint a dynamic Sub at `anchor_resource` for `promoter_id`.
    ///
    /// Enumeration ADDS; only anchor-terminal removes. At most one
    /// dynamic Sub per `(promoter_id, anchor_resource)` —
    /// [`Self::promoter_already_promoted`] below queries live
    /// `SubRegistry` truth and early-returns if a dynamic Sub for this
    /// `(promoter, anchor)` already exists (no cached map to drift).
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
    /// **`promote_path` is diagnostic-only.** The caller builds it as
    /// an owned `Arc<Path>` so the [`Diagnostic::PromotionKindObserved`]
    /// payload can move it (last use, no clone).
    pub(crate) fn try_promote(
        &mut self,
        promoter_id: PromoterId,
        anchor_resource: ResourceId,
        promote_path: Arc<Path>,
        observed_kind: EntryKind,
        now: Instant,
        out: &mut StepOutput,
    ) {
        // Dedup gate: at most one dynamic Sub per `(promoter, anchor)`.
        // Derived from live `SubRegistry` truth — no cached map to
        // drift, so a stale entry is unrepresentable by construction.
        if self.promoter_already_promoted(promoter_id, anchor_resource) {
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

        // `format_compact!` writes straight into a `CompactString` —
        // no intermediate `String` allocation before the move into
        // `SubParams.name`.
        let synthesized = format_compact!("{promoter_name}@{}", promote_path.display());

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

        // The `SubRegistry` insert inside `attach_sub_inner` *is* the
        // single source of the `(promoter, anchor)` dynamic-Sub fact —
        // no Promoter-side mirror to write. The `.expect` records the
        // liveness invariant rather than masking a soft early-return;
        // the returned `SubId` is intentionally unused (the registry
        // owns it now).
        self.attach_sub_inner(req, now, out).expect(
            "promoter forward-pass anchored at a freshly ensured live User slot; \
             the engine's Resource-arm liveness check cannot trip",
        );

        // Diagnostic. Last use of `promote_path` — move into the
        // variant.
        out.diagnostics.push(Diagnostic::PromotionKindObserved {
            promoter: promoter_id,
            path: promote_path,
            kind: observed_kind.into(),
        });

        // Fan-out warning — one-shot per Promoter lifetime, the count
        // now registry-derived behind a cheap pre-gate.
        self.maybe_warn_fanout(promoter_id, out);
    }

    /// Emit the one-shot [`Diagnostic::PromoterFanoutThreshold`] iff
    /// `promoter_id`'s *live* dynamic-Sub count first crosses
    /// [`FANOUT_WARNING_THRESHOLD`]. The count is derived from
    /// `SubRegistry` (the dedup map whose `.len()` once fed the latch
    /// was deleted — the Promoter keeps no mirror).
    ///
    /// [`Promoter::fanout_warned`] is the cheap pre-gate: an
    /// already-warned (pathological) Promoter never re-runs the
    /// O(total Subs) scan on its later promotions, so total scan cost
    /// is bounded by the pre-warning prefix of each Promoter's life.
    /// The scan only runs on a genuinely *new* promotion anyway — the
    /// dedup gate (and the read-only fast-path in
    /// [`Self::promote_or_descend_candidate`]) skip already-promoted
    /// anchors before reaching here. `latch_fanout_warning`'s own
    /// atomic check-and-latch remains the structural one-shot; this
    /// pre-gate is additive, not its replacement.
    fn maybe_warn_fanout(&mut self, promoter_id: PromoterId, out: &mut StepOutput) {
        if self
            .promoters
            .get(promoter_id)
            .is_none_or(Promoter::fanout_warned)
        {
            return;
        }
        let count = self
            .subs
            .iter()
            .filter(|(_, s)| s.source_promoter == Some(promoter_id))
            .count();
        if let Some(count) = self
            .promoters
            .get_mut(promoter_id)
            .and_then(|q| q.latch_fanout_warning(FANOUT_WARNING_THRESHOLD, count))
        {
            out.diagnostics.push(Diagnostic::PromoterFanoutThreshold {
                promoter: promoter_id,
                count,
            });
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
        out: &mut StepOutput,
    ) {
        let has_proxy = self
            .promoters
            .get(promoter_id)
            .is_some_and(|q| match q.state() {
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

        // Enqueue. The queue is `BTreeSet`-idempotent — concurrent
        // events at the same proxy collapse to one enumeration.
        if let Some(q) = self.promoters.get_mut(promoter_id) {
            q.enqueue_enumeration(resource);
        }
        self.dispatch_next_enumeration(promoter_id, out);
    }

    /// Notify operators that a dynamic Sub minted by `promoter_id` has
    /// reaped — its anchor disappeared and the all-dynamic teardown
    /// branch of [`Engine::on_anchor_terminal_event`] is unwinding the
    /// Profile. Pure narration: emits [`Diagnostic::DynamicSubReaped`]
    /// and touches no engine state.
    ///
    /// The Promoter holds no `(anchor → sub)` mirror to drop —
    /// `dynamic_subs` was deleted in favour of the derived gate
    /// ([`Self::promoter_already_promoted`]) — and the Sub's removal
    /// from `SubRegistry` is the caller's job. `anchor_path` is the
    /// operator-facing payload (the `Diagnostic` variant is path-keyed
    /// for log readability).
    ///
    /// `&self` + the `unused_self` allow keep API-shape symmetry with
    /// [`Self::dispatch_promoter_enumeration_failed`], the other
    /// diagnostic-only Promoter sibling.
    ///
    /// **Always emitted.** The sole caller
    /// ([`Engine::on_anchor_terminal_all_dynamic`]) resolves
    /// `promoter_id` from the still-attached Sub *before* removing it
    /// from the registry, so the notification always corresponds to a
    /// real reap. There is no presence gate that a concurrent
    /// [`Self::reap_promoter_inner`] could race to silently swallow the
    /// notification — the operator reliably sees the reap.
    #[allow(clippy::unused_self)]
    pub(crate) fn on_dynamic_sub_reaped(
        &self,
        promoter_id: PromoterId,
        sub_id: SubId,
        anchor_path: &Arc<Path>,
        out: &mut StepOutput,
    ) {
        out.diagnostics.push(Diagnostic::DynamicSubReaped {
            promoter: promoter_id,
            sub: sub_id,
            path: Arc::clone(anchor_path),
        });
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
    /// 2. Detach every dynamic Sub this Promoter minted. The set is
    ///    derived by scanning `SubRegistry` for
    ///    `source_promoter == promoter_id` (the dedup map was deleted;
    ///    the registry is the single source). Collect first to drop
    ///    the `subs` borrow, then detach each via
    ///    [`Engine::detach_sub_inner`]; each detach runs the standard
    ///    deferred-reap-or-immediate-reap branch on the corresponding
    ///    Profile.
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
    ///    - [`Self::release_promoter_prefix_parent_claim`] releases the
    ///      preserved [`specter_core::ContribKey::PromoterPrefixParent`]
    ///      recovery edge (idempotent — `None` for a never-materialised
    ///      or root-prefix Promoter). Placed last, exactly as
    ///      `reap_profile` runs `release_watch_root_parent_claim` last
    ///      of its claim quartet before `profiles.detach`.
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
        // the owner's slot — the descent slot (PrefixPending) or the
        // enumeration slot (Active) — and emits `ProbeOp::Cancel` iff
        // one was in flight. `pending_enumerations` drains as a side
        // effect of the proxy-release pass below: every queued entry
        // corresponds to a registered proxy, and
        // `release_promoter_proxy_claim` removes its own queue entry
        // inside `unregister_proxy`. A missed disarm is not silently
        // tolerated: the armed slot would reach
        // `release_promoter_descent_prefix_claim`'s `Active{empty}` flip
        // (PrefixPending) or `promoters.remove` (Active) and trip
        // `ProbeSlot`'s Drop tripwire — the Promoter dual of
        // `reap_profile`'s structural enforcement.
        self.cancel_owner_probe(ProbeOwner::Promoter(promoter_id), out);

        // Reap is the only "done with this owner forever" edge; drop
        // its `DispatchLedger` high-water so the debug-only `BTreeMap`
        // doesn't grow with the cumulative count of ever-attached
        // Promoters. Correctness-preserving (the next attach at the
        // same SlotMap slot bumps the generation, producing a distinct
        // `ProbeOwner`); release-only the call compiles out.
        #[cfg(debug_assertions)]
        self.dispatch_ledger
            .forget(ProbeOwner::Promoter(promoter_id));

        // 2. Detach every dynamic Sub this Promoter minted. Derived
        // from live `SubRegistry` truth (the dedup map was deleted —
        // nothing to drain, nothing to drift) by scanning for
        // `source_promoter == promoter_id`. Collect first so the
        // `subs` read borrow drops before the `detach_sub_inner`
        // `&mut self` chain. O(total Subs), cold (reap is reload /
        // shutdown, never the hot path); each `detach_sub_inner`
        // decrements the Profile refcount and reaps the underlying
        // Profile when the dynamic Sub was its last attachment.
        let sub_ids: Vec<SubId> = self
            .subs
            .iter()
            .filter(|(_, s)| s.source_promoter == Some(promoter_id))
            .map(|(sid, _)| sid)
            .collect();
        for sub_id in sub_ids {
            self.detach_sub_inner(sub_id, DetachReason::PromoterReaped, out);
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
        // the slot. The cancel-first precondition (descent slot
        // disarmed) is satisfied by the cancel above, and the flip's
        // discard is itself the `ProbeSlot` Drop enforcement point.
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
            .map(|q| match q.state() {
                PromoterState::Active { proxies, .. } => proxies.keys().copied().collect(),
                PromoterState::PrefixPending(_) => Vec::new(),
            })
            .unwrap_or_default();
        for r in proxy_list {
            self.unregister_proxy(promoter_id, r, out);
        }

        // Release the preserved parent-edge recovery contribution.
        // Idempotent (`take_prefix_parent` ⇒ `None` for a
        // never-materialised or root-prefix Promoter, no double
        // release). Placed last among the per-Resource releases,
        // mirroring `reap_profile`'s `release_watch_root_parent_claim`
        // before `profiles.detach`. No cancel-first concern — it
        // neither flips state nor drops a `ProbeSlot`.
        self.release_promoter_prefix_parent_claim(promoter_id, out);

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
